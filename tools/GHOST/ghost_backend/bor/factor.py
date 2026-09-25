"""One owned LU per BOR mode with bounded physical-RHS validation."""
import math
import numpy as np
from scipy.linalg import get_lapack_funcs
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import matrix_inf_norm, first_nonfinite


def residual_storage_settings():
    """(policy, directory) of the calling thread's execution options.

    Mode tasks run on executor threads, which do not inherit context
    variables, so the sweep resolves these once and hands them down."""
    from ghost_backend.execution.options import option, temporary_directory
    policy = option('dense_residual_storage', 'auto')
    try:
        directory = temporary_directory()
    except ValueError:
        if policy == 'disk':
            raise
        directory = None
    return policy, directory


def _residual_spool_selected(matrix, owned, policy):
    """Spool the original coefficients (``linalg.residual_spool``) instead of
    keeping them next to the LU: the same ``dense_residual_storage`` policy as
    the 2-D dense factor ('auto' spools a system of 512 MiB or more only when
    its copy does not fit the memory available as it is factored; the worker
    plan prices the copy)."""
    from ghost_backend.linalg.residual_spool import auto_spooled
    eligible = (owned and isinstance(matrix, np.ndarray) and matrix.dtype == np.complex128
                and matrix.ndim == 2 and matrix.flags.owndata and matrix.flags.writeable
                and (matrix.flags.f_contiguous or matrix.flags.c_contiguous))
    return eligible and (policy == 'disk' or policy == 'auto' and auto_spooled(matrix.nbytes))


class ModalFactor:
    """LU of one modal system with original-coefficient residuals.

    ``owned=True`` lets a large system be factored in its own buffer: the
    original coefficients go to a disk spool for the residuals, so one mode
    holds one dense matrix instead of the matrix plus its LU copy.  The
    caller must not use ``matrix`` afterwards (``solve_am`` never does).
    """

    def __init__(self, matrix, mode, monitor_cond, checkpoint=None, owned=False,
                 residual_storage=None):
        from ghost_backend.bor.solver import BOR_CONDITION_EST_MAX
        self.a = matrix
        self.mode = mode
        self.checkpoint = checkpoint or (lambda: None)
        self.diagnostics = None
        self.checkpoint()
        if first_nonfinite(matrix) is not None:
            raise RuntimeError('BoR mode m={} produced a non-finite system matrix.'.format(mode))
        self.matrix_inf = matrix_inf_norm(matrix)
        self.event = dict(factorizations=1, rhs_batches=0, max_rhs_columns=0,
                          max_backward_error=0., max_relative_residual=0., refinement_steps=0,
                          residual_storage='memory')
        getrf, self.getrs, gecon = get_lapack_funcs(('getrf', 'getrs', 'gecon'), (matrix,))
        norm = None
        if monitor_cond:
            # Row-stripped norm avoids an extra full real matrix from abs(A);
            # taken before an in-place factorization overwrites A.
            sums = np.zeros(len(matrix))
            for start in range(0, len(matrix), 64):
                self.checkpoint()
                sums += np.sum(abs(matrix[start:start + 64]), axis=0)
            norm = float(np.max(sums))
        # 0: LU of A (solve A x = b); 1: LU of A^T in a C-ordered buffer
        # (solve with the transposed factors).
        self.trans = 0
        spool = None
        policy, directory = residual_storage or residual_storage_settings()
        if directory is not None and _residual_spool_selected(matrix, owned, policy):
            from ghost_backend.linalg.residual_spool import ResidualSpool
            try:
                spool = ResidualSpool(matrix, directory, self.checkpoint)
            except OSError:
                if policy == 'disk':
                    raise
        if spool is not None:
            buffer = matrix if matrix.flags.f_contiguous else matrix.T
            self.trans = 0 if matrix.flags.f_contiguous else 1
            try:
                self.lu, self.piv, info = timed_stage('factorization')(getrf)(buffer, overwrite_a=True)
            except BaseException:
                spool.close()
                raise
            self.a = spool
            self.event.update(residual_storage='disk', original_matrix_disk_bytes=int(matrix.nbytes))
        else:
            # Keep A for original-coefficient residuals; LAPACK owns one F-order copy.
            self.lu, self.piv, info = timed_stage('factorization')(getrf)(
                np.array(matrix, dtype=complex, order='F', copy=True), overwrite_a=True)
        if info:
            raise RuntimeError('BoR mode m={} LU factorization failed (LAPACK info={}).'.format(mode, info))
        self.condition = math.nan
        if monitor_cond:
            # ||A||_1 is the infinity norm of the factored A^T.
            reciprocal, info = timed_stage('condition_estimate')(gecon)(
                self.lu, norm, norm='I' if self.trans else '1')
            if info or not math.isfinite(float(reciprocal)) or reciprocal <= 0 or norm <= 0:
                raise RuntimeError('BoR mode m={} condition estimation failed.'.format(mode))
            self.condition = 1. / float(reciprocal)
            if not math.isfinite(self.condition) or self.condition > BOR_CONDITION_EST_MAX:
                raise RuntimeError('BoR mode m={} estimated 1-norm condition {} exceeds the release limit {}.'.format(
                    mode, self.condition, BOR_CONDITION_EST_MAX))

    def close(self):
        """Release the residual spool (also released when the factor is collected)."""
        close = getattr(self.a, 'close', None)
        if close is not None:
            close()

    def inverse(self, rhs):
        self.checkpoint()
        value, info = timed_stage('rhs_solve')(self.getrs)(self.lu, self.piv, rhs, trans=self.trans)
        if info:
            raise RuntimeError('BoR mode m={} LU solve failed (LAPACK info={}).'.format(self.mode, info))
        return value

    def errors(self, x, b):
        residual = self.a @ x - b
        norms = np.linalg.norm(residual, axis=0)
        bnorms = np.linalg.norm(b, axis=0)
        relative = norms / np.where(bnorms > 0., bnorms, 1.)
        numerator = np.max(abs(residual), axis=0)
        denominator = self.matrix_inf * np.max(abs(x), axis=0) + np.max(abs(b), axis=0)
        backward = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.)
        backward[(denominator <= 0.) & (numerator > 0.)] = np.inf
        return residual, relative, backward

    def solve(self, rhs):
        from ghost_backend.bor.solver import BOR_LINEAR_BACKWARD_ERROR_MAX, BOR_LINEAR_RESIDUAL_MAX
        b = np.asarray(rhs, complex)
        if b.ndim != 2 or b.shape[0] != len(self.a) or not b.shape[1] or first_nonfinite(b) is not None:
            raise RuntimeError('BoR mode m={} produced an invalid or non-finite excitation.'.format(self.mode))
        x = self.inverse(b)
        residual, relative, backward = self.errors(x, b)
        for attempt in range(2):
            if np.max(relative) <= BOR_LINEAR_RESIDUAL_MAX and np.max(backward) <= BOR_LINEAR_BACKWARD_ERROR_MAX:
                break
            candidate = x + self.inverse(-residual)
            updated = self.errors(candidate, b)
            if np.max(updated[1]) >= np.max(relative) and np.max(updated[2]) >= np.max(backward):
                break
            x = candidate
            residual, relative, backward = updated
            self.event['refinement_steps'] += 1
        if (first_nonfinite(x) is not None or not np.all(np.isfinite(relative))
                or not np.all(np.isfinite(backward)) or np.max(backward) > BOR_LINEAR_BACKWARD_ERROR_MAX):
            raise RuntimeError('BoR mode m={} normwise linear backward error {} exceeds the release limit {}.'.format(
                self.mode, float(np.max(backward)), BOR_LINEAR_BACKWARD_ERROR_MAX))
        self.relative_residual = relative
        self.event['rhs_batches'] += 1
        self.event['max_rhs_columns'] = max(self.event['max_rhs_columns'], b.shape[1])
        self.event['max_backward_error'] = max(self.event['max_backward_error'], float(np.max(backward)))
        self.event['max_relative_residual'] = max(self.event['max_relative_residual'], float(np.max(relative)))
        return x


def compressed_storage_budget(options, workers):
    """Resolve the same per-worker cap for planning and factor construction."""
    from ghost_backend.compressed.runtime import automatic_storage_bytes
    configured = options['compressed_storage_mib']
    total = automatic_storage_bytes() if configured == 0 else configured * 1024**2
    return int(total) // max(1, int(workers))


def compressed_factor(oracle, mode, monitor_cond, options, workers, checkpoint=None):
    from scipy.sparse.linalg import LinearOperator, onenormest
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.factor import CompressedFactor
    from ghost_backend.bor.solver import BOR_CONDITION_EST_MAX
    # 0 sizes the cap from the solve memory limit; a fixed cap starves an
    # electrically large body, whose modes each need their own share.
    budget = compressed_storage_budget(options, workers)
    coordinates = oracle.row_coordinates
    if coordinates is None:
        coordinates = np.arange(oracle.n, dtype=float)[:, None]
    operator = timed_stage('modal_compressed_assembly')(StreamedOperator)(oracle, coordinates,
        tile=options['compression_tile'], budget=budget, checkpoint=checkpoint)
    factor = CompressedFactor(operator, label='BoR mode m={}'.format(mode), checkpoint=checkpoint,
                              storage_budget_bytes=budget, check_precision=False)
    factor.event['refinement_steps'] = 0
    factor.condition = math.nan
    if monitor_cond:
        inverse = LinearOperator(operator.shape, matvec=lambda z: factor.inverse(z),
                                 rmatvec=lambda z: factor.inverse(z, trans=2), dtype=complex)
        # Preserve BOR's unscaled 1-norm condition criterion.
        factor.condition = float(np.max(operator.column_norm + operator.column_error)) * float(onenormest(inverse))
        if not math.isfinite(factor.condition) or factor.condition > BOR_CONDITION_EST_MAX:
            raise RuntimeError('BoR mode m={} estimated 1-norm condition {} exceeds the release limit {}.'.format(
                mode, factor.condition, BOR_CONDITION_EST_MAX))
    return factor
