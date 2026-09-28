"""Reusable CPU factorization with bounded solves and shared residual evidence."""
import math

import numpy as np
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import matrix_inf_norm, first_nonfinite, checked_row_norms
from ghost_backend.linalg.refined_lu import requested_precision


# LU factors P A P^T with the stride permutation p_i = (i s) mod N,
# s ~ 0.618 N.  Boundary-element systems in contour order have large pivot
# growth under partial pivoting (2.5e3 at N = 2,888 and up to 3e4 at N = 9,000
# on P1 closed bodies), which put the backward error over the 1e-12 gate and
# cost a refinement solve and a second residual product on every angle batch
# (linear_solve 5.6 s against 2.8-3.4 s at N = 5,000); spreading neighbouring
# panels over the elimination order keeps it at 1-30 for the same LU time.
LU_STRIDE_FRACTION = 0.6180339887498949


def stride_permutation(n):
    """``p`` with ``p_i = (i s) mod n``, ``s`` the coprime stride nearest ``0.618 n``."""
    n = int(n)
    stride = max(1, int(round(LU_STRIDE_FRACTION * n)))
    while math.gcd(stride, n) != 1:
        stride += 1
    return (np.arange(n, dtype=np.intp) * stride) % n


def _permute_in_place(matrix, perm):
    """``matrix[:] = matrix[perm][:, perm]`` without a second full-size array."""
    n = len(perm)
    for axis in (0, 1):
        done = np.zeros(n, dtype=bool)
        for start in range(n):
            if done[start] or perm[start] == start:
                done[start] = True
                continue
            index = start
            saved = (matrix[index] if axis == 0 else matrix[:, index]).copy()
            while True:
                done[index] = True
                source = perm[index]
                if source == start:
                    break
                if axis == 0:
                    matrix[index] = matrix[source]
                else:
                    matrix[:, index] = matrix[:, source]
                index = source
            if axis == 0:
                matrix[index] = saved
            else:
                matrix[:, index] = saved


class DenseFactor:
    def __init__(self, matrix, diagnostics=None, label='dense system', evidence=None,
                 checkpoint=None, force_double=False, coordinates=None, owned_matrix=False):
        import ghost_backend.twod.solver as rcs
        self.a = np.asarray(matrix, dtype=np.complex128)
        self.diagnostics, self.label = diagnostics, label
        self.checkpoint = checkpoint or (lambda: None)
        self.checkpoint()
        row_max = None
        if self.a.ndim == 2:
            # One pass for the finite check, the infinity norm and the row
            # maxima of the equilibration (formerly three passes).
            first, self.matrix_inf, row_max = checked_row_norms(self.a)
            if first is not None:
                rcs._ensure_finite_linear_system(self.a, label=label)
        else:
            rcs._ensure_finite_linear_system(self.a, label=label)
            self.matrix_inf = matrix_inf_norm(self.a)
        self.lu = self.piv = self.mixed = None
        self._perm = None
        self._residual_spool = None
        self._spooled_buffer = None
        # Certification: the equilibration's column pass, before any in-place LU.
        self._condition_scaling = (rcs._equilibrated_scaling_and_norm_1(self.a, row_max=row_max)
                                   if diagnostics is not None and row_max is not None else None)
        self._owned_matrix = owned_matrix
        self.hierarchical = None
        self.fallback_reason = ''
        requested, threshold = rcs._requested_dense_backend()
        if requested != 'cpu' and diagnostics is not None:
            self.fallback_reason = 'CPU LU required for certified condition estimation'
        self.relative_residual = np.empty(0)
        self.event = dict(unknowns=len(self.a), factorizations=0, rhs_batches=0,
                          max_rhs_columns=0, max_backward_error=0., max_relative_residual=0.)
        self.reported_factorizations = 0
        self.reported_hierarchical_builds = 0
        if evidence is not None:
            evidence.append(self.event)
        from ghost_backend.linalg.hierarchical import (
            automatic_hierarchical,
            factor_mode,
            HierarchicalFactor,
            HierarchicalRejected,
        )
        self.factor_mode = factor_mode()
        if self.factor_mode == 'compressed':
            raise ValueError('This factorization requires a geometry-built compressed operator.')
        if self.factor_mode != 'dense' and requested_precision() == 'mixed':
            raise ValueError('Hierarchical CPU factorization requires double precision.')
        # Large dense systems are factored hierarchically by default: accepted
        # by the same exact-matrix backward-error gate as LU, with LU as fallback.
        automatic = (self.factor_mode in ('dense', 'auto') and requested_precision() == 'double'
                     and self.a.ndim == 2 and automatic_hierarchical(len(self.a)))
        if self.factor_mode == 'hierarchical' or automatic:
            try:
                self.hierarchical = timed_stage('factorization')(HierarchicalFactor)(
                    self.a, coordinates, self.checkpoint, self.matrix_inf)
                self._sync_hierarchical_builds()
                self.event['hierarchical'] = self.hierarchical.evidence
            except (HierarchicalRejected, np.linalg.LinAlgError, ValueError, RuntimeWarning) as exc:
                if self.factor_mode == 'hierarchical':
                    raise RuntimeError('Hierarchical factorization rejected this system: {}'.format(exc)) from exc
                self.fallback_reason = 'Hierarchical factorization fell back to dense LU: {}'.format(exc)
        if self.hierarchical is None and requested_precision() == 'mixed' and not force_double and rcs._SCIPY_LINALG is not None:
            try:
                self.mixed = rcs.RefinedLU(self.a)
                self.event['factorizations'] += 1
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
                self.fallback_reason = 'Mixed precision fell back to double LU: {}'.format(exc)
        if self.mixed is None and self.hierarchical is None:
            self._factor_double()
        if diagnostics is not None:
            try:
                self._condition()
            except BaseException:
                self.close()
                raise

    def _sync_hierarchical_builds(self):
        if self.hierarchical is not None:
            builds = self.hierarchical.evidence.get('builds', 1)
            self.event['factorizations'] += builds-self.reported_hierarchical_builds
            self.reported_hierarchical_builds = builds

    def _factor_double(self):
        import ghost_backend.twod.solver as rcs
        self._sync_hierarchical_builds()
        self.mixed = None
        self.hierarchical = None
        if rcs._SCIPY_LINALG is not None:
            from ghost_backend.linalg.residual_spool import ResidualSpool, selected
            use_spool = (self._residual_spool is None and requested_precision() == 'double'
                         and selected(self.a, self._owned_matrix, self.factor_mode))
            if use_spool:
                from ghost_backend.execution.options import temporary_directory, option
                scaling = self._condition_scaling
                if scaling is None and self.diagnostics is not None:
                    scaling = rcs._equilibrated_scaling_and_norm_1(self.a)
                try:
                    spool = ResidualSpool(self.a, temporary_directory(), self.checkpoint)
                except OSError:
                    if option('dense_residual_storage','auto') == 'disk':
                        raise
                else:
                    original = self.a
                    perm = stride_permutation(len(original))
                    try:
                        # The original is on disk now: permute its buffer in place.
                        _permute_in_place(original, perm)
                        self.lu, self.piv = timed_stage('factorization')(rcs._SCIPY_LINALG.lu_factor)(
                            original, overwrite_a=True, check_finite=False)
                    except BaseException:
                        spool.close()
                        raise
                    self._perm = perm
                    self.a = self._residual_spool = spool
                    self._spooled_buffer = original
                    self._condition_scaling = scaling
                    self.event.update(residual_storage='disk', residual_rows='all',
                        original_matrix_disk_bytes=original.nbytes,
                        original_matrix_buffer_bytes=spool.buffer_bytes)
                    self.event['factorizations'] += 1
                    self.event['lu_order'] = 'stride'
                    return
            perm = stride_permutation(len(self.a))
            # The permuted copy replaces the one lu_factor would make, in
            # LAPACK's column order so that it is factored in place.
            permuted = np.empty(self.a.shape, dtype=np.complex128, order='F')
            for column, source in enumerate(perm):
                if column % 4096 == 0:
                    self.checkpoint()
                permuted[:, column] = self.a[perm, source]
            self.lu, self.piv = timed_stage('factorization')(rcs._SCIPY_LINALG.lu_factor)(
                permuted, overwrite_a=True, check_finite=False)
            self._perm = perm
            self.event['factorizations'] += 1
            self.event['lu_order'] = 'stride'

    def _lu_solve(self, b, trans=0):
        """Solve with the stride-ordered LU: ``A x = b`` (or ``A^H x = b`` for trans=2)."""
        import ghost_backend.twod.solver as rcs
        if self._perm is None:
            return rcs._SCIPY_LINALG.lu_solve((self.lu, self.piv), b, trans=trans, check_finite=False)
        solved = rcs._SCIPY_LINALG.lu_solve((self.lu, self.piv), b[self._perm], trans=trans,
                                            check_finite=False)
        x = np.empty_like(solved)
        x[self._perm] = solved
        return x

    def _condition(self):
        import ghost_backend.twod.solver as rcs
        failed = False
        if self.hierarchical is not None:
            try:
                estimate = rcs._equilibrated_condition_from_lu(self.a, None, None, self.hierarchical.solve,
                                                               scaling=self._condition_scaling)
                method = 'equilibrated_1norm_hierarchical_refined_inverse'
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, RuntimeWarning) as exc:
                if self.factor_mode == 'hierarchical':
                    raise
                self.fallback_reason = 'Hierarchical condition check fell back to dense LU: {}'.format(exc)
                failed = True
        elif self.mixed is not None:
            try:
                estimate = rcs._equilibrated_condition_from_lu(
                    self.a, self.mixed.lu, self.mixed.piv, self.mixed.solve,
                    scaling=self._condition_scaling)
                method = 'equilibrated_1norm_refined_inverse'
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
                self.fallback_reason = 'Mixed precision fell back to double LU: {}'.format(exc)
                failed = True
        elif self.lu is not None:
            estimate = rcs._equilibrated_condition_from_lu(self.a, self.lu, self.piv, self._lu_solve,
                scaling=self._condition_scaling)
            method = 'equilibrated_1norm_lu_onenormest'
        else:

            row = np.maximum(np.max(np.abs(self.a), axis=1), 1e-300)
            eq = self.a / row[:, None]
            col = np.maximum(np.max(np.abs(eq), axis=0), 1e-300)
            estimate = float(np.linalg.cond(eq / col[None, :], p=1))
            method = 'equilibrated_1norm_numpy_fallback'
        if failed:


            self._factor_double()
            return self._condition()
        self._sync_hierarchical_builds()
        self.diagnostics.update(condition_est=estimate, condition_method=method,
                                condition_label=self.label)

    def close(self):
        if self._residual_spool is not None:
            self._residual_spool.close()

    def restore_original(self):
        if self._residual_spool is not None:
            self._residual_spool.restore_into(self._spooled_buffer)

    def _apply_inverse(self, b, return_residual=False):
        import ghost_backend.twod.solver as rcs
        if self.hierarchical is not None:
            try:
                result = timed_stage('rhs_solve')(self.hierarchical.solve)(b, return_residual=return_residual)
                self._sync_hierarchical_builds()
                return result
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, RuntimeWarning) as exc:
                if self.factor_mode == 'hierarchical':
                    raise
                self.fallback_reason = 'Hierarchical residual check fell back to dense LU: {}'.format(exc)


            self._factor_double()
            if self.diagnostics is not None:
                self._condition()
        if self.mixed is not None:
            try:
                return self.mixed.solve(b, return_residual=return_residual)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
                self.fallback_reason = 'Mixed precision fell back to double LU: {}'.format(exc)
            self._factor_double()
            if self.diagnostics is not None:
                self._condition()
        if self.lu is None:
            self.event['factorizations'] += 1
            x = np.linalg.solve(self.a, b)
        else:
            x = timed_stage('rhs_solve')(self._lu_solve)(b)
        return (x, None) if return_residual else x

    @timed_stage('linear_solve')
    def solve(self, rhs):
        import ghost_backend.twod.solver as rcs
        self.checkpoint()
        b = np.asarray(rhs, dtype=np.complex128)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != len(self.a) or not b.shape[1]:
            raise ValueError('RHS must have one or more columns and match the system.')
        if first_nonfinite(b) is not None:
            raise ValueError('Nonfinite RHS')
        x, inner_residual = self._apply_inverse(b, return_residual=True)
        if inner_residual is not None:


            np.negative(inner_residual, out=inner_residual)
            self.event['residual_products_reused'] = self.event.get('residual_products_reused', 0) + 1
        def metrics(candidate, rhs, residual=None):
            if residual is None:
                residual = self.a @ candidate - rhs
            ri = np.max(np.abs(residual), axis=0)
            den = self.matrix_inf * np.max(np.abs(candidate), axis=0) + np.max(np.abs(rhs), axis=0)
            errors = np.divide(ri, den, out=np.zeros_like(ri), where=den > 0)
            errors[(den <= 0) & (ri > 0)] = np.inf
            return residual, errors
        residual, errors = metrics(x, b, inner_residual)
        inner_residual = None
        steps = 0
        for attempt in range(2):
            # Only the columns above the gate are refined: one more solve and
            # residual product for them, not for the whole batch.
            columns = np.flatnonzero(~(errors <= rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX))
            if not columns.size:
                break
            candidate = x[:, columns] + self._apply_inverse(-residual[:, columns])
            updated, candidate_errors = metrics(candidate, b[:, columns])
            better = candidate_errors < errors[columns]
            if not np.any(better):
                break
            improved = columns[better]
            x[:, improved] = candidate[:, better]
            residual[:, improved] = updated[:, better]
            errors[improved] = candidate_errors[better]
            steps += 1
        error = float(np.max(errors))
        if not np.isfinite(error) or error > rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX:
            raise RuntimeError('{} normwise backward error {} exceeds the release limit {}.'.format(
                self.label, error, rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX))
        norm = np.linalg.norm(b, axis=0)
        self.relative_residual = np.linalg.norm(residual, axis=0) / np.where(norm <= rcs.EPS, 1., norm)
        self.event['rhs_batches'] += 1
        self.event['max_rhs_columns'] = max(self.event['max_rhs_columns'], b.shape[1])
        self.event['max_backward_error'] = max(self.event['max_backward_error'], error)
        self.event['max_relative_residual'] = max(self.event['max_relative_residual'], float(np.max(self.relative_residual)))
        if self.diagnostics is not None:
            self.diagnostics.update(linear_backward_error=self.event['max_backward_error'],
                                    linear_backward_error_limit=rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX,
                                    linear_refinement_steps=max(self.diagnostics.get('linear_refinement_steps', 0), steps))
            if self.mixed is not None:
                self.diagnostics['mixed_precision_corrections'] = self.mixed.max_corrections
        requested, _ = rcs._requested_dense_backend()
        rcs._record_dense_backend_event(requested=requested, used='cpu_hierarchical' if self.hierarchical is not None else 'cpu_mixed_lu' if self.mixed is not None else 'cpu',
            n=len(self.a), label=self.label, fallback_reason=self.fallback_reason, gpu_device='',
            refinement_steps=steps, factorizations=self.event['factorizations'] - self.reported_factorizations,
            rhs_columns=b.shape[1], hierarchical=self.event.get('hierarchical'),
            sweep_compression=self.event.get('sweep_compression'),
            condition_method=(self.diagnostics or {}).get('condition_method'))
        self.reported_factorizations = self.event['factorizations']
        return x[:, 0] if vector else x
