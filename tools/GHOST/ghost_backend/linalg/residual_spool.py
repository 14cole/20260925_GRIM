"""Disk-backed original coefficients for full, bounded-memory residuals."""
import shutil
import tempfile
import weakref
import numpy as np


class ResidualSpool:
    block_bytes = 16 * 1024**2

    def __init__(self, matrix, directory, checkpoint):
        self.shape, self.dtype, self.ndim = matrix.shape, matrix.dtype, 2
        self.checkpoint = checkpoint
        if shutil.disk_usage(directory).free < matrix.nbytes + 64*1024**2:
            raise OSError('Insufficient temporary disk space for full original-matrix residuals.')
        self.file = tempfile.TemporaryFile(prefix='ghost-residual-', suffix='.bin', dir=directory)
        self._cleanup = weakref.finalize(self, self.file.close)
        try:
            for start, stop in self._rows():
                self.checkpoint()
                block = np.ascontiguousarray(matrix[start:stop])
                self.file.write(memoryview(block).cast('B'))
            self.file.flush()
        except BaseException:
            self.close()
            raise

    # Row blocks of at least this many rows: OpenBLAS products of fewer rows
    # can take seconds on many threads (see execution.options.blas_core_budget).
    min_block_rows = 256

    def _block_rows(self):
        return max(self.min_block_rows, self.block_bytes // (16*self.shape[1]))

    @property
    def buffer_bytes(self):
        """Bytes of one row block read back from the spool."""
        return 16 * self.shape[1] * min(self.shape[0], self._block_rows())

    def _rows(self):
        count = self._block_rows()
        for start in range(0, self.shape[0], count):
            yield start, min(start+count, self.shape[0])

    def __len__(self):
        return self.shape[0]

    def __matmul__(self, rhs):
        rhs = np.asarray(rhs)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.shape[1]:
            raise ValueError('Residual RHS does not match the original matrix.')
        result = np.empty((self.shape[0],rhs.shape[1]),complex)
        self.file.seek(0)
        for start, stop in self._rows():
            self.checkpoint()
            block = np.fromfile(self.file,dtype=np.complex128,count=(stop-start)*self.shape[1])
            if block.size != (stop-start)*self.shape[1]:
                raise OSError('Original-matrix residual spool is incomplete.')
            result[start:stop] = block.reshape(stop-start,self.shape[1]) @ rhs
        return result[:,0] if vector else result

    def close(self):
        self._cleanup()

    def restore_into(self, matrix):
        """Restore an owned assembly buffer needed by the next polarization."""
        self.file.seek(0)
        for start, stop in self._rows():
            self.checkpoint()
            block = np.fromfile(self.file,dtype=np.complex128,count=(stop-start)*self.shape[1])
            if block.size != (stop-start)*self.shape[1]:
                raise OSError('Original-matrix residual spool is incomplete.')
            matrix[start:stop] = block.reshape(stop-start,self.shape[1])


# 'auto' keeps the original coefficients next to their LU whenever that copy
# fits.  Both admission plans already price it (the 2-D dense estimate two
# matrices, BoR three per mode worker), and each residual product of a spooled
# matrix re-reads the whole file: the certified 10 GHz airfoil (7.8 GB system)
# spent 64 s of 207 s re-reading it about 20 times, and ran in 145 s kept in
# memory within its 16.7 GB estimate.  A large matrix is spooled only when the
# memory available as it is factored cannot hold the copy -- a guard against
# pressure the plan did not foresee, not a size rule.
AUTO_SPOOL_MIN_BYTES = 512 * 1024**2
COPY_MARGIN_BYTES = 256 * 1024**2


def copy_fits(nbytes):
    """Whether a copy of ``nbytes`` (plus a margin) fits the memory available now.

    Unknown availability counts as not fitting, which keeps a large matrix spooled.
    """
    from ghost_backend.twod.solver import _detect_available_gb
    try:
        available = float(_detect_available_gb()) * 1024**3
    except Exception:
        return False
    return available > 0 and available >= nbytes + max(COPY_MARGIN_BYTES, nbytes // 8)


def auto_spooled(nbytes):
    """The 'auto' residual-storage decision for an eligible owned matrix."""
    return nbytes >= AUTO_SPOOL_MIN_BYTES and not copy_fits(nbytes)


def selected(matrix, owned, factor_mode):
    from ghost_backend.execution.options import option
    policy = option('dense_residual_storage','auto')
    # 'dense' and 'auto' factor large systems hierarchically; an LU fallback
    # that finds no room for its copy spools the original like any other.
    eligible = (owned and factor_mode in ('dense', 'auto') and matrix.flags.owndata
                and matrix.flags.f_contiguous and matrix.flags.writeable)
    return eligible and (policy == 'disk' or policy == 'auto' and auto_spooled(matrix.nbytes))
