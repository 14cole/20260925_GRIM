"""BoR far build: the 2 pi Galerkin factor of the bracket families is folded into the transform tables.

``_banded_stream`` scales every bracket output (four ``[pairs, orders]`` arrays per tile) by 2 pi in
place after the projection; multiplying the far smaller half-grid cosine/sine tables (one
``[half, orders]`` pair per sample-count group) before the GEMM gives the same products to rounding
and removes four full passes over the tile outputs (about 4% of a tile's thread time).  The FFT
projection path keeps the scaling on its output.
"""
import numpy as np
from ghost_backend.bor import kernels as K
from ghost_backend.bor import streaming as ST

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['cyl_s6', 'sph_direct', 'sph_coated']


def banded_modal_kernels(kind, coordinates, k, m_max, near_mask, modes=None,
                         work_bytes=K.FFT_BUILD_BUDGET, threads=1, out_dtype=np.complex128, scale=1.0):
    if kind not in ('g', 'mfie', 'ibc'):
        raise ValueError('Unknown banded modal kernel.')
    dtype = K._check_out_dtype(out_dtype)
    shape = np.broadcast(*coordinates).shape
    arrays = [np.broadcast_to(np.asarray(a, float), shape).ravel() for a in coordinates]
    rp, zp, rq, zq = arrays if kind == 'g' else (arrays[0], arrays[1], arrays[4], arrays[5])
    modes = (np.asarray(list(range(m_max+2)) if kind == 'g' else list(range(-m_max, m_max+1)), int)
             if modes is None else np.asarray(modes, int).ravel())
    count = len(rp)
    outputs = [np.zeros((count, len(modes)), dtype) for _ in range(1 if kind == 'g' else 4)]
    active = np.flatnonzero(~np.broadcast_to(near_mask, shape).ravel())
    gap = np.hypot(rp[active]-rq[active], zp[active]-zq[active])
    if np.any(gap <= 0):
        raise ValueError('Coincident point pairs must use singular near quadrature.')
    bracket = kind != 'g'
    top = max(int(m_max) + (0 if bracket else 1), int(np.max(np.abs(modes))) if modes.size else 0)
    sizes = K._far_sample_counts(bracket, rp[active], rq[active], gap, k, top)
    if np.any(sizes > K.N_XI_SAFETY_CAP):
        raise ValueError('Banded azimuthal quadrature exceeds its sample safety cap; route close pairs to near integration.')
    wavenumber = complex(k)
    sampler = K._native_pair_sampler(kind)
    if sampler is None and active.size:
        K._notice_native_fallback(('sample_g_pairs' if kind == 'g' else 'sample_brackets_pairs',))
    threads = max(1, int(threads))
    order = np.argsort(sizes, kind='stable')
    group_sizes, starts = np.unique(sizes[order], return_index=True)
    stops = np.r_[starts[1:], len(order)]
    rows = 1 if kind == 'g' else 4
    per_sample = rows if sampler is not None else (4 if kind == 'g' else 18)
    for size, start, stop in zip(group_sizes, starts, stops):
        indices = active[order[start:stop]]
        size = int(size)
        half = size // 2 + 1
        projected_pairs = min(len(indices), max(1, int(work_bytes // (
            16 * ((per_sample + 4) * half + len(modes))))))
        projection = K._half_grid_projection_method(projected_pairs, half, len(modes))
        overhead = {'complex': 0, 'real': 1, 'fft': 4}[projection]
        cosine, sine, sin2, cx, sx = K._half_grid_tables(size, modes, bracket, projection)
        tail = 1.0
        if scale != 1.0:
            if projection == 'fft':
                tail = scale                      # the transform has no table to scale
            else:
                cosine = cosine * scale
                sine = None if sine is None else sine * scale
        chunk = max(1, int(work_bytes // (16 * ((per_sample + overhead) * half + len(modes)))))
        for first in range(0, len(indices), chunk):
            ids = indices[first:first+chunk]
            pair = [a[ids] for a in arrays]
            if kind == 'g':
                samples = None if sampler is None else sampler(pair, wavenumber, sin2, threads)
                if samples is None:
                    a, b, c, d = pair
                    distance = np.sqrt(((a-c)**2+(b-d)**2)[:, None]+4*(a*c)[:, None]*sin2)
                    samples = np.exp(-1j*wavenumber*distance)/(4*np.pi*distance)
                projected = K._project_half_grid(samples, cosine, modes, size, projection)
                outputs[0][ids] = projected if tail == 1.0 else projected * tail
            else:
                samples = None if sampler is None else sampler(pair, wavenumber, cx, sx, threads)
                if samples is None:
                    xi = 2*np.pi*np.arange(half)/size - np.pi
                    samples = (K._mfie_brackets(*pair, k, xi) if kind == 'mfie'
                               else K._ibc_brackets_grid(*pair, k, np.broadcast_to(xi, (len(ids), half))))
                tt, tf, ft, ff = samples
                for slot, values, table, odd in ((0, tt, cosine, False), (1, tf, sine, True),
                                                 (2, ft, sine, True), (3, ff, cosine, False)):
                    projected = K._project_half_grid(values, table, modes, size, projection, odd=odd)
                    outputs[slot][ids] = projected if tail == 1.0 else projected * tail
    result = tuple(o.reshape(shape+(len(modes),)) for o in outputs)
    return result[0] if kind == 'g' else result


def _banded_stream(stream, rows, kind, modes, sources=None, strict_upper=False, near_free=False):
    if hasattr(stream, 'solver'):
        sp = sq = stream.solver
        near_sources = sp._near_sources_by_element
    else:
        sp, sq, near_sources = stream.sp, stream.sq, stream._near_sources
    gp, gq = sp.g, sq.g
    go_p, go_q = sp.gauss_order, sq.gauss_order
    first, stop, _ = rows.indices(sp.P)
    f0, f1 = (0, sq.gen.n_elems) if sources is None else (int(sources[0]), int(sources[1]))
    c0, c1 = f0 * go_q, f1 * go_q
    if near_free and not strict_upper:
        near = np.zeros((1, 1), bool)
    else:
        near = np.zeros((stop - first, c1 - c0), bool)
        for e in range(first // go_p, (stop + go_p - 1) // go_p):
            r = slice(max(0, e * go_p - first), min(stop - first, (e + 1) * go_p - first))
            for f in near_sources[e]:
                if f0 <= f < f1:
                    near[r, (f - f0) * go_q:(f - f0 + 1) * go_q] = True
            if strict_upper and e >= f0:
                near[r, :(min(e + 1, f1) - f0) * go_q] = True
    names = ('rho', 'z') if kind == 'g' else ('rho', 'z', 'trho', 'tz')
    args = tuple(getattr(gp, name)[rows, None] for name in names)
    args += tuple(getattr(gq, name)[None, c0:c1] for name in names)
    work = getattr(stream, '_work_bytes', None)
    if work is None:
        work = K.FFT_BUILD_BUDGET / max(1, getattr(stream, '_workers', 1))
    return banded_modal_kernels(kind, args, stream.k, stream.m_max, near, modes, work_bytes=work,
                                threads=getattr(stream, '_native_threads', 1),
                                scale=1.0 if kind == 'g' else 2.0 * np.pi)


def apply():
    K.banded_modal_kernels = banded_modal_kernels
    ST._banded_stream = _banded_stream
