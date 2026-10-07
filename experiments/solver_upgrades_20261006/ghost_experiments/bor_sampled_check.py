"""BoR graded near rules: the coarse level is evaluated on a strided subset of each layout chunk.

The project evaluates every point of a layout group at the coarse and the fine level and accepts a
point when the two agree to NEAR_ANGULAR_RTOL.  The fine level is the published value either way;
the coarse level (40% of all angular samples) only decides acceptance, and on every body measured
100% of the points pass at the first level.  Here the fine level is still evaluated for every point;
the coarse level is evaluated for every ``STRIDE``-th point of the chunk (the first included).  When
all probed points pass, the whole chunk is accepted at the fine level (bitwise the project's values);
when any probed point fails, the chunk falls back to the project's complete check.
"""
import numpy as np
from ghost_backend.bor import kernels as K

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['cyl_s2', 'cyl_c2', 'cyl_s6', 'sph_direct', 'sph_diel', 'sph_coated']
STRIDE = 4
STATS = dict(chunks=0, points=0, probed=0, fallbacks=0)


def _store(kind, out, ids, fine, offset):
    if kind != 'g':
        out[0][ids] = fine[0][:, :, offset:]
        out[1][ids] = fine[1][:, :, offset:]
    else:
        out[ids] = 2.0 * (fine[:, 0][:, offset:] + 1j * fine[:, 1][:, offset:])


def _refine_near_chunk_sampled(kind, arrays, ids, stable, base_orders, evaluate, next_orders, out, offset=0):
    bracket = kind != 'g'
    use_stable = stable
    coarse_orders = np.asarray(base_orders, dtype=np.int64)
    fine_orders = next_orders(coarse_orders)
    if fine_orders is None:
        raise ValueError("BoR near angular quadrature exceeds its accuracy limit; refine the mesh or reduce modal bandwidth.")
    fine = evaluate(ids, fine_orders, use_stable)
    STATS['chunks'] += 1
    STATS['points'] += len(ids)
    if len(ids) >= 2 * STRIDE:
        probe = np.arange(0, len(ids), STRIDE)
        STATS['probed'] += len(probe)
        coarse_probe = evaluate(ids[probe], coarse_orders, use_stable)
        fine_probe = (fine[0][probe], fine[1][probe]) if bracket else fine[probe]
        error, scale = K._near_check(kind, fine_probe, coarse_probe)
        if np.all(np.isfinite(error) & (error <= K.NEAR_ANGULAR_RTOL * np.maximum(scale, 1e-280))):
            _store(kind, out, ids, fine, offset)
            return
        STATS['fallbacks'] += 1
    else:
        STATS['probed'] += len(ids)
    # The project's complete check (its loop, with the fine level already evaluated).
    coarse = evaluate(ids, coarse_orders, use_stable)
    while True:
        error, scale = K._near_check(kind, fine, coarse)
        converged = np.isfinite(error) & (error <= K.NEAR_ANGULAR_RTOL * np.maximum(scale, 1e-280))
        accepted = ids[converged]
        if bracket:
            out[0][accepted] = fine[0][converged][:, :, offset:]
            out[1][accepted] = fine[1][converged][:, :, offset:]
        else:
            out[accepted] = 2.0 * (fine[:, 0][converged][:, offset:] + 1j * fine[:, 1][converged][:, offset:])
        if np.all(converged):
            return
        pending = ~converged
        ids = ids[pending]
        coarse = (fine[0][pending], fine[1][pending]) if bracket else fine[pending]
        error, scale = error[pending], scale[pending]
        grown = next_orders(fine_orders)
        if grown is None:
            if bracket and not use_stable:
                use_stable = True
                coarse_orders = np.asarray(base_orders, dtype=np.int64)
                fine_orders = next_orders(coarse_orders)
                coarse = evaluate(ids, coarse_orders, use_stable)
                fine = evaluate(ids, fine_orders, use_stable)
                continue
            worst = int(np.argmax(error / np.maximum(scale, 1e-280)))
            coordinates = tuple(float(v[ids[worst]]) for v in arrays)
            name = {'g': 'modal_kernels_near', 'mfie': 'mfie_kernels_near', 'ibc': 'ibc_kernels_near'}[kind]
            raise ValueError(
                f"BoR near angular quadrature did not converge at the maximum order: "
                f"{name}, point={coordinates}, relative change="
                f"{float(error[worst] / max(scale[worst], 1e-280)):.3g}.")
        coarse_orders, fine_orders = fine_orders, grown
        fine = evaluate(ids, fine_orders, use_stable)


def _report():
    if STATS['chunks']:
        import sys
        print('[experiment] bor_sampled_check', STATS, file=sys.stderr, flush=True)


def apply():
    import atexit
    atexit.register(_report)
    K._refine_near_chunk = _refine_near_chunk_sampled
