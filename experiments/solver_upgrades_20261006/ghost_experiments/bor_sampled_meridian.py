"""BoR disjoint near pairs: the coarse meridian level is evaluated on a strided subset of a batch.

``_converged_disjoint_batch`` integrates every disjoint pair at a coarse meridian order (6 points
per element) and at the doubled order (12), accepting the fine blocks when the two agree to
``rtol`` (2e-5); 100% of the pairs of the measured bodies pass at the first level with a change
of 7e-8.  Here the fine level is evaluated first for every pair; the coarse level is evaluated
for every ``STRIDE``-th pair (the first included) and, when all of them pass, every pair of the
batch is accepted at the fine level (the project's values, bitwise).  Any probed failure falls
back to the project's complete refinement of the batch.
"""
import math
import numpy as np
from ghost_backend.bor import solver as S

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['cyl_s2', 'cyl_c2', 'cyl_s6', 'sph_direct', 'sph_diel', 'sph_coated']
STRIDE = 4
STATS = dict(batches=0, pairs=0, probed=0, fallbacks=0)


def _converged_disjoint_batch_sampled(gp, gq, pairs, k, m_max, kinds,
                                      order=12, rtol=2e-5, max_order=192, signed=True, mode_start=0):
    requested_order = int(order)
    if requested_order < 2 or not 2 * requested_order <= max_order <= 384 or not 0 < rtol < 1:
        raise ValueError('BoR near quadrature needs order >= 2, 2*order <= max_order <= 384, and 0 < rtol < 1.')
    results = [None] * len(pairs)
    states = []
    for index, (e, f) in enumerate(pairs):
        zero_mfie = False
        if 'mfie' in kinds:
            nodes = np.vstack((gp.nodes[e:e+2], gq.nodes[f:f+2]))
            z_tolerance = 8*np.finfo(float).eps*max(float(np.max(np.abs(nodes))), np.finfo(float).tiny)
            zero_mfie = float(np.ptp(nodes[:, 1])) <= z_tolerance
            if zero_mfie and tuple(kinds) == ('mfie',):
                width = 2*m_max+1 if signed else m_max+1-mode_start
                results[index] = ({'mfie': np.zeros((4, width, 2, 2), complex)}, int(order), 0.)
                continue
        gap = S._segment_distance(gp.nodes[e], gp.nodes[e + 1], gq.nodes[f], gq.nodes[f + 1])
        graded = gap < 0.25 * max(gp.lengths[e], gq.lengths[f])
        states.append(dict(index=index, e=e, f=f, gap=gap, graded=graded, zero_mfie=zero_mfie,
                           n=requested_order if graded else max(2, requested_order // 2)))

    def evaluate(group):
        output = [None]*len(group)
        for zero in (False, True):
            selected = [(i, st) for i, st in enumerate(group) if st['zero_mfie'] == zero]
            if not selected:
                continue
            wanted = tuple(kind for kind in kinds if not (zero and kind == 'mfie'))
            blocks = S._contract_near_batch(
                [(gp, st['e'], gq, st['f'],
                  S._gap_graded_points(gp, st['e'], gq, st['f'], st['n']) if st['graded']
                  else S._regular_cell_points(st['n'])) for _, st in selected],
                k, m_max, wanted, signed, mode_start)
            for (i, _), value in zip(selected, blocks):
                if zero:
                    width = 2*m_max+1 if signed else m_max+1-mode_start
                    value['mfie'] = np.zeros((4,width,2,2),complex)
                output[i] = value
        return output

    def block_error(fine, coarse):
        errors = []
        for kind in kinds:
            scale = np.max(np.abs(fine[kind]), axis=(1, 2, 3))
            floor = max(float(np.max(scale)) * 1e-8, 1e-280)
            error = np.max(np.abs(fine[kind] - coarse[kind]), axis=(1, 2, 3))
            errors.append(float(np.max(error / np.maximum(scale, floor))))
        return max(errors)

    STATS['batches'] += 1
    STATS['pairs'] += len(states)
    # Fine level first for every pair; the coarse level only for the probed pairs.
    for st in states:
        st['n_coarse'] = st['n']
        st['n'] = max(requested_order, min(2 * st['n'], max_order))
    fine_all = evaluate(states)
    probe = list(range(0, len(states), STRIDE)) if len(states) >= 2 * STRIDE else list(range(len(states)))
    STATS['probed'] += len(probe)
    for i in probe:
        states[i]['n'], states[i]['n_fine'] = states[i]['n_coarse'], states[i]['n']
    coarse_probe = evaluate([states[i] for i in probe])
    for i in probe:
        states[i]['n'] = states[i]['n_fine']
    worst = 0.
    ok = True
    for i, coarse in zip(probe, coarse_probe):
        error = block_error(fine_all[i], coarse)
        worst = max(worst, error)
        if not (math.isfinite(error) and error <= rtol):
            ok = False
    if ok:
        for st, fine in zip(states, fine_all):
            results[st['index']] = (fine, st['n'], worst)
        return results
    STATS['fallbacks'] += 1
    # The project's complete refinement: coarse for every pair, then the doubling loop.
    for st in states:
        st['n'] = st['n_coarse']
    for st, blocks in zip(states, evaluate(states)):
        st['coarse'] = blocks
    active = states
    while active:
        for st in active:
            st['n'] = max(requested_order, min(2 * st['n'], max_order))
        still = []
        for st, fine in zip(active, evaluate(active)):
            error = block_error(fine, st['coarse'])
            if math.isfinite(error) and error <= rtol:
                results[st['index']] = (fine, st['n'], error)
                continue
            if st['n'] >= max_order:
                raise ValueError(f"BoR near meridian quadrature did not converge for elements ({st['e']}, {st['f']}); gap={st['gap']:.6g}, order={st['n']}, relative block change={error:.3g}. Refine the mesh or increase near_max_order.")
            st['coarse'] = fine
            still.append(st)
        active = still
    return results


def _report():
    if STATS['batches']:
        import sys
        print('[experiment] bor_sampled_meridian', STATS, file=sys.stderr, flush=True)


def apply():
    import atexit
    atexit.register(_report)
    S._converged_disjoint_batch = _converged_disjoint_batch_sampled
