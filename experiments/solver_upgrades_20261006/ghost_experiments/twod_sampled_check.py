"""2-D polynomial near rule: the 20-point check runs on a strided subset of each task batch.

``near_blocks`` evaluates every active task (self, touching and adaptive pairs) at the 20- and
36-point Duffy rules and accepts the 36-point moments when the two agree to
NEAR_PAIR_QUADRATURE_RTOL; the 20-point rule (24% of the samples) only serves that check, and on
the measured bodies every pair converges at 36.  Here the 36-point moments are evaluated for every
task; the 20-point moments only for every ``STRIDE``-th task of the batch plus the first task of
every (kind, shared-end) group.  When all probed tasks pass, the whole batch is accepted at 36
(the project's values, bitwise); otherwise the batch falls back to the project's complete check
(the probed moments come from the moment cache).
"""
import numpy as np
from ghost_backend.twod import polynomial_quadrature as pq

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['air_c1', 'air_c3', 'coat_c10', 'air_c10']
STRIDE = 4
STATS = dict(batches=0, tasks=0, probed=0, fallbacks=0)


def near_blocks_sampled(pairs, k, obs_derivative=True, threads=None):
    from ghost_backend.twod.operators import (NEAR_PAIR_QUADRATURE_RTOL,
                                              NEAR_PAIR_QUADRATURE_MAX_DEPTH, get_assembly_threads)
    pairs = list(pairs)
    threads = get_assembly_threads() if threads is None else max(1, int(threads))
    checkpoint = pq.solve_checkpoint()
    cache = pq._MOMENT_CACHE.get()
    rtol = NEAR_PAIR_QUADRATURE_RTOL
    table = pq._table_for(k, pairs)
    values, parents = {}, {}
    pending = [pq._Task(i, i, 0, ((0., 1.), (0., 1.))) for i in range(len(pairs))]
    next_node = len(pairs)

    def accept(node, value):
        values[node] = value
        while node in parents:
            parent, first, second = parents[node]
            if first not in values or second not in values:
                break
            a, b = values.pop(first), values.pop(second)
            del parents[first], parents[second]
            node = parent
            values[node] = tuple(x+y for x, y in zip(a, b))

    def verdict(task, low_moment, high_moment):
        obs, src = pairs[task.pair]
        low, high = pq._project(low_moment, obs, src), pq._project(high_moment, obs, src)
        scale = max(np.max(abs(high[0])), np.max(abs(high[1])), obs.length * src.length * 1e-10)
        error = max(np.max(abs(a-b)) for a, b in zip(low, high))
        return high, scale, error <= rtol * scale

    while pending:
        if checkpoint is not None:
            checkpoint()
        active = pending[-pq._ACTIVE_TASKS:]
        del pending[-len(active):]
        for task in active:
            pq._classify(task, *pairs[task.pair])
        highs = pq._moments(active, pairs, k, obs_derivative, 36, cache, threads, checkpoint, table)
        STATS['batches'] += 1
        STATS['tasks'] += len(active)
        groups = {}
        for index, task in enumerate(active):
            groups.setdefault((task.kind, task.shared), []).append(index)
        probe = sorted(set(range(0, len(active), STRIDE)) | {members[0] for members in groups.values()})
        STATS['probed'] += len(probe)
        lows_probe = pq._moments([active[i] for i in probe], pairs, k, obs_derivative, 20, cache, threads,
                                 checkpoint, table)
        probed_ok = all(verdict(active[i], low, highs[i])[2] for i, low in zip(probe, lows_probe))
        refine, following = [], []
        if probed_ok:
            for task, high_moment in zip(active, highs):
                obs, src = pairs[task.pair]
                accept(task.node, pq._project(high_moment, obs, src))
        else:
            STATS['fallbacks'] += 1
            lows = pq._moments(active, pairs, k, obs_derivative, 20, cache, threads, checkpoint, table)
            for task, low_moment, high_moment in zip(active, lows, highs):
                obs, src = pairs[task.pair]
                high, scale, converged = verdict(task, low_moment, high_moment)
                if converged:
                    accept(task.node, high)
                elif obs.panel_index == src.panel_index or task.depth >= 2*NEAR_PAIR_QUADRATURE_MAX_DEPTH:
                    refine.append((task, high, scale))
                else:
                    oi, si = task.intervals
                    if (oi[1]-oi[0])*obs.length >= (si[1]-si[0])*src.length:
                        mid = sum(oi) / 2
                        split = [((oi[0], mid), si), ((mid, oi[1]), si)]
                    else:
                        mid = sum(si) / 2
                        split = [(oi, (si[0], mid)), (oi, (mid, si[1]))]
                    parents[next_node] = parents[next_node+1] = (task.node, next_node, next_node+1)
                    following.extend(pq._Task(next_node + j, task.pair, task.depth + 1, interval)
                                     for j, interval in enumerate(split))
                    next_node += 2
        if refine:
            finest_moments = pq._moments([r[0] for r in refine], pairs, k, obs_derivative, 72,
                                         cache, threads, checkpoint, table)
            unresolved = []
            for (task, high, scale), moment in zip(refine, finest_moments):
                obs, src = pairs[task.pair]
                finest = pq._project(moment, obs, src)
                if max(np.max(abs(a-b)) for a, b in zip(high, finest)) > rtol * scale:
                    unresolved.append((task, finest, scale))
                else:
                    accept(task.node, finest)
            if unresolved:
                extra_moments = pq._moments([u[0] for u in unresolved], pairs, k, obs_derivative, 144,
                                            cache, threads, checkpoint, table)
                for (task, finest, scale), moment in zip(unresolved, extra_moments):
                    obs, src = pairs[task.pair]
                    extra = pq._project(moment, obs, src)
                    error = max(np.max(abs(a-b)) for a, b in zip(finest, extra))
                    if error > rtol * scale:
                        raise ValueError('Polynomial near quadrature did not converge; refine the mesh. '
                                         'Panels {} / {}, k={}, intervals={}, relative change={:.3g}.'.format(
                                             obs.panel_index, src.panel_index, k, task.intervals, error/scale))
                    accept(task.node, extra)
        pending.extend(following)

    return [values[index] for index in range(len(pairs))]


def _report():
    if STATS['batches']:
        import sys
        print('[experiment] twod_sampled_check', STATS, file=sys.stderr, flush=True)


def apply():
    import atexit
    atexit.register(_report)
    pq.near_blocks = near_blocks_sampled
    # Callers import the name at call time (operators._near_pair_blocks, _assemble_multi).
