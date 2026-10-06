"""Hit rate of the box-rule moment cache in a certified 3 GHz airfoil solve."""
import sys, time
sys.path.insert(0, r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST')
import numpy as np
import ghost_backend.twod.operators as ops
from ghost_backend.twod import polynomial_quadrature as pq
counts = dict(calls=0, pairs=0, fresh=0, nocache=0)
original = ops._box_monomial_moments
def counting(obs_elems, src_elems, *a, **k):
    counts['fresh'] += len(obs_elems)
    return original(obs_elems, src_elems, *a, **k)
ops._box_monomial_moments = counting
orig_poly = ops._box_blocks_polynomial
def wrapped(elements, obs_ids, src_ids, *a, **k):
    counts['calls'] += 1; counts['pairs'] += len(obs_ids)
    if pq._MOMENT_CACHE.get() is None: counts['nocache'] += len(obs_ids)
    return orig_poly(elements, obs_ids, src_ids, *a, **k)
ops._box_blocks_polynomial = wrapped
if __name__ == '__main__':
    sys.path.insert(0, r'C:\Users\14col\AppData\Local\Temp\claude\C--Users-14col-Documents-20261005-GRIM-claude\32f846e0-15e4-47e5-9b9b-7dbb7d989d14\scratchpad\fixes')
    from bench import load_snapshot, AIRFOIL
    from ghost_backend.execution.options import automatic_options
    from ghost_backend.twod import solver as S
    snap, mdir = load_snapshot(AIRFOIL)
    t = time.perf_counter()
    r = S.solve_monostatic_rcs_2d_certified(geometry_snapshot=snap, frequencies_ghz=[3.0], elevations_deg=list(np.linspace(0, 180, 181)),
        geometry_units='inches', material_base_dir=mdir, max_panels=100000, solver_method='auto', execution_options=automatic_options())
    print('wall %.2f' % (time.perf_counter() - t), counts)
