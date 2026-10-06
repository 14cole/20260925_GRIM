"""Old (backup) vs new polynomial near_blocks: self, touching, regular and bisected pairs; lossy and real k;
1 vs 4 threads bitwise."""
import importlib.util
import sys
import time
sys.path.insert(0, r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST')
sys.path.insert(0, r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST\ghost_backend\tests')
import numpy as np
import ghost_backend.twod.polynomial_quadrature as new_pq
from ghost_backend.twod.basis import enrich
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.execution.cpu import CPUState, _STATE
from test_audit_fixes_2d_operators import mesh_of, circle

BACKUP = (r'C:\Users\14col\AppData\Local\Temp\claude\C--Users-14col-Documents-20261005-GRIM-claude'
          r'\32f846e0-15e4-47e5-9b9b-7dbb7d989d14\scratchpad\backup_before_fixes\ghost_backend\twod\polynomial_quadrature.py')
spec = importlib.util.spec_from_file_location('old_pq', BACKUP)
old_pq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_pq)


def elements(degree, radius=0.05, count=24, center=(0., 0.)):
    import copy
    with execution_scope(validate_options(dict(basis_order=degree))):
        mesh = mesh_of(circle(radius, count, center))   # _build_linear_mesh enriches to the scoped degree
    return mesh.elements


def pairs_for(degree):
    outer = elements(degree)
    inner = elements(degree, radius=0.0499, center=(0., 0.))  # 0.1 mm gap, elements ~13 mm: forces bisection
    pairs = []
    for i in range(6):
        pairs.append((outer[i], outer[i]))                 # self
        pairs.append((outer[i], outer[(i + 1) % 24]))      # touching
        pairs.append((outer[i], outer[(i + 3) % 24]))      # regular near
        pairs.append((outer[i], inner[i]))                 # very close -> bisected intervals
        pairs.append((outer[i], inner[(i + 1) % 24]))
    return pairs


def compare(label, a, b, tol):
    worst = 0.
    for (sa, ka), (sb, kb) in zip(a, b):
        for x, y in ((sa, sb), (ka, kb)):
            scale = max(np.max(np.abs(x)), 1e-300)
            worst = max(worst, float(np.max(np.abs(x - y))) / scale)
    print('%-40s max relative difference %.3e %s' % (label, worst, 'OK' if worst <= tol else 'FAIL'))
    return worst


results = []
for degree in (2, 3):
    pairs = pairs_for(degree)
    for k in (60.0 - 9.0j, 120.0, 400.0 - 150.0j):
        for obs_derivative in (True, False):
            with _STATE.override(CPUState()):
                t0 = time.perf_counter()
                old = old_pq.near_blocks(pairs, k, obs_derivative, threads=1)
                t1 = time.perf_counter()
                new1 = new_pq.near_blocks(pairs, k, obs_derivative, threads=1)
                t2 = time.perf_counter()
                new4 = new_pq.near_blocks(pairs, k, obs_derivative, threads=4)
            worst = compare('P%d k=%s obs=%s old vs new (1 thread)' % (degree, k, obs_derivative), old, new1, 1e-12)
            results.append(worst)
            bitwise = all(np.array_equal(x, y) for (sa, ka), (sb, kb) in zip(new1, new4) for x, y in ((sa, sb), (ka, kb)))
            print('    new 1 vs 4 threads bitwise: %s   (old %.3f s, new %.3f s)' % (bitwise, t1 - t0, t2 - t1))
            assert bitwise
print('WORST', max(results))
