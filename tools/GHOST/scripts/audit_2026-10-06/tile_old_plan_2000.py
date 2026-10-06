"""Far-build wall of the previous tile plan (full-width rows) on the 2000-element sphere, for comparison."""
import sys
import time
sys.path.insert(0, r"C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST")
import numpy as np
from ghost_backend.bor import solver as bs, streaming as st

n, f = 2000, 5.0
solver = bs.BorPecSolver(bs.sphere_generatrix(0.1, n), f * 1e9)
m_max, _ = bs._bor_mode_limits(solver.k, 0.1, np.asarray([0., 45., 90.]), None)
stream = st.StreamingFarBlocks(solver, m_max, efie=True, mfie=True, workers=8)
new_plan = (stream._tile_rows, stream._tile_sources)
# The previous planner: the largest full-width row count that fits the per-tile share.
with_min = st.STREAM_TILE_MIN_ROWS
st.STREAM_TILE_MIN_ROWS = 10**9   # never triggers the narrow-source branch
old = st._plan_banded_tiles(n, solver.gauss_order, n, solver.gauss_order, min(stream.mode_block, m_max + 1) + 2,
                            min(stream.mode_block, m_max + 1), True, True, st.BOR_STREAM_TILE_BUDGET_GB, 8, stream._nx_e if hasattr(stream, '_nx_e') else 0)
st.STREAM_TILE_MIN_ROWS = with_min
print('new plan rows=%d sources=%d; previous plan rows=%d sources=%d' % (new_plan[0], new_plan[1], old[1], old[2]), flush=True)
for label, (rows, sources) in (('previous', (old[1], old[2])), ('new', new_plan)):
    stream._tile_rows, stream._tile_sources = rows, sources
    t0 = time.perf_counter()
    stream._build_range(0, m_max)
    print('%s plan rows=%d sources=%d build %.2f s' % (label, rows, sources, time.perf_counter() - t0), flush=True)
    if label == 'previous':
        reference = stream.Z.copy()
    else:
        print('agreement %.1e' % (np.max(np.abs(stream.Z - reference)) / np.max(np.abs(reference))))
stream.close()
