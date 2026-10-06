"""Far-build wall for tile shapes (rows x sources) at the same live set: n=800 sphere, 5 GHz."""
import sys
import time
sys.path.insert(0, r"C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST")
import numpy as np
from ghost_backend.bor import solver as bs
from ghost_backend.bor import streaming as st


def main(n_elems=800, f_ghz=5.0):
    points = bs.sphere_generatrix(0.1, n_elems)
    solver = bs.BorPecSolver(points, f_ghz * 1e9)
    m_max, tail = bs._bor_mode_limits(solver.k, 0.1, np.asarray([0., 45., 90.]), None)
    stream = st.StreamingFarBlocks(solver, m_max, efie=True, mfie=True, workers=8)
    ne = solver.gen.n_elems
    default = (stream._tile_rows, stream._tile_sources)
    print(f"n={ne} m_max={m_max} default plan rows={default[0]} sources={default[1]} workers={stream._workers} "
          f"native_threads={stream._native_threads} work_bytes={stream._work_bytes/1e6:.1f} MB", flush=True)
    reference = None
    shapes = [default, (4, max(1, ne // 4)), (8, max(1, ne // 8)), (16, max(1, ne // 16)), (8, ne), (32, max(1, ne // 32))]
    for rows, sources in shapes:
        stream._tile_rows, stream._tile_sources = rows, sources
        t0 = time.perf_counter()
        stream._build_range(0, m_max)
        wall = time.perf_counter() - t0
        if reference is None:
            reference = (stream.Z.copy(), stream.K.copy())
            agreement = ''
        else:
            dz = np.max(np.abs(stream.Z - reference[0])) / np.max(np.abs(reference[0]))
            dk = np.max(np.abs(stream.K - reference[1])) / np.max(np.abs(reference[1]))
            agreement = f'agreement efie {dz:.1e} mfie {dk:.1e}'
        tiles = len(range(0, ne, rows)) * len(range(0, ne, sources))
        print(f"rows={rows:3d} sources={sources:4d} tiles={tiles:5d} wall {wall:6.2f} s {agreement}", flush=True)
    stream.close()


if __name__ == "__main__":
    main(*(float(a) if '.' in a else int(a) for a in sys.argv[1:]))
