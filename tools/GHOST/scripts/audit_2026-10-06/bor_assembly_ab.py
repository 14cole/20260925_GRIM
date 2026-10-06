"""tables vs streaming on small/medium PEC spheres and the cylinder (workers=4)."""
import sys, time
sys.path.insert(0, r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST')
import numpy as np

def run(n, f, assembly, kind='pec'):
    from ghost_backend.bor.solver import solve_bor, solve_bor_dielectric, sphere_generatrix
    pts = sphere_generatrix(0.1, n)
    t = time.perf_counter()
    if kind == 'pec':
        r = solve_bor(pts, f, list(np.linspace(0, 180, 37)), formulation='cfie', workers=4, assembly=assembly)
    else:
        r = solve_bor_dielectric(pts, f, list(np.linspace(0, 180, 19)), eps_r=3-0.1j, mu_r=1-0.02j, workers=4, assembly=assembly)
    return time.perf_counter() - t, r

if __name__ == '__main__':
    for kind, n, f in (('pec', 40, 1.0e9), ('pec', 100, 4.77e9), ('pec', 200, 4.77e9), ('pec', 300, 3.0e9), ('diel', 60, 2.0e9)):
        out = {}
        for assembly in ('tables', 'streaming', 'tables', 'streaming'):
            wall, r = run(n, f, assembly, kind)
            out.setdefault(assembly, []).append(wall)
            amp = np.asarray(r['amp_vv'])
            out.setdefault(assembly + '_amp', amp)
        d = float(np.max(np.abs(out['tables_amp'] - out['streaming_amp'])) / np.max(np.abs(out['tables_amp'])))
        st = r['runtime_profile']['stage_seconds']
        print('%s n=%d f=%.2f GHz unknowns=%d modes %d/%d: tables %s s | streaming %s s | VV diff %.1e | last stages %s' % (
            kind, n, f/1e9, r['n_unknowns'], r['modes_used'], r['mode_cap'], ['%.2f' % w for w in out['tables']],
            ['%.2f' % w for w in out['streaming']], d, {k: round(v, 2) for k, v in st.items() if v > 0.05}), flush=True)
