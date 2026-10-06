"""Accuracy and timing bench for the audit fixes: one fresh process per case.

py bench.py run OUTDIR [--set quick|full] [--only case,case]
py bench.py compare REFDIR NEWDIR [--tol 1e-8]
py bench.py case NAME OUTDIR          (internal: one case in this process)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST'
GEO = ROOT + r'\ghost_backend\geometry\geometries'
MAT = ROOT + r'\ghost_backend\validation\material_examples'
AIRFOIL = r'C:\Users\14col\Documents\20261005_GRIM_claude\airfoil.geo'
CYL = GEO + r'\BOR\pec_cylinder_r4_h10_in.geo'

CASES = {
    # 2-D dispatch API, production profile (automatic_options), 181 angles.
    'air_c1': dict(kind='2d', mode='certified', geo=AIRFOIL, units='inches', ghz=1.0, set='quick'),
    'air_c3': dict(kind='2d', mode='certified', geo=AIRFOIL, units='inches', ghz=3.0, set='quick'),
    'air_s3': dict(kind='2d', mode='survey', geo=AIRFOIL, units='inches', ghz=3.0, set='quick'),
    'air_s3c': dict(kind='2d', mode='survey', geo=AIRFOIL, units='inches', ghz=3.0, set='full',
                    options=dict(factorization='compressed')),
    'air_c10': dict(kind='2d', mode='certified', geo=AIRFOIL, units='inches', ghz=10.0, set='full'),
    'ibc_c1': dict(kind='2d', mode='certified', geo=MAT + r'\type2_csv_ibc.geo', units='meters', ghz=1.0, set='quick'),
    'two_c3': dict(kind='2d', mode='certified', geo=MAT + r'\type5_two_dielectrics.geo', units='meters', ghz=3.0, set='quick'),
    'thin_c1': dict(kind='2d', mode='certified', geo=MAT + r'\type1_thin_dielectric.geo', units='meters', ghz=1.0, set='quick'),
    'coat_c10': dict(kind='2d', mode='certified', geo=MAT + r'\type4_pec_backed_coating.geo', units='meters', ghz=10.0, set='full'),
    # BoR dispatch API (certified / survey), 181 aspects, 4 workers (driver default).
    'cyl_s2': dict(kind='bor', mode='survey', geo=CYL, units='inches', ghz=2.0, workers=4, set='quick'),
    'cyl_c2': dict(kind='bor', mode='certified', geo=CYL, units='inches', ghz=2.0, workers=4, set='quick'),
    'cyl_s6': dict(kind='bor', mode='survey', geo=CYL, units='inches', ghz=6.0, workers=4, set='full'),
    # BoR direct APIs.
    'sph_direct': dict(kind='bor_direct', fn='solve_bor', a=0.1, n=100, ghz=4.77, angles=37, workers=4, set='quick'),
    'sph_diel': dict(kind='bor_direct', fn='solve_bor_dielectric', a=0.1, n=60, ghz=2.0, angles=19, workers=4,
                     eps_r=3.0 - 0.1j, mu_r=1.0 - 0.02j, set='quick'),
    'sph_coated': dict(kind='bor_direct', fn='solve_bor_coated_pec', a=0.1, n=60, ghz=2.0, angles=19, workers=4,
                       eps_r=3.0 - 0.1j, mu_r=1.0 - 0.02j, set='quick'),
}


def load_snapshot(geo):
    from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot
    geo = Path(geo)
    snap = build_geometry_snapshot(*parse_geometry(geo.read_text(encoding='utf-8-sig')))
    snap['source_path'] = str(geo)
    return snap, str(geo.parent)


def _channels(samples):
    import numpy as np
    by = {}
    for row in samples:
        label = row.get('polarization', row.get('channel', '?'))
        by.setdefault(str(label), []).append(row)
    out = {}
    for label, rows in by.items():
        theta = np.array([r['theta_inc_deg'] for r in rows], float)
        amp = np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in rows])
        db = np.array([r['rcs_db'] for r in rows], float)
        res = np.array([r.get('linear_residual', 0.) for r in rows], float)
        out[label + '_theta'] = theta
        out[label + '_amp'] = amp
        out[label + '_db'] = db
        out[label + '_res'] = res
    return out


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 200 else '<ndarray %s>' % (o.shape,)
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def run_case(name, outdir, root=None):
    root = root or ROOT
    sys.path.insert(0, root)
    import numpy as np
    c = CASES[name]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    info = dict(case=name, spec={k: (str(v) if isinstance(v, complex) else v) for k, v in c.items()})
    t_import = time.perf_counter()
    if c['kind'] == '2d':
        from ghost_backend.execution.options import automatic_options
        from ghost_backend.twod import solver as S
        fn = S.solve_monostatic_rcs_2d_certified if c['mode'] == 'certified' else S.solve_monostatic_rcs_2d_survey
        snap, mdir = load_snapshot(c['geo'])
        options = automatic_options()
        options.update(c.get('options', {}))
        kwargs = dict(geometry_snapshot=snap, frequencies_ghz=[float(c['ghz'])],
                      elevations_deg=list(np.linspace(0., 180., c.get('angles', 181))),
                      geometry_units=c['units'], material_base_dir=mdir, max_panels=100000,
                      solver_method='auto', execution_options=options)
        info['import_seconds'] = time.perf_counter() - t_import
        t1 = time.perf_counter()
        result = fn(**kwargs)
        wall = time.perf_counter() - t1
        channels = _channels(result['samples'])
        md = result['metadata']
        am = md.get('adaptive_mesh') or {}
        info.update(wall=wall,
                    stage_seconds=(md.get('runtime_profile') or {}).get('stage_seconds'),
                    backend=(md.get('backend_selection') or {}).get('selected'),
                    planning_seconds=(md.get('backend_selection') or {}).get('planning_seconds'),
                    adaptive_used=am.get('used'), adaptive_fallback=am.get('fallback'),
                    steps=[{k: s.get(k) for k in ('polynomial_degree', 'panels', 'nodes', 'seconds', 'backend')}
                           for s in (am.get('steps') or [])],
                    panel_count=md.get('panel_count'), nodes=md.get('linear_node_count'),
                    degree=md.get('polynomial_degree'), threads=md.get('execution_threads'),
                    certified=md.get('mesh_convergence_certified'),
                    warnings=md.get('warnings'))
        # Per-channel condition estimates and factor variants when present.
        systems = []
        for key in ('experimental_cpu', 'cpu_kernel_execution'):
            st = md.get(key)
            if isinstance(st, dict):
                for s in st.get('systems') or []:
                    if isinstance(s, dict):
                        systems.append({k: s.get(k) for k in ('label', 'unknowns', 'factor_variant',
                                                             'condition_est', 'factor_work_seconds')})
        info['systems'] = systems[:12]
        cm = md.get('channel_metadata') or {}
        info['channel_condition'] = {ch: (v.get('condition_est') if isinstance(v, dict) else None) for ch, v in cm.items()}
    elif c['kind'] == 'bor':
        from ghost_backend.bor.dispatch import solve_monostatic_rcs_bor_survey, solve_monostatic_rcs_bor_certified
        from ghost_backend.twod.samples import compact_samples
        fn = solve_monostatic_rcs_bor_certified if c['mode'] == 'certified' else solve_monostatic_rcs_bor_survey
        fn = compact_samples()(fn)
        snap, mdir = load_snapshot(c['geo'])
        kwargs = dict(geometry_snapshot=snap, frequencies_ghz=[float(c['ghz'])],
                      elevations_deg=list(np.linspace(0., 180., c.get('angles', 181))),
                      geometry_units=c['units'], material_base_dir=mdir, workers=c['workers'],
                      cfie_alpha=0.5, bor_options={})
        info['import_seconds'] = time.perf_counter() - t_import
        t1 = time.perf_counter()
        result = fn(**kwargs)
        wall = time.perf_counter() - t1
        channels = _channels(result['samples'])
        md = result['metadata']
        pfs = md.get('per_frequency') or []
        pf = pfs[0] if pfs else {}
        info.update(wall=wall, stage_seconds=(md.get('runtime_profile') or {}).get('stage_seconds'),
                    modes_used=pf.get('modes_used'), mode_cap=pf.get('mode_cap'),
                    mode_tasks_started=pf.get('mode_tasks_started'), unknowns=pf.get('n_unknowns'),
                    assembly=pf.get('assembly'), near_preparation=pf.get('near_preparation'),
                    certified=md.get('mesh_convergence_certified'), warnings=md.get('warnings'))
    else:
        from ghost_backend.bor import solver as B
        pts = B.sphere_generatrix(c['a'], c['n'])
        thetas = list(np.linspace(0., 180., c['angles']))
        if c['fn'] == 'solve_bor':
            fn, kwargs = B.solve_bor, dict(points=pts, freq_hz=c['ghz'] * 1e9, thetas_deg=thetas,
                                           formulation='cfie', workers=c['workers'])
        elif c['fn'] == 'solve_bor_dielectric':
            fn, kwargs = B.solve_bor_dielectric, dict(points=pts, freq_hz=c['ghz'] * 1e9, thetas_deg=thetas,
                                                      eps_r=c['eps_r'], mu_r=c['mu_r'], workers=c['workers'])
        else:
            fn, kwargs = B.solve_bor_coated_pec, dict(points_outer=B.sphere_generatrix(1.2 * c['a'], c['n']),
                                                      points_core=B.sphere_generatrix(0.8 * c['a'], int(0.8 * c['n'])),
                                                      freq_hz=c['ghz'] * 1e9, thetas_deg=thetas,
                                                      eps_r=c['eps_r'], mu_r=c['mu_r'], workers=c['workers'])
        info['import_seconds'] = time.perf_counter() - t_import
        t1 = time.perf_counter()
        result = fn(**kwargs)
        wall = time.perf_counter() - t1
        channels = {}
        for ch in ('vv', 'hh'):
            channels[ch.upper() + '_theta'] = np.asarray(result['theta_deg'], float)
            channels[ch.upper() + '_amp'] = np.asarray(result['amp_' + ch], complex)
            channels[ch.upper() + '_db'] = 10 * np.log10(np.maximum(np.asarray(result['sigma_' + ch], float), 1e-300))
        info.update(wall=wall, stage_seconds=(result.get('runtime_profile') or {}).get('stage_seconds'),
                    modes_used=result.get('modes_used'), mode_cap=result.get('mode_cap'),
                    mode_tasks_started=result.get('mode_tasks_started'), unknowns=result.get('n_unknowns'),
                    assembly=result.get('assembly'), near_preparation=result.get('near_preparation'),
                    stream_sampling_backend=result.get('stream_sampling_backend'))
    np.savez(outdir / (name + '.npz'), **channels)
    with open(outdir / (name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1, default=_json_default)
    print('[%s] wall %.3f s' % (name, wall), flush=True)


def run_all(outdir, which, only, root=None, repeat=1):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = [n for n, c in CASES.items() if (only and n in only) or (not only and (which == 'full' or c['set'] == 'quick'))]
    env = dict(os.environ)
    env.pop('GHOST_HIERARCHICAL_MIN_UNKNOWNS', None)
    for name in names:
      for _ in range(max(1, int(repeat))):
        log = outdir / (name + '.log')
        t0 = time.perf_counter()
        with open(log, 'w', encoding='utf-8') as f:
            proc = subprocess.run([sys.executable, __file__, 'case', name, str(outdir)] + (['--root', root] if root else []),
                                  cwd=root or ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        print('%-11s exit %d  process %.1f s' % (name, proc.returncode, time.perf_counter() - t0), flush=True)
        if proc.returncode:
            print(log.read_text(encoding='utf-8')[-3000:])


def compare(ref, new, tol):
    import numpy as np
    ref, new = Path(ref), Path(new)
    worst = 0.
    rows = []
    for path in sorted(ref.glob('*.npz')):
        name = path.stem
        other = new / path.name
        if not other.exists():
            rows.append((name, 'MISSING', None, None, None))
            continue
        a, b = np.load(path), np.load(other)
        ia = json.load(open(ref / (name + '.json'), encoding='utf-8'))
        ib = json.load(open(new / (name + '.json'), encoding='utf-8'))
        case_worst, db_worst, detail = 0., 0., []
        for key in a.files:
            if not key.endswith('_amp'):
                continue
            x, y = a[key], b[key]
            if x.shape != y.shape:
                detail.append('%s shape %s vs %s' % (key, x.shape, y.shape))
                case_worst = float('inf')
                continue
            peak = max(float(np.max(np.abs(x))), 1e-300)
            rel = float(np.max(np.abs(x - y))) / peak
            case_worst = max(case_worst, rel)
            dbk = key[:-4] + '_db'
            if dbk in a.files and dbk in b.files:
                db_worst = max(db_worst, float(np.max(np.abs(a[dbk] - b[dbk]))))
            detail.append('%s %.2e' % (key[:-4], rel))
        worst = max(worst, case_worst)
        rows.append((name, 'PASS' if case_worst <= tol else 'FAIL', case_worst, db_worst,
                     (ia.get('wall'), ib.get('wall'), ' '.join(detail))))
    print('%-11s %-7s %-10s %-10s %8s %8s %6s  %s' % ('case', 'verdict', 'peak-rel', 'max dB', 'ref s', 'new s', 'ratio', 'channels'))
    for name, verdict, rel, db, extra in rows:
        if extra is None:
            print('%-11s %-7s' % (name, verdict))
            continue
        wr, wn, detail = extra
        ratio = (wn / wr) if (wr and wn) else float('nan')
        print('%-11s %-7s %-10.2e %-10.2e %8.2f %8.2f %6.2f  %s' % (name, verdict, rel, db, wr or 0, wn or 0, ratio, detail))
    print('worst peak-relative change: %.3e (tolerance %.1e)' % (worst, tol))
    return 0 if worst <= tol else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('outdir'); r.add_argument('--set', default='quick'); r.add_argument('--only', default='')
    r.add_argument('--root', default=None); r.add_argument('--repeat', type=int, default=1)
    c = sub.add_parser('compare'); c.add_argument('ref'); c.add_argument('new'); c.add_argument('--tol', type=float, default=1e-8)
    k = sub.add_parser('case'); k.add_argument('name'); k.add_argument('outdir'); k.add_argument('--root', default=None)
    a = ap.parse_args()
    if a.cmd == 'run':
        run_all(a.outdir, a.set, [s for s in a.only.split(',') if s], a.root, a.repeat)
    elif a.cmd == 'compare':
        sys.exit(compare(a.ref, a.new, a.tol))
    else:
        run_case(a.name, a.outdir, a.root)


if __name__ == '__main__':
    main()
