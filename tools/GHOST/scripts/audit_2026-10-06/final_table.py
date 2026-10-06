"""Markdown table of the final bench against the baseline, for AUDIT_FIXES_2026-10-06.md."""
import glob
import json
import os
import numpy as np

S = r'C:\Users\14col\AppData\Local\Temp\claude\C--Users-14col-Documents-20261005-GRIM-claude\32f846e0-15e4-47e5-9b9b-7dbb7d989d14\scratchpad\fixes'
LABELS = {
    'air_c1': 'Airfoil certified 1 GHz (414 panels, P2 -> P3)',
    'air_c3': 'Airfoil certified 3 GHz (864 panels)',
    'air_c10': 'Airfoil certified 10 GHz (2,472 panels, HODLR on the P3 system)',
    'air_s3': 'Airfoil survey 3 GHz (5,666 panels, 8,608 unknowns, dense)',
    'air_s3c': 'Airfoil survey 3 GHz, compressed backend forced',
    'ibc_c1': 'TYPE 2 CSV IBC square, certified 1 GHz (P1 pair)',
    'two_c3': 'TYPE 5 two dielectrics, certified 3 GHz (P1 pair)',
    'thin_c1': 'TYPE 1 thin dielectric strip, certified 1 GHz',
    'coat_c10': 'TYPE 4 PEC-backed coating, certified 10 GHz (hp, refined once)',
    'cyl_s2': 'BoR PEC cylinder survey 2 GHz (126 unknowns)',
    'cyl_c2': 'BoR PEC cylinder certified 2 GHz (base + fine)',
    'cyl_s6': 'BoR PEC cylinder survey 6 GHz (370 unknowns)',
    'sph_direct': 'BoR PEC sphere ka = 10, direct API (202 unknowns)',
    'sph_diel': 'BoR dielectric sphere, direct API (244 unknowns)',
    'sph_coated': 'BoR coated PEC sphere, direct API (342 unknowns)',
}
rows = []
worst = 0.
for name in LABELS:
    ref, new = os.path.join(S, 'baseline', name), os.path.join(S, 'final', name)
    if not (os.path.exists(ref + '.npz') and os.path.exists(new + '.npz')):
        continue
    a, b = np.load(ref + '.npz'), np.load(new + '.npz')
    ia, ib = json.load(open(ref + '.json', encoding='utf-8')), json.load(open(new + '.json', encoding='utf-8'))
    rel = 0.
    for key in a.files:
        if key.endswith('_amp'):
            rel = max(rel, float(np.max(np.abs(a[key] - b[key]))) / max(float(np.max(np.abs(a[key]))), 1e-300))
    worst = max(worst, rel)
    change = 100. * (ib['wall'] / ia['wall'] - 1.)
    rows.append('| %s | %.2f | %.2f | %+.0f%% | %.1e |' % (LABELS[name], ia['wall'], ib['wall'], change, rel))
table = ['| Case | Before (s) | After (s) | Change | Peak-relative change |', '|---|---:|---:|---:|---:|'] + rows
table.append('')
table.append('Worst peak-relative change over the set: %.1e (rule: 1e-8).' % worst)
print('\n'.join(table))
