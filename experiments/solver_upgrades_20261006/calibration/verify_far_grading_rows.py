"""Denser verification of the new far-rule rows of far_grading_extension (and candidates beyond |k| L = 3).

Each cell (row, degree, column) is tested on more configurations than the calibration sweep: four
|k| L values across the bin, four ratios per column, eight panel angles, four centre directions and
four length ratios, with the standard (three arg k) or attenuating (four) wavenumber set, i.e.
6,144 or 8,192 element pairs per cell.  The kernels at the 48-point reference and at every candidate
order 5..16 are sampled once per pair and shared by the three degrees.  Prints per row the smallest
order whose blocks stay within 1e-12 of the reference (same error measures as the calibration) and,
for the proposed rows, PASS/FAIL per entry.
"""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import hankel2

ROOT = Path(__file__).resolve().parents[3] / 'tools' / 'GHOST'
sys.path.insert(0, str(ROOT))
from ghost_backend.twod.basis import values as basis_values  # noqa: E402

TOLERANCE = 1e-12
ORDERS = list(range(5, 17))
COLUMNS = {'near': (3.0, 3.5, 4.0, 4.5), 'mid': (5.0, 6.0, 7.5, 9.0), 'far': (10.0, 12.0, 15.0, 20.0)}
ANGLES = np.linspace(0.0, np.pi, 8, endpoint=False) + 0.05
DIRECTIONS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
LENGTHS = (1.0, 0.5, 0.2, 0.1)
STANDARD_ARGS = (0.0, -0.35, -0.7)
ATTENUATING_ARGS = (-0.8, -1.1, -1.45, -1.55)
# proposed rows: (bound, previous bound, ((near, mid, far) per degree) or None for a survey only)
PROPOSED = {
    'standard': ((1.00, 0.50, ((7, 6, 6), (7, 6, 6), (8, 7, 7))),
                 (2.00, 1.50, ((7, 7, 7), (8, 7, 7), (8, 8, 8))),
                 (2.50, 2.00, ((7, 7, 7), (8, 8, 8), (8, 8, 8))),
                 (4.00, 3.00, None), (5.00, 4.00, None), (6.00, 5.00, None), (8.00, 6.00, None)),
    'attenuating': ((1.00, 0.50, ((7, 6, 6), (7, 7, 6), (8, 7, 7))),
                    (2.00, 1.50, ((8, 7, 7), (8, 8, 7), (9, 8, 8))),
                    (2.50, 2.00, ((8, 7, 7), (8, 8, 8), (9, 8, 8))),
                    (4.00, 3.00, None), (5.00, 4.00, None), (6.00, 5.00, None), (8.00, 6.00, None)),
}
_RULES = {}
_BASIS = {}


def rule(q):
    if q not in _RULES:
        x, w = np.polynomial.legendre.leggauss(q)
        _RULES[q] = ((x + 1) / 2, w / 2)
        _BASIS[q] = {d: basis_values(_RULES[q][0], d).T * _RULES[q][1] for d in (1, 2, 3)}
    return _RULES[q]


def kernels(q, wave, ls, angle, phi_c, distance):
    """Green's function, K' and D kernels and |h| on the q x q product rule (observation length 1)."""
    x, w = rule(q)
    ts = np.array([np.cos(angle), np.sin(angle)])
    ns = np.array([-ts[1], ts[0]])
    c = distance * np.array([np.cos(phi_c), np.sin(phi_c)])
    p = np.column_stack(((x - .5), 0 * x))
    s = c + (x[:, None] - .5) * ls * ts
    d = p[:, None, :] - s[None, :, :]
    r = np.linalg.norm(d, axis=2)
    green = .25j * hankel2(0, wave * r)
    h = .25j * wave * hankel2(1, wave * r)
    return green, -h * d[:, :, 1] / r, h * (d @ ns) / r, np.abs(h)


def blocks(q, kern, degree, ls):
    weight = _BASIS[q][degree]
    return tuple(weight @ k @ weight.T * ls for k in kern)


def smallest_orders(bound, previous, args):
    """Per (degree, column): the smallest order within TOLERANCE over the bin's configurations."""
    kls = np.linspace(previous, bound, 5)[1:]
    need = {}
    for column, ratios in COLUMNS.items():
        worst = {(d, q): 0.0 for d in (1, 2, 3) for q in ORDERS}
        for kl in kls:
            for arg in args:
                wave = kl * np.exp(1j * arg)
                for ratio in ratios:
                    for angle in ANGLES:
                        for phi_c in DIRECTIONS:
                            for ls in LENGTHS:
                                ref = kernels(48, wave, ls, angle, phi_c, ratio)
                                cand = {q: kernels(q, wave, ls, angle, phi_c, ratio) for q in ORDERS}
                                for d in (1, 2, 3):
                                    sr, kr, dr, scale_r = blocks(48, ref, d, ls)
                                    s_scale = np.max(np.abs(sr))
                                    h_scale = np.max(scale_r)
                                    for q in ORDERS:
                                        s, kb, db, _ = blocks(q, cand[q], d, ls)
                                        err = max(np.max(np.abs(s - sr)) / s_scale,
                                                  max(np.max(np.abs(kb - kr)), np.max(np.abs(db - dr))) / h_scale)
                                        if err > worst[(d, q)]:
                                            worst[(d, q)] = err
        for d in (1, 2, 3):
            need[(d, column)] = next((q for q in ORDERS if worst[(d, q)] <= TOLERANCE), None)
    return need


if __name__ == '__main__':
    only = sys.argv[1:]
    for medium, args in (('standard', STANDARD_ARGS), ('attenuating', ATTENUATING_ARGS)):
        if only and medium not in only:
            continue
        print('%s medium (arg k %s)' % (medium, args), flush=True)
        for bound, previous, proposed in PROPOSED[medium]:
            started = time.perf_counter()
            need = smallest_orders(bound, previous, args)
            found = tuple(tuple(need[(d, c)] for c in ('near', 'mid', 'far')) for d in (1, 2, 3))
            verdict = ''
            if proposed is not None:
                ok = all(need[(d, c)] is not None and need[(d, c)] <= proposed[d - 1][i]
                         for d in (1, 2, 3) for i, c in enumerate(('near', 'mid', 'far')))
                verdict = '  proposed %s: %s' % (proposed, 'PASS' if ok else 'FAIL')
            print('    (%.2f, %s),  # measured smallest orders, %.0f s%s'
                  % (bound, found, time.perf_counter() - started, verdict), flush=True)
