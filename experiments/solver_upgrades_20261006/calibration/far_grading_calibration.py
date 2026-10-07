"""Calibrate finer |k| L bins of the 2-D far-rule grading tables (same method as the project's tables).

For every candidate row (|k| L upper bound), ratio column (closest centre-distance ratio 3, 5, 10),
polynomial degree and medium (arg k from 0 to -45 degrees; strongly attenuating below -45) the
smallest Gauss order whose element-pair blocks stay within 1e-12 of an independent 48-point rule:
S relative to the block's largest entry, K' and D relative to the integral of the kernel magnitude,
over |k| L at the bound and at half the bin, ratios at the column's lower edge and 1.5x it, length
ratios 1, 0.3 and 0.1, and orientations.  Prints the tables in the project's layout.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import hankel2

ROOT = Path(__file__).resolve().parents[3] / 'tools' / 'GHOST'
sys.path.insert(0, str(ROOT))
from ghost_backend.twod.basis import values as basis_values, derivative_matrix  # noqa: E402

TOLERANCE = 1e-12
ROWS = (0.15, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
COLUMNS = (3.0, 5.0, 10.0)
ORIENTATIONS = ((0., 0., 1.), (1.1, .5, .3), (2.9, 1.2, 1.), (np.pi / 2, 0., .5), (0.4, 2.0, 0.1), (2.2, 0.9, 0.3))


def blocks(q, wave, lo, ls, angle, phi_c, degree, distance):
    x, w = np.polynomial.legendre.leggauss(q)
    x, w = (x + 1) / 2, w / 2
    ts = np.array([np.cos(angle), np.sin(angle)])
    ns = np.array([-ts[1], ts[0]])
    c = distance * np.array([np.cos(phi_c), np.sin(phi_c)])
    p = np.column_stack(((x - .5) * lo, 0 * x))
    s = c + (x[:, None] - .5) * ls * ts
    d = p[:, None, :] - s[None, :, :]
    r = np.linalg.norm(d, axis=2)
    green = .25j * hankel2(0, wave * r)
    h = .25j * wave * hankel2(1, wave * r)
    phi = basis_values(x, degree)
    weight = phi.T * w
    mass = weight @ green @ weight.T
    k_block = weight @ (-h * d[:, :, 1] / r) @ weight.T * lo * ls
    d_block = weight @ (h * (d @ ns) / r) @ weight.T * lo * ls
    scale = np.max(weight @ np.abs(h) @ weight.T) * lo * ls
    return mass * lo * ls, k_block, d_block, scale


def worst_error(q, degree, kls, ratios, args):
    worst = 0.
    for kl in kls:
        for arg in args:
            wave = kl * np.exp(1j * arg)
            for ratio in ratios:
                for angle, phi_c, ls in ORIENTATIONS:
                    s, kb, db, scale = blocks(q, wave, 1., ls, angle, phi_c, degree, ratio)
                    sr, kr, dr, _ = blocks(48, wave, 1., ls, angle, phi_c, degree, ratio)
                    worst = max(worst, np.max(abs(s - sr)) / np.max(abs(sr)),
                                max(np.max(abs(kb - kr)), np.max(abs(db - dr))) / scale)
    return worst


def calibrate(args, label):
    print('%s table: rows (|k| L bound, ((near, mid, far) per degree 1..3))' % label, flush=True)
    previous = 0.
    for bound in ROWS:
        kls = (bound, 0.5 * (previous + bound)) if previous else (bound, 0.5 * bound)
        row = []
        for degree in (1, 2, 3):
            orders = []
            for column, upper in zip(COLUMNS, COLUMNS[1:] + (COLUMNS[-1] * 2,)):
                ratios = (column, 0.5 * (column + upper))
                order = next(q for q in range(2, 17) if worst_error(q, degree, kls, ratios, args) <= TOLERANCE)
                orders.append(order)
            row.append(tuple(orders))
        print('    (%.2f, %s),' % (bound, tuple(row)), flush=True)
        previous = bound


if __name__ == '__main__':
    calibrate((0., -0.35, -0.7), 'standard (arg k 0 to -40 degrees)')
    calibrate((-0.8, -1.1, -1.45, -1.55), 'attenuating (arg k -46 to -89 degrees)')
