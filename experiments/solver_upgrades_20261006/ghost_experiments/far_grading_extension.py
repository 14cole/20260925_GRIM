"""2-D far-rule grading: finer |k| L rows between 0.5 and 3 (calibration/far_grading_calibration.py).

The project's tables jump from the 0.50 row to 1.50 and then 3.00.  A mesh at ten elements per
wavelength puts its tiles in (0.5, 1.0], where the calibrated order for degree 2 (the first
certification pass) is one point lower on each axis than the 1.50 row; the rows 2.00 and 2.50
likewise sit below the 3.00 row.  The project's own rows are kept verbatim and the new rows are
inserted between them; every new entry is at most the enclosing project entry, so grading never
becomes coarser than today.  STATS counts the tiles per row and the orders the project would have
used.
"""
import atexit
from collections import Counter

from ghost_backend.twod import operators as ops

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['air_c1', 'air_c3', 'air_s3', 'coat_c10']
STATS = Counter()

# rows: (|k| L bound, ((near, mid, far) for degree 1, 2, 3)); '*' marks the new rows
STANDARD = (
    (0.15, ((7, 6, 5), (7, 6, 5), (8, 7, 6))),
    (0.50, ((7, 6, 5), (7, 6, 6), (8, 7, 6))),
    (1.00, ((7, 6, 6), (7, 6, 6), (8, 7, 7))),   # *
    (1.50, ((7, 6, 6), (8, 7, 7), (8, 7, 7))),
    (2.00, ((7, 7, 7), (8, 7, 7), (8, 8, 8))),   # *
    (2.50, ((7, 7, 7), (8, 8, 8), (8, 8, 8))),   # *
    (3.00, ((8, 8, 8), (8, 8, 8), (9, 9, 9))),
)
ATTENUATING = (
    (0.15, ((7, 6, 5), (7, 6, 5), (8, 7, 6))),
    (0.50, ((7, 6, 5), (7, 6, 6), (8, 7, 6))),
    (1.00, ((7, 6, 6), (7, 7, 6), (8, 7, 7))),   # *
    (1.50, ((7, 7, 7), (8, 7, 7), (8, 8, 8))),
    (2.00, ((8, 7, 7), (8, 8, 7), (9, 8, 8))),   # *
    (2.50, ((8, 7, 7), (8, 8, 8), (9, 8, 8))),   # *
    (3.00, ((8, 8, 8), (9, 8, 8), (9, 9, 9))),
)
_PROJECT = (ops._FAR_ORDER_TABLE, ops._FAR_ORDER_TABLE_ATTENUATING)


def _lookup(table, kl_max, ratio_min, cap, degree):
    """The project's table lookup (``_graded_far_order`` without its switches)."""
    row = min(max(int(degree), 1), 3) - 1
    for bound, orders in table:
        if kl_max <= bound:
            near, mid, far = orders[row]
            order = near if ratio_min < 5.0 else (mid if ratio_min < 10.0 else far)
            return max(2, min(int(cap), int(order)))
    return int(cap)


def _check_tables():
    for new, old in zip((STANDARD, ATTENUATING), _PROJECT):
        bounds = [bound for bound, _ in new]
        assert bounds == sorted(bounds)
        for bound, orders in old:
            assert (bound, orders) in new, (bound, orders)      # project rows kept verbatim
        for bound, orders in new:
            for degree in (1, 2, 3):
                for ratio in (3.0, 5.0, 10.0):
                    assert _lookup(new, bound, ratio, 99, degree) <= _lookup(old, bound, ratio, 99, degree)


def _report():
    if STATS:
        print('[experiment] far_grading_extension', dict(sorted(STATS.items())), flush=True)


def apply():
    _check_tables()
    ops._FAR_ORDER_TABLE = STANDARD
    ops._FAR_ORDER_TABLE_ATTENUATING = ATTENUATING
    original = ops._graded_far_order

    def graded(kl_max, ratio_min, cap, degree=1, attenuating=False):
        order = original(kl_max, ratio_min, cap, degree, attenuating)
        before = _lookup(_PROJECT[1] if attenuating else _PROJECT[0], kl_max, ratio_min, cap, degree)
        if before == int(cap) and order == int(cap):
            STATS['tiles(kl>3 or ungraded)'] += 1
        else:
            bound = next((b for b, _ in STANDARD if kl_max <= b), None)
            STATS['tiles row %.2f: %d->%d' % (bound, before, order)] += 1
        return order

    ops._graded_far_order = graded
    atexit.register(_report)
