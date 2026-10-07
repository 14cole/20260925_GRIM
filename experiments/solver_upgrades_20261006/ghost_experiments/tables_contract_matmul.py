"""BoR dense-tables path: the nine EFIE left contractions of a chunk are grouped by left weight.

``_efie_tables_into`` contracts every mode's ``[P, P]`` Green tables with the element bases:
three source-side products, then nine ``einsum('eai,eafb->eifb')`` left contractions with an
inner dimension of three (the Gauss order), which spend most of their time in operand reshapes.
Here the nine are five: the right-hand sides sharing a left weight are concatenated along their
last axis and contracted by one batched ``matmul`` per weight, then scattered into the four
quadrants with their scales.  The material solvers (dielectric, coated, junctions) still assemble
from tables below 2 GB; conductors stream.
"""
import numpy as np
from ghost_backend.bor import solver as S

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['sph_diel', 'sph_coated']


def _grouped_left_into(left, pieces, e0):
    """``pieces``: (Y slice [ce*o, neq, 2], target, scale) sharing ``left`` [ce, o, 2]."""
    ce, o, _ = left.shape
    neq = pieces[0][0].shape[1]
    columns = np.concatenate([piece[0] for piece in pieces], axis=2)          # [ce*o, neq, 2g]
    g = len(pieces)
    A = np.matmul(left.transpose(0, 2, 1), columns.reshape(ce, o, neq * 2 * g))  # [ce, 2, neq*2g]
    A = A.reshape(ce, 2, neq, g, 2)
    for s, (_, target, scale) in enumerate(pieces):
        block = A[:, :, :, s, :]
        if scale != 1:
            block = block * scale
        target[e0:e0 + ce, 0:neq] += block[:, 0, :, 0]
        target[e0:e0 + ce, 1:neq + 1] += block[:, 0, :, 1]
        target[e0 + 1:e0 + ce + 1, 0:neq] += block[:, 1, :, 0]
        target[e0 + 1:e0 + ce + 1, 1:neq + 1] += block[:, 1, :, 1]


def _efie_tables_into(targets, m, k, gp, ne_p, op, gq, ne_q, oq, table, scale):
    am = abs(int(m))
    rw_p, rw_q = gp.rho * gp.w, gq.rho * gq.w
    lp = dict(rw_tr=S._local_basis(gp, ne_p, op, rw_p * gp.trho),
              rw_tz=S._local_basis(gp, ne_p, op, rw_p * gp.tz),
              rw=S._local_basis(gp, ne_p, op, rw_p),
              w=S._local_basis(gp, ne_p, op, gp.w),
              dw=S._local_basis(gp, ne_p, op, gp.w, derivative=True))
    right_g = np.concatenate([S._local_basis(gq, ne_q, oq, rw_q * gq.tz),
                              S._local_basis(gq, ne_q, oq, gq.w, derivative=True),
                              S._local_basis(gq, ne_q, oq, gq.w)], axis=2)
    right_c = np.concatenate([S._local_basis(gq, ne_q, oq, rw_q * gq.trho),
                              S._local_basis(gq, ne_q, oq, rw_q)], axis=2)
    right_s = np.concatenate([S._local_basis(gq, ne_q, oq, rw_q),
                              S._local_basis(gq, ne_q, oq, rw_q * gq.trho)], axis=2)
    inv_k2 = 1.0 / k ** 2
    jm = 1j * m
    tt, tf, ft, ff = targets
    chunk = S._local_rows(ne_p, op, table.shape[1])
    for e0 in range(0, ne_p, chunk):
        e1 = min(ne_p, e0 + chunk)
        rows = slice(e0 * op, e1 * op)
        g_lo = np.asarray(table[rows, :, abs(am - 1)], dtype=np.complex128)
        g_0 = np.asarray(table[rows, :, am], dtype=np.complex128)
        g_hi = np.asarray(table[rows, :, am + 1], dtype=np.complex128)
        gc = 0.5 * (g_lo + g_hi)
        gs = (g_lo - g_hi) / 2j
        if m < 0:
            gs = -gs
        del g_lo, g_hi
        Yg = S._contract_right(g_0, right_g)
        Yc = S._contract_right(gc, right_c)
        Ys = S._contract_right(gs, right_s)
        del g_0, gc, gs
        L = {name: value[e0:e1] for name, value in lp.items()}
        _grouped_left_into(L['rw_tr'], [(Yc[..., 0:2], tt, scale), (Ys[..., 0:2], tf, scale)], e0)
        _grouped_left_into(L['rw_tz'], [(Yg[..., 0:2], tt, scale)], e0)
        _grouped_left_into(L['dw'], [(Yg[..., 2:4], tt, -inv_k2 * scale), (Yg[..., 4:6], tf, -jm * inv_k2 * scale)], e0)
        _grouped_left_into(L['rw'], [(Ys[..., 2:4], ft, -scale), (Yc[..., 2:4], ff, scale)], e0)
        _grouped_left_into(L['w'], [(Yg[..., 2:4], ft, jm * inv_k2 * scale), (Yg[..., 4:6], ff, -(m * m) * inv_k2 * scale)], e0)


def _bracket_tables_into(targets, m, m_max, gp, ne_p, op, gq, ne_q, oq, tables, scale, source_weight=None):
    left = S._local_basis(gp, ne_p, op, gp.w * gp.rho)
    right_factor = gq.w * gq.rho if source_weight is None else gq.w * gq.rho * source_weight
    right = S._local_basis(gq, ne_q, oq, right_factor)
    chunk = S._local_rows(ne_p, op, tables[0].shape[1])
    index = abs(int(m))
    for table in tables:
        if index >= table.shape[-1]:
            raise ValueError("BoR bracket table was built for a smaller mode cap.")
    for e0 in range(0, ne_p, chunk):
        e1 = min(ne_p, e0 + chunk)
        rows = slice(e0 * op, e1 * op)
        pieces = []
        for uv, target in enumerate(targets):
            sign = (-1.0 if uv in (1, 2) and m < 0 else 1.0)
            Y = S._contract_right(np.asarray(tables[uv][rows, :, index], dtype=np.complex128), right)
            pieces.append((Y, target, 2.0 * np.pi * sign * scale))
        _grouped_left_into(left[e0:e1], pieces, e0)


def apply():
    S._efie_tables_into = _efie_tables_into
    S._bracket_tables_into = _bracket_tables_into
