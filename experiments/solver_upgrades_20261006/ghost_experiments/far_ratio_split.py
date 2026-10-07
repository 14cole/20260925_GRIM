"""2-D far tiles: pairs at centre-distance ratio >= 10 use the calibrated 'far' order of their tile.

``_far_tile`` grades one quadrature order per tile from the tile's longest panel and its closest
far pair (``ratio_min``), so a tile whose closest pair is at ratio 3 evaluates every pair, most of
them at ratio 10 and beyond, with the ratio-below-5 column of the calibration table (8 points on
the airfoil's air tiles where the far column gives 6-7).  Here the native block is evaluated
twice per tile: the pairs at ratio >= 10 with the table's far column for the tile's |k| L, the
rest with the tile's order as before; both columns were calibrated to 1e-12 against a 48-point
rule, so each pair keeps its calibrated accuracy.  W-requesting formulations and unequal
observation/source rules keep the single-rule path.

Implemented by rewriting the project's ``_assemble_multi`` source at import time (the tile
closure is not patchable on its own); the anchors must match the project text exactly.
"""
import inspect
import textwrap
from ghost_backend.twod import operators as ops

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['air_c1', 'air_c3', 'air_s3', 'coat_c10']

_OLD = '''        native = None
        # The native ABI currently takes one quadrature rule for both axes.
        # Unequal rules use the exact same kernels in the two-rule NumPy path.
        if native_far and tile_obs_order == tile_src_order:
            from ghost_backend.twod.assembly.native.far import far_block
            native = far_block(far_table, k0, obs_pts, src_pts, qw_obs, phi_obs_arr,
                               obs_norm, src_norm, any_far, obs_normal_deriv,
                               want_s, want_k, mirrored or want_d)
'''
_NEW = '''        native = None
        # The native ABI currently takes one quadrature rule for both axes.
        # Unequal rules use the exact same kernels in the two-rule NumPy path.
        if native_far and tile_obs_order == tile_src_order:
            from ghost_backend.twod.assembly.native.far import far_block
            far_column = max(rule_floor, _graded_far_order(kl_max, 10.0, far_obs_order, width - 1, attenuating))
            split_far = None
            if not want_w and far_column < tile_obs_order and any_far.any():
                split_far = any_far & ((centre_dist / scale) >= 10.0)
                if not split_far.any() or split_far.all():
                    split_far = None
            if split_far is None:
                native = far_block(far_table, k0, obs_pts, src_pts, qw_obs, phi_obs_arr,
                                   obs_norm, src_norm, any_far, obs_normal_deriv,
                                   want_s, want_k, mirrored or want_d)
            else:
                near_part = far_block(far_table, k0, obs_pts, src_pts, qw_obs, phi_obs_arr,
                                      obs_norm, src_norm, any_far & ~split_far, obs_normal_deriv,
                                      want_s, want_k, mirrored or want_d)
                t_far, qw_far, phi_far = _rule(far_column)
                obs_far = obs_p0[:, None, :] + t_far[None, :, None] * obs_seg[:, None, :]
                src_far = src_p0[:, None, :] + t_far[None, :, None] * src_seg[:, None, :]
                far_part = far_block(far_table, k0, obs_far, src_far, qw_far, phi_far,
                                     obs_norm, src_norm, split_far, obs_normal_deriv,
                                     want_s, want_k, mirrored or want_d)
                if near_part is not None and far_part is not None:
                    native = tuple(None if a is None else a + b for a, b in zip(near_part, far_part))
                    _FAR_SPLIT_STATS['split_tiles'] += 1
                elif near_part is None and far_part is None:
                    native = None
                else:
                    native = None
'''
_FAR_SPLIT_STATS = dict(split_tiles=0)


def apply():
    source = textwrap.dedent(inspect.getsource(ops._assemble_multi))
    if source.count(_OLD) != 1:
        raise RuntimeError('far_ratio_split: the _assemble_multi anchor was not found (project text changed).')
    namespace = ops.__dict__
    namespace['_FAR_SPLIT_STATS'] = _FAR_SPLIT_STATS
    exec(compile(source.replace(_OLD, _NEW), ops.__file__, 'exec'), namespace)
    import atexit, sys
    atexit.register(lambda: _FAR_SPLIT_STATS['split_tiles'] and print('[experiment] far_ratio_split', _FAR_SPLIT_STATS, file=sys.stderr, flush=True))
