"""Compressed 2-D backend: the TM partner operator stays in RAM when both operators fit comfortably.

``build_pair`` finalizes the TE and TM operators from each shared tile query and spools the TM
partner to disk (one SHA-256 per array, written and read back) so that only one operator is
resident while TE is solved.  When the forecast allowance of both operators is at most half of
the compressed storage budget, the partner is kept in RAM instead (no spool write, no read-back,
no digests); the spool remains for everything larger.
"""
from ghost_backend.compressed import polarization_cache as pc

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['air_s3c']
STATS = dict(kept=0, spooled=0)
_original = pc.build_pair


def _forecast_allowance_bytes():
    from ghost_backend.twod.assembly.session import current_session
    session = current_session()
    cached = getattr(session, 'memory_storage', None) if session is not None else None
    if not cached:
        return None
    values = [entry.get('operator_allowance_bytes') for entry in cached.values() if isinstance(entry, dict)]
    values = [int(v) for v in values if v]
    return max(values) if values else None


def build_pair(oracle, coordinates, tile=512, budget=512*1024**2, checkpoint=None, spool_directory=None):
    if spool_directory is not None:
        allowance = _forecast_allowance_bytes()
        if allowance is not None and 2 * allowance <= 0.5 * budget:
            spool_directory = None
            STATS['kept'] += 1
        else:
            STATS['spooled'] += 1
    return _original(oracle, coordinates, tile=tile, budget=budget, checkpoint=checkpoint,
                     spool_directory=spool_directory)


def _report():
    if STATS['kept'] or STATS['spooled']:
        print('[experiment] compressed_tm_in_ram', dict(STATS), flush=True)


def apply():
    import atexit
    pc.build_pair = build_pair
    # The runtime loads the partner before querying it; a resident operator has nothing to load.
    if not hasattr(pc.StreamedOperator, 'load'):
        pc.StreamedOperator.load = lambda self: None
    atexit.register(_report)
