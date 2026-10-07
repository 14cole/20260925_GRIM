"""Compressed 2-D backend: the admission samples' tile compressions run on threads.

``memory.sample_operator`` prices the compressed operator from about 20 sampled tiles per
separation band: each sample queries the oracle (serial here, as in the project; the oracle's
column caches are not shared between threads) and then compresses the tile.  The compression
(``compress_tile`` acts on its own tile only) is the part that runs on a thread pool here, one
band at a time, with the tile order and the pilot records unchanged.
"""
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ghost_backend.compressed import memory as M

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['air_s3c']
THREADS = 8


def sample_operator(oracle, coordinates, tile=512, checkpoint=None, pilot_identity=None):
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.pilots import save
    checkpoint = checkpoint or (lambda: None)
    shell = StreamedOperator(oracle, coordinates, tile=tile, assemble=False,
        budget=max(16*M.MIB, 128*oracle.n), checkpoint=checkpoint)
    lengths = np.asarray([len(ids) for ids in shell.groups], dtype=np.int64)
    groups = len(lengths)
    diagonal = 16*int(np.dot(lengths, lengths))
    expected = allowance = float(shell.bytes + diagonal)
    bands = []
    rng = np.random.RandomState(1904)
    low = 1
    pool = ThreadPoolExecutor(max_workers=THREADS)
    try:
        while low < groups:
            checkpoint()
            high = min(2*low, groups)
            gaps = np.arange(low, high, dtype=np.int64)
            cumulative = np.cumsum(groups-gaps)
            population = 2*int(cumulative[-1])
            chosen = set()
            for upper in range(population-min(M.SAMPLES_PER_BAND,population), population):
                pick = int(rng.randint(0, upper+1, dtype=np.int64))
                chosen.add(upper if pick in chosen else pick)
            queried = []
            for picked in sorted(chosen):
                checkpoint()
                reverse = picked >= cumulative[-1]
                position = int(picked % cumulative[-1])
                index = int(np.searchsorted(cumulative, position, side='right'))
                gap = int(gaps[index])
                i = position-int(cumulative[index-1] if index else 0)
                j = i+gap
                if reverse:
                    i, j = j, i
                rows, cols = shell.groups[i], shell.groups[j]
                if hasattr(oracle, 'prepare_columns'):
                    oracle.prepare_columns(cols)
                raw, tail = oracle.get_with_error(rows, cols)
                if (raw.shape != (len(rows),len(cols)) or tail.shape != raw.shape or
                        not np.all(np.isfinite(raw)) or not np.all(np.isfinite(tail)) or np.any(tail<0)):
                    raise ValueError('Invalid coefficient tile in compressed memory forecast.')
                queried.append((i, j, rows, cols, raw, tail))
            compressed_tiles = list(pool.map(lambda q: shell.compress_tile(q[0], q[1], q[4], q[5]), queried))
            ratios = []
            for (i, j, rows, cols, raw, tail), compressed in zip(queried, compressed_tiles):
                payload = compressed[3]
                ratios.append(sum(a.nbytes for a in payload if a is not None)/float(raw.nbytes))
                save(pilot_identity, rows, cols, compressed)
            queried = compressed_tiles = None
            prefix = np.r_[0, np.cumsum(lengths)]
            indices = np.arange(groups)
            starts = np.minimum(groups, indices+low)
            stops = np.minimum(groups, indices+high)
            entries = 2*int(np.dot(lengths, prefix[stops]-prefix[starts]))
            mean = float(np.mean(ratios))
            sem = float(np.std(ratios, ddof=1)/math.sqrt(len(ratios))) if len(ratios)>1 else 0.
            upper = min(1., max(mean+2*sem, 1.15*mean, 2./max(1,int(lengths.max()))))
            if len(ratios) == population:
                upper = mean
            expected += 16*entries*mean
            allowance += 16*entries*upper
            bands.append(dict(first_gap=low, last_gap=high-1, samples=len(ratios),
                              mean_ratio=mean, allowance_ratio=upper))
            low = high
    finally:
        pool.shutdown(wait=True)
    return dict(method='sampled_spatial_tiles', operator_bytes=int(math.ceil(expected)),
        operator_allowance_bytes=int(math.ceil(allowance)), samples=sum(b['samples'] for b in bands),
        tile=tile, groups=groups, bands=bands, sampled=True)


def apply():
    M.sample_operator = sample_operator
