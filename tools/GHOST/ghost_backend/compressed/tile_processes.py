"""Assemble and compress compressed-operator tiles in worker processes.

Tile assembly is Python-heavy, so threads contend for the interpreter lock. Each
worker process holds its own copy of the coefficient oracle and a kernel-table
state, assembles whole tiles and returns them already compressed; the caller
stores results in tile order exactly as the in-process path does.
"""
from contextlib import contextmanager
from collections import deque
import multiprocessing as mp
import os
import pickle
import sys

MIB = 1024**2
# Interpreter, libraries, kernel tables and one tile's assembly workspace.
WORKER_BYTES = 384*MIB
MAX_WORKERS = 8
# Smaller operators finish before worker start-up would pay for itself.
MIN_TILES = 256
COUNTERS = ('calls', 'entries', 'dropped_routes')

_WORKER = None


@contextmanager
def _without_main_module():
    """Start workers without re-importing the caller's __main__.

    Workers need only ghost_backend; importing a GUI entry point is wasted
    start-up, and an unguarded script would run its solve again in every worker.
    """
    main = sys.modules.get('__main__')
    saved = {name: getattr(main, name) for name in ('__spec__', '__file__') if hasattr(main, name)}
    try:
        if main is not None:
            main.__spec__ = None
            if '__file__' in saved:
                del main.__file__
        yield
    finally:
        for name, value in saved.items():
            setattr(main, name, value)


def _ignore_checkpoint():
    pass


def _compressor(operator):
    """Picklable stand-in exposing only StreamedOperator.compress_tile."""
    from ghost_backend.compressed.operator import StreamedOperator
    shell = StreamedOperator.__new__(StreamedOperator)
    shell.__dict__.update(groups=operator.groups, group_bounds=operator.group_bounds,
                          tolerance=operator.tolerance, compression=operator.compression,
                          checkpoint=_ignore_checkpoint)
    return shell


def _sources(oracle):
    return list(getattr(oracle, 'oracles', [oracle]))


def prepare(oracle, operators):
    """(worker count, worker payload) for this operator, or (0, None) to stay in process."""
    from ghost_backend.execution.options import current_options, environment_value, effective_assembly_threads
    tiles = len(operators[0].groups)**2
    if (not getattr(oracle, 'process_tiles', False) or tiles < MIN_TILES or
            mp.current_process().daemon or current_options() is None):
        return 0, None
    configured = environment_value('GHOST_TILE_PROCESSES', '').strip()
    allocation = effective_assembly_threads()
    count = min(MAX_WORKERS, allocation, int(configured) if configured else allocation)
    if count < 2:
        return 0, None
    payload = pickle.dumps((oracle, [_compressor(op) for op in operators], dict(current_options(), assembly_threads=1)),
                           protocol=pickle.HIGHEST_PROTOCOL)
    from ghost_backend.twod.solver import _solve_memory_limit_gb
    from ghost_backend.compressed.runtime import storage_budget
    spare = int(_solve_memory_limit_gb()*1024**3) - storage_budget()
    count = min(count, max(0, spare)//(WORKER_BYTES + 2*len(payload)))
    return (int(count), payload) if count >= 2 else (0, None)


def _initialize(payload):
    global _WORKER
    from ghost_backend.execution.thread_control import threadpool_limits
    from ghost_backend.execution.cpu import CPUState
    import ghost_backend.twod.assembly.native.far as far
    import scipy.linalg  # noqa: F401 -- SciPy's own BLAS must be loaded before limiting
    oracle, compressors, options = pickle.loads(payload)
    # Workers are the unit of parallelism: one BLAS thread and unsplit far blocks.
    limits = threadpool_limits(1)
    far.SPLIT_WORKERS = 1
    _WORKER = oracle, compressors, options, CPUState(), limits


def _tile(task):
    from ghost_backend.execution.cpu import _STATE
    from ghost_backend.execution.options import execution_scope
    i, j, missing = task
    oracle, compressors, options, state, _ = _WORKER
    groups = compressors[0].groups
    sources = _sources(oracle)
    before = [[getattr(o, name) for name in COUNTERS] for o in sources]
    with execution_scope(options, assembly_threads=1), _STATE.override(state):
        from ghost_backend.compressed.operator import tile_values
        values = tile_values(oracle, groups[i], groups[j], missing)
    counters = [([getattr(o, name)-old for name, old in zip(COUNTERS, previous)], o.max_entries)
                for o, previous in zip(sources, before)]
    return {index: compressors[index].compress_tile(i,j,*value) for index,value in values.items()}, counters


def compressed_tiles(oracle, operators, workers, payload, checkpoint):
    """Yield per-tile compress_tile results, one per operator, in assembly order.

    The parent oracle's query counters are advanced as the workers report them.
    If a worker process dies, the remaining tiles are assembled in this process.
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError
    from concurrent.futures.process import BrokenProcessPool
    from ghost_backend.execution.runtime import single_thread_worker_environment
    groups = operators[0].groups
    def tiles():
        return ((i, j, [index for index,op in enumerate(operators) if (i,j) not in op.pilot_tiles])
                for j in range(len(groups)) for i in range(len(groups)))
    remaining = iter(tiles())
    sources = _sources(oracle)
    done = 0
    pending = deque()
    window = max(1, 2*int(workers))
    from ghost_backend.compressed.worker_pool import acquire, run_tile
    owner = acquire(workers)
    path = owner.prepare(payload) if owner is not None else None
    executor = owner.executor if owner is not None else ProcessPoolExecutor(
        workers, mp_context=mp.get_context('spawn'), initializer=_initialize, initargs=(payload,))
    successful = False
    def fill():
        # Starting a process imports numerical runtimes before its initializer.
        # Keep both bootstrap protections around every bounded submission.
        with _without_main_module(), single_thread_worker_environment():
            while len(pending) < window:
                task = next(remaining, None)
                if task is None:
                    break
                checkpoint()
                from concurrent.futures import Future
                if task[2]:
                    future = executor.submit(run_tile,path,task) if owner is not None else executor.submit(_tile,task)
                else:
                    future = Future();future.set_result(({}, [([0]*len(COUNTERS),o.max_entries) for o in sources]))
                pending.append((future,task))
    try:
        fill()
        while pending:
            checkpoint()
            future,task = pending[0]
            while True:
                try:
                    compressed, counters = future.result(timeout=.1)
                    break
                except TimeoutError:
                    checkpoint()
            pending.popleft()
            for source, (deltas, largest) in zip(sources, counters):
                for name, delta in zip(COUNTERS, deltas):
                    setattr(source, name, getattr(source, name)+delta)
                source.max_entries = max(source.max_entries, largest)
            done += 1
            from ghost_backend.compressed.operator import take_pilot
            yield [compressed[index] if index in compressed else take_pilot(op,*task[:2])
                   for index,op in enumerate(operators)]
            # The consumer has stored/released the preceding result before
            # another job enters the window. Futures cannot retain the full
            # operator behind a slow early tile.
            del compressed, counters, future
            fill()
        successful = True
    except BrokenProcessPool:
        pass
    finally:
        for future,task in pending:
            future.cancel()
        pending.clear()
        if owner is not None:
            if not successful:
                owner.close()
        else:
            for process in list((getattr(executor, '_processes', None) or {}).values()):
                if process.is_alive():
                    process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
    from itertools import islice
    for i, j, missing in islice(tiles(), done, None):
        checkpoint()
        from ghost_backend.compressed.operator import tile_values, take_pilot
        values = tile_values(oracle,groups[i],groups[j],missing) if missing else {}
        yield [op.compress_tile(i,j,*values[index]) if index in values else take_pilot(op,i,j)
               for index,op in enumerate(operators)]
