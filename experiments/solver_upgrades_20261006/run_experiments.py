"""Run GHOST solver experiments: accuracy against the current project's fields, and timing.

py run_experiments.py reference                   # record the project's own fields and timings
py run_experiments.py run NAME[,NAME...] [--cases a,b] [--label L]
py run_experiments.py compare LABEL [--tol 1e-8]
py run_experiments.py list

Each run executes the bench cases in fresh processes with ``GHOST_EXPERIMENTS`` set and this folder
on ``PYTHONPATH`` (``sitecustomize.py`` applies the overlays in every process, the solver's spawned
workers included).  Results go to ``results/<label>/``; the project's reference to ``results/reference/``.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = (HERE / '..' / '..' / 'tools' / 'GHOST').resolve()
BENCH = ROOT / 'scripts' / 'audit_2026-10-06' / 'bench.py'
RESULTS = HERE / 'results'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import ghost_experiments  # noqa: E402


def _bench():
    spec = importlib.util.spec_from_file_location('ghost_bench', BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cases(label, experiments, cases):
    outdir = RESULTS / label
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(HERE) + os.pathsep + env.get('PYTHONPATH', '')
    if experiments:
        env['GHOST_EXPERIMENTS'] = ','.join(experiments)
    else:
        env.pop('GHOST_EXPERIMENTS', None)
    env.pop('GHOST_HIERARCHICAL_MIN_UNKNOWNS', None)
    for case in cases:
        log = outdir / (case + '.log')
        started = time.perf_counter()
        with open(log, 'w', encoding='utf-8') as handle:
            proc = subprocess.run([sys.executable, str(BENCH), 'case', case, str(outdir)], cwd=str(ROOT), env=env,
                                  stdout=handle, stderr=subprocess.STDOUT, timeout=3600)
        print('%-12s %-11s exit %d  %.1f s' % (label, case, proc.returncode, time.perf_counter() - started), flush=True)
        if proc.returncode:
            print(log.read_text(encoding='utf-8')[-2500:])
    (outdir / 'experiments.json').write_text(json.dumps(dict(experiments=experiments, cases=cases)), encoding='utf-8')


def compare(label, tol=1e-8, reference='reference'):
    import numpy as np
    ref, new = RESULTS / reference, RESULTS / label
    rows = []
    worst = 0.
    for path in sorted(new.glob('*.npz')):
        name = path.stem
        other = ref / path.name
        if not other.exists():
            rows.append((name, 'NO REF', None, None, None))
            continue
        a, b = np.load(other), np.load(path)
        ia = json.load(open(ref / (name + '.json'), encoding='utf-8'))
        ib = json.load(open(new / (name + '.json'), encoding='utf-8'))
        rel = 0.
        for key in a.files:
            if key.endswith('_amp'):
                rel = max(rel, float(np.max(np.abs(a[key] - b[key]))) / max(float(np.max(np.abs(a[key]))), 1e-300))
        worst = max(worst, rel)
        rows.append((name, 'PASS' if rel <= tol else 'FAIL', rel, ia.get('wall'), ib.get('wall')))
    print('%-12s %-7s %-10s %8s %8s %6s' % ('case', 'verdict', 'peak-rel', 'ref s', 'new s', 'ratio'))
    for name, verdict, rel, wr, wn in rows:
        if rel is None:
            print('%-12s %-7s' % (name, verdict))
        else:
            print('%-12s %-7s %-10.2e %8.2f %8.2f %6.2f' % (name, verdict, rel, wr or 0, wn or 0, (wn / wr) if wr and wn else float('nan')))
    print('worst peak-relative change: %.2e (tolerance %.0e)' % (worst, tol))
    return rows, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    ref = sub.add_parser('reference')
    ref.add_argument('--cases', default='')
    run = sub.add_parser('run')
    run.add_argument('experiments')
    run.add_argument('--cases', default='')
    run.add_argument('--label', default=None)
    cmp = sub.add_parser('compare')
    cmp.add_argument('label')
    cmp.add_argument('--tol', type=float, default=1e-8)
    cmp.add_argument('--reference', default='reference')
    args = parser.parse_args()
    bench = _bench()
    if args.command == 'list':
        for name, text in ghost_experiments.EXPERIMENTS.items():
            print('%-28s %s' % (name, text))
        return
    if args.command == 'reference':
        cases = [c for c in args.cases.split(',') if c] or [n for n, c in bench.CASES.items()]
        run_cases('reference', [], cases)
        return
    if args.command == 'run':
        names = list(ghost_experiments.EXPERIMENTS) if args.experiments == 'all' else [n for n in args.experiments.split(',') if n]
        cases = [c for c in args.cases.split(',') if c]
        if not cases:
            seen = []
            for name in names:
                module = importlib.import_module('ghost_experiments.' + name)
                for case in module.CASES:
                    if case not in seen:
                        seen.append(case)
            cases = seen
        label = args.label or '+'.join(names)
        run_cases(label, names, cases)
        compare(label)
        return
    compare(args.label, args.tol, args.reference)


if __name__ == '__main__':
    main()
