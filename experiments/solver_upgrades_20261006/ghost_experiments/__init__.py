"""GHOST solver experiments: overlays that monkeypatch ``tools/GHOST/ghost_backend`` in place.

Each experiment module defines ``DESCRIPTION``, ``CASES`` (the bench cases it affects) and
``apply()``; ``GHOST_EXPERIMENTS=name,name`` (or ``all``) applies them through ``sitecustomize``
in every process, including the solver's spawned worker processes.  Nothing here edits the
project: a passing experiment is ported by hand afterwards.
"""
import importlib
import os

EXPERIMENTS = {
    'bor_sampled_check': 'BoR graded near rules: coarse level evaluated on every 4th point of a layout group',
    'bor_sampled_meridian': 'BoR disjoint near pairs: coarse meridian level evaluated on every 4th pair of a batch',
    'twod_sampled_check': '2-D polynomial near rule: 20-point check on every 4th task of a batch',
    'dense_condition_probes': 'dense condition estimate: onenormest with four probe columns',
    'tables_contract_matmul': 'BoR dense-tables path: left contractions grouped per weight as batched matmuls',
    'far_grading_extension': '2-D far rule grading for attenuating media beyond |k| L = 3 (calibrated)',
    'compressed_tm_in_ram': 'compressed 2-D backend: keep the TM partner in RAM when it fits',
    'compressed_admission_threads': 'compressed 2-D backend: compress the admission samples on threads',
    'far_tile_glue': 'BoR far build: 2 pi folded into the bracket transform tables',
    'far_ratio_split': '2-D far tiles: pairs at ratio >= 10 evaluated with the calibrated far column',
}
APPLIED = []


def apply(name):
    if name in APPLIED:
        return
    if name not in EXPERIMENTS:
        raise KeyError('Unknown GHOST experiment: {}'.format(name))
    module = importlib.import_module('ghost_experiments.' + name)
    module.apply()
    APPLIED.append(name)


def apply_from_environment():
    names = os.environ.get('GHOST_EXPERIMENTS', '').strip()
    if not names:
        return
    selected = list(EXPERIMENTS) if names == 'all' else [n.strip() for n in names.split(',') if n.strip()]
    for name in selected:
        apply(name)
