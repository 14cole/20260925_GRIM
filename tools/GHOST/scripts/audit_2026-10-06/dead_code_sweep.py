"""Dead-code sweep of ghost_backend (applied on 6 October 2026; every anchor asserts, so a rerun on the swept tree stops at the first one): remove production definitions with no production callers,
keep test oracles by moving them beside the tests, delete research prototypes with their tests."""
import ast
import builtins
from pathlib import Path

ROOT = Path(r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST')
PKG = ROOT / 'ghost_backend'
TESTS = PKG / 'tests'


def _spans(path, names):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            found[node.name] = (start, node.end_lineno)
    missing = set(names) - set(found)
    assert not missing, (path, missing)
    return src, lines, found


def remove_defs(path, names):
    src, lines, found = _spans(path, names)
    keep = [True] * len(lines)
    for name, (start, end) in found.items():
        for i in range(start, end):
            keep[i] = False
        j = end
        while j < len(lines) and lines[j].strip() == '' and j - end < 2:
            keep[j] = False
            j += 1
    path.write_text(''.join(line for line, k in zip(lines, keep) if k), encoding='utf-8')
    print('removed', ', '.join(names), 'from', path.relative_to(ROOT))


def extract_defs(path, names):
    src, lines, found = _spans(path, names)
    return [''.join(lines[found[name][0]:found[name][1]]) for name in names]


def replace(path, pairs):
    text = path.read_text(encoding='utf-8')
    for old, new in pairs:
        assert text.count(old) >= 1, (path, old[:60])
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')
    print('updated', path.relative_to(ROOT))


# 1. Legacy BoR near rules become a test helper module.
kernels = PKG / 'bor' / 'kernels.py'
legacy_names = ['_modal_kernels_near_rule', '_project_pm_brackets', '_mfie_kernels_near_rule', '_ibc_kernels_near_rule']
sources = extract_defs(kernels, legacy_names)
module_tree = ast.parse(kernels.read_text(encoding='utf-8'))
module_level = set()
for node in module_tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        module_level.add(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    module_level.add(n.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            module_level.add((alias.asname or alias.name).split('.')[0])
used = set()
for source in sources:
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
needed = sorted(name for name in used & module_level if name not in legacy_names and not hasattr(builtins, name))
header = ('"""Legacy two-piece BoR near rules (the pre-graded NumPy samplers), kept beside the tests as\n'
          'independent references for the graded native rules of ``ghost_backend.bor.kernels``."""\n'
          'from ghost_backend.bor.kernels import (\n    ' + ',\n    '.join(needed) + ',\n)\n\n\n')
(TESTS / 'legacy_near_rules.py').write_text(header + '\n\n'.join(sources), encoding='utf-8')
print('wrote tests/legacy_near_rules.py importing', needed)
remove_defs(kernels, legacy_names)

# 2. The EFIE reciprocity diagnostic of the old storage becomes a helper of its only test.
solver = PKG / 'bor' / 'solver.py'
asymmetry = extract_defs(solver, ['_efie_near_asymmetry'])[0]
remove_defs(solver, ['_map_near_pairs', 'bor_basis_bytes', '_efie_near_asymmetry'])
reuse = TESTS / 'test_bor_compute_reuse.py'
replace(reuse, [
    ('class CompactModalStorageTests(unittest.TestCase):',
     asymmetry.rstrip('\r\n') + '\n\n\nclass CompactModalStorageTests(unittest.TestCase):'),
    ('bor._efie_near_asymmetry(', '_efie_near_asymmetry('),
])

# 3. Other production removals.
remove_defs(PKG / 'bor' / 'streaming.py', ['_contract_source_side'])
remove_defs(PKG / 'execution' / 'provenance.py', ['write_output_attestation', 'verify_output_attestation', '_artifact_path',
    'write_artifact_manifest', 'write_artifact_in_progress', 'verify_artifact_manifest', 'verify_component_output_manifest'])
remove_defs(PKG / 'twod' / 'solver.py', ['_residual_norm_many', '_make_elem_mask', '_solve_te_robin_mfie'])
remove_defs(PKG / 'io' / 'naming.py', ['format_base', 'group_solver_files'])
remove_defs(PKG / 'io' / 'grim.py', ['compute_linear_from_dbke'])
replace(PKG / 'twod' / 'basis.py', [
    ('''def mass_block(element):
    degree = len(element.node_ids) - 1
    return element.length * _reference_blocks(degree)[0]


def stiffness_block(element):
    degree = len(element.node_ids) - 1
    return _reference_blocks(degree)[1] / element.length


@lru_cache(maxsize=4)
def _reference_blocks(degree):
    q, w = np.polynomial.legendre.leggauss(degree + 1)
    phi = values((q + 1) / 2, degree)
    dphi = values((q + 1) / 2, degree, True)
    mass = (phi.T * (w / 2)) @ phi
    stiffness = (dphi.T * (w / 2)) @ dphi
    mass.flags.writeable = stiffness.flags.writeable = False
    return mass, stiffness
''',
     '''def mass_block(element):
    degree = len(element.node_ids) - 1
    return element.length * _reference_mass(degree)


@lru_cache(maxsize=4)
def _reference_mass(degree):
    q, w = np.polynomial.legendre.leggauss(degree + 1)
    phi = values((q + 1) / 2, degree)
    mass = (phi.T * (w / 2)) @ phi
    mass.flags.writeable = False
    return mass
'''),
])

# 4. Research prototypes and their tests.
for rel in ('ghost_backend/bor/polynomial.py', 'ghost_backend/compressed/analytic_far.py', 'ghost_backend/twod/nystrom.py',
            'ghost_backend/tests/test_bor_polynomial_meridian.py', 'ghost_backend/tests/test_analytic_far_prototype.py',
            'ghost_backend/tests/test_nystrom.py'):
    (ROOT / rel).unlink()
    print('deleted', rel)
replace(ROOT / 'scripts' / 'check_headless.py', [("'nystrom',", '')])

# 5. Driver imports that nothing uses.
for name in ('run_local_bor.py', 'run_local_monostatic.py', 'run_hpc_bor_monostatic.py', 'run_hpc_monostatic.py'):
    replace(PKG / name, [('import ghost_backend.execution.provenance as _workflow_provenance\n', '')])
replace(PKG / 'run_local_bor.py', [('''    configuration_source_records,
    copy_configuration,
)''', '''    configuration_source_records,
)''')])

# 6. Tests that referenced the moved or removed helpers.
replace(TESTS / 'test_bor_performance_fixes.py', [
    ('from ghost_backend.bor import solver as bor\n', 'from ghost_backend.bor import solver as bor\nimport legacy_near_rules as legacy\n'),
    ('kernels._mfie_kernels_near_rule', 'legacy._mfie_kernels_near_rule'),
    ('kernels._project_pm_brackets', 'legacy._project_pm_brackets'),
    ('bor._map_near_pairs(lambda p: p * 2, range(40), 4)', 'list(bor._iter_near_pairs(lambda p: p * 2, range(40), 4))'),
    ('bor._map_near_pairs(failing, range(40), 4)', 'list(bor._iter_near_pairs(failing, range(40), 4))'),
])
replace(TESTS / 'test_audit_fixes_bor_kernels.py', [
    ('from ghost_backend.bor import streaming\n', 'from ghost_backend.bor import streaming\nimport legacy_near_rules as legacy\n'),
    ('kernels._modal_kernels_near_rule', 'legacy._modal_kernels_near_rule'),
])
replace(TESTS / 'test_september_round12_near_rounding.py', [
    ('from ghost_backend.bor import kernels, solver as bor\n', 'from ghost_backend.bor import kernels, solver as bor\nimport legacy_near_rules as legacy\n'),
    ('kernels._mfie_kernels_near_rule', 'legacy._mfie_kernels_near_rule'),
    ('kernels._ibc_kernels_near_rule', 'legacy._ibc_kernels_near_rule'),
])
perf = TESTS / 'test_performance_updates.py'
text = perf.read_text(encoding='utf-8')
anchor = next(line for line in text.splitlines() if line.startswith('from ghost_backend') or line.startswith('import ghost_backend'))
replace(perf, [
    (anchor + '\n', anchor + '\nimport legacy_near_rules as legacy\n'),
    ('kernels._project_pm_brackets', 'legacy._project_pm_brackets'),
])
replace(TESTS / 'test_direct_solver_methods.py', [
    ('for solver in (rcs._solve_te_robin_mfie, rcs._solve_multi_region_indirect):', 'for solver in (rcs._solve_multi_region_indirect,):'),
])
replace(ROOT / 'BOR_PERFORMANCE.md', [('`_contract_source_side`): 209 ms', '`_contract_source_group`): 209 ms')])
print('sweep done')
