"""Audit fixes of 2026-09-27 (see AUDIT_FIXES_2026-09-27.md).

Each test class checks one correctness fix of the 27 September audit against
an independent reference, an invariance, or the arithmetic it must reproduce.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.operators as ops
from ghost_backend.twod.geometry import LinearElement

DENSE_P1 = dict(factorization='dense', mesh_strategy='global', basis_order=1,
                assembly_threads=1, blas_threads=1)


def _element(p0, p1, index):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    tangent = (p1 - p0) / length
    return LinearElement(name='e', seg_type=2, ibc_flag=0, pos_mat=0, neg_mat=0,
                         node_ids=(2 * index, 2 * index + 1), p0=p0, p1=p1,
                         center=0.5 * (p0 + p1), tangent=tangent,
                         normal=np.array([-tangent[1], tangent[0]]), length=length,
                         panel_index=index)


def _pairs(points):
    return [dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
            for a, b in zip(points[:-1], points[1:])]


class CollinearNearPairTests(unittest.TestCase):
    """R-3: collinear separated-near pairs converge on any orientation."""

    def pair(self, angle_deg, origin=(0.3, 0.8), length=0.015, gap=0.002):
        d = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
        o = np.asarray(origin, float)
        return (_element(o, o + length * d, 0),
                _element(o + (length + gap) * d, o + (2 * length + gap) * d, 1))

    def test_rotated_collinear_pair_converges_with_a_null_double_layer(self):
        k = 2 * np.pi / 0.3
        aligned = ops._integrate_linear_pair_adaptive_sk(*self.pair(0.0), k, True, 16, 16)
        for origin in ((0.3, 0.8), (0.0, 0.0), (-40.0, 25.0)):
            obs, src = self.pair(30.0, origin)
            self.assertTrue(ops.requires_adaptive(obs, src))
            s_block, k_block = ops._integrate_linear_pair_adaptive_sk(obs, src, k, True, 16, 16)
            np.testing.assert_allclose(s_block, aligned[0], rtol=1e-11, atol=0)
            self.assertLess(np.linalg.norm(k_block),
                            1e-12 * np.linalg.norm(s_block) / obs.length)

    def test_double_layer_only_request_returns_no_single_layer(self):
        obs, src = self.pair(30.0)
        s_block, k_block = ops._integrate_linear_pair_adaptive_sk(
            obs, src, 2 * np.pi / 0.3, False, 16, 16, compute_single_layer=False)
        self.assertFalse(np.any(s_block))
        self.assertLess(np.linalg.norm(k_block), 1e-15)

    def test_inclined_side_with_a_sliver_primitive_solves_like_the_aligned_one(self):
        def snapshot(angle_deg):
            c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
            rotation = np.array([[c, -s], [s, c]])
            points = [rotation @ np.array(p, float) for p in
                      ([0, 0], [0, 1], [0.4, 1], [0.402, 1], [1, 1], [1, 0], [0, 0])]
            points[-1] = points[0]
            return dict(segments=[dict(name='sq', seg_type=2, properties=['2', '0', '0', '0', '0'],
                                       point_pairs=_pairs(points))], ibcs=[], dielectrics=[])

        def amplitudes(angle_deg, elevations):
            result = rcs.solve_monostatic_rcs_2d_single_polarization(
                snapshot(angle_deg), [1.0], elevations, polarization='TM',
                geometry_units='meters', strict_quality_gate=False, execution_options=DENSE_P1)
            return np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in result['samples']])

        aligned = amplitudes(0.0, [0.0, 20.0, 45.0])
        rotated = amplitudes(30.0, [30.0, 50.0, 75.0])
        np.testing.assert_allclose(np.abs(rotated), np.abs(aligned), rtol=1e-9)


class SheetBoundWaveMeshTests(unittest.TestCase):
    """A-3: a free TYPE 1 card is meshed for the bound wave of twice its law."""

    LAMBDA = 299_792_458.0 / 1e9

    def card(self, per_side=None):
        h = self.LAMBDA / 2
        corners = np.array([[-h, -h], [-h, h], [h, h], [h, -h], [-h, -h]])
        if per_side is None:
            points, n = corners, 0
        else:
            points = [a + t * (b - a) for a, b in zip(corners[:-1], corners[1:])
                      for t in np.arange(per_side) / per_side]
            points, n = np.array(points + [corners[-1]]), 1
        return dict(name='card', seg_type=1, properties=['1', str(n), '1', '0', '0'],
                    point_pairs=_pairs(points))

    def solve(self, card, reactance):
        law = [['1', 'constant', '2.0', repr(float(reactance)), '0', '0']]
        result = rcs.solve_monostatic_rcs_2d_single_polarization(
            dict(segments=[card], ibcs=law, dielectrics=[]), [1.0], [0.0, 15.0, 30.0, 45.0],
            polarization='TE', geometry_units='meters', strict_quality_gate=False,
            execution_options=DENSE_P1, max_panels=20000)
        amplitudes = np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in result['samples']])
        return amplitudes, result['metadata']['panel_count']

    def test_index_is_that_of_an_opaque_surface_of_twice_the_impedance(self):
        from ghost_backend.twod import geometry as geo
        z = 2.0 + 1000.0j
        self.assertAlmostEqual(geo._bound_surface_wave_index(2 * z),
                               math.sqrt(1 + (2000.0 / 376.730313668) ** 2), places=2)

    def test_inductive_card_is_meshed_for_its_bound_wave(self):
        resistive_panels = self.solve(self.card(), 0.0)[1]
        reference = self.solve(self.card(360), 600.0)[0]
        amplitudes, panels = self.solve(self.card(), 600.0)
        self.assertGreaterEqual(panels, 3 * resistive_panels)
        # 6.4 % on the 80-panel mesh the card had before.
        self.assertLess(np.max(np.abs(amplitudes - reference)) / np.max(np.abs(reference)), 5e-3)


class ThinLayerCurvatureGuardTests(unittest.TestCase):
    """R-4: the thin-layer curvature guard measures the drawing, not the mesh."""

    def polygon(self, radius, count, per_primitive):
        theta = np.linspace(0.0, -2 * np.pi, count + 1)
        points = radius * np.column_stack((np.cos(theta), np.sin(theta)))
        points[-1] = points[0]
        return dict(segments=[dict(name='layer', seg_type=1,
                                   properties=['1', str(per_primitive), '1', '0', '0'],
                                   point_pairs=_pairs(points))],
                    ibcs=[['1', 'thin_dielectric', '0.0012', '2']],
                    dielectrics=[['2', '10', '-1', '1', '0']])

    def solve(self, snapshot):
        result = rcs.solve_monostatic_rcs_2d_single_polarization(
            snapshot, [1.5], [0.0], polarization='TM', geometry_units='meters',
            strict_quality_gate=False, execution_options=DENSE_P1)
        return complex(result['samples'][0]['rcs_amp_real'], result['samples'][0]['rcs_amp_imag'])

    def test_splitting_primitives_keeps_a_faceted_circle_admissible(self):
        coarse = self.solve(self.polygon(0.1, 64, 1))
        for per_primitive in (5, 8):
            fine = self.solve(self.polygon(0.1, 64, per_primitive))
            self.assertLess(abs(fine - coarse) / abs(coarse), 1e-5)

    def test_curvature_of_the_drawing_does_not_depend_on_the_split(self):
        from ghost_backend.twod.formulations import thin_layer
        values = []
        for per_primitive in (1, 6):
            points = np.array([[0.1 * math.cos(t), 0.1 * math.sin(t)]
                               for t in np.linspace(0.0, -2 * np.pi, 65)])
            panels = []
            for a, b in zip(points[:-1], points[1:]):
                for i in range(per_primitive):
                    p = a + (b - a) * i / per_primitive
                    q = a + (b - a) * (i + 1) / per_primitive if i + 1 < per_primitive else b
                    length = float(np.linalg.norm(q - p))
                    tangent = (q - p) / length
                    panels.append(rcs.Panel('layer', 1, 1, 0, 0, p, q, (p + q) / 2, tangent,
                                            np.array([-tangent[1], tangent[0]]), length))
            mesh = thin_layer._continuous_oriented_mesh(rcs._build_linear_mesh(panels))
            values.append(thin_layer._drawing_curvature(mesh))
        self.assertAlmostEqual(values[1], values[0], delta=1e-6 * values[0])
        self.assertAlmostEqual(values[0], 1 / 0.1, delta=0.01 / 0.1)

    def test_a_tightly_curved_layer_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, 'curvature radius exceeds 0.05'):
            self.solve(self.polygon(0.012, 64, 1))


class JunctionPhiBasisTests(unittest.TestCase):
    """A-1: junctions tie the through (t) currents only; phi currents stay free.

    A vacuum coating or patch must be invisible, so its error must track the
    mesh without it. Tied phi currents made it 10x (partial coating) and 2x
    (patch) worse at 6 elements per hemisphere.
    """

    ASPECTS = [0., 30., 60., 90., 120., 150., 180.]

    @staticmethod
    def hemisphere(a, n, upper=True, bulge=0.0):
        th = np.linspace(0.0, 0.5 * math.pi, n + 1) + (0.0 if upper else 0.5 * math.pi)
        r = a * (1.0 + bulge * np.sin(2 * th))
        return np.column_stack([r * np.sin(th), r * np.cos(th)])

    def max_error_db(self, result, reference):
        return max(abs(10 * math.log10(v / reference))
                   for v in list(result['sigma_vv']) + list(result['sigma_hh']))

    def test_vacuum_partial_coating_tracks_the_plain_conductor(self):
        from ghost_backend.bor import solver as bor
        from ghost_backend.bor.kernels import C0
        from ghost_backend.validation import sphere
        a, n = 0.1, 6
        f = 3.0 * C0 / (2 * math.pi * a)
        reference = sphere.sigma_pec_sphere(a, f)
        coated = bor.solve_bor_partial_coating(
            self.hemisphere(a, n, bulge=0.12), self.hemisphere(a, n), [self.hemisphere(a, n, False)],
            f, self.ASPECTS, eps_r=1.0, mu_r=1.0)
        plain = bor.solve_bor(bor.sphere_generatrix(a, 2 * n), f, self.ASPECTS, formulation='cfie')
        self.assertLess(self.max_error_db(coated, reference),
                        1.25 * self.max_error_db(plain, reference))

    def test_vacuum_patch_tracks_the_coated_sphere(self):
        from ghost_backend.bor import solver as bor
        from ghost_backend.bor.kernels import C0
        from ghost_backend.validation import sphere
        a_core, a, eps, n = 0.08, 0.1, 2.5 - 0.1j, 6
        f = 3.0 * C0 / (2 * math.pi * a)
        reference = sphere.sigma_coated_pec_sphere(a_core, a, eps, 1.0, f)
        patched = bor.solve_bor_coating_patch(
            self.hemisphere(a, n, bulge=0.12), self.hemisphere(a, n), [self.hemisphere(a, n, False)],
            bor.sphere_generatrix(a_core, 2 * n), f, self.ASPECTS, eps_inner=eps, mu_inner=1.0,
            eps_patch=1.0, mu_patch=1.0)
        coated = bor.solve_bor_coated_pec(bor.sphere_generatrix(a, 2 * n),
                                          bor.sphere_generatrix(a_core, 2 * n), f, self.ASPECTS, eps, 1.0)
        self.assertLess(self.max_error_db(patched, reference),
                        1.25 * self.max_error_db(coated, reference))


class BackendAdmissionTests(unittest.TestCase):
    """P-1: every backend that fits the budget stays a retry; compressed
    storage that cannot fit its cap is refused before assembly."""

    GIB = 1024 ** 3

    def test_budget_fitting_backends_outside_the_margin_stay_retries(self):
        from ghost_backend.execution.policy import rank_candidates
        candidates = dict(dense=dict(cost=1.0, peak_gb=7.94), compressed=dict(cost=1.4, peak_gb=4.0))
        # 9 GiB budget, 20 % margin: compressed leads, dense (fits 9 GiB) is its retry.
        self.assertEqual(rank_candidates(candidates, 9.0), ['compressed', 'dense'])
        self.assertEqual(rank_candidates(candidates, 12.0), ['dense', 'compressed'])
        self.assertEqual(rank_candidates(candidates, 7.0), ['compressed'])

    def plan(self, operator_bytes, storage_limit, partner):
        from ghost_backend.compressed import memory
        sample = dict(method='sampled', operator_bytes=operator_bytes,
                      operator_allowance_bytes=operator_bytes, sampled=True, samples=8)
        return memory.forecast(4000, 8000, 361, 64, 1, storage_limit,
                               dict(compressed_storage=sample, compressed_partner=partner))

    def test_forecast_prices_the_reserved_partner_against_the_cap(self):
        from ghost_backend.compressed import memory
        inverse = memory.inverse_storage(8000)[0]
        operator = 2 * self.GIB
        limit = operator + inverse + self.GIB     # one polarization fits, two do not
        alone = self.plan(operator, limit, partner=False)
        paired = self.plan(operator, limit, partner=True)
        self.assertTrue(alone['storage_fits'])
        self.assertFalse(paired['storage_fits'])
        self.assertEqual(paired['storage_required_bytes'], 2 * operator + inverse)
        # The spooled partner is priced against the cap, not as resident RAM.
        self.assertEqual(paired['peak_bytes'], alone['peak_bytes'])

    def test_unsampled_ceiling_cannot_refuse(self):
        from ghost_backend.compressed import memory
        plan = memory.forecast(4000, 8000, 361, 64, 1, self.GIB, dict(compressed_partner=True))
        self.assertTrue(plan['storage_fits'])

    def test_gate_refuses_before_assembly(self):
        paired = self.plan(2 * self.GIB, 3 * self.GIB, partner=True)
        with self.assertRaisesRegex(MemoryError, 'reserved polarization partner'):
            rcs._compressed_storage_gate(dict(memory_estimate=paired), 'The test solve')
        rcs._compressed_storage_gate(dict(memory_estimate=self.plan(self.GIB, 8 * self.GIB, False)), 'x')
        rcs._compressed_storage_gate({}, 'x')


class MemoryProbeTests(unittest.TestCase):
    """H-4 and F3: SLURM per-CPU memory, nested cgroups, page cache, zero
    headroom and Windows commit."""

    GIB = 1024 ** 3

    def test_per_cpu_memory_without_cpus_per_task_uses_the_node_allocation(self):
        import os
        from unittest import mock
        base = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
        env = dict(base, SLURM_MEM_PER_CPU='3900', SLURM_CPUS_ON_NODE='128', SLURM_JOB_ID='7')
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(rcs, '_allocation_rss_bytes', return_value=0) as rss:
            self.assertEqual(rcs._slurm_available_bytes(), 3900 * 128 * 1024 ** 2)
        # The job's memory on the node is shared by all of its local processes.
        self.assertEqual(rss.call_args[0][0], ('SLURM_JOB_ID',))

    def test_nested_groups_are_read_innermost_first(self):
        from ghost_backend.execution import cgroup
        text = '0::/system.slice/slurmstepd.scope/job_7/step_0\n'
        groups = cgroup.memory_groups(lambda path: text)
        self.assertEqual([g[0] for g in groups], [
            '/sys/fs/cgroup/system.slice/slurmstepd.scope/job_7/step_0/memory.max',
            '/sys/fs/cgroup/system.slice/slurmstepd.scope/job_7/memory.max',
            '/sys/fs/cgroup/system.slice/slurmstepd.scope/memory.max',
            '/sys/fs/cgroup/system.slice/memory.max'])
        v1 = cgroup.memory_groups(lambda path: '4:memory:/slurm/uid_1/job_7\n3:cpu,cpuacct:/x\n')
        self.assertEqual(v1[0][:2], ('/sys/fs/cgroup/memory/slurm/uid_1/job_7/memory.limit_in_bytes',
                                     '/sys/fs/cgroup/memory/slurm/uid_1/job_7/memory.usage_in_bytes'))
        self.assertEqual(cgroup.memory_groups(lambda path: '0::/\n'), [])

    def test_tightest_nested_headroom_excludes_page_cache(self):
        from unittest import mock
        from ghost_backend.execution import cgroup
        job = ('/job/memory.max', '/job/memory.current', '/job/memory.stat', 'inactive_file')
        slice_ = ('/slice/memory.max', '/slice/memory.current', '/slice/memory.stat', 'inactive_file')
        values = {'/job/memory.max': 8 * self.GIB, '/job/memory.current': 7 * self.GIB,
                  '/slice/memory.max': 64 * self.GIB, '/slice/memory.current': 10 * self.GIB}
        cache = {'/job/memory.stat': 3 * self.GIB, '/slice/memory.stat': 0}
        with mock.patch.object(rcs, '_cgroup_memory_groups', return_value=[job, slice_]), \
                mock.patch.object(rcs, '_read_cgroup_int', side_effect=values.get), \
                mock.patch.object(cgroup, 'stat_value', lambda read, path, key: cache[path]):
            # 8 GiB - (7 GiB - 3 GiB of reclaimable cache); the root probe saw none of it.
            self.assertEqual(rcs._cgroup_available_bytes(), 4 * self.GIB)

    def test_zero_headroom_is_not_overridden_by_an_explicit_budget(self):
        from unittest import mock
        from ghost_backend.execution.options import execution_scope, validate_options
        with execution_scope(validate_options(dict(ram_budget_gib=8.0))), \
                mock.patch.object(rcs, '_detect_available_gb', return_value=0.0):
            with mock.patch.object(rcs, '_available_memory_bounds', return_value=[0]):
                self.assertEqual(rcs._configured_solve_memory_limit_gb(), 0.0)
            with mock.patch.object(rcs, '_available_memory_bounds', return_value=[]):
                self.assertEqual(rcs._configured_solve_memory_limit_gb(), 8.0)

    def test_scheduler_takes_the_tighter_of_slurm_and_cgroup(self):
        import os
        from unittest import mock
        from ghost_backend.hpc import scheduler
        with mock.patch.dict(os.environ, {'SLURM_MEM_PER_NODE': str(64 * 1024)}), \
                mock.patch.object(scheduler, '_cgroup_limit_gib', return_value=8.0):
            self.assertEqual(scheduler.detect_memory_gb(), 8.0)

    def test_scheduler_reads_the_nested_cgroup_limit(self):
        from unittest import mock
        from ghost_backend.hpc import scheduler
        files = {'/proc/self/cgroup': '0::/system.slice/job_7\n',
                 '/sys/fs/cgroup/system.slice/job_7/memory.max': str(8 * self.GIB),
                 '/sys/fs/cgroup/system.slice/memory.max': 'max'}

        class FakePath:
            def __init__(self, path):
                self.path = str(path)

            def read_text(self):
                if self.path not in files:
                    raise OSError(self.path)
                return files[self.path]

        with mock.patch.object(scheduler, 'Path', FakePath):
            self.assertEqual(scheduler._cgroup_limit_gib(), 8.0)

    def test_windows_availability_is_bounded_by_commit(self):
        from types import SimpleNamespace
        from unittest import mock

        class Kernel32:
            @staticmethod
            def GlobalMemoryStatusEx(pointer):
                pointer._obj.ullAvailPhys = 12 * MemoryProbeTests.GIB
                pointer._obj.ullTotalPageFile = 40 * MemoryProbeTests.GIB
                pointer._obj.ullAvailPageFile = 5 * MemoryProbeTests.GIB
                return 1

        with mock.patch.object(rcs.os, 'name', 'nt'), \
                mock.patch.object(rcs.ctypes, 'windll', SimpleNamespace(kernel32=Kernel32()), create=True):
            self.assertEqual(rcs._windows_available_bytes(), 5 * self.GIB)
            self.assertEqual(rcs._windows_commit_available_bytes(), 5 * self.GIB)


class ProvenanceFingerprintTests(unittest.TestCase):
    """H-2 and H-3: fingerprints cover the build and the solver, not the host
    or the checkout's driver configuration."""

    def test_driver_configuration_and_gui_edits_keep_the_source_fingerprint(self):
        import tempfile
        from ghost_backend.execution.provenance import backend_source_fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ('run_hpc_monostatic.py', 'ui/__init__.py', 'ui/window.py',
                             'twod/solver.py', 'helper.py'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            before = backend_source_fingerprint(str(root))
            (root / 'run_hpc_monostatic.py').write_text('CONFIG = {"frequencies_ghz": [9.0]}')
            (root / 'ui/window.py').write_text('changed')
            self.assertEqual(before, backend_source_fingerprint(str(root)))
            (root / 'helper.py').write_text('changed')
            self.assertNotEqual(before, backend_source_fingerprint(str(root)))
            before = backend_source_fingerprint(str(root))
            (root / 'twod/solver.py').write_text('changed')
            self.assertNotEqual(before, backend_source_fingerprint(str(root)))

    def test_runtime_fingerprint_ignores_host_features(self):
        from unittest import mock
        import platform
        from ghost_backend.execution import provenance
        payload = provenance.runtime_environment_payload()
        self.assertNotIn('platform_release', payload)
        self.assertNotIn('platform_processor', payload)
        self.assertNotIn('found', payload['numpy_config'].get('SIMD Extensions', {}))
        before = provenance.runtime_environment_fingerprint()
        config = dict(np.__config__.CONFIG)
        simd = dict(config.get('SIMD Extensions', {}), found=['X86_V2'])
        with mock.patch.dict(np.__config__.CONFIG, {'SIMD Extensions': simd}), \
                mock.patch.object(platform, 'release', return_value='other-kernel'), \
                mock.patch.object(platform, 'processor', return_value='other-cpu'):
            self.assertEqual(before, provenance.runtime_environment_fingerprint())


class BorPreviewPricingTests(unittest.TestCase):
    """P-3 and P-4: BoR previews and planners price the factors and far
    stores the solve builds."""

    def test_mirror_symmetry_test_is_shared_by_solve_and_preview(self):
        from ghost_backend.bor import solver as bor
        sphere = bor.sphere_generatrix(0.1, 32)
        self.assertTrue(bor.generatrix_mirror_symmetric(sphere))
        self.assertTrue(bor.generatrix_mirror_symmetric(sphere, (75 - 20j, None)))
        taper = np.linspace(10.0, 50.0, len(sphere) - 1)
        self.assertFalse(bor.generatrix_mirror_symmetric(sphere, (taper,)))
        cone = np.column_stack([np.linspace(0.0, 0.1, 17), np.linspace(0.2, 0.0, 17)])
        cone = np.vstack([cone, [[0.0, 0.0]]])
        self.assertFalse(bor.generatrix_mirror_symmetric(cone))

    def spy_plans(self, call):
        from unittest import mock
        from ghost_backend.bor import solver as bor
        seen = []
        original = bor.plan_bor_mode_workers

        def spy(*args, **kwargs):
            seen.append(dict(hierarchical=kwargs.get('hierarchical', False),
                             mirrored=kwargs.get('mirrored', False)))
            return original(*args, **kwargs)

        with mock.patch.object(bor, 'plan_bor_mode_workers', spy):
            call()
        return seen

    def test_snapshot_preview_prices_mirror_halves_of_a_symmetric_conductor(self):
        from ghost_backend.bor import dispatch, solver as bor
        points = bor.sphere_generatrix(0.1, 64)
        pairs = [dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                 for a, b in zip(points[:-1], points[1:])]
        snapshot = dict(segments=[dict(name='s', seg_type=2, properties=['2', '0', '0', '0', '0'],
                                       point_pairs=pairs)], ibcs=[], dielectrics=[])
        seen = self.spy_plans(lambda: dispatch.estimate_bor_resources(
            snapshot, 3.0, [0.0, 90.0], geometry_units='meters', mesh_certification=False, workers=2))
        self.assertTrue(seen)
        self.assertEqual(seen[-1], dict(hierarchical=True, mirrored=True))

    def test_direct_preview_prices_the_factors_solve_bor_uses(self):
        from ghost_backend.bor import dispatch, solver as bor
        supplied = dict(points=bor.sphere_generatrix(0.1, 64), freq_hz=3.0e9, thetas_deg=[0.0, 90.0])
        seen = self.spy_plans(lambda: dispatch._resolve_direct_plan(supplied))
        self.assertEqual(seen[0], dict(hierarchical=True, mirrored=True))
        dielectric = dict(supplied, eps_r=4.0)
        seen = self.spy_plans(lambda: dispatch._resolve_direct_plan(dielectric))
        self.assertEqual(seen[0], dict(hierarchical=False, mirrored=False))

    def test_compressed_self_stores_are_fixed_and_do_not_shape_the_range(self):
        from unittest import mock
        from ghost_backend.bor import compressed_far as cf, streaming as st
        m_max, budget = 16, 0.012
        crosses = [(120, 90, True, False)]
        selfs = [st.self_stream_spec(120, True), st.self_stream_spec(120, True),
                 st.self_stream_spec(90, True)]
        with mock.patch.object(cf, 'FAR_COMPRESSION_MIN_NODES', 80):
            stores = sum(cf.estimate_compressed_far_gb(n, m_max, 'efie', True) for n in (120, 120, 90))
            block, held, workers = st.plan_combined_streaming_mode_block(m_max, selfs + crosses, budget, 4)
            alone = st.plan_combined_streaming_mode_block(m_max, crosses, budget, 4)
            self.assertEqual((block, workers), alone[::2])
            self.assertAlmostEqual(held, alone[1] + stores, places=12)
            self.assertAlmostEqual(st.compressed_self_store_gb(m_max, selfs + crosses), stores, places=12)
            # Never spilled: not part of the per-mode spill cost.
            self.assertAlmostEqual(st.combined_stream_mode_gb(m_max, selfs + crosses),
                                   st.combined_stream_mode_gb(m_max, crosses), places=12)
        # Below the compression threshold a self stream is a streamed block, as before.
        dense = st.plan_combined_streaming_mode_block(m_max, selfs + crosses, 8.0, 4)
        legacy = st.plan_combined_streaming_mode_block(
            m_max, [(n, n, True, False) for n in (120, 120, 90)] + crosses, 8.0, 4)
        self.assertEqual(dense, legacy)


class SpillReleaseFlushTests(unittest.TestCase):
    """P-5: POSIX drops spilled pages without a synchronous msync per range."""

    def release(self, os_name):
        from unittest import mock
        from ghost_backend.bor import streaming
        calls = []

        class Mapping:
            def flush(self, offset, size):
                calls.append('flush')

            def madvise(self, option, start, length):
                calls.append('drop')

        array = np.zeros(4 * streaming._SPILL_PAGE, dtype=np.complex128)
        holder = mock.MagicMock(wraps=array)
        holder._mmap = Mapping()
        holder.dtype = array.dtype
        holder.ctypes.data = 0
        with mock.patch.object(streaming.os, 'name', os_name), \
                mock.patch.object(streaming.mmap, 'MADV_DONTNEED', 4, create=True), \
                mock.patch.object(streaming, '_virtual_unlock', lambda: (lambda base, size: calls.append('unlock'))):
            streaming._release_spilled(holder, [(0, array.size)])
        return calls

    def test_posix_drops_pages_without_flushing(self):
        self.assertEqual(self.release('posix'), ['drop'])

    def test_windows_still_flushes_before_unlocking(self):
        self.assertEqual(self.release('nt'), ['flush', 'unlock'])


if __name__ == '__main__':
    unittest.main()
