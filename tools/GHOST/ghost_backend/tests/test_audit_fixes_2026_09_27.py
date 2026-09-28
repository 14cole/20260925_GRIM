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
        from unittest import mock
        from ghost_backend.twod import geometry
        # Sizing alone: the corners of both cards are also graded (fix 18).
        with mock.patch.object(geometry, 'EDGE_GRADING_MIN_PANELS', 10 ** 9):
            resistive_panels = self.solve(self.card(), 0.0)[1]
            panels = self.solve(self.card(), 600.0)[1]
        self.assertGreaterEqual(panels, 3 * resistive_panels)
        reference = self.solve(self.card(360), 600.0)[0]
        amplitudes = self.solve(self.card(), 600.0)[0]
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


# ---------------------------------------------------------------------------
# Second round: speed items and grading.


class ThinLayerSizingTests(unittest.TestCase):
    """T2-12: a thin layer is meshed for its guided mode, not its bulk wavelength."""

    def test_slab_mode_index_is_the_first_order_thin_slab_mode(self):
        from ghost_backend.twod import geometry as geo
        k0d = 2 * math.pi * 1.5e9 / 299_792_458.0 * 0.0003
        self.assertAlmostEqual(geo._thin_slab_mode_index(10.0, 1.0, 0.0003, 1.5),
                               math.sqrt(1 + (k0d * 9 / 2) ** 2), places=12)
        self.assertEqual(geo._thin_slab_mode_index(1.0, 1.0, 0.001, 3.0), 1.0)

    def test_thin_layer_polygon_gets_the_free_space_density(self):
        theta = np.linspace(0.0, -2 * np.pi, 65)
        points = 0.1 * np.column_stack((np.cos(theta), np.sin(theta)))
        points[-1] = points[0]

        def solve(per_primitive):
            snapshot = dict(segments=[dict(name='layer', seg_type=1,
                                           properties=['1', str(per_primitive), '1', '0', '0'],
                                           point_pairs=_pairs(points))],
                            ibcs=[['1', 'thin_dielectric', '0.0003', '2']],
                            dielectrics=[['2', '10', '-1', '1', '0']])
            result = rcs.solve_monostatic_rcs_2d_single_polarization(
                snapshot, [1.5], [0.0, 45.0, 90.0], polarization='TE', geometry_units='meters',
                strict_quality_gate=False, execution_options=DENSE_P1)
            amplitudes = np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in result['samples']])
            return amplitudes, result['metadata']['panel_count']

        reference, _ = solve(12)
        amplitudes, panels = solve(0)
        self.assertEqual(panels, 64)            # 256 at the bulk wavelength before
        self.assertLess(np.max(np.abs(amplitudes - reference)) / np.max(np.abs(reference)), 1e-4)


class StrideOrderedLuTests(unittest.TestCase):
    """T2-10: dense LU factors P A P^T with a stride permutation."""

    def matrix(self, n=301, seed=3):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) + 4 * np.eye(n)
        return np.asfortranarray(a), rng.standard_normal((n, 6)) + 1j * rng.standard_normal((n, 6))

    def test_permutation_and_in_place_application_are_exact(self):
        from ghost_backend.linalg import dense
        for n in (1, 2, 7, 64, 301):
            perm = dense.stride_permutation(n)
            self.assertEqual(sorted(perm.tolist()), list(range(n)))
            a = np.asfortranarray(np.arange(n * n, dtype=complex).reshape(n, n))
            expected = a[np.ix_(perm, perm)]
            dense._permute_in_place(a, perm)
            np.testing.assert_array_equal(a, expected)

    def test_memory_and_spooled_factors_solve_and_estimate_like_natural_order(self):
        import tempfile
        import scipy.linalg as la
        from ghost_backend.linalg.dense import DenseFactor
        from ghost_backend.execution.options import execution_scope, validate_options
        a, b = self.matrix()
        reference = np.linalg.solve(a, b)
        natural = rcs._equilibrated_condition_from_lu(a, *la.lu_factor(a))
        diagnostics = {}
        factor = DenseFactor(a.copy(order='F'), diagnostics=diagnostics)
        self.assertEqual(factor.event['lu_order'], 'stride')
        np.testing.assert_allclose(factor.solve(b), reference, rtol=1e-11, atol=1e-13)
        self.assertAlmostEqual(diagnostics['condition_est'] / natural, 1.0, places=8)
        with tempfile.TemporaryDirectory() as directory, \
                execution_scope(validate_options(dict(dense_residual_storage='disk',
                                                      temporary_directory=directory))):
            owned = a.copy(order='F')
            spooled = DenseFactor(owned, owned_matrix=True)
            self.assertEqual(spooled.event['residual_storage'], 'disk')
            np.testing.assert_allclose(spooled.solve(b), reference, rtol=1e-11, atol=1e-13)
            spooled.restore_original()
            np.testing.assert_array_equal(owned, a)
            spooled.close()

    def test_only_columns_above_the_gate_are_refined(self):
        from unittest import mock
        from ghost_backend.linalg.dense import DenseFactor
        a, b = self.matrix()
        factor = DenseFactor(a.copy(order='F'))
        first = factor._apply_inverse(b)
        residual = a @ first - b
        errors = (np.max(np.abs(residual), axis=0)
                  / (factor.matrix_inf * np.max(np.abs(first), axis=0) + np.max(np.abs(b), axis=0)))
        gate = float(np.median(errors))
        widths = []
        original = factor._apply_inverse

        def spy(rhs, return_residual=False):
            widths.append(rhs.shape[1])
            return original(rhs, return_residual=return_residual)

        with mock.patch.object(rcs, 'DENSE_LINEAR_BACKWARD_ERROR_MAX', gate), \
                mock.patch.object(factor, '_apply_inverse', spy):
            factor.solve(b)
        self.assertEqual(widths[0], b.shape[1])
        self.assertEqual(widths[1], int(np.sum(errors > gate)))

    def test_closed_p1_system_no_longer_needs_refinement(self):
        from unittest import mock
        import ghost_backend.twod.fields as fields
        t = np.linspace(0, -2 * np.pi, 513)
        points = np.column_stack((0.5 * np.cos(t), 0.15 * np.sin(t)))
        points[-1] = points[0]
        snapshot = dict(segments=[dict(name='e', seg_type=2, properties=['2', '-20', '0', '0', '0'],
                                       point_pairs=_pairs(points))], ibcs=[], dielectrics=[])
        captured = []
        original = fields._solve_fields

        def capture(mesh, matrix, k0, angles, rhs_builder, diagnostics, label, **kw):
            captured.append((np.array(matrix, copy=True),
                             rhs_builder(np.linspace(0, 360, 256, endpoint=False))))
            return original(mesh, matrix, k0, angles, rhs_builder, diagnostics, label, **kw)

        with mock.patch.object(fields, '_solve_fields', capture):
            rcs.solve_monostatic_rcs_2d_single_polarization(
                snapshot, [12.0], [0.0], polarization='TE', geometry_units='meters',
                strict_quality_gate=False, execution_options=DENSE_P1)
        from ghost_backend.linalg.dense import DenseFactor
        matrix, rhs = captured[0]
        diagnostics = {}
        DenseFactor(np.asfortranarray(matrix), diagnostics=diagnostics).solve(rhs)
        # Natural order: growth 277, 4 of 256 columns over 1e-12, one refinement step.
        self.assertEqual(diagnostics['linear_refinement_steps'], 0)
        self.assertLess(diagnostics['linear_backward_error'], 1e-13)


def _junction_square(side=3.2 * 0.299792458):
    """A 6.4-wavelength (1 GHz) square: PEC on three sides and half the fourth, 75-20j on the rest."""
    middle, top_right, bottom_right, bottom_left, top_left = (
        np.array(p) for p in ([0, side], [side, side], [side, -side], [-side, -side], [-side, side]))

    def segment(name, points, ibc):
        return dict(name=name, seg_type=2, properties=['2', '0', str(ibc), '0', '0'],
                    point_pairs=_pairs(points))
    return dict(segments=[segment('pec', [top_right, bottom_right, bottom_left, top_left, middle], 0),
                          segment('ibc', [middle, top_right], 1)],
                ibcs=[['1', 'constant', '75', '-20', '0', '0']], dielectrics=[])


class HpCoarseningTests(unittest.TestCase):
    """T2-1 with the A-5 guard: coarser hp candidates, checked one grading step deeper at junctions."""

    def test_check_mesh_grades_each_junction_side_deeper(self):
        from ghost_backend.twod import geometry as geo
        from ghost_backend.twod import adaptive_geometry as ag
        self.assertEqual(ag.HP_COARSENING, 6.)
        snapshot = dict(_junction_square(), _2d_hp_coarsening=ag.HP_COARSENING, _2d_hp_refinements={})
        wavelength = 0.299792458
        candidate = geo._build_panels(snapshot, 1.0, wavelength, frequencies_ghz=[1.0])
        check = geo._build_panels(dict(snapshot, _2d_hp_singular_levels=ag.HP_CHECK_SINGULAR_LEVELS),
                                  1.0, wavelength, frequencies_ghz=[1.0])
        # Two junctions (the ends of the impedance side), two sides each.
        self.assertEqual(len(check) - len(candidate), 4 * ag.HP_CHECK_SINGULAR_LEVELS)
        shortest = min(panel.length for panel in check)
        self.assertAlmostEqual(min(panel.length for panel in candidate) / shortest,
                               2 ** ag.HP_CHECK_SINGULAR_LEVELS, places=6)

    def test_forecast_and_solve_use_the_same_check_mesh(self):
        from unittest import mock
        from ghost_backend.twod import adaptive_geometry as ag
        from ghost_backend.twod.preparation import prepare_geometry
        snapshot = _junction_square()
        materials = prepare_geometry(snapshot, None, 'meters')[2]
        with mock.patch.object(ag, 'MIN_AUTOMATIC_REFERENCE_PANELS', 0):
            meshes = ag.candidate_meshes(snapshot, materials, 1.5, True, [1.0], 1.0)
        (_, base, base_degree), (_, check, check_degree) = meshes
        self.assertEqual((base_degree, check_degree), (2, 3))
        self.assertNotIn('_2d_hp_singular_levels', base)
        self.assertEqual(check['_2d_hp_singular_levels'], ag.HP_CHECK_SINGULAR_LEVELS)
        with mock.patch.object(ag, 'MIN_AUTOMATIC_REFERENCE_PANELS', 0):
            result = rcs.solve_monostatic_rcs_2d_certified(
                snapshot, [1.0], list(np.linspace(20.0, 160.0, 15)), geometry_units='meters',
                execution_options=dict(mesh_strategy='adaptive', factorization='dense',
                                       assembly_threads=2, blas_threads=2))
        steps = result['metadata']['adaptive_mesh']['steps']
        self.assertTrue(result['metadata']['adaptive_mesh']['used'])
        self.assertEqual(steps[1]['panels'] - steps[0]['panels'], 4 * ag.HP_CHECK_SINGULAR_LEVELS)


class UnitBlasTests(unittest.TestCase):
    """P-6: BoR driver units widen BLAS from the import-time pin to their CPU allocation."""

    @staticmethod
    def blas_threads():
        from ghost_backend.execution.thread_control import threadpool_info
        return max(int(pool['num_threads']) for pool in threadpool_info() if pool.get('user_api') == 'blas')

    def test_scope_widens_to_the_allocation_and_restores_the_pin(self):
        from ghost_backend.execution.thread_control import threadpool_limits
        from ghost_backend.execution.options import physical_core_count
        from ghost_backend.hpc import scheduler
        with threadpool_limits(limits=1, user_api='blas'):
            with scheduler.cpu_allocation_scope(3), scheduler.unit_blas_scope(True) as threads:
                self.assertEqual(threads, min(3, physical_core_count()))
                self.assertEqual(self.blas_threads(), threads)
            self.assertEqual(self.blas_threads(), 1)
            with scheduler.cpu_allocation_scope(3), scheduler.unit_blas_scope(False) as threads:
                self.assertIsNone(threads)
                self.assertEqual(self.blas_threads(), 1)

    def test_drivers_widen_automatic_units_only(self):
        from unittest import mock
        from ghost_backend.execution.thread_control import threadpool_limits
        from ghost_backend import run_local_bor, run_hpc_bor_monostatic
        self.assertIsNone(run_local_bor.BLAS_THREADS_PER_WORKER)
        self.assertIsNone(run_hpc_bor_monostatic.BLAS_THREADS_PER_WORKER)
        seen = []

        def solve(*_args, **_kwargs):
            seen.append(self.blas_threads())
            return 'written', 'path'
        with threadpool_limits(limits=1, user_api='blas'), \
                mock.patch.object(run_local_bor, '_solve_and_export', solve), \
                mock.patch.object(run_hpc_bor_monostatic, '_solve_and_export', solve):
            run_local_bor._solve_and_export_star(({}, {}, 'dir', 2, True))
            run_local_bor._solve_and_export_star(({}, {}, 'dir', 2, False))
            run_local_bor._solve_and_export_star(({}, {}, 'dir', 2))
            run_hpc_bor_monostatic._solve_and_export_star(({}, {}, 'base', 'run', 2, True))
            run_hpc_bor_monostatic._solve_and_export_star(({}, {}, 'base', 'run', 2, False))
        self.assertEqual(seen, [2, 1, 1, 2, 1])

    def test_configurations_accept_the_automatic_setting(self):
        from ghost_backend.runs.config import validate_settings
        from ghost_backend.hpc.bundle import BundleError, _validate_settings
        keys = {'BLAS_THREADS_PER_WORKER'}
        self.assertEqual(validate_settings({'BLAS_THREADS_PER_WORKER': None}, keys),
                         {'BLAS_THREADS_PER_WORKER': None})
        self.assertEqual(validate_settings({'BLAS_THREADS_PER_WORKER': 2}, keys)['BLAS_THREADS_PER_WORKER'], 2)
        with self.assertRaises(ValueError):
            validate_settings({'BLAS_THREADS_PER_WORKER': 0}, keys)
        self.assertIsNone(_validate_settings('bor', {'BLAS_THREADS_PER_WORKER': None})['BLAS_THREADS_PER_WORKER'])
        with self.assertRaises(BundleError):
            _validate_settings('bor', {'BLAS_THREADS_PER_WORKER': 0})


class CompressedProductThreadTests(unittest.TestCase):
    """F4: compressed products take their threads from the allocation, one BLAS thread each."""

    def operator(self, n=260):
        from ghost_backend.compressed.operator import StreamedOperator

        class Exact:
            dropped_routes = calls = entries = 0

            def __init__(self, a):
                self.a, self.n = a, len(a)

            def get_with_error(self, rows, cols):
                value = self.a[np.ix_(rows, cols)].copy()
                self.calls += 1
                self.entries += value.size
                return value, np.zeros(value.shape)
        rng = np.random.RandomState(17)
        u = rng.randn(n, 5) + 1j * rng.randn(n, 5)
        a = np.eye(n) * 20 + u @ (rng.randn(5, n) + 1j * rng.randn(5, n)) / n
        return a, StreamedOperator(Exact(a), np.arange(n)[:, None], tile=64)

    def test_threads_follow_the_allocation_and_results_do_not(self):
        from contextlib import contextmanager
        from unittest import mock
        from ghost_backend.compressed import operator as op
        from ghost_backend.execution import options
        from ghost_backend.execution.options import physical_core_count
        from ghost_backend.hpc.scheduler import cpu_allocation_scope
        a, operator = self.operator()
        b = np.arange(len(a) * 12).reshape(len(a), 12) * (1 + 0.5j)
        entered = []
        original = options.single_thread_blas

        @contextmanager
        def spy():
            entered.append(True)
            with original():
                yield
        results = {}
        with mock.patch.object(options, 'single_thread_blas', spy):
            for cpus in (1, 3):
                with cpu_allocation_scope(cpus):
                    self.assertEqual(op.product_threads(), min(cpus, physical_core_count(), op.MATMUL_WORKERS))
                    results[cpus] = operator.matmul(b)
                    operator.equilibrate()
        # One CPU: the serial product and a one-thread equilibration pool.
        self.assertEqual(len(entered), 3)
        scale = np.max(np.abs(a @ b))
        self.assertLess(np.max(np.abs(results[1] - results[3])) / scale, 1e-15)
        self.assertLess(np.max(np.abs(results[3] - a @ b)) / scale, 1e-13)


class BorChainFineMeshTests(unittest.TestCase):
    """TB-1: BoR certification refines each chain 1.5x, in mirror pairs."""

    @staticmethod
    def sphere(primitives, radius=0.1, halves=False):
        theta = np.linspace(0.0, np.pi, primitives + 1)
        rho, z = radius * np.sin(theta), radius * np.cos(theta)
        rho[0] = rho[-1] = 0.0
        z[0], z[-1] = radius, -radius
        pairs = _pairs(np.column_stack((rho, z)))
        parts = [pairs[:primitives // 2], pairs[primitives // 2:]] if halves else [pairs]
        return dict(segments=[dict(name='s%d' % i, seg_type=2, properties=['2', '0', '0', '0', '0'],
                                   point_pairs=part) for i, part in enumerate(parts)],
                    ibcs=[], dielectrics=[])

    def test_densely_drawn_sphere_is_refined_1_5x_and_stays_symmetric(self):
        import copy
        from ghost_backend.bor import dispatch
        from ghost_backend.bor.solver import generatrix_mirror_symmetric
        wavelength = 0.0299792458
        for primitives, halves in ((512, False), (511, False), (512, True)):
            with self.subTest(primitives=primitives, halves=halves):
                snapshot = self.sphere(primitives, halves=halves)
                fine = copy.deepcopy(snapshot)
                fine['_bor_certification_refinement_factor'] = 1.5
                fine['_bor_certification_base_segment_n'] = ['0'] * len(snapshot['segments'])
                base_chains = dispatch._chains_from_snapshot(snapshot, 1.0)
                fine_chains = dispatch._chains_from_snapshot(fine, 1.0)
                self.assertEqual(dispatch._run_element_count(base_chains, wavelength), primitives)
                self.assertEqual(dispatch._run_element_count(fine_chains, wavelength),
                                 math.ceil(1.5 * primitives))   # 2x (per primitive) before
                points = dispatch._mesh_generatrix(fine_chains, wavelength, 10 ** 5, 1e-12)[0]
                self.assertEqual(len(points) - 1, math.ceil(1.5 * primitives))
                self.assertTrue(generatrix_mirror_symmetric(points))

    def test_paired_spread_is_closed_under_reversal(self):
        from ghost_backend.twod.geometry import _mirror_paired_spread
        for size in (1, 2, 7, 12, 511):
            for count in range(0, size + 1, max(1, size // 5)):
                chosen = set(_mirror_paired_spread(list(range(size)), count))
                self.assertGreaterEqual(len(chosen), count)
                self.assertLessEqual(len(chosen), count + 1)
                self.assertEqual(chosen, {size - 1 - i for i in chosen})


class EdgeGradingTests(unittest.TestCase):
    """A-4: the linear 2D mesh grades free ends, branch points and corners like junctions."""

    WAVELENGTH = 0.299792458

    def snapshot(self, points, kind=2, n=-20, ibcs=(), flag=0, **extra):
        return dict(segments=[dict(name='s', seg_type=kind, properties=[str(kind), str(n), str(flag), '0', '0'],
                                   point_pairs=_pairs(points))],
                    ibcs=[list(row) for row in ibcs], dielectrics=[['2', '10', '-1', '1', '0']], **extra)

    def polygon(self, sides, radius, rotate=0.0):
        t = rotate - np.linspace(0, 2 * np.pi, sides + 1)
        points = radius * np.column_stack((np.cos(t), np.sin(t)))
        points[-1] = points[0]
        return points

    def test_graded_vertices_and_untouched_ones(self):
        from ghost_backend.twod import geometry as geo
        lam = self.WAVELENGTH
        square = self.polygon(4, 2 * lam / math.sqrt(2), math.pi / 4)
        strip = np.array([[-lam, 0.0], [lam, 0.0]])

        def count(snapshot):
            return len(geo._build_panels(snapshot, 1.0, lam, frequencies_ghz=[1.0]))
        self.assertEqual(count(self.snapshot(square)), 160 + 4 * 8)
        self.assertEqual(count(self.snapshot(strip)), 40 + 2 * 4)             # one side per free end
        self.assertEqual(count(self.snapshot(self.polygon(64, lam))), 128)      # a drawn curve
        self.assertEqual(count(self.snapshot(square, n=10)), 40)                # explicit counts are kept
        small = self.polygon(4, 0.2 * lam / math.sqrt(2), math.pi / 4)
        self.assertEqual(count(self.snapshot(small)), 16)                       # 4-panel sides: under the minimum
        self.assertEqual(count(self.snapshot(square, n=0, _2d_hp_coarsening=6.0, _2d_hp_refinements={})), 28)
        thin = self.snapshot(strip, kind=1, flag=1, ibcs=[['1', 'thin_dielectric', '0.0003', '2']])
        self.assertEqual(count(thin), 40)                                       # qualified thin layer

    def test_strip_end_error_falls_16x(self):
        lam = self.WAVELENGTH
        strip = np.array([[-lam, 0.0], [lam, 0.0]])
        t = 0.5 * (1 - np.cos(np.pi * np.arange(601) / 600))
        reference_points = strip[0] + t[:, None] * (strip[1] - strip[0])
        angles = list(np.linspace(0, 180, 37))

        def solve(snapshot):
            result = rcs.solve_monostatic_rcs_2d_single_polarization(
                snapshot, [1.0], angles, polarization='TM', geometry_units='meters',
                strict_quality_gate=False, execution_options=DENSE_P1)
            return np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in result['samples']])
        reference = solve(self.snapshot(reference_points, n=1))
        error = np.max(np.abs(solve(self.snapshot(strip)) - reference)) / np.max(np.abs(reference))
        self.assertLess(error, 2.5e-4)          # 2.4e-3 on the uniform 40 panels


class RimGradingTests(unittest.TestCase):
    """A-2: BoR conductor generatrices are graded at rims, corners and tips."""

    A, L = 0.1, 0.2

    def snapshot(self, points):
        return dict(segments=[dict(name='body', seg_type=2, properties=['2', '0', '0', '0', '0'],
                                   point_pairs=_pairs(np.asarray(points, float)))], ibcs=[], dielectrics=[])

    def elements(self, points, wavelength):
        from ghost_backend.bor import dispatch
        chains = dispatch._chains_from_snapshot(self.snapshot(points), 1.0)
        groups, _tol, _axis_tol = dispatch._prepare_bor_groups(chains, dispatch._classify(chains))
        return dispatch._run_element_count(groups[0], wavelength)

    def test_rims_and_tips_are_graded_and_poles_are_not(self):
        from ghost_backend.bor.solver import sphere_generatrix
        from ghost_backend.bor.kernels import C0
        wavelength = 2 * math.pi * self.A / 3.0
        cylinder = [[0, self.L / 2], [self.A, self.L / 2], [self.A, -self.L / 2], [0, -self.L / 2]]
        self.assertEqual(self.elements(cylinder, wavelength), 40 + 2 * 8)
        sphere = sphere_generatrix(self.A, 64)
        self.assertEqual(self.elements(sphere, C0 / 3e9), 64)
        # A 30 degree cone tip on a flat base: the tip gains one side, the rim
        # only its slant side (the base, 6 elements, is below the minimum of 8).
        cone = [[0, self.L], [self.L * math.tan(math.radians(15)), 0], [0, 0]]
        lengths = [math.hypot(*np.subtract(q, p)) for p, q in zip(cone[:-1], cone[1:])]
        uniform = [max(1, math.ceil(length / (wavelength / 20))) for length in lengths]
        self.assertEqual(uniform[1], 6)
        self.assertEqual(self.elements(cone, wavelength), sum(uniform) + 4 + 4)
        # Explicit counts and corrugations keep their elements.
        explicit = self.snapshot(cylinder)
        explicit['segments'][0]['properties'][1] = '5'
        from ghost_backend.bor import dispatch
        chains = dispatch._chains_from_snapshot(explicit, 1.0)
        groups = dispatch._prepare_bor_groups(chains, dispatch._classify(chains))[0]
        self.assertEqual(dispatch._run_element_count(groups[0], wavelength), 15)

    def test_cylinder_rim_error_falls_15x(self):
        from ghost_backend.bor import dispatch
        from ghost_backend.bor import solver as bor
        from ghost_backend.bor.kernels import C0
        frequency = 3.0 * C0 / (2 * math.pi * self.A)
        aspects = [0., 20., 45., 70., 90., 110., 135., 160., 180.]
        step = C0 / frequency / 120

        def run(p0, p1, levels_start, levels_end):
            n = max(2, int(math.ceil(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step)))
            t = set(np.linspace(0, 1, n + 1)) | {0.5 ** k / n for k in range(1, levels_start + 1)} \
                | {1 - 0.5 ** k / n for k in range(1, levels_end + 1)}
            t = np.array(sorted(t))[:, None]
            return np.asarray(p0) + t * (np.subtract(p1, p0))
        a, half = self.A, self.L / 2
        reference_points = np.vstack([run((0, half), (a, half), 0, 4), run((a, half), (a, -half), 4, 4)[1:],
                                      run((a, -half), (0, -half), 4, 0)[1:]])
        reference = bor.solve_bor(reference_points, frequency, aspects, formulation='cfie')
        expected = np.concatenate([reference['sigma_vv'], reference['sigma_hh']])
        result = dispatch.solve_monostatic_rcs_bor_survey(
            geometry_snapshot=self.snapshot([[0, half], [a, half], [a, -half], [0, -half]]),
            frequencies_ghz=[frequency / 1e9], elevations_deg=aspects, geometry_units='meters')
        sigma = np.concatenate([[row['rcs_linear'] for row in result['co_solved_samples'][pol]]
                                for pol in ('VV', 'HH')])
        self.assertEqual(result['metadata']['per_frequency'][0]['mesh_elements_total'], 56)
        self.assertLess(np.max(np.abs(10 * np.log10(sigma / expected))), 0.006)   # 0.060 dB uniform


class BorOwnShareTests(unittest.TestCase):
    """H-1: a BoR array task runs its own share before it takes its peers' units."""

    def test_no_task_is_starved_by_a_fast_starter(self):
        import os
        import re
        import subprocess
        import tempfile
        from ghost_backend.bor import solver as bor
        from ghost_backend.hpc.common import configure_driver, latest_run_dir
        backend = ROOT / 'ghost_backend'
        points = bor.sphere_generatrix(0.1, 24)
        rows = '\n'.join('{:.12g} {:.12g} {:.12g} {:.12g}'.format(*a, *b) for a, b in zip(points[:-1], points[1:]))
        tasks, frequencies = 3, [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'geometry').mkdir()
            (work / 'geometry' / 'sphere.geo').write_text(
                'Title: PEC sphere\nSegment: pec 2\nproperties: 2 1 0 0 0\n' + rows
                + '\nIBCS_Resistances:\nDielectrics:\n')
            driver = configure_driver(backend / 'run_hpc_bor_monostatic.py', work / 'driver.py', dict(
                GEOMETRY_DIRS=[str(work / 'geometry')], FREQUENCIES_GHZ=frequencies, AZIMUTHS_DEG=[0.],
                ELEVATIONS_DEG=[0., 30.], OUTPUT_DIR=str(work / 'runs'), N_NODES=tasks, N_JOBS=1,
                GEOMETRY_UNITS='meters', MESH_CERTIFICATION=False, WORKERS_PER_UNIT=1, SUBMIT=False))
            env = dict(os.environ, PYTHONPATH=str(backend.parent), OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            submit = subprocess.run([sys.executable, str(driver)], env=env, cwd=work,
                                    capture_output=True, text=True, timeout=600)
            self.assertEqual(submit.returncode, 0, submit.stdout + submit.stderr)
            run_dir = latest_run_dir(work / 'runs')
            workers = [subprocess.Popen([sys.executable, str(driver), '--worker', str(run_dir), '0', str(task)],
                                        env=env, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True)
                       for task in range(tasks)]
            outputs = [worker.communicate(timeout=900)[0] for worker in workers]
            self.assertTrue(all(worker.returncode == 0 for worker in workers), '\n'.join(outputs))
            wrote = [int(re.search(r'wrote=(\d+)', output).group(1)) for output in outputs]
            planned = [int(re.search(r'planned here: (\d+)', output).group(1)) for output in outputs]
            pools = [int(re.search(r'pool: (\d+) x', output).group(1)) for output in outputs]
            self.assertEqual(sum(wrote), len(frequencies))
            self.assertTrue(all(count > 0 for count in wrote), wrote)
            # The pool was sized from every candidate (6), so a task that
            # started first could claim the whole run before its peers began.
            self.assertEqual(pools, planned)
            self.assertEqual(sum(planned), len(frequencies))


if __name__ == '__main__':
    unittest.main()
