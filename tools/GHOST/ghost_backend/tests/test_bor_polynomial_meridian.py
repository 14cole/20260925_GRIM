"""True polynomial BoR currents: basis, pole constraints and physical fields."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor
from ghost_backend.bor.polynomial import lagrange_basis, touching_points, PolynomialMeridian, solve_bor_polynomial
from ghost_backend.bor.options import option_scope, validate_options


class PolynomialBasisTests(unittest.TestCase):
    def test_partition_derivative_and_cubic_reproduction(self):
        x = np.linspace(0, 1, 31)
        for degree in (1,2,3):
            basis, derivative, nodes = lagrange_basis(degree,x)
            np.testing.assert_allclose(basis.sum(axis=0),1,atol=3e-15)
            np.testing.assert_allclose(derivative.sum(axis=0),0,atol=5e-15)
            for power in range(degree+1):
                np.testing.assert_allclose(nodes**power @ basis,x**power,atol=3e-15)
                expected = 0*x if power == 0 else power*x**(power-1)
                np.testing.assert_allclose(nodes**power @ derivative,expected,atol=5e-15)

    def test_true_interior_unknowns_and_regular_cartesian_poles(self):
        with option_scope(validate_options(dict(factorization='dense'))):
            surface = PolynomialMeridian(bor.sphere_generatrix(.02,8),1e9,3)
        self.assertEqual(surface.Nn,25)
        for mode in (-2,-1,0,1,2):
            q = surface.basis_transform(mode)
            values = q @ np.ones(q.shape[1])
            for node,element in ((0,0),(surface.Nn-1,surface.gen.n_elems-1)):
                if abs(mode) == 1:
                    self.assertEqual(values[surface.Nn+node],1j*mode*np.sign(surface.gen.trho[element]))
                else:
                    self.assertEqual(values[node],0)
                    self.assertEqual(values[surface.Nn+node],0)

    def test_degree_one_recovers_production_operator_fields(self):
        points = bor.sphere_generatrix(.025,10)
        common = dict(freq_hz=1e9,thetas_deg=[0.,90.,180.],n_modes=8,workers=1,
                      bor_options=dict(factorization='dense'))
        reference = bor.solve_bor(points,gauss_order=4,**dict(common,
            bor_options=dict(factorization='dense',near_refinement=2)))
        actual = solve_bor_polynomial(points,basis_degree=1,**common)
        for field in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[field],reference[field],rtol=3e-4,atol=1e-10)
        self.assertLessEqual(actual['polynomial_near_relative_change'],2e-5)

    def test_degree_one_same_quadrature_recovers_all_production_near_blocks(self):
        with option_scope(validate_options(dict(factorization='dense'))):
            surface = PolynomialMeridian(bor.sphere_generatrix(.025,8),1e9,1)
        for e,f in ((0,0),(3,3),(2,3),(3,2)):
            points = touching_points(e,f,12)
            actual = surface._near_blocks(e,f,3,points)
            reference = bor._contract_near_points(surface.gen,e,surface.gen,f,surface.k,3,
                ('efie','mfie'),points,signed=False)
            for kind in reference:
                np.testing.assert_allclose(actual[kind],reference[kind],rtol=3e-13,atol=1e-17)

    def test_cubic_fields_match_refined_linear_same_geometry(self):
        points = bor.sphere_generatrix(.025,12)
        refined = np.vstack(((points[:-1,None,:]+np.arange(4)[None,:,None]/4*
                              np.diff(points,axis=0)[:,None,:]).reshape(-1,2),points[-1]))
        common = dict(freq_hz=1e9,thetas_deg=[0.,45.,90.,150.,180.],n_modes=9,workers=1,
                      bor_options=dict(factorization='dense'))
        actual = solve_bor_polynomial(points,basis_degree=3,**common)
        reference = bor.solve_bor(refined,**common)
        for field in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[field],reference[field],rtol=.006,atol=2e-7)
        self.assertEqual(actual['n_unknowns'],74)
        self.assertLess(actual['n_unknowns'],reference['n_unknowns'])

    def test_closed_surface_and_supported_backend_are_required(self):
        with self.assertRaisesRegex(ValueError,'closed'):
            solve_bor_polynomial(bor.sphere_generatrix(.02,8)[1:-1],1e9,[0.],
                                 bor_options=dict(factorization='dense'))
        with self.assertRaisesRegex(ValueError,'dense'):
            solve_bor_polynomial(bor.sphere_generatrix(.02,8),1e9,[0.],
                                 bor_options=dict(factorization='compressed'))
