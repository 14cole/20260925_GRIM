"""Independent dense checks of the isolated, analytically bounded prototype."""
import numpy as np
import pytest
from scipy import sparse
from scipy.special import hankel2, jv

from ghost_backend.compressed.analytic_far import AnalyticFarRejected, graf_tail_bound, single_layer_block


def nodes(a=.3, radius=1.1, seed=97):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, 73)
    sources = np.column_stack((np.cos(theta), np.sin(theta)))*rng.uniform(0, a, len(theta))[:, None]
    theta = rng.uniform(-np.pi, np.pi, 61)
    targets = np.column_stack((np.cos(theta), np.sin(theta)))*rng.uniform(radius, radius*1.4, len(theta))[:, None]
    return targets, sources


@pytest.mark.parametrize('k,a,radius', [(0.1,.2,1.2), (3.,.3,1.1), (30.,.2,1.), (150.,.05,1.)])
def test_scalar_expansion_meets_analytic_budget_against_independent_dense_kernel(k, a, radius):
    targets, sources = nodes(a, radius)
    block = single_layer_block(targets, sources, k, center=[0, 0], absolute_tolerance=2e-10)
    from ghost_backend.twod.assembly.kernels import values
    exact = values(k, np.linalg.norm(targets[:, None]-sources[None, :], axis=2))[..., 0]
    # The separate small floating-point allowance acknowledges the documented
    # analytic-only scope; it is not part of the reported truncation bound.
    assert np.max(abs(block.dense()-exact)) <= block.evidence['analytic_kernel_tail_bound']+3e-13
    assert block.evidence['max_coefficient_tail_bound'] <= 2e-10
    assert block.evidence['coefficient_pair_evaluations'] == 0
    assert not block.evidence['production_certified']
    if k == 150:
        assert block.evidence['order'] < k*radius/2  # finite tail head avoids distant-target over-ranking


@pytest.mark.parametrize('order', [0, 2, 7, 16])
def test_uniform_bound_includes_all_angles_and_source_radii_not_only_sampled_boundary(order):
    k, a, radius = 5., .4, 1.1
    source_radii = np.linspace(0, a, 19)
    angles = np.linspace(-np.pi, np.pi, 101)
    y = source_radii[:, None]
    distance = np.sqrt(radius**2+y**2-2*radius*y*np.cos(angles))
    exact = hankel2(0, k*distance)
    orders = np.arange(-order, order+1)
    approximate = np.einsum('n,jn,nl->jl', hankel2(orders, k*radius), jv(orders[None, :], k*source_radii[:, None]),
                            np.exp(-1j*orders[:, None]*angles))
    assert np.max(abs(exact-approximate)) <= graf_tail_bound(k, a, radius, order)+3e-14


@pytest.mark.parametrize('use_sparse', [False, True])
def test_signed_weighted_galerkin_bounds_and_transpose_products(use_sparse):
    targets, sources = nodes()
    rng = np.random.default_rng(252)
    wt = rng.standard_normal((13, len(targets)))+1j*rng.standard_normal((13, len(targets)))
    ws = rng.standard_normal((len(sources), 17))+1j*rng.standard_normal((len(sources), 17))
    wt[rng.random(wt.shape) < .7] = 0
    ws[rng.random(ws.shape) < .7] = 0
    block = single_layer_block(targets, sources, 8., center=[0, 0],
        test_weights=sparse.csr_matrix(wt) if use_sparse else wt,
        source_weights=sparse.csr_matrix(ws) if use_sparse else ws, absolute_tolerance=1e-8)
    kernel = .25j*hankel2(0, 8*np.linalg.norm(targets[:, None]-sources[None, :], axis=2))
    exact = wt @ kernel @ ws
    error = abs(block.dense()-exact)
    assert np.all(error <= block.entry_tail_left[:, None]*block.entry_tail_right[None, :]+2e-12)
    assert np.all(error.sum(axis=1) <= block.row_tail_bound+4e-11)
    assert np.all(error.sum(axis=0) <= block.column_tail_bound+4e-11)
    for trans, matrix in ((0, exact), (1, exact.T), (2, exact.conj().T)):
        rhs = rng.standard_normal((matrix.shape[1], 3))+1j
        np.testing.assert_allclose(block.matmul(rhs, trans), matrix @ rhs, atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize('override', [dict(k=3+.01j), dict(k=-3), dict(kernel='double_layer'),
    dict(kernel='hypersingular'), dict(medium='lossy'), dict(medium='heterogeneous'),
    dict(max_order=0, absolute_tolerance=1e-15), dict(max_tail_terms=1)])
def test_unsupported_or_unqualified_work_rejects_instead_of_publishing_a_sampled_operator(override):
    targets, sources = nodes()
    arguments = dict(k=3., center=[0, 0])
    arguments.update(override)
    with pytest.raises(AnalyticFarRejected):
        single_layer_block(targets, sources, **arguments)


def test_overlap_and_nonfinite_inputs_reject():
    targets, sources = nodes()
    with pytest.raises(AnalyticFarRejected):
        single_layer_block(sources, sources, 3., center=[0, 0])
    targets[0, 0] = np.nan
    with pytest.raises(AnalyticFarRejected):
        single_layer_block(targets, sources, 3.)


def test_centered_sources_have_exact_zero_truncation_tail():
    targets, _ = nodes()
    sources = np.zeros((4, 2))
    block = single_layer_block(targets, sources, 3., center=[0, 0])
    exact = .25j*hankel2(0, 3*np.linalg.norm(targets, axis=1))[:, None]*np.ones((1, 4))
    np.testing.assert_allclose(block.dense(), exact, rtol=1e-14, atol=1e-14)
    assert block.evidence['order'] == 0
    assert block.evidence['analytic_kernel_tail_bound'] == 0
