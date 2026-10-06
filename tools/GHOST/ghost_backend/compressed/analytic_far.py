"""Experimental, analytically bounded real-k 2D single-layer far blocks.

This module is deliberately not connected to production dispatch. It forms
separated Galerkin factors without evaluating the target/source distance grid.
Only truncation of the analytic expansion is bounded here: SciPy special-
function evaluation, roundoff, and production norm/scaling integration still
need qualification before these factors can replace the authoritative operator.

Graf's addition theorem: https://dlmf.nist.gov/10.23 (10.23.7).
Bounds used below: Meng, "Explicit error bound of the fast multipole method
for scattering problems in 2-D", https://arxiv.org/abs/1806.08512, equations
(18), (19), (24) and Lemmas 1 and 3. These require real positive arguments.
Their H^(1) magnitude bounds also apply to its real-axis conjugate H^(2).
For q >= x, C_n(x) <= C_{q+1}(x) for n >= q+1, so

  2 sum[n>q] |H_n(x) J_n(y)|
    <= 2 sqrt(2) C_{q+1}(x) / pi * r**(q+1) / ((q+1)*(1-r)),

where r=y/x<1. The omitted orders p+1,...,q are bounded individually.
Consequently retained order p need not exceed k times the target distance.
No sampled residual is used to select or accept an expansion.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.special import gammaln, hankel2, jv, logsumexp, yv


class AnalyticFarRejected(ValueError):
    """The restricted expansion cannot safely represent this request."""


def _real_positive(value, name):
    value = complex(value)
    if not np.isfinite(value) or value.imag or value.real <= 0:
        raise AnalyticFarRejected(f'{name} must be finite, positive, and real.')
    return value.real


def graf_tail_bound(k, source_radius, target_radius, order, *, max_tail_terms=4096):
    """Uniform absolute H0^(2) truncation bound for radii <=a and >=R.

    This is an analytic truncation bound evaluated in floating point, not an
    interval-arithmetic enclosure of special-function implementation errors.
    Unsupported, nonfinite, or excessive intermediate work rejects explicitly.
    """
    k = _real_positive(k, 'Wavenumber')
    a, radius = float(source_radius), float(target_radius)
    if (not math.isfinite(a) or not math.isfinite(radius) or a < 0 or radius <= a
            or type(order) is not int or order < 0
            or type(max_tail_terms) is not int or max_tail_terms < 1):
        raise AnalyticFarRejected('Invalid order, radii, or separation.')
    if a == 0:
        return 0.
    x, y = k*radius, k*a
    if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0:
        raise AnalyticFarRejected('Scaled radii are outside floating-point range.')
    q = max(order, math.ceil(x))
    if q > max_tail_terms:
        raise AnalyticFarRejected('Analytic tail evaluation exceeds its work cap.')
    n = q+1
    # C_n is positive in the theorem's n >= x+1 domain. Nonfinite Y_n
    # rejects, even when an unqualified rescaling might recover its product.
    yn = float(yv(n, x))
    if not math.isfinite(yn) or yn >= 0:
        raise AnalyticFarRejected('Unstable Hankel tail bound.')
    log_c = max(0., math.log(-yn)+n*math.log(x/2)+math.log(math.pi)-float(gammaln(n)))
    ratio = a/radius
    log_tail = (math.log(2*math.sqrt(2)/math.pi)+log_c
                +n*math.log(ratio)-math.log(n)-math.log1p(-ratio))
    logs = [log_tail]
    if order < q:
        orders = np.arange(order+1, q+1)
        h = abs(hankel2(orders, x))
        if not np.all(np.isfinite(h)) or np.any(h <= 0):
            raise AnalyticFarRejected('Unstable finite Hankel tail sum.')
        # Both J bounds hold for every real source radius <= a. In
        # particular J_n(ka) itself is NOT a bound when it crosses a zero.
        log_j = np.minimum(-.5*math.log(2), orders*math.log(y/2)-gammaln(orders+1))
        logs.append(float(logsumexp(math.log(2)+np.log(h)+log_j)))
    log_bound = float(logsumexp(logs))
    if not math.isfinite(log_bound) or log_bound > math.log(np.finfo(float).max):
        raise AnalyticFarRejected('Nonfinite analytic truncation bound.')
    # Modest numerical inflation, explicitly not a claim of interval rigor.
    bound = math.exp(max(log_bound, math.log(np.finfo(float).tiny)))
    return float(np.nextafter(bound*(1+128*np.finfo(float).eps*(q+2)), np.inf))


def _points(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise AnalyticFarRejected(f'{name} must contain real coordinates.')
    points = np.asarray(raw, float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.all(np.isfinite(points)):
        raise AnalyticFarRejected(f'{name} must be a nonempty finite n-by-2 array.')
    return points


def _weights(value, points, test):
    if value is None:
        return sparse.eye(points, format='csr'), np.ones(points)
    value = sparse.csr_matrix(value, dtype=complex) if sparse.issparse(value) else np.asarray(value, complex)
    if (value.ndim != 2 or value.shape[1 if test else 0] != points
            or not min(value.shape) or not np.all(np.isfinite(value.data if sparse.issparse(value) else value))):
        raise AnalyticFarRejected('Quadrature/basis weights have invalid shape or values.')
    mass = np.asarray(abs(value).sum(axis=1 if test else 0)).ravel()
    return value, mass


@dataclass
class AnalyticFarBlock:
    left: np.ndarray
    right: np.ndarray
    entry_tail_left: np.ndarray
    entry_tail_right: np.ndarray
    evidence: dict

    @property
    def row_tail_bound(self):
        return self.entry_tail_left*float(np.sum(self.entry_tail_right))

    @property
    def column_tail_bound(self):
        return float(np.sum(self.entry_tail_left))*self.entry_tail_right

    def matmul(self, rhs, trans=0):
        if trans == 0:
            return self.left @ (self.right @ rhs)
        if trans == 1:
            return self.right.T @ (self.left.T @ rhs)
        if trans == 2:
            return self.right.conj().T @ (self.left.conj().T @ rhs)
        raise ValueError('Invalid transpose mode.')

    def dense(self):
        """Materialize represented coefficients for independent prototype tests."""
        return self.left @ self.right


def single_layer_block(targets, sources, k, *, center=None, test_weights=None,
                       source_weights=None, absolute_tolerance=1e-10, max_order=128,
                       max_tail_terms=4096, kernel='single_layer', medium='real_homogeneous'):
    """Factor Wt @ (i/4 H0^(2)(k|x-y|)) @ Ws for separated real-k nodes.

    Wt has shape (test DOFs, targets); Ws has shape (sources, source DOFs).
    Their entries include quadrature and basis weights, including signs.
    The selected bound is per weighted coefficient, not relative or sampled.
    Derivative kernels, lossy media, and nonseparated groups reject explicitly.
    Returned error arrays bound analytic truncation only, not total roundoff.
    """
    if kernel != 'single_layer' or medium != 'real_homogeneous':
        raise AnalyticFarRejected('Only the homogeneous real-k scalar single-layer kernel is supported.')
    k = _real_positive(k, 'Wavenumber')
    tolerance = _real_positive(absolute_tolerance, 'Absolute tolerance')
    if type(max_order) is not int or max_order < 0:
        raise AnalyticFarRejected('Expansion order cap must be a nonnegative integer.')
    targets, sources = _points(targets, 'Targets'), _points(sources, 'Sources')
    if center is None:
        center = np.mean(sources, axis=0)
    else:
        center = np.asarray(center)
        if center.shape != (2,):
            raise AnalyticFarRejected('Center must contain two real coordinates.')
        center = _points(center[None, :], 'Center')[0]
    source_delta, target_delta = sources-center, targets-center
    source_radii, target_radii = np.linalg.norm(source_delta, axis=1), np.linalg.norm(target_delta, axis=1)
    a, radius = float(np.max(source_radii)), float(np.min(target_radii))
    if not math.isfinite(radius) or radius <= a:
        raise AnalyticFarRejected('Source circle must exclude every target node.')
    wt, row_mass = _weights(test_weights, len(targets), True)
    ws, column_mass = _weights(source_weights, len(sources), False)
    weight_bound = float(np.max(row_mass))*float(np.max(column_mass))/4
    for order in range(max_order+1):
        tail = graf_tail_bound(k, a, radius, order, max_tail_terms=max_tail_terms)
        if tail*weight_bound <= tolerance:
            break
    else:
        raise AnalyticFarRejected('Expansion cannot meet the requested absolute truncation budget.')
    orders = np.arange(-order, order+1)
    target_angles = np.arctan2(target_delta[:, 1], target_delta[:, 0])
    source_angles = np.arctan2(source_delta[:, 1], source_delta[:, 0])
    # Match twod.assembly.kernels.values and the regional Galerkin convention.
    left = .25j*hankel2(orders[None, :], k*target_radii[:, None])*np.exp(1j*target_angles[:, None]*orders)
    right = jv(orders[:, None], k*source_radii[None, :])*np.exp(-1j*orders[:, None]*source_angles)
    left, right = np.asarray(wt @ left), np.asarray((ws.T @ right.T).T)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise AnalyticFarRejected('Nonfinite analytic expansion factors.')
    return AnalyticFarBlock(left, right, row_mass*(tail/4), column_mass,
        dict(construction='experimental_real_helmholtz_graf', order=order, rank=2*order+1,
             coefficient_pair_evaluations=0, source_nodes=len(sources), target_nodes=len(targets),
             analytic_kernel_tail_bound=tail/4, max_coefficient_tail_bound=tail*weight_bound,
             source_radius=a, nearest_target_radius=radius, production_certified=False,
             error_scope='analytic truncation only; floating-point evaluation and production integration unqualified'))
