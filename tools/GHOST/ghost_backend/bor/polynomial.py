"""Restricted polynomial-current Galerkin prototype for closed PEC meridians.

The geometry remains the supplied piecewise-linear meridian. Degrees two and
three add true continuous polynomial current unknowns within each element;
they do not reinterpret a larger Gauss rule as a higher-order approximation.
This research API is deliberately separate from production dispatch until
curved geometry, material interfaces and large-case memory layouts are qualified.
"""
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.sparse import csr_matrix

from ghost_backend.bor import solver as bor
from ghost_backend.bor.kernels import kernels_for_mode
from ghost_backend.bor.near_storage import mode_sign
from ghost_backend.bor.options import configured, current_options


def lagrange_basis(degree, coordinates):
    """Lobatto interpolation values and derivatives on the unit panel."""
    degree = int(degree)
    if degree not in (1, 2, 3):
        raise ValueError('Polynomial BoR basis degree must be one, two, or three.')
    nodes = (np.array([0., 1.]) if degree == 1 else np.array([0., .5, 1.])
             if degree == 2 else np.array([0., (1-1/math.sqrt(5))/2,
                                           (1+1/math.sqrt(5))/2, 1.]))
    values, derivatives = [], []
    for j, node in enumerate(nodes):
        other = np.delete(nodes, j)
        basis = Polynomial.fromroots(other) / np.prod(node-other)
        values.append(basis(coordinates))
        derivatives.append(basis.deriv()(coordinates))
    return np.asarray(values), np.asarray(derivatives), nodes


def touching_points(e, f, order):
    """Duffy triangles with polynomial grading of the logarithmic diagonal."""
    x, wx = bor.cached_leggauss(order)
    y, wy = bor.cached_leggauss(order+1)
    x, y, wx, wy = (x+1)/2, (y+1)/2, wx/2, wy/2
    if e == f:
        # Double grading in both coordinates eventually rounds distinct
        # physical points to the same ring. Grade only the diagonal here.
        u = x[:,None]
        a, b = np.broadcast_arrays(u, u*(1-y[None,:]**2))
        jacobian = u*2*y[None,:]
    else:
        u = x[:,None]**2
        a, b = np.broadcast_arrays(u, u*y[None,:])
        jacobian = 2*x[:,None]*u*np.ones((1,len(y)))
    weights = (wx[:,None]*wy[None,:]*jacobian).ravel()
    a, b = np.r_[a.ravel(),b.ravel()], np.r_[b.ravel(),a.ravel()]
    if f == e+1:
        a = 1-a
    elif e == f+1:
        b = 1-b
    return a,b,np.r_[weights,weights]


class PolynomialMeridian(bor.BorPecSolver):
    """Polynomial currents, exact divergence derivatives and existing kernels."""
    def __init__(self, points, frequency, degree, near_depth=4):
        if int(degree) != degree or degree not in (1, 2, 3):
            raise ValueError('Polynomial BoR basis degree must be one, two, or three.')
        self.degree = int(degree)
        refinement = current_options()['near_refinement']
        super().__init__(points, frequency, gauss_order=max(4, self.degree+3)+2*refinement,
                         near_depth=near_depth)
        if not self.gen.node_on_axis(0) or not self.gen.node_on_axis(self.gen.n_elems):
            raise ValueError('Polynomial BoR currently requires a closed axis-to-axis PEC surface.')
        if self._compressed:
            raise ValueError('Polynomial BoR currently supports dense factorization only.')
        self.Nn = self.degree*self.gen.n_elems+1
        self.connectivity = np.arange(self.gen.n_elems)[:, None]*self.degree+np.arange(self.degree+1)
        shape, deriv, nodes = lagrange_basis(self.degree, self.g.T1)
        self.current_nodes = np.vstack((
            (self.gen.nodes[:-1, None, :] + nodes[None, :-1, None]*
             np.diff(self.gen.nodes, axis=0)[:, None, :]).reshape(-1, 2), self.gen.nodes[-1]))
        rows = self.g.elem[None, :]*self.degree+np.arange(self.degree+1)[:, None]
        cols = np.broadcast_to(np.arange(self.P), rows.shape)
        self.test_basis = csr_matrix((shape.ravel(), (rows.ravel(), cols.ravel())), shape=(self.Nn, self.P))
        derivative = self.g.trho[None, :]*shape + self.g.rho[None, :]*deriv/self.gen.lengths[self.g.elem][None, :]
        self.divergence_basis = csr_matrix((derivative.ravel(), (rows.ravel(), cols.ravel())),
                                          shape=(self.Nn, self.P))
        self.polynomial_near = {}
        self.near_change = 0.
        self.near_refinement_max = 0

    def _test_accumulate(self, values):
        return self.test_basis @ values

    def _basis_evaluate(self, values):
        return self.test_basis.T @ values

    def basis_mask(self, mode):
        active = np.ones(2*self.Nn, dtype=bool)
        active[[0, self.Nn-1]] = abs(mode) == 1
        active[[self.Nn, 2*self.Nn-1]] = False
        return active

    def basis_transform(self, mode):
        active = np.flatnonzero(self.basis_mask(mode))
        rows, cols = active.tolist(), np.arange(len(active)).tolist()
        data = [1.+0j]*len(active)
        if abs(mode) == 1:
            for node, element in ((0, 0), (self.Nn-1, self.gen.n_elems-1)):
                rows.append(self.Nn+node)
                cols.append(int(np.searchsorted(active, node)))
                data.append(1j*mode*np.sign(self.gen.trho[element]))
        return csr_matrix((data, (rows, cols)), shape=(2*self.Nn, len(active)))

    def _near_blocks(self, e, f, cap, points):
        s, t, weights = points
        count, width = self.degree+1, cap+1
        result = {kind: np.zeros((4, width, count, count), complex) for kind in ('efie', 'mfie')}
        modes = np.arange(width)
        for start in range(0, len(s), 256):
            if self._checkpoint is not None:
                self._checkpoint()
            sl = slice(start, start+256)
            rp, zp, trp, tzp, *_, lp = bor._points_on_element(self.gen, e, s[sl])
            rq, zq, trq, tzq, *_, lq = bor._points_on_element(self.gen, f, t[sl])
            tp, dtp, _ = lagrange_basis(self.degree, s[sl])
            tq, dtq, _ = lagrange_basis(self.degree, t[sl])
            dp, dq = trp*tp+rp*dtp/lp, trq*tq+rq*dtq/lq
            w = weights[sl]*lp*lq
            chunk = (rp, zp, trp, tzp, rq, zq, trq, tzq, tp, tq, dp, dq, w)
            kernels = bor._near_chunk_kernels([chunk], self.k, cap, ('efie', 'mfie'), signed=False)[0]

            def rows(a, b):
                return (a[:, None]*b[None, :]).reshape(count*count, -1)

            def contract(a, kernel):
                return (a @ kernel).reshape(count, count, width).transpose(2, 0, 1)

            tt = rows(tp, tq)
            g = kernels['efie']
            gn = g[:, modes]
            gc = (g[:, abs(modes-1)] + g[:, modes+1])*.5
            gs = (g[:, abs(modes-1)] - g[:, modes+1])/(2j)
            rr = (rp*rq*w)[:, None]
            scalar = w[:, None]*gn/self.k**2
            result['efie'][0] += contract(tt, rr*(trp*trq*gc+tzp*tzq*gn))-contract(rows(dp,dq), scalar)
            result['efie'][1] += contract(tt, rr*trp*gs)-1j*modes[:,None,None]*contract(rows(dp,tq), scalar)
            result['efie'][2] += contract(tt, -rr*trq*gs)+1j*modes[:,None,None]*contract(rows(tp,dq), scalar)
            result['efie'][3] += contract(tt, rr*gc)-modes[:,None,None]**2*contract(tt, scalar)
            for uv, value in enumerate(kernels['mfie']):
                result['mfie'][uv] += contract(tt, 2*np.pi*rr*value)
        return result

    def prepare_polynomial(self, cap):
        """Check actual polynomial near blocks, including self/shared panels."""
        self._angular_top = int(cap)+1
        self._kernel_tables(cap)
        self._mfie_tables(cap)
        for e, sources in enumerate(self._near_sources_by_element):
            for f in sources:
                adjacent = abs(e-f) <= 1

                def integrate(level):
                    order = 12*2**(level+current_options()['near_refinement'])
                    points = (touching_points(e, f, order)
                              if adjacent else bor._gap_graded_points(self.gen, e, self.gen, f, order))
                    if e == f and self.gen.node_on_axis(e+1):
                        # Apply the same pole grading at the south endpoint.
                        points = (1-points[0],1-points[1],points[2])
                    return self._near_blocks(e, f, cap, points)

                coarse = integrate(0)
                for level in range(1, 4):
                    fine = integrate(level)
                    errors = []
                    for kind in fine:
                        scale = np.max(abs(fine[kind]), axis=(1,2,3))
                        floor = max(float(np.max(scale))*1e-8, 1e-280)
                        delta = np.max(abs(fine[kind]-coarse[kind]), axis=(1,2,3))
                        errors.append(float(np.max(delta/np.maximum(scale, floor))))
                    change = max(errors)
                    if math.isfinite(change) and change <= 2e-5:
                        self.near_change = max(self.near_change, change)
                        self.near_refinement_max = max(self.near_refinement_max, level)
                        self.polynomial_near[(e,f)] = fine
                        break
                    coarse = fine
                else:
                    raise ValueError(f'Polynomial BoR near block ({e},{f}) did not converge: {change:.3g}.')

    def assemble_polynomial(self, mode, cap, alpha):
        g, basis = self.g, self.test_basis
        efie = list(bor._pair_blocks(mode, self.k, g.rho,g.trho,g.tz,basis,self.divergence_basis,g.w,
                                    g.rho,g.trho,g.tz,basis,self.divergence_basis,g.w,
                                    *kernels_for_mode(self._G_table, mode)))
        weighted = basis.multiply(g.w*g.rho)
        mfie = [2*np.pi*mode_sign(uv,mode)*(weighted @ table[:,:,abs(mode)] @ weighted.T)
                for uv,table in enumerate(self._K_tables)]
        for (e,f), data in self.polynomial_near.items():
            index = np.ix_(self.connectivity[e], self.connectivity[f])
            for uv in range(4):
                efie[uv][index] += data['efie'][uv,abs(mode)]*mode_sign(uv,mode)
                mfie[uv][index] += data['mfie'][uv,abs(mode)]*mode_sign(uv,mode)
        mass = (weighted @ basis.T).toarray()*2*np.pi
        mfie = [.5*mass-mfie[0], -mfie[1], -mfie[2], .5*mass-mfie[3]]
        blocks = [alpha*1j*self.k*self.eta*2*np.pi*t + (1-alpha)*bor.ETA0*k for t,k in zip(efie,mfie)]
        full = np.block([[blocks[0],blocks[1]], [blocks[2],blocks[3]]])
        q = self.basis_transform(mode)
        return q.conj().T @ full @ q, q


@configured
def solve_bor_polynomial(points, freq_hz, thetas_deg, basis_degree=3, n_modes=None,
                         mode_tol=1e-6, cfie_alpha=.5, workers=1, check_abort=None):
    """Experimental dense CFIE on one closed PEC surface; no material shortcuts.

    Caller-supplied points define the geometry at every degree. Consequently
    p-convergence certifies the current approximation on that geometry, and
    never substitutes for a separate geometric mesh-convergence check.
    """
    if current_options()['factorization'] != 'dense':
        raise ValueError('Polynomial BoR requires bor_options={"factorization": "dense"}.')
    alpha = float(cfie_alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('Polynomial BoR CFIE alpha must lie strictly between zero and one.')
    surface = PolynomialMeridian(bor._validate_solve_bor_generatrix(points, 'cfie'), freq_hz, basis_degree)
    surface._checkpoint = check_abort
    thetas = bor._validated_bor_aspects(thetas_deg)
    if thetas.ndim != 1 or not len(thetas) or not np.all(np.isfinite(thetas)) or np.any((thetas<0)|(thetas>180)):
        raise ValueError('Polynomial BoR aspects must be a nonempty finite grid in [0,180].')
    cap, tail = bor._bor_mode_limits(surface.k, np.max(surface.gen.nodes[:,0]), thetas, n_modes)
    pairs = sum(map(len, surface._near_sources_by_element))
    # Explicitly price the prototype's full point tables, polynomial near
    # blocks and dense assembly temporaries before any operator preparation.
    retained = 16*(surface.P**2*(5*cap+6)+pairs*8*(cap+1)*(basis_degree+1)**2)
    workspace = (max(bor.FFT_BUILD_BUDGET,16*12*surface.P**2)+16*24*surface.Nn**2
                 +bor.NEAR_BATCH_KERNEL_BYTES+bor.ANGULAR_CACHE_BUDGET_BYTES)
    assembly_gb = (retained+workspace)/1e9

    def rhs(mode, theta, pol):
        return alpha*surface.rhs_mode(mode,theta,pol)+(1-alpha)*bor.ETA0*surface.rhs_mfie_mode(mode,theta,pol)

    def farfield(mode, values, theta, pol):
        return surface.farfield_mode(mode,values,theta)[0 if pol == 'VV' else 1]

    fields, used, stats = bor._mode_sweep(2*surface.Nn,thetas,('VV','HH'),cap,mode_tol,
        lambda mode: surface.assemble_polynomial(mode,cap,alpha), rhs, farfield,
        prepare=surface.prepare_polynomial, workers=workers, check_abort=check_abort,
        monitor_cond=True, min_mode_before_tail=tail, assembly_peak_gb=assembly_gb,
        signed_mode_symmetry=True, axial_mode_only=bor._all_axial_aspects(thetas),
        rhs_batch=lambda mode,angles,pols: surface.rhs_vv_hh_batch(mode,angles,
            efie_scale=alpha,mfie_scale=(1-alpha)*bor.ETA0),
        farfield_batch=lambda mode,values,angles,pols: surface.farfield_vv_hh_batch(mode,values,angles))
    bor._require_mode_convergence(stats,mode_tol)
    return dict(theta_deg=thetas.tolist(), amp_vv=fields[0].tolist(), amp_hh=fields[1].tolist(),
        sigma_vv=(4*np.pi*abs(fields[0])**2).tolist(), sigma_hh=(4*np.pi*abs(fields[1])**2).tolist(),
        modes_used=used,n_unknowns=2*surface.Nn,formulation='cfie',boundary_model='pec',
        assembly='polynomial_dense_prototype',basis_degree=int(basis_degree),
        geometry_approximation='supplied_piecewise_linear_meridian',
        polynomial_near_relative_change=surface.near_change,
        polynomial_near_refinement_max=surface.near_refinement_max,**stats)
