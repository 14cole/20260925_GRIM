"""Shared CPU backend relative runtime prior.

The prior ranks work; it is not a wall-time or optimality guarantee. It retains
the measured dense preference on small/medium systems. Both desktop and
execution-node scheduling use it.
"""
import math

MODEL = 'geometry_work_v4'
BACKENDS = ('dense', 'compressed')


def relative_cost(resources, n_angles, mode):
    n=max(1,int(resources['nodes'])); d=max(1,int(resources['system_dofs']))
    angles=max(1,int(n_angles)); kernels=max(1.,resources.get('operator_matrices',3)/3.)
    if resources.get('analytic_zero'):
        return .001
    assembly=7e-7*n*n*kernels
    dense=.025 + assembly + 4e-12*d**3 + 2e-10*d*d*angles
    if mode == 'dense':
        return dense
    if mode == 'compressed':
        return 1.4*dense
    raise ValueError('Unknown automatic backend.')


def available_candidates(candidates):
    """Copy candidates for ranking; every backend runs on any CPU host."""
    return {mode:dict(c) for mode,c in candidates.items()}


def rank_candidates(candidates, budget_gib, margin=.2):
    available=available_candidates(candidates)
    if not available:
        raise RuntimeError('No compatible solver backend is available.')
    for mode,c in available.items():
        if mode not in BACKENDS or any(not math.isfinite(float(c[k])) or c[k] <= 0 for k in ('cost','peak_gb')):
            raise ValueError('Invalid automatic backend forecast.')
    admitted=[m for m,c in available.items() if c['peak_gb'] <= budget_gib]
    if not admitted:
        description=', '.join('{} {:.2f} GiB'.format(m,c['peak_gb']) for m,c in available.items())
        raise MemoryError('No compatible backend fits the {:.2f} GiB solve budget ({}).'.format(budget_gib,description))
    # Backends inside the margin lead. Every other backend that fits the budget
    # stays in the order as a retry: dropping it let a compressed build that
    # failed late end the run although dense (5.7 GiB measured against its
    # 7.9 GiB forecast) fit the 9 GiB budget.
    key=lambda m:(available[m]['cost'],BACKENDS.index(m))
    margin_fitting=[m for m in admitted if available[m]['peak_gb'] <= (1-margin)*budget_gib]
    return sorted(margin_fitting,key=key)+sorted((m for m in admitted if m not in margin_fitting),key=key)
