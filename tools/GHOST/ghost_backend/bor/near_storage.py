"""Store nonnegative near-field modes using exact tangential-block parity.

For EFIE, MFIE and unrotated IBC brackets, tt/ff are even in m and tf/ft
are odd. This holds for complex media as well: no complex conjugation is
involved. Apply the parity before the PMCHWT row rotation.
"""
import numpy as np


def mode_sign(component, mode):
    return -1 if mode < 0 and component in (1, 2) else 1


def mode_blocks(values, mode):
    """Read one signed mode from [4, m_max+1, ...] retained coefficients."""
    result = values[:, abs(mode)]
    if mode < 0:
        signs = np.array([1, -1, -1, 1]).reshape((4,) + (1,) * (result.ndim - 1))
        result = result * signs
    return result
