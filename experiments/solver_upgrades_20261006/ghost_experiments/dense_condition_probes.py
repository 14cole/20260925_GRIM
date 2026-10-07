"""Dense condition estimate: SciPy's block 1-norm estimator with four probe columns instead of two.

Each estimator iteration applies the inverse (two LU solves) to ``t`` columns; a LAPACK solve of
2 or 4 columns streams the whole factor once, so wider probes cost the same pass and may converge
in fewer iterations.  The estimate is a diagnostic for the 1e6 gate; fields are unchanged.
"""
import numpy as np
from ghost_backend.twod import solver as S

DESCRIPTION = __doc__.strip().splitlines()[0]
CASES = ['air_s3', 'air_c3', 'air_c10']
PROBES = 4


def _deterministic_onenormest(operator):
    with S._CONDITION_ESTIMATE_LOCK:
        saved = np.random.get_state()
        try:
            np.random.seed(S._CONDITION_ESTIMATE_SEED)
            return float(S._SCIPY_SPARSE_LINALG.onenormest(operator, t=PROBES))
        finally:
            np.random.set_state(saved)


def apply():
    S._deterministic_onenormest = _deterministic_onenormest
