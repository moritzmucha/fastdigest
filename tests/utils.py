import sys
import math
from fastdigest import TDigest


RTOL = 0.0
ATOL = 1e-12
DEFAULT_MAX_CENTROIDS = 1000


def check_median(digest: TDigest, expected: float) -> None:
    quantile_est = digest.quantile(0.5)
    assert math.isclose(quantile_est, expected, rel_tol=1e-3), (
        f"Expected median ~{expected}, got {quantile_est}"
    )

def check_tdigest_equality(orig: TDigest, new: TDigest) -> None:
    assert isinstance(new, TDigest), (
        f"Expected TDigest, got {type(new).__name__}"
    )
    assert new.max_centroids == orig.max_centroids, (
        f"Expected max_centroids={orig.max_centroids}, "
        f"got {new.max_centroids}"
    )
    assert new.n_values == orig.n_values, (
        f"Expected {orig.n_values} values, got {new.n_values}"
    )
    assert new.n_centroids == orig.n_centroids, (
        f"Expected {orig.n_centroids} centroids, got {new.n_centroids}"
    )
    for q in [0.25, 0.5, 0.75]:
        orig_val = orig.quantile(q)
        new_val = new.quantile(q)
        assert math.isclose(
            orig_val, new_val, rel_tol=RTOL, abs_tol=ATOL
        ), f"Quantile {q} mismatch: orig {orig_val} vs new {new_val}"
