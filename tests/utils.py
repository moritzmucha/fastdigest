import sys
import math
from typing import Iterable, List, Sequence
from fastdigest import TDigest


EPS = sys.float_info.epsilon
RTOL = 0.000
ATOL = 1e-12
DEFAULT_MAX_CENTROIDS = 1000
SAMPLE_QUANTILES = [0.01, 0.25, 0.5, 0.75, 0.99]


def quantile(seq: Sequence[float], q: float):
    """
    Calculate the q-th quantile of a sequence of floats using linear interpolation.
    
    Parameters:
        seq (Sequence[float]): A sequence of floats.
        q (float): A number between 0 and 1 indicating the desired quantile.
        
    Returns:
        float: The q-th quantile of the sequence.
    
    Raises:
        ValueError: If the sequence is empty or q is not between 0 and 1.
    """
    if not seq:
        raise ValueError("Sequence must not be empty")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    
    s = sorted(seq)
    n = len(s)
    # Position using linear interpolation: p = (n-1) * q
    pos = (n - 1) * q
    lower = int(pos)
    upper = lower + 1
    if upper >= n:
        return s[lower]
    weight = pos - lower
    return s[lower] * (1 - weight) + s[upper] * weight

def calculate_sample_quantiles(
        data: Iterable[float],
        quantiles: Iterable[float] = SAMPLE_QUANTILES
    ) -> List[float]:
    return [quantile(data, q) for q in quantiles]

def check_sample_quantiles(
        digest: TDigest,
        expected: Iterable[float],
        quantiles: Iterable[float] = SAMPLE_QUANTILES,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> None:
    estimated = [digest.quantile(q) for q in quantiles]
    for q, exp, est in zip(quantiles, expected, estimated):
        assert math.isclose(est, exp, rel_tol=rtol, abs_tol=atol), (
            f"Expected p{int(100 * q):02} ~{exp}, got {est}"
        )

def check_tdigest_equality(orig: TDigest, new: TDigest) -> None:
    assert isinstance(new, TDigest), (
        f"Expected TDigest, got {type(new).__name__}"
    )
    assert orig == new, "Equality check failed"
    expected = [orig.quantile(q) for q in SAMPLE_QUANTILES]
    check_sample_quantiles(new, expected)
