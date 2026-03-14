import sys
import math
from bisect import bisect_right
from typing import Iterable, List, Sequence
from fastdigest import TDigest


EPS = sys.float_info.epsilon
RTOL = 0.000
ATOL = 1e-12
DEFAULT_MAX_CENTROIDS = 1000
SAMPLE_QUANTILES = [0.01, 0.25, 0.5, 0.75, 0.99]
SAMPLE_RANKS = [-float("inf"), 25.0, 50.0, 75.0, 100.0, float("inf")]


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


def rank(seq: Sequence[float], x: float):
    """
    Calculate the normalized rank (CDF position) of value x in a sequence of floats
    using linear interpolation.

    Parameters:
        seq (Sequence[float]): A sequence of floats.
        x (float): Value whose rank is requested.

    Returns:
        float: A number between 0 and 1 representing the relative rank.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not seq:
        raise ValueError("Sequence must not be empty")

    s = sorted(seq)
    n = len(s)

    if x <= s[0]:
        return 0.0
    if x >= s[-1]:
        return 1.0

    i = bisect_right(s, x) - 1
    lo, hi = s[i], s[i + 1]

    if hi == lo:
        pos = i
    else:
        pos = i + (x - lo) / (hi - lo)

    return pos / (n - 1)


def calculate_sample_quantiles(
    data: Iterable[float], quantiles: Iterable[float] = SAMPLE_QUANTILES
) -> List[float]:
    return [quantile(data, q) for q in quantiles]


def calculate_sample_ranks(
    data: Iterable[float], ranks: Iterable[float] = SAMPLE_RANKS
) -> List[float]:
    return [rank(data, x) for x in ranks]


def compare_values(
    func_name: str,
    parameters: Iterable[float],
    expected: Iterable[float],
    results: Iterable[float],
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    for p, exp, res in zip(parameters, expected, results):
        assert math.isclose(res, exp, rel_tol=rtol, abs_tol=atol), (
            f"{func_name}({p}): expected ~{exp}, got {res}"
        )


def check_sample_quantiles(
    digest: TDigest,
    expected: Iterable[float],
    quantiles: Iterable[float] = SAMPLE_QUANTILES,
) -> None:
    estimated = [digest.quantile(q) for q in quantiles]
    compare_values("quantile", quantiles, expected, estimated)


def check_sample_ranks(
    digest: TDigest,
    expected: Iterable[float],
    ranks: Iterable[float] = SAMPLE_RANKS,
) -> None:
    estimated = [digest.cdf(x) for x in ranks]
    compare_values("cdf", ranks, expected, estimated)


def check_tdigest_equality(orig: TDigest, new: TDigest) -> None:
    assert isinstance(new, TDigest), (
        f"Expected TDigest, got {type(new).__name__}"
    )
    assert orig == new, "Equality check failed"
    expected = [orig.quantile(q) for q in SAMPLE_QUANTILES]
    check_sample_quantiles(new, expected)
