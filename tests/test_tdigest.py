import pytest
import math
import random
import pickle
import collections.abc
from copy import copy, deepcopy
from typing import Callable, Sequence, List
from fastdigest import TDigest
from utils import (
    EPS,
    RTOL,
    ATOL,
    DEFAULT_MAX_CENTROIDS,
    calculate_sample_quantiles,
    check_sample_quantiles,
    check_tdigest_equality,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def empty_digest() -> TDigest:
    return TDigest()

@pytest.fixture
def sample_values() -> List[int]:
    return list(range(1, 101))

# -------------------------------------------------------------------
# Initialization and property tests
# -------------------------------------------------------------------
def test_init() -> None:
    d = TDigest()
    assert d.max_centroids == DEFAULT_MAX_CENTROIDS
    assert d.n_values == 0
    assert d.n_centroids == 0
    d = TDigest(max_centroids=3)
    assert d.max_centroids == 3
    with pytest.raises(TypeError):
        TDigest([1, 2, 3])

@pytest.mark.parametrize("values", [
    [1, 2, 3, 4, 5],
    range(1, 6),
    (5, 3, 4, 2, 1),
])
def test_from_values(values: Sequence[int]) -> None:
    d = TDigest.from_values(values)
    assert d.max_centroids == DEFAULT_MAX_CENTROIDS
    assert d.n_values == len(values)
    assert d.n_centroids == len(values)

    d = TDigest.from_values(values, max_centroids=3)
    assert d.max_centroids == 3
    assert d.n_values == len(values)
    assert d.n_centroids == 3

    d = TDigest.from_values([])
    assert d == TDigest()

def test_max_centroids(
        sample_values: Sequence[int], empty_digest: TDigest
    ) -> None:
    d = TDigest.from_values(sample_values)
    assert d.max_centroids == DEFAULT_MAX_CENTROIDS
    d = TDigest.from_values(sample_values, max_centroids=3)
    assert isinstance(d.max_centroids, int) and d.max_centroids == 3
    assert empty_digest.max_centroids == DEFAULT_MAX_CENTROIDS
    d = TDigest(3)
    assert d.max_centroids == 3
    d.max_centroids = 10
    assert d.max_centroids == 10

def test_n_values_and_n_centroids(empty_digest: TDigest) -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    assert isinstance(d.n_values, int) and d.n_values == 3
    assert isinstance(d.n_centroids, int) and d.n_centroids == 3
    d = empty_digest
    assert d.n_values == 0
    assert d.n_centroids == 0
    d.update(1)
    assert d.n_values == 1
    assert d.n_centroids == 1

def test_is_empty(empty_digest: TDigest) -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    assert not d.is_empty
    d = empty_digest
    assert d.is_empty
    d.update(1)
    assert not d.is_empty

def test_centroids(empty_digest: TDigest) -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    centroids = d.centroids
    assert isinstance(centroids, list) and len(centroids) == 3
    assert all(isinstance(t, tuple) for t in centroids)
    assert all(isinstance(v, float) for v in centroids[0])
    assert isinstance(empty_digest.centroids, list)

# -------------------------------------------------------------------
# Merge tests (merge, merge_inplace, __add__, __iadd__)
# -------------------------------------------------------------------
@pytest.mark.parametrize("merge_func", [
    lambda d1, d2: d1.merge(d2),
    lambda d1, d2: d1 + d2,
])
def test_merge_operations(
        merge_func: Callable[[TDigest, TDigest], TDigest]
    ) -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    expected = calculate_sample_quantiles(range(1, 101))
    merged = merge_func(d1, d2)
    check_sample_quantiles(merged, expected)
    assert merged.n_values == 100

def test_merge_with_max_centroids() -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    d1.max_centroids = 3
    merged = d1.merge(d2)
    assert merged.n_values == 100
    d2.max_centroids = 50
    merged = d1.merge(d2)
    assert 3 < merged.n_centroids <= 50 + 1, (
        f"Expected between 4 and 51 centroids, got {merged.n_centroids}"
    )
    d2.max_centroids = 3
    merged = d1.merge(d2)
    assert merged.n_centroids in (3, 4), (
        f"Expected 3-4 centroids, got {merged.n_centroids}"
    )

def test_merge_inplace() -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    expected = calculate_sample_quantiles(range(1, 101))
    d1.merge_inplace(d2)
    check_sample_quantiles(d1, expected)
    assert d1.n_values == 100
    d1.max_centroids = 3
    d1.merge_inplace(d2)
    assert d1.n_centroids <= 3 + 1
    d2.max_centroids = 50
    d1.merge_inplace(d2)
    assert d1.n_centroids <= 3 + 1
    empty = TDigest()
    d = TDigest.from_values(range(1, 51))
    d.merge_inplace(empty)
    expected = calculate_sample_quantiles(range(1, 51))
    check_sample_quantiles(d, expected)
    empty.merge_inplace(d)
    check_sample_quantiles(empty, expected)

@pytest.mark.parametrize("iadd_op", [
    lambda d1, d2: d1 + d2,
    lambda d1, d2: d1.__iadd__(d2) or d1,
])
def test_add_iadd(iadd_op: Callable[[TDigest, TDigest], TDigest]) -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    expected = calculate_sample_quantiles(range(1, 101))
    result = iadd_op(d1, d2)
    check_sample_quantiles(result, expected)

def test_add_with_empty_max_centroids(empty_digest: TDigest) -> None:
    digest = TDigest.from_values(range(101))
    digest.max_centroids = 3
    empty_digest.max_centroids = 3
    merged = digest + empty_digest
    assert len(merged) <= 3 + 1

# -------------------------------------------------------------------
# Update tests (batch_update and update)
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    "update_method, start_range, update_input, max_centroids", [
        ("batch_update", range(1, 51), range(51, 101), 1000),
        ("batch_update", range(51, 101), range(1, 51), 20),
        ("batch_update", range(1, 101), [], 1000),
        ("update", range(1, 100), [100], 20),
        ("update", range(100, 2, -1), [1, 2], 1000)
    ]
)
def test_updates(
        update_method: str,
        start_range: Sequence[int],
        update_input: Sequence[int],
        max_centroids: int
    ) -> None:
    d = TDigest.from_values(list(start_range), max_centroids=max_centroids)
    if update_method == "update":
        for x in update_input:
            getattr(d, update_method)(x)
    else:
        getattr(d, update_method)(update_input)
    expected = calculate_sample_quantiles(range(1, 101))
    check_sample_quantiles(d, expected)
    expected_n = len(start_range) + len(update_input)
    assert d.n_values == expected_n
    assert d.n_centroids <= max_centroids + 1

# -------------------------------------------------------------------
# Quantile tests (quantile, percentile, median, iqr, min, max)
# -------------------------------------------------------------------
def test_quantile_median_min_max(empty_digest: TDigest) -> None:
    data = list(range(2, 199))
    random.shuffle(data)
    d = TDigest.from_values(data)
    expected = calculate_sample_quantiles(range(2, 199))
    check_sample_quantiles(d, expected)
    with pytest.raises(ValueError):
        empty_digest.quantile(0.5)
    p = d.percentile(50)
    assert math.isclose(p, 100.0, rel_tol=RTOL, abs_tol=ATOL)
    m = d.median()
    assert math.isclose(m, 100.0, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(d.iqr(), 98.0, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(d.min(), 2.0, rel_tol=RTOL, abs_tol=EPS)
    assert math.isclose(d.max(), 198.0, rel_tol=RTOL, abs_tol=EPS)

# -------------------------------------------------------------------
# CDF tests (cdf, probability)
# -------------------------------------------------------------------
def test_cdf_methods(empty_digest: TDigest) -> None:
    d = TDigest.from_values(range(1, 101))
    rank_est = d.cdf(50)
    expected_rank = (50 - 1) / (100 - 1)
    assert 0 <= rank_est <= 1
    assert math.isclose(
        rank_est, expected_rank, rel_tol=RTOL, abs_tol=ATOL
    )
    with pytest.raises(ValueError):
        empty_digest.cdf(50)
    p_est = d.probability(80, 100)
    expected_p = ((100 - 1) - (80 - 1)) / (100 - 1)
    assert math.isclose(
        p_est, expected_p, rel_tol=RTOL, abs_tol=ATOL
    )

# -------------------------------------------------------------------
# Mean tests (mean, trimmed_mean, sum)
# -------------------------------------------------------------------
def test_mean_trimmed_mean_sum(empty_digest: TDigest) -> None:
    values = list(range(1, 101))
    d = TDigest.from_values(values)
    assert math.isclose(d.mean(), 50.5, rel_tol=RTOL, abs_tol=EPS)
    values[-1] = 10_000
    d = TDigest.from_values(values)
    trimmed = d.trimmed_mean(0.01, 0.99)
    assert math.isclose(trimmed, 50.5, rel_tol=RTOL, abs_tol=ATOL)
    with pytest.raises(ValueError):
        d.trimmed_mean(0.9, 0.1)
    with pytest.raises(ValueError):
        empty_digest.trimmed_mean(0.01, 0.99)
    true_sum = sum(values)
    assert math.isclose(d.sum(), true_sum, rel_tol=RTOL, abs_tol=ATOL)

# -------------------------------------------------------------------
# Serialization tests: to/from dict and pickle
# -------------------------------------------------------------------
def test_to_from_dict() -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    d_dict = d.to_dict()
    assert isinstance(d_dict, dict)
    new_d = TDigest.from_dict(d_dict)
    check_tdigest_equality(d, new_d)
    d = TDigest.from_values(range(1, 101), max_centroids=3)
    d_dict = d.to_dict()
    new_d = TDigest.from_dict(d_dict)
    check_tdigest_equality(d, new_d)
    d = TDigest()
    d_dict = d.to_dict()
    assert isinstance(d_dict, dict)
    new_d = TDigest.from_dict(d_dict)
    assert isinstance(new_d, TDigest)
    assert d == new_d
    d = TDigest(3)
    d_dict = d.to_dict()
    new_d = TDigest.from_dict(d_dict)
    assert isinstance(new_d, TDigest)
    assert d == new_d

@pytest.mark.parametrize("copy_func", [
    lambda d: d.copy(),
    lambda d: copy(d),
    lambda d: deepcopy(d),
])
def test_copy_methods(copy_func: Callable[[TDigest], TDigest]) -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    d_copy = copy_func(d)
    check_tdigest_equality(d, d_copy)
    assert id(d_copy) != id(d)
    empty = TDigest()
    empty_copy = copy_func(empty)
    assert len(empty_copy) == 0

def test_pickle_unpickle() -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    dumped = pickle.dumps(d)
    unpickled = pickle.loads(dumped)
    check_tdigest_equality(d, unpickled)
    d = TDigest.from_values(range(1, 101), max_centroids=3)
    dumped = pickle.dumps(d)
    unpickled = pickle.loads(dumped)
    check_tdigest_equality(d, unpickled)
    d = TDigest()
    dumped = pickle.dumps(d)
    unpickled = pickle.loads(dumped)
    assert d == unpickled

# -------------------------------------------------------------------
# Bool, length, representation, iteration, and equality tests
# -------------------------------------------------------------------
def test_bool_len_repr(empty_digest: TDigest) -> None:
    assert not empty_digest
    d = TDigest.from_values([1.0, 2.0, 3.0])
    assert d
    length = len(d)
    assert isinstance(length, int)
    assert length == d.n_centroids, (
        f"Expected {d.n_centroids}, got {length}"
    )
    rep = repr(d)
    assert rep == f"TDigest(max_centroids={DEFAULT_MAX_CENTROIDS})", (
        f"__repr__ output unexpected: {rep}"
    )
    d = TDigest.from_values([1.0, 2.0, 3.0], max_centroids=100)
    rep = repr(d)
    assert rep == "TDigest(max_centroids=100)", (
        f"__repr__ output unexpected: {rep}"
    )

def test_iter() -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    iterator = iter(d)
    assert isinstance(iterator, collections.abc.Iterator)
    assert list(iterator) == d.centroids

def test_equality() -> None:
    d1 = TDigest.from_values([1.0, 2.0, 3.0])
    d2 = TDigest.from_values([2.0, 1.0, 3.0])
    d3 = TDigest.from_values([1.0, 2.0, 3.1])
    d4 = TDigest.from_values([1.0, 2.0, 3.0], max_centroids=3)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    empty1 = TDigest()
    empty2 = TDigest()
    assert empty1 == empty2
    assert d1 != empty1
