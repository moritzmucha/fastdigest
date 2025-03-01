import pytest
import math
import pickle
from copy import copy, deepcopy
from typing import Callable, Optional, Sequence, Union, List
from fastdigest import TDigest
from utils import (
    RTOL,
    ATOL,
    DEFAULT_MAX_CENTROIDS,
    check_median,
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
    (1, 2, 3, 4, 5),
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
    d.max_centroids = None
    assert d.max_centroids is None
    d.max_centroids = 3
    assert d.max_centroids == 3

def test_n_values_and_n_centroids(empty_digest: TDigest) -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    assert isinstance(d.n_values, int) and d.n_values == 3
    assert isinstance(d.n_centroids, int) and d.n_centroids == 3
    assert empty_digest.n_values == 0
    assert empty_digest.n_centroids == 0

# -------------------------------------------------------------------
# Compression test
# -------------------------------------------------------------------
def test_compress() -> None:
    d = TDigest.from_values(range(1, 101))
    d.compress(5)
    assert 3 <= d.n_centroids <= 5, (
        f"Expected between 3 and 5 centroids, got {d.n_centroids}"
    )
    check_median(d, 50.5)
    empty = TDigest()
    empty.compress(5)
    assert len(empty) == 0

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
    merged = merge_func(d1, d2)
    check_median(merged, 50.5)
    assert merged.n_values == 100

def test_merge_with_max_centroids() -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    d1.max_centroids = 3
    merged = d1.merge(d2)
    assert merged.n_values == 100
    d2.max_centroids = 50
    merged = d1.merge(d2)
    assert 3 < merged.n_centroids <= 50, (
        f"Expected between 4 and 50 centroids, got {merged.n_centroids}"
    )
    d2.max_centroids = 3
    merged = d1.merge(d2)
    assert merged.n_centroids == 3, (
        f"Expected 3 centroids, got {merged.n_centroids}"
    )

def test_merge_inplace() -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    d1.merge_inplace(d2)
    check_median(d1, 50.5)
    assert d1.n_values == 100
    d1.max_centroids = 3
    d1.merge_inplace(d2)
    assert d1.n_centroids == 3
    d2.max_centroids = 50
    d1.merge_inplace(d2)
    assert d1.n_centroids == 3
    empty = TDigest()
    d = TDigest.from_values(range(1, 51))
    d.merge_inplace(empty)
    check_median(d, 25.5)
    empty.merge_inplace(d)
    check_median(empty, 25.5)

@pytest.mark.parametrize("iadd_op", [
    lambda d1, d2: d1 + d2,
    lambda d1, d2: d1.__iadd__(d2) or d1,
])
def test_add_iadd(iadd_op: Callable[[TDigest, TDigest], TDigest]) -> None:
    d1 = TDigest.from_values(range(1, 51))
    d2 = TDigest.from_values(range(51, 101))
    result = iadd_op(d1, d2)
    check_median(result, 50.5)

def test_add_with_empty_max_centroids(empty_digest: TDigest) -> None:
    digest = TDigest.from_values(range(101))
    digest.max_centroids = 3
    empty_digest.max_centroids = 3
    merged = digest + empty_digest
    assert len(merged) == 3

# -------------------------------------------------------------------
# Update tests (batch_update and update)
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    "update_method, update_input, start_range, max_centroids", [
        ("batch_update", range(51, 101), range(1, 51), None),
        ("batch_update", range(51, 101), range(1, 51), 10),
        ("batch_update", [], range(1, 101), None),
        ("update", 100, range(1, 100), 99)
    ]
)
def test_updates(
        update_method: str,
        update_input: Union[range, int],
        start_range: range,
        max_centroids: Optional[int]
    ) -> None:
    d = TDigest.from_values(list(start_range), max_centroids=max_centroids)
    getattr(d, update_method)(update_input)
    check_median(d, 50.5)
    expected_n = (
        len(start_range) +
        (len(update_input) if update_method == "batch_update" else 1)
    )
    assert d.n_values == expected_n
    if max_centroids is not None:
        assert d.n_centroids <= max_centroids

# -------------------------------------------------------------------
# Quantile, percentile, and CDF tests
# -------------------------------------------------------------------
def test_quantile_percentile_cdf(empty_digest: TDigest) -> None:
    d = TDigest.from_values(range(2, 199))
    check_median(d, 100.0)
    with pytest.raises(ValueError):
        empty_digest.quantile(0.5)
    p = d.percentile(50)
    assert math.isclose(p, 100, rel_tol=RTOL, abs_tol=ATOL)
    with pytest.raises(ValueError):
        empty_digest.percentile(50)
    d2 = TDigest.from_values(range(1, 101))
    rank_est = d2.cdf(50)
    expected_rank = (50 - 1) / (100 - 1)
    assert 0 <= rank_est <= 1
    assert math.isclose(
        rank_est, expected_rank, rel_tol=RTOL, abs_tol=ATOL
    )
    with pytest.raises(ValueError):
        empty_digest.cdf(50)

# -------------------------------------------------------------------
# Trimmed mean test
# -------------------------------------------------------------------
def test_trimmed_mean(empty_digest: TDigest) -> None:
    values = list(range(101))
    values.append(10_000)
    d = TDigest.from_values(values)
    trimmed = d.trimmed_mean(0.01, 0.99)
    assert math.isclose(trimmed, 50.5, rel_tol=RTOL, abs_tol=ATOL)
    with pytest.raises(ValueError):
        d.trimmed_mean(0.9, 0.1)
    with pytest.raises(ValueError):
        empty_digest.trimmed_mean(0.01, 0.99)

# -------------------------------------------------------------------
# Serialization tests: to/from dict and pickle
# -------------------------------------------------------------------
def test_to_from_dict() -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    d_dict: dict = d.to_dict()
    assert isinstance(d_dict, dict)
    new_d = TDigest.from_dict(d_dict)
    assert d == new_d
    check_tdigest_equality(d, new_d)
    d = TDigest.from_values(range(1, 101), max_centroids=3)
    d_dict = d.to_dict()
    new_d = TDigest.from_dict(d_dict)
    assert d == new_d
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
    assert d == d_copy
    check_tdigest_equality(d, d_copy)
    assert id(d_copy) != id(d)
    empty = TDigest()
    empty_copy = copy_func(empty)
    assert len(empty_copy) == 0

def test_pickle_unpickle() -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
    dumped = pickle.dumps(d)
    unpickled = pickle.loads(dumped)
    assert d == unpickled
    check_tdigest_equality(d, unpickled)
    d = TDigest.from_values(range(1, 101), max_centroids=3)
    dumped = pickle.dumps(d)
    unpickled = pickle.loads(dumped)
    assert d == unpickled
    check_tdigest_equality(d, unpickled)
    d = TDigest()
    dumped = pickle.dumps(d)
    unpickled = pickle.loads(dumped)
    assert d == unpickled

# -------------------------------------------------------------------
# Length, representation, and equality tests
# -------------------------------------------------------------------
def test_len_repr() -> None:
    d = TDigest.from_values([1.0, 2.0, 3.0])
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

def test_equality() -> None:
    d1 = TDigest.from_values([1.0, 2.0, 3.0])
    d2 = TDigest.from_values([1.0, 2.0, 3.0])
    d3 = TDigest.from_values([1.0, 2.0, 3.1])
    d4 = TDigest.from_values([1.0, 2.0, 3.0], max_centroids=3)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    empty1 = TDigest()
    empty2 = TDigest()
    assert empty1 == empty2
    assert d1 != empty1
