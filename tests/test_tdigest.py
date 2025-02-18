import math
import pickle
import pytest
from fastdigest import TDigest

def test_init():
    # Test proper initialization with a non-empty list
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    digest = TDigest(values)
    assert len(digest) > 0

    # Test that an empty list raises ValueError
    with pytest.raises(ValueError):
        TDigest([])

def test_n_values():
    digest = TDigest([1.0, 2.0, 3.0])
    n_values = digest.n_values
    assert isinstance(n_values, int), (
        f"Expected int, got {type(n_values).__name__}"
    )
    assert n_values == 3, f"Expected 3, got {n_values}"

def test_n_centroids():
    digest = TDigest([1.0, 2.0, 3.0])
    n_centroids = digest.n_centroids
    assert isinstance(n_centroids, int), (
        f"Expected int, got {type(n_centroids).__name__}"
    )
    assert n_centroids == 3, f"Expected 3, got {n_centroids}"

def test_len():
    digest = TDigest([1.0, 2.0, 3.0])
    length = len(digest)
    assert isinstance(length, int), (
        f"Expected int, got {type(length).__name__}"
    )
    assert length == 3, f"Expected 3, got {length}"

def test_repr():
    digest = TDigest([1.0, 2.0, 3.0])
    rep = repr(digest)
    assert rep == "TDigest(n_values=3, n_centroids=3)", (
        f"__repr__ output unexpected: {rep}"
    )

def test_estimate_quantile():
    # Create a digest from 1..100
    digest = TDigest(range(1, 101))
    # For a uniformly distributed dataset, the median should be near 50.5
    q = 0.5
    quantile_est = digest.estimate_quantile(q)
    expected = 1 + q * (100 - 1)  # 1 + 0.5*99 = 50.5
    assert math.isclose(quantile_est, expected, rel_tol=1e-3), (
        f"Expected ~{expected}, got {quantile_est}"
    )

def test_estimate_rank():
    digest = TDigest(range(1, 101))
    x = 50
    rank_est = digest.estimate_rank(x)
    # For uniform data, expected rank is (x - min)/(max - min)
    expected = (50 - 1) / (100 - 1)
    assert 0 <= rank_est <= 1, "Rank should be between 0 and 1"
    assert math.isclose(rank_est, expected, rel_tol=1e-3), (
        f"Expected ~{expected}, got {rank_est}"
    )

def test_merge():
    # Create two TDigest instances from non-overlapping ranges
    digest1 = TDigest(range(1, 51))
    digest2 = TDigest(range(51, 101))
    merged = digest1.merge(digest2)
    # The median of the merged data should be around 50.5
    quantile_est = merged.estimate_quantile(0.5)
    expected = 50.5
    assert math.isclose(quantile_est, expected, rel_tol=1e-3), (
        f"Expected median ~{expected}, got {quantile_est}"
    )

def test_compress():
    digest = TDigest(range(1, 101))
    # Compress the digest to at most 5 centroids. Note that for N values
    # ingested, it will never go below min(N, 3) centroids.
    digest.compress(5)
    compressed_centroids = len(digest)
    assert 3 <= compressed_centroids <= 5, (
        f"Expected between 3 and 5 centroids, got {compressed_centroids}"
    )
    # Check that quantile estimates remain plausible after compression
    quantile_est = digest.estimate_quantile(0.5)
    expected = 50.5
    assert math.isclose(quantile_est, expected, rel_tol=1e-3), (
        f"Expected median ~{expected}, got {quantile_est}"
    )

def test_trimmed_mean():
    values = list(range(101))
    values.append(10_000)
    digest = TDigest(values)
    # 1st percentile is 1.01, 99th percentile is 99.99. (2 + 99) / 2 = 50.5
    trimmed = digest.trimmed_mean(0.01, 0.99)
    expected = 50.5
    assert math.isclose(trimmed, expected, rel_tol=1e-3), (
        f"Expected trimmed mean ~{expected}, got {trimmed}"
    )
    # Ensure that providing invalid quantiles raises a ValueError.
    with pytest.raises(ValueError):
        digest.trimmed_mean(0.9, 0.1)

def check_tdigest_equality(original, new):
    # Sanity checks
    assert isinstance(new, TDigest), (
        f"Expected TDigest, got {type(new).__name__}"
    )
    assert new.n_values == original.n_values, (
        f"Expected {original.n_values} values, got {new.n_values}"
    )
    assert new.n_centroids == original.n_centroids, (
        f"Expected {original.n_centroids} centroids, "
        f"got {new.n_centroids}"
    )

    # Verify that quantile estimates match within a reasonable tolerance
    for q in [0.25, 0.5, 0.75]:
        orig_val = original.estimate_quantile(q)
        new_val = new.estimate_quantile(q)
        assert math.isclose(orig_val, new_val, rel_tol=1e-9), (
            f"Quantile {q} mismatch: original {orig_val} vs new {new_val}"
        )

def test_to_from_dict():
    original = TDigest(range(1, 101))
    digest_dict = original.to_dict()
    assert isinstance(digest_dict, dict), (
        f"Expected dict, got {type(digest_dict).__name__}"
    )
    new = TDigest.from_dict(digest_dict)
    check_tdigest_equality(original, new)

def test_pickle_unpickle():
    original = TDigest(range(1, 101))
    dumped = pickle.dumps(original)
    unpickled = pickle.loads(dumped)
    check_tdigest_equality(original, unpickled)

if __name__ == "__main__":
    test_init()
    test_n_values()
    test_n_centroids()
    test_len()
    test_repr()
    test_estimate_quantile()
    test_estimate_rank()
    test_merge()
    test_compress()
    test_trimmed_mean()
    test_to_from_dict()
    test_pickle_unpickle()
