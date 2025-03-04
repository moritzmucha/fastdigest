from fastdigest import TDigest, merge_all
from utils import calculate_sample_quantiles, check_sample_quantiles


def test_merge_all() -> None:
    min_max_centroids = 10
    digests = [
        TDigest.from_values(range(i, i+10)) for i in range(1, 100, 10)
    ]
    # Append an empty digest
    digests.append(TDigest())
    merged = merge_all(iter(digests))
    expected = calculate_sample_quantiles(range(1, 101))
    check_sample_quantiles(merged, expected)
    assert merged.n_values == 100, (
        f"Expected 100 values, got {merged.n_values}"
    )
    max_c = min_max_centroids
    merged = merge_all(digests, max_centroids=max_c)
    check_sample_quantiles(merged, expected)
    assert merged.n_centroids <= max_c + 1, (
        f"Expected {max_c} centroids, got {merged.n_centroids}"
    )
    for i, d in enumerate(digests[:-1]):
        d.max_centroids = min_max_centroids + i
    merged = merge_all(digests)
    assert merged.n_values == 100, (
        f"Expected 100 values, got {merged.n_values}"
    )
    min_c = min_max_centroids + 9
    max_c = 50
    digests[-1].max_centroids = max_c
    merged = merge_all(digests)
    check_sample_quantiles(merged, expected)
    assert min_c <= merged.n_centroids <= max_c + 1, (
        f"Expected between {min_c} and {max_c} centroids, "
        f"got {merged.n_centroids}"
    )
    empty_digests = [TDigest(max_centroids=i) for i in range(10)]
    merged_empty = merge_all(empty_digests)
    assert merged_empty == TDigest(max_centroids=9)
    merged_empty = merge_all([], max_centroids=3)
    assert merged_empty == TDigest(max_centroids=3)
