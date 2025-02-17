from typing import Dict, List, Sequence, Any

class TDigest:
    def __init__(self, values: Sequence[float]) -> None:
        """
        Initialize a TDigest with a non-empty sequence of float values.

        :param values: Sequence of float values
        """
        ...

    @property
    def n_centroids(self) -> int:
        """
        Number of centroids in the TDigest.

        :return: Number of centroids
        """
        ...

    @property
    def n_values(self) -> int:
        """
        Total number of data points ingested.

        :return: The sum of all centroid weights, rounded to the nearest integer.
        """
        ...

    def estimate_quantile(self, q: float) -> float:
        """
        Estimate the value at a given cumulative probability (quantile).

        :param q: Float between 0 and 1 representing cumulative probability
        :return: Estimated quantile value
        """
        ...

    def estimate_rank(self, x: float) -> float:
        """
        Estimate the cumulative probability (rank) of a given value x.

        :param x: Value for which to compute the rank
        :return: Float between 0 and 1 representing cumulative probability
        """
        ...

    def merge(self, other: "TDigest") -> "TDigest":
        """
        Merge this TDigest with another, returning a new TDigest.

        :param other: Other TDigest instance
        :return: New TDigest representing the merged data
        """
        ...

    def compress(self, max_centroids: int) -> None:
        """
        Compress the TDigest in-place to `max_centroids`.

        :param max_centroids: Maximum number of centroids allowed

        **Note:** compression below `min(n_values, 3)` centroids is not possible.
        """
        ...

    def trimmed_mean(self, q1: float, q2: float) -> float:
        """
        Estimate the trimmed mean (truncated mean) of the data,
        excluding values below the `q1` and above the `q2` quantiles.

        :param q1: Lower quantile threshold (0 <= q1 < q2)
        :param q2: Upper quantile threshold (q1 < q2 <= 1)
        :return: The trimmed mean value
        """
        ...

    def to_dict(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Return a dictionary representation of the TDigest.

        The returned dictionary contains a key "centroids" that maps to a list of
        centroids, where each centroid is a dictionary with keys "m" and "c".

        :return: Dictionary representation of the TDigest
        """
        ...

    @staticmethod
    def from_dict(tdigest_dict: Dict[str, Any]) -> "TDigest":
        """
        Construct a TDigest from a dictionary representation.

        The dictionary must have a key "centroids" mapping to a list of centroids.
        Each centroid should be a dictionary with keys "m" (float) and "c" (float).

        :param tdigest_dict: Dictionary with centroids
        :return: TDigest instance
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of centroids in the TDigest.

        :return: Number of centroids
        """
        ...

    def __repr__(self) -> str:
        """
        Return a string representation summarizing the TDigest.

        :return: string representation of the TDigest
        """
        ...
