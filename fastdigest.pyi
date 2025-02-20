from typing import Dict, List, Sequence, Tuple, Union, Any

class TDigest:
    def __init__(self, values: Sequence[Union[float, int]]) -> None:
        """
        Initialize a TDigest with a non-empty sequence of numerical values.

        :param values: Sequence of float or int values
        """
        ...

    @property
    def n_values(self) -> int:
        """
        Total number of data points ingested.

        :return: Sum of all centroid weights, rounded to the nearest integer
        """
        ...

    @property
    def n_centroids(self) -> int:
        """
        Number of centroids in the TDigest.

        :return: Number of centroids
        """
        ...

    def compress(self, max_centroids: int) -> None:
        """
        Compress the TDigest in-place to `max_centroids` (or fewer)
        centroids.

        **Note:** there is a lower limit of `min(n_values, 3)` centroids.

        :param max_centroids: Maximum number of centroids allowed
        """
        ...

    def merge(self, other: "TDigest") -> "TDigest":
        """
        Merge this TDigest with another, returning a new TDigest.

        :param other: Other TDigest instance
        :return: New TDigest representing the merged data
        """
        ...

    def merge_inplace(self, other: "TDigest") -> None:
        """
        Merge another TDigest into this one in-place.

        :param other: Other TDigest instance
        :return: The modified TDigest (self)
        """
        ...
    
    def batch_update(self, values: Sequence[Union[float, int]]) -> None:
        """
        Update the TDigest in-place with a non-empty sequence of numbers.

        This is equivalent to creating a temporary TDigest from the values
        and merging it into `self`.

        :param values: Sequence of values to add
        """
        ...

    def update(self, value: Union[float, int]) -> None:
        """
        Update the TDigest in-place with a single value.

        This is equivalent to `self.batch_update([value])`.

        :param value: Single value to add
        """
        ...

    def quantile(self, q: float) -> float:
        """
        Estimate the value at a given cumulative probability (quantile).

        :param q: Float between 0 and 1 representing cumulative probability
        :return: Estimated quantile value
        """
        ...

    def percentile(self, p: Union[float, int]) -> float:
        """
        Estimate the value at a given cumulative probability (percentile).

        Convenience method, same as `quantile(p/100)`.

        :param p: Number between 0 and 100 (cumulative probability in percent)
        :return: Estimated percentile value
        """
        ...

    def rank(self, x: float) -> float:
        """
        Estimate the cumulative probability (rank) of a given value x.

        :param x: Value for which to compute the rank
        :return: Float between 0 and 1 representing cumulative probability
        """
        ...

    def trimmed_mean(self, q1: float, q2: float) -> float:
        """
        Estimate the trimmed mean (truncated mean) of the data,
        excluding values below the `q1` and above the `q2` quantiles.

        :param q1: Lower quantile threshold (0 <= q1 < q2)
        :param q2: Upper quantile threshold (q1 < q2 <= 1)
        :return: Trimmed mean value
        """
        ...

    def to_dict(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Return a dictionary representation of the TDigest.

        The returned dict contains a key "centroids" that maps to a list of
        centroids, where each centroid is a dict with keys "m" and "c".

        :return: Dictionary representation of the TDigest
        """
        ...

    @staticmethod
    def from_dict(tdigest_dict: Dict[str, Any]) -> "TDigest":
        """
        Construct a TDigest from a dictionary representation.

        The dict must have a key "centroids" mapping to a list of centroids.
        Each centroid should be a dict with keys "m" (float) and "c" (float).

        :param tdigest_dict: Dictionary with centroids
        :return: TDigest instance
        """
        ...

    def copy(self) -> "TDigest":
        """
        Returns a copy of the TDigest instance.

        :return: Copy of the TDigest instance
        """
        ...

    def __copy__(self) -> "TDigest":
        """
        Returns a copy of the TDigest instance.

        :return: Copy of the TDigest instance
        """
        ...

    def __deepcopy__(self) -> "TDigest":
        """
        Returns a copy of the TDigest instance.

        :return: Copy of the TDigest instance
        """
        ...

    def __reduce__(self) -> Tuple[object, Tuple[Any, ...]]:
        """
        Enables pickle support by returning a tuple (callable, args) that
        can be used to reconstruct the TDigest.

        :return: Tuple of (reconstruction function, arguments)
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

        :return: String representation of the TDigest
        """
        ...

    def __add__(self, other: "TDigest") -> "TDigest":
        """
        Merge this TDigest with another, returning a new TDigest.
        
        Equivalent to `self.merge(other)`, but using the `+` operator.

        :param other: Other TDigest instance
        :return: New TDigest representing the merged data
        """
        ...

    def __iadd__(self, other: "TDigest") -> "TDigest":
        """
        Merge another TDigest into this one in-place.
        
        Equivalent to `self.merge_inplace(other)`, but using the `+=` operator.

        :param other: Other TDigest instance
        """
        ...
