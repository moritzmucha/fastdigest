from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

class TDigest:
    """
    Class containing the t-digest data structure.

    :param optional max_centroids:
        Number of centroids to maintain. A lower value enables a
        smaller memory footprint and faster computation speed at the
        cost of some accuracy. 0 disables compression. Default is 1000.
    """

    def __init__(self, max_centroids: int = 1000) -> None:
        """
        Creates an empty new TDigest instance.

        :param optional max_centroids:
            Number of centroids to maintain. A lower value enables a
            smaller memory footprint and faster computation speed at the
            cost of some accuracy. 0 disables compression. Default is 1000.
        """
        ...

    @staticmethod
    def from_values(
        x: Sequence[float],
        w: Optional[Union[Sequence[float], float]] = None,
        max_centroids: int = 1000,
    ) -> "TDigest":
        """
        Creates a new TDigest from a sequence of numeric values.

        :param x: Sequence of numeric values.
        :param optional w:
            Weights. This can be either a sequence of the same length as `x`,
            or a scalar that will be used as the weight for the entire batch.
            If `None` (default), each value has a weight of 1.
        :param optional max_centroids:
            Number of centroids to maintain. A lower value enables a
            smaller memory footprint and faster computation speed at the
            cost of some accuracy. 0 disables compression. Default is 1000.
        """
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "TDigest":
        """
        Deserialize a TDigest from its binary representation.

        :param data: Binary representation produced by `to_bytes()`.
        :return: TDigest instance.
        """
        ...

    @staticmethod
    def from_dict(tdigest_dict: Dict[str, Any]) -> "TDigest":
        """
        Construct a TDigest from a dictionary representation.

        The dict must have a key "centroids" mapping to a list of centroids.
        Each centroid should be a dict with keys "m" (float) and "c" (float).

        :param tdigest_dict: Dictionary with centroids.
        :return: TDigest instance.
        """
        ...

    @property
    def max_centroids(self) -> int:
        """
        Instance parameter controlling the maximum size (number of centroids)
        of the data structure.

        :return: Maximum number of centroids parameter.
        """
        ...

    @max_centroids.setter
    def max_centroids(self, value: int) -> None:
        ""
        ...

    @property
    def mass(self) -> float:
        """
        Total weight of all data points fed into this TDigest.

        :return: Sum of all centroid weights.
        """
        ...

    @property
    def n_values(self) -> int:
        """
        Total number of individual data points fed into this TDigest.

        :return: Number of values ingested.
        """
        ...

    @property
    def n_centroids(self) -> int:
        """
        Number of centroids currently in this TDigest.

        :return: Number of centroids.
        """
        ...

    @property
    def is_empty(self) -> bool:
        """
        True if no data has been ingested yet.

        :return: True if empty, False otherwise.
        """
        ...

    @property
    def centroids(self) -> List[Tuple[float, float]]:
        """
        List of centroids in the TDigest as tuples of (mean, weight).

        :return: List of (mean, weight) tuples.
        """
        ...

    def merge(self, other: "TDigest") -> "TDigest":
        """
        Merges this TDigest with another, returning a new instance.

        Equivalent to the `+` operator.

        If the instances have different `max_centroids` parameters, the result
        will use the higher value.

        :param other: Other TDigest instance.
        :return: New TDigest representing the merged data.
        """
        ...

    def merge_inplace(self, other: "TDigest") -> None:
        """
        Merges another TDigest into `self`, modifying the calling object
        in-place.

        Equivalent to the `+=` operator.

        :param other: Other TDigest instance.
        """
        ...

    def batch_update(
        self,
        x: Sequence[float],
        w: Optional[Union[Sequence[float], float]] = None,
    ) -> None:
        """
        Updates the TDigest in-place with a sequence of numeric values.

        :param x: Sequence of values to add.
        :param optional w:
            Weights. This can be either a sequence of the same length as `x`,
            or a scalar that will be used as the weight for the entire batch.
            If `None` (default), each value has a weight of 1.
        """
        ...

    def update(self, x: float, w: Optional[float] = None) -> None:
        """
        Updates the TDigest in-place with a numeric value.

        Optimized for low-latency single updates (by writing to an internal
        buffer that is merged lazily).

        :param x: Value to add.
        :param optional w: Weight for `x`. Default is `None` (1).
        """
        ...

    def quantile(self, q: float) -> float:
        """
        Estimates the value at the given relative rank/cumulative probability
        `q`.

        Inverse function of `cdf(x)`.

        :param q: Float between 0 and 1.
        :return: Estimated quantile value.
        """
        ...

    def quantile_vec(self, q: Sequence[float]) -> List[float]:
        """
        Estimates the value at the given relative ranks/cumulative probabilities
        `q`.

        Inverse function of `cdf_vec(x)`.

        :param q: Sequence of floats between 0 and 1.
        :return: List of estimated quantile values.
        """
        ...

    def percentile(self, p: float) -> float:
        """
        Estimates the value at a given cumulative probability (percentile).

        Equivalent to `quantile(p/100)`.

        :param p: Number between 0 and 100 (cumulative probability in percent).
        :return: Estimated percentile value.
        """
        ...

    def median(self) -> float:
        """
        Estimates the median value.

        Equivalent to `quantile(0.5)`.

        :return: Estimated median.
        """
        ...

    def iqr(self) -> float:
        """
        Estimates the interquartile range (IQR).

        Equivalent to `quantile(0.75) - quantile(0.25)`.

        :return: Estimated IQR.
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Estimates the cumulative distribution function (CDF) at the value `x`.

        Inverse function of `quantile(q)`.

        :param x: Value for which to compute the CDF.
        :return: Float between 0 and 1 representing cumulative probability.
        """
        ...

    def cdf_vec(self, x: Sequence[float]) -> List[float]:
        """
        Estimates the cumulative distribution function (CDF) at the values `x`.

        Inverse function of `quantile_vec(q)`.

        :param x: Sequence of values for which to compute the CDF.
        :return: List of CDF(x) floats between 0 and 1.
        """
        ...

    def probability(self, x1: float, x2: float) -> float:
        """
        Estimates the probability of finding a value in the interval
        [`x1`, `x2`].

        Equivalent to `cdf(x2) - cdf(x1)`.

        :param x1: Lower bound of the interval.
        :param x2: Upper bound of the interval.
        :return: Float between 0 and 1 representing probability.
        """
        ...

    def sum(self) -> float:
        """
        Returns the sum of all ingested values.

        :return: Sum of all values.
        """
        ...

    def mean(self) -> float:
        """
        Returns the arithmetic mean of all ingested values.

        :return: Arithmetic mean.
        """
        ...

    def trimmed_mean(self, q1: float, q2: float) -> float:
        """
        Estimates the trimmed mean (truncated mean) of the data,
        excluding values below the `q1` and above the `q2` quantiles.

        :param q1: Lower quantile threshold (`0 <= q1 < q2`).
        :param q2: Upper quantile threshold (`q1 < q2 <= 1`).
        :return: Trimmed mean value.
        """
        ...

    def min(self) -> float:
        """
        Returns the minimum of all ingested values.

        :return: Minimum value.
        """
        ...

    def max(self) -> float:
        """
        Returns the maximum of all ingested values.

        :return: Maximum value.
        """
        ...

    def to_bytes(self) -> bytes:
        """
        Returns a serialized binary representation of the TDigest.

        :return: Binary representation of the TDigest.
        """
        ...

    def to_dict(self) -> Dict[str, Union[float, int, List[Dict[str, float]]]]:
        """
        Returns a dictionary representation of the TDigest.

        The returned dict contains a "centroids" list, where each centroid
        is represented as a dict with keys "m" and "c".
        It also contains instance parameters such as `max_centroids`.

        :return: Dictionary representation of the TDigest.
        """
        ...

    def equals(self, other: "TDigest") -> bool:
        """
        Checks equality between two TDigest instances.

        Returns True if all centroids, properties, and `max_centroids` are
        exactly identical, otherwise False.

        Raises TypeError if `other` is not a TDigest.

        :param other: Other TDigest instance.
        :return: True if equal, False otherwise.
        """
        ...

    def copy(self) -> "TDigest":
        """
        Returns a copy of the TDigest instance.

        :return: Copy of the TDigest instance.
        """
        ...

    def __copy__(self) -> "TDigest":
        """
        Returns a copy of the TDigest instance.

        :return: Copy of the TDigest instance.
        """
        ...

    def __deepcopy__(self) -> "TDigest":
        """
        Returns a copy of the TDigest instance.

        :return: Copy of the TDigest instance.
        """
        ...

    def __reduce__(self) -> Tuple[object, Tuple[Any, ...]]:
        """
        Enables pickle support by returning a tuple (callable, args) that
        can be used to reconstruct the TDigest.

        :return: Tuple of (reconstruction function, arguments).
        """
        ...

    def __bool__(self) -> bool:
        """
        Returns True if the TDigest is not empty.

        :return: True if not empty, False otherwise.
        """
        ...

    def __len__(self) -> int:
        """
        Returns the number of centroids in the TDigest.

        :return: Number of centroids.
        """
        ...

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """
        Returns an iterator over the list of centroids.

        :return: Iterator over centroid (mean, weight) tuples.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the TDigest.

        :return: String representation of the TDigest.
        """
        ...

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between TDigest and another object.

        Equivalent to `self.equals(other)`, but comparisons with different
        types will return False instead of raising an error.

        :param other: Any Python object.
        :return: True if equal, False otherwise.
        """
        ...

    def __ne__(self, other: Any) -> bool:
        """
        Check inequality between TDigest and another object.

        Equivalent to `not self.equals(other)`, but comparisons with different
        types will return True instead of raising an error.

        :param other: Any Python object.
        :return: False if equal, True otherwise.
        """
        ...

    def __add__(self, other: "TDigest") -> "TDigest":
        """
        Merge this TDigest with another, returning a new TDigest.

        Equivalent to `self.merge(other)`, but using the `+` operator.

        :param other: Other TDigest instance.
        :return: New TDigest representing the merged data.
        """
        ...

    def __iadd__(self, other: "TDigest") -> "TDigest":
        """
        Merge another TDigest into this one in-place.

        Equivalent to `self.merge_inplace(other)`, but using the `+=` operator.

        :param other: Other TDigest instance.
        :return: The modified TDigest instance.
        """
        ...

def merge_all(
    digests: Iterable[TDigest], max_centroids: Optional[int] = None
) -> TDigest:
    """
    Merge an iterable of TDigests into a single new instance.

    If `max_centroids` is provided, the new instance will use this value.
    Otherwise, will inherit the largest `max_centroids` parameter found in the
    input TDigests.

    :param digests: Iterable of TDigest instances to merge.
    :param optional max_centroids:
        Parameter to be used for the new instance.
        If `None` (default), the value is determined from the input TDigests.
    :return: New TDigest representing the merged data.
    """
    ...

__version__: str
