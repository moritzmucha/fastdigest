from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    Optional,
    Union,
    Any
)

class TDigest:
    def __init__(
            self,
            max_centroids: int = 1000
        ) -> None:
        """
        Initialize an empty TDigest instance.

        :param max_centroids:
            Number of centroids to maintain. A lower value enables a
            smaller memory footprint and faster computation speed at the
            cost of some accuracy. 0 disables compression. Default is 1000.
        """
        ...

    @staticmethod
    def from_values(
            values: Sequence[Union[float, int]],
            max_centroids: int = 1000
        ) -> "TDigest":
        """
        Initialize a TDigest with a sequence of numerical values.

        :param values: Sequence of float or int values.
        :param max_centroids:
            Number of centroids to maintain. A lower value enables a
            smaller memory footprint and faster computation speed at the
            cost of some accuracy. 0 disables compression. Default is 1000.
        """
        ...

    @property
    def max_centroids(self) -> int:
        """
        The maximum number of centroids instance parameter.
        
        :return: Maximum number of centroids parameter.
        """
        ...

    @max_centroids.setter
    def max_centroids(self, value: int) -> None: ...

    @property
    def n_values(self) -> int:
        """
        Total number of data points ingested.

        :return: Sum of all centroid weights, rounded to the nearest integer.
        """
        ...

    @property
    def n_centroids(self) -> int:
        """
        Number of centroids in the TDigest.

        :return: Number of centroids.
        """
        ...

    @property
    def is_empty(self) -> bool:
        """
        True if no data has been ingested yet.

        :return: True if empty.
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
        Merge this TDigest with another, returning a new TDigest.

        Equivalent to the `+` operator.

        The resulting TDigest will use the higher of the two instances'
        `max_centroids` parameters.

        :param other: Other TDigest instance.
        :return: New TDigest representing the merged data.
        """
        ...

    def merge_inplace(self, other: "TDigest") -> None:
        """
        Merge another TDigest into `self`, modifying the calling object
        in-place.
        
        Equivalent to the `+=` operator.

        :param other: Other TDigest instance.
        """
        ...

    def batch_update(self, values: Sequence[Union[float, int]]) -> None:
        """
        Update the TDigest in-place with a sequence of numbers.

        This directly performs a merge, which is faster than looping over
        `update` in most cases.

        :param values: Sequence of values to add.
        """
        ...

    def update(self, value: Union[float, int]) -> None:
        """
        Update the TDigest in-place with a single value.

        This writes to a stack-allocated buffer before merging, which can be
        significantly faster than `batch_update` if you have to iteratively
        add one observed value at a time, e.g. in streaming applications.

        :param value: Single value to add.
        """
        ...

    def quantile(self, q: float) -> float:
        """
        Estimate the value at a given cumulative probability (quantile).

        Inverse function of `cdf(x)`.

        :param q: Float between 0 and 1 representing cumulative probability.
        :return: Estimated quantile value.
        """
        ...

    def percentile(self, p: Union[float, int]) -> float:
        """
        Estimate the value at a given cumulative probability (percentile).

        Equivalent to `quantile(p/100)`.

        :param p: Number between 0 and 100 (cumulative probability in percent).
        :return: Estimated percentile value.
        """
        ...

    def median(self) -> float:
        """
        Estimate the median value.

        Equivalent to `quantile(0.5)`.

        :return: Estimated median.
        """
        ...

    def iqr(self, q1: float, q2: float) -> float:
        """
        Estimate the interquartile range (IQR).

        Equivalent to `quantile(0.75) - quantile(0.25)`.

        :return: Estimated IQR.
        """
        ...

    def min(self) -> float:
        """
        Return the minimum of all ingested values.

        This is an exact value, not an estimate.

        :return: Minimum value.
        """
        ...

    def max(self) -> float:
        """
        Return the maximum of all ingested values.

        This is an exact value, not an estimate.

        :return: Maximum value.
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Estimate the cumulative distribution function (CDF) at the value `x`,
        aka the cumulative probability or relative rank.

        Inverse function of `quantile(q)`.

        :param x: Value for which to compute the cdf.
        :return: Float between 0 and 1 representing cumulative probability.
        """
        ...

    def probability(self, x1: float, x2: float) -> float:
        """
        Estimate the probability of finding a value in the interval
        [`x1`, `x2`].

        Equivalent to `cdf(x2) - cdf(x1)`.

        :return: Float between 0 and 1 representing probability.
        """
        ...

    def sum(self) -> float:
        """
        Return the sum of all ingested values.

        This is an exact value (aside from accumulated floating-point error),
        not an estimate.

        :return: Sum of all values.
        """
        ...

    def mean(self) -> float:
        """
        Calculate the arithmetic mean of all ingested values.

        This is an exact value (aside from accumulated floating-point error),
        not an estimate.

        :return: Arithmetic mean.
        """
        ...

    def trimmed_mean(self, q1: float, q2: float) -> float:
        """
        Estimate the trimmed mean (truncated mean) of the data,
        excluding values below the `q1` and above the `q2` quantiles.

        :param q1: Lower quantile threshold (0 <= q1 < q2).
        :param q2: Upper quantile threshold (q1 < q2 <= 1).
        :return: Trimmed mean value.
        """
        ...

    def to_dict(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Return a dictionary representation of the TDigest.

        The returned dict contains a "centroids" list, where each centroid
        is represented as a dict with keys "m" and "c".
        It also contains instance parameters such as `max_centroids`.

        :return: Dictionary representation of the TDigest.
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
        Return True if the TDigest is not empty.

        :return: True if not empty.
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of centroids in the TDigest.

        :return: Number of centroids.
        """
        ...

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """
        Return an iterator over the list of centroids.

        :return: Iterator over centroid (mean, weight) tuples.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a string representation of the TDigest.

        :return: String representation of the TDigest.
        """
        ...

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between two TDigest instances.

        Returns True if all centroids are the same and `max_centroids` has the
        same value, otherwise False.

        :param other: Other TDigest instance.
        :return: Bool representing equality.
        """
        ...

    def __ne__(self, other: Any) -> bool:
        """
        Check inequality between two TDigest instances.

        Returns False if all centroids are the same and `max_centroids` has the
        same value, otherwise True.

        :param other: Other TDigest instance.
        :return: Bool representing inequality.
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
        """
        ...

def merge_all(
        digests: Iterable[TDigest],
        max_centroids: Optional[int] = None
    ) -> TDigest:
    """
    Merge an iterable of TDigest instances into a single TDigest.

    If `max_centroids` is provided, this value will be set in the new TDigest.

    Otherwise, the resulting TDigest will use the maximum of the parameters of
    the input instances.

    :param digests: Iterable of TDigest instances to merge.
    :param optional max_centroids:
        Parameter to be used for the new instance.
        If None, the value is determined from the source TDigests.
        Default is None.
    :return: New TDigest representing the merged data.
    """
    ...

__version__: str
