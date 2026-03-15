### Contents

- [Initialization](#initialization)
  - [TDigest()](#tdigest)
  - [TDigest.from_values(x, w=None)](#tdigestfrom_valuesx-wnone)
- [Mathematical functions](#mathematical-functions)
  - [self.quantile(q)](#selfquantileq)
  - [self.percentile(p)](#selfpercentilep)
  - [self.median()](#selfmedian)
  - [self.iqr()](#selfiqr)
  - [self.cdf(x)](#selfcdfx)
  - [self.probability(x1, x2)](#selfprobabilityx1-x2)
  - [self.sum()](#selfsum)
  - [self.mean()](#selfmean)
  - [self.trimmed_mean(q1, q2)](#selftrimmed_meanq1-q2)
  - [self.min()](#selfmin)
  - [self.max()](#selfmax)
  - [self.mad()](#selfmad)
  - [self.var()](#selfvar)
  - [self.std()](#selfstd)
  - [self.is_normal()](#selfis_normal)
- [Vectorized mathematical functions](#vectorized-mathematical-functions)
  - [self.quantile_vec(q)](#selfquantile_vecq)
  - [self.cdf_vec(x)](#selfcdf_vecx)
- [Updating a TDigest](#updating-a-tdigest)
  - [self.update(x, w=None)](#selfupdatex-wnone)
  - [self.batch_update(x, w=None)](#selfbatch_updatex-wnone)
- [Merging TDigests](#merging-tdigests)
  - [self.merge(other)](#selfmergeother)
  - [self.merge_inplace(other)](#selfmerge_inplaceother)
  - [merge_all(digests)](#merge_alldigests)
- [Serialization](#serialization)
  - [self.to_dict()](#selfto_dict)
  - [TDigest.from_dict(tdigest_dict)](#tdigestfrom_dicttdigest_dict)
  - [self.to_bytes()](#selfto_bytes)
  - [TDigest.from_bytes(data)](#tdigestfrom_bytesdata)
- [Comparison](#comparison)
  - [self.equals(other)](#selfequalsother)
- [Other methods and properties](#other-methods-and-properties)
  - [self.copy()](#selfcopy)
  - [self.max_centroids](#selfmax_centroids)
  - [self.mass](#selfmass)
  - [self.n_values](#selfn_values)
  - [self.n_centroids](#selfn_centroids)
  - [self.is_empty](#selfis_empty)
  - [self.centroids](#selfcentroids)
  - [Magic methods / operators](#magic-methods--operators)

### Initialization

#### TDigest()

Creates a new TDigest instance.

```python
from fastdigest import TDigest

digest = TDigest()
print(digest)
```
    TDigest(max_centroids=1000)

> **Note:** The `max_centroids` parameter controls how large the data structure is allowed to grow. A lower value means more compression, enabling a smaller memory footprint and faster computation speed at the cost of some precision.
>
> The default value of 1000 offers a great balance of speed and high precision.
>
> Setting `max_centroids` to 0 disables compression entirely. This will incur a significant performance cost on all operations and is not recommended.

#### TDigest.from_values(x, w=None)

Creates a TDigest directly from any sequence of numeric values `x`. The optional weights `w` can be either a sequence of the same length as `x`, or a scalar that will be used as the weight for the entire batch.

Static method.

```python
import numpy as np
from fastdigest import TDigest

digest = TDigest.from_values([2.71, 3.14, 1.42])  # from list
digest = TDigest.from_values((42,))               # from tuple
digest = TDigest.from_values(range(101))          # from range

digest = TDigest.from_values([1, 2], w=[1, 2])  # weighted individually
digest = TDigest.from_values([1, 2], w=2.0)     # weighted with scalar

data = np.random.random(10_000)
digest = TDigest.from_values(data)  # from NumPy array

print(f"{digest}: {len(digest)} centroids from {digest.n_values} values")
```
    TDigest(max_centroids=1000): 988 centroids from 10000 values

### Mathematical functions

#### self.quantile(q)

Estimates the value at the quantile `q` (between 0 and 1).

Inverse function of [`cdf(x)`](#selfcdfx).

Also available as vectorized [`quantile_vec(q)`](#selfquantile_vecq).

```python
from fastdigest import TDigest
import numpy as np

normally_distributed_data = np.random.normal(0, 1, 10_000)
digest = TDigest.from_values(normally_distributed_data)

print(f"         Median: {digest.quantile(0.5):.3f}")
print(f"99th percentile: {digest.quantile(0.99):.3f}")
```
             Median: 0.001
    99th percentile: 2.274

#### self.percentile(p)

Estimates the value at the `p`th percentile.

Alias for [`quantile(p/100)`](#selfquantileq).

```python
digest = TDigest.from_values(normally_distributed_data)

print(f"         Median: {digest.percentile(50):.3f}")
print(f"99th percentile: {digest.percentile(99):.3f}")
```
             Median: 0.001
    99th percentile: 2.274

#### self.median()

Estimates the median value.

Alias for [`quantile(0.5)`](#selfquantileq).

```python
digest = TDigest.from_values(normally_distributed_data)

print(f"Median: {digest.median():.3f}")
```
    Median: 0.001

#### self.iqr()

Estimates the interquartile range (IQR).

Alias for [`quantile(0.75) - quantile(0.25)`](#selfquantileq).

```python
digest = TDigest.from_values(normally_distributed_data)

print(f"IQR: {digest.iqr():.3f}")
```
    IQR: 1.334

#### self.cdf(x)

Estimates the relative rank (cumulative probability) of the value `x`.

Inverse function of [`quantile(q)`](#selfquantileq).

Also available as vectorized [`cdf_vec(x)`](#selfcdf_vecx).

```python
digest = TDigest.from_values(normally_distributed_data)

print(f"cdf(0.0) = {digest.cdf(0.0):.3f}")
print(f"cdf(1.0) = {digest.cdf(1.0):.3f}")
```
    cdf(0.0) = 0.500
    cdf(1.0) = 0.846

#### self.probability(x1, x2)

Estimates the probability of finding a value in the interval [`x1`, `x2`].

Alias for [`cdf(x2) - cdf(x1)`](#selfcdfx).

```python
digest = TDigest.from_values(normally_distributed_data)
prob = digest.probability(-2.0, 2.0)
prob_pct = 100 * prob

print(f"Probability of value between ±2: {prob_pct:.1f}%")
```
    Probability of value between ±2: 95.4%

#### self.sum()

Returns the sum of all ingested values.

```python
digest = TDigest.from_values(range(11))

print(f"Sum: {digest.sum()}")
```
    Sum: 55.0

#### self.mean()

Returns the arithmetic mean of the distribution.

```python
digest = TDigest.from_values(range(11))

print(f"Mean value: {digest.mean()}")
```
    Mean value: 5.0

#### self.trimmed_mean(q1, q2)

Estimates the truncated mean between the two quantiles `q1` and `q2`.

```python
data = list(range(11))
data[-1] = 100_000  # extreme outlier
digest = TDigest.from_values(data)
mean = digest.mean()
trimmed_mean = digest.trimmed_mean(0.1, 0.9)

print(f"        Mean: {mean}")
print(f"Trimmed mean: {trimmed_mean}")
```
            Mean: 9095.0
    Trimmed mean: 5.0

#### self.min()

Returns the lowest ingested value.

```python
digest = TDigest.from_values(range(-50, 51))

print(f"Minimum: {digest.min():+.1f}")
```
    Minimum: -50.0

#### self.max()

Returns the highest ingested value.

```python
digest = TDigest.from_values(range(-50, 51))

print(f"Maximum: {digest.max():+.1f}")
```
    Maximum: +50.0

#### self.mad()

Estimates the median absolute deviation (MAD) of the distribution.

```python
skewed_data = np.random.standard_gamma(5, 10_000)
digest = TDigest.from_values(skewed_data)

print(f"MAD: {digest.mad():.3f}")
```
    MAD: 1.429

#### self.var()

Estimates the population variance of the distribution.

```python
normally_distributed_data = np.random.normal(0, 1, 10_000)
digest = TDigest.from_values(normally_distributed_data)

print(f"Variance: {digest.var():.3f}")
```
    Variance: 1.010

#### self.std()

Estimates the standard deviation of the distribution.

Alias for [`var() ** 0.5`](#selfvar).

```python
digest = TDigest.from_values(normally_distributed_data)

print(f"Standard deviation: {digest.std():.3f}")
```
    Standard deviation: 1.005

#### self.is_normal()

Performs a Kolmogorov-Smirnov test to determine if the ingested data follows a normal distribution.

```python
normally_distributed_data = np.random.normal(0, 1, 10_000)
normal_digest = TDigest.from_values(normally_distributed_data)

skewed_data = np.random.standard_gamma(5, 10_000)
skewed_digest = TDigest.from_values(skewed_data)

print(normal_digest.is_normal())
print(skewed_digest.is_normal())
```
    True
    False

> **Note:** The significance threshold of the test can be adjusted via the optional argument `alpha` (0.05 by default).

### Vectorized mathematical functions

These methods take a sequence (e.g. list, array) argument and return the results as a list.
They are significantly faster than looping over [`quantile(q)`](#selfquantileq)/[`cdf(x)`](#selfcdfx) when estimating many ($n \gg 1$) values at once.

#### self.quantile_vec(q)

Estimates the values at the quantiles `q` (between 0 and 1).

```python
from fastdigest import TDigest

digest = TDigest.from_values(range(41))
results = digest.quantile_vec([0.25, 0.5, 0.75])
print(results)
```
    [10.0, 20.0, 30.0]

#### self.cdf_vec(x)

Estimates the relative ranks (cumulative probabilities) of the values `x`.

```python
digest = TDigest.from_values(range(41))
results = digest.cdf_vec([10, 20, 30])
print(results)
```
    [0.25, 0.5, 0.75]

### Updating a TDigest

#### self.update(x, w=None)

Updates a digest in-place with a single value `x`, with optional weight `w`.

```python
from fastdigest import TDigest

digest = TDigest.from_values([1, 2, 3, 4, 5, 6])
digest.update(7)
digest.update(42, w=5.0)

print(f"{digest}: {digest.n_values} values, combined weight of {digest.mass}")
```
    TDigest(max_centroids=1000): 8 values, combined weight of 12.0

> **Note:** This writes to a stack-allocated buffer before merging, which is significantly faster than [`batch_update`](#selfbatch_updatex-wnone) for small ad-hoc updates, e.g. in streaming applications.

#### self.batch_update(x, w=None)

Updates a digest in-place by merging a sequence of many values `x` at once. The optional weights `w` can be either a sequence of the same length as `x`, or a scalar that will be used as the weight for the entire batch.

```python
import numpy as np

digest = TDigest()
digest.batch_update([1, 2, 3, 4, 5, 6])
digest.batch_update(np.arange(7, 11))  # using numpy array

digest.batch_update([1, 2], w=[1, 2])  # weighted individually
digest.batch_update([1, 2], w=2.0)     # weighted with scalar

digest.batch_update([5])  # can also just be one value ...
digest.batch_update([])   # ... or empty

print(f"{digest}: {digest.n_values} values, combined weight of {digest.mass}")
```
    TDigest(max_centroids=1000): 15 values, combined weight of 18.0

> **Note:** This directly performs a merge, which is faster than looping over [`update`](#selfupdatex-wnone) if you have the data in advance.

### Merging TDigests

#### self.merge(other)

Creates a new TDigest instance from two digests.

Alias: [`+` operator](#magic-methods--operators)

```python
from fastdigest import TDigest

digest1 = TDigest.from_values(range(50), max_centroids=1000)
digest2 = TDigest.from_values(range(50, 101), max_centroids=3)

merged = digest1 + digest2  # alias for digest1.merge(digest2)

print(f"{merged}: {len(merged)} centroids from {merged.n_values} values")
```
    TDigest(max_centroids=1000): 53 centroids from 101 values

> **Note:** When merging TDigests with different `max_centroids` parameters, the larger value is used for the new instance.

#### self.merge_inplace(other)

Updates a digest in-place with the centroids from an `other` TDigest.

Alias: [`+=` operator](#magic-methods--operators)

```python
digest = TDigest.from_values(range(50), max_centroids=30)
tmp_digest = TDigest.from_values(range(50, 101))

digest += tmp_digest  # alias for: digest.merge_inplace(tmp_digest)

print(f"{digest}: {len(digest)} centroids from {digest.n_values} values")
```
    TDigest(max_centroids=30): 30 centroids from 101 values

> **Note:** Using this method leaves the `max_centroids` parameter of the calling TDigest unchanged.

#### merge_all(digests)

Creates a new TDigest instance from an iterable of `digests` that are efficiently merged in a single operation.

Module-level function.

```python
from fastdigest import merge_all

# create a list of 10 digests from (non-overlapping) ranges
partial_digests = []
for i in range(10):
    partial_data = range(i * 10, (i+1) * 10)
    digest = TDigest.from_values(partial_data, max_centroids=30)
    partial_digests.append(digest)

# merge all digests and create a new instance
merged = merge_all(partial_digests)

print(f"{merged}: {len(merged)} centroids from {merged.n_values} values")
```
    TDigest(max_centroids=30): 30 centroids from 100 values

> **Note:** This function has an optional argument `max_centroids`. If `None` (default), the new instance inherits the largest `max_centroids` parameter of the input digests. Otherwise, the specified value is used.

### Serialization

#### self.to_dict()

Returns a dictionary representation of the TDigest.

```python
import json
from fastdigest import TDigest

digest = TDigest.from_values(range(101), max_centroids=3)
tdigest_dict = digest.to_dict()

print(json.dumps(tdigest_dict, indent=2))
```
```
{
  "max_centroids": 3,
  "mass": 101.0,
  "sum": 5050.0,
  "min": 0.0,
  "max": 100.0,
  "n_values": 101,
  "centroids": [
    {
      "m": 10.5,
      "c": 22.0
    },
    {
      "m": 49.5,
      "c": 56.0
    },
    {
      "m": 89.0,
      "c": 23.0
    }
  ]
}
```

> **Note:** In the "centroids" list, each centroid is represented as a dict with keys "m" (mean) and "c" (count/weight).
The "max_centroids", "mass", "sum", "min", "max" and "n_values" keys are optional — if missing, their values are inferred from the centroids/set to default.
This allows full backward compatibility with dicts created by the *tdigest* Python library.

#### TDigest.from_dict(tdigest_dict)

Creates a new TDigest instance from the `tdigest_dict`.

Static method.

```python
restored = TDigest.from_dict(tdigest_dict)

print(f"{restored}: {len(restored)} centroids from {restored.n_values} values")
```
    TDigest(max_centroids=3): 3 centroids from 101 values

#### self.to_bytes()

Returns a serialized binary representation of the TDigest.

```python
digest = TDigest.from_values(range(101), max_centroids=3)

with open("digest.bin", "wb") as f:
    f.write(digest.to_bytes())
```

> **Note:** This is *much* faster and more efficient than [`to_dict`](#selfto_dict).

#### TDigest.from_bytes(data)

Creates a new TDigest instance from the serialized binary `data`.

Static method.

```python
with open("digest.bin", "rb") as f:
    restored = TDigest.from_bytes(f.read())

print(f"{restored}: {len(restored)} centroids from {restored.n_values} values")
```
    TDigest(max_centroids=3): 3 centroids from 101 values

<br>

> **Note:** You can also use the `pickle` module for serialization. This uses [`to_bytes`](#selfto_bytes)/[`from_bytes`](#tdigestfrom_bytesdata) internally but produces a different format that is not interchangeable with TDigest's native methods.

### Comparison

#### self.equals(other)

Returns `True` if both TDigests have identical centroids, properties and `max_centroids`, otherwise `False`.

Raises `TypeError` if `other` is not a TDigest.

Alternative (without type strictness): [`==`, `!=` operators](#magic-methods--operators)

```python
from fastdigest import TDigest

digest = TDigest.from_values(range(101))
restored = TDigest.from_dict(digest.to_dict())
print(f"digest == restored: {digest.equals(restored)}")
```
    digest == restored: True

### Other methods and properties

#### self.copy()

Returns a copy of the instance.

#### self.max_centroids

Returns the `max_centroids` parameter. Can also be assigned to, changing future behavior of the instance.

#### self.mass

Returns the total ingested weight. Equivalent to [`float(n_values)`](#selfn_values) if no weighted updates were used.

#### self.n_values

Returns the total number of individually ingested values (disregarding weights).

#### self.n_centroids

Returns the number of centroids in the digest.

#### self.is_empty

Returns `True` if no data has been ingested yet.

#### self.centroids

Returns the centroids as a list of (mean, weight) tuples.

#### Magic methods / operators

- `self == other`: alias for [`self.equals(other)`](#selfequalsother) but with `TypeError` suppressed → other types return `False`
- `self != other`: alias for [`not self.equals(other)`](#selfequalsother) but with `TypeError` suppressed → other types return `True`
- `self + other`: alias for [`self.merge(other)`](#selfmergeother)
- `self += other`: alias for [`self.merge_inplace(other)`](#selfmerge_inplaceother)
- `bool(digest)`: alias for [`not digest.is_empty`](#selfis_empty)
- `len(digest)`: alias for [`digest.n_centroids`](#selfn_centroids)
- `iter(digest)`: returns an iterator over [`digest.centroids`](#selfcentroids)
- `copy(digest)`, `deepcopy(digest)`: alias for [`digest.copy()`](#selfcopy)
- `str(digest)`, `repr(digest)`: returns a string representation
