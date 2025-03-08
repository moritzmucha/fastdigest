### Contents

- [Initialization](#initialization)
  - [TDigest()](#tdigest)
  - [TDigest.from_values(values)](#tdigestfrom_valuesvalues)
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
- [Updating a TDigest](#updating-a-tdigest)
  - [self.update(value)](#selfupdatevalue)
  - [self.batch_update(values)](#selfbatch_updatevalues)
- [Merging TDigest objects](#merging-tdigest-objects)
  - [self.merge(other)](#selfmergeother)
  - [self.merge_inplace(other)](#selfmerge_inplaceother)
  - [merge_all(digests)](#merge_alldigests)
- [Dict conversion](#dict-conversion)
  - [self.to_dict()](#selfto_dict)
  - [TDigest.from_dict(tdigest_dict)](#tdigestfrom_dicttdigest_dict)
- [Other methods and properties](#other-methods-and-properties)
  - [self.centroids](#selfcentroids)
  - [self.is_empty](#selfis_empty)
  - [self.max_centroids](#selfmax_centroids)
  - [self.n_centroids](#selfn_centroids)
  - [self.n_values](#selfn_values)
  - [Magic methods](#magic-methods)

### Initialization

#### TDigest()

Create a new TDigest instance by simply calling the class init method.

```python
from fastdigest import TDigest

digest = TDigest()
digest
```
    TDigest(max_centroids=1000)

**Note:** The `max_centroids` parameter controls how large the data structure is allowed to grow. A lower value means more compression, enabling a smaller memory footprint and faster computation speed at the cost of some accuracy. The default value of 1000 offers a great balance.

Setting `max_centroids` to 0 disables compression entirely. This will incur a significant performance cost on all operations and is not recommended.

#### TDigest.from_values(values)

Static method to initialize a TDigest directly from any sequence of numerical values.

```python
import numpy as np

digest = TDigest.from_values([1.42, 2.71, 3.14])  # from list
digest = TDigest.from_values((42,))               # from tuple
digest = TDigest.from_values(range(101))          # from range

data = np.random.random(10_000)
digest = TDigest.from_values(data)  # from NumPy array

print(f"{digest}: {len(digest)} centroids from {digest.n_values} values")
```
    TDigest(max_centroids=1000): 988 centroids from 10000 values

### Mathematical functions

#### self.quantile(q)

Estimate the value at the quantile `q` (between 0 and 1).

This is the inverse function of [cdf(x)](#selfcdfx).

```python
# using a standard normal distribution
digest = TDigest.from_values(np.random.normal(0, 1, 10_000))

print(f"         Median: {digest.quantile(0.5):.3f}")
print(f"99th percentile: {digest.quantile(0.99):.3f}")
```
             Median: 0.001
    99th percentile: 2.274

#### self.percentile(p)

Estimate the value at the `p`th percentile. Alias for `quantile(p/100)`.

```python
print(f"         Median: {digest.percentile(50):.3f}")
print(f"99th percentile: {digest.percentile(99):.3f}")
```
             Median: 0.001
    99th percentile: 2.274

#### self.median()

Estimate the median value. Alias for `quantile(0.5)`.

```python
print(f"Median: {digest.median():.3f}")
```
    Median: 0.001

#### self.iqr()

Estimate the interquartile range (IQR), calculated as the 75th minus the 25th percentile.

```python
print(f"IQR: {digest.iqr():.3f}")
```
    IQR: 1.334

#### self.cdf(x)

Estimate the relative rank (cumulative probability) of the value `x`.

This is the inverse function of [quantile(q)](#selfquantileq).

```python
print(f"cdf(0.0) = {digest.cdf(0.0):.3f}")
print(f"cdf(1.0) = {digest.cdf(1.0):.3f}")
```
    cdf(0.0) = 0.500
    cdf(1.0) = 0.846

#### self.probability(x1, x2)

Estimate the probability of finding a value in the interval [`x1`, `x2`].

```python
prob = digest.probability(-2.0, 2.0)
prob_pct = 100 * prob
print(f"Probability of value between ±2: {prob_pct:.1f}%")
```
    Probability of value between ±2: 95.4%

### self.sum()

Return the sum of all ingested values. This is an exact value (aside from accumulated floating-point error).

```python
data = list(range(11))
digest = TDigest.from_values(data)

print(f"Sum: {digest.sum()}")
```
    Sum: 55.0

### self.mean()

Calculate the arithmetic mean of all ingested values. This is an exact value (aside from accumulated floating-point error).

```python
data = list(range(11))
digest = TDigest.from_values(data)

print(f"Mean value: {digest.mean()}")
```
    Mean value: 5.0

### self.trimmed_mean(q1, q2)

Estimate the truncated mean between the two quantiles `q1` and `q2`.

```python
# inserting an outlier that we want to ignore
data[-1] = 100_000
digest = TDigest.from_values(data)
mean = digest.mean()
trimmed_mean = digest.trimmed_mean(0.1, 0.9)

print(f"        Mean: {mean}")
print(f"Trimmed mean: {trimmed_mean}")
```
            Mean: 9095.0
    Trimmed mean: 5.0

#### self.min()

Return the lowest ingested value. This is an exact value.

```python
print(f"Minimum: {digest.min():+.3f}")
```
    Minimum: -3.545

#### self.max()

Return the highest ingested value. This is an exact value.

```python
print(f"Maximum: {digest.max():+.3f}")
```
    Maximum: +4.615

### Updating a TDigest

#### self.update(value)

Update a digest in-place with a single value.

```python
digest = TDigest.from_values([1, 2, 3, 4, 5, 6])
digest.update(42)

print(f"{digest}: {digest.n_values} values")
```
    TDigest(max_centroids=1000): 7 values

**Note:** This writes to a stack-allocated buffer before merging, which is significantly faster than `batch_update` for rapid iteration with one value (or few values) at a time, e.g. in streaming applications.

#### self.batch_update(values)

Update a digest in-place by merging a sequence of many values at once.

```python
digest = TDigest()
digest.batch_update([1, 2, 3, 4, 5, 6])
digest.batch_update(np.arange(7, 11))  # using numpy array
digest.batch_update([5])  # can also just be one value ...
digest.batch_update([])   # ... or empty

print(f"{digest}: {digest.n_values} values")
```
    TDigest(max_centroids=1000): 11 values

**Note:** This directly performs a merge, which is faster than looping over `update` if you have the data in advance.

### Merging TDigest objects

#### self.merge(other)

Use this method or the `+` operator to create a new TDigest instance from two digests.

```python
digest1 = TDigest.from_values(range(50), max_centroids=1000)
digest2 = TDigest.from_values(range(50, 101), max_centroids=3)

merged = digest1 + digest2  # alias for digest1.merge(digest2)

print(f"{merged}: {len(merged)} centroids from {merged.n_values} values")
```
    TDigest(max_centroids=1000): 53 centroids from 101 values

**Note:** When merging TDigests with different `max_centroids` parameters, the larger value is used for the new instance.

#### self.merge_inplace(other)

Use this method or the `+=` operator to locally update a TDigest with the centroids from an `other`.

```python
digest = TDigest.from_values(range(50), max_centroids=30)
tmp_digest = TDigest.from_values(range(50, 101))

digest += tmp_digest  # alias for: digest.merge_inplace(tmp_digest)

print(f"{digest}: {len(digest)} centroids from {digest.n_values} values")
```
    TDigest(max_centroids=30): 30 centroids from 101 values

**Note:** Using this method leaves the `max_centroids` parameter of the calling TDigest unchanged.

#### merge_all(digests)

Use this function to easily merge a list (or other iterable) of many TDigests.

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

**Note:** This function has an optional `max_centroids` keyword argument. If `None` (default), the `max_centroids` parameter for the new instance is automatically determined as the maximum of the input parameters. Otherwise, the specified value is used instead.

### Dict conversion

#### self.to_dict()

Obtain a dictionary representation of the TDigest.

```python
import json

digest = TDigest.from_values(range(101), max_centroids=3)
tdigest_dict = digest.to_dict()

print(json.dumps(tdigest_dict, indent=2))
```
    {
      "max_centroids": 3,
      "min": 0.0,
      "max": 100.0,
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

**Note:** In the "centroids" list, each centroid is represented as a dict with keys "m" (mean) and "c" (count). The "max_centroids" key is optional. The "min" and "max" keys are not needed when importing legacy dicts from Python *tdigest*, but mandatory for serializing and deserializing digests created by *fastDigest*.

#### TDigest.from_dict(tdigest_dict)

Static method to create a new TDigest instance from the `tdigest_dict`.

```python
digest = TDigest.from_dict(tdigest_dict)

print(f"{digest}: {digest.n_values} values")
```
    TDigest(max_centroids=3): 101 values

### Other methods and properties

#### self.centroids

Returns the centroids as a list of (mean, weight) tuples.

#### self.is_empty

Returns `True` if no data has been ingested yet.

#### self.max_centroids

Returns the `max_centroids` parameter of the instance. Can also be used to change it.

#### self.n_centroids

Returns the number of centroids in the digest.

#### self.n_values

Returns the total number of values ingested.

#### Magic methods

- `digest1 == digest2`: returns `True` if both instances have identical centroids and parameters (within f64 accuracy)
- `self + other`: alias for `self.merge(other)`
- `self += other`: alias for `self.merge_inplace(other)`
- `bool(digest)`: alias for `not digest.is_empty`
- `len(digest)`: alias for `digest.n_centroids`
- `iter(digest)`: returns an iterator over `digest.centroids`
- `repr(digest)`, `str(digest)`: returns a string representation
