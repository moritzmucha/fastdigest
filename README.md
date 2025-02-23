# fastDigest

[![PyPI](https://img.shields.io/pypi/v/fastdigest.svg)](https://pypi.org/project/fastdigest)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://github.com/moritzmucha/fastdigest/actions/workflows/build.yml/badge.svg)](https://github.com/moritzmucha/fastdigest/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

***fastDigest*** is a Python extension module that provides a lightning-fast implementation of the [t-digest algorithm](https://github.com/tdunning/t-digest). Built on a highly optimized Rust backend, *fastDigest* enables lightweight and accurate quantile and rank estimation for streaming data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Installing from PyPI](#installing-from-pypi)
  - [Installing from source](#installing-from-source)
- [Usage](#usage)
  - [Initialization](#initialization)
    - [Initializing a new TDigest](#initializing-a-new-tdigest)
    - [Creating a TDigest from data](#creating-a-tdigest-from-data)
  - [Estimating quantile and rank (CDF)](#estimating-quantile-and-rank-cdf)
  - [Estimating the trimmed mean](#estimating-the-trimmed-mean)
  - [Compressing a TDigest](#compressing-a-tdigest)
  - [Updating a TDigest](#updating-a-tdigest)
  - [Merging TDigest objects](#merging-tdigest-objects)
    - [Merging TDigests into a new instance](#merging-tdigests-into-a-new-instance)
    - [Merging into a TDigest in-place](#merging-into-a-tdigest-in-place)
    - [Merging a list of TDigests](#merging-a-list-of-tdigests)
  - [Dict conversion](#dict-conversion)
    - [Exporting a TDigest to a dict](#exporting-a-tdigest-to-a-dict)
    - [Restoring a TDigest from a dict](#restoring-a-tdigest-from-a-dict)
  - [Migration](#migration)
- [Benchmarks](#benchmarks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Quantile & CDF estimation**: Compute highly accurate quantile and rank (CDF) estimates.
- **Trimmed mean**: Calculate the truncated mean in close approximation.
- **Merging digests**: Merge many t-digests into one, enabling parallel computing and Big Data workflows such as MapReduce.
- **Updating**: Update a t-digest incrementally with streaming data, or batches of a dataset too large to fit in memory otherwise.
- **Serialization**: Use the `to_dict`/`from_dict` methods (e.g. for JSON conversion) or the `pickle` module for easy serialization.
- **Pythonic API**: Use built-in Python operators and functions such as `+` for merging, `==` for checking equality, or `len` to return the number of centroids in a digest.
- **Blazing fast**: Say goodbye to performance headaches — thanks to its Rust backbone, this module is hundreds of times faster than existing Python implementations.

## Installation

### Installing from PyPI

Compiled wheels are available on PyPI. Simply install via pip:

```bash
pip install fastdigest
```

### Installing from source

If you want to build and install *fastDigest* from source, you need **Rust** and **maturin**.

1. Install *maturin* via pip:

```bash
pip install maturin
```

2. Install the Rust toolchain: see https://rustup.rs

3. Build and install the package:

```bash
maturin build --release
pip install target/wheels/fastdigest-0.5.1-<platform-tag>.whl
```

## Usage

### Initialization

#### Initializing a new TDigest

Create an empty new TDigest instance:

```python
from fastdigest import TDigest

digest = TDigest()
```

Specify the `max_centroids` parameter to enable automatic compression:

```python
from fastdigest import TDigest

digest = TDigest(max_centroids=1000)
```

#### Creating a TDigest from data

To initialize a TDigest directly from any sequence of numbers, use `TDigest.from_values`:

```python
import numpy as np
from fastdigest import TDigest

digest = TDigest.from_values([1.42, 2.71, 3.14])        # from list
digest = TDigest.from_values((42,))                     # from tuple
digest = TDigest.from_values(range(101))                # from range
digest = TDigest.from_values(np.linspace(0, 100, 101))  # from numpy array
```

You can also provide the `max_centroids` parameter to `from_values`:

```python
import numpy as np
from fastdigest import TDigest

data = np.random.random(10_000)
digest = TDigest.from_values(data, max_centroids=1000)

print(f"Compressed: {len(digest)} centroids")  # 988 centroids
```

### Estimating quantile and rank (CDF)

Estimate the value at a given quantile `q` using `quantile(q)` or `percentile(100 * q)`:

```python
from fastdigest import TDigest

digest = TDigest.from_values(range(1001), max_centroids=3)
print("         Median:", digest.quantile(0.5))
print("99th percentile:", digest.quantile(0.99))

# same thing, different method:
print("         Median:", digest.percentile(50))
print("99th percentile:", digest.percentile(99))
```

Or do the reverse - find the `cdf` value (cumulative probability or rank) of a given value:

```python
from fastdigest import TDigest

digest = TDigest.from_values(range(1001), max_centroids=3)
print("Rank of 500:", digest.cdf(500))
print("Rank of 990:", digest.cdf(990))
```

### Estimating the trimmed mean

Estimate the truncated mean, i.e. the arithmetic mean of all data points between two quantiles:

```python
from fastdigest import TDigest

values = list(range(10))
values.append(1000)  # outlier that we want to ignore
digest = TDigest.from_values(values)
result = digest.trimmed_mean(0.1, 0.9)

print(f"Trimmed mean: {result}")  # result: 5.0
```

### Compressing a TDigest

If you don't specify the `max_centroids` parameter in a TDigest instance, it will be **uncompressed**; meaning it has one centroid per data point. You can manually call the `compress` method to shrink the object in-place, reducing memory usage while mostly maintaining accuracy:

```python
import numpy as np
from fastdigest import TDigest

# generate a large dataset from a skewed distribution
data = np.random.gumbel(0, 0.1, 10_000)

digest = TDigest.from_values(data)
p99 = digest.quantile(0.99)  # estimate the 99th percentile
print(f"{len(digest):5} centroids: {p99=:.3f}")

digest.compress(1000)  # compress to 1000 or fewer centroids
p99 = digest.quantile(0.99)
print(f"{len(digest):5} centroids: {p99=:.3f}")

digest.compress(100)  # compress to 100 or fewer centroids
p99 = digest.quantile(0.99)
print(f"{len(digest):5} centroids: {p99=:.3f}")
```

### Updating a TDigest

To update an existing TDigest in-place with a sequence of values, use `batch_update`:

```python
import numpy as np
from fastdigest import TDigest

digest = TDigest.from_values([1, 2, 3])
digest.batch_update([4, 5, 6])
digest.batch_update(np.arange(7, 10))  # using numpy array
```

To update with a single value, use `update`:

```python
from fastdigest import TDigest

digest = TDigest.from_values([1, 2, 3])
digest.update(4)
```

**Note:** If you have more than one value to add, it is always preferable to use `batch_update` rather than looping over `update`.

### Merging TDigest objects

#### Merging TDigests into a new instance

Use the `+` operator to merge two digests, creating a new TDigest instance:

```python
from fastdigest import TDigest

digest1 = TDigest.from_values(range(50))
digest2 = TDigest.from_values(range(50, 101))
digest3 = TDigest.from_values(range(101, 201))
merged_digest = digest1 + digest2 + digest3
# alias for:
# merged_digest = digest1.merge(digest2).merge(digest3)
```

**Note:** If `max_centroids` is specified in both instances and the combined `n_centroids` is greater than `max_centroids`, compression will be performed immediately.

When merging two TDigests with different `max_centroids` values, the larger value is used. `None` counts as larger than any other value, since it means no compression.

#### Merging into a TDigest in-place

You can also merge in-place using the `+=` operator:

```python
from fastdigest import TDigest

digest = TDigest.from_values(range(50))
tmp_digest = TDigest.from_values(range(50, 101))
digest += tmp_digest  # alias for: digest.merge_inplace(tmp_digest)

# verify that the result is the same as a new instance from the same data
digest == TDigest.from_values(range(101))  # True
```

**Note:** When using `merge_inplace` or `+=`, the calling TDigest's `max_centroids` parameter remains unchanged.

#### Merging a list of TDigests

The `merge_all` function offers an easy way to merge a sequence of many TDigests. The `max_centroids` value for the new instance can be optionally specified as a keyword argument, otherwise it is determined from the input TDigests.

```python
from fastdigest import TDigest, merge_all

# create a list of 10 digests from (non-overlapping) ranges
digests = [TDigest.from_values(range(i, i+10)) for i in range(0, 100, 10)]

# merge all digests and create a new instance compressed to 3 centroids
merged = merge_all(digests, max_centroids=3)

# verify that the result is the same as a new instance from the same data
merged == TDigest.from_values(range(100), max_centroids=3)  # True
```

### Dict conversion

#### Exporting a TDigest to a dict

Obtain a dictionary representation of the digest by calling `to_dict`:

```python
from fastdigest import TDigest
import json

digest = TDigest.from_values(range(101), max_centroids=3)
tdigest_dict = digest.to_dict()
print(json.dumps(tdigest_dict, indent=2))
```

#### Restoring a TDigest from a dict

Use `TDigest.from_dict(d)` to create a new TDigest instance. The dict has to contain a list of `centroids`, with each centroid itself being a dict with keys `m` (mean) and `c` (weight or count). The `max_centroids` key is optional.

```python
from fastdigest import TDigest

data = {
    "max_centroids": 3,
    "centroids": [
        {"m": 0.0, "c": 1.0},
        {"m": 50.0, "c": 99.0},
        {"m": 100.0, "c": 1.0}
    ]
}
digest = TDigest.from_dict(data)
```

### Migration

The *fastDigest* API is designed to be mostly compatible with the *tdigest* Python library. Migrating is as simple as changing ...

```python
from tdigest import TDigest
```

... to:

```python
from fastdigest import TDigest
```

Dicts created by *tdigest* can also natively be used by *fastDigest*. For functional continuity, set `max_centroids` to 1000 after importing:

```python
from fastdigest import TDigest

imported_digest = TDigest.from_dict(legacy_dict)
imported_digest.max_centroids = 1000
```

## Benchmarks

Constructing a TDigest and estimating the median of 1,000,000 uniformly distributed random values (average of 10 consecutive runs):

| Library            | Time (ms) | Speedup         |
|--------------------|-----------|-----------------|
| tdigest            | ~12,800   | -               |
| fastdigest         | ~51       | **250x** faster |

*Environment*: Python 3.13.2, Fedora 41 (Workstation), AMD Ryzen 5 7600X

If you want to try it yourself, install *fastDigest* as well as [*tdigest*](https://github.com/CamDavidsonPilon/tdigest) and run:

```bash
python benchmark.py
```

## License

*fastDigest* is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Credit goes to Ted Dunning for inventing the [t-digest](https://github.com/tdunning/t-digest). Special thanks to Andy Lok for creating the efficient [*tdigests* Rust library](https://github.com/andylokandy/tdigests), as well as all [*PyO3* contributors](https://github.com/pyo3).
