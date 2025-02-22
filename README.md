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
  - [Creating a TDigest from values](#creating-a-tdigest-from-values)
  - [Estimating quantiles and ranks](#estimating-quantiles-and-ranks)
  - [Estimating the trimmed mean](#estimating-the-trimmed-mean)
  - [Compressing the TDigest](#compressing-the-tdigest)
  - [Merging TDigest objects](#merging-tdigest-objects)
  - [Updating a TDigest](#updating-a-tdigest)
  - [Exporting a TDigest to a dict](#exporting-a-tdigest-to-a-dict)
  - [Restoring a TDigest from a dict](#restoring-a-tdigest-from-a-dict)
- [Benchmarks](#benchmarks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Quantile & rank estimation**: Compute highly accurate quantile and rank estimates.
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

### Creating a TDigest from values

Initialize a TDigest directly from any non-empty sequence of numbers:

```python
import numpy as np
from fastdigest import TDigest

digest = TDigest([1.42, 2.71, 3.14])        # from list
digest = TDigest((42,))                     # from tuple
digest = TDigest(range(101))                # from range
digest = TDigest(np.linspace(0, 100, 101))  # from numpy array
```

Specify the `max_centroids` parameter to enable automatic compression:

```python
import numpy as np
from fastdigest import TDigest

data = np.random.random(10_000)
digest = TDigest(data, max_centroids=1000)

print(f"Compressed: {len(digest)} centroids")  # 988 centroids
```

### Estimating quantiles and ranks

Estimate the value at a given quantile `q` using `quantile(q)` or `percentile(100 * q)`:

```python
from fastdigest import TDigest

digest = TDigest(range(1001), max_centroids=3)
print("         Median:", digest.quantile(0.5))
print("99th percentile:", digest.quantile(0.99))

# same thing, different method:
print("         Median:", digest.percentile(50))
print("99th percentile:", digest.percentile(99))
```

Or do the reverse - find the cumulative probability (`rank`) of a given value:

```python
from fastdigest import TDigest

digest = TDigest(range(1001), max_centroids=3)
print("Rank of 500:", digest.rank(500))
print("Rank of 990:", digest.rank(990))
```

### Estimating the trimmed mean

Estimate the truncated mean, i.e. the arithmetic mean of all data points between two quantiles:

```python
from fastdigest import TDigest

values = list(range(10))
values.append(1000)  # outlier that we want to ignore
digest = TDigest(values)
result = digest.trimmed_mean(0.1, 0.9)

print(f"Trimmed mean: {result}")  # result: 5.0
```

### Compressing the TDigest

If you don't specify the `max_centroids` parameter at initialization, the TDigest object will be **uncompressed**; meaning it has one centroid per data point. You can manually call the `compress` method to shrink the object in-place, reducing memory usage while mostly maintaining accuracy:

```python
import numpy as np
from fastdigest import TDigest

# generate a large dataset from a skewed distribution
data = np.random.gumbel(0, 0.1, 10_000)

digest = TDigest(data)
p99 = digest.quantile(0.99)  # estimate the 99th percentile
print(f"{len(digest):5} centroids: {p99=:.3f}")

digest.compress(1000)  # compress to 1000 or fewer centroids
p99 = digest.quantile(0.99)
print(f"{len(digest):5} centroids: {p99=:.3f}")

digest.compress(100)  # compress to 100 or fewer centroids
p99 = digest.quantile(0.99)
print(f"{len(digest):5} centroids: {p99=:.3f}")
```

### Merging TDigest objects

#### Merging TDigests into a new instance

Use the `+` operator to merge two digests, creating a new TDigest instance:

```python
from fastdigest import TDigest

digest1 = TDigest(range(50))
digest2 = TDigest(range(50, 101))
merged_digest = digest1 + digest2  # alias for digest1.merge(digest2)
```

**Note:** If `max_centroids` is specified in both instances and the combined `n_centroids` is greater than `max_centroids`, compression will be performed immediately.

When merging two TDigests with different `max_centroids` values, the larger value is used. `None` counts as larger than any other value, since it means no compression.

#### Merging into a TDigest in-place

You can also merge in-place using the `+=` operator:

```python
from fastdigest import TDigest

digest = TDigest(range(50))
temp_digest = TDigest(range(50, 101))
digest += temp_digest  # alias for digest.merge_inplace(temp_digest)

# verify that the result is the same as a new instance from the same data
digest == TDigest(range(101))  # True
```

**Note:** When using `merge_inplace` or `+=`, the calling TDigest's `max_centroids` parameter always remains unchanged.

This means you can effectively chain-merge many uncompressed digests and perform a single compression step at the end by combining `+=` and `+`:

```python
from fastdigest import TDigest

digest = TDigest(range(101), max_centroids=3)
tmp_digest1 = TDigest(range(101, 201))
tmp_digest2 = TDigest(range(201, 301))
tmp_digest3 = TDigest(range(301, 401))
digest += tmp_digest1 + tmp_digest2 + tmp_digest3

print(f"Result: {len(digest)} centroids")  # 3 centroids
```

#### Merging a list of TDigests

The `merge_all` function offers an easy way to merge a sequence of many TDigests. The `max_centroids` value for the new instance can be optionally specified as a keyword argument, otherwise it is determined from the input TDigests.

```python
from fastdigest import TDigest, merge_all

# create a list of 10 digests from (non-overlapping) ranges
digests = [TDigest(range(i, i+10)) for i in range(0, 100, 10)]

# merge all digests and create a new instance compressed to 3 centroids
merged = merge_all(digests, max_centroids=3)

# verify that the result is the same as a new instance from the same data
merged == TDigest(range(100), max_centroids=3)  # True
```

### Updating a TDigest

To update an existing TDigest in-place with a new sequence/array of values, use `batch_update`:

```python
from fastdigest import TDigest

digest = TDigest([1, 2, 3])
digest.batch_update([4, 5, 6])
```

To update with a single value, use `update`:

```python
from fastdigest import TDigest

digest = TDigest([1, 2, 3])
digest.update(4)
```

**Note:** If you have more than one value to add, it is always preferable to use `batch_update` rather than looping over `update`.

### Exporting a TDigest to a dict

Obtain a dictionary representation of the digest by calling `to_dict`:

```python
from fastdigest import TDigest
import json

digest = TDigest(range(101), max_centroids=3)
tdigest_dict = digest.to_dict()
print(json.dumps(tdigest_dict, indent=2))
```

### Restoring a TDigest from a dict

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

**Note:** dicts created by the *tdigest* Python library can also natively be used by *fastDigest*. For functional continuity, set `max_centroids` to 1000 after importing:

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
