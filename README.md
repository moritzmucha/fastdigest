# fastDigest

[![PyPI](https://img.shields.io/pypi/v/fastdigest.svg)](https://pypi.org/project/fastdigest)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://github.com/moritzmucha/fastdigest/actions/workflows/build.yml/badge.svg)](https://github.com/moritzmucha/fastdigest/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

***fastDigest*** is a Python extension module that provides a lightning-fast implementation of the t‑digest algorithm. Built on top of the efficient *tdigests* Rust library, *fastDigest* enables lightweight and accurate quantile and rank estimation for streaming data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Installing from PyPI](#installing-from-pypi)
  - [Installing from source](#installing-from-source)
- [Usage](#usage)
  - [Creating a TDigest from values](#creating-a-tdigest-from-values)
  - [Compressing the TDigest](#compressing-the-tdigest)
  - [Merging TDigest objects](#merging-tdigest-objects)
  - [Updating a TDigest](#updating-a-tdigest)
  - [Estimating quantiles and ranks](#estimating-quantiles-and-ranks)
  - [Estimating the trimmed mean](#estimating-the-trimmed-mean)
  - [Exporting a TDigest to a dict](#exporting-a-tdigest-to-a-dict)
  - [Restoring a TDigest from a dict](#restoring-a-tdigest-from-a-dict)
- [Benchmarks](#benchmarks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Quantile & rank estimation**: Compute highly accurate quantile and rank estimates from large datasets with a low memory footprint.
- **Trimmed mean**: Calculate the truncated mean in close approximation.
- **Merging digests**: Merge many t‑digests into one, enabling parallel computing workflows such as MapReduce big data processing.
- **Updating**: Use convenience methods to update a t-digest incrementally.
- **Flexible compression**: Decide when and how much t-digests are compressed.
- **Serialization**: Use the `to_dict`/`from_dict` methods (e.g. for JSON conversion) or the `pickle` module for easy serialization.

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
pip install target/wheels/fastdigest-0.3.1-<platform-tag>.whl
```

## Usage

### Creating a TDigest from values

Initialize a TDigest directly from any non‑empty sequence of numbers:

```python
from fastdigest import TDigest

digest = TDigest([1.42, 2.71, 3.14])  # from list
digest = TDigest((42,))               # from tuple
digest = TDigest(range(101))          # from range
```

This can also be a NumPy array:

```python
from fastdigest import TDigest
import numpy as np

array = np.linspace(0, 100, 101)
digest = TDigest(array)
```

### Compressing the TDigest

The TDigest is **uncompressed** after initialization - meaning it has one centroid per data point. Call the `compress(max_centroids)` method to shrink the TDigest object in-place:

```python
from fastdigest import TDigest

digest = TDigest(range(101))
print(f"Before: {len(digest)} centroids")
digest.compress(10)  # compress to 10 (or fewer) centroids
print(f" After: {len(digest)} centroids")
```

### Merging TDigest objects

Use the `+` operator to merge TDigests, combining their data:

```python
from fastdigest import TDigest

digest1 = TDigest(range(50))
digest2 = TDigest(range(50, 101))
merged_digest = digest1 + digest2  # alias for digest1.merge(digest2)
```

You can also merge in-place using the `+=` operator:

```python
from fastdigest import TDigest

digest = TDigest(range(50))
temp_digest = TDigest(range(50, 101))
digest += temp_digest  # alias for digest.merge_inplace(temp_digest)
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

### Estimating quantiles and ranks

Estimate the value at a given quantile `q` using `quantile(q)` or `percentile(100 * q)`:

```python
from fastdigest import TDigest

digest = TDigest(range(1001))
digest.compress(3)
print("         Median:", digest.quantile(0.5))
print("99th percentile:", digest.quantile(0.99))

# same thing, different method:
print("         Median:", digest.percentile(50))
print("99th percentile:", digest.percentile(99))
```

Or do the reverse - find the cumulative probability (rank) of a given value:

```python
from fastdigest import TDigest

digest = TDigest(range(1001))
digest.compress(3)
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
digest.trimmed_mean(0.1, 0.9)  # result: 5.0
```

### Exporting a TDigest to a dict

Obtain a dictionary representation of the digest for serialization or interoperability:

```python
from fastdigest import TDigest
import json

digest = TDigest(range(101))
digest.compress(3)
tdigest_dict = digest.to_dict()
print(json.dumps(tdigest_dict, indent=2))
```

### Restoring a TDigest from a dict

Construct a TDigest from a dict containing a list of centroids. Each centroid is itself a dict with keys "m" (mean) and "c" (weight or count):

```python
from fastdigest import TDigest

data = {
    "centroids": [
        {"m": 0.0, "c": 1.0},
        {"m": 50.0, "c": 99.0},
        {"m": 100.0, "c": 1.0}
    ]
}
digest = TDigest.from_dict(data)
```

**Note:** If you have been working with the older *tdigest* Python library, you may be glad to hear that dicts created by its `to_dict` method can also natively be used by *fastDigest*.

## Benchmarks

Constructing a TDigest and estimating the median of 1,000,000 uniformly distributed random values (average of 10 consecutive runs):

| Library            | Time (ms) | Speedup         |
|--------------------|-----------|-----------------|
| tdigest            | ~12,800   | -               |
| fastdigest         | ~51       | **250x** faster |

*Environment*: Python 3.13.2, Fedora 41 (Workstation), AMD Ryzen 5 7600X

If you want to try it yourself, install *fastDigest* as well as [*tdigest*](https://pypi.org/project/tdigest/) and run:

```bash
python benchmark.py
```

## License

*fastDigest* is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Credit goes to Ted Dunning for inventing [the t-digest algorithm](https://github.com/tdunning/t-digest). Special thanks to Andy Lok for creating the [*tdigests* Rust library](https://github.com/andylokandy/tdigests), as well as all [*PyO3* contributors](https://github.com/pyo3).
