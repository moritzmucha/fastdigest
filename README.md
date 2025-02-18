# fastDigest

[![PyPI](https://img.shields.io/pypi/v/fastdigest.svg)](https://pypi.org/project/fastdigest)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://github.com/moritzmucha/fastdigest/actions/workflows/build.yml/badge.svg)](https://github.com/moritzmucha/fastdigest/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

***fastDigest*** is a Python extension module that provides a lightning-fast implementation of the t‑digest algorithm using Rust and PyO3. Built on top of the efficient *tdigests* library, *fastDigest* enables lightweight and accurate quantile and rank estimation for streaming data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating a TDigest from Values](#creating-a-tdigest-from-values)
  - [Merging Two TDigest Objects](#merging-two-tdigest-objects)
  - [Compressing the TDigest](#compressing-the-tdigest)
  - [Estimating Quantiles and Ranks](#estimating-quantiles-and-ranks)
  - [Estimating the Trimmed Mean](#estimating-the-trimmed-mean)
  - [Exporting a TDigest to a Dict](#exporting-a-tdigest-to-a-dict)
  - [Restoring a TDigest from a Dict](#restoring-a-tdigest-from-a-dict)
- [Benchmarks](#benchmarks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Quantile & Rank Estimation**: Compute highly accurate quantile and rank estimates from large datasets with a low memory footprint.
- **Trimmed Mean**: Calculate the truncated mean in close approximation.
- **Merging Digests**: Merge two t‑digests into one. This can be used to handle streaming data.
- **Flexible Compression**: Decide when and how much t-digests are compressed.
- **Serialization**: Use `to_dict`/`from_dict` methods (e.g. for JSON conversion) or the `pickle` module for easy serialization.

## Installation

Compiled wheels are available on PyPI. Simply install via pip:

```bash
pip install fastdigest
```

If you want to build and install *fastDigest* from source, you need **Rust** and **maturin**.

1. Install *maturin* via pip:

```bash
pip install maturin
```

2. Install the Rust toolchain: see https://rustup.rs

3. Build and install the package:

```bash
maturin build --release
pip install target/wheels/fastdigest-0.3.0-<platform-tag>.whl
```

## Usage

### Creating a TDigest from Values

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

### Merging Two TDigest Objects

Merge two digests to combine their data:

```python
from fastdigest import TDigest

digest1 = TDigest(range(50))
digest2 = TDigest(range(50, 101))
merged_digest = digest1.merge(digest2)
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

### Estimating Quantiles and Ranks

Estimate the value at a given quantile:

```python
from fastdigest import TDigest

# Simple example
digest = TDigest(range(101))
digest.compress(3)
print("         Median:", digest.estimate_quantile(0.5))
print("90th percentile:", digest.estimate_quantile(0.9))
```

```python
from fastdigest import TDigest
import numpy as np

# Example using a standard normal distribution
normal_dist = np.random.normal(0, 1, 10_000)
digest = TDigest(normal_dist)
digest.compress(100)
z_score = digest.estimate_quantile((1 + 0.954) / 2)  # using symmetry
print(f"Standard deviations at 95.4% confidence: {z_score:.2f}")
```

Or the reverse - the cumulative probability (rank) of a given value:

```python
# Continuing from normal distribution example
confidence = 2 * digest.estimate_rank(2.0) - 1  # reverse order of operations
confidence_pct = 100 * confidence
print(f"Confidence at 2.0 standard deviations: {confidence_pct:.2f}%")
```

### Estimating the Trimmed Mean

Estimate the truncated mean, i.e. the arithmetic mean of all data points between two quantiles:

```python
from fastdigest import TDigest

values = list(range(10))
values.append(1000)
digest = TDigest(values)
digest.trimmed_mean(0.1, 0.9)
```

### Exporting a TDigest to a Dict

Obtain a dictionary representation of the digest for serialization or interoperability:

```python
from fastdigest import TDigest
import json

digest = TDigest(range(101))
digest.compress(3)
tdigest_dict = digest.to_dict()
print(json.dumps(tdigest_dict, indent=2))
```

### Restoring a TDigest from a Dict

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
