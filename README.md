# fastDigest

[![PyPI](https://img.shields.io/pypi/v/fastdigest.svg)](https://pypi.org/project/fastdigest/)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://github.com/moritzmucha/fastdigest/actions/workflows/build.yml/badge.svg)](https://github.com/moritzmucha/fastdigest/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

***fastDigest*** is a Rust-powered Python extension module that provides a lightning-fast implementation of the [t-digest data structure and algorithm](https://github.com/tdunning/t-digest), offering a lightweight suite of online statistics for streaming and distributed data.

## Contents

- [Features](#features)
- [Installation](#installation)
  - [Installing from PyPI](#installing-from-pypi)
  - [Installing from source](#installing-from-source)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Mathematical functions](#mathematical-functions)
  - [Updating a TDigest](#updating-a-tdigest)
  - [Merging TDigest objects](#merging-tdigest-objects)
  - [Dict conversion](#dict-conversion)
  - [Migration](#migration)
- [Benchmarks](#benchmarks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Online statistics**: Compute highly accurate estimates of quantiles, the CDF, and many derived quantities such as the (trimmed) mean.
- **Updating**: Update a t-digest incrementally with streaming data or batches of large datasets.
- **Merging**: Merge many t-digests into one, enabling parallel compute operations such as map-reduce.
- **Serialization**: Use the `to_dict`/`from_dict` methods or the `pickle` module for serialization.
- **Easy API**: The *fastDigest* API is designed to be intuitive and to keep high overlap with popular libraries.
- **Blazing fast**: Thanks to its Rust backbone, this module is hundreds of times faster than other Python implementations.

## Installation

### Installing from PyPI

Compiled wheels are available on PyPI. Simply install via pip:

```bash
pip install fastdigest
```

### Installing from source

To build and install *fastDigest* from source, you will need Rust and *maturin*.

1. Install the Rust toolchain &rarr; see https://rustup.rs

2. Install *maturin* via pip:

```bash
pip install maturin
```

3. Build and install the package:

```bash
maturin build --release
pip install target/wheels/fastdigest-0.8.3-<platform-tag>.whl
```

## Usage

The following examples give you a quick start. See the [API reference](https://github.com/moritzmucha/fastdigest/blob/main/API.md) for the full documentation.

### Initialization

Simply call `TDigest()`, or use `TDigest.from_values` to create a digest directly from any sequence of numeric values:

```python
from fastdigest import TDigest

digest = TDigest()
digest = TDigest.from_values([1.42, 2.71, 3.14])
```

### Mathematical functions

Estimate the value at the rank `q` using `quantile(q)`:

```python
digest = TDigest.from_values(range(101))
print("99th percentile:", digest.quantile(0.99))
```

Or the inverse - use `cdf` to find the rank (cumulative probability) of a given value:

```python
print("cdf(990) =", digest.cdf(990))
```

Compute the arithmetic `mean`, or the `trimmed_mean` between two quantiles:

```python
data = list(range(101))
data[-1] = 100_000  # inserting an outlier
digest = TDigest.from_values(data)
print(f"        Mean: {digest.mean():.1f}")
print(f"Trimmed mean: {digest.trimmed_mean(0.1, 0.9)}")
```

### Updating a TDigest

Use `batch_update` to merge a sequence of many values at once, or `update` to add one value at a time:

```python
digest = TDigest()
digest.batch_update([0, 1, 2])
digest.update(3)
```

Note that there can be significant performance differences between these methods depending on use-case.

### Merging TDigest objects

Use the `+` operator to create a new instance from two TDigests, or `+=` to merge in-place:

```python
digest1 = TDigest.from_values(range(20))
digest2 = TDigest.from_values(range(20, 51))
digest3 = TDigest.from_values(range(51, 101))

digest1 += digest2
merged_new = digest1 + digest3
```

The `merge_all` function offers an easy way to merge an iterable of many TDigests:

```python
from fastdigest import TDigest, merge_all

digests = [TDigest.from_values(range(i, i+10)) for i in range(0, 100, 10)]
merged = merge_all(digests)
```

### Dict conversion

Obtain a dictionary representation by calling `to_dict()` and load it into a new instance with `TDigest.from_dict`:

```python
from fastdigest import TDigest
import json

digest = TDigest.from_values(range(101))
td_dict = digest.to_dict()
print(json.dumps(td_dict, indent=2))
restored = TDigest.from_dict(td_dict)
```

### Migration

The *fastDigest* API is designed to be backward compatible with the *tdigest* Python library. Migrating is as simple as changing your `import` statement.

Dicts created by *tdigest* can also natively be used by *fastDigest*.

## Benchmarks

Constructing a TDigest and estimating the median of 1,000,000 uniformly distributed random values (average of 10 consecutive runs):

| Library            | Time (ms) | Speedup         |
|--------------------|-----------|-----------------|
| tdigest            | ~12,800   | -               |
| fastdigest         | ~32       | **400x** faster |

*Environment*: Python 3.13.2, Fedora 41 (Workstation), AMD Ryzen 5 7600X

If you want to try it yourself, install *fastDigest* as well as [*tdigest*](https://github.com/CamDavidsonPilon/tdigest) and run:

```bash
python benchmark.py
```

## License

*fastDigest* is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Credit goes to Ted Dunning for inventing the [t-digest](https://github.com/tdunning/t-digest). Special thanks to Andy Lok and Paul Meng for creating the [*tdigests*](https://github.com/andylokandy/tdigests) and [*tdigest*](https://github.com/MnO2/t-digest) Rust libraries, respectively, as well as to all [*PyO3* contributors](https://github.com/pyo3).
