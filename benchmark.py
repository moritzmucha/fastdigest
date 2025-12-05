import time
import random
from statistics import mean, stdev
from argparse import ArgumentParser
from typing import Sequence, Tuple, Type, TypeVar, Union
from fastdigest import TDigest

# Constants:
MAX_CENTROIDS = 1000          # should be same as fastdigest's default
TIME_UNITS_PER_SECOND = 1000  # granularity of time measurements
TIME_UNIT = "ms"              # for console output

# Run parameter defaults:
P = 50         # percentile to estimate
N = 1_000_000  # size of the dataset
R = 1          # number of benchmark runs

try:
    from tdigest import TDigest as LegacyTDigest
except ImportError:
    LegacyTDigest = None

try:
    from pytdigest import TDigest as PyTDigest
    import numpy as np
except ImportError:
    PyTDigest = None


T = TypeVar("T")
def compute(
    cls: Type[T],
    dataset: Sequence[float],
    incremental: bool = False,
    p: Union[float, int] = P,
) -> Tuple[float, float]:
    start = time.perf_counter()
    digest = cls()
    if incremental:
        for x in dataset:
            digest.update(x)
    else:
        digest.batch_update(dataset)
    result = digest.percentile(p)
    elapsed = TIME_UNITS_PER_SECOND * (time.perf_counter() - start)
    return result, elapsed


def compute_pytd(
    dataset: Sequence[float], incremental: bool = False, p: Union[float, int] = P
) -> Tuple[float, float]:
    dataset = np.array(dataset)
    start = time.perf_counter()
    digest = PyTDigest(MAX_CENTROIDS)
    if incremental:
        for x in dataset:
            digest = digest.compute(x)
    else:
        digest = digest.compute(dataset)
    result = digest.inverse_cdf(p / 100)
    elapsed = TIME_UNITS_PER_SECOND * (time.perf_counter() - start)
    return result, elapsed


def run_benchmark(
    cls: Type[T],
    name: str,
    incremental: bool = False,
    p: Union[float, int] = P,
    n: int = N,
    r: int = R,
    baseline: Union[float, int] = 0,
) -> float:
    result = 0.0
    times = []
    for i in range(r):
        random.seed(i)
        data = [random.uniform(0, 100) for _ in range(n)]
        prog_str = f"running... ({i + 1}/{r})"
        if i == 0:
            line = f"{name:>10}: {prog_str:17}"
        else:
            line = f"{name:>10}: {prog_str:17} | last result: {result:.3f} "
        print("\r" + line, end="", flush=True)
        if cls == PyTDigest:
            result, elapsed = compute_pytd(data, incremental, p)
        else:
            result, elapsed = compute(cls, data, incremental, p)
        times.append(elapsed)
    t_mean = mean(times)
    if r > 1:
        t_std = stdev(times)
        time_str = f"({t_mean:,.0f} ± {t_std:,.0f}) {TIME_UNIT}"
    else:
        time_str = f"{t_mean:,.0f} {TIME_UNIT}"
    speed = baseline / t_mean if baseline >= 0 else 1.0
    if speed == 1.0 or speed >= 10.0:
        speed_str = f" | rel. speed: {speed:>3.0f}x"
    elif speed:
        speed_str = f" | rel. speed: {speed:>3.1f}x"
    else:
        speed_str = ""
    new_line = f"{name:>10}: {time_str:>17}{speed_str}"
    blank_len = max(0, len(line) - len(new_line))
    blank_str = " " * blank_len
    print("\r" + new_line + blank_str, flush=True)
    return t_mean


def main():
    parser = ArgumentParser(
        description=(
            "Benchmark fastdigest against other t-digest libraries "
            "(tdigest, pytdigest)."
        )
    )
    parser.add_argument(
        "-i",
        "--incremental",
        action="store_true",
        help="merge one value at a time, using update() instead of batch_update()",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        type=float,
        default=float(P),
        help=f"percentile to estimate (default: {P})",
    )
    parser.add_argument(
        "-n",
        "--n-values",
        type=int,
        default=N,
        help=f"size of the dataset (default: {N:_})",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=R,
        help=f"number of benchmark runs (default: {R:_})",
    )
    args = parser.parse_args()
    i = args.incremental
    n = args.n_values
    p = args.percentile
    r = args.repeat

    if not 0 <= p <= 100:
        print("p must be between 0 and 100.")
        return
    if n < 1:
        print("n must be at least 1.")
        return
    if r < 1:
        print("r must be at least 1.")
        return

    print()
    baseline = -1
    for cls, lib in ((LegacyTDigest, "tdigest"), (PyTDigest, "pytdigest")):
        if cls is None:
            continue
        t = run_benchmark(cls, lib, i, p, n, r, baseline)
        if baseline == -1:
            baseline = t
    run_benchmark(TDigest, "fastdigest", i, p, n, r, max(baseline, 0))
    print()


if __name__ == "__main__":
    main()
