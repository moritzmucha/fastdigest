import time
import random
import argparse
from fastdigest import TDigest
from statistics import mean, stdev
from typing import Callable, Sequence, Tuple

N = 1_000_000  # default size of dataset
Q = 0.5        # default quantile to estimate
R = 1          # default number of benchmark runs

try:
    from tdigest import TDigest as LegacyTDigest
except ImportError:
    LegacyTDigest = None
    print(
        "Warning: Legacy 'tdigest' library not found. "
        "Install it to run the full benchmark.\n"
    )

def benchmark_fastdigest(
        dataset: Sequence[float],
        q: float,
        max_centroids: int = 1000
    ) -> Tuple[float, float]:
    start = time.perf_counter()
    digest = TDigest(dataset)
    digest.compress(max_centroids)
    result = digest.quantile(q)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    return result, elapsed_ms

def benchmark_tdigest(
        dataset: Sequence[float],
        q: float
    ) -> Tuple[float, float]:
    start = time.perf_counter()
    digest = LegacyTDigest()
    digest.batch_update(dataset)
    result = digest.percentile(100. * q)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    return result, elapsed_ms

def run_benchmark(
        benchmark: Callable[[Sequence[float], float], Tuple[float, float]],
        name: str,
        n: int = N,
        q: float = Q,
        r: int = R
    ) -> float:
    times = []
    for i in range(r):
        random.seed(i)
        data = [random.uniform(0, 100) for _ in range(n)]
        progress_str = f"running... ({i+1}/{r})"
        if i == 0:
            print(f"\r{name:>14}: {progress_str:17}", end="", flush=True)
        else:
            print(
                f"\r{name:>14}: {progress_str:17} | last result: {result:.3f}",
                end="",
                flush=True
            )
        result, elapsed_ms = benchmark(data, q)
        times.append(elapsed_ms)
    t_mean = mean(times)
    if r > 1:
        t_std = stdev(times)
        time_str = f"({t_mean:,.0f} ± {t_std:,.0f}) ms"
    else:
        time_str = f"{t_mean:,.0f} ms"
    blank_str = " " * (max(len(progress_str), 17) - max(len(time_str), 17))
    print(
        f"\r{name:>14}: {time_str:17} | last result: {result:.3f}",
        end = blank_str + "\n",
        flush = True
    )
    return t_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark fastDigest against the older tdigest library."
    )
    parser.add_argument(
        "-n", "--n-values",
        type = int,
        default = N,
        help = f"size of the dataset (default: {N:_})"
    )
    parser.add_argument(
        "-q", "--quantile",
        type = float,
        default = Q,
        help = f"quantile to estimate (default: {Q})"
    )
    parser.add_argument(
        "-r", "--repeat",
        type = int,
        default = R,
        help = f"number of benchmark runs (default: {R:_})"
    )
    args = parser.parse_args()
    n = args.n_values
    q = args.quantile
    r = args.repeat

    if LegacyTDigest is not None:
        t_legacy = run_benchmark(benchmark_tdigest, "Legacy tdigest", n, q, r)

    t_fast = run_benchmark(benchmark_fastdigest, "fastDigest", n, q, r)

    if LegacyTDigest is not None:
        print(f"{'Speedup':>14}: {t_legacy / t_fast:.0f}x")
