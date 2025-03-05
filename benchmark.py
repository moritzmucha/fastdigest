import time
import random
import argparse
from fastdigest import TDigest
from statistics import mean, stdev
from typing import Sequence, Tuple, Type, TypeVar, Union

# Default values:
P = 50         # percentile to estimate
N = 1_000_000  # size of the dataset
R = 1          # number of benchmark runs

try:
    from tdigest import TDigest as LegacyTDigest
except ImportError:
    LegacyTDigest = None
    print(
        "Warning: Legacy 'tdigest' library not found. "
        "Install it to run the full benchmark.\n"
    )

T = TypeVar("T")
def compute(
        cls: Type[T],
        dataset: Sequence[float],
        incremental: bool = False,
        p: Union[float, int] = P
    ) -> Tuple[float, float]:
    start = time.perf_counter()
    digest = cls()
    if incremental:
        for x in dataset:
            digest.update(x)
    else:
        digest.batch_update(dataset)
    result = digest.percentile(p)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    return result, elapsed_ms

def run_benchmark(
        cls: Type[T],
        name: str,
        incremental: bool = False,
        p: Union[float, int] = P,
        n: int = N,
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
        result, elapsed_ms = compute(cls, data, incremental, p)
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

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fastDigest against the older tdigest library."
    )
    parser.add_argument(
        "-i", "--incremental",
        action = "store_true",
        help = "use update() instead of batch_update()"
    )
    parser.add_argument(
        "-p", "--percentile",
        type = float,
        default = float(P),
        help = f"percentile to estimate (default: {P})"
    )
    parser.add_argument(
        "-n", "--n-values",
        type = int,
        default = N,
        help = f"size of the dataset (default: {N:_})"
    )
    parser.add_argument(
        "-r", "--repeat",
        type = int,
        default = R,
        help = f"number of benchmark runs (default: {R:_})"
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

    if LegacyTDigest is not None:
        t_legacy = run_benchmark(LegacyTDigest, "Legacy tdigest", i, p, n, r)

    t_fast = run_benchmark(TDigest, "fastDigest", i, p, n, r)

    if LegacyTDigest is not None:
        print(f"{'Speedup':>14}: {t_legacy / t_fast:.0f}x")

if __name__ == '__main__':
    main()
