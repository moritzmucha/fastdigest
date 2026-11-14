import pytest
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Thread
from fastdigest import TDigest, merge_all


def run_workers(*workers):
    barrier = Barrier(len(workers))

    def run(worker):
        barrier.wait()
        return worker()

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        return list(pool.map(run, workers))


def independent_digests():
    def summary(digest):
        return (
            digest.n_values,
            digest.mass(),
            digest.sum(),
            digest.min(),
            digest.max(),
        )

    def calculate():
        values = [float(i % 137) for i in range(10_000)]
        weights = [1.0 + i % 3 for i in range(len(values))]
        first = TDigest.from_values(values, weights)
        second = TDigest.from_values(values)
        # Leave weighted observations in both online caches before merging.
        first.update(-1.0, 2.0)
        second.update(150.0, 3.0)
        first.batch_update(values, 2.0)
        batch_bytes = first.to_bytes()
        merged = first.merge(second)
        combined = merge_all(d for d in (first, second, first))
        first.merge_inplace(second)
        # Merging tied means in a different order can change rounding.
        assert summary(first) == pytest.approx(summary(merged))
        assert first + second == first.merge(second)
        first += second
        return batch_bytes, summary(first), summary(combined)

    expected = calculate()
    for actual in run_workers(*([calculate] * 4)):
        assert actual[0] == expected[0]
        assert actual[1] == pytest.approx(expected[1])
        assert actual[2] == pytest.approx(expected[2])


def shared_updates():
    digest = TDigest()
    digest.update(1.0, 2.0)
    source = TDigest()
    for _ in range(7):
        source.update(1.0, 3.0)

    rounds = 100
    batch = [1.0] * 300

    def update():
        for _ in range(rounds):
            digest.update(1.0, 2.0)

    def batch_update():
        for _ in range(rounds):
            digest.batch_update(batch, 2.0)

    def merge_inplace():
        for _ in range(rounds):
            digest.merge_inplace(source)

    def read():
        for _ in range(rounds):
            assert digest.n_values > 0
            assert digest.quantile(0.5) == 1.0
            assert digest.copy().mean() == 1.0
            assert TDigest.from_dict(digest.to_dict()).mean() == 1.0
            assert TDigest.from_bytes(digest.to_bytes()).mean() == 1.0
            assert digest.merge(source).mean() == 1.0
            assert merge_all([digest, source, digest]).mean() == 1.0
            digest.equals(source)

    run_workers(update, batch_update, batch_update, merge_inplace, read, read)
    assert digest.n_values == 1 + rounds * (1 + 2 * len(batch) + 7)
    expected_mass = 2 + rounds * (2 + 4 * len(batch) + 21)
    assert digest.mass() == expected_mass
    assert digest.sum() == expected_mass
    assert source.n_values == 7
    assert source.mass() == 21


def opposing_merges():
    left, right = TDigest(), TDigest()
    left.update(1.0)
    right.update(1.0)

    def merge_into(target, source):
        for _ in range(12):
            target.merge_inplace(source)
            assert target.merge(source).mean() == 1.0
            target.equals(source)

    run_workers(
        lambda: merge_into(left, right), lambda: merge_into(right, left)
    )
    for digest in (left, right):
        assert digest.n_values > 1
        assert digest.mass() == digest.n_values
        assert digest.sum() == digest.n_values


def self_merges():
    digest = TDigest()
    digest.update(1.0, 2.0)

    def merge_self():
        for _ in range(4):
            digest.merge_inplace(digest)

    run_workers(*([merge_self] * 4))
    assert digest.n_values == 2**16
    assert digest.mass() == 2**17
    assert digest.sum() == 2**17
    assert digest.merge(digest).n_values == 2**17
    digest += digest
    assert digest.n_values == 2**17


def gil_progress(method):
    values = [float(i % 7919) for i in range(100_000)]
    digest = TDigest.from_values(values, max_centroids=20_000)
    other = digest.copy()
    operations = {
        "from_values": lambda: TDigest.from_values(values, 2.0),
        "batch_update": lambda: digest.batch_update(values, 2.0),
        "merge": lambda: digest.merge(other),
        "merge_inplace": lambda: digest.merge_inplace(other),
        "merge_all": lambda: merge_all([digest, other] * 8),
    }
    operation = operations[method]
    ready, start, progressed = Event(), Event(), Event()

    def observer():
        ready.set()
        start.wait()
        progressed.set()

    observer_thread = Thread(target=observer)
    observer_thread.start()
    ready.wait()
    previous_interval = sys.getswitchinterval()
    try:
        # Prevent normal bytecode scheduling from satisfying this check.
        # Only the extension calls below should give the observer the GIL.
        sys.setswitchinterval(1000.0)
        start.set()
        for _ in range(20):
            operation()
            if progressed.is_set():
                break
        assert progressed.is_set(), f"{method} did not release the GIL"
    finally:
        sys.setswitchinterval(previous_interval)
        observer_thread.join()


def run_in_subprocess(*args):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "scenario",
    ["independent_digests", "shared_updates", "opposing_merges", "self_merges"],
)
def test_concurrent_digests(scenario):
    run_in_subprocess(scenario)


@pytest.mark.parametrize(
    "method",
    ["from_values", "batch_update", "merge", "merge_inplace", "merge_all"],
)
def test_releases_gil(method):
    if sys.implementation.name != "cpython":
        pytest.skip("GIL scheduling check requires CPython")
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        pytest.skip("No GIL to release on this interpreter")
    run_in_subprocess("gil_progress", method)


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
