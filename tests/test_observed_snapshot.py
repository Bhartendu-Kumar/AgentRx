"""Observed model snapshot (W1c) — what the LLM server actually served.

These tests exercise the accumulator in isolation. The actual LLMAgent
wiring is not tested here (it would require a live or mocked Azure/TRAPI
client); the round-trip through provenance is covered indirectly via
agentrx.pipeline.provenance tests + the wiring in run.py finally block.
"""
from __future__ import annotations

import threading

import pytest

from agentrx.llm_clients import observed_snapshot


@pytest.fixture(autouse=True)
def _clean_snapshot():
    observed_snapshot.reset()
    yield
    observed_snapshot.reset()


def test_empty_snapshot_returns_none():
    assert observed_snapshot.snapshot() is None


def test_single_record_populates_snapshot():
    observed_snapshot.record(served_model="gpt-5-2025-04-16", system_fingerprint="fp_abc")
    snap = observed_snapshot.snapshot()
    assert snap is not None
    assert snap["total_calls"] == 1
    assert snap["served_models"] == {"gpt-5-2025-04-16": 1}
    assert snap["system_fingerprints"] == {"fp_abc": 1}
    assert snap["first_seen_utc"] is not None
    assert snap["last_seen_utc"] is not None


def test_repeated_records_aggregate_counts():
    for _ in range(3):
        observed_snapshot.record(served_model="gpt-5", system_fingerprint="fp_A")
    for _ in range(2):
        observed_snapshot.record(served_model="gpt-5", system_fingerprint="fp_B")
    snap = observed_snapshot.snapshot()
    assert snap["total_calls"] == 5
    assert snap["served_models"] == {"gpt-5": 5}
    assert snap["system_fingerprints"] == {"fp_A": 3, "fp_B": 2}


def test_none_values_recorded_as_string_null():
    observed_snapshot.record(served_model=None, system_fingerprint=None)
    snap = observed_snapshot.snapshot()
    assert snap["served_models"] == {"null": 1}
    assert snap["system_fingerprints"] == {"null": 1}


def test_model_drift_visible_in_snapshot():
    """The whole point: catch silent server-side model upgrades."""
    observed_snapshot.record(served_model="gpt-5-2025-04-16", system_fingerprint="fp_old")
    observed_snapshot.record(served_model="gpt-5-2025-08-07", system_fingerprint="fp_new")
    snap = observed_snapshot.snapshot()
    assert snap["served_models"] == {"gpt-5-2025-04-16": 1, "gpt-5-2025-08-07": 1}
    assert len(snap["system_fingerprints"]) == 2


def test_record_never_raises_on_garbage_input():
    class Boom:
        def __str__(self):
            raise RuntimeError("kaboom")
    # Must not propagate.
    observed_snapshot.record(served_model=Boom(), system_fingerprint=Boom())
    # And must not have advanced the counter, since the keys couldn't be made.
    snap = observed_snapshot.snapshot()
    assert snap is None


def test_thread_safety_of_record():
    """N threads * K calls each must produce exactly N*K total_calls."""
    N_THREADS = 8
    K_CALLS = 50

    def worker():
        for i in range(K_CALLS):
            observed_snapshot.record(
                served_model=f"thread-model-{i % 3}",
                system_fingerprint=f"fp-{i % 4}",
            )

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = observed_snapshot.snapshot()
    assert snap["total_calls"] == N_THREADS * K_CALLS
    assert sum(snap["served_models"].values()) == N_THREADS * K_CALLS
    assert sum(snap["system_fingerprints"].values()) == N_THREADS * K_CALLS


def test_reset_clears_state():
    observed_snapshot.record(served_model="x", system_fingerprint="y")
    assert observed_snapshot.snapshot() is not None
    observed_snapshot.reset()
    assert observed_snapshot.snapshot() is None
