"""Cross-stage accumulator: what the LLM server actually served.

The configured model_spec in ``run_config.json`` says what we *asked for*.
This module records what the server *gave us* per call (``response.model``
and ``response.system_fingerprint``), aggregates across the entire run, and
exposes a snapshot for ``run.py`` to merge into ``run_config.json``.

Why this matters
----------------
Azure OpenAI silently routes ``gpt-5`` to whatever underlying version the
deployment is currently pinned to, and the system fingerprint changes when
the served binary changes. A reproduction that gets a different number on
the same prompt-mode/exec-mode/with-context cell next month has exactly two
likely causes: prompt regression in our code (caught by tests) or a model
drift on the server (caught ONLY by recording the snapshot).

Design
------
Process-global accumulator behind a single ``threading.Lock``. Each
``LLMAgent.get_llm_response`` calls ``record(served_model=..., system_fingerprint=...)``
after every successful completion. Failures inside record() are swallowed
locally; provenance must never crash a run.

Snapshot shape (returned by ``snapshot()``):

    {
        "total_calls": int,
        "served_models": {model_name: count, ...},
        "system_fingerprints": {fp_or_null_str: count, ...},
        "first_seen_utc": ISO 8601 str,
        "last_seen_utc": ISO 8601 str,
    }

Returns ``None`` when no calls were recorded (so an LLM-less ``--skip-judge
--skip-nl-checks`` run records ``observed_model_snapshot: null`` rather than
a confusing empty dict).
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_served_models: dict[str, int] = {}
_system_fingerprints: dict[str, int] = {}
_total_calls: int = 0
_first_seen_utc: str | None = None
_last_seen_utc: str | None = None


def record(*, served_model: Any, system_fingerprint: Any) -> None:
    """Record one LLM call's observed model identifiers.

    Never raises; if the inputs are unusable the call is silently dropped.
    ``served_model`` and ``system_fingerprint`` may be None, str, or any
    object with a useful ``__str__``; they are coerced and the literal
    string ``"null"`` is used as the key for missing values so the snapshot
    JSON keeps them visible.
    """
    global _total_calls, _first_seen_utc, _last_seen_utc
    try:
        sm_key = "null" if served_model is None else str(served_model)
        fp_key = "null" if system_fingerprint is None else str(system_fingerprint)
        now = datetime.now(timezone.utc).isoformat()
        with _lock:
            _served_models[sm_key] = _served_models.get(sm_key, 0) + 1
            _system_fingerprints[fp_key] = _system_fingerprints.get(fp_key, 0) + 1
            _total_calls += 1
            if _first_seen_utc is None:
                _first_seen_utc = now
            _last_seen_utc = now
    except Exception:
        # Provenance never breaks a run.
        return


def snapshot() -> dict[str, Any] | None:
    """Return an immutable snapshot of the accumulator, or None if empty."""
    with _lock:
        if _total_calls == 0:
            return None
        return {
            "total_calls": _total_calls,
            "served_models": dict(_served_models),
            "system_fingerprints": dict(_system_fingerprints),
            "first_seen_utc": _first_seen_utc,
            "last_seen_utc": _last_seen_utc,
        }


def reset() -> None:
    """Test-only hook. Clears all accumulator state."""
    global _served_models, _system_fingerprints, _total_calls
    global _first_seen_utc, _last_seen_utc
    with _lock:
        _served_models = {}
        _system_fingerprints = {}
        _total_calls = 0
        _first_seen_utc = None
        _last_seen_utc = None
