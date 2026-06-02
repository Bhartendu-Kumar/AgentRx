"""Aggregator codebase-identity tests (G3).

The W1b pipeline writes ``<run_dir>/run_config.json`` containing the
producing AgentRx git SHA + full RunConfig + input sha256s. R04 (G2)
ignored it. This test file pins the G3 contract: ``load_run_identity``
reads that file back and ``aggregate_invocation`` attaches the
resulting ``RunIdentity`` to every CellReport, so the report answers
*which code produced this number, against which inputs, using which
recipe*. The aggregator also stamps its own git SHA at report time.

A reproduction record without a commit SHA + input checksum is a
floating number -- you can't clone, re-run, or compare. G3 closes that.

Tests pin:
  - load_run_identity field-by-field on a well-formed payload
  - graceful None on missing dir / missing file / corrupt JSON / non-dict
  - degraded RunIdentity when older payloads lack package / data_provenance
  - flow through aggregate_invocation: OK + METRIC_MISSING + MISSING_SUMMARY
    paths all carry run_identity (because run_dir exists); MISSING_RUN_DIR
    leaves it None
  - report_to_json top-level pins aggregator_commit_sha (string or null)
    and generated_at_utc as ISO-8601 UTC
  - report_to_json round-trips run_identity into a nested dict with the
    full run_config_payload preserved verbatim
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentrx.reproduction.aggregator import (  # noqa: E402
    RunIdentity,
    _aggregator_commit_sha,
    aggregate_invocation,
    load_run_identity,
    report_to_json,
)
from agentrx.reproduction.paper_claims import get_claim  # noqa: E402


# --------------------------------------------------------------------------
# Helpers (re-derived locally to keep this file standalone)
# --------------------------------------------------------------------------


def _run(**overrides) -> dict:
    base = {
        "Correct cases": 5, "Incorrect cases": 5,
        "Correct step number predictions": 4, "Incorrect step number predictions": 6,
        "Overall average distance": 3.0,
        "Step accuracy within +-1": 0.4,
        "Step accuracy within +-3": 0.6,
        "Step accuracy within +-5": 0.8,
        "model_name": "gpt-5",
        "api_version": "2025-01-01-preview",
        "total_prompt_tokens": 1000, "total_output_tokens": 200,
        "total_execution_time_sec": 30.0,
    }
    base.update(overrides)
    return base


def _summary(runs: list[dict]) -> dict:
    return {"num_runs": len(runs), "individual_run_summaries": runs}


def _write_summary(rd: Path, summary: dict) -> None:
    out = rd / "judge_output" / "analysis" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))


def _write_run_config(
    rd: Path,
    *,
    commit_sha: str | None = "deadbeef" * 5,
    version: str | None = "0.1.0",
    input_sha256: str | None = "a" * 64,
    gt_sha256: str | None = "b" * 64,
    timestamp: str | None = "2026-06-02T13:00:00+00:00",
    run_config: dict | None = None,
    omit_package: bool = False,
    omit_data_provenance: bool = False,
) -> Path:
    payload: dict = {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "domain": "tau",
        "endpoint": "https://example.invalid/v1",
        "run_config": run_config or {
            "prompt_mode": "static", "exec_mode": "oneshot",
            "num_runs": 3, "with_context": True,
        },
        "model_spec": {"deployment": "gpt-5", "api_version": "2025-01-01-preview"},
    }
    if not omit_package:
        payload["package"] = {
            "agentrx_commit_sha": commit_sha,
            "agentrx_version": version,
            "python_version": "3.11.13",
        }
    if not omit_data_provenance:
        payload["data_provenance"] = {
            "input_sha256": input_sha256,
            "ground_truth_sha256": gt_sha256,
            "ground_truth_path": "/abs/path/gt.json",
        }
    p = rd / "run_config.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


# --------------------------------------------------------------------------
# load_run_identity: field-by-field
# --------------------------------------------------------------------------


def test_load_run_identity_basic(tmp_path):
    _write_run_config(tmp_path)
    rid = load_run_identity(tmp_path)
    assert rid is not None
    assert rid.producing_agentrx_commit_sha == "deadbeef" * 5
    assert rid.producing_agentrx_version == "0.1.0"
    assert rid.timestamp_utc == "2026-06-02T13:00:00+00:00"
    assert rid.input_sha256 == "a" * 64
    assert rid.ground_truth_sha256 == "b" * 64
    assert rid.run_config_path.endswith("run_config.json")
    # Full payload preserved verbatim so a reader can introspect anything.
    assert rid.run_config_payload is not None
    assert rid.run_config_payload["run_config"]["prompt_mode"] == "static"
    assert rid.run_config_payload["run_config"]["num_runs"] == 3


def test_load_run_identity_returns_none_when_dir_missing(tmp_path):
    assert load_run_identity(tmp_path / "nope") is None


def test_load_run_identity_returns_none_when_file_missing(tmp_path):
    # Dir exists but no run_config.json.
    assert load_run_identity(tmp_path) is None


def test_load_run_identity_returns_none_when_json_corrupt(tmp_path):
    (tmp_path / "run_config.json").write_text("{not valid json")
    assert load_run_identity(tmp_path) is None


def test_load_run_identity_returns_none_when_payload_is_list(tmp_path):
    # Non-dict payloads must not crash field projection.
    (tmp_path / "run_config.json").write_text("[1, 2, 3]")
    assert load_run_identity(tmp_path) is None


def test_load_run_identity_degrades_when_package_missing(tmp_path):
    _write_run_config(tmp_path, omit_package=True)
    rid = load_run_identity(tmp_path)
    assert rid is not None
    assert rid.producing_agentrx_commit_sha is None
    assert rid.producing_agentrx_version is None
    # Other fields still present.
    assert rid.input_sha256 == "a" * 64
    assert rid.run_config_payload is not None


def test_load_run_identity_degrades_when_data_provenance_missing(tmp_path):
    _write_run_config(tmp_path, omit_data_provenance=True)
    rid = load_run_identity(tmp_path)
    assert rid is not None
    assert rid.input_sha256 is None
    assert rid.ground_truth_sha256 is None
    assert rid.producing_agentrx_commit_sha == "deadbeef" * 5


def test_load_run_identity_handles_null_commit_sha(tmp_path):
    # When the producing checkout was not a git repo, W1b writes null.
    _write_run_config(tmp_path, commit_sha=None)
    rid = load_run_identity(tmp_path)
    assert rid is not None
    assert rid.producing_agentrx_commit_sha is None


# --------------------------------------------------------------------------
# Flow through aggregate_invocation
# --------------------------------------------------------------------------


def test_run_identity_flows_onto_ok_cell_report(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run(), _run()]))
    _write_run_config(rd, commit_sha="cafef00d" * 5)
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "OK"
    assert r.run_identity is not None
    assert r.run_identity.producing_agentrx_commit_sha == "cafef00d" * 5
    assert r.run_identity.input_sha256 == "a" * 64


def test_run_identity_flows_onto_metric_missing_cell_report(tmp_path):
    """Even when the metric can't project, the identity of the producing
    code is meaningful -- it tells the user where to debug."""
    rd = tmp_path / "rd"
    rd.mkdir()
    bad = _run()
    del bad["Overall average distance"]
    _write_summary(rd, _summary([bad]))
    _write_run_config(rd)
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t5_tau_oneshot_baseline_dist")],
    )
    r = reports[0]
    assert r.status == "METRIC_MISSING"
    assert r.run_identity is not None
    assert r.run_identity.producing_agentrx_commit_sha == "deadbeef" * 5


def test_run_identity_flows_onto_missing_summary_cell_report(tmp_path):
    """run_config.json without a summary still tells you which code was
    attempted -- valuable for debugging missed/crashed judge runs."""
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_run_config(rd)
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "MISSING_SUMMARY"
    assert r.run_identity is not None
    assert r.run_identity.producing_agentrx_commit_sha == "deadbeef" * 5


def test_run_identity_is_none_when_run_dir_missing(tmp_path):
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=tmp_path / "nope",
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "MISSING_RUN_DIR"
    assert r.run_identity is None


def test_run_identity_is_none_when_run_config_absent_but_summary_present(tmp_path):
    """summary.json without run_config.json is legal (older runs).
    run_identity is None; the cell still reports OK with provenance."""
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run()]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "OK"
    assert r.run_identity is None
    assert r.provenance is not None  # G2 still works


# --------------------------------------------------------------------------
# Report-level metadata
# --------------------------------------------------------------------------


def test_aggregator_commit_sha_returns_string_or_none():
    """Must not raise, must be either a hex string or None."""
    sha = _aggregator_commit_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) == 40 and all(
        c in "0123456789abcdef" for c in sha
    ))


def test_report_to_json_pins_aggregator_commit_sha(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run()]))
    _write_run_config(rd)
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    payload = json.loads(report_to_json(reports))
    assert "aggregator_commit_sha" in payload
    # Either a 40-char hex SHA or null. Both are valid.
    sha = payload["aggregator_commit_sha"]
    assert sha is None or (isinstance(sha, str) and len(sha) == 40)


def test_report_to_json_pins_generated_at_utc(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run()]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    payload = json.loads(report_to_json(reports))
    assert "generated_at_utc" in payload
    ts = payload["generated_at_utc"]
    assert isinstance(ts, str)
    # ISO-8601 with UTC offset. Round-trips through datetime.fromisoformat.
    import datetime as _dt
    parsed = _dt.datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == _dt.timedelta(0)


def test_report_to_json_round_trips_run_identity(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run()]))
    _write_run_config(rd, commit_sha="feedface" * 5)
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    payload = json.loads(report_to_json(reports))
    cell = payload["reports"][0]
    assert "run_identity" in cell
    rid = cell["run_identity"]
    assert rid["producing_agentrx_commit_sha"] == "feedface" * 5
    assert rid["input_sha256"] == "a" * 64
    # Full payload preserved verbatim so a reader can recover the recipe.
    assert rid["run_config_payload"]["run_config"]["prompt_mode"] == "static"


def test_report_to_json_run_identity_is_null_when_absent(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run()]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    payload = json.loads(report_to_json(reports))
    cell = payload["reports"][0]
    assert cell["run_identity"] is None
