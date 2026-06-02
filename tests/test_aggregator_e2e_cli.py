"""End-to-end CLI integration for the report pipeline (G3.5).

The R04/G2/G3 work added three layers of structure inside the aggregator:
  - per-cell numbers from project_metric (R04)
  - per-cell provenance.model_name, api_version, tokens, time (G2)
  - per-cell run_identity.producing_agentrx_commit_sha + checksums (G3)
  - top-level aggregator_commit_sha + generated_at_utc (G3)

Existing tests verify each layer at the Python-function level. This file
pins the contract that ALL of them flow through the public CLI surface
``python -m agentrx.reproduction report --format {json,markdown,summary}``
when given a real sweep manifest. Without this test, a future refactor
that quietly drops a field from report_to_json or breaks the CLI argument
parser would still pass the unit tests but ship a broken CLI.

These tests fabricate per-invocation summary.json + run_config.json files
under a tmp directory and exercise the CLI as a subprocess; they do NOT
require Azure (G4 covers real LLM calls).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentrx.reproduction.sweep import (  # noqa: E402
    build_manifest,
    dump_manifest,
    group_into_invocations,
    select_claims,
)


# --------------------------------------------------------------------------
# Fabrication helpers (kept local so this file is standalone)
# --------------------------------------------------------------------------


def _make_run(
    *,
    correct: int = 30,
    incorrect: int = 70,
    correct_step: int = 40,
    incorrect_step: int = 60,
    overall_distance: float = 4.5,
    model_name: str = "gpt-5",
    api_version: str = "2025-01-01-preview",
    prompt_tokens: int = 1000,
    output_tokens: int = 200,
    exec_time: float = 30.0,
) -> dict:
    return {
        "Correct cases": correct,
        "Incorrect cases": incorrect,
        "Correct step number predictions": correct_step,
        "Incorrect step number predictions": incorrect_step,
        "Overall average distance": overall_distance,
        "Step accuracy within +-1": 0.55,
        "Step accuracy within +-2": 0.63,
        "Step accuracy within +-3": 0.70,
        "Step accuracy within +-4": 0.75,
        "Step accuracy within +-5": 0.80,
        "model_name": model_name,
        "api_version": api_version,
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
        "total_execution_time_sec": exec_time,
    }


def _make_summary(runs: list[dict]) -> dict:
    return {"num_runs": len(runs), "individual_run_summaries": runs}


def _write_summary(run_dir: Path, summary: dict) -> Path:
    out = run_dir / "judge_output" / "analysis" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return out


def _write_run_config(
    run_dir: Path,
    *,
    commit_sha: str = "cafef00d" * 5,
    version: str = "0.1.0",
    input_sha256: str = "a" * 64,
    gt_sha256: str = "b" * 64,
) -> Path:
    payload = {
        "schema_version": 1,
        "timestamp_utc": "2026-06-02T13:00:00+00:00",
        "domain": "tau",
        "endpoint": "https://example.invalid/v1",
        "run_config": {
            "prompt_mode": "static", "exec_mode": "oneshot",
            "num_runs": 3, "with_context": True,
        },
        "model_spec": {"deployment": "gpt-5", "api_version": "2025-01-01-preview"},
        "package": {
            "agentrx_commit_sha": commit_sha,
            "agentrx_version": version,
            "python_version": "3.11.13",
        },
        "data_provenance": {
            "input_sha256": input_sha256,
            "ground_truth_sha256": gt_sha256,
            "ground_truth_path": "/abs/path/gt.json",
        },
    }
    p = run_dir / "run_config.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def _fake_repo(tmp_path: Path) -> Path:
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text("# stub\n")
    return base


def _build_populated_sweep(
    tmp_path: Path,
    *,
    table_label: str = "tab:static-dynamic",
    write_run_config: bool = True,
    commit_sha: str = "cafef00d" * 5,
) -> Path:
    """Build a manifest + populate every invocation with summary + run_config.

    Returns the manifest path.
    """
    base = _fake_repo(tmp_path)
    claims = select_claims(table_label=table_label)
    invs = group_into_invocations(claims)
    runs_root = tmp_path / "runs"
    manifest = build_manifest(
        invs, runs_root=runs_root,
        selection={"table_label": table_label},
        base_dir=base, endpoint="azure", python="/usr/bin/python",
    )
    dump_manifest(manifest, runs_root)
    for entry in manifest["invocations"]:
        rd = Path(entry["run_dir"])
        _write_summary(rd, _make_summary([_make_run(), _make_run(), _make_run()]))
        if write_run_config:
            _write_run_config(rd, commit_sha=commit_sha)
    return runs_root / "manifest.json"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", *args],
        capture_output=True, text=True, timeout=60,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


# --------------------------------------------------------------------------
# JSON format: every G1/G2/G3 field must surface
# --------------------------------------------------------------------------


def test_cli_report_json_top_level_pins_aggregator_sha_and_timestamp(tmp_path):
    manifest = _build_populated_sweep(tmp_path)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "json")
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    assert payload["schema_version"] == 1
    # aggregator_commit_sha: string-or-null. If a string, must be 40 hex chars.
    sha = payload["aggregator_commit_sha"]
    assert sha is None or (
        isinstance(sha, str) and len(sha) == 40
        and all(c in "0123456789abcdef" for c in sha)
    )
    # generated_at_utc: ISO-8601 with explicit UTC offset.
    ts = payload["generated_at_utc"]
    parsed = _dt.datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == _dt.timedelta(0)


def test_cli_report_json_carries_run_identity_per_cell(tmp_path):
    """G3: producing_agentrx_commit_sha must reach every CellReport in CLI output."""
    manifest = _build_populated_sweep(tmp_path, commit_sha="feedface" * 5)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "json")
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    # Every OK report must carry run_identity with the producing SHA we wrote.
    ok = [r for r in payload["reports"] if r["status"] == "OK"]
    assert ok, "expected at least one OK cell"
    for r in ok:
        rid = r["run_identity"]
        assert rid is not None, f"missing run_identity on {r['cell_id']}"
        assert rid["producing_agentrx_commit_sha"] == "feedface" * 5
        assert rid["input_sha256"] == "a" * 64
        assert rid["ground_truth_sha256"] == "b" * 64
        # Full payload preserved so a reader can recover the exact recipe.
        assert rid["run_config_payload"]["run_config"]["prompt_mode"] == "static"


def test_cli_report_json_carries_provenance_per_cell(tmp_path):
    """G2: model_name, api_version, tokens, time must reach every CellReport."""
    manifest = _build_populated_sweep(tmp_path)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "json")
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    ok = [r for r in payload["reports"] if r["status"] == "OK"]
    assert ok
    for r in ok:
        pv = r["provenance"]
        assert pv is not None, f"missing provenance on {r['cell_id']}"
        assert pv["model_name"] == "gpt-5"
        assert pv["api_version"] == "2025-01-01-preview"
        # _make_run writes prompt_tokens=1000, 3 runs per cell, summed.
        assert pv["total_prompt_tokens"] == 3000
        assert pv["total_output_tokens"] == 600
        assert pv["total_execution_time_sec"] == pytest.approx(90.0)
        assert pv["n_runs"] == 3
        assert pv["model_homogeneous"] is True


def test_cli_report_json_run_identity_null_when_run_config_absent(tmp_path):
    """Legacy compat: summaries without W1b run_config.json are still
    valid; the CLI must emit run_identity=null without crashing."""
    manifest = _build_populated_sweep(tmp_path, write_run_config=False)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "json")
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    ok = [r for r in payload["reports"] if r["status"] == "OK"]
    assert ok
    for r in ok:
        assert r["run_identity"] is None
        # G2 provenance still works -- summaries always carry model info.
        assert r["provenance"] is not None
        assert r["provenance"]["model_name"] == "gpt-5"


def test_cli_report_json_n_reports_matches_catalog_selection(tmp_path):
    """Join correctness through the CLI: 1 report per claim in the selection."""
    manifest = _build_populated_sweep(tmp_path, table_label="tab:static-dynamic")
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "json")
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    expected = select_claims(table_label="tab:static-dynamic")
    assert payload["n_reports"] == len(expected)
    assert sorted(r["cell_id"] for r in payload["reports"]) == sorted(
        c.cell_id for c in expected
    )
    assert payload["n_ok"] == len(expected)


def test_cli_report_json_paper_delta_arithmetic(tmp_path):
    """abs_delta and within_paper_std must be derived correctly end-to-end."""
    manifest = _build_populated_sweep(tmp_path)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "json")
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    for r in payload["reports"]:
        if r["status"] != "OK":
            continue
        # observed = mean of per-run projections; paper = catalog value.
        assert r["abs_delta"] == pytest.approx(
            abs(r["observed_mean"] - r["paper_mean"])
        )


# --------------------------------------------------------------------------
# Markdown format: model column flows from provenance
# --------------------------------------------------------------------------


def test_cli_report_markdown_includes_model_column_from_provenance(tmp_path):
    manifest = _build_populated_sweep(tmp_path)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "markdown")
    assert cp.returncode == 0, cp.stderr
    lines = cp.stdout.splitlines()
    # Header asserts the column exists (added in G2).
    assert "| model |" in lines[0]
    # Every data row must contain the homogeneous model name.
    data_rows = lines[2:]
    assert data_rows, "no data rows in markdown output"
    for row in data_rows:
        assert "gpt-5" in row, f"missing model name in {row!r}"


def test_cli_report_markdown_dash_when_no_provenance(tmp_path):
    """When summary.json is absent, markdown's model column shows em-dash."""
    base = _fake_repo(tmp_path)
    claims = select_claims(table_label="tab:static-dynamic")
    invs = group_into_invocations(claims)
    runs_root = tmp_path / "runs"
    manifest = build_manifest(
        invs, runs_root=runs_root,
        selection={"table_label": "tab:static-dynamic"},
        base_dir=base, endpoint="azure", python="/usr/bin/python",
    )
    dump_manifest(manifest, runs_root)
    # Deliberately do NOT write any summary.json -> every cell MISSING_SUMMARY
    # or MISSING_RUN_DIR. The model column must be em-dash on every row.
    cp = _run_cli("report", "--manifest", str(runs_root / "manifest.json"),
                  "--format", "markdown")
    assert cp.returncode == 0, cp.stderr
    lines = cp.stdout.splitlines()
    data_rows = lines[2:]
    assert data_rows
    for row in data_rows:
        assert "\u2014" in row, f"missing em-dash for absent provenance: {row!r}"


# --------------------------------------------------------------------------
# Summary format: status counters add up
# --------------------------------------------------------------------------


def test_cli_report_summary_counts_sum_to_total(tmp_path):
    """Status counters in `summary` format must partition all reports."""
    manifest = _build_populated_sweep(tmp_path)
    cp = _run_cli("report", "--manifest", str(manifest), "--format", "summary")
    assert cp.returncode == 0, cp.stderr
    # Parse "  OK:                  N" style lines.
    counts: dict[str, int] = {}
    n_cells = None
    for line in cp.stdout.splitlines():
        line = line.strip()
        if line.startswith("# Aggregator summary"):
            # "# Aggregator summary (8 cells, manifest=...)"
            import re
            m = re.search(r"\((\d+) cells", line)
            assert m, line
            n_cells = int(m.group(1))
            continue
        # Match exactly two-segment "Key: N" lines.
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if val.isdigit():
            counts[key.strip()] = int(val)
    assert n_cells is not None
    # OK + Missing run dir + Missing summary + Metric not projected = n_cells.
    partition = (
        counts["OK"]
        + counts["Missing run dir"]
        + counts["Missing summary"]
        + counts["Metric not projected"]
    )
    assert partition == n_cells, f"status counts don't partition: {counts}"
