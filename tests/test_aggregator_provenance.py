"""Aggregator provenance tests (G2).

Distinct from tests/test_provenance.py (which covers the W1b RunConfig
provenance dump). This file covers the aggregator's per-cell provenance:
model_name, api_version, token totals, execution time, and homogeneity
detection across runs.

The aggregator's project_metric tells you the NUMBER; provenance tells
you what produced it. A reproduction number without (model, api_version,
prompt_tokens, output_tokens, execution_time) is unverifiable.

Tests pin:
  - field-by-field extraction from a well-formed per-run summary
  - sum semantics across runs (tokens, time)
  - model homogeneity detection (single model vs MIXED)
  - graceful absence: every field allowed to be None when the
    corresponding per-run key is missing (older judge outputs)
  - provenance flows through aggregate_invocation onto every OK and
    METRIC_MISSING CellReport, and is None when the run dir / summary
    is missing entirely
  - JSON serialisation round-trips provenance into a nested dict
  - Markdown rendering surfaces the model column with MIXED:... format
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentrx.reproduction.aggregator import (  # noqa: E402
    CellReport,
    Provenance,
    aggregate_invocation,
    extract_provenance,
    render_cell_reports_markdown,
    report_to_json,
)
from agentrx.reproduction.paper_claims import get_claim  # noqa: E402


def _run(**overrides) -> dict:
    base = {
        "Correct cases": 5,
        "Incorrect cases": 5,
        "Correct step number predictions": 4,
        "Incorrect step number predictions": 6,
        "Overall average distance": 3.0,
        "Step accuracy within +-1": 0.4,
        "Step accuracy within +-3": 0.6,
        "Step accuracy within +-5": 0.8,
        "model_name": "gpt-5",
        "api_version": "2025-01-01-preview",
        "total_prompt_tokens": 1000,
        "total_output_tokens": 200,
        "total_execution_time_sec": 30.0,
    }
    base.update(overrides)
    return base


def _summary(runs: list[dict]) -> dict:
    return {"num_runs": len(runs), "individual_run_summaries": runs}


# --------------------------------------------------------------------------
# extract_provenance: field-by-field
# --------------------------------------------------------------------------


def test_extract_provenance_basic():
    p = extract_provenance(_summary([_run(), _run(), _run()]))
    assert p.model_name == "gpt-5"
    assert p.api_version == "2025-01-01-preview"
    assert p.total_prompt_tokens == 3000
    assert p.total_output_tokens == 600
    assert p.total_execution_time_sec == pytest.approx(90.0)
    assert p.n_runs == 3
    assert p.model_homogeneous is True
    assert p.distinct_model_names == ["gpt-5"]


def test_extract_provenance_detects_mixed_models():
    p = extract_provenance(_summary([
        _run(model_name="gpt-5"),
        _run(model_name="gpt-4o"),
        _run(model_name="gpt-5"),
    ]))
    assert p.model_homogeneous is False
    assert p.distinct_model_names == ["gpt-4o", "gpt-5"]
    # When mixed, model_name is still set (alphabetically first) but the
    # model_homogeneous flag is what callers MUST check.
    assert p.model_name in {"gpt-4o", "gpt-5"}


def test_extract_provenance_graceful_when_keys_missing():
    """Older judge outputs don't carry model_name/tokens. Every field
    must degrade to None individually instead of raising."""
    bare = {
        "Correct cases": 1, "Incorrect cases": 0,
        "Correct step number predictions": 1, "Incorrect step number predictions": 0,
        "Overall average distance": 0.0,
        "Step accuracy within +-1": 1.0, "Step accuracy within +-3": 1.0,
        "Step accuracy within +-5": 1.0,
    }
    p = extract_provenance(_summary([bare]))
    assert p.model_name is None
    assert p.api_version is None
    assert p.total_prompt_tokens is None
    assert p.total_output_tokens is None
    assert p.total_execution_time_sec is None
    assert p.n_runs == 1
    # No model names at all -> trivially homogeneous (no contradictions).
    assert p.model_homogeneous is True
    assert p.distinct_model_names == []


def test_extract_provenance_partial_keys_some_runs():
    """One run missing tokens must NOT poison the sum -- present runs are
    still summed; absent runs contribute nothing."""
    incomplete = _run()
    del incomplete["total_prompt_tokens"]
    p = extract_provenance(_summary([_run(), incomplete, _run()]))
    assert p.total_prompt_tokens == 2000  # 2 runs * 1000


def test_extract_provenance_no_runs():
    p = extract_provenance({"num_runs": 0, "individual_run_summaries": []})
    assert p.model_name is None
    assert p.api_version is None
    assert p.total_prompt_tokens is None
    assert p.n_runs == 0
    assert p.model_homogeneous is True


def test_extract_provenance_missing_individual_runs_key():
    p = extract_provenance({"num_runs": 0})
    assert p.n_runs == 0


# --------------------------------------------------------------------------
# Flow through aggregate_invocation
# --------------------------------------------------------------------------


def _write_summary(rd: Path, summary: dict) -> None:
    out = rd / "judge_output" / "analysis" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))


def test_provenance_flows_onto_ok_cell_report(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run(), _run()]))
    reports = aggregate_invocation(
        invocation_slug="slug",
        run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "OK"
    assert r.provenance is not None
    assert r.provenance.model_name == "gpt-5"
    assert r.provenance.total_prompt_tokens == 2000


def test_provenance_flows_onto_metric_missing_cell_report(tmp_path):
    """Provenance is still meaningful when the metric can't project --
    the user needs to know which model produced the unprojectable run."""
    rd = tmp_path / "rd"
    rd.mkdir()
    bad = _run()
    del bad["Overall average distance"]
    _write_summary(rd, _summary([bad]))
    reports = aggregate_invocation(
        invocation_slug="slug",
        run_dir=rd,
        claims=[get_claim("t5_tau_oneshot_baseline_dist")],
    )
    r = reports[0]
    assert r.status == "METRIC_MISSING"
    assert r.provenance is not None
    assert r.provenance.model_name == "gpt-5"


def test_provenance_is_none_when_run_dir_missing(tmp_path):
    reports = aggregate_invocation(
        invocation_slug="slug",
        run_dir=tmp_path / "nope",
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "MISSING_RUN_DIR"
    assert r.provenance is None


def test_provenance_is_none_when_summary_missing(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    r = reports[0]
    assert r.status == "MISSING_SUMMARY"
    assert r.provenance is None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_report_to_json_round_trips_provenance(tmp_path):
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _summary([_run(), _run()]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd,
        claims=[get_claim("t4_tau_full_step")],
    )
    payload = json.loads(report_to_json(reports))
    prov = payload["reports"][0]["provenance"]
    assert prov["model_name"] == "gpt-5"
    assert prov["total_prompt_tokens"] == 2000
    assert prov["model_homogeneous"] is True
    assert prov["distinct_model_names"] == ["gpt-5"]


def test_markdown_shows_homogeneous_model_name():
    rep = CellReport(
        cell_id="x", table_label="tab:y", domain="tau", metric="step_index_acc",
        paper_mean=48.3, paper_std=0.0, unit="percent",
        run_dir="/r", invocation_slug="s", status="OK",
        observed_mean=48.3, observed_std=0.0, n_runs=2,
        abs_delta=0.0, within_paper_std=True,
        provenance=Provenance(
            model_name="gpt-5", api_version="v1",
            total_prompt_tokens=100, total_output_tokens=50,
            total_execution_time_sec=10.0, n_runs=2,
            model_homogeneous=True, distinct_model_names=["gpt-5"],
        ),
    )
    md = render_cell_reports_markdown([rep])
    lines = md.splitlines()
    assert "| model |" in lines[0]
    assert "| gpt-5 |" in lines[2]


def test_markdown_flags_mixed_models():
    rep = CellReport(
        cell_id="x", table_label="tab:y", domain="tau", metric="step_index_acc",
        paper_mean=10.0, paper_std=1.0, unit="percent",
        run_dir="/r", invocation_slug="s", status="OK",
        observed_mean=11.0, observed_std=0.5, n_runs=3,
        abs_delta=1.0, within_paper_std=True,
        provenance=Provenance(
            model_name="gpt-4o", api_version="v1",
            total_prompt_tokens=300, total_output_tokens=60,
            total_execution_time_sec=45.0, n_runs=3,
            model_homogeneous=False, distinct_model_names=["gpt-4o", "gpt-5"],
        ),
    )
    md = render_cell_reports_markdown([rep])
    assert "MIXED:gpt-4o,gpt-5" in md


def test_markdown_dash_when_no_provenance():
    rep = CellReport(
        cell_id="x", table_label="tab:y", domain="tau", metric="step_index_acc",
        paper_mean=10.0, paper_std=1.0, unit="percent",
        run_dir="/r", invocation_slug="s", status="MISSING_RUN_DIR",
    )
    md = render_cell_reports_markdown([rep])
    lines = md.splitlines()
    # The model cell should be an em-dash since provenance is None.
    assert "| \u2014 |" in lines[2]


# --------------------------------------------------------------------------
# Real-corpus integration
# --------------------------------------------------------------------------


CORPUS = REPO_ROOT / "tests" / "fixtures" / "judge_summaries"


@pytest.mark.parametrize("path", sorted(CORPUS.rglob("*.summary.json")),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_provenance_extractable_from_real_corpus(path: Path):
    """Every vendored real summary must yield a populated Provenance."""
    d = json.loads(path.read_text())
    p = extract_provenance(d)
    assert p.n_runs >= 1
    # Real summaries in our corpus always carry these.
    assert p.model_name is not None, f"{path}: no model_name in any run"
    assert p.api_version is not None, f"{path}: no api_version in any run"
    assert p.total_prompt_tokens is not None and p.total_prompt_tokens > 0
    assert p.total_output_tokens is not None and p.total_output_tokens > 0
    assert p.total_execution_time_sec is not None
    assert p.total_execution_time_sec > 0
    assert p.model_homogeneous is True, (
        f"{path}: corpus fixture is unexpectedly heterogeneous: "
        f"{p.distinct_model_names}"
    )
