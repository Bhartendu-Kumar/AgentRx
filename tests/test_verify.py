"""Reproduction-verdict tests (verify gate).

``verify`` adds the single missing concern between ``report`` (what the
numbers are) and a CI gate (did we reproduce the paper). These tests pin:

  - the per-cell policy: every branch (PASS / FAIL / INCONCLUSIVE) for
    every status the aggregator can emit
  - tolerance resolution: paper_std * multiplier is primary; abs_tol is a
    fallback used ONLY when paper_std is None; neither -> INCONCLUSIVE
  - the std_multiplier scaling and the zero-std edge
  - the overall gate: green iff EVERY cell is PASS
  - serialisation round-trips and exit codes through the real CLI
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentrx.reproduction.aggregator import CellReport  # noqa: E402
from agentrx.reproduction.verify import (  # noqa: E402
    FAIL,
    INCONCLUSIVE,
    PASS,
    classify_cell,
    classify_reports,
    render_verdict_summary,
    verdict_to_json,
)


# --------------------------------------------------------------------------
# Helpers: build a CellReport directly (decoupled from the catalog/sweep)
# --------------------------------------------------------------------------


def _report(
    *,
    cell_id: str = "cell_x",
    metric: str = "step_index_acc",
    domain: str = "tau",
    status: str = "OK",
    paper_mean: float = 32.2,
    paper_std: float | None = 3.2,
    observed_mean: float | None = 33.0,
    abs_delta: float | None = 0.8,
    within_paper_std: bool | None = True,
) -> CellReport:
    return CellReport(
        cell_id=cell_id,
        table_label="tab:ablations",
        domain=domain,
        metric=metric,
        paper_mean=paper_mean,
        paper_std=paper_std,
        unit="percent",
        run_dir="/tmp/whatever",
        invocation_slug="slug",
        status=status,
        observed_mean=observed_mean,
        observed_std=0.0,
        n_runs=3,
        abs_delta=abs_delta,
        within_paper_std=within_paper_std,
    )


# --------------------------------------------------------------------------
# Per-cell policy
# --------------------------------------------------------------------------


def test_ok_within_band_is_pass():
    # |33.0 - 32.2| = 0.8 <= 3.2
    v = classify_cell(_report(observed_mean=33.0, abs_delta=0.8))
    assert v.verdict == PASS
    assert v.tolerance == pytest.approx(3.2)
    assert v.tolerance_kind == "paper_std*k"


def test_ok_outside_band_is_fail():
    # |40.0 - 32.2| = 7.8 > 3.2
    v = classify_cell(_report(observed_mean=40.0, abs_delta=7.8))
    assert v.verdict == FAIL
    assert v.tolerance == pytest.approx(3.2)


def test_std_multiplier_widens_band():
    # |40.0 - 32.2| = 7.8; with k=3 -> tol = 9.6, so PASS
    r = _report(observed_mean=40.0, abs_delta=7.8)
    assert classify_cell(r, std_multiplier=1.0).verdict == FAIL
    assert classify_cell(r, std_multiplier=3.0).verdict == PASS


def test_zero_std_requires_exact_match():
    # paper_std=0 -> tol=0; only |delta|==0 passes
    assert classify_cell(_report(paper_std=0.0, abs_delta=0.0)).verdict == PASS
    assert classify_cell(_report(paper_std=0.0, abs_delta=0.1)).verdict == FAIL


def test_no_std_no_abstol_is_inconclusive():
    v = classify_cell(_report(paper_std=None, within_paper_std=None))
    assert v.verdict == INCONCLUSIVE
    assert v.tolerance is None
    assert v.tolerance_kind is None


def test_no_std_with_abstol_uses_abstol():
    r = _report(paper_std=None, observed_mean=49.0, paper_mean=48.3,
                abs_delta=0.7, within_paper_std=None)
    v_pass = classify_cell(r, abs_tol=1.0)
    assert v_pass.verdict == PASS
    assert v_pass.tolerance == pytest.approx(1.0)
    assert v_pass.tolerance_kind == "abs_tol"
    v_fail = classify_cell(r, abs_tol=0.5)
    assert v_fail.verdict == FAIL


def test_paper_std_takes_precedence_over_abstol():
    # paper_std present -> abs_tol is ignored entirely
    r = _report(paper_std=3.2, observed_mean=40.0, abs_delta=7.8)
    v = classify_cell(r, abs_tol=100.0)  # would pass under abs_tol
    assert v.verdict == FAIL
    assert v.tolerance_kind == "paper_std*k"


@pytest.mark.parametrize("status", ["MISSING_RUN_DIR", "MISSING_SUMMARY"])
def test_missing_artifacts_are_fail(status):
    v = classify_cell(_report(status=status, observed_mean=None,
                              abs_delta=None, within_paper_std=None))
    assert v.verdict == FAIL
    assert status in v.reason


def test_metric_missing_is_inconclusive():
    v = classify_cell(_report(status="METRIC_MISSING", observed_mean=None,
                              abs_delta=None, within_paper_std=None))
    assert v.verdict == INCONCLUSIVE


# --------------------------------------------------------------------------
# Overall gate
# --------------------------------------------------------------------------


def test_gate_green_only_when_all_pass():
    reports = [
        _report(cell_id="a", observed_mean=33.0, abs_delta=0.8),
        _report(cell_id="b", observed_mean=31.0, abs_delta=1.2),
    ]
    sv = classify_reports(reports)
    assert sv.passed is True
    assert (sv.n_cells, sv.n_pass, sv.n_fail, sv.n_inconclusive) == (2, 2, 0, 0)


def test_gate_red_on_single_fail():
    reports = [
        _report(cell_id="a", observed_mean=33.0, abs_delta=0.8),
        _report(cell_id="b", observed_mean=99.0, abs_delta=66.8),
    ]
    sv = classify_reports(reports)
    assert sv.passed is False
    assert sv.n_fail == 1


def test_gate_red_on_inconclusive():
    reports = [
        _report(cell_id="a", observed_mean=33.0, abs_delta=0.8),
        _report(cell_id="b", status="METRIC_MISSING", observed_mean=None,
                abs_delta=None, within_paper_std=None),
    ]
    sv = classify_reports(reports)
    assert sv.passed is False
    assert sv.n_inconclusive == 1


def test_empty_selection_is_not_green():
    sv = classify_reports([])
    assert sv.passed is False
    assert sv.n_cells == 0


def test_cells_sorted_by_cell_id():
    reports = [
        _report(cell_id="z"),
        _report(cell_id="a"),
        _report(cell_id="m"),
    ]
    sv = classify_reports(reports)
    assert [c.cell_id for c in sv.cells] == ["a", "m", "z"]


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_verdict_to_json_round_trips():
    sv = classify_reports([_report(observed_mean=33.0, abs_delta=0.8)])
    doc = json.loads(verdict_to_json(sv))
    assert doc["schema_version"] == 1
    assert doc["passed"] is True
    assert doc["n_pass"] == 1
    assert isinstance(doc["generated_at_utc"], str)
    assert doc["cells"][0]["verdict"] == PASS
    assert doc["cells"][0]["tolerance"] == pytest.approx(3.2)


def test_summary_render_has_overall_and_per_cell():
    sv = classify_reports([
        _report(cell_id="a", observed_mean=33.0, abs_delta=0.8),
        _report(cell_id="b", observed_mean=99.0, abs_delta=66.8),
    ])
    text = render_verdict_summary(sv)
    assert "Reproduction verdict: FAIL" in text
    assert "[PASS" in text
    assert "[FAIL" in text


# --------------------------------------------------------------------------
# CLI integration: exit codes + json through the real entry point
# --------------------------------------------------------------------------


def _write_manifest_and_summary(
    tmp_path: Path, cell_id: str, runs: list[dict]
) -> Path:
    """Write a one-invocation manifest + a judge summary the cell reads."""
    run_dir = tmp_path / "inv"
    out = run_dir / "judge_output" / "analysis" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"num_runs": len(runs), "individual_run_summaries": runs}, indent=2
    ))
    manifest = {
        "schema_version": 1,
        "invocations": [
            {"slug": "inv", "run_dir": str(run_dir), "cell_ids": [cell_id]},
        ],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    return mpath


def _run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", *argv],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_cli_verify_exit_zero_when_reproduced(tmp_path):
    # t5_tau_oneshot_baseline_step: paper 32.2 +/- 3.2, metric step_index_acc.
    # Build runs whose per-run step accuracy averages ~0.322 -> 32.2%.
    runs = [
        {"Correct step number predictions": 32, "Incorrect step number predictions": 68},
        {"Correct step number predictions": 33, "Incorrect step number predictions": 67},
        {"Correct step number predictions": 31, "Incorrect step number predictions": 69},
    ]
    mpath = _write_manifest_and_summary(tmp_path, "t5_tau_oneshot_baseline_step", runs)
    proc = _run_cli("verify", "--manifest", str(mpath), "--format", "json")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["passed"] is True
    assert doc["cells"][0]["verdict"] == "PASS"


def test_cli_verify_exit_one_when_far_off(tmp_path):
    runs = [
        {"Correct step number predictions": 90, "Incorrect step number predictions": 10},
        {"Correct step number predictions": 90, "Incorrect step number predictions": 10},
        {"Correct step number predictions": 90, "Incorrect step number predictions": 10},
    ]
    mpath = _write_manifest_and_summary(tmp_path, "t5_tau_oneshot_baseline_step", runs)
    proc = _run_cli("verify", "--manifest", str(mpath), "--format", "json")
    assert proc.returncode == 1
    doc = json.loads(proc.stdout)
    assert doc["passed"] is False
    assert doc["cells"][0]["verdict"] == "FAIL"


def test_cli_verify_no_std_cell_is_inconclusive_without_abstol(tmp_path):
    # t5_tau_oneshot_checklistvio_step: paper 48.3 with NO std.
    runs = [
        {"Correct step number predictions": 48, "Incorrect step number predictions": 52},
    ]
    mpath = _write_manifest_and_summary(
        tmp_path, "t5_tau_oneshot_checklistvio_step", runs)
    # Without --abs-tol: inconclusive -> exit 1.
    proc = _run_cli("verify", "--manifest", str(mpath), "--format", "json")
    assert proc.returncode == 1
    doc = json.loads(proc.stdout)
    assert doc["cells"][0]["verdict"] == "INCONCLUSIVE"
    # With a generous --abs-tol it can pass.
    proc2 = _run_cli("verify", "--manifest", str(mpath),
                     "--abs-tol", "5", "--format", "json")
    assert proc2.returncode == 0
    doc2 = json.loads(proc2.stdout)
    assert doc2["cells"][0]["verdict"] == "PASS"
    assert doc2["cells"][0]["tolerance_kind"] == "abs_tol"
