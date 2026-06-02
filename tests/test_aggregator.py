"""Aggregator tests (R04).

The aggregator joins three things:

  - the catalog's ``cell_id -> (metric, paper_value)``
  - the sweep manifest's ``invocation -> run_dir`` map
  - each invocation's ``judge_output/analysis/summary.json``

…and emits one ``CellReport`` per claim in paper units. Tests pin:
  - paper-unit projection arithmetic for every metric the catalog uses
  - graceful degradation on missing run dir / missing summary / unprojectable metric
  - join correctness: every claim that the sweep planned must appear in the report
  - delta math (``abs_delta``, ``within_paper_std``)
  - rendering (json schema + markdown header) is stable
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentrx.reproduction.aggregator import (  # noqa: E402
    CellReport,
    MetricMissing,
    aggregate_invocation,
    aggregate_sweep,
    load_judge_summary,
    project_metric,
    render_cell_reports_markdown,
    report_to_json,
)
from agentrx.reproduction.paper_claims import get_claim  # noqa: E402
from agentrx.reproduction.sweep import (  # noqa: E402
    build_manifest,
    dump_manifest,
    group_into_invocations,
    select_claims,
)


# --------------------------------------------------------------------------
# Helpers: build a minimal judge summary that satisfies project_metric
# --------------------------------------------------------------------------


def _make_run(
    *,
    correct: int = 20,
    incorrect: int = 10,
    correct_step: int = 15,
    incorrect_step: int = 15,
    overall_distance: float = 3.0,
    tol1: float = 0.55,
    tol3: float = 0.70,
    tol5: float = 0.80,
) -> dict:
    return {
        "Correct cases": correct,
        "Incorrect cases": incorrect,
        "Correct step number predictions": correct_step,
        "Incorrect step number predictions": incorrect_step,
        "Overall average distance": overall_distance,
        "Step accuracy within +-1": tol1,
        "Step accuracy within +-2": (tol1 + tol3) / 2,
        "Step accuracy within +-3": tol3,
        "Step accuracy within +-4": (tol3 + tol5) / 2,
        "Step accuracy within +-5": tol5,
    }


def _make_summary(runs: list[dict]) -> dict:
    return {"num_runs": len(runs), "individual_run_summaries": runs}


def _write_summary(run_dir: Path, summary: dict) -> Path:
    out = run_dir / "judge_output" / "analysis" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return out


# --------------------------------------------------------------------------
# project_metric: paper-unit arithmetic
# --------------------------------------------------------------------------


def test_project_step_index_acc_returns_percent_with_pstdev():
    s = _make_summary([
        _make_run(correct_step=10, incorrect_step=20),  # 0.3333
        _make_run(correct_step=18, incorrect_step=12),  # 0.6000
        _make_run(correct_step=15, incorrect_step=15),  # 0.5000
    ])
    mean, std, n = project_metric("step_index_acc", s)
    per_run = [10/30, 18/30, 15/30]
    assert n == 3
    assert mean == pytest.approx(statistics.mean(per_run) * 100, rel=1e-9)
    assert std == pytest.approx(statistics.pstdev(per_run) * 100, rel=1e-9)


def test_project_category_acc_returns_percent():
    s = _make_summary([
        _make_run(correct=12, incorrect=18),  # 0.4000
        _make_run(correct=21, incorrect=9),   # 0.7000
    ])
    mean, std, n = project_metric("category_acc", s)
    per_run = [12/30, 21/30]
    assert n == 2
    assert mean == pytest.approx(statistics.mean(per_run) * 100, rel=1e-9)
    assert std == pytest.approx(statistics.pstdev(per_run) * 100, rel=1e-9)


def test_project_avg_step_distance_passes_through_units():
    """avg_step_distance is in STEPS, not percent. Must NOT be scaled."""
    s = _make_summary([
        _make_run(overall_distance=2.4),
        _make_run(overall_distance=3.0),
        _make_run(overall_distance=3.6),
    ])
    mean, std, n = project_metric("avg_step_distance", s)
    assert n == 3
    assert mean == pytest.approx(3.0, rel=1e-9)
    assert std == pytest.approx(statistics.pstdev([2.4, 3.0, 3.6]), rel=1e-9)


@pytest.mark.parametrize("metric,tol", [
    ("step_acc_at_plus_minus_1", 1),
    ("step_acc_at_plus_minus_3", 3),
    ("step_acc_at_plus_minus_5", 5),
])
def test_project_tolerance_metrics_return_percent(metric, tol):
    s = _make_summary([
        _make_run(tol1=0.40, tol3=0.55, tol5=0.65),
        _make_run(tol1=0.50, tol3=0.65, tol5=0.75),
    ])
    mean, std, n = project_metric(metric, s)
    expected_key = {1: "tol1", 3: "tol3", 5: "tol5"}[tol]
    if expected_key == "tol1":
        per_run = [0.40, 0.50]
    elif expected_key == "tol3":
        per_run = [0.55, 0.65]
    else:
        per_run = [0.65, 0.75]
    assert n == 2
    assert mean == pytest.approx(statistics.mean(per_run) * 100, rel=1e-9)
    assert std == pytest.approx(statistics.pstdev(per_run) * 100, rel=1e-9)


def test_project_single_run_yields_zero_std():
    """The aggregator must NOT crash on single-run sweeps; pstdev([x]) == 0."""
    s = _make_summary([_make_run(correct=10, incorrect=10)])
    mean, std, n = project_metric("category_acc", s)
    assert n == 1
    assert mean == pytest.approx(50.0, rel=1e-9)
    assert std == 0.0


def test_project_missing_individual_runs_raises_missing():
    with pytest.raises(MetricMissing, match="individual_run_summaries"):
        project_metric("step_index_acc", {"num_runs": 0})


def test_project_empty_runs_raises_missing():
    with pytest.raises(MetricMissing):
        project_metric("step_index_acc",
                       {"individual_run_summaries": []})


def test_project_step_index_acc_skips_run_with_zero_cases():
    s = _make_summary([
        _make_run(correct_step=10, incorrect_step=20),
        {"Correct step number predictions": 0,
         "Incorrect step number predictions": 0,  # contributes nothing
         "Correct cases": 0, "Incorrect cases": 0,
         "Overall average distance": 0.0},
    ])
    mean, std, n = project_metric("step_index_acc", s)
    assert n == 1
    assert mean == pytest.approx(10/30 * 100, rel=1e-9)


def test_project_unknown_metric_raises_missing_not_keyerror():
    with pytest.raises(MetricMissing, match="no projection implemented"):
        project_metric("agent_acc", _make_summary([_make_run()]))


def test_project_avg_step_distance_missing_key_in_all_runs_raises():
    s = _make_summary([
        {"Correct cases": 1, "Incorrect cases": 0,
         "Correct step number predictions": 1, "Incorrect step number predictions": 0},
    ])
    with pytest.raises(MetricMissing, match="Overall average distance"):
        project_metric("avg_step_distance", s)


# --------------------------------------------------------------------------
# load_judge_summary
# --------------------------------------------------------------------------


def test_load_judge_summary_returns_none_when_missing(tmp_path):
    assert load_judge_summary(tmp_path) is None


def test_load_judge_summary_returns_none_when_corrupt(tmp_path):
    p = tmp_path / "judge_output" / "analysis" / "summary.json"
    p.parent.mkdir(parents=True)
    p.write_text("not json {")
    assert load_judge_summary(tmp_path) is None


def test_load_judge_summary_roundtrips(tmp_path):
    s = _make_summary([_make_run()])
    _write_summary(tmp_path, s)
    got = load_judge_summary(tmp_path)
    assert got == s


# --------------------------------------------------------------------------
# aggregate_invocation: per-invocation join + status
# --------------------------------------------------------------------------


def test_aggregate_invocation_status_missing_run_dir(tmp_path):
    claim = get_claim("t4_tau_full_step")
    reports = aggregate_invocation(
        invocation_slug="slug",
        run_dir=tmp_path / "does_not_exist",
        claims=[claim],
    )
    assert len(reports) == 1
    assert reports[0].status == "MISSING_RUN_DIR"
    assert reports[0].observed_mean is None
    assert reports[0].cell_id == "t4_tau_full_step"


def test_aggregate_invocation_status_missing_summary(tmp_path):
    claim = get_claim("t4_tau_full_step")
    rd = tmp_path / "rd"
    rd.mkdir()
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd, claims=[claim],
    )
    assert reports[0].status == "MISSING_SUMMARY"


def test_aggregate_invocation_status_metric_missing(tmp_path):
    # A claim whose metric has no projection (agent_acc isn't in catalog
    # today; we synthesise the situation by writing a summary that lacks
    # the keys avg_step_distance needs).
    claim = get_claim("t5_tau_oneshot_baseline_dist")  # avg_step_distance
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _make_summary([
        {"Correct cases": 5, "Incorrect cases": 5,
         "Correct step number predictions": 2,
         "Incorrect step number predictions": 8},
    ]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd, claims=[claim],
    )
    assert reports[0].status == "METRIC_MISSING"
    assert "Overall average distance" in reports[0].notes


def test_aggregate_invocation_ok_within_paper_std(tmp_path):
    """Paper t4_tau_full_step says 48.3 +/- 0.0; we feed a summary that
    matches exactly and assert the report says OK + within_paper_std=True."""
    claim = get_claim("t4_tau_full_step")  # paper_value = 48.3 +/- 0.0
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _make_summary([
        _make_run(correct_step=483, incorrect_step=517),
        _make_run(correct_step=483, incorrect_step=517),
        _make_run(correct_step=483, incorrect_step=517),
    ]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd, claims=[claim],
    )
    r = reports[0]
    assert r.status == "OK"
    assert r.n_runs == 3
    assert r.observed_mean == pytest.approx(48.3, rel=1e-9)
    assert r.observed_std == pytest.approx(0.0, abs=1e-9)
    assert r.abs_delta == pytest.approx(0.0, abs=1e-9)
    assert r.within_paper_std is True


def test_aggregate_invocation_ok_outside_paper_std():
    """Catalog claim t4_tau_baseline_step is 32.2 +/- 3.2 -- a 10-pt miss
    must come back within_paper_std=False."""
    claim = get_claim("t4_tau_baseline_step")
    # In-memory only: no need to touch disk if we drive the projection
    # directly through aggregate_invocation; instead we test the delta math
    # via the public surface using a temp run_dir.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "rd"
        rd.mkdir()
        # Per-run step acc = 22.0% -> abs_delta = 10.2 -> outside 3.2
        _write_summary(rd, _make_summary([
            _make_run(correct_step=22, incorrect_step=78),
            _make_run(correct_step=22, incorrect_step=78),
        ]))
        reports = aggregate_invocation(
            invocation_slug="slug", run_dir=rd, claims=[claim],
        )
    r = reports[0]
    assert r.status == "OK"
    assert r.observed_mean == pytest.approx(22.0, rel=1e-9)
    assert r.abs_delta == pytest.approx(10.2, abs=1e-9)
    assert r.within_paper_std is False


def test_aggregate_invocation_within_paper_std_is_none_when_paper_std_none(tmp_path):
    # t5_magentic_oneshot_baseline_step paper = 31.8 with std=None.
    claim = get_claim("t5_magentic_oneshot_baseline_step")
    assert claim.paper_value is not None
    assert claim.paper_value.std is None
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _make_summary([
        _make_run(correct_step=318, incorrect_step=682),
    ]))
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd, claims=[claim],
    )
    r = reports[0]
    assert r.status == "OK"
    assert r.within_paper_std is None
    assert r.abs_delta == pytest.approx(0.0, abs=1e-9)


def test_aggregate_invocation_emits_one_report_per_claim(tmp_path):
    """An invocation that covers step + cat + dist must produce three
    distinct reports."""
    rd = tmp_path / "rd"
    rd.mkdir()
    _write_summary(rd, _make_summary([_make_run()]))
    claims = [
        get_claim("t5_tau_oneshot_baseline_step"),
        get_claim("t5_tau_oneshot_baseline_cat"),
        get_claim("t5_tau_oneshot_baseline_dist"),
    ]
    reports = aggregate_invocation(
        invocation_slug="slug", run_dir=rd, claims=claims,
    )
    assert [r.cell_id for r in reports] == [c.cell_id for c in claims]
    assert all(r.status == "OK" for r in reports)
    # Three different metrics -> three different observed values.
    metrics = {r.metric: r.observed_mean for r in reports}
    assert set(metrics.keys()) == {"step_index_acc", "category_acc", "avg_step_distance"}


# --------------------------------------------------------------------------
# aggregate_sweep: end-to-end through a real manifest
# --------------------------------------------------------------------------


def _fake_repo(tmp_path: Path) -> Path:
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text("# stub\n")
    return base


def test_aggregate_sweep_joins_manifest_to_summaries(tmp_path):
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

    # Populate every invocation with a summary so every cell projects.
    for entry in manifest["invocations"]:
        rd = Path(entry["run_dir"])
        _write_summary(rd, _make_summary([
            _make_run(correct=30, incorrect=70, correct_step=40, incorrect_step=60,
                      overall_distance=4.5),
        ]))

    reports = aggregate_sweep(runs_root / "manifest.json")
    # Every catalog claim in the selection must appear EXACTLY once.
    assert sorted(r.cell_id for r in reports) == sorted(c.cell_id for c in claims)
    assert all(r.status == "OK" for r in reports)
    # Reports must be cell_id-sorted for stable diffs.
    assert [r.cell_id for r in reports] == sorted(r.cell_id for r in reports)


def test_aggregate_sweep_marks_partial_completion(tmp_path):
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
    # Only the first invocation gets a summary; the rest are missing.
    first = manifest["invocations"][0]
    _write_summary(Path(first["run_dir"]), _make_summary([_make_run()]))

    reports = aggregate_sweep(runs_root / "manifest.json")
    statuses = {r.status for r in reports}
    assert "OK" in statuses
    assert "MISSING_SUMMARY" in statuses or "MISSING_RUN_DIR" in statuses
    # Reports for the populated invocation are OK; reports for the others
    # have observed=None.
    by_invocation = {}
    for r in reports:
        by_invocation.setdefault(r.invocation_slug, []).append(r)
    ok_invocation = first["slug"]
    assert all(r.status == "OK" for r in by_invocation[ok_invocation])
    for slug, rs in by_invocation.items():
        if slug == ok_invocation:
            continue
        assert all(r.status in ("MISSING_RUN_DIR", "MISSING_SUMMARY") for r in rs)
        assert all(r.observed_mean is None for r in rs)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_report_to_json_is_stable_and_typed():
    reports = [
        CellReport(
            cell_id="b", table_label="tab:x", domain="tau", metric="category_acc",
            paper_mean=10.0, paper_std=1.0, unit="percent",
            run_dir="/r", invocation_slug="s", status="OK",
            observed_mean=11.0, observed_std=0.5, n_runs=3,
            abs_delta=1.0, within_paper_std=True,
        ),
        CellReport(
            cell_id="a", table_label="tab:x", domain="tau", metric="step_index_acc",
            paper_mean=20.0, paper_std=None, unit="percent",
            run_dir="/r", invocation_slug="s", status="MISSING_SUMMARY",
        ),
    ]
    payload = json.loads(report_to_json(reports))
    assert payload["schema_version"] == 1
    assert payload["n_reports"] == 2
    assert payload["n_ok"] == 1
    assert [r["cell_id"] for r in payload["reports"]] == ["a", "b"]


def test_render_markdown_has_stable_header():
    md = render_cell_reports_markdown([
        CellReport(
            cell_id="t4_tau_full_step", table_label="tab:static-dynamic",
            domain="tau", metric="step_index_acc",
            paper_mean=48.3, paper_std=0.0, unit="percent",
            run_dir="/r", invocation_slug="s", status="OK",
            observed_mean=48.3, observed_std=0.0, n_runs=3,
            abs_delta=0.0, within_paper_std=True,
        ),
    ])
    lines = md.splitlines()
    assert lines[0].startswith("| cell_id | table | domain | metric |")
    assert lines[1].startswith("|---|")
    assert "t4_tau_full_step" in lines[2]
    assert "48.30 +/- 0.00" in lines[2]
    assert "yes" in lines[2]


# --------------------------------------------------------------------------
# CLI smoke
# --------------------------------------------------------------------------


def test_cli_report_help_lists_format_choices():
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", "report", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert cp.returncode == 0, cp.stderr
    for flag in ("--manifest", "--format", "summary", "json", "markdown"):
        assert flag in cp.stdout, f"flag/choice {flag!r} missing from report --help"


def test_cli_report_summary_after_populated_sweep(tmp_path):
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
    for entry in manifest["invocations"]:
        _write_summary(Path(entry["run_dir"]),
                       _make_summary([_make_run()]))
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", "report",
         "--manifest", str(runs_root / "manifest.json"),
         "--format", "summary"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert cp.returncode == 0, cp.stderr
    assert "Aggregator summary" in cp.stdout
    assert "OK:" in cp.stdout
    # All 8 Table 4 cells should be OK.
    assert "OK:                  8" in cp.stdout


def test_cli_report_json_format_emits_valid_payload(tmp_path):
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
    for entry in manifest["invocations"]:
        _write_summary(Path(entry["run_dir"]),
                       _make_summary([_make_run()]))
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", "report",
         "--manifest", str(runs_root / "manifest.json"),
         "--format", "json"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout)
    assert payload["schema_version"] == 1
    assert payload["n_reports"] == len(claims)
    assert all("paper_mean" in r for r in payload["reports"])
