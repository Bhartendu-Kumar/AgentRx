"""The judge stage's output contract is invariant under ``num_runs``.

Pre-fix, ``run.py::run_judge`` only invoked
``judge_module.create_aggregate_summary`` when ``config.num_runs > 1``.
That made ``judge_output/analysis/summary.json`` -- the canonical
aggregate file -- a function of a knob it has no business depending on,
and any single-run invocation produced a partial run dir that the R04
aggregator (``agentrx.reproduction.aggregator.load_judge_summary``)
rejected with ``status=MISSING_SUMMARY``.

These tests fence three things:

  1. ``create_aggregate_summary`` is correct at n=1 (mean = sole value,
     std = 0 from the existing ``compute_stats`` guard).
  2. The n=1 aggregate round-trips cleanly through the R04 aggregator's
     loader and provenance extractor.
  3. The call inside ``run.py::run_judge`` is NOT guarded by any
     ``num_runs``-comparing ``if`` -- the AST contract pin prevents
     accidental reintroduction of the guard.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Fixture: a minimal but schema-valid run{N}.json that
# ``load_and_analyze_run_for_metrics`` can consume via its
# ``'summary' in data`` short path.
# --------------------------------------------------------------------------


def _make_run_summary(
    *,
    model_name: str = "gpt-5",
    api_version: str = "2025-01-01-preview",
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    exec_time_sec: float = 1.5,
) -> dict:
    """Mirror the per-run ``summary`` block judge.analysis() writes today."""
    return {
        "model_name": model_name,
        "api_version": api_version,
        "Correct cases": 1,
        "Incorrect cases": 0,
        "Average distance for correct cases": 0.0,
        "Average distance for incorrect cases": 0,
        "Overall average distance": 0.0,
        "Normalized average distance for correct cases": 0.0,
        "Normalized average distance for incorrect cases": 0,
        "Normalized overall average distance": 0.0,
        "Correct step number predictions": 1,
        "Incorrect step number predictions": 0,
        "Step number accuracy": 1.0,
        "Step accuracy within +-1": 1.0,
        "Step accuracy within +-2": 1.0,
        "Step accuracy within +-3": 1.0,
        "Step accuracy within +-4": 1.0,
        "Step accuracy within +-5": 1.0,
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "total_execution_time_sec": exec_time_sec,
    }


def _write_run_file(judge_out_dir: Path, run_number: int, summary: dict) -> Path:
    runs_dir = judge_out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    p = runs_dir / f"run{run_number}.json"
    p.write_text(json.dumps({"summary": summary, "detailed_results": []}))
    return p


# --------------------------------------------------------------------------
# (1) Behavioral: create_aggregate_summary is correct at n=1
# --------------------------------------------------------------------------


def test_create_aggregate_summary_writes_file_at_n1(tmp_path, capsys):
    from agentrx.judge import judge as judge_module

    judge_out_dir = tmp_path / "judge_output"
    _write_run_file(judge_out_dir, 1, _make_run_summary(
        prompt_tokens=11707, output_tokens=3588, exec_time_sec=117.8934,
    ))

    judge_module.create_aggregate_summary(str(judge_out_dir), 1)

    summary_path = judge_out_dir / "analysis" / "summary.json"
    assert summary_path.is_file(), (
        "create_aggregate_summary(n=1) must write "
        "judge_output/analysis/summary.json -- it is the judge stage's "
        "output contract, not an n>1 luxury"
    )

    payload = json.loads(summary_path.read_text())
    assert payload["num_runs"] == 1
    assert len(payload["individual_run_summaries"]) == 1
    irs = payload["individual_run_summaries"][0]
    assert irs["model_name"] == "gpt-5"
    assert irs["api_version"] == "2025-01-01-preview"
    assert irs["total_prompt_tokens"] == 11707
    assert irs["total_output_tokens"] == 3588
    assert irs["total_execution_time_sec"] == pytest.approx(117.8934)

    # aggregate_statistics must be present and well-formed at n=1.
    agg = payload["aggregate_statistics"]
    assert agg["overall_correct_cases"] == 1
    assert agg["overall_incorrect_cases"] == 0
    assert agg["overall_accuracy"] == 1.0

    # compute_stats at n=1 must yield std/variance == 0.0 (NOT raise).
    stab = payload["stability_metrics"]
    assert stab["accuracy"]["mean"] == 1.0
    assert stab["accuracy"]["std_dev"] == 0.0
    assert stab["accuracy"]["variance"] == 0.0


# --------------------------------------------------------------------------
# (2) Consumer contract: the n=1 summary round-trips through the R04
#     aggregator's loader + provenance extractor.
# --------------------------------------------------------------------------


def test_aggregate_summary_n1_round_trips_through_r04_aggregator(tmp_path):
    from agentrx.judge import judge as judge_module
    from agentrx.reproduction.aggregator import (
        extract_provenance,
        load_judge_summary,
    )

    # Mirror real run-dir layout: <run_dir>/judge_output/runs/run1.json
    run_dir = tmp_path / "g4_like_run"
    judge_out_dir = run_dir / "judge_output"
    _write_run_file(judge_out_dir, 1, _make_run_summary(
        model_name="gpt-5",
        prompt_tokens=11707,
        output_tokens=3588,
        exec_time_sec=117.8934,
    ))

    judge_module.create_aggregate_summary(str(judge_out_dir), 1)

    # R04 loader points at <run_dir>, walks down to judge_output/analysis/summary.json.
    summary = load_judge_summary(run_dir)
    assert summary is not None, (
        "load_judge_summary must consume the n=1 aggregate the judge stage "
        "wrote -- otherwise every single-run cell would be MISSING_SUMMARY"
    )

    prov = extract_provenance(summary)
    assert prov.n_runs == 1
    assert prov.model_name == "gpt-5"
    assert prov.api_version == "2025-01-01-preview"
    assert prov.total_prompt_tokens == 11707
    assert prov.total_output_tokens == 3588
    assert prov.total_execution_time_sec == pytest.approx(117.8934)
    assert prov.model_homogeneous is True
    assert prov.distinct_model_names == ["gpt-5"]


# --------------------------------------------------------------------------
# (3) Source-level contract: run.py::run_judge calls
#     create_aggregate_summary unconditionally w.r.t. num_runs.
# --------------------------------------------------------------------------


def _find_function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _references_num_runs(test_expr: ast.AST) -> bool:
    """True if the expression mentions a ``num_runs`` attribute or name."""
    for n in ast.walk(test_expr):
        if isinstance(n, ast.Attribute) and n.attr == "num_runs":
            return True
        if isinstance(n, ast.Name) and n.id == "num_runs":
            return True
    return False


def _enclosing_if_tests_for_call(
    func: ast.FunctionDef, target_callable_name: str
) -> list[ast.expr]:
    """Return the ``If.test`` of every ``If`` inside ``func`` whose body
    (or orelse) transitively contains a call to ``target_callable_name``.

    ``target_callable_name`` matches the trailing attribute of the call
    (so ``judge_module.create_aggregate_summary(...)`` matches
    ``"create_aggregate_summary"``).
    """
    def _contains_target_call(node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                callee = n.func
                if isinstance(callee, ast.Attribute) and callee.attr == target_callable_name:
                    return True
                if isinstance(callee, ast.Name) and callee.id == target_callable_name:
                    return True
        return False

    enclosing: list[ast.expr] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        body_has = any(_contains_target_call(s) for s in node.body)
        orelse_has = any(_contains_target_call(s) for s in node.orelse)
        if body_has or orelse_has:
            enclosing.append(node.test)
    return enclosing


def test_run_judge_calls_create_aggregate_summary_unconditionally():
    """The R07 guard must not return: ``create_aggregate_summary`` is
    called from ``run_judge`` regardless of ``num_runs``."""
    run_py = REPO_ROOT / "run.py"
    tree = ast.parse(run_py.read_text())
    run_judge = _find_function_def(tree, "run_judge")

    # Sanity: the call exists at all.
    callee_names = []
    for n in ast.walk(run_judge):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            callee_names.append(n.func.attr)
    assert "create_aggregate_summary" in callee_names, (
        "run_judge no longer calls create_aggregate_summary -- the judge "
        "stage's output contract is broken"
    )

    # Contract: no enclosing If's test may reference num_runs.
    enclosing_tests = _enclosing_if_tests_for_call(
        run_judge, "create_aggregate_summary"
    )
    bad = [ast.unparse(t) for t in enclosing_tests if _references_num_runs(t)]
    assert not bad, (
        "run_judge guards create_aggregate_summary with a num_runs check: "
        f"{bad}. That guard (R07) made judge_output/analysis/summary.json "
        "missing for --num-runs 1 runs. The aggregate file is the judge "
        "stage's output contract, not an n>1 luxury -- remove the guard."
    )
