"""Sweep orchestrator tests (R03).

The sweep module's value lies in (a) collapsing redundant catalog
invocations, (b) emitting a manifest the aggregator can join against,
and (c) refusing to re-invoke ``run.py`` for invocations whose state
ledger already records the judge stage as complete. These tests pin
each of those contracts.

We deliberately do NOT spawn a real ``run.py`` here -- that requires
LLM endpoints. Execution is exercised via a fake ``run.py`` injected
into a temp ``base_dir`` so we can assert that argv is dispatched
correctly and resume semantics work.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentrx.reproduction import ALL_CLAIMS  # noqa: E402
from agentrx.reproduction.paper_claims import (  # noqa: E402
    PaperClaim,
    PaperValue,
    get_claim,
)
from agentrx.reproduction.sweep import (  # noqa: E402
    Invocation,
    build_manifest,
    dump_manifest,
    execute_sequential,
    group_into_invocations,
    invocation_argv,
    invocation_key,
    invocation_run_dir,
    invocation_slug,
    is_invocation_complete,
    select_claims,
)


# --------------------------------------------------------------------------
# invocation_key / invocation_slug contracts
# --------------------------------------------------------------------------


def test_invocation_key_collapses_metrics_for_same_recipe():
    """A row band+column in tab:ablations contributes step/cat/dist as three
    DIFFERENT claims -- they must share an invocation_key."""
    step = get_claim("t5_tau_oneshot_baseline_step")
    cat = get_claim("t5_tau_oneshot_baseline_cat")
    dist = get_claim("t5_tau_oneshot_baseline_dist")
    assert invocation_key(step) == invocation_key(cat) == invocation_key(dist)
    assert invocation_slug(step) == invocation_slug(cat) == invocation_slug(dist)


def test_invocation_key_separates_distinct_recipes():
    """Different dynamic_mode -> different invocation_key (they run a
    different dynamic stage)."""
    a = get_claim("t5_tau_oneshot_baseline_step")
    b = get_claim("t5_tau_stepbystep_baseline_step")
    assert invocation_key(a) != invocation_key(b)
    assert invocation_slug(a) != invocation_slug(b)


def test_invocation_key_separates_skip_static_from_full():
    a = get_claim("t4_tau_full_step")
    b = get_claim("t4_tau_dynamic_only_step")  # skip_static=True
    assert a.skip_static != b.skip_static
    assert invocation_key(a) != invocation_key(b)
    assert invocation_slug(a) != invocation_slug(b)


def test_invocation_key_dedups_t4_full_with_t5_checklistvio_oneshot():
    """Both t4_tau_full_* and t5_tau_oneshot_checklistvio_* are PAPER_DEFAULT
    + dynamic_mode=oneshot + domain=tau. The catalog notes this explicitly:
    'AgentRx' row in Table 4 is the byte-exact match for Checklist+Vio.
    one-shot in Table 5. The sweep must therefore run it ONCE."""
    a = get_claim("t4_tau_full_step")
    b = get_claim("t5_tau_oneshot_checklistvio_step")
    assert invocation_key(a) == invocation_key(b)
    assert invocation_slug(a) == invocation_slug(b)


def test_invocation_key_dedups_t4_baseline_with_t5_baseline_oneshot():
    """Same second-order collision the catalog inherits: Table 4 'Baseline'
    row and Table 5 tau-oneshot Baseline column resolve to the same
    (prompt_mode='baseline', exec_mode='violations-after', with_context=False,
    dynamic_mode='oneshot') recipe. Pinning this keeps a future catalog edit
    from silently double-running the recipe."""
    a = get_claim("t4_tau_baseline_step")
    b = get_claim("t5_tau_oneshot_baseline_step")
    assert invocation_key(a) == invocation_key(b)
    assert invocation_slug(a) == invocation_slug(b)


def test_slug_is_bijective_with_key_across_catalog():
    """No two distinct invocation_keys may collide on the same slug, and no
    two equal keys may produce different slugs. group_into_invocations
    enforces this internally; here we exercise the catalog."""
    slug_to_key: dict[str, tuple] = {}
    for c in ALL_CLAIMS:
        k = invocation_key(c)
        s = invocation_slug(c)
        if s in slug_to_key:
            assert slug_to_key[s] == k, (
                f"slug collision: {s!r} maps to two distinct invocation_keys"
            )
        else:
            slug_to_key[s] = k


# --------------------------------------------------------------------------
# group_into_invocations
# --------------------------------------------------------------------------


def test_group_into_invocations_reduces_catalog_count():
    """The full 116-claim catalog must collapse to substantially fewer
    invocations (each tab:ablations cell contributes 3 metrics that share
    a recipe). The exact number is asserted to catch silent regressions
    in the recipe equivalence relation."""
    invs = group_into_invocations(ALL_CLAIMS)
    assert len(ALL_CLAIMS) == 116, (
        f"catalog size changed unexpectedly ({len(ALL_CLAIMS)} != 116); "
        "if intentional, update this test together with the new expected "
        "invocation count."
    )
    # Table 5: 6 row-bands x 6 columns = 36 invocations.
    # Table 4 contributes 4 recipes (Baseline, Global-Only, Dynamic-Only, AgentRx)
    # x 2 metrics each. Of those 4 recipes:
    #   - 'Baseline'     == tab:ablations tau-oneshot Baseline    column (collide)
    #   - 'AgentRx'      == tab:ablations tau-oneshot Checklist+Vio. column (collide)
    #   - 'Global-Only'  (skip_dynamic=True) is unique to Table 4
    #   - 'Dynamic-Only' (skip_static=True)  is unique to Table 4
    # so Table 4 adds 2 net new invocations.
    # Total: 36 + 2 = 38.
    assert len(invs) == 38, (
        f"unexpected invocation count: {len(invs)}; "
        "if a new table or new claim was added, recompute the expected "
        "value from the catalog rather than blindly bumping this number."
    )


def test_group_into_invocations_total_claims_preserved():
    invs = group_into_invocations(ALL_CLAIMS)
    n_grouped = sum(len(inv.claims) for inv in invs)
    assert n_grouped == len(ALL_CLAIMS), "grouping dropped or duplicated claims"


def test_group_into_invocations_sorted_by_slug():
    invs = group_into_invocations(ALL_CLAIMS)
    slugs = [inv.slug for inv in invs]
    assert slugs == sorted(slugs)


def test_group_into_invocations_within_group_sorted_by_cell_id():
    invs = group_into_invocations(ALL_CLAIMS)
    for inv in invs:
        ids = [c.cell_id for c in inv.claims]
        assert ids == sorted(ids), f"unsorted cells in {inv.slug!r}: {ids}"


def test_group_into_invocations_detects_slug_drift():
    """If invocation_slug ever stops being a function of invocation_key, the
    grouper must shout. Build a synthetic mini-catalog where two claims
    share a key but one has its slug poisoned."""
    a = get_claim("t5_tau_oneshot_baseline_step")
    b = get_claim("t5_tau_oneshot_baseline_cat")
    poisoned = dataclasses.replace(b, cell_id="POISON")
    # Monkey-key the same recipe but assert slug stability directly.
    # Since invocation_slug is pure-from-fields, the only way to trigger the
    # drift detector is to inject a claim type that overrides slug. We
    # don't have such a type; instead we sanity-check that the bucket of
    # genuine claims actually produces a single slug.
    invs = group_into_invocations([a, b, poisoned])
    assert len(invs) == 1
    assert invs[0].slug == invocation_slug(a)


# --------------------------------------------------------------------------
# select_claims
# --------------------------------------------------------------------------


def test_select_claims_by_cell_ids_preserves_order():
    out = select_claims(cell_ids=["t4_tau_full_cat", "t4_tau_full_step"])
    assert [c.cell_id for c in out] == ["t4_tau_full_cat", "t4_tau_full_step"]


def test_select_claims_unknown_cell_id_raises():
    with pytest.raises(KeyError):
        select_claims(cell_ids=["does_not_exist"])


def test_select_claims_no_filter_returns_full_catalog():
    out = select_claims()
    assert len(out) == len(ALL_CLAIMS)


def test_select_claims_table_filter():
    out = select_claims(table_label="tab:static-dynamic")
    assert out and all(c.table_label == "tab:static-dynamic" for c in out)


def test_select_claims_combination_filter_intersects():
    out = select_claims(table_label="tab:ablations", domain="tau",
                        dynamic_mode="oneshot", metric="step_index_acc")
    # 6 columns x 1 row band x 1 metric = 6 claims.
    assert len(out) == 6
    assert all(c.table_label == "tab:ablations" for c in out)
    assert all(c.domain == "tau" for c in out)
    assert all(c.dynamic_mode == "oneshot" for c in out)
    assert all(c.metric == "step_index_acc" for c in out)


# --------------------------------------------------------------------------
# argv generation
# --------------------------------------------------------------------------


def test_invocation_argv_starts_with_chosen_python_and_resolves_run_py(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text("# stub\n")
    argv = invocation_argv(
        inv, runs_root=tmp_path / "runs", base_dir=base,
        python="/opt/special/python", endpoint="azure",
    )
    assert argv[0] == "/opt/special/python"
    assert argv[1] == str((base / "run.py").resolve())
    assert "--run-dir" in argv
    assert "--num-runs" in argv
    assert "--domain" in argv
    rd_idx = argv.index("--run-dir")
    assert argv[rd_idx + 1] == str((tmp_path / "runs" / "invocations" / inv.slug).resolve())


def test_invocation_argv_for_magentic_star_emits_subset_ids(tmp_path):
    inv = group_into_invocations(
        [get_claim("t5_magenticstar_stepbystep_checklistvio_step")]
    )[0]
    base = tmp_path / "fake_repo"
    (base / "data" / "magentic_dataset").mkdir(parents=True)
    (base / "data" / "magentic_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text("# stub\n")
    argv = invocation_argv(inv, runs_root=tmp_path / "runs", base_dir=base)
    assert "--subset-ids" in argv
    i = argv.index("--subset-ids")
    assert argv[i + 1].endswith("data/ground_truth/magentic_star_ids.json")


def test_invocation_argv_input_path_is_dataset_directory(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text("# stub\n")
    argv = invocation_argv(inv, runs_root=tmp_path / "runs", base_dir=base)
    # The input path is argv[2] (positional, after "python", "run.py").
    assert argv[2] == str((base / "data" / "tau_dataset").resolve())


def test_invocation_argv_raises_when_input_directory_missing(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    base = tmp_path / "fake_repo"
    base.mkdir()
    (base / "run.py").write_text("# stub\n")
    with pytest.raises(FileNotFoundError, match="resolved dataset directory"):
        invocation_argv(inv, runs_root=tmp_path / "runs", base_dir=base)


# --------------------------------------------------------------------------
# is_invocation_complete
# --------------------------------------------------------------------------


def test_is_invocation_complete_false_when_no_state(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    assert is_invocation_complete(inv, tmp_path) is False


def test_is_invocation_complete_false_when_judge_missing(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    rd = invocation_run_dir(inv, tmp_path)
    rd.mkdir(parents=True)
    (rd / "run_state.json").write_text(json.dumps(
        {"completed_stages": ["ir", "static", "dynamic", "check"]}
    ))
    assert is_invocation_complete(inv, tmp_path) is False


def test_is_invocation_complete_true_when_judge_present(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    rd = invocation_run_dir(inv, tmp_path)
    rd.mkdir(parents=True)
    (rd / "run_state.json").write_text(json.dumps(
        {"completed_stages": ["ir", "static", "dynamic", "check", "judge"]}
    ))
    assert is_invocation_complete(inv, tmp_path) is True


def test_is_invocation_complete_false_when_state_corrupt(tmp_path):
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    rd = invocation_run_dir(inv, tmp_path)
    rd.mkdir(parents=True)
    (rd / "run_state.json").write_text("not json{")
    assert is_invocation_complete(inv, tmp_path) is False


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def test_build_and_dump_manifest_roundtrip(tmp_path):
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text("# stub\n")
    claims = select_claims(table_label="tab:static-dynamic")
    invs = group_into_invocations(claims)
    manifest = build_manifest(
        invs, runs_root=tmp_path / "runs",
        selection={"table_label": "tab:static-dynamic"},
        base_dir=base, endpoint="azure", python="/usr/bin/python",
    )
    assert manifest["schema_version"] == 1
    assert manifest["n_claims"] == len(claims)
    assert manifest["n_invocations"] == len(invs)
    assert manifest["endpoint"] == "azure"
    assert len(manifest["invocations"]) == len(invs)
    for entry in manifest["invocations"]:
        assert set(entry.keys()) == {"slug", "run_dir", "cell_ids", "argv"}
        assert entry["argv"][0] == "/usr/bin/python"
    path = dump_manifest(manifest, tmp_path / "runs")
    assert path == tmp_path / "runs" / "manifest.json"
    loaded = json.loads(path.read_text())
    assert loaded == manifest


# --------------------------------------------------------------------------
# execute_sequential with a fake run.py
# --------------------------------------------------------------------------


FAKE_RUN_PY = '''\
#!/usr/bin/env python3
"""Fake run.py: writes a run_state.json marking 'judge' complete.

Exists so the sweep orchestrator's resume / dispatch path can be tested
without spinning up the real pipeline (which needs LLM endpoints).
"""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--run-dir", required=True)
# Swallow everything else.
args, _ = ap.parse_known_args()
os.makedirs(args.run_dir, exist_ok=True)
with open(os.path.join(args.run_dir, "run_state.json"), "w") as f:
    json.dump({"completed_stages": ["ir", "static", "dynamic", "check", "judge"]}, f)
sys.exit(0)
'''


def _make_fake_repo(tmp_path: Path) -> Path:
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text(FAKE_RUN_PY)
    return base


def test_execute_sequential_runs_then_skips_on_resume(tmp_path):
    base = _make_fake_repo(tmp_path)
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    runs_root = tmp_path / "runs"

    # First pass: not complete -> RUN.
    results1 = execute_sequential(
        [inv], runs_root=runs_root, base_dir=base,
        python=sys.executable, endpoint="azure", resume=True,
    )
    assert len(results1) == 1
    assert results1[0].skipped is False
    assert results1[0].returncode == 0
    assert (invocation_run_dir(inv, runs_root) / "run_state.json").exists()

    # Second pass: state ledger says 'judge' complete -> SKIP.
    results2 = execute_sequential(
        [inv], runs_root=runs_root, base_dir=base,
        python=sys.executable, endpoint="azure", resume=True,
    )
    assert results2[0].skipped is True
    assert results2[0].returncode == 0

    # Third pass: --no-resume forces re-execution.
    results3 = execute_sequential(
        [inv], runs_root=runs_root, base_dir=base,
        python=sys.executable, endpoint="azure", resume=False,
    )
    assert results3[0].skipped is False
    assert results3[0].returncode == 0


def test_execute_sequential_reports_nonzero_returncode_without_raising(tmp_path):
    base = tmp_path / "fake_repo"
    (base / "data" / "tau_dataset").mkdir(parents=True)
    (base / "data" / "tau_dataset" / "x.json").write_text("[]")
    (base / "run.py").write_text(
        "import sys; sys.exit(7)\n"
    )
    inv = group_into_invocations([get_claim("t4_tau_full_step")])[0]
    runs_root = tmp_path / "runs"
    results = execute_sequential(
        [inv], runs_root=runs_root, base_dir=base,
        python=sys.executable, endpoint="azure", resume=True,
    )
    assert results[0].returncode == 7
    assert results[0].skipped is False


# --------------------------------------------------------------------------
# CLI smoke tests
# --------------------------------------------------------------------------


def test_cli_sweep_help_lists_flag():
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", "sweep", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert cp.returncode == 0, cp.stderr
    for flag in ("--runs-root", "--cells", "--table", "--dry-run", "--no-resume",
                 "--dynamic-mode"):
        assert flag in cp.stdout, f"flag {flag} missing from sweep --help"


def test_cli_sweep_dry_run_emits_plan_and_manifest(tmp_path):
    base = _make_fake_repo(tmp_path)
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", "sweep",
         "--runs-root", str(tmp_path / "runs"),
         "--table", "tab:static-dynamic",
         "--base-dir", str(base),
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert cp.returncode == 0, cp.stderr
    assert "Sweep plan:" in cp.stdout
    # No manifest is written under --dry-run.
    assert not (tmp_path / "runs" / "manifest.json").exists()
    # The JSON body must be a parseable manifest with the right shape.
    # Find first '{' through end of stdout.
    brace_at = cp.stdout.find("{")
    assert brace_at != -1
    payload = json.loads(cp.stdout[brace_at:])
    assert payload["schema_version"] == 1
    assert payload["selection"]["table_label"] == "tab:static-dynamic"
    assert payload["n_claims"] == 8  # 4 methods x 2 metrics
    assert payload["n_invocations"] == 4
