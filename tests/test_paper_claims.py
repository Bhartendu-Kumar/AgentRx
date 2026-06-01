"""Paper-claims catalog (W2): schema integrity and reproducibility.

The tests check three invariants:
  1. The catalog is internally consistent (unique cell_ids, every paper_value
     has a numeric mean, dataset/GT paths are mapped for every domain claimed).
  2. Every claim's run_config is a valid RunConfig that PAPER_DEFAULT could
     have produced via dataclasses.replace.
  3. claim_to_cmdline produces a CLI line that a downstream parser could parse
     without ambiguity (no missing required args).
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from agentrx.pipeline.profiles import PAPER_DEFAULT, RunConfig
from agentrx.reproduction import (
    ALL_CLAIMS,
    PaperClaim,
    PaperValue,
    cell_id_unique,
    claims_for,
)
from agentrx.reproduction.cli import claim_to_cmdline
from agentrx.reproduction.paper_claims import (
    _DATASET_GLOB,
    _GROUND_TRUTH,
    _SUBSET_IDS_FILE,
    get_claim,
)


def test_catalog_is_nonempty():
    assert len(ALL_CLAIMS) > 0


def test_cell_ids_are_unique():
    # The constructor already calls this, but it's the load-bearing invariant
    # so it gets its own test.
    cell_id_unique(ALL_CLAIMS)


def test_every_claim_has_known_domain_paths():
    for c in ALL_CLAIMS:
        assert c.domain in _DATASET_GLOB, c.cell_id
        assert c.domain in _GROUND_TRUTH, c.cell_id
        assert c.domain in _SUBSET_IDS_FILE, c.cell_id


def test_paper_values_are_finite_numbers():
    for c in ALL_CLAIMS:
        if c.paper_value is None:
            continue
        pv = c.paper_value
        assert isinstance(pv.mean, (int, float))
        assert pv.mean == pv.mean  # not NaN
        if pv.std is not None:
            assert pv.std >= 0, f"{c.cell_id}: negative std {pv.std}"


def test_every_runconfig_is_a_real_runconfig():
    for c in ALL_CLAIMS:
        assert isinstance(c.run_config, RunConfig)
        # And it must round-trip through asdict cleanly.
        d = dataclasses.asdict(c.run_config)
        rebuilt = RunConfig(**d)
        assert rebuilt == c.run_config


def test_skip_static_and_skip_dynamic_are_not_both_true():
    """A claim with both static and dynamic stages skipped has no invariants,
    which is degenerate. Catalog must not contain such a recipe."""
    for c in ALL_CLAIMS:
        assert not (c.skip_static and c.skip_dynamic), c.cell_id


def test_table_4_static_dynamic_has_expected_eight_cells():
    t4 = claims_for(table_label="tab:static-dynamic")
    assert len(t4) == 8, f"expected 8 cells in tab:static-dynamic, got {len(t4)}"
    # All tau-bench.
    assert all(c.domain == "tau" for c in t4)
    # 4 step-acc + 4 cat-acc.
    metrics = sorted(c.metric for c in t4)
    assert metrics.count("step_index_acc") == 4
    assert metrics.count("category_acc") == 4


@pytest.mark.parametrize("cell_id,method,expected_step,expected_cat", [
    ("t4_tau_baseline",     "baseline",     32.2, 25.3),
    ("t4_tau_global_only",  "global_only",  41.4, 28.7),
    ("t4_tau_dynamic_only", "dynamic_only", 43.7, 36.8),
    ("t4_tau_full",         "agentrx_full", 48.3, 39.1),
])
def test_table_4_paper_values_match_latex(cell_id, method, expected_step, expected_cat):
    step = get_claim(f"{cell_id}_step").paper_value.mean
    cat = get_claim(f"{cell_id}_cat").paper_value.mean
    assert step == expected_step, f"{cell_id} step expected {expected_step} got {step}"
    assert cat == expected_cat, f"{cell_id} cat expected {expected_cat} got {cat}"


def test_table_4_recipe_axes_distinguish_the_four_methods():
    baseline = get_claim("t4_tau_baseline_step")
    global_only = get_claim("t4_tau_global_only_step")
    dynamic_only = get_claim("t4_tau_dynamic_only_step")
    full = get_claim("t4_tau_full_step")

    # Baseline: no violations, baseline prompt mode
    assert baseline.run_config.with_context is False
    assert baseline.run_config.prompt_mode == "baseline"
    assert baseline.skip_static is False
    assert baseline.skip_dynamic is False

    # Global-Only: skip dynamic
    assert global_only.skip_dynamic is True
    assert global_only.skip_static is False
    assert global_only.run_config == PAPER_DEFAULT  # axes inherited

    # Dynamic-Only: skip static
    assert dynamic_only.skip_static is True
    assert dynamic_only.skip_dynamic is False
    assert dynamic_only.run_config == PAPER_DEFAULT

    # Full: PAPER_DEFAULT, no skipping
    assert full.skip_static is False
    assert full.skip_dynamic is False
    assert full.run_config == PAPER_DEFAULT


# --------------------------------------------------------------------------
# claim_to_cmdline
# --------------------------------------------------------------------------


def _opt_index(argv, opt):
    return argv.index(opt)


def test_cmdline_includes_all_required_arguments():
    c = get_claim("t4_tau_full_step")
    argv = claim_to_cmdline(c, input_path="/tmp/x.json", run_name="cell_test")
    # Required positional + flags.
    assert argv[0] == "python"
    assert argv[1] == "run.py"
    assert argv[2] == "/tmp/x.json"
    for required in ["--domain", "--endpoint", "--prompt-mode", "--exec-mode",
                     "--num-runs", "--prompt-style", "--dynamic-mode",
                     "--ground-truth", "--run-name"]:
        assert required in argv, f"{required} missing in cmdline"


def test_cmdline_for_skip_dynamic_claim_passes_skip_dynamic():
    c = get_claim("t4_tau_global_only_step")
    argv = claim_to_cmdline(c, input_path="/tmp/x.json")
    assert "--skip-dynamic" in argv
    assert "--skip-static" not in argv


def test_cmdline_for_skip_static_claim_passes_skip_static():
    c = get_claim("t4_tau_dynamic_only_step")
    argv = claim_to_cmdline(c, input_path="/tmp/x.json")
    assert "--skip-static" in argv
    assert "--skip-dynamic" not in argv


def test_cmdline_for_no_context_claim_passes_no_context():
    c = get_claim("t4_tau_baseline_step")
    argv = claim_to_cmdline(c, input_path="/tmp/x.json")
    assert "--no-context" in argv


def test_cmdline_for_full_claim_omits_no_context():
    c = get_claim("t4_tau_full_step")
    argv = claim_to_cmdline(c, input_path="/tmp/x.json")
    assert "--no-context" not in argv


def test_cmdline_threads_python_check_timeout_and_num_runs():
    c = get_claim("t4_tau_full_step")
    argv = claim_to_cmdline(c, input_path="/tmp/x.json")
    # --num-runs <N>
    i = _opt_index(argv, "--num-runs")
    assert argv[i + 1] == str(c.run_config.num_runs)
    # --python-check-timeout-sec <F>
    i = _opt_index(argv, "--python-check-timeout-sec")
    assert float(argv[i + 1]) == c.run_config.python_check_timeout_sec


# --------------------------------------------------------------------------
# CLI subprocess sanity (catches argparse/import wiring regressions)
# --------------------------------------------------------------------------


def _run_cli(*args, expect_zero=True) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.reproduction", *args],
        capture_output=True, text=True, timeout=30,
    )
    if expect_zero:
        assert cp.returncode == 0, f"stderr:\n{cp.stderr}\nstdout:\n{cp.stdout}"
    return cp


def test_cli_list_runs_and_prints_known_cell():
    cp = _run_cli("list")
    assert "t4_tau_full_cat" in cp.stdout


def test_cli_show_emits_valid_json():
    cp = _run_cli("show", "t4_tau_full_cat")
    payload = json.loads(cp.stdout)
    assert payload["cell_id"] == "t4_tau_full_cat"
    assert payload["paper_value"]["mean"] == 39.1
    assert payload["run_config"]["prompt_mode"] == PAPER_DEFAULT.prompt_mode


def test_cli_cmd_emits_python_runpy_invocation():
    cp = _run_cli("cmd", "t4_tau_full_cat", "--input", "/tmp/x.json")
    out = cp.stdout.strip()
    assert out.startswith("python run.py /tmp/x.json")
    assert "--prompt-mode combined" in out
    assert "--ground-truth" in out
