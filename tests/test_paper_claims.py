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


# --------------------------------------------------------------------------
# Table 5 (tab:ablations)
# --------------------------------------------------------------------------


_T5_DOMAIN_DYNAMIC_PAIRS = {
    ("tau", "oneshot"),
    ("tau", "stepbystep"),
    ("flash", "oneshot"),
    ("flash", "stepbystep"),
    ("magentic", "oneshot"),
    ("magentic_star", "stepbystep"),
}


def test_table_5_has_expected_cell_count():
    # 6 columns x 6 row-bands x 3 metrics = 108
    t5 = claims_for(table_label="tab:ablations")
    assert len(t5) == 108, f"expected 108 cells in tab:ablations, got {len(t5)}"


def test_table_5_domain_dynamic_pairs_are_exactly_the_paper_six():
    t5 = claims_for(table_label="tab:ablations")
    pairs = {(c.domain, c.dynamic_mode) for c in t5}
    assert pairs == _T5_DOMAIN_DYNAMIC_PAIRS


def test_table_5_each_row_band_has_eighteen_cells():
    # 6 columns x 3 metrics = 18 cells per row band
    t5 = claims_for(table_label="tab:ablations")
    from collections import Counter
    counts = Counter((c.domain, c.dynamic_mode) for c in t5)
    for pair, n in counts.items():
        assert n == 18, f"row band {pair} has {n} cells, expected 18"


def test_table_5_metric_distribution_is_balanced():
    t5 = claims_for(table_label="tab:ablations")
    from collections import Counter
    counts = Counter(c.metric for c in t5)
    # 6 row bands x 6 columns = 36 cells per metric
    assert counts["step_index_acc"] == 36
    assert counts["category_acc"] == 36
    assert counts["avg_step_distance"] == 36


@pytest.mark.parametrize("col_slug,prompt_mode,exec_mode,with_context", [
    ("baseline",          "baseline", "violations-after", False),
    ("stepthencat",       "baseline", "stepbystep",       False),
    ("baselinevio",       "baseline", "violations-after", True),
    ("stepthencatvio",    "baseline", "stepbystep",       True),
    ("taxonomychecklist", "combined", "violations-after", False),
    ("checklistvio",      "combined", "violations-after", True),
])
def test_table_5_column_axis_is_well_defined(col_slug, prompt_mode, exec_mode, with_context):
    """Every cell in column ``col_slug`` must have the same (prompt_mode,
    exec_mode, with_context) triple."""
    t5 = [c for c in claims_for(table_label="tab:ablations")
          if c.cell_id.endswith(f"_{col_slug}_step")
          or c.cell_id.endswith(f"_{col_slug}_cat")
          or c.cell_id.endswith(f"_{col_slug}_dist")]
    assert len(t5) == 18, f"column {col_slug}: expected 18 cells, got {len(t5)}"
    for c in t5:
        assert c.run_config.prompt_mode == prompt_mode, c.cell_id
        assert c.run_config.exec_mode == exec_mode, c.cell_id
        assert c.run_config.with_context == with_context, c.cell_id


@pytest.mark.parametrize("cell_id,expected_mean,expected_std", [
    # Sentinel cells. If we ever break the transcription, at least one of
    # these breaks first.
    ("t5_tau_oneshot_baseline_step",            32.2, 3.2),
    ("t5_tau_oneshot_checklistvio_step",        48.3, None),
    ("t5_tau_oneshot_checklistvio_cat",         39.1, 1.6),
    ("t5_tau_oneshot_stepthencatvio_step",      54.0, 1.6),  # best step-acc in paper
    ("t5_flash_stepbystep_checklistvio_cat",    60.3, 1.3),  # best cat-acc in paper
    ("t5_magentic_oneshot_taxonomychecklist_step", 31.8, None),
    ("t5_magenticstar_stepbystep_checklistvio_step", 46.9, 3.5),
])
def test_table_5_sentinel_values_match_paper(cell_id, expected_mean, expected_std):
    c = get_claim(cell_id)
    assert c.paper_value.mean == expected_mean
    assert c.paper_value.std == expected_std


def test_table_5_avg_step_distance_uses_steps_unit():
    t5 = claims_for(table_label="tab:ablations", metric="avg_step_distance")
    for c in t5:
        assert c.paper_value.unit == "steps", c.cell_id


def test_table_5_accuracy_metrics_use_percent_unit():
    t5_step = claims_for(table_label="tab:ablations", metric="step_index_acc")
    t5_cat = claims_for(table_label="tab:ablations", metric="category_acc")
    for c in t5_step + t5_cat:
        assert c.paper_value.unit == "percent", c.cell_id


def test_table_5_magentic_star_uses_subset_ids_file():
    star = [c for c in claims_for(table_label="tab:ablations")
            if c.domain == "magentic_star"]
    assert len(star) == 18
    for c in star:
        assert c.subset_ids_file() == "data/ground_truth/magentic_star_ids.json"


def test_table_5_magentic_full_does_not_use_subset_ids_file():
    full = [c for c in claims_for(table_label="tab:ablations")
            if c.domain == "magentic"]
    assert len(full) == 18
    for c in full:
        assert c.subset_ids_file() is None


def test_table_4_dynamic_mode_is_oneshot():
    """The W2b correction: cross-table numeric evidence shows Tab 4 uses
    one-shot constraint generation, not the CLI default stepbystep."""
    t4 = claims_for(table_label="tab:static-dynamic")
    for c in t4:
        assert c.dynamic_mode == "oneshot", c.cell_id


def test_total_catalog_size_is_table_4_plus_table_5():
    assert len(ALL_CLAIMS) == 8 + 108


def test_cli_cmd_for_table_5_threads_dynamic_mode():
    cp = _run_cli("cmd", "t5_magenticstar_stepbystep_checklistvio_cat",
                  "--input", "/tmp/x.json")
    out = cp.stdout.strip()
    assert "--dynamic-mode stepbystep" in out
    assert "--prompt-mode combined" in out
    assert "--exec-mode violations-after" in out
    # magentic_star -> domain flag is "magentic"
    assert "--domain magentic" in out
