"""Sanity tests for the sweep driver scripts under ``scripts/sweeps/``.

We do not run the sweeps here — that would launch the full pipeline against
real datasets and LLM endpoints. We assert only the structural contracts
that, if broken, would make a sweep silently process the wrong trajectories:

  - ``run_magentic44.sh`` exists, is executable, and the Python one-liner
    it embeds (``from agentrx.pipeline.globals import MAGENTIC_TASK_IDS``)
    produces exactly the IDs the catalog expects (44 unique, no blanks).
  - ``run_magentic27.sh`` exists and is executable.
  - The dataset on disk contains a JSON file for every MAGENTIC_TASK_IDS
    entry. A drift between the global and the dataset is a setup bug.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEPS = REPO_ROOT / "scripts" / "sweeps"


@pytest.mark.parametrize("script", [
    "run_magentic44.sh",
    "run_magentic27.sh",
    "run_tau29.sh",
])
def test_sweep_script_exists_and_is_executable(script):
    p = SWEEPS / script
    assert p.is_file(), f"missing sweep script: {p}"
    assert os.access(p, os.X_OK), f"sweep script not executable: {p}"


def test_magentic44_script_id_extraction_matches_globals():
    """The script extracts IDs by running:
       python -c "from agentrx.pipeline.globals import MAGENTIC_TASK_IDS; [print(t) for t in MAGENTIC_TASK_IDS]"
    Verify this one-liner yields the same list the code knows about."""
    from agentrx.pipeline.globals import MAGENTIC_TASK_IDS

    cp = subprocess.run(
        [sys.executable, "-c",
         "from agentrx.pipeline.globals import MAGENTIC_TASK_IDS; "
         "[print(t) for t in MAGENTIC_TASK_IDS]"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    assert cp.returncode == 0, cp.stderr
    out_ids = [line for line in cp.stdout.splitlines() if line.strip()]
    assert out_ids == list(MAGENTIC_TASK_IDS)
    assert len(out_ids) == 44, f"expected 44 magentic ids, got {len(out_ids)}"
    assert len(set(out_ids)) == len(out_ids), "duplicate IDs in MAGENTIC_TASK_IDS"


def test_magentic44_script_references_correct_dataset_and_ground_truth():
    """If somebody refactors the script's --domain or --ground-truth argument,
    the catalog will silently start producing the wrong numbers. Pin the
    invocation shape."""
    script_text = (SWEEPS / "run_magentic44.sh").read_text()
    assert "--domain magentic" in script_text
    assert "data/ground_truth/magentic_one_ground_truth.json" in script_text
    assert "data/magentic_dataset/" in script_text
    # Idempotency check is on run3.json (n=3 paper default).
    assert "run3.json" in script_text
    # Run-name prefix distinguishes m44 from m27 outputs.
    assert "azure_m44_" in script_text


def test_magentic_task_ids_have_corresponding_dataset_files():
    """Every entry in MAGENTIC_TASK_IDS must have a json file on disk under
    data/magentic_dataset/. Drift between the global and the dataset means
    the m44 sweep will fail mid-run on whichever ID is missing."""
    from agentrx.pipeline.globals import MAGENTIC_TASK_IDS

    dataset_dir = REPO_ROOT / "data" / "magentic_dataset"
    if not dataset_dir.is_dir():
        pytest.skip(f"{dataset_dir} not present in this checkout")
    missing = [
        tid for tid in MAGENTIC_TASK_IDS
        if not (dataset_dir / f"{tid}.json").is_file()
    ]
    assert not missing, (
        f"{len(missing)} of {len(MAGENTIC_TASK_IDS)} MAGENTIC_TASK_IDS lack "
        f"a corresponding .json under data/magentic_dataset/: {missing[:5]}..."
    )
