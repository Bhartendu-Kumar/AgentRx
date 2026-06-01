"""--subset-ids CLI + ID-filter integration (W5a).

The subset filter has three observable surfaces that must stay in lockstep:

  1. ``run._load_subset_ids`` parses the shipped JSON shape (matches
     ``data/ground_truth/magentic_star_ids.json``) and a bare-list fallback.
  2. ``run.py --subset-ids PATH`` is accepted by argparse and surfaced
     through to ``run_pipeline``'s provenance write.
  3. ``claim_to_cmdline`` emits ``--subset-ids`` exactly for the claims
     whose ``subset_ids_file()`` is non-None (today: only the Magentic*
     rows of Table 5). Other claims must NOT emit the flag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# run.py is not a package, so import via path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import run  # noqa: E402

from agentrx.reproduction import ALL_CLAIMS  # noqa: E402
from agentrx.reproduction.paper_claims import get_claim  # noqa: E402
from agentrx.reproduction.cli import claim_to_cmdline  # noqa: E402


# --------------------------------------------------------------------------
# _load_subset_ids
# --------------------------------------------------------------------------


def test_load_subset_ids_accepts_shipped_object_shape(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({
        "description": "test",
        "count": 2,
        "ids": [
            {"trajectory_id": "a", "substeps": 5},
            {"trajectory_id": "b", "substeps": 8},
        ],
    }))
    got = run._load_subset_ids(str(p))
    assert got == {"a", "b"}


def test_load_subset_ids_accepts_object_shape_with_string_entries(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"ids": ["a", "b", "c"]}))
    assert run._load_subset_ids(str(p)) == {"a", "b", "c"}


def test_load_subset_ids_accepts_bare_list(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps(["a", "b", "c"]))
    assert run._load_subset_ids(str(p)) == {"a", "b", "c"}


def test_load_subset_ids_rejects_unknown_shape(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"oops": "not an ids list"}))
    with pytest.raises(ValueError, match="expected either a list of ids"):
        run._load_subset_ids(str(p))


def test_load_subset_ids_rejects_unknown_entry(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"ids": [{"oops": "no trajectory_id"}]}))
    with pytest.raises(ValueError, match="trajectory_id"):
        run._load_subset_ids(str(p))


def test_load_subset_ids_rejects_empty_object_form(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"ids": []}))
    with pytest.raises(ValueError, match="empty"):
        run._load_subset_ids(str(p))


def test_load_subset_ids_rejects_empty_list_form(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="empty"):
        run._load_subset_ids(str(p))


def test_load_subset_ids_on_shipped_magentic_star_file():
    """Pin contract: the shipped Magentic* file loads cleanly and returns 27 ids."""
    shipped = REPO_ROOT / "data" / "ground_truth" / "magentic_star_ids.json"
    if not shipped.exists():
        pytest.skip(f"shipped subset file not present: {shipped}")
    ids = run._load_subset_ids(str(shipped))
    assert isinstance(ids, set)
    assert len(ids) == 27, (
        "Magentic* paper subset must be exactly 27 trajectory ids "
        "(see eval.tex tab:ablations Magentic* row band)."
    )
    # Every id must be a non-empty string.
    assert all(isinstance(x, str) and x for x in ids)


# --------------------------------------------------------------------------
# claim_to_cmdline integration
# --------------------------------------------------------------------------


def _argv_for(cell_id: str) -> list[str]:
    return claim_to_cmdline(get_claim(cell_id), input_path="/tmp/x.json")


def test_cmdline_emits_subset_ids_for_every_magentic_star_claim():
    star_claims = [c for c in ALL_CLAIMS if c.domain == "magentic_star"]
    assert star_claims, (
        "regression guard: the catalog must continue to cover at least one "
        "Magentic* claim once Magentic* is added to ALL_CLAIMS"
    )
    for c in star_claims:
        argv = claim_to_cmdline(c, input_path="/tmp/x.json")
        assert "--subset-ids" in argv, (
            f"claim {c.cell_id} has subset_ids_file={c.subset_ids_file()!r} "
            "but cmdline omits --subset-ids"
        )
        i = argv.index("--subset-ids")
        assert argv[i + 1] == c.subset_ids_file(), (
            f"claim {c.cell_id}: cmdline subset path != catalog subset path"
        )


def test_cmdline_omits_subset_ids_for_non_subset_claims():
    for c in ALL_CLAIMS:
        if c.subset_ids_file() is not None:
            continue
        argv = claim_to_cmdline(c, input_path="/tmp/x.json")
        assert "--subset-ids" not in argv, (
            f"claim {c.cell_id} has no subset but cmdline emits --subset-ids"
        )


# --------------------------------------------------------------------------
# run.py argparse smoke
# --------------------------------------------------------------------------


def test_run_py_help_lists_subset_ids():
    """run.py --help must surface --subset-ids so the contract is discoverable."""
    cp = subprocess.run(
        [sys.executable, "run.py", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert cp.returncode == 0, cp.stderr
    assert "--subset-ids" in cp.stdout
