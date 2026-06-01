"""Provenance dump must round-trip RunConfig, model spec, file hashes (W1b)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from agentrx.pipeline.profiles import PAPER_DEFAULT, RunConfig
from agentrx.pipeline.provenance import (
    PROVENANCE_FILENAME,
    SCHEMA_VERSION,
    build_provenance,
    dump_provenance,
    load_provenance,
)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_build_provenance_has_required_top_level_keys():
    payload = build_provenance(
        input_path="/nonexistent/input.json",
        domain="tau",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir", "static", "dynamic", "check", "judge", "report"],
    )
    required = {
        "schema_version",
        "timestamp_utc",
        "input",
        "domain",
        "endpoint",
        "ground_truth",
        "stages_planned",
        "stages_completed",
        "run_config",
        "model_spec",
        "package",
        "data_provenance",
        "observed_model_snapshot",
        "extra",
    }
    assert required.issubset(payload.keys())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["domain"] == "tau"
    assert payload["endpoint"] == "azure"
    assert payload["stages_completed"] == []
    assert payload["observed_model_snapshot"] is None


def test_run_config_is_serialised_as_asdict():
    payload = build_provenance(
        input_path="x",
        domain="tau",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
    )
    expected = dataclasses.asdict(PAPER_DEFAULT)
    assert payload["run_config"] == expected
    # All RunConfig fields must appear in the provenance dump (D1: nothing hidden).
    for f in dataclasses.fields(RunConfig):
        assert f.name in payload["run_config"]


def test_data_provenance_hashes_real_files(tmp_path: Path):
    input_file = tmp_path / "in.json"
    input_file.write_bytes(b'{"hello": "world"}\n')
    gt_file = tmp_path / "gt.json"
    gt_file.write_bytes(b'{"truth": 1}\n')

    payload = build_provenance(
        input_path=str(input_file),
        domain="tau",
        endpoint="azure",
        ground_truth_path=str(gt_file),
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
    )
    assert payload["data_provenance"]["input_sha256"] == _sha256(input_file.read_bytes())
    assert payload["data_provenance"]["ground_truth_sha256"] == _sha256(gt_file.read_bytes())
    assert payload["data_provenance"]["ground_truth_path"] == str(gt_file)


def test_data_provenance_handles_missing_files():
    payload = build_provenance(
        input_path="/does/not/exist.json",
        domain="tau",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
    )
    assert payload["data_provenance"]["input_sha256"] is None
    assert payload["data_provenance"]["ground_truth_sha256"] is None
    assert payload["data_provenance"]["ground_truth_path"] is None
    assert payload["ground_truth"] is None


@pytest.mark.parametrize("endpoint", ["azure", "trapi", "copilot"])
def test_model_spec_carries_endpoint_type(endpoint):
    payload = build_provenance(
        input_path="x",
        domain="tau",
        endpoint=endpoint,
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
    )
    assert payload["model_spec"]["endpoint_type"] == endpoint
    assert payload["endpoint"] == endpoint


def test_package_section_carries_python_version():
    payload = build_provenance(
        input_path="x",
        domain="tau",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
    )
    py = payload["package"]["python_version"]
    assert isinstance(py, str)
    assert len(py.split(".")) == 3


def test_dump_and_load_round_trip(tmp_path: Path):
    payload = build_provenance(
        input_path="x",
        domain="magentic",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir", "check"],
        stages_completed=["ir"],
    )
    written_path = dump_provenance(str(tmp_path), payload)
    assert written_path.endswith(PROVENANCE_FILENAME)
    loaded = load_provenance(str(tmp_path))
    assert loaded is not None
    assert loaded["domain"] == "magentic"
    assert loaded["stages_completed"] == ["ir"]
    # Raw JSON on disk parses cleanly (no trailing garbage).
    with open(written_path, "r", encoding="utf-8") as f:
        json.load(f)


def test_load_provenance_returns_none_when_missing(tmp_path: Path):
    assert load_provenance(str(tmp_path)) is None


def test_observed_model_snapshot_is_passthrough():
    snap = {"served_model": "gpt-5-2025-04-16", "system_fingerprint": "fp_abc"}
    payload = build_provenance(
        input_path="x",
        domain="tau",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
        observed_model_snapshot=snap,
    )
    assert payload["observed_model_snapshot"] == snap


def test_alternate_runconfig_is_serialised_faithfully():
    custom = dataclasses.replace(PAPER_DEFAULT, skip_nl=True, python_check_timeout_sec=5.0)
    payload = build_provenance(
        input_path="x",
        domain="tau",
        endpoint="azure",
        ground_truth_path=None,
        judge_config=custom,
        stages_planned=["check"],
    )
    assert payload["run_config"]["skip_nl"] is True
    assert payload["run_config"]["python_check_timeout_sec"] == 5.0
