"""Single-file dump of *everything* that determined a run.

The premise (deficit D1): if you cannot reconstruct a run from one artifact,
you cannot meaningfully reproduce the paper. ``runs/<name>/run_config.json``
is that artifact. It is written immediately after the run directory is
created (so a crashed run still leaves behind enough information to bisect)
and rewritten at pipeline end with ``stages_completed`` filled in and any
observed model-snapshot the LLM client recorded.

Schema version is explicit; consumers must check it. The schema is
deliberately flat-ish (one nested level for grouped concerns) so JSON diffs
between two runs read cleanly.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import agentrx.pipeline.globals as g
from agentrx.pipeline.profiles import RunConfig

PROVENANCE_FILENAME = "run_config.json"
SCHEMA_VERSION = 1


def _sha256_of_file(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _agentrx_commit_sha() -> str | None:
    """Best-effort git HEAD of the agentrx package source tree.

    Returns None when git is unavailable or the source tree is not a checkout
    (e.g. installed from a wheel). We never raise: provenance must not break
    a run.
    """
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("ascii", errors="replace").strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _agentrx_version() -> str | None:
    """Read the installed package version, else fall back to the pyproject literal."""
    try:
        from importlib.metadata import version  # py>=3.8
        return version("agentrx")
    except Exception:
        pass
    # Fallback: parse pyproject.toml at the repo root next to this package.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        try:
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("version") and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return None


def _resolve_model_spec(endpoint: str) -> dict[str, Any]:
    """Snapshot the endpoint-shaped model identifiers at run time.

    This is the *configured* spec (what we asked for). The *observed* spec
    (what the server actually served) is captured separately by the LLM
    client and merged in later via ``observed_model_snapshot``.
    """
    if endpoint == "azure":
        return {
            "endpoint_type": "azure",
            "endpoint_url": g.ENDPOINT or None,
            "api_version": g.API_VERSION or None,
            "deployment_name": g.DEPLOYMENT or None,
            "model_name": g.MODEL_NAME or None,
            "embedding_model_name": g.EMBEDDING_MODEL_NAME or None,
        }
    if endpoint == "trapi":
        return {
            "endpoint_type": "trapi",
            "endpoint_url": g.TRAPI_ENDPOINT_PREFIX or None,
            "api_version": g.TRAPI_API_VERSION or None,
            "deployment_name": g.TRAPI_DEPLOYMENT_NAME or None,
            "model_name": g.TRAPI_MODEL_NAME or None,
            "model_version": g.TRAPI_MODEL_VERSION or None,
            "trapi_instance": g.TRAPI_INSTANCE or None,
        }
    if endpoint == "copilot":
        return {
            "endpoint_type": "copilot",
            "endpoint_url": None,
            "api_version": None,
            "deployment_name": None,
            "model_name": g.MODEL_NAME or None,
        }
    return {"endpoint_type": endpoint}


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def build_provenance(
    *,
    input_path: str,
    domain: str,
    endpoint: str,
    ground_truth_path: str | None,
    judge_config: RunConfig,
    stages_planned: list[str],
    stages_completed: list[str] | None = None,
    observed_model_snapshot: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the provenance dict. Pure; does not touch the filesystem outside hashing."""
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": os.path.abspath(input_path) if input_path else None,
        "domain": domain,
        "endpoint": endpoint,
        "ground_truth": os.path.abspath(ground_truth_path) if ground_truth_path else None,
        "stages_planned": list(stages_planned),
        "stages_completed": list(stages_completed) if stages_completed is not None else [],
        "run_config": dataclasses.asdict(judge_config),
        "model_spec": _resolve_model_spec(endpoint),
        "package": {
            "agentrx_commit_sha": _agentrx_commit_sha(),
            "agentrx_version": _agentrx_version(),
            "python_version": _python_version(),
        },
        "data_provenance": {
            "input_sha256": _sha256_of_file(input_path),
            "ground_truth_sha256": _sha256_of_file(ground_truth_path),
            "ground_truth_path": (
                os.path.abspath(ground_truth_path) if ground_truth_path else None
            ),
        },
        "observed_model_snapshot": observed_model_snapshot,
        "extra": dict(extra) if extra else {},
    }


def dump_provenance(run_dir: str, payload: dict[str, Any]) -> str:
    """Atomically write ``run_dir/run_config.json``. Returns the written path."""
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, PROVENANCE_FILENAME)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp_path, out_path)
    return out_path


def load_provenance(run_dir: str) -> dict[str, Any] | None:
    """Read back ``run_dir/run_config.json`` or return None if missing/corrupt."""
    path = os.path.join(run_dir, PROVENANCE_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
