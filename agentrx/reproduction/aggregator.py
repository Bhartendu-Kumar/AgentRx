"""Aggregator: join a sweep manifest to per-invocation judge outputs and
project each catalog claim's metric into paper units, with a paper-delta.

This module closes the catalog -> sweep -> aggregator triangle: the catalog
declares ``cell_id -> (metric, paper_value)``, the sweep produces
``<run_dir>/judge_output/analysis/summary.json`` per invocation, and this
module emits one ``CellReport`` per claim with ``observed_mean +/- observed_std``
in the same units the paper printed (``%`` for accuracies, ``steps`` for
distance).

Design (first principles)
-------------------------
1. *Paper units are sacred.* If the paper prints ``32.2 +/- 3.2``, observed
   must come back as a percent number. The judge writes accuracy as a
   fraction in ``[0,1]``; ``_project_*`` is the single place we multiply
   by 100. Distance metrics pass through.

2. *Std is computed from per-run accuracy, not from a precomputed scalar.*
   ``analyze_metrics`` does the same: per-run fraction = correct/total,
   then ``pstdev`` over the run vector (ddof=0). This matches what
   ``compute_accuracy_std`` in agentrx/reports/analyze_metrics.py does
   for the legacy single-run reports.

3. *The aggregator never invents data.* If a metric cannot be projected
   (no per-run cases, missing summary, missing key), the cell's
   ``status`` reflects that and ``observed_mean/std`` are ``None``. We
   do not silently substitute zeros.

4. *Manifest is the join key.* We do NOT walk ``runs_root`` looking for
   directories; we read the manifest authored by the sweep so the sweep
   plan and the report are byte-aligned.

5. *Provenance is part of the report, not metadata.* A reproduction
   number without ``(model_name, api_version, prompt_tokens, output_tokens,
   execution_time)`` is unverifiable. ``extract_provenance`` walks the
   same per-run summaries that drove the metric projection and carries
   the model identity + per-cell token + wall-clock totals into the
   CellReport. Model homogeneity within an invocation is checked: if
   different runs of the same cell used different model_names, that is
   recorded explicitly rather than silently averaged.

6. *Codebase identity is part of the report.* The W1b pipeline writes
   ``<run_dir>/run_config.json`` containing the producing AgentRx git
   SHA + full RunConfig + input sha256s. ``load_run_identity`` reads
   that file back and attaches the identity to each CellReport. The
   aggregator also stamps its OWN git SHA at report-generation time
   into the top-level JSON, so a reader can always answer: which code
   ran the experiment, and which code summarised it.
"""
from __future__ import annotations

import datetime as _dt
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agentrx.reproduction.paper_claims import (
    Metric,
    PaperClaim,
    get_claim,
)

__all__ = [
    "CellReport",
    "MetricMissing",
    "Provenance",
    "RunIdentity",
    "aggregate_invocation",
    "aggregate_sweep",
    "extract_provenance",
    "load_judge_summary",
    "load_run_identity",
    "project_metric",
    "render_cell_reports_markdown",
    "report_to_json",
]


class MetricMissing(LookupError):
    """A paper metric cannot be projected from the summary as-is.

    Raised when the per-run accuracy vector is empty, when a required
    summary key is absent, or when a metric is recognised but currently
    has no implemented projection. The aggregator catches this and
    surfaces it as ``status='METRIC_MISSING'`` on the cell report rather
    than letting it crash the run.
    """


# --------------------------------------------------------------------------
# Metric projection
# --------------------------------------------------------------------------


def _per_run_step_index_acc(run: dict) -> float | None:
    c = run.get("Correct step number predictions")
    i = run.get("Incorrect step number predictions")
    if c is None or i is None:
        return None
    total = c + i
    return (c / total) if total > 0 else None


def _per_run_category_acc(run: dict) -> float | None:
    c = run.get("Correct cases")
    i = run.get("Incorrect cases")
    if c is None or i is None:
        return None
    total = c + i
    return (c / total) if total > 0 else None


def _per_run_avg_step_distance(run: dict) -> float | None:
    v = run.get("Overall average distance")
    return None if v is None else float(v)


def _per_run_tolerance(run: dict, tol: int) -> float | None:
    key = f"Step accuracy within +-{tol}"
    v = run.get(key)
    return None if v is None else float(v)


_TOLERANCE_METRIC_TO_INT: dict[str, int] = {
    "step_acc_at_plus_minus_1": 1,
    "step_acc_at_plus_minus_3": 3,
    "step_acc_at_plus_minus_5": 5,
}


def _scale_pct(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values) * 100.0, statistics.pstdev(values) * 100.0


def _scale_raw(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def project_metric(metric: Metric, summary: dict) -> tuple[float, float, int]:
    """Return ``(observed_mean, observed_std, n_runs)`` in paper units.

    ``summary`` is the deserialised
    ``judge_output/analysis/summary.json`` (see
    ``agentrx/judge/judge.py::create_aggregate_summary``).

    Raises ``MetricMissing`` if the metric cannot be projected (empty
    per-run vector, missing keys, or unimplemented metric).
    """
    runs = summary.get("individual_run_summaries")
    if not isinstance(runs, list) or not runs:
        raise MetricMissing(
            f"summary has no 'individual_run_summaries' or it is empty "
            f"(metric={metric!r})"
        )

    if metric == "step_index_acc":
        per_run = [v for v in (_per_run_step_index_acc(r) for r in runs) if v is not None]
        if not per_run:
            raise MetricMissing(f"no run has step-prediction counts (metric={metric!r})")
        m, s = _scale_pct(per_run)
        return m, s, len(per_run)

    if metric == "category_acc":
        per_run = [v for v in (_per_run_category_acc(r) for r in runs) if v is not None]
        if not per_run:
            raise MetricMissing(f"no run has Correct/Incorrect cases (metric={metric!r})")
        m, s = _scale_pct(per_run)
        return m, s, len(per_run)

    if metric == "avg_step_distance":
        per_run = [v for v in (_per_run_avg_step_distance(r) for r in runs) if v is not None]
        if not per_run:
            raise MetricMissing(f"no run has Overall average distance (metric={metric!r})")
        m, s = _scale_raw(per_run)
        return m, s, len(per_run)

    if metric in _TOLERANCE_METRIC_TO_INT:
        tol = _TOLERANCE_METRIC_TO_INT[metric]
        per_run = [v for v in (_per_run_tolerance(r, tol) for r in runs) if v is not None]
        if not per_run:
            raise MetricMissing(
                f"no run has 'Step accuracy within +-{tol}' (metric={metric!r})"
            )
        m, s = _scale_pct(per_run)
        return m, s, len(per_run)

    # Catalog has more metric names (per-category, agent-attribution, tokens)
    # that no current claim uses. Surface them as MISSING rather than
    # silently dropping; future commits add the projections.
    raise MetricMissing(
        f"metric {metric!r} has no projection implemented yet"
    )


# --------------------------------------------------------------------------
# CellReport
# --------------------------------------------------------------------------

Status = str  # "OK" | "MISSING_RUN_DIR" | "MISSING_SUMMARY" | "METRIC_MISSING"


@dataclass
class Provenance:
    """What ran, on what model, at what cost.

    Populated from the per-run blocks of ``summary.json``. Token and
    execution-time fields are SUMS across runs (the user can divide by
    ``n_runs`` to recover per-run averages without losing information).
    ``model_homogeneous`` records whether every run within the invocation
    reported the same ``model_name`` -- a heterogeneous mix is a real
    reproduction concern that the report must not hide.
    """
    model_name: str | None
    api_version: str | None
    total_prompt_tokens: int | None
    total_output_tokens: int | None
    total_execution_time_sec: float | None
    n_runs: int
    model_homogeneous: bool
    distinct_model_names: list[str]


def _sum_or_none(runs: Sequence[dict], key: str) -> int | float | None:
    """Sum ``key`` across runs; return ``None`` if no run has the key."""
    values = [r[key] for r in runs if key in r and r[key] is not None]
    if not values:
        return None
    return sum(values)


def extract_provenance(summary: dict) -> Provenance:
    """Pull model identity and cost totals out of a judge summary.

    Always returns a ``Provenance``; fields are ``None`` when the
    corresponding per-run keys are absent (older judge outputs).
    """
    runs = summary.get("individual_run_summaries") or []
    model_names_raw = [r.get("model_name") for r in runs if r.get("model_name")]
    distinct = sorted(set(model_names_raw))
    api_versions = [r.get("api_version") for r in runs if r.get("api_version")]
    return Provenance(
        model_name=distinct[0] if len(distinct) == 1 else (distinct[0] if distinct else None),
        api_version=api_versions[0] if api_versions else None,
        total_prompt_tokens=_sum_or_none(runs, "total_prompt_tokens"),
        total_output_tokens=_sum_or_none(runs, "total_output_tokens"),
        total_execution_time_sec=_sum_or_none(runs, "total_execution_time_sec"),
        n_runs=len(runs),
        model_homogeneous=(len(distinct) <= 1),
        distinct_model_names=distinct,
    )


@dataclass
class CellReport:
    cell_id: str
    table_label: str
    domain: str
    metric: str
    paper_mean: float
    paper_std: float | None
    unit: str
    run_dir: str
    invocation_slug: str
    status: Status
    observed_mean: float | None = None
    observed_std: float | None = None
    n_runs: int | None = None
    abs_delta: float | None = None
    within_paper_std: bool | None = None
    notes: str = ""
    provenance: Provenance | None = None
    run_identity: RunIdentity | None = None


def _compute_delta(observed_mean: float, paper_mean: float,
                   paper_std: float | None) -> tuple[float, bool | None]:
    delta = abs(observed_mean - paper_mean)
    if paper_std is None:
        return delta, None
    return delta, delta <= paper_std


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load_judge_summary(run_dir: str | Path) -> dict | None:
    """Load ``<run_dir>/judge_output/analysis/summary.json`` or return None.

    ``None`` is the explicit *"this invocation didn't produce a judge
    summary"* signal; callers translate it to ``MISSING_SUMMARY`` on the
    cell report instead of trying to recover.
    """
    p = Path(run_dir) / "judge_output" / "analysis" / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class RunIdentity:
    """Which codebase + recipe produced an invocation's outputs.

    Snapshotted from ``<run_dir>/run_config.json`` (written by
    ``agentrx/pipeline/provenance.py``, W1b). Carries the producing
    AgentRx git SHA so a reader can clone the exact commit, and the
    full RunConfig + model_spec + data sha256s so a reader can
    reconstruct the recipe and confirm input identity.

    ``run_config_payload`` is the whole W1b dict verbatim. This costs a
    few KB per cell but makes each CellReport a self-contained
    reproduction record -- the only external dependency is the git
    repository at ``producing_agentrx_commit_sha``.
    """
    producing_agentrx_commit_sha: str | None
    producing_agentrx_version: str | None
    timestamp_utc: str | None
    input_sha256: str | None
    ground_truth_sha256: str | None
    run_config_path: str | None
    run_config_payload: dict | None


def load_run_identity(run_dir: str | Path) -> RunIdentity | None:
    """Read ``<run_dir>/run_config.json`` and project the identity fields.

    Returns ``None`` if the file is missing or corrupt. Returns a
    populated ``RunIdentity`` with possibly-None inner fields when the
    file exists but is an older schema that lacks some keys.
    """
    p = Path(run_dir) / "run_config.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    pkg = d.get("package") or {}
    data = d.get("data_provenance") or {}
    return RunIdentity(
        producing_agentrx_commit_sha=pkg.get("agentrx_commit_sha"),
        producing_agentrx_version=pkg.get("agentrx_version"),
        timestamp_utc=d.get("timestamp_utc"),
        input_sha256=data.get("input_sha256"),
        ground_truth_sha256=data.get("ground_truth_sha256"),
        run_config_path=str(p),
        run_config_payload=d,
    )


def _aggregator_commit_sha() -> str | None:
    """Return the git SHA of the AgentRx checkout containing aggregator.py.

    Walks up from this file to find a git repo. Returns None if git is
    unavailable or the file is not inside a git working tree -- the
    report still works without it, but the SHA is the canonical way to
    pin which aggregator code generated which numbers.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            repo = parent
            break
    else:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if sha else None


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate_invocation(
    *,
    invocation_slug: str,
    run_dir: str | Path,
    claims: Sequence[PaperClaim],
) -> list[CellReport]:
    """Aggregate one invocation's judge output into per-claim CellReports."""
    rd = Path(run_dir)
    reports: list[CellReport] = []
    summary = load_judge_summary(rd) if rd.is_dir() else None
    identity = load_run_identity(rd) if rd.is_dir() else None
    base_status: Status | None = None
    if not rd.is_dir():
        base_status = "MISSING_RUN_DIR"
    elif summary is None:
        base_status = "MISSING_SUMMARY"

    for c in claims:
        pv = c.paper_value
        assert pv is not None, (
            f"claim {c.cell_id} has no paper_value; the catalog must not "
            "emit claims without published numbers for the aggregator."
        )
        common = dict(
            cell_id=c.cell_id,
            table_label=c.table_label,
            domain=c.domain,
            metric=c.metric,
            paper_mean=pv.mean,
            paper_std=pv.std,
            unit=pv.unit,
            run_dir=str(rd.resolve()) if rd.is_absolute() or rd.exists() else str(rd),
            invocation_slug=invocation_slug,
        )
        if base_status is not None:
            reports.append(CellReport(
                status=base_status, run_identity=identity, **common,
            ))
            continue
        prov = extract_provenance(summary)
        try:
            mean, std, n = project_metric(c.metric, summary)
        except MetricMissing as e:
            reports.append(CellReport(
                status="METRIC_MISSING", notes=str(e),
                provenance=prov, run_identity=identity, **common,
            ))
            continue
        delta, within = _compute_delta(mean, pv.mean, pv.std)
        reports.append(CellReport(
            status="OK",
            observed_mean=mean,
            observed_std=std,
            n_runs=n,
            abs_delta=delta,
            within_paper_std=within,
            provenance=prov,
            run_identity=identity,
            **common,
        ))
    return reports


def aggregate_sweep(manifest_path: str | Path) -> list[CellReport]:
    """Aggregate every invocation in a sweep manifest into a flat report.

    Output is sorted by ``cell_id`` so reports are diff-friendly across
    sweep runs.
    """
    mpath = Path(manifest_path)
    manifest = json.loads(mpath.read_text())
    out: list[CellReport] = []
    for entry in manifest.get("invocations", []):
        slug = entry["slug"]
        run_dir = entry["run_dir"]
        cell_ids = entry["cell_ids"]
        claims = [get_claim(cid) for cid in cell_ids]
        out.extend(aggregate_invocation(
            invocation_slug=slug, run_dir=run_dir, claims=claims,
        ))
    out.sort(key=lambda r: r.cell_id)
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def report_to_json(reports: Sequence[CellReport]) -> str:
    """Serialise reports as a stable JSON document (sorted by cell_id).

    Top-level metadata pins the aggregator's own git SHA + generation
    timestamp so a reader can answer both *"which experiment produced
    these numbers"* (per-cell ``run_identity.producing_agentrx_commit_sha``)
    and *"which aggregator summarised them"* (top-level
    ``aggregator_commit_sha``).
    """
    payload = {
        "schema_version": 1,
        "aggregator_commit_sha": _aggregator_commit_sha(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_reports": len(reports),
        "n_ok": sum(1 for r in reports if r.status == "OK"),
        "reports": [asdict(r) for r in sorted(reports, key=lambda r: r.cell_id)],
    }
    return json.dumps(payload, indent=2)


def _fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


def render_cell_reports_markdown(reports: Sequence[CellReport]) -> str:
    """Render reports as a Markdown table. Stable column order for diffs."""
    header = (
        "| cell_id | table | domain | metric | paper | observed | n | "
        "abs_delta | within_paper_std | model | status |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in sorted(reports, key=lambda x: x.cell_id):
        paper = _fmt_pm(r.paper_mean, r.paper_std)
        observed = _fmt_pm(r.observed_mean, r.observed_std)
        ad = "—" if r.abs_delta is None else f"{r.abs_delta:.2f}"
        wps = "—" if r.within_paper_std is None else ("yes" if r.within_paper_std else "no")
        nr = "—" if r.n_runs is None else str(r.n_runs)
        if r.provenance is None or r.provenance.model_name is None:
            model = "—"
        elif r.provenance.model_homogeneous:
            model = r.provenance.model_name
        else:
            model = "MIXED:" + ",".join(r.provenance.distinct_model_names)
        rows.append(
            f"| {r.cell_id} | {r.table_label} | {r.domain} | {r.metric} | "
            f"{paper} | {observed} | {nr} | {ad} | {wps} | {model} | {r.status} |"
        )
    return "\n".join(rows) + "\n"
