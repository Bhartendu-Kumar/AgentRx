"""Reproduction verdict: turn per-cell paper-vs-observed reports into a
PASS / FAIL gate with a single exit code.

This is the missing concern between ``report`` (which says *what the
numbers are*) and a human or CI job (which needs to know *did this
checkout reproduce the paper*). It adds NO new measurement logic: it
consumes the ``CellReport`` objects the aggregator already produces and
applies one explicit, first-principles tolerance policy.

Design (first principles)
-------------------------
1. *Never silently pass what we could not measure.* A cell is ``PASS``
   only if the pipeline produced a comparable number AND that number
   lands inside the tolerance band. A missing run, missing summary, or
   unprojectable metric is never green.

2. *The band is the paper's own dispersion, not a number we invent.*
   When the paper published a std, the tolerance is ``paper_std *
   std_multiplier`` (multiplier defaults to 1.0, i.e. "inside one
   reported sigma"). ``abs_tol`` is only a fallback for cells the paper
   published without a std; if neither is available the cell is
   ``INCONCLUSIVE`` rather than assumed-good.

3. *Inconclusive is not pass.* Both ``METRIC_MISSING`` (the metric has
   no projection yet) and "OK but no tolerance available" yield
   ``INCONCLUSIVE``. The overall gate is green only when EVERY selected
   cell reached a definitive ``PASS`` -- you cannot claim to have
   reproduced a cell you could not even judge.

Per-cell policy
---------------
    status == OK, |observed - paper| <= tol       -> PASS
    status == OK, |observed - paper| >  tol        -> FAIL
    status == OK, no tolerance available           -> INCONCLUSIVE
    status == METRIC_MISSING                       -> INCONCLUSIVE
    status in {MISSING_RUN_DIR, MISSING_SUMMARY}   -> FAIL

Tolerance
---------
    tol = paper_std * std_multiplier   when paper_std is not None
    tol = abs_tol                      when paper_std is None and abs_tol given
    (no tol)                           otherwise
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from agentrx.reproduction.aggregator import CellReport

__all__ = [
    "CellVerdict",
    "SweepVerdict",
    "classify_cell",
    "classify_reports",
    "render_verdict_summary",
    "verdict_to_json",
]


PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CellVerdict:
    """The pass/fail decision for one cell, with the band it was judged by."""

    cell_id: str
    metric: str
    domain: str
    status: str                 # the underlying CellReport.status
    verdict: str                # PASS | FAIL | INCONCLUSIVE
    reason: str
    paper_mean: float | None
    observed_mean: float | None
    abs_delta: float | None
    tolerance: float | None
    tolerance_kind: str | None  # "paper_std*k" | "abs_tol" | None


@dataclass
class SweepVerdict:
    """The aggregate gate over a whole sweep selection."""

    passed: bool
    n_cells: int
    n_pass: int
    n_fail: int
    n_inconclusive: int
    std_multiplier: float
    abs_tol: float | None
    cells: list[CellVerdict]


def _tolerance_for(
    report: CellReport, std_multiplier: float, abs_tol: float | None
) -> tuple[float | None, str | None]:
    """Resolve the tolerance band for one report.

    The paper's own std is primary; ``abs_tol`` is the explicit operator
    fallback used only when the paper published no std. We never invent a
    band, so the return is ``(None, None)`` when neither is available.
    """
    if report.paper_std is not None:
        return report.paper_std * std_multiplier, "paper_std*k"
    if abs_tol is not None:
        return abs_tol, "abs_tol"
    return None, None


def classify_cell(
    report: CellReport,
    *,
    std_multiplier: float = 1.0,
    abs_tol: float | None = None,
) -> CellVerdict:
    """Apply the per-cell policy to a single ``CellReport``."""
    common = dict(
        cell_id=report.cell_id,
        metric=report.metric,
        domain=report.domain,
        status=report.status,
        paper_mean=report.paper_mean,
        observed_mean=report.observed_mean,
        abs_delta=report.abs_delta,
    )

    if report.status in ("MISSING_RUN_DIR", "MISSING_SUMMARY"):
        return CellVerdict(
            verdict=FAIL,
            reason=f"no comparable number produced ({report.status})",
            tolerance=None,
            tolerance_kind=None,
            **common,
        )

    if report.status == "METRIC_MISSING":
        return CellVerdict(
            verdict=INCONCLUSIVE,
            reason="metric has no projection yet (METRIC_MISSING)",
            tolerance=None,
            tolerance_kind=None,
            **common,
        )

    if report.status != "OK":
        # Defensive: an unknown status is never silently passed.
        return CellVerdict(
            verdict=INCONCLUSIVE,
            reason=f"unrecognised status {report.status!r}",
            tolerance=None,
            tolerance_kind=None,
            **common,
        )

    tol, kind = _tolerance_for(report, std_multiplier, abs_tol)
    if tol is None:
        return CellVerdict(
            verdict=INCONCLUSIVE,
            reason="paper published no std and no --abs-tol supplied",
            tolerance=None,
            tolerance_kind=None,
            **common,
        )

    assert report.abs_delta is not None, (
        "OK report must carry abs_delta; aggregator computed it from "
        "observed_mean and paper_mean."
    )
    if report.abs_delta <= tol:
        return CellVerdict(
            verdict=PASS,
            reason=f"|delta|={report.abs_delta:.4g} <= tol={tol:.4g} ({kind})",
            tolerance=tol,
            tolerance_kind=kind,
            **common,
        )
    return CellVerdict(
        verdict=FAIL,
        reason=f"|delta|={report.abs_delta:.4g} > tol={tol:.4g} ({kind})",
        tolerance=tol,
        tolerance_kind=kind,
        **common,
    )


def classify_reports(
    reports: Sequence[CellReport],
    *,
    std_multiplier: float = 1.0,
    abs_tol: float | None = None,
) -> SweepVerdict:
    """Classify every report and fold into a single sweep verdict.

    The gate is green (``passed=True``) only when every cell is ``PASS``:
    a sweep that left any cell ``FAIL`` or ``INCONCLUSIVE`` did not
    conclusively reproduce its selection.
    """
    cells = [
        classify_cell(r, std_multiplier=std_multiplier, abs_tol=abs_tol)
        for r in sorted(reports, key=lambda x: x.cell_id)
    ]
    n_pass = sum(1 for c in cells if c.verdict == PASS)
    n_fail = sum(1 for c in cells if c.verdict == FAIL)
    n_inconclusive = sum(1 for c in cells if c.verdict == INCONCLUSIVE)
    return SweepVerdict(
        passed=(len(cells) > 0 and n_fail == 0 and n_inconclusive == 0),
        n_cells=len(cells),
        n_pass=n_pass,
        n_fail=n_fail,
        n_inconclusive=n_inconclusive,
        std_multiplier=std_multiplier,
        abs_tol=abs_tol,
        cells=cells,
    )


def verdict_to_json(verdict: SweepVerdict) -> str:
    """Serialise the verdict as a stable JSON document."""
    payload = {
        "schema_version": 1,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "passed": verdict.passed,
        "n_cells": verdict.n_cells,
        "n_pass": verdict.n_pass,
        "n_fail": verdict.n_fail,
        "n_inconclusive": verdict.n_inconclusive,
        "std_multiplier": verdict.std_multiplier,
        "abs_tol": verdict.abs_tol,
        "cells": [asdict(c) for c in verdict.cells],
    }
    return json.dumps(payload, indent=2)


def render_verdict_summary(verdict: SweepVerdict) -> str:
    """Render a human-readable one-line-per-cell verdict block."""
    lines: list[str] = []
    overall = "PASS" if verdict.passed else "FAIL"
    lines.append(
        f"# Reproduction verdict: {overall} "
        f"({verdict.n_pass}/{verdict.n_cells} cells PASS, "
        f"{verdict.n_fail} FAIL, {verdict.n_inconclusive} INCONCLUSIVE; "
        f"std_multiplier={verdict.std_multiplier}, abs_tol={verdict.abs_tol})"
    )
    for c in verdict.cells:
        lines.append(f"  [{c.verdict:<12}] {c.cell_id:<40} {c.reason}")
    return "\n".join(lines) + "\n"
