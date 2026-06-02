"""Sweep orchestrator: drive ``run.py`` for many catalog claims with provenance.

This module closes the loop the catalog opened: the catalog says
*"reproducing cell ``t5_tau_oneshot_baseline_step`` means running ``run.py``
with these flags"*, and this module says *"reproducing N cells means
running ``run.py`` ``M <= N`` times because cells that share a recipe
share their pipeline output, with manifest + per-invocation logs so the
audit trail is mechanical."*

Design (first principles)
-------------------------
1. Two catalog claims that differ ONLY in which metric the aggregator
   later extracts (e.g. ``step_index_acc`` vs ``category_acc`` on the
   same row band+column of ``tab:ablations``) are NOT two pipeline
   invocations -- they read the same ``judge_output/runs/run*.json``.
   ``invocation_key()`` makes that equivalence explicit; the catalog's
   116 claims collapse to ~40 invocations.

2. The slug for an invocation is fully derived from its key, so two
   sweeps that select overlapping claims share run directories
   automatically. ``--run-dir`` + ``run.py``'s state ledger gives us
   resume for free; this module only decides "do I need to invoke
   ``run.py`` AT ALL for this invocation, or is it already done."

3. We do not parallelise here. ``run.py`` has its own LLM retry; adding
   process-level concurrency requires endpoint-rotation (R06) and is a
   separate concern. The sequential dispatcher is the natural primitive
   to bolt parallelism onto later.

4. The manifest (``<runs_root>/manifest.json``) is the single source of
   truth a downstream aggregator (R04) reads to know which cells map to
   which run directory.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from agentrx.reproduction.cli import claim_to_cmdline
from agentrx.reproduction.paper_claims import (
    ALL_CLAIMS,
    Domain,
    DynamicMode,
    Metric,
    PaperClaim,
    get_claim,
)

__all__ = [
    "Invocation",
    "InvocationResult",
    "build_manifest",
    "dump_manifest",
    "execute_sequential",
    "group_into_invocations",
    "invocation_argv",
    "invocation_key",
    "invocation_run_dir",
    "invocation_slug",
    "is_invocation_complete",
    "resolve_input_path",
    "select_claims",
]


# --------------------------------------------------------------------------
# Invocation grouping
# --------------------------------------------------------------------------


def invocation_key(c: PaperClaim) -> tuple:
    """Two claims sharing this key produce byte-identical ``run.py`` output.

    The key excludes ``metric`` and ``paper_value`` (extracted post-hoc by
    the aggregator) and ``notes`` / ``cell_id`` / ``table_*`` (pure
    metadata). Everything that actually influences what ``run.py`` does
    is captured here.
    """
    cfg = c.run_config
    return (
        c.domain,
        c.dynamic_mode,
        c.skip_static,
        c.skip_dynamic,
        cfg.prompt_mode,
        cfg.exec_mode,
        cfg.with_context,
        cfg.num_runs,
        cfg.prompt_style,
        cfg.include_nl_check_violations,
        cfg.skip_nl,
        cfg.python_check_timeout_sec,
    )


def invocation_slug(c: PaperClaim) -> str:
    """Filesystem-safe short name describing this invocation.

    Stability contract: claims with equal ``invocation_key(c)`` MUST
    produce equal ``invocation_slug(c)``. ``group_into_invocations``
    enforces this at build time so a regression in either function is
    caught the first time the catalog is grouped.
    """
    cfg = c.run_config
    parts = [
        c.domain,
        c.dynamic_mode,
        cfg.prompt_mode,
        cfg.exec_mode,
        "ctx" if cfg.with_context else "noctx",
        cfg.prompt_style,
        f"n{cfg.num_runs}",
        f"pyto{cfg.python_check_timeout_sec:g}",
    ]
    if c.skip_static:
        parts.append("nostatic")
    if c.skip_dynamic:
        parts.append("nodynamic")
    if not cfg.include_nl_check_violations:
        parts.append("nonlvio")
    if cfg.skip_nl:
        parts.append("noNL")
    return "_".join(parts)


@dataclass(frozen=True)
class Invocation:
    """One pipeline run that collectively reproduces 1+ catalog cells."""
    slug: str
    key: tuple
    claims: tuple[PaperClaim, ...]

    @property
    def representative(self) -> PaperClaim:
        return self.claims[0]

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(c.cell_id for c in self.claims)


def group_into_invocations(claims: Iterable[PaperClaim]) -> list[Invocation]:
    """Collapse claims into invocations, sorted by slug.

    Within each invocation, claims are sorted by ``cell_id`` for
    deterministic manifest output. The slug<->key bijection is checked
    here; a violation indicates ``invocation_key`` and ``invocation_slug``
    have drifted apart.
    """
    buckets: dict[tuple, list[PaperClaim]] = {}
    for c in claims:
        buckets.setdefault(invocation_key(c), []).append(c)
    out: list[Invocation] = []
    seen_slugs: dict[str, tuple] = {}
    for key, group in buckets.items():
        slug = invocation_slug(group[0])
        for c in group[1:]:
            other = invocation_slug(c)
            if other != slug:
                raise AssertionError(
                    "invocation_slug drift within an invocation_key bucket: "
                    f"{group[0].cell_id} -> {slug!r} vs "
                    f"{c.cell_id} -> {other!r}; "
                    "invocation_key and invocation_slug must agree on equivalence."
                )
        prev_key = seen_slugs.get(slug)
        if prev_key is not None and prev_key != key:
            raise AssertionError(
                f"two distinct invocation_keys produced the same slug {slug!r}; "
                "invocation_slug needs another disambiguating field."
            )
        seen_slugs[slug] = key
        ordered = tuple(sorted(group, key=lambda c: c.cell_id))
        out.append(Invocation(slug=slug, key=key, claims=ordered))
    out.sort(key=lambda inv: inv.slug)
    return out


# --------------------------------------------------------------------------
# Claim selection
# --------------------------------------------------------------------------


def select_claims(
    *,
    cell_ids: Sequence[str] | None = None,
    table_label: str | None = None,
    domain: Domain | None = None,
    metric: Metric | None = None,
    dynamic_mode: DynamicMode | None = None,
    skip_static: bool | None = None,
    skip_dynamic: bool | None = None,
) -> list[PaperClaim]:
    """Pick a subset of the catalog by axis filters and/or an explicit id list.

    When ``cell_ids`` is non-empty the pool starts from those ids (raises
    ``KeyError`` for unknown ids) and is then further narrowed by the
    other filters. When ``cell_ids`` is None/empty the pool starts from
    ``ALL_CLAIMS``. The original axis-filter implementation lives in
    ``paper_claims.claims_for``; this wrapper adds the cell_ids axis and
    the recipe-shape axes that the sweep needs.
    """
    if cell_ids:
        pool: list[PaperClaim] = [get_claim(cid) for cid in cell_ids]
    else:
        pool = list(ALL_CLAIMS)
    if table_label is not None:
        pool = [c for c in pool if c.table_label == table_label]
    if domain is not None:
        pool = [c for c in pool if c.domain == domain]
    if metric is not None:
        pool = [c for c in pool if c.metric == metric]
    if dynamic_mode is not None:
        pool = [c for c in pool if c.dynamic_mode == dynamic_mode]
    if skip_static is not None:
        pool = [c for c in pool if c.skip_static is skip_static]
    if skip_dynamic is not None:
        pool = [c for c in pool if c.skip_dynamic is skip_dynamic]
    return pool


# --------------------------------------------------------------------------
# Input resolution
# --------------------------------------------------------------------------


def resolve_input_path(claim: PaperClaim, *, base_dir: str | Path = ".") -> str:
    """Resolve the catalog's ``input_glob()`` to a concrete directory path.

    All three shipped datasets are directories of one-file-per-trajectory
    inputs (``data/tau_dataset/*.json``, ``data/magentic_dataset/*.json``,
    ``data/flash_dataset/*.jsonl``). ``run.py`` accepts a directory and
    iterates; ``--subset-ids`` further filters by filename stem when the
    claim is a subset (e.g. Magentic*).
    """
    glob = claim.input_glob()
    base = Path(base_dir).resolve()
    parent = (base / glob).parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"resolved dataset directory does not exist: {parent} "
            f"(catalog glob={glob!r}, claim={claim.cell_id!r}, "
            f"base_dir={base})"
        )
    return str(parent)


# --------------------------------------------------------------------------
# Argv + run directory
# --------------------------------------------------------------------------


def invocation_run_dir(inv: Invocation, runs_root: str | Path) -> Path:
    return Path(runs_root) / "invocations" / inv.slug


def invocation_argv(
    inv: Invocation,
    *,
    runs_root: str | Path,
    base_dir: str | Path = ".",
    python: str = "python",
    endpoint: str = "azure",
) -> list[str]:
    """Build the ``run.py`` argv that materialises this invocation."""
    input_path = resolve_input_path(inv.representative, base_dir=base_dir)
    argv = claim_to_cmdline(inv.representative, input_path=input_path,
                            endpoint=endpoint)
    argv[0] = python
    argv[1] = str((Path(base_dir) / "run.py").resolve())
    argv += ["--run-dir", str(invocation_run_dir(inv, runs_root).resolve())]
    return argv


# --------------------------------------------------------------------------
# Resume / completion detection
# --------------------------------------------------------------------------


def is_invocation_complete(inv: Invocation, runs_root: str | Path) -> bool:
    """An invocation is complete iff ``run.py``'s state ledger says so.

    We do not introspect per-trajectory output files because ``run.py`` is
    the authoritative source for "what completed" via its
    ``run_state.json`` ``completed_stages`` list. ``"judge"`` is the
    terminal stage we care about for paper cells; the optional ``"report"``
    stage is a downstream convenience not required for the aggregator.
    """
    state_path = invocation_run_dir(inv, runs_root) / "run_state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return "judge" in set(state.get("completed_stages", []))


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def build_manifest(
    invocations: list[Invocation],
    *,
    runs_root: str | Path,
    selection: dict,
    base_dir: str | Path,
    endpoint: str,
    python: str,
) -> dict:
    """Serialisable description of the sweep plan (no execution side effects)."""
    return {
        "schema_version": 1,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "runs_root": str(Path(runs_root).resolve()),
        "base_dir": str(Path(base_dir).resolve()),
        "endpoint": endpoint,
        "python": python,
        "selection": selection,
        "n_claims": sum(len(inv.claims) for inv in invocations),
        "n_invocations": len(invocations),
        "invocations": [
            {
                "slug": inv.slug,
                "run_dir": str(invocation_run_dir(inv, runs_root).resolve()),
                "cell_ids": list(inv.cell_ids),
                "argv": invocation_argv(
                    inv,
                    runs_root=runs_root,
                    base_dir=base_dir,
                    python=python,
                    endpoint=endpoint,
                ),
            }
            for inv in invocations
        ],
    }


def dump_manifest(manifest: dict, runs_root: str | Path) -> Path:
    path = Path(runs_root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return path


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


@dataclass
class InvocationResult:
    slug: str
    cell_ids: tuple[str, ...]
    skipped: bool
    returncode: int | None = None
    wall_sec: float | None = None
    log_path: str | None = None


def execute_sequential(
    invocations: list[Invocation],
    *,
    runs_root: str | Path,
    base_dir: str | Path = ".",
    python: str = "python",
    endpoint: str = "azure",
    resume: bool = True,
    log_stream=None,
) -> list[InvocationResult]:
    """Run each invocation in series, honouring ``resume``.

    Returns one ``InvocationResult`` per invocation in the input order.
    Does NOT raise on per-invocation non-zero exit codes; the caller
    decides how to handle partial completion (the manifest plus per-
    invocation logs constitute the audit trail). Concurrency, retry, and
    endpoint rotation are out of scope for this dispatcher.
    """
    results: list[InvocationResult] = []
    out = log_stream or sys.stdout
    base = Path(base_dir).resolve()
    for i, inv in enumerate(invocations, 1):
        rd = invocation_run_dir(inv, runs_root)
        rd.mkdir(parents=True, exist_ok=True)
        prefix = f"[{i}/{len(invocations)}]"
        if resume and is_invocation_complete(inv, runs_root):
            print(f"{prefix} SKIP {inv.slug} ({len(inv.claims)} cell(s)) "
                  "-- already complete", file=out, flush=True)
            results.append(InvocationResult(
                slug=inv.slug, cell_ids=inv.cell_ids,
                skipped=True, returncode=0, wall_sec=0.0,
            ))
            continue
        argv = invocation_argv(inv, runs_root=runs_root, base_dir=base_dir,
                               python=python, endpoint=endpoint)
        log_path = rd / "sweep_invocation.log"
        print(f"{prefix} RUN  {inv.slug} ({len(inv.claims)} cell(s))",
              file=out, flush=True)
        t0 = _dt.datetime.now()
        with open(log_path, "ab") as log_fh:
            cp = subprocess.run(argv, stdout=log_fh, stderr=subprocess.STDOUT,
                                cwd=str(base))
        wall = (_dt.datetime.now() - t0).total_seconds()
        rc = cp.returncode
        status = "DONE" if rc == 0 else f"FAIL(rc={rc})"
        print(f"{prefix} {status} {inv.slug} {wall:.1f}s -> {log_path}",
              file=out, flush=True)
        results.append(InvocationResult(
            slug=inv.slug, cell_ids=inv.cell_ids,
            skipped=False, returncode=rc, wall_sec=wall,
            log_path=str(log_path),
        ))
    return results
