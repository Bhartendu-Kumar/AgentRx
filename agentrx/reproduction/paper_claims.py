"""Paper-claims catalog: one row of one paper table -> one ``PaperClaim``.

Schema design constraints
-------------------------
1. Every claim must carry its full recipe (``RunConfig`` + dataset + GT)
   so reproducing it is mechanical: no human decisions, no guessing.
2. Every claim cites its source: ``table_label`` is the LaTeX ``\\label{...}``
   used in ``resources/paper/eval.tex``. A reader can grep the .tex to
   verify our transcription.
3. The published number lives on ``paper_value`` (``PaperValue(mean, std)``)
   exactly as written in the paper. Where the paper omits std (single
   number, no $\\pm$), ``std`` is ``None`` — we never invent stds.
4. ``run_config`` is always a real ``RunConfig`` constructed from
   ``PAPER_DEFAULT`` via ``dataclasses.replace`` so any future field on
   ``RunConfig`` is correctly inherited.
5. ``dynamic_mode`` and the ``skip_*`` flags are NOT on ``RunConfig`` by
   design (see ``profiles.py`` rationale). They are first-class fields here
   because they are recipe inputs.

What this file does NOT do
--------------------------
- It does not enumerate every cell in every paper table on initial commit.
  Coverage is added in named, reviewable chunks. The first chunk (this
  commit) covers ``tab:static-dynamic`` (Table 4: tau Global vs Dynamic
  ablations) in full. Subsequent commits add ``tab:ablations`` (Table 5)
  and ``tab:step_cat_metrics_per_domain_transposed`` (Table 6).
- It does not run anything. ``agentrx/reproduction/cli.py`` is the runner.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

from agentrx.pipeline.profiles import PAPER_DEFAULT, RunConfig

Domain = Literal["tau", "magentic", "magentic_star", "flash"]
Metric = Literal[
    "step_index_acc",
    "category_acc",
    "avg_step_distance",
    # Tolerance columns from tab:step_cat_metrics_per_domain_transposed:
    "step_acc_at_plus_minus_1",
    "step_acc_at_plus_minus_3",
    "step_acc_at_plus_minus_5",
    "critical_category_acc",
    "any_category_acc",
    "earliest_category_acc",
    "terminal_category_acc",
    # Token-stats table:
    "avg_tokens_per_step",
    "avg_chars_per_step",
    "avg_tokens_per_trajectory",
    # tab:whowhen agent-attribution accuracy:
    "agent_acc",
]
DynamicMode = Literal["oneshot", "stepbystep"]


@dataclass(frozen=True)
class PaperValue:
    """A single number reported in the paper, with optional standard deviation.

    ``unit`` is informational; numeric comparison code should know what
    metric it's looking at and convert if needed.
    """
    mean: float
    std: float | None = None
    unit: str = "percent"


# Datasets shipped in this repo. ``magentic_star`` shares the trajectory
# pool with ``magentic`` but is filtered to the 27 ids in
# ``data/ground_truth/magentic_star_ids.json``.
_DATASET_GLOB: dict[str, str] = {
    "tau": "data/tau_dataset/*.json",
    "magentic": "data/magentic_dataset/*.json",
    "magentic_star": "data/magentic_dataset/*.json",
    "flash": "data/flash_dataset/*.jsonl",
}
_GROUND_TRUTH: dict[str, str] = {
    "tau": "data/ground_truth/tau_ground_truth.json",
    "magentic": "data/ground_truth/magentic_one_ground_truth.json",
    "magentic_star": "data/ground_truth/magentic_one_ground_truth.json",
    "flash": "data/ground_truth/flash_ground_truth.json",
}
_SUBSET_IDS_FILE: dict[str, str | None] = {
    "tau": None,
    "magentic": None,
    "magentic_star": "data/ground_truth/magentic_star_ids.json",
    "flash": None,
}


@dataclass(frozen=True)
class PaperClaim:
    """One reproducible cell of one paper table."""

    # Identification -----------------------------------------------------
    cell_id: str               # globally unique; see cell_id_unique()
    table_label: str           # LaTeX label, e.g. "tab:static-dynamic"
    table_short_name: str      # human-readable, e.g. "Table 4 (static vs dynamic)"
    domain: Domain
    metric: Metric

    # The published number ----------------------------------------------
    paper_value: PaperValue | None  # None when not catalogued yet

    # Recipe -------------------------------------------------------------
    run_config: RunConfig
    dynamic_mode: DynamicMode = "stepbystep"
    skip_static: bool = False
    skip_dynamic: bool = False

    # Provenance / caveats ----------------------------------------------
    notes: str = ""

    # Derived ------------------------------------------------------------
    def input_glob(self) -> str:
        return _DATASET_GLOB[self.domain]

    def ground_truth_path(self) -> str:
        return _GROUND_TRUTH[self.domain]

    def subset_ids_file(self) -> str | None:
        return _SUBSET_IDS_FILE[self.domain]


def cell_id_unique(claims: Sequence[PaperClaim]) -> None:
    """Validator: every cell_id must be unique. Raises ValueError if not."""
    seen: dict[str, int] = {}
    for c in claims:
        seen[c.cell_id] = seen.get(c.cell_id, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        raise ValueError(f"duplicate PaperClaim cell_ids: {dups}")


# --------------------------------------------------------------------------
# Helper constructors. The catalog below is verbose by design; helpers keep
# axis-orthogonal cells visibly aligned so a reviewer can diff them by eye.
# --------------------------------------------------------------------------


def _claim(
    cell_id: str,
    table_label: str,
    table_short_name: str,
    domain: Domain,
    metric: Metric,
    paper_value: PaperValue | None,
    *,
    prompt_mode: str = PAPER_DEFAULT.prompt_mode,
    exec_mode: str = PAPER_DEFAULT.exec_mode,
    with_context: bool = PAPER_DEFAULT.with_context,
    dynamic_mode: DynamicMode = "stepbystep",
    skip_static: bool = False,
    skip_dynamic: bool = False,
    notes: str = "",
) -> PaperClaim:
    cfg = replace(
        PAPER_DEFAULT,
        prompt_mode=prompt_mode,
        exec_mode=exec_mode,
        with_context=with_context,
    )
    return PaperClaim(
        cell_id=cell_id,
        table_label=table_label,
        table_short_name=table_short_name,
        domain=domain,
        metric=metric,
        paper_value=paper_value,
        run_config=cfg,
        dynamic_mode=dynamic_mode,
        skip_static=skip_static,
        skip_dynamic=skip_dynamic,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Table 4 (tab:static-dynamic, eval.tex L119-L137)
# Domain: tau-bench only. Mean +/- std over n=3.
# 4 methods x 2 metrics = 8 cells.
# Citation: paper text "We run this experiment only on tau-bench because
# Flash and Magentic do not have a domain policy."
# --------------------------------------------------------------------------

_T4 = "tab:static-dynamic"
_T4_NAME = "Table 4 (static vs dynamic ablation, tau-bench)"

_TABLE_4_CLAIMS: list[PaperClaim] = [
    # Baseline: no violations injected, plain taxonomy.
    _claim("t4_tau_baseline_step",    _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(32.2, 3.2),
           prompt_mode="baseline", with_context=False, exec_mode="violations-after",
           notes="Paper 'Baseline' row: prompt_mode=baseline, no violation context."),
    _claim("t4_tau_baseline_cat",     _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(25.3, 1.6),
           prompt_mode="baseline", with_context=False, exec_mode="violations-after",
           notes="Paper 'Baseline' row."),

    # Global-Only: skip dynamic stage; only schema/policy constraints fire.
    _claim("t4_tau_global_only_step", _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(41.4, 2.8),
           skip_dynamic=True,
           notes="Paper 'Global-Only' row: --skip-dynamic so only static "
                 "(schema+policy) invariants generate violations."),
    _claim("t4_tau_global_only_cat",  _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(28.7, 1.6),
           skip_dynamic=True,
           notes="Paper 'Global-Only' row."),

    # Dynamic-Only: skip static stage; only prefix-conditioned constraints.
    _claim("t4_tau_dynamic_only_step", _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(43.7, 1.6),
           skip_static=True,
           notes="Paper 'Dynamic-Only' row: --skip-static so only dynamic "
                 "(prefix-conditioned) invariants generate violations."),
    _claim("t4_tau_dynamic_only_cat",  _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(36.8, 1.6),
           skip_static=True,
           notes="Paper 'Dynamic-Only' row."),

    # AgentRx: PAPER_DEFAULT — both static and dynamic, combined prompt mode.
    _claim("t4_tau_full_step",         _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(48.3, 0.0),
           notes="Paper '\\tool' row: PAPER_DEFAULT. std=0 in paper means n=3 "
                 "yielded the same step-index value across all 3 runs."),
    _claim("t4_tau_full_cat",          _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(39.1, 1.6),
           notes="Paper '\\tool' row: PAPER_DEFAULT."),
]


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

ALL_CLAIMS: list[PaperClaim] = []
ALL_CLAIMS.extend(_TABLE_4_CLAIMS)
cell_id_unique(ALL_CLAIMS)


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------


def claims_for(
    *,
    table_label: str | None = None,
    domain: Domain | None = None,
    metric: Metric | None = None,
) -> list[PaperClaim]:
    """Filter ALL_CLAIMS. Any None argument is treated as 'any'."""
    out = list(ALL_CLAIMS)
    if table_label is not None:
        out = [c for c in out if c.table_label == table_label]
    if domain is not None:
        out = [c for c in out if c.domain == domain]
    if metric is not None:
        out = [c for c in out if c.metric == metric]
    return out


def get_claim(cell_id: str) -> PaperClaim:
    for c in ALL_CLAIMS:
        if c.cell_id == cell_id:
            return c
    raise KeyError(f"no PaperClaim with cell_id={cell_id!r}")
