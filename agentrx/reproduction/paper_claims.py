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
#
# Cross-table evidence for dynamic_mode="oneshot": tab:static-dynamic's
# AgentRx row reports tau step=48.3 cat=39.1, which is an exact match for
# tab:ablations Checklist+Vio. column ONE-SHOT row band (48.3 step, 39.1
# +/- 1.6 cat). The Step-by-Step row band of the same cell reports 37.9
# step / 35.6 cat. So Tab 4's "AgentRx" uses one-shot constraint generation,
# and by parsimony the other three rows in Tab 4 do too. (The Baseline row
# 32.2 +/- 3.2 step also matches tab:ablations Baseline col ONE-SHOT row.)
# --------------------------------------------------------------------------

_T4 = "tab:static-dynamic"
_T4_NAME = "Table 4 (static vs dynamic ablation, tau-bench)"

_TABLE_4_CLAIMS: list[PaperClaim] = [
    # Baseline: no violations injected, plain taxonomy.
    _claim("t4_tau_baseline_step",    _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(32.2, 3.2),
           prompt_mode="baseline", with_context=False, exec_mode="violations-after",
           dynamic_mode="oneshot",
           notes="Paper 'Baseline' row: prompt_mode=baseline, no violation context."),
    _claim("t4_tau_baseline_cat",     _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(25.3, 1.6),
           prompt_mode="baseline", with_context=False, exec_mode="violations-after",
           dynamic_mode="oneshot",
           notes="Paper 'Baseline' row."),

    # Global-Only: skip dynamic stage; only schema/policy constraints fire.
    _claim("t4_tau_global_only_step", _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(41.4, 2.8),
           skip_dynamic=True, dynamic_mode="oneshot",
           notes="Paper 'Global-Only' row: --skip-dynamic so only static "
                 "(schema+policy) invariants generate violations. "
                 "dynamic_mode is irrelevant when dynamic is skipped; "
                 "recorded for invocation completeness."),
    _claim("t4_tau_global_only_cat",  _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(28.7, 1.6),
           skip_dynamic=True, dynamic_mode="oneshot",
           notes="Paper 'Global-Only' row."),

    # Dynamic-Only: skip static stage; only prefix-conditioned constraints.
    _claim("t4_tau_dynamic_only_step", _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(43.7, 1.6),
           skip_static=True, dynamic_mode="oneshot",
           notes="Paper 'Dynamic-Only' row: --skip-static so only dynamic "
                 "(prefix-conditioned) invariants generate violations."),
    _claim("t4_tau_dynamic_only_cat",  _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(36.8, 1.6),
           skip_static=True, dynamic_mode="oneshot",
           notes="Paper 'Dynamic-Only' row."),

    # AgentRx: PAPER_DEFAULT (combined+violations-after+with_context=True),
    # one-shot constraint generation (see header note).
    _claim("t4_tau_full_step",         _T4, _T4_NAME, "tau", "step_index_acc",
           PaperValue(48.3, 0.0),
           dynamic_mode="oneshot",
           notes="Paper '\\tool' row: PAPER_DEFAULT + dynamic_mode=oneshot. "
                 "Exact match for tab:ablations Checklist+Vio. one-shot row. "
                 "std=0 in paper means n=3 yielded identical step-index."),
    _claim("t4_tau_full_cat",          _T4, _T4_NAME, "tau", "category_acc",
           PaperValue(39.1, 1.6),
           dynamic_mode="oneshot",
           notes="Paper '\\tool' row: PAPER_DEFAULT + dynamic_mode=oneshot."),
]


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

ALL_CLAIMS: list[PaperClaim] = []
ALL_CLAIMS.extend(_TABLE_4_CLAIMS)


# --------------------------------------------------------------------------
# Table 5 (tab:ablations, eval.tex L139-L194)
# 6 judge columns x 6 row-bands x 3 metrics = 108 cells.
#
# Column-axis mapping (judge configuration), with code citations
# (judge.py / run.py from this branch HEAD):
#
#   Column                | prompt_mode | exec_mode         | with_context
#   ----------------------+-------------+-------------------+-------------
#   Baseline              | baseline    | violations-after  | False
#   Step-then-Cat.        | baseline    | stepbystep        | False
#   Baseline+Vio.         | baseline    | violations-after  | True
#   Step-then-Cat.+Vio.   | baseline    | stepbystep        | True
#   Taxonomy Checklist    | combined    | violations-after  | False
#   Checklist+Vio.        | combined    | violations-after  | True
#
# Rationale for Taxonomy Checklist / Checklist+Vio. using prompt_mode
# "combined" rather than "checklist": tab:static-dynamic's AgentRx row
# (tau, step=48.3, cat=39.1) is the byte-exact match for the Checklist+Vio.
# one-shot row of tab:ablations (tau, step=48.3, cat=39.1 +/- 1.6), and
# the PAPER_DEFAULT recipe uses prompt_mode="combined". The PromptMode
# enum's bare "checklist" variant is a strict subset (taxonomy without
# examples) used in supplementary ablations not in this table; the paper's
# "Taxonomy Checklist" column name refers to the rendered prompt section,
# not the enum value.
#
# Row-band axis (constraint generation):
#   - "One-shot ... Generation" -> dynamic_mode = "oneshot"
#   - "Step-by-Step ... Generation" -> dynamic_mode = "stepbystep"
#
# Magentic is split: One-Shot row uses domain="magentic" (full 44-task
# set), Step-by-Step row uses domain="magentic_star" (27-task filtered
# subset). This is stated explicitly in paper text: "We evaluated the
# step-by-step setting only on Magentic* because of long-horizon
# trajectories."
# --------------------------------------------------------------------------

_T5 = "tab:ablations"
_T5_NAME = "Table 5 (judge x constraint-gen ablations)"


@dataclass(frozen=True)
class _T5Col:
    """One column of tab:ablations: a judge-configuration axis."""
    slug: str           # short id for cell_ids
    label: str          # human label as printed in paper
    prompt_mode: str
    exec_mode: str
    with_context: bool


_T5_COLUMNS: list[_T5Col] = [
    _T5Col("baseline",            "Baseline",
           "baseline", "violations-after", False),
    _T5Col("stepthencat",         "Step-then-Cat.",
           "baseline", "stepbystep",       False),
    _T5Col("baselinevio",         "Baseline+Vio.",
           "baseline", "violations-after", True),
    _T5Col("stepthencatvio",      "Step-then-Cat.+Vio.",
           "baseline", "stepbystep",       True),
    _T5Col("taxonomychecklist",   "Taxonomy Checklist",
           "combined", "violations-after", False),
    _T5Col("checklistvio",        "Checklist+Vio.",
           "combined", "violations-after", True),
]


@dataclass(frozen=True)
class _T5Cell:
    """One numeric cell of tab:ablations as transcribed from eval.tex."""
    mean: float
    std: float | None = None


@dataclass(frozen=True)
class _T5RowBand:
    """One row-band of tab:ablations: domain + constraint-gen strategy +
    three metric rows, each with six column values."""
    domain: Domain
    dynamic_mode: DynamicMode
    slug: str           # short id, e.g. "tau_oneshot"
    # Each list is exactly len(_T5_COLUMNS) long.
    step_index_acc: list[_T5Cell]
    category_acc: list[_T5Cell]
    avg_step_distance: list[_T5Cell]


# Verbatim transcription of eval.tex L160-L194. Cell order matches
# _T5_COLUMNS: Baseline, Step-then-Cat., Baseline+Vio., Step-then-Cat.+Vio.,
# Taxonomy Checklist, Checklist+Vio.
_T5_ROW_BANDS: list[_T5RowBand] = [
    # --- Tau-Bench / One-shot ---
    _T5RowBand(
        domain="tau", dynamic_mode="oneshot", slug="tau_oneshot",
        step_index_acc=[
            _T5Cell(32.2, 3.2), _T5Cell(32.2, 1.6), _T5Cell(47.1, 1.6),
            _T5Cell(54.0, 1.6), _T5Cell(32.2, 1.6), _T5Cell(48.3, None),
        ],
        category_acc=[
            _T5Cell(25.3, 1.6), _T5Cell(27.6, 2.8), _T5Cell(37.9, 2.8),
            _T5Cell(40.2, 1.6), _T5Cell(25.3, 1.6), _T5Cell(39.1, 1.6),
        ],
        avg_step_distance=[
            _T5Cell(5.7, 0.8),  _T5Cell(6.0, 1.0),  _T5Cell(2.8, 0.3),
            _T5Cell(2.4, 0.5),  _T5Cell(5.8, 0.7),  _T5Cell(3.0, 0.3),
        ],
    ),
    # --- Tau-Bench / Step-by-Step ---
    _T5RowBand(
        domain="tau", dynamic_mode="stepbystep", slug="tau_stepbystep",
        step_index_acc=[
            _T5Cell(32.2, 3.2), _T5Cell(32.2, 1.6), _T5Cell(41.4, 2.8),
            _T5Cell(36.8, 1.6), _T5Cell(32.2, 1.6), _T5Cell(37.9, 2.8),
        ],
        category_acc=[
            _T5Cell(25.3, 1.6), _T5Cell(27.6, 2.8), _T5Cell(35.6, 1.6),
            _T5Cell(34.5, None), _T5Cell(25.3, 1.6), _T5Cell(35.6, 1.6),
        ],
        avg_step_distance=[
            _T5Cell(5.7, 0.8),  _T5Cell(6.0, 1.0),  _T5Cell(3.3, 0.6),
            _T5Cell(4.1, 0.1),  _T5Cell(5.8, 0.7),  _T5Cell(3.6, 0.1),
        ],
    ),
    # --- Flash / One-shot ---
    _T5RowBand(
        domain="flash", dynamic_mode="oneshot", slug="flash_oneshot",
        step_index_acc=[
            _T5Cell(80.9, 2.3), _T5Cell(73.0, 1.3), _T5Cell(81.8, 1.3),
            _T5Cell(70.6, 2.3), _T5Cell(83.3, 2.3), _T5Cell(80.1, 1.3),
        ],
        category_acc=[
            _T5Cell(53.9, 3.6), _T5Cell(55.5, 3.6), _T5Cell(53.9, 7.6),
            _T5Cell(52.3, 2.3), _T5Cell(57.9, 2.7), _T5Cell(58.0, 3.6),
        ],
        avg_step_distance=[
            _T5Cell(0.25, 0.2), _T5Cell(0.34, 1.3), _T5Cell(0.2, None),
            _T5Cell(0.36, None), _T5Cell(0.2, 0.3), _T5Cell(0.2, None),
        ],
    ),
    # --- Flash / Step-by-Step ---
    _T5RowBand(
        domain="flash", dynamic_mode="stepbystep", slug="flash_stepbystep",
        step_index_acc=[
            _T5Cell(80.9, 2.3), _T5Cell(73.0, 1.3), _T5Cell(76.2, None),
            _T5Cell(68.2, 5.4), _T5Cell(83.3, 2.3), _T5Cell(76.1, 2.3),
        ],
        category_acc=[
            _T5Cell(53.9, 3.6), _T5Cell(55.5, 3.6), _T5Cell(57.1, 6.2),
            _T5Cell(59.5, 2.3), _T5Cell(57.9, 2.7), _T5Cell(60.3, 1.3),
        ],
        avg_step_distance=[
            _T5Cell(0.25, 0.2), _T5Cell(0.34, 1.3), _T5Cell(0.3, None),
            _T5Cell(0.4, None), _T5Cell(0.2, 0.3), _T5Cell(0.3, None),
        ],
    ),
    # --- Magentic / One-shot ---
    _T5RowBand(
        domain="magentic", dynamic_mode="oneshot", slug="magentic_oneshot",
        step_index_acc=[
            _T5Cell(31.8, None), _T5Cell(29.5, 2.3), _T5Cell(25.0, 2.3),
            _T5Cell(27.3, 1.3),  _T5Cell(31.8, None), _T5Cell(24.2, 1.3),
        ],
        category_acc=[
            _T5Cell(36.4, 3.6), _T5Cell(37.8, 5.7), _T5Cell(31.1, 3.5),
            _T5Cell(34.1, 4.5), _T5Cell(37.1, 5.7), _T5Cell(25.0, 1.3),
        ],
        avg_step_distance=[
            _T5Cell(22.0, 1.0), _T5Cell(13.4, 2.0), _T5Cell(28.0, 1.2),
            _T5Cell(13.7, 2.3), _T5Cell(25.5, 1.2), _T5Cell(28.3, 1.0),
        ],
    ),
    # --- Magentic* / Step-by-Step ---
    _T5RowBand(
        domain="magentic_star", dynamic_mode="stepbystep", slug="magenticstar_stepbystep",
        step_index_acc=[
            _T5Cell(42.0, 1.8), _T5Cell(40.7, None), _T5Cell(45.7, 1.8),
            _T5Cell(40.7, 5.2), _T5Cell(42.0, 1.8), _T5Cell(46.9, 3.5),
        ],
        category_acc=[
            _T5Cell(39.5, 3.5), _T5Cell(40.7, 5.2), _T5Cell(43.2, 1.8),
            _T5Cell(42.0, 1.8), _T5Cell(35.8, 1.8), _T5Cell(44.4, 3.0),
        ],
        avg_step_distance=[
            _T5Cell(5.0, 0.7), _T5Cell(4.9, 0.8), _T5Cell(4.9, 0.9),
            _T5Cell(5.2, 0.4), _T5Cell(6.6, 0.3), _T5Cell(4.8, 0.8),
        ],
    ),
]


def _build_table_5_claims() -> list[PaperClaim]:
    out: list[PaperClaim] = []
    metric_rows: list[tuple[str, str]] = [
        ("step_index_acc",    "step"),
        ("category_acc",      "cat"),
        ("avg_step_distance", "dist"),
    ]
    for band in _T5_ROW_BANDS:
        cells_by_metric: dict[str, list[_T5Cell]] = {
            "step_index_acc":    band.step_index_acc,
            "category_acc":      band.category_acc,
            "avg_step_distance": band.avg_step_distance,
        }
        for metric_name, metric_slug in metric_rows:
            cells = cells_by_metric[metric_name]
            assert len(cells) == len(_T5_COLUMNS), (
                f"row band {band.slug} metric {metric_name}: expected "
                f"{len(_T5_COLUMNS)} cells, got {len(cells)}"
            )
            for col, cell in zip(_T5_COLUMNS, cells):
                cell_id = f"t5_{band.slug}_{col.slug}_{metric_slug}"
                unit = "percent" if metric_name != "avg_step_distance" else "steps"
                notes = (
                    f"tab:ablations row band {band.slug!r}, column "
                    f"{col.label!r}. dynamic_mode={band.dynamic_mode}; "
                    f"prompt_mode={col.prompt_mode}, exec_mode="
                    f"{col.exec_mode}, with_context={col.with_context}."
                )
                if col.slug in ("taxonomychecklist", "checklistvio"):
                    notes += (
                        " 'Taxonomy Checklist' column resolved to "
                        "prompt_mode='combined' (see module-level rationale "
                        "in paper_claims.py)."
                    )
                out.append(_claim(
                    cell_id, _T5, _T5_NAME, band.domain, metric_name,
                    PaperValue(cell.mean, cell.std, unit=unit),
                    prompt_mode=col.prompt_mode,
                    exec_mode=col.exec_mode,
                    with_context=col.with_context,
                    dynamic_mode=band.dynamic_mode,
                    notes=notes,
                ))
    return out


_TABLE_5_CLAIMS: list[PaperClaim] = _build_table_5_claims()
ALL_CLAIMS.extend(_TABLE_5_CLAIMS)

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
