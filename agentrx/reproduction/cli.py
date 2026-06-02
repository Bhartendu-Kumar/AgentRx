"""Command-line wrapper around the paper-claims catalog.

Usage examples:

    python -m agentrx.reproduction list
    python -m agentrx.reproduction list --table tab:static-dynamic
    python -m agentrx.reproduction show t4_tau_full_cat
    python -m agentrx.reproduction cmd  t4_tau_full_cat --input data/tau_dataset/foo.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

from agentrx.reproduction.paper_claims import (
    ALL_CLAIMS,
    PaperClaim,
    claims_for,
    get_claim,
)


def _format_claim_line(c: PaperClaim) -> str:
    pv = c.paper_value
    pv_str = "?" if pv is None else (
        f"{pv.mean}" + (f" +/- {pv.std}" if pv.std is not None else "")
    )
    return (
        f"  {c.cell_id:<36} {c.table_label:<22} {c.domain:<14} "
        f"{c.metric:<22} paper={pv_str}"
    )


def _cmd_list(args: argparse.Namespace) -> int:
    claims = claims_for(
        table_label=args.table,
        domain=args.domain,
        metric=args.metric,
    )
    if not claims:
        print("(no claims matched)")
        return 0
    print(f"# {len(claims)} claim(s):")
    for c in claims:
        print(_format_claim_line(c))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    c = get_claim(args.cell_id)
    payload = {
        "cell_id": c.cell_id,
        "table_label": c.table_label,
        "table_short_name": c.table_short_name,
        "domain": c.domain,
        "metric": c.metric,
        "paper_value": (None if c.paper_value is None
                        else dataclasses.asdict(c.paper_value)),
        "run_config": dataclasses.asdict(c.run_config),
        "dynamic_mode": c.dynamic_mode,
        "skip_static": c.skip_static,
        "skip_dynamic": c.skip_dynamic,
        "input_glob": c.input_glob(),
        "ground_truth_path": c.ground_truth_path(),
        "subset_ids_file": c.subset_ids_file(),
        "notes": c.notes,
    }
    print(json.dumps(payload, indent=2))
    return 0


def claim_to_cmdline(c: PaperClaim, *, input_path: str, run_name: str | None = None,
                     endpoint: str = "azure") -> list[str]:
    """Render the ``run.py`` command line that reproduces a single claim.

    Returns the argv list (NOT a shell string) so callers can either
    subprocess.run(...) it directly or shlex.join(...) it for display.
    The first element is ``"python"``; the caller may swap that for a
    specific interpreter path.
    """
    cfg = c.run_config
    argv: list[str] = [
        "python", "run.py", input_path,
        "--domain", "tau" if c.domain == "tau" else ("magentic" if c.domain.startswith("magentic") else "flash"),
        "--endpoint", endpoint,
        "--prompt-mode", cfg.prompt_mode,
        "--exec-mode", cfg.exec_mode,
        "--num-runs", str(cfg.num_runs),
        "--prompt-style", cfg.prompt_style,
        "--python-check-timeout-sec", str(cfg.python_check_timeout_sec),
        "--dynamic-mode", c.dynamic_mode,
    ]
    if not cfg.with_context:
        argv.append("--no-context")
    if cfg.include_nl_check_violations:
        argv.append("--include-nl-violations")
    else:
        argv.append("--exclude-nl-violations")
    if cfg.skip_nl:
        argv.append("--skip-nl-checks")
    if c.skip_static:
        argv.append("--skip-static")
    if c.skip_dynamic:
        argv.append("--skip-dynamic")
    argv += ["--ground-truth", c.ground_truth_path()]
    subset = c.subset_ids_file()
    if subset is not None:
        argv += ["--subset-ids", subset]
    if run_name:
        argv += ["--run-name", run_name]
    return argv


def _cmd_cmd(args: argparse.Namespace) -> int:
    c = get_claim(args.cell_id)
    argv = claim_to_cmdline(c, input_path=args.input, run_name=args.run_name,
                            endpoint=args.endpoint)
    print(shlex.join(argv))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    # Imported lazily so ``import agentrx.reproduction.cli`` stays cheap for
    # the list/show/cmd subcommands used by tests.
    from agentrx.reproduction.sweep import (
        build_manifest,
        dump_manifest,
        execute_sequential,
        group_into_invocations,
        select_claims,
    )

    cell_ids = [s for s in (args.cells.split(",") if args.cells else []) if s]
    selection = {
        "cell_ids": cell_ids or None,
        "table_label": args.table,
        "domain": args.domain,
        "metric": args.metric,
        "dynamic_mode": args.dynamic_mode,
    }
    try:
        claims = select_claims(
            cell_ids=cell_ids or None,
            table_label=args.table,
            domain=args.domain,
            metric=args.metric,
            dynamic_mode=args.dynamic_mode,
        )
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not claims:
        print("error: no claims matched the given filters", file=sys.stderr)
        return 2
    invocations = group_into_invocations(claims)
    manifest = build_manifest(
        invocations,
        runs_root=args.runs_root,
        selection=selection,
        base_dir=args.base_dir,
        endpoint=args.endpoint,
        python=args.python,
    )
    print(f"# Sweep plan: {len(claims)} claim(s) -> {len(invocations)} "
          f"invocation(s) -> {args.runs_root}")
    for inv in invocations:
        cells_preview = ", ".join(inv.cell_ids[:3])
        more = "" if len(inv.claims) <= 3 else f" ... (+{len(inv.claims) - 3} more)"
        print(f"  - {inv.slug}: {len(inv.claims)} cell(s) [{cells_preview}{more}]")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    manifest_path = dump_manifest(manifest, args.runs_root)
    print(f"# Wrote manifest: {manifest_path}")
    results = execute_sequential(
        invocations,
        runs_root=args.runs_root,
        base_dir=args.base_dir,
        python=args.python,
        endpoint=args.endpoint,
        resume=not args.no_resume,
    )
    failed = [r for r in results if r.returncode not in (0, None)]
    skipped = sum(r.skipped for r in results)
    print(f"# Sweep complete: {len(results)} invocation(s), "
          f"{skipped} skipped, {len(failed)} failed")
    return 1 if failed else 0


def _cmd_report(args: argparse.Namespace) -> int:
    from agentrx.reproduction.aggregator import (
        aggregate_sweep,
        render_cell_reports_markdown,
        report_to_json,
    )

    reports = aggregate_sweep(args.manifest)
    if args.format == "json":
        print(report_to_json(reports))
    elif args.format == "markdown":
        print(render_cell_reports_markdown(reports), end="")
    elif args.format == "summary":
        n_ok = sum(1 for r in reports if r.status == "OK")
        n_within = sum(1 for r in reports if r.within_paper_std is True)
        n_outside = sum(1 for r in reports if r.within_paper_std is False)
        n_no_std = sum(1 for r in reports
                       if r.status == "OK" and r.within_paper_std is None)
        n_missing_rd = sum(1 for r in reports if r.status == "MISSING_RUN_DIR")
        n_missing_sum = sum(1 for r in reports if r.status == "MISSING_SUMMARY")
        n_missing_metric = sum(1 for r in reports if r.status == "METRIC_MISSING")
        print(f"# Aggregator summary ({len(reports)} cells, manifest={args.manifest})")
        print(f"  OK:                  {n_ok}")
        print(f"  Within paper std:    {n_within}")
        print(f"  Outside paper std:   {n_outside}")
        print(f"  OK but no paper std: {n_no_std}")
        print(f"  Missing run dir:     {n_missing_rd}")
        print(f"  Missing summary:     {n_missing_sum}")
        print(f"  Metric not projected:{n_missing_metric}")
    else:  # pragma: no cover - argparse 'choices' prevents this
        raise AssertionError(f"unknown format {args.format!r}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Walk a directory of judge summaries and check project_metric on every one.

    Diagnoses schema drift between what the judge writes and what the
    aggregator reads. Distinct from `report`: `report` projects the
    catalog's metrics onto a planned sweep; `audit` runs every supported
    metric against every summary it finds, regardless of catalog.

    Exit code:
        0  no summary raised an unexpected exception
        1  at least one UNEXPECTED exception, OR a value was out of bounds
           (percent outside [0,100], negative std, negative distance)
        2  no summaries found at all (likely a wrong --runs-root)
    """
    from agentrx.reproduction.aggregator import MetricMissing, project_metric

    METRICS = [
        "step_index_acc", "category_acc", "avg_step_distance",
        "step_acc_at_plus_minus_1", "step_acc_at_plus_minus_3",
        "step_acc_at_plus_minus_5",
    ]
    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"error: --runs-root {root} is not a directory", file=sys.stderr)
        return 2

    files: list[Path] = []
    for p in root.rglob("summary.json"):
        if "judge_output/analysis" in str(p):
            files.append(p)
    if not files:
        print(f"error: no judge_output/analysis/summary.json files under {root}",
              file=sys.stderr)
        return 2

    n_ok = 0
    n_metric_missing = 0
    n_unexpected = 0
    n_out_of_bounds = 0
    corrupt: list[tuple[Path, str]] = []
    failures: list[tuple[Path, str, str]] = []

    for p in files:
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            corrupt.append((p, repr(e)))
            continue
        for m in METRICS:
            try:
                mean, std, _ = project_metric(m, d)
            except MetricMissing:
                n_metric_missing += 1
                continue
            except Exception as e:  # noqa: BLE001 - audit MUST catch everything
                n_unexpected += 1
                failures.append((p, m, f"{type(e).__name__}: {e}"))
                continue
            ok = True
            if m == "avg_step_distance":
                if mean < 0:
                    failures.append((p, m, f"negative distance {mean}"))
                    ok = False
            else:
                if not (0.0 <= mean <= 100.0):
                    failures.append((p, m, f"percent out of [0,100]: {mean}"))
                    ok = False
            if std < 0:
                failures.append((p, m, f"negative std {std}"))
                ok = False
            if not ok:
                n_out_of_bounds += 1
            else:
                n_ok += 1

    print(f"# Audit: {len(files)} summary.json file(s) under {root}")
    print(f"  corrupt JSON:        {len(corrupt)}")
    print(f"  metric+file pairs OK:           {n_ok}")
    print(f"  MetricMissing (expected misses):{n_metric_missing}")
    print(f"  UNEXPECTED exceptions:          {n_unexpected}")
    print(f"  out-of-bounds values:           {n_out_of_bounds}")
    if args.verbose:
        for p, m, msg in failures[:20]:
            print(f"  FAIL  {p}  [{m}]  {msg}")
        for p, msg in corrupt[:5]:
            print(f"  CORRUPT  {p}  {msg}")
    return 1 if (n_unexpected or n_out_of_bounds or corrupt) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentrx.reproduction",
        description="Query the paper-claims catalog and emit reproduction commands.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List claims (optionally filtered).")
    p_list.add_argument("--table", default=None, help="LaTeX label, e.g. tab:static-dynamic")
    p_list.add_argument("--domain", default=None, choices=["tau", "magentic", "magentic_star", "flash"])
    p_list.add_argument("--metric", default=None)
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Print one claim as JSON.")
    p_show.add_argument("cell_id")
    p_show.set_defaults(func=_cmd_show)

    p_cmd = sub.add_parser("cmd", help="Emit the run.py command line for one claim.")
    p_cmd.add_argument("cell_id")
    p_cmd.add_argument("--input", required=True, help="Trajectory file path.")
    p_cmd.add_argument("--run-name", default=None)
    p_cmd.add_argument("--endpoint", default="azure", choices=["azure", "trapi", "copilot"])
    p_cmd.set_defaults(func=_cmd_cmd)

    p_sweep = sub.add_parser(
        "sweep",
        help="Run a sweep of catalog claims via run.py, with invocation "
             "grouping and a manifest.",
    )
    p_sweep.add_argument("--runs-root", required=True,
                         help="Output root directory; manifest.json and "
                              "invocations/<slug>/ live under here.")
    p_sweep.add_argument("--cells", default=None,
                         help="Comma-separated cell_ids to run.")
    p_sweep.add_argument("--table", default=None,
                         help="Filter by LaTeX table label, e.g. tab:ablations.")
    p_sweep.add_argument("--domain", default=None,
                         choices=["tau", "magentic", "magentic_star", "flash"])
    p_sweep.add_argument("--metric", default=None)
    p_sweep.add_argument("--dynamic-mode", default=None,
                         choices=["oneshot", "stepbystep"])
    p_sweep.add_argument("--endpoint", default="azure",
                         choices=["azure", "trapi", "copilot"])
    p_sweep.add_argument("--base-dir", default=".",
                         help="Repo root that contains run.py and data/ "
                              "(default: current working directory).")
    p_sweep.add_argument("--python", default=sys.executable,
                         help="Python interpreter to invoke run.py with "
                              "(default: this interpreter).")
    p_sweep.add_argument("--dry-run", action="store_true",
                         help="Print the plan and manifest JSON; do not write "
                              "or execute anything.")
    p_sweep.add_argument("--no-resume", action="store_true",
                         help="Re-run invocations even if their state ledger "
                              "already records the judge stage as complete.")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_report = sub.add_parser(
        "report",
        help="Aggregate a completed sweep's judge outputs into a per-cell "
             "paper-vs-observed report.",
    )
    p_report.add_argument("--manifest", required=True,
                          help="Path to a sweep manifest.json.")
    p_report.add_argument("--format", default="summary",
                          choices=["summary", "json", "markdown"],
                          help="summary: human counts; json: full structured "
                               "reports; markdown: one table row per cell.")
    p_report.set_defaults(func=_cmd_report)

    p_audit = sub.add_parser(
        "audit",
        help="Walk a runs-root and project every supported metric against "
             "every judge summary.json; report schema drift / bad values.",
    )
    p_audit.add_argument("--runs-root", required=True,
                         help="Directory to walk recursively for "
                              "judge_output/analysis/summary.json files.")
    p_audit.add_argument("--verbose", action="store_true",
                         help="Also list the first 20 failures.")
    p_audit.set_defaults(func=_cmd_audit)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
