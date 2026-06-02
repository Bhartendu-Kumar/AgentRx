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

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
