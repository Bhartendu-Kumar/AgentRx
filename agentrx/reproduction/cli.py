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
    if run_name:
        argv += ["--run-name", run_name]
    return argv


def _cmd_cmd(args: argparse.Namespace) -> int:
    c = get_claim(args.cell_id)
    argv = claim_to_cmdline(c, input_path=args.input, run_name=args.run_name,
                            endpoint=args.endpoint)
    print(shlex.join(argv))
    return 0


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

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
