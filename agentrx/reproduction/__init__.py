"""Declarative catalog of paper claims and how to reproduce them.

The premise (deficit D2): if you cannot ask the question *"what was the
paper number for tau-bench step-acc, Baseline+Vio., one-shot?"* and get a
single typed answer plus a reproducible command, then "reproduce the paper"
is a phrase, not an engineering task.

This package provides:

  - ``PaperClaim`` and ``PaperValue`` dataclasses — one cell of one paper
    table is one ``PaperClaim``; the paper-published number is one
    ``PaperValue`` attached to it.
  - ``ALL_CLAIMS`` — the enumerated catalog, populated from the camera-ready
    LaTeX (``resources/paper/eval.tex``) by hand. New tables get appended
    as we work through them.
  - ``claims_for(...)`` — query helper.
  - ``cli`` module — ``python -m agentrx.reproduction list/show/cmd ...``
    to enumerate cells and emit the exact ``run.py`` command line that
    reproduces them.
"""
from agentrx.reproduction.paper_claims import (  # noqa: F401
    ALL_CLAIMS,
    Domain,
    Metric,
    PaperClaim,
    PaperValue,
    claims_for,
    cell_id_unique,
)
