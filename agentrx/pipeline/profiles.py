"""Centralized judge-stage configuration.

The judge stage exposes a small set of orthogonal axes that the AgentRx paper
varies per table cell. They are bundled into a single frozen dataclass so that
every call site (run.py, downstream sweep scripts, tests) refers to one source
of truth instead of repeating string literals.

There is no profile *registry* by design. The dataclass IS the centralization
primitive: paper-default recipe lives in PAPER_DEFAULT, and ablations are
expressed as `dataclasses.replace(PAPER_DEFAULT, exec_mode="stepbystep")`.
Orthogonal pipeline-stage knobs (dynamic-mode, skip-static, ...) deliberately
remain on their own CLI flags rather than being bundled into a named recipe.
"""
from dataclasses import dataclass
from typing import Literal

PromptMode = Literal["baseline", "checklist", "examples", "combined"]
ExecMode = Literal["violations-after", "stepbystep", "violations-before"]


@dataclass(frozen=True)
class RunConfig:
    """Judge-stage knobs needed to reproduce a paper table cell.

    prompt_mode   which taxonomy block the judge sees (see judge.py::build_taxonomy_text)
    exec_mode     when violations are surfaced relative to category labelling
    with_context  inject deduplicated violation context into the judge prompt
                  (runtime value also requires the upstream check stage to have
                  produced a context directory; see run.py::run_judge)
    """
    prompt_mode: PromptMode
    exec_mode: ExecMode
    with_context: bool


PAPER_DEFAULT = RunConfig(
    prompt_mode="combined",
    exec_mode="violations-after",
    with_context=True,
)
