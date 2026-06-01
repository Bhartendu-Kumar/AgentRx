"""``agentrx.llm_clients`` — package marker.

This package collects the per-endpoint LLM client adapters used by the
AgentRx pipeline. The actual clients live in dedicated submodules:

  - :mod:`agentrx.llm_clients.azure`              Azure OpenAI client
  - :mod:`agentrx.llm_clients.trapi`              TRAPI client
  - :mod:`agentrx.llm_clients.copilot_cli`        Copilot CLI client
  - :mod:`agentrx.llm_clients.endpoint_registry`  Canonical endpoint inventory
  - :mod:`agentrx.llm_clients.model_spec`         Typed ``ModelSpec`` dataclass
  - :mod:`agentrx.llm_clients.observed_snapshot`  Observed-model recorder

Historical note. Prior to the W4a cleanup this ``__init__.py`` carried a
1884-line legacy duplicate of :mod:`agentrx.judge.judge` (an alternative
judge entry point that was never wired into ``run.py``, never referenced
by any test, never documented as an entry point, and had drifted out of
sync with the active judge module). It was removed so the canonical judge
entry point lives in exactly one place: :mod:`agentrx.judge.judge`,
invoked via ``run.py``. Submodule imports such as ``from agentrx.llm_clients
import observed_snapshot`` continue to work because Python resolves them
against the submodule files directly, not against this namespace.
"""
