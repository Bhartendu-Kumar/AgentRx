"""``ModelSpec``: a single typed representation of "which model, where, how".

Background. The judge / IR / invariant pipelines all need to call an LLM,
but the *exact* model invocation is currently scattered across:

  - ``agentrx.pipeline.globals`` env-var reads (g.ENDPOINT, g.DEPLOYMENT,
    g.API_VERSION, g.TRAPI_*, ...)
  - per-client field assignments in ``azure.py`` / ``trapi.py`` /
    ``copilot_cli.py`` that re-derive the same identifiers from globals
  - the inline dict that ``provenance._resolve_model_spec`` emits to
    ``runs/<x>/run_config.json``
  - ad-hoc hardcoded URL tables (e.g.
    ``scripts/run_tau_ablation.py:ENDPOINTS``)

This module is the unification: one dataclass with one ``to_provenance_dict``
contract that exactly preserves the wire format the provenance file already
uses. New code (e.g. sweeps that need to pin per-cell endpoints) constructs
``ModelSpec`` via :func:`from_registry`; legacy code paths continue to work
because :func:`from_globals` reproduces the prior behaviour byte-for-byte.

What this module deliberately does NOT do
-----------------------------------------
- It does not yet replace the per-client constructors. ``LLMAgent.__init__``
  still reads globals. The migration to spec-based construction is a
  separate change with its own test surface.
- It does not perform any I/O (no token requests, no LLM calls). It is a
  pure dataclass plus a small set of factories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import agentrx.pipeline.globals as g
from agentrx.llm_clients import endpoint_registry as _registry

EndpointKind = Literal["azure", "trapi", "copilot"]


@dataclass(frozen=True)
class ModelSpec:
    """Typed identifier of one LLM invocation target.

    Fields
    ------
    endpoint_kind
        Which client backend is going to be used.
    endpoint_url
        Base URL of the endpoint, with trailing slash. ``None`` for the
        ``copilot`` backend, which has no Azure-shaped endpoint.
    api_version
        Azure / TRAPI ``api-version`` query parameter.
    deployment_name
        The Azure / TRAPI deployment alias (what the client passes as
        ``model=`` in chat completions for Azure-shaped endpoints).
    model_name
        The published model id (``gpt-5``, ``gpt-5.1``, ...). Equal to
        ``deployment_name`` when the deployment is named after the model,
        which is the convention everywhere except a small set of aliased
        deployments.
    model_version
        TRAPI-only: pinned model version string (e.g. ``2025-04-16``).
    trapi_instance
        TRAPI-only: instance segment appended to the TRAPI endpoint prefix.
    embedding_model_name
        Azure-only: name of the embedding deployment used by code paths
        that need embeddings (kept here because the provenance schema
        currently surfaces it; downstream consumers may ignore).
    """

    endpoint_kind: EndpointKind
    endpoint_url: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    trapi_instance: Optional[str] = None
    embedding_model_name: Optional[str] = None

    def to_provenance_dict(self) -> dict[str, Any]:
        """Render this spec into the dict shape that
        ``runs/<x>/run_config.json::model_spec`` has used since W1b.

        The set of keys emitted depends on ``endpoint_kind`` to preserve
        the existing wire format byte-for-byte.
        """
        if self.endpoint_kind == "azure":
            return {
                "endpoint_type": "azure",
                "endpoint_url": self.endpoint_url or None,
                "api_version": self.api_version or None,
                "deployment_name": self.deployment_name or None,
                "model_name": self.model_name or None,
                "embedding_model_name": self.embedding_model_name or None,
            }
        if self.endpoint_kind == "trapi":
            return {
                "endpoint_type": "trapi",
                "endpoint_url": self.endpoint_url or None,
                "api_version": self.api_version or None,
                "deployment_name": self.deployment_name or None,
                "model_name": self.model_name or None,
                "model_version": self.model_version or None,
                "trapi_instance": self.trapi_instance or None,
            }
        if self.endpoint_kind == "copilot":
            return {
                "endpoint_type": "copilot",
                "endpoint_url": None,
                "api_version": None,
                "deployment_name": None,
                "model_name": self.model_name or None,
            }
        # Unknown kinds get the minimal envelope so provenance stays writable
        # even if a new endpoint type is added before this module learns it.
        return {"endpoint_type": self.endpoint_kind}


def from_globals(endpoint_kind: EndpointKind) -> ModelSpec:
    """Build a ``ModelSpec`` from the current process-global env-var state.

    This is the path used by ``provenance.build_provenance`` and by every
    legacy code site that relies on ``agentrx.pipeline.globals`` already
    being populated from ``.env``.

    Unknown ``endpoint_kind`` values produce a bare ``ModelSpec`` carrying
    only the kind, matching the historical fall-through behaviour of
    ``provenance._resolve_model_spec``.
    """
    if endpoint_kind == "azure":
        return ModelSpec(
            endpoint_kind="azure",
            endpoint_url=g.ENDPOINT or None,
            api_version=g.API_VERSION or None,
            deployment_name=g.DEPLOYMENT or None,
            model_name=g.MODEL_NAME or None,
            embedding_model_name=g.EMBEDDING_MODEL_NAME or None,
        )
    if endpoint_kind == "trapi":
        return ModelSpec(
            endpoint_kind="trapi",
            endpoint_url=g.TRAPI_ENDPOINT_PREFIX or None,
            api_version=g.TRAPI_API_VERSION or None,
            deployment_name=g.TRAPI_DEPLOYMENT_NAME or None,
            model_name=g.TRAPI_MODEL_NAME or None,
            model_version=g.TRAPI_MODEL_VERSION or None,
            trapi_instance=g.TRAPI_INSTANCE or None,
        )
    if endpoint_kind == "copilot":
        return ModelSpec(
            endpoint_kind="copilot",
            model_name=g.MODEL_NAME or None,
        )
    return ModelSpec(endpoint_kind=endpoint_kind)


def from_registry(
    *,
    model_name: str,
    endpoint_url: str,
    embedding_model_name: Optional[str] = None,
) -> ModelSpec:
    """Build an azure-kind ``ModelSpec`` by resolving deployment + api_version
    from the canonical endpoint registry.

    Use this in sweep scripts that need to pin one cell to one endpoint
    without mutating ``os.environ`` or relying on subprocess re-imports.

    Raises ``KeyError`` if ``endpoint_url`` is not in the registry or
    ``model_name`` is not deployed there.
    """
    deployment, api_version = _registry.resolve_deployment(model_name, endpoint_url)
    return ModelSpec(
        endpoint_kind="azure",
        endpoint_url=endpoint_url.rstrip("/") + "/",
        api_version=api_version,
        deployment_name=deployment,
        model_name=model_name,
        embedding_model_name=embedding_model_name,
    )
