"""Canonical endpoint registry for AgentRx.

This is the single source of truth for which Azure OpenAI deployments live on
which endpoint. Loads ``config/endpoints.json`` and exposes typed helpers:

  - ``get_default_endpoints()`` -> the gpt-5 endpoint pool used by sweeps
  - ``get_endpoints_for_model(model)`` -> every URL that hosts ``model``
  - ``resolve_deployment(model, endpoint_url)`` -> (deployment, api_version)
  - ``list_models()`` -> every distinct model across all endpoints
  - ``describe(model=None)`` -> human-readable inventory summary

Scripts and clients MUST import from here instead of inlining endpoint URLs.
A grep for ``openai.azure.com`` across the repo should find matches only in:

  - ``config/endpoints.json`` (the registry itself)
  - ``.env`` / ``.env.example`` (runtime configuration, gitignored)
  - ``runs/`` artifacts (legitimate provenance from past runs)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


# config/endpoints.json sits at the repo root: <repo>/config/endpoints.json
# This file lives at <repo>/agentrx/llm_clients/endpoint_registry.py, so:
#   parents[0] = agentrx/llm_clients/
#   parents[1] = agentrx/
#   parents[2] = <repo root>
_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "endpoints.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not _REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Endpoint registry not found at {_REGISTRY_PATH}. "
            "Expected config/endpoints.json at the AgentRx repo root."
        )
    return json.loads(_REGISTRY_PATH.read_text())


def get_default_endpoints() -> list[str]:
    """Return the gpt-5 endpoint pool used by sweeps."""
    return list(_load()["default_endpoints_for_gpt5"])


def get_default_model() -> str:
    return str(_load()["default_model"])


def get_default_api_version() -> str:
    return str(_load()["default_api_version"])


def get_endpoints_for_model(model_name: str) -> list[str]:
    """Return every endpoint URL that has ``model_name`` as a deployment.

    The returned URLs all end in a trailing slash. Order is registry order.
    """
    out: list[str] = []
    for ep in _load()["endpoints"]:
        if model_name in ep.get("deployments", {}):
            out.append(ep["url"])
    return out


def resolve_deployment(model_name: str, endpoint_url: str) -> tuple[str, str]:
    """Return ``(deployment_name, api_version)`` for ``model_name`` on
    ``endpoint_url``. Raises KeyError if the model is not deployed there or
    the endpoint is unknown."""
    url_norm = endpoint_url.rstrip("/") + "/"
    for ep in _load()["endpoints"]:
        if ep["url"].rstrip("/") + "/" == url_norm:
            dep = ep["deployments"].get(model_name)
            if dep is None:
                raise KeyError(
                    f"Model {model_name!r} not deployed on {endpoint_url!r}. "
                    f"Available on this endpoint: "
                    f"{sorted(ep['deployments'].keys())}"
                )
            return dep["deployment"], dep["api_version"]
    raise KeyError(f"Endpoint {endpoint_url!r} not found in registry")


def get_endpoint_by_name(name: str) -> dict:
    """Return the full registry entry dict for the endpoint named ``name``.

    The ``name`` is the registry's ``name`` field (typically the Azure resource
    name, e.g. ``aiops-llm-eus2``). Raises KeyError if not found."""
    for ep in _load()["endpoints"]:
        if ep["name"] == name:
            return dict(ep)
    raise KeyError(f"Endpoint name {name!r} not found in registry")


def list_endpoints() -> list[dict]:
    """Return all endpoint entries as a fresh list of dicts."""
    return [dict(ep) for ep in _load()["endpoints"]]


def list_models() -> list[str]:
    """Return every distinct model name across all endpoints, sorted."""
    seen: set[str] = set()
    for ep in _load()["endpoints"]:
        seen.update(ep.get("deployments", {}).keys())
    return sorted(seen)


def describe(model_name: Optional[str] = None) -> str:
    """Human-readable inventory summary. If ``model_name`` is given, scope
    output to that model only."""
    reg = _load()
    lines: list[str] = []
    if model_name is None:
        lines.append(
            f"Endpoint registry: {len(reg['endpoints'])} endpoints, "
            f"{len(list_models())} distinct models"
        )
        lines.append(f"Default model: {reg['default_model']}")
        lines.append("")
        for ep in reg["endpoints"]:
            models = ", ".join(sorted(ep["deployments"].keys()))
            lines.append(
                f"  {ep['name']:24s} {ep['location']:18s} {models}"
            )
    else:
        hits = get_endpoints_for_model(model_name)
        lines.append(
            f"Model {model_name!r} is deployed on {len(hits)} endpoint(s):"
        )
        for url in hits:
            dep, api = resolve_deployment(model_name, url)
            lines.append(f"  {url:60s} deployment={dep:24s} api={api}")
    return "\n".join(lines)


def _reset_cache_for_tests() -> None:
    """Test hook to reload the registry after a test rewrites the file."""
    _load.cache_clear()


if __name__ == "__main__":
    import sys
    print(describe(sys.argv[1] if len(sys.argv) > 1 else None))
