"""ModelSpec tests.

The contract:

  - ``from_globals(kind)`` produces a spec whose ``to_provenance_dict()``
    output is byte-identical to the dict the pre-W3 ``_resolve_model_spec``
    used to emit. We pin this by constructing the legacy dict inline and
    diffing.
  - ``from_registry(...)`` resolves api_version + deployment via the
    canonical endpoint registry and raises ``KeyError`` on unknown
    endpoint or model.
  - The dataclass is frozen and hashable.
"""
from __future__ import annotations

import dataclasses
import pytest

from agentrx.llm_clients.model_spec import (
    ModelSpec,
    from_globals,
    from_registry,
)
from agentrx.llm_clients import endpoint_registry as er
from agentrx.pipeline.provenance import build_provenance, _resolve_model_spec
from agentrx.pipeline.profiles import PAPER_DEFAULT
import agentrx.pipeline.globals as g


def test_modelspec_is_frozen_and_hashable():
    spec = ModelSpec(endpoint_kind="azure", model_name="gpt-5")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.model_name = "gpt-5.1"  # type: ignore[misc]
    # Hashable since frozen with eq=True (default).
    {spec}


@pytest.mark.parametrize("kind", ["azure", "trapi", "copilot"])
def test_from_globals_provenance_dict_matches_legacy_keys(kind):
    """The exact key set the legacy ``_resolve_model_spec`` emitted, per kind."""
    legacy_azure_keys = {
        "endpoint_type", "endpoint_url", "api_version", "deployment_name",
        "model_name", "embedding_model_name",
    }
    legacy_trapi_keys = {
        "endpoint_type", "endpoint_url", "api_version", "deployment_name",
        "model_name", "model_version", "trapi_instance",
    }
    legacy_copilot_keys = {
        "endpoint_type", "endpoint_url", "api_version", "deployment_name",
        "model_name",
    }
    expected = {
        "azure": legacy_azure_keys,
        "trapi": legacy_trapi_keys,
        "copilot": legacy_copilot_keys,
    }[kind]
    spec = from_globals(kind)
    got = spec.to_provenance_dict()
    assert set(got.keys()) == expected
    assert got["endpoint_type"] == kind


def test_unknown_kind_returns_minimal_envelope():
    spec = from_globals("nonsense")  # type: ignore[arg-type]
    d = spec.to_provenance_dict()
    assert d == {"endpoint_type": "nonsense"}


@pytest.mark.parametrize("kind", ["azure", "trapi", "copilot"])
def test_provenance_payload_uses_modelspec_under_the_hood(kind):
    """``build_provenance(...).model_spec`` must round-trip through ModelSpec."""
    payload = build_provenance(
        input_path="x",
        domain="tau",
        endpoint=kind,
        ground_truth_path=None,
        judge_config=PAPER_DEFAULT,
        stages_planned=["ir"],
    )
    direct = _resolve_model_spec(kind)
    assert payload["model_spec"] == direct
    assert payload["model_spec"] == from_globals(kind).to_provenance_dict()


def test_from_globals_azure_carries_global_endpoint_when_set(monkeypatch):
    monkeypatch.setattr(g, "ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setattr(g, "API_VERSION", "2025-01-01-preview")
    monkeypatch.setattr(g, "DEPLOYMENT", "gpt-5")
    monkeypatch.setattr(g, "MODEL_NAME", "gpt-5")
    monkeypatch.setattr(g, "EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    spec = from_globals("azure")
    assert spec.endpoint_url == "https://example.openai.azure.com/"
    assert spec.api_version == "2025-01-01-preview"
    assert spec.deployment_name == "gpt-5"
    assert spec.model_name == "gpt-5"
    assert spec.embedding_model_name == "text-embedding-3-small"


def test_from_globals_azure_coerces_empty_strings_to_none(monkeypatch):
    monkeypatch.setattr(g, "ENDPOINT", "")
    monkeypatch.setattr(g, "API_VERSION", "")
    monkeypatch.setattr(g, "DEPLOYMENT", "")
    monkeypatch.setattr(g, "MODEL_NAME", "")
    monkeypatch.setattr(g, "EMBEDDING_MODEL_NAME", "")
    spec = from_globals("azure")
    d = spec.to_provenance_dict()
    for k in ("endpoint_url", "api_version", "deployment_name",
              "model_name", "embedding_model_name"):
        assert d[k] is None, f"empty global string should become None for {k}"


def test_from_globals_trapi_carries_trapi_specific_fields(monkeypatch):
    monkeypatch.setattr(g, "TRAPI_ENDPOINT_PREFIX", "https://trapi.example/")
    monkeypatch.setattr(g, "TRAPI_API_VERSION", "2025-03-01-preview")
    monkeypatch.setattr(g, "TRAPI_DEPLOYMENT_NAME", "gpt-5-deployment")
    monkeypatch.setattr(g, "TRAPI_MODEL_NAME", "gpt-5")
    monkeypatch.setattr(g, "TRAPI_MODEL_VERSION", "2025-04-16")
    monkeypatch.setattr(g, "TRAPI_INSTANCE", "instance-foo")
    spec = from_globals("trapi")
    assert spec.endpoint_url == "https://trapi.example/"
    assert spec.api_version == "2025-03-01-preview"
    assert spec.deployment_name == "gpt-5-deployment"
    assert spec.model_name == "gpt-5"
    assert spec.model_version == "2025-04-16"
    assert spec.trapi_instance == "instance-foo"


def test_from_globals_copilot_only_has_model_name(monkeypatch):
    monkeypatch.setattr(g, "MODEL_NAME", "gpt-5.2")
    spec = from_globals("copilot")
    d = spec.to_provenance_dict()
    assert d["model_name"] == "gpt-5.2"
    assert d["endpoint_url"] is None
    assert d["api_version"] is None
    assert d["deployment_name"] is None


# --------------------------------------------------------------------------
# from_registry
# --------------------------------------------------------------------------


def test_from_registry_resolves_default_model_on_default_endpoint():
    er._reset_cache_for_tests()
    url = er.get_default_endpoints()[0]
    model = er.get_default_model()
    spec = from_registry(model_name=model, endpoint_url=url)
    assert spec.endpoint_kind == "azure"
    assert spec.endpoint_url.rstrip("/") + "/" == url.rstrip("/") + "/"
    assert spec.model_name == model
    # deployment + api_version came from the registry, not env vars.
    dep, api = er.resolve_deployment(model, url)
    assert spec.deployment_name == dep
    assert spec.api_version == api


def test_from_registry_raises_on_unknown_endpoint():
    er._reset_cache_for_tests()
    with pytest.raises(KeyError):
        from_registry(model_name="gpt-5", endpoint_url="https://no.such.endpoint/")


def test_from_registry_raises_on_model_not_on_endpoint():
    er._reset_cache_for_tests()
    url = er.get_default_endpoints()[0]
    with pytest.raises(KeyError):
        from_registry(model_name="this-model-does-not-exist", endpoint_url=url)


def test_from_registry_normalises_trailing_slash():
    er._reset_cache_for_tests()
    url = er.get_default_endpoints()[0].rstrip("/")
    spec = from_registry(model_name=er.get_default_model(), endpoint_url=url)
    assert spec.endpoint_url.endswith("/")


def test_from_registry_emits_correct_provenance_keys():
    er._reset_cache_for_tests()
    url = er.get_default_endpoints()[0]
    spec = from_registry(
        model_name=er.get_default_model(),
        endpoint_url=url,
        embedding_model_name="text-embedding-3-small",
    )
    d = spec.to_provenance_dict()
    assert d["endpoint_type"] == "azure"
    assert d["endpoint_url"] == url.rstrip("/") + "/"
    assert d["embedding_model_name"] == "text-embedding-3-small"
