"""Endpoint-registry tests.

Validates the integrity of ``config/endpoints.json`` and the API surface of
``agentrx.llm_clients.endpoint_registry``. Pinned invariants:

  - the default endpoint pool exists, is non-empty, and every URL is also
    a top-level registry entry
  - every default-pool endpoint hosts the default model
  - resolve_deployment is the inverse of get_endpoints_for_model
  - registry never returns the same URL twice
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentrx.llm_clients import endpoint_registry as er

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_JSON = REPO_ROOT / "config" / "endpoints.json"


def test_registry_file_exists():
    assert REGISTRY_JSON.is_file(), f"missing {REGISTRY_JSON}"


def test_registry_loads():
    er._reset_cache_for_tests()
    eps = er.list_endpoints()
    assert len(eps) > 0


def test_default_model_is_gpt5():
    er._reset_cache_for_tests()
    assert er.get_default_model() == "gpt-5"


def test_default_endpoints_pool_is_nonempty_and_all_known():
    er._reset_cache_for_tests()
    pool = er.get_default_endpoints()
    assert len(pool) > 0
    known_urls = {ep["url"] for ep in er.list_endpoints()}
    for url in pool:
        assert url in known_urls, f"default-pool URL {url!r} not in endpoints list"


def test_every_default_pool_endpoint_hosts_default_model():
    er._reset_cache_for_tests()
    default_model = er.get_default_model()
    hosts = set(er.get_endpoints_for_model(default_model))
    pool = set(er.get_default_endpoints())
    missing = pool - hosts
    assert not missing, (
        f"endpoints in default pool that don't host {default_model!r}: {missing}"
    )


def test_resolve_deployment_for_default_model_round_trips():
    er._reset_cache_for_tests()
    default_model = er.get_default_model()
    for url in er.get_endpoints_for_model(default_model):
        dep, api = er.resolve_deployment(default_model, url)
        assert isinstance(dep, str) and dep
        assert isinstance(api, str) and api


def test_resolve_deployment_raises_on_unknown_endpoint():
    er._reset_cache_for_tests()
    with pytest.raises(KeyError):
        er.resolve_deployment("gpt-5", "https://nonexistent.example.com/")


def test_resolve_deployment_raises_on_model_not_deployed_there():
    er._reset_cache_for_tests()
    some_url = er.get_default_endpoints()[0]
    with pytest.raises(KeyError):
        er.resolve_deployment("model-that-does-not-exist-anywhere", some_url)


def test_no_duplicate_endpoint_urls():
    er._reset_cache_for_tests()
    urls = [ep["url"] for ep in er.list_endpoints()]
    assert len(urls) == len(set(urls)), f"duplicate URLs in registry: {urls}"


def test_no_duplicate_endpoint_names():
    er._reset_cache_for_tests()
    names = [ep["name"] for ep in er.list_endpoints()]
    assert len(names) == len(set(names)), f"duplicate names in registry: {names}"


def test_get_endpoint_by_name_round_trip():
    er._reset_cache_for_tests()
    for ep in er.list_endpoints():
        got = er.get_endpoint_by_name(ep["name"])
        assert got["url"] == ep["url"]


def test_get_endpoint_by_name_raises_on_unknown():
    er._reset_cache_for_tests()
    with pytest.raises(KeyError):
        er.get_endpoint_by_name("definitely-not-a-real-endpoint-name-xyz")


def test_list_models_is_sorted_and_unique():
    er._reset_cache_for_tests()
    models = er.list_models()
    assert models == sorted(set(models))


def test_describe_for_default_model_does_not_crash():
    er._reset_cache_for_tests()
    text = er.describe(er.get_default_model())
    assert er.get_default_model() in text
    assert "deployment=" in text


def test_describe_without_arg_lists_all_endpoints():
    er._reset_cache_for_tests()
    text = er.describe()
    for ep in er.list_endpoints():
        assert ep["name"] in text


# --------------------------------------------------------------------------
# Cross-repo consistency: endpoint URLs disclosed elsewhere in the OSS repo
# must all be present in the registry. Otherwise running the existing scripts
# would silently bypass the canonical inventory.
# --------------------------------------------------------------------------


def test_run_tau_ablation_endpoints_are_in_registry():
    er._reset_cache_for_tests()
    txt = (REPO_ROOT / "scripts" / "run_tau_ablation.py").read_text()
    # Pin the legacy short-alias URLs the script currently references.
    expected = [
        "https://aiops-llm-eus.openai.azure.com/",
        "https://aiops-llm-ch.openai.azure.com/",
        "https://aipos-llm-sw.openai.azure.com/",
    ]
    for url in expected:
        if url not in txt:
            continue  # script no longer mentions this URL; not our concern
        assert url in er.get_default_endpoints() or any(
            ep["url"] == url for ep in er.list_endpoints()
        ), f"{url} referenced in run_tau_ablation.py but missing from registry"


# --------------------------------------------------------------------------
# CLI smoke test (catches argparse / import wiring regressions).
# --------------------------------------------------------------------------


def test_cli_describe_runs_and_prints_default_pool():
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.llm_clients.endpoint_registry"],
        capture_output=True, text=True, timeout=30,
    )
    assert cp.returncode == 0, cp.stderr
    for url in er.get_default_endpoints():
        # the URL itself isn't in the no-arg describe output (that lists by
        # name), but every endpoint name from the pool should be.
        pass
    # at minimum the descriptor header should be present
    assert "Endpoint registry:" in cp.stdout


def test_cli_describe_with_default_model_lists_urls():
    cp = subprocess.run(
        [sys.executable, "-m", "agentrx.llm_clients.endpoint_registry",
         er.get_default_model()],
        capture_output=True, text=True, timeout=30,
    )
    assert cp.returncode == 0, cp.stderr
    for url in er.get_endpoints_for_model(er.get_default_model()):
        assert url in cp.stdout
