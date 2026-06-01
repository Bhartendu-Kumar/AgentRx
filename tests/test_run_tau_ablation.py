"""Static contract tests for scripts/run_tau_ablation.py.

This script orchestrates the 4-cell tau-29 judge ablation matrix. Pre-W3c
it embedded its own ``ENDPOINTS`` dict mapping short keys (``"eus"``,
``"ch"``, ``"sw"``) to Azure URLs. Post-W3c the script holds no endpoint
URLs of its own: cells reference endpoints by registry ``name``, and the
URL is resolved at script start via
``agentrx.llm_clients.endpoint_registry``.

These tests pin that contract so a future re-introduction of inlined URLs
will trip CI.
"""
from __future__ import annotations

import importlib

import pytest

from agentrx.llm_clients import endpoint_registry as er


@pytest.fixture(scope="module")
def ablation_module():
    """Import the script as a module (it sits in ``scripts/``)."""
    er._reset_cache_for_tests()
    mod = importlib.import_module("scripts.run_tau_ablation")
    return mod


def test_script_has_no_inlined_endpoints_dict(ablation_module):
    """The pre-W3c ``ENDPOINTS`` dict must not be reintroduced."""
    assert not hasattr(ablation_module, "ENDPOINTS"), (
        "scripts/run_tau_ablation.py must source endpoint URLs from "
        "agentrx.llm_clients.endpoint_registry, not from an inlined dict"
    )


def test_cells_reference_registry_endpoint_names(ablation_module):
    """Every cell's endpoint key must resolve in the registry."""
    cells = ablation_module.CELLS
    assert isinstance(cells, list) and len(cells) > 0
    known = {e["name"] for e in er.list_endpoints()}
    for name, flags, ep_name in cells:
        assert isinstance(name, str)
        assert isinstance(flags, list)
        assert ep_name in known, (
            f"cell {name!r} references endpoint {ep_name!r} not in registry; "
            f"known: {sorted(known)}"
        )


def test_cells_do_not_pin_contended_endpoint(ablation_module):
    """``aiops-llm-eus2`` is reserved for the live flash sweep."""
    bad = [c for c in ablation_module.CELLS if c[2] == "aiops-llm-eus2"]
    assert not bad, (
        f"cells {[c[0] for c in bad]} pin aiops-llm-eus2 which the live "
        "flash sweep holds; pick another endpoint"
    )


def test_resolve_cell_endpoints_returns_urls(ablation_module):
    resolved = ablation_module._resolve_cell_endpoints()
    assert isinstance(resolved, dict)
    referenced = {c[2] for c in ablation_module.CELLS}
    assert set(resolved.keys()) == referenced
    for name, url in resolved.items():
        assert url.startswith("https://")
        assert url.endswith("/")
        # the URL must match what the registry says
        assert er.get_endpoint_by_name(name)["url"] == url


def test_resolve_cell_endpoints_fails_fast_on_unknown(monkeypatch, ablation_module):
    """If somebody adds a typo'd cell, the script must fail at startup."""
    fake_cells = list(ablation_module.CELLS) + [
        ("typo_cell", ["--prompt-style", "paper"], "aiops-llm-nowhere"),
    ]
    monkeypatch.setattr(ablation_module, "CELLS", fake_cells)
    with pytest.raises(KeyError):
        ablation_module._resolve_cell_endpoints()


def test_resolve_cell_endpoints_fails_on_contended_endpoint(monkeypatch, ablation_module):
    fake_cells = [
        ("bad_cell", ["--prompt-style", "paper"], "aiops-llm-eus2"),
    ]
    monkeypatch.setattr(ablation_module, "CELLS", fake_cells)
    with pytest.raises(RuntimeError, match="aiops-llm-eus2"):
        ablation_module._resolve_cell_endpoints()


def test_three_distinct_endpoints_exercised(ablation_module):
    """The matrix should exercise three different endpoints, not collapse to one."""
    distinct = {c[2] for c in ablation_module.CELLS}
    assert len(distinct) >= 3, (
        f"matrix only exercises {len(distinct)} endpoint(s): {distinct}; "
        "load-balancing across the default pool is the whole point"
    )
