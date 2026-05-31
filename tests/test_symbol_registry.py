"""Tests for SymbolRegistry — canonical symbol management and alias resolution."""

import pytest
from src.data.symbol_registry import SymbolRegistry


@pytest.fixture
def registry() -> SymbolRegistry:
    return SymbolRegistry(
        symbols=["EURUSD", "GBPUSD", "BTCUSD"],
        aliases={
            "EURUSD": ["EURUSD", "EURUSD.r", "EURUSDm"],
            "BTCUSD": ["BTCUSD", "BTCUSD.r", "BTCUSDm", "BTCUSD.cash"],
        },
    )


class TestSymbolRegistrySymbols:

    def test_get_symbols_returns_configured_list(self, registry):
        assert registry.get_symbols() == ["EURUSD", "GBPUSD", "BTCUSD"]

    def test_get_symbols_returns_copy(self, registry):
        s1 = registry.get_symbols()
        s1.append("XAUUSD")
        assert "XAUUSD" not in registry.get_symbols()

    def test_empty_symbol_list(self):
        reg = SymbolRegistry(symbols=[])
        assert reg.get_symbols() == []


class TestSymbolRegistryCandidates:

    def test_candidates_includes_symbol_itself(self, registry):
        assert "EURUSD" in registry.get_candidates("EURUSD")

    def test_candidates_includes_aliases(self, registry):
        candidates = registry.get_candidates("EURUSD")
        assert "EURUSD.r" in candidates
        assert "EURUSDm" in candidates

    def test_candidates_no_duplicates(self, registry):
        candidates = registry.get_candidates("EURUSD")
        assert len(candidates) == len(set(candidates))

    def test_candidates_for_symbol_without_aliases(self, registry):
        candidates = registry.get_candidates("GBPUSD")
        assert candidates == ["GBPUSD"]

    def test_candidates_order_symbol_first(self, registry):
        candidates = registry.get_candidates("EURUSD")
        assert candidates[0] == "EURUSD"


class TestSymbolRegistryResolve:

    def test_resolve_exact_match(self, registry):
        assert registry.resolve("EURUSD", ["EURUSD", "GBPUSD"]) == "EURUSD"

    def test_resolve_first_alias_wins(self, registry):
        # EURUSD not available, EURUSD.r is
        result = registry.resolve("EURUSD", ["EURUSD.r", "GBPUSD"])
        assert result == "EURUSD.r"

    def test_resolve_second_alias_fallback(self, registry):
        # Neither EURUSD nor EURUSD.r; EURUSDm is
        result = registry.resolve("EURUSD", ["EURUSDm"])
        assert result == "EURUSDm"

    def test_resolve_returns_none_when_no_match(self, registry):
        result = registry.resolve("EURUSD", ["XAUUSD", "USDJPY"])
        assert result is None

    def test_resolve_empty_available_list(self, registry):
        assert registry.resolve("EURUSD", []) is None

    def test_resolve_no_aliases_exact_only(self):
        reg = SymbolRegistry(symbols=["USDJPY"])
        assert reg.resolve("USDJPY", ["USDJPY"]) == "USDJPY"
        assert reg.resolve("USDJPY", ["USDJPY.r"]) is None

    def test_resolve_btcusd_cash_variant(self, registry):
        result = registry.resolve("BTCUSD", ["BTCUSD.cash"])
        assert result == "BTCUSD.cash"


class TestSymbolRegistryFromConfig:

    def test_from_config_parses_symbols(self):
        config = {"symbols": ["EURUSD", "GBPUSD"], "symbol_aliases": {}}
        reg = SymbolRegistry.from_config(config)
        assert reg.get_symbols() == ["EURUSD", "GBPUSD"]

    def test_from_config_parses_aliases(self):
        config = {
            "symbols": ["EURUSD"],
            "symbol_aliases": {"EURUSD": ["EURUSD", "EURUSD.r"]},
        }
        reg = SymbolRegistry.from_config(config)
        assert "EURUSD.r" in reg.get_candidates("EURUSD")

    def test_from_config_empty_config(self):
        reg = SymbolRegistry.from_config({})
        assert reg.get_symbols() == []

    def test_from_config_missing_aliases_key(self):
        config = {"symbols": ["EURUSD"]}
        reg = SymbolRegistry.from_config(config)
        assert reg.get_symbols() == ["EURUSD"]
        assert reg.get_candidates("EURUSD") == ["EURUSD"]
