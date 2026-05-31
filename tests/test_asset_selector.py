"""Tests for AssetSelector — tradeable universe management."""

import pytest

from src.refinement.asset_selector import (
    AssetSelector,
    DEFAULT_EXCLUDED,
    DEFAULT_RECOMMENDED,
)


class TestAssetSelectorDefault:

    def test_default_recommended_assets(self):
        sel = AssetSelector.default()
        for symbol in ["SPY", "VOO", "DIA"]:
            assert sel.is_tradeable(symbol)

    def test_default_excluded_assets(self):
        sel = AssetSelector.default()
        for symbol in ["QQQ", "XLE", "XLF", "IWM"]:
            assert not sel.is_tradeable(symbol)

    def test_recommended_property(self):
        sel = AssetSelector.default()
        assert isinstance(sel.recommended, list)
        assert "SPY" in sel.recommended

    def test_excluded_property(self):
        sel = AssetSelector.default()
        assert isinstance(sel.excluded, list)
        assert "QQQ" in sel.excluded


class TestIsTradeble:

    def test_recommended_but_also_excluded_is_not_tradeable(self):
        sel = AssetSelector(recommended=["SPY"], excluded=["SPY"])
        assert sel.is_tradeable("SPY") is False

    def test_not_in_either_list_is_not_tradeable(self):
        sel = AssetSelector(recommended=["SPY"], excluded=[])
        assert sel.is_tradeable("AAPL") is False

    def test_in_recommended_and_not_excluded_is_tradeable(self):
        sel = AssetSelector(recommended=["SPY"], excluded=[])
        assert sel.is_tradeable("SPY") is True


class TestFilter:

    def test_filter_keeps_only_tradeable(self):
        sel = AssetSelector(recommended=["SPY", "DIA"], excluded=["QQQ"])
        result = sel.filter(["SPY", "QQQ", "DIA", "XLE"])
        assert sorted(result) == ["DIA", "SPY"]

    def test_empty_input_returns_empty(self):
        sel = AssetSelector.default()
        assert sel.filter([]) == []

    def test_all_excluded_returns_empty(self):
        sel = AssetSelector(recommended=[], excluded=["SPY"])
        assert sel.filter(["SPY"]) == []


class TestGetMetadata:

    def test_metadata_present_for_known_assets(self):
        sel = AssetSelector.default()
        meta = sel.get_metadata("SPY")
        assert meta is not None
        assert meta.symbol == "SPY"
        assert meta.recommended is True

    def test_metadata_shows_exclusion_reason(self):
        sel = AssetSelector.default()
        meta = sel.get_metadata("QQQ")
        assert meta is not None
        assert meta.recommended is False

    def test_unknown_symbol_returns_none(self):
        sel = AssetSelector.default()
        assert sel.get_metadata("UNKNOWN") is None


class TestConstants:

    def test_default_recommended_contains_spy(self):
        assert "SPY" in DEFAULT_RECOMMENDED

    def test_default_excluded_contains_qqq(self):
        assert "QQQ" in DEFAULT_EXCLUDED
