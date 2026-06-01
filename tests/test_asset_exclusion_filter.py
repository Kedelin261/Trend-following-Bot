"""Tests for AssetExclusionFilter — Phase 5.4."""

import pytest
from src.amplification_validation.asset_exclusion_filter import AssetExclusionFilter


class TestAssetExclusionFilter:

    def test_allows_symbol_not_in_set(self):
        f = AssetExclusionFilter(frozenset({"SCHD"}))
        assert f.allows("SPY") is True

    def test_blocks_excluded_symbol(self):
        f = AssetExclusionFilter(frozenset({"SCHD"}))
        assert f.allows("SCHD") is False

    def test_case_insensitive_block(self):
        f = AssetExclusionFilter(frozenset({"schd"}))
        assert f.allows("SCHD") is False

    def test_case_insensitive_input(self):
        f = AssetExclusionFilter(frozenset({"SCHD"}))
        assert f.allows("schd") is False

    def test_empty_exclusion_set_allows_all(self):
        f = AssetExclusionFilter(frozenset())
        for sym in ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]:
            assert f.allows(sym) is True

    def test_multiple_excluded_assets(self):
        f = AssetExclusionFilter(frozenset({"SCHD", "XLV"}))
        assert f.allows("SCHD") is False
        assert f.allows("XLV") is False
        assert f.allows("SPY") is True
        assert f.allows("VOO") is True

    def test_excluded_assets_property(self):
        excl = frozenset({"SCHD", "IWM"})
        f = AssetExclusionFilter(excl)
        assert f.excluded_assets == frozenset({"SCHD", "IWM"})

    def test_excludes_only_schd_scenario(self):
        """Phase 5.3 finding: only SCHD excluded, others pass."""
        f = AssetExclusionFilter(frozenset({"SCHD"}))
        pass_through = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV"]
        for sym in pass_through:
            assert f.allows(sym) is True, f"{sym} should be allowed"
        assert f.allows("SCHD") is False

    def test_repr(self):
        f = AssetExclusionFilter(frozenset({"SCHD"}))
        r = repr(f)
        assert "SCHD" in r
        assert "AssetExclusionFilter" in r

    def test_baseline_profile_allows_all(self):
        """BASELINE scenario has empty exclusion set."""
        from src.amplification_validation.filter_profiles import SCENARIO_BASELINE
        f = AssetExclusionFilter(SCENARIO_BASELINE.excluded_assets)
        for sym in ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]:
            assert f.allows(sym) is True

    def test_remove_schd_profile(self):
        """REMOVE_SCHD scenario excludes exactly SCHD."""
        from src.amplification_validation.filter_profiles import SCENARIO_REMOVE_SCHD
        f = AssetExclusionFilter(SCENARIO_REMOVE_SCHD.excluded_assets)
        assert f.allows("SCHD") is False
        assert f.allows("SPY") is True
