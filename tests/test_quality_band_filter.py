"""Tests for QualityBandFilter — Phase 5.4."""

import pytest
from src.amplification_validation.quality_band_filter import QualityBandFilter


class TestQualityBandFilter:

    def test_disabled_when_none(self):
        f = QualityBandFilter(None, None)
        assert f.is_active is False
        assert f.allows(65.0) is True
        assert f.allows(0.0) is True
        assert f.allows(100.0) is True

    def test_blocks_score_in_band(self):
        f = QualityBandFilter(60.0, 69.9)
        assert f.allows(65.0) is False
        assert f.allows(60.0) is False   # inclusive lower bound
        assert f.allows(69.9) is False   # inclusive upper bound

    def test_allows_score_below_band(self):
        f = QualityBandFilter(60.0, 69.9)
        assert f.allows(59.9) is True
        assert f.allows(40.0) is True

    def test_allows_score_above_band(self):
        f = QualityBandFilter(60.0, 69.9)
        assert f.allows(70.0) is True
        assert f.allows(90.0) is True
        assert f.allows(100.0) is True

    def test_allows_boundary_just_above(self):
        f = QualityBandFilter(60.0, 69.9)
        # 70.0 is just above 69.9 — must be allowed
        assert f.allows(70.0) is True

    def test_phase53_exact_band(self):
        """Phase 5.3 finding: band 60-69 has PF=0.91 — no edge."""
        f = QualityBandFilter(60.0, 69.9)
        # Representative scores in the bad band
        for s in [60.0, 63.5, 68.0, 69.9]:
            assert f.allows(s) is False, f"Score {s} should be blocked"
        # Representative scores outside
        for s in [40.0, 55.0, 70.0, 80.0, 90.0, 100.0]:
            assert f.allows(s) is True, f"Score {s} should be allowed"

    def test_is_active_when_set(self):
        f = QualityBandFilter(60.0, 69.9)
        assert f.is_active is True

    def test_score_lo_hi_properties(self):
        f = QualityBandFilter(60.0, 69.9)
        assert f.score_lo == 60.0
        assert f.score_hi == 69.9

    def test_raises_on_lo_greater_than_hi(self):
        with pytest.raises(ValueError, match="score_lo"):
            QualityBandFilter(70.0, 60.0)

    def test_raises_on_only_one_none(self):
        with pytest.raises(ValueError):
            QualityBandFilter(60.0, None)
        with pytest.raises(ValueError):
            QualityBandFilter(None, 69.9)

    def test_repr_disabled(self):
        f = QualityBandFilter(None, None)
        assert "disabled" in repr(f)

    def test_repr_active(self):
        f = QualityBandFilter(60.0, 69.9)
        r = repr(f)
        assert "60" in r
        assert "69" in r

    def test_baseline_profile_no_quality_filter(self):
        """BASELINE has no quality filter."""
        from src.amplification_validation.filter_profiles import SCENARIO_BASELINE
        f = QualityBandFilter(
            SCENARIO_BASELINE.excluded_score_lo,
            SCENARIO_BASELINE.excluded_score_hi,
        )
        assert f.is_active is False

    def test_remove_quality_profile(self):
        """REMOVE_QUALITY_60_69 uses [60.0, 69.9]."""
        from src.amplification_validation.filter_profiles import (
            SCENARIO_REMOVE_QUALITY_60_69,
        )
        f = QualityBandFilter(
            SCENARIO_REMOVE_QUALITY_60_69.excluded_score_lo,
            SCENARIO_REMOVE_QUALITY_60_69.excluded_score_hi,
        )
        assert f.is_active is True
        assert f.allows(65.0) is False
        assert f.allows(80.0) is True
