"""Tests for TimeframeRegistry — standard timeframe validation and enumeration."""

import pytest
from src.data.timeframe_registry import TimeframeRegistry, STANDARD_TIMEFRAMES


class TestStandardTimeframes:

    def test_expected_timeframes_defined(self):
        expected = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}
        assert set(STANDARD_TIMEFRAMES) == expected

    def test_eight_timeframes_total(self):
        assert len(STANDARD_TIMEFRAMES) == 8

    def test_no_duplicates(self):
        assert len(STANDARD_TIMEFRAMES) == len(set(STANDARD_TIMEFRAMES))


class TestTimeframeRegistryIsValid:

    @pytest.mark.parametrize("tf", STANDARD_TIMEFRAMES)
    def test_all_standard_timeframes_are_valid(self, tf):
        assert TimeframeRegistry.is_valid(tf) is True

    @pytest.mark.parametrize("tf", ["2H", "monthly", "1D", "m1", "h1", "", "DAILY"])
    def test_invalid_timeframes_rejected(self, tf):
        assert TimeframeRegistry.is_valid(tf) is False

    def test_case_sensitive(self):
        assert TimeframeRegistry.is_valid("h1") is False
        assert TimeframeRegistry.is_valid("H1") is True


class TestTimeframeRegistryGetAll:

    def test_get_all_returns_list(self):
        result = TimeframeRegistry.get_all()
        assert isinstance(result, list)

    def test_get_all_contains_all_standard(self):
        result = TimeframeRegistry.get_all()
        assert set(result) == set(STANDARD_TIMEFRAMES)

    def test_get_all_returns_copy(self):
        r1 = TimeframeRegistry.get_all()
        r1.append("FAKE")
        r2 = TimeframeRegistry.get_all()
        assert "FAKE" not in r2


class TestTimeframeRegistryValidate:

    @pytest.mark.parametrize("tf", STANDARD_TIMEFRAMES)
    def test_validate_returns_valid_timeframe(self, tf):
        assert TimeframeRegistry.validate(tf) == tf

    def test_validate_raises_for_invalid(self):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            TimeframeRegistry.validate("INVALID")

    def test_validate_error_message_includes_valid_list(self):
        with pytest.raises(ValueError, match="H1"):
            TimeframeRegistry.validate("BAD")
