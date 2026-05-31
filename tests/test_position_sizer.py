"""Tests for PositionSizer — position size calculation and RiskProfile integration."""

import pytest

from src.risk.models import RiskProfile
from src.risk.position_sizer import PositionSizer


@pytest.fixture
def sizer() -> PositionSizer:
    return PositionSizer()


@pytest.fixture
def standard_profile() -> RiskProfile:
    return RiskProfile(
        account_size=10_000.0,
        cash_available=10_000.0,
        risk_per_trade_percent=1.0,
    )


class TestCalculatePositionSize:

    def test_spec_example(self, sizer):
        """Spec: $100 risk / $2 per share = 50 shares."""
        assert sizer.calculate_position_size(100.0, 2.0) == 50

    def test_spec_atr_example(self, sizer):
        """Spec full example: $100 / $12.50 = 8 shares."""
        assert sizer.calculate_position_size(100.0, 12.50) == 8

    def test_zero_risk_per_share_returns_zero(self, sizer):
        assert sizer.calculate_position_size(100.0, 0.0) == 0

    def test_negative_risk_per_share_returns_zero(self, sizer):
        assert sizer.calculate_position_size(100.0, -5.0) == 0

    def test_floors_to_integer(self, sizer):
        # $100 / $3 = 33.33 → floor → 33
        assert sizer.calculate_position_size(100.0, 3.0) == 33

    def test_returns_int_type(self, sizer):
        result = sizer.calculate_position_size(100.0, 5.0)
        assert isinstance(result, int)

    def test_large_account(self, sizer):
        # $1,000 risk / $5 per share = 200 shares
        assert sizer.calculate_position_size(1_000.0, 5.0) == 200

    def test_tiny_risk_returns_zero(self, sizer):
        # $0.50 risk / $5 per share = 0.1 → floor → 0
        assert sizer.calculate_position_size(0.50, 5.0) == 0

    def test_forex_small_pip_value(self, sizer):
        # $100 / $0.0020 = 50,000 (standard lot territory)
        result = sizer.calculate_position_size(100.0, 0.002)
        assert result == 50_000


class TestRiskProfileIntegration:

    def test_dollar_risk_from_standard_profile(self, sizer, standard_profile):
        # 1% of $10,000 = $100
        dr = sizer.dollar_risk_from_profile(standard_profile)
        assert dr == pytest.approx(100.0)

    def test_size_from_profile_matches_manual(self, sizer, standard_profile):
        dr   = sizer.dollar_risk_from_profile(standard_profile)
        size = sizer.calculate_position_size(dr, 12.50)
        assert size == 8

    def test_higher_risk_percent_gives_larger_size(self, sizer):
        p1 = RiskProfile(10_000, 10_000, risk_per_trade_percent=1.0)
        p2 = RiskProfile(10_000, 10_000, risk_per_trade_percent=2.0)
        dr1 = sizer.dollar_risk_from_profile(p1)
        dr2 = sizer.dollar_risk_from_profile(p2)
        size1 = sizer.calculate_position_size(dr1, 5.0)
        size2 = sizer.calculate_position_size(dr2, 5.0)
        assert size2 > size1

    def test_smaller_cash_gives_smaller_size(self, sizer):
        p_rich = RiskProfile(50_000, 50_000, risk_per_trade_percent=1.0)
        p_poor = RiskProfile(5_000,  5_000,  risk_per_trade_percent=1.0)
        dr_rich = sizer.dollar_risk_from_profile(p_rich)
        dr_poor = sizer.dollar_risk_from_profile(p_poor)
        assert sizer.calculate_position_size(dr_rich, 5.0) > \
               sizer.calculate_position_size(dr_poor, 5.0)


class TestRiskProfile:

    def test_dollar_risk_calculation(self):
        p = RiskProfile(account_size=10_000, cash_available=10_000, risk_per_trade_percent=1.0)
        assert p.dollar_risk_per_trade == pytest.approx(100.0)

    def test_cash_available_used_not_account_size(self):
        # cash_available < account_size (e.g. margin used)
        p = RiskProfile(account_size=10_000, cash_available=5_000, risk_per_trade_percent=1.0)
        assert p.dollar_risk_per_trade == pytest.approx(50.0)

    def test_from_config_defaults(self):
        p = RiskProfile.from_config({})
        assert p.account_size == 10_000.0
        assert p.risk_per_trade_percent == 1.0
        assert p.max_daily_loss_percent == 3.0
        assert p.max_weekly_loss_percent == 5.0

    def test_from_config_custom(self):
        cfg = {"risk": {"account_size": 25_000, "risk_per_trade_percent": 0.5}}
        p = RiskProfile.from_config(cfg)
        assert p.account_size == 25_000.0
        assert p.dollar_risk_per_trade == pytest.approx(125.0)

    def test_max_daily_loss_dollar(self):
        p = RiskProfile(account_size=10_000, cash_available=10_000,
                        max_daily_loss_percent=3.0)
        assert p.max_daily_loss_dollar == pytest.approx(300.0)

    def test_max_weekly_loss_dollar(self):
        p = RiskProfile(account_size=10_000, cash_available=10_000,
                        max_weekly_loss_percent=5.0)
        assert p.max_weekly_loss_dollar == pytest.approx(500.0)
