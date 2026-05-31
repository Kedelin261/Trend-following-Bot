"""Opportunity analyzer — quantifies trade frequency per timeframe.

Answers: 'How many independent opportunities per year does each
timeframe provide, and how does the combined portfolio compare?'

No broker code. No API calls. Pure analytics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.timeframe.timeframe_profile import TimeframeProfile

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR  = 52


@dataclass
class OpportunityMetrics:
    """Trade frequency statistics for one asset + one timeframe."""

    symbol:           str
    timeframe:        str
    total_trades:     int
    candle_count:     int
    trading_years:    float
    trades_per_year:  float
    trades_per_month: float
    trades_per_week:  float
    win_rate:         float
    expectancy:       float
    profit_factor:    float
    meets_density_goal: bool = False
    density_note:       str  = ""


@dataclass
class PortfolioOpportunityMetrics:
    """Aggregate opportunity flow across all assets and timeframes."""

    timeframe:              str
    asset_metrics:          List[OpportunityMetrics]
    total_trades:           int
    aggregate_trades_per_year: float
    aggregate_monthly_opps: float
    aggregate_weekly_opps:  float
    meets_100_target:       bool
    meets_150_target:       bool
    meets_200_target:       bool


class OpportunityAnalyzer:
    """Computes trade frequency from BacktestResults + TimeframeProfile.

    Parameters
    ----------
    min_annual_trades : annual trades needed for 'sufficient' density rating
    """

    def __init__(self, min_annual_trades: float = 20.0) -> None:
        self.min_annual_trades = min_annual_trades

    def analyze(
        self,
        results:      BacktestResults,
        tf_profile:   TimeframeProfile,
        candle_count: int,
    ) -> OpportunityMetrics:
        """Compute frequency metrics for one asset / timeframe result."""
        years         = max(tf_profile.years_from_bars(candle_count), 0.001)
        per_year      = results.total_trades / years
        per_month     = per_year / MONTHS_PER_YEAR
        per_week      = per_year / WEEKS_PER_YEAR

        meets = per_year >= self.min_annual_trades
        note  = (
            f"On pace for {per_year:.0f} trades/yr"
            if meets else
            f"⚠ {per_year:.0f} trades/yr (need ≥ {self.min_annual_trades:.0f})"
        )

        return OpportunityMetrics(
            symbol             = results.symbol,
            timeframe          = tf_profile.timeframe,
            total_trades       = results.total_trades,
            candle_count       = candle_count,
            trading_years      = round(years, 2),
            trades_per_year    = round(per_year, 1),
            trades_per_month   = round(per_month, 2),
            trades_per_week    = round(per_week, 3),
            win_rate           = results.win_rate,
            expectancy         = results.expectancy,
            profit_factor      = results.profit_factor,
            meets_density_goal = meets,
            density_note       = note,
        )

    def analyze_portfolio(
        self,
        results_map:   Dict[str, BacktestResults],
        tf_profile:    TimeframeProfile,
        candle_counts: Dict[str, int],
    ) -> PortfolioOpportunityMetrics:
        """Aggregate opportunity metrics across all assets for one timeframe."""
        metrics = [
            self.analyze(bt, tf_profile, candle_counts.get(sym, tf_profile.bars_per_year))
            for sym, bt in results_map.items()
        ]

        total_trades = sum(m.total_trades for m in metrics)
        agg_per_year = sum(m.trades_per_year for m in metrics)

        return PortfolioOpportunityMetrics(
            timeframe               = tf_profile.timeframe,
            asset_metrics           = metrics,
            total_trades            = total_trades,
            aggregate_trades_per_year = round(agg_per_year, 1),
            aggregate_monthly_opps  = round(agg_per_year / MONTHS_PER_YEAR, 1),
            aggregate_weekly_opps   = round(agg_per_year / WEEKS_PER_YEAR, 2),
            meets_100_target        = total_trades >= 100,
            meets_150_target        = total_trades >= 150,
            meets_200_target        = total_trades >= 200,
        )
