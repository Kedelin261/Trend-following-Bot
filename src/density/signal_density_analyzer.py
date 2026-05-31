"""Signal density analyzer — quantifies trade frequency for a single asset.

Answers: 'How many opportunities does this strategy generate per year?'

No broker code. No API calls. Pure analytics.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.backtest.models import BacktestResults, BacktestTrade

TRADING_DAYS_PER_YEAR   = 252
TRADING_MONTHS_PER_YEAR = 12


@dataclass
class SignalDensityMetrics:
    """Trade frequency statistics for one asset backtest."""

    symbol:              str
    timeframe:           str
    total_trades:        int
    candle_count:        int
    trading_years:       float
    trades_per_year:     float
    trades_per_month:    float
    avg_holding_bars:    float
    win_rate:            float
    expectancy:          float
    profit_factor:       float
    meets_density_goal:  bool    # True when trades_per_year >= min_annual_trades
    density_note:        str = ""


class SignalDensityAnalyzer:
    """Computes frequency metrics from a BacktestResults object.

    Parameters
    ----------
    min_annual_trades : trades per year needed to be considered 'sufficient'
    bars_per_year     : number of candle bars per trading year (252 for D1)
    """

    def __init__(
        self,
        min_annual_trades: float = 20.0,
        bars_per_year:     int   = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.min_annual_trades = min_annual_trades
        self.bars_per_year     = bars_per_year

    def analyze(
        self,
        results:      BacktestResults,
        candle_count: int,
    ) -> SignalDensityMetrics:
        """Compute density metrics from backtest results."""
        trading_years   = max(candle_count / self.bars_per_year, 0.001)
        trades_per_year = results.total_trades / trading_years
        trades_per_month = trades_per_year / TRADING_MONTHS_PER_YEAR

        avg_holding = (
            sum(t.holding_period for t in results.trades) / len(results.trades)
            if results.trades else 0.0
        )

        meets = trades_per_year >= self.min_annual_trades
        note  = (
            f"On pace for {trades_per_year:.0f} trades/year"
            if meets else
            f"⚠ Only {trades_per_year:.0f} trades/year (need ≥ {self.min_annual_trades:.0f})"
        )

        return SignalDensityMetrics(
            symbol             = results.symbol,
            timeframe          = results.timeframe,
            total_trades       = results.total_trades,
            candle_count       = candle_count,
            trading_years      = round(trading_years, 2),
            trades_per_year    = round(trades_per_year, 1),
            trades_per_month   = round(trades_per_month, 2),
            avg_holding_bars   = round(avg_holding, 1),
            win_rate           = results.win_rate,
            expectancy         = results.expectancy,
            profit_factor      = results.profit_factor,
            meets_density_goal = meets,
            density_note       = note,
        )

    def analyze_multi(
        self,
        asset_results: dict,          # {symbol: BacktestResults}
        candle_counts: dict,          # {symbol: int}
    ) -> List[SignalDensityMetrics]:
        """Analyze density for multiple assets."""
        return [
            self.analyze(bt, candle_counts.get(sym, 252))
            for sym, bt in asset_results.items()
        ]
