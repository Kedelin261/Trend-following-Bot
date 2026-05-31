"""Portfolio density analyzer — measures opportunity flow across all assets.

Answers: 'Can this multi-asset portfolio generate 100+ high-quality trades?'

New promotion criteria (replaces per-asset 30-trade threshold):
  Aggregate Trades ≥ 100
  Profit Factor   ≥ 1.50 (aggregate)
  Expectancy      > $0   (aggregate)
  Max Drawdown    < 15 % (worst asset)
  At least 2 assets individually pass quality checks

No broker code. No API calls. Pure analytics.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults, StrategyHealth

logger = logging.getLogger(__name__)

PORTFOLIO_MIN_TRADES   = 100
PORTFOLIO_MIN_PF       = 1.50
PORTFOLIO_MIN_EXP      = 0.0
PORTFOLIO_MAX_DD       = 15.0
PORTFOLIO_MIN_ASSETS   = 2     # must individually pass

TRADING_DAYS_PER_YEAR  = 252


@dataclass
class PortfolioDensityReport:
    """Portfolio-level opportunity analysis."""

    asset_trade_counts:   Dict[str, int]
    total_trades:         int
    trading_years:        float
    trades_per_year:      float
    monthly_opportunities: float

    # Aggregate quality metrics (weighted by trade count)
    aggregate_pf:         float
    aggregate_expectancy: float
    worst_drawdown:       float
    assets_passing:       int    # count of assets that individually pass quality

    # Thresholds
    meets_100_trades:     bool
    meets_150_trades:     bool
    meets_200_trades:     bool
    promoted:             bool

    promotion_reason:     str
    notes:                List[str]
    warnings:             List[str]


class PortfolioDensityAnalyzer:
    """Aggregates backtest results across assets to compute portfolio-level density.

    Parameters
    ----------
    min_trades    : aggregate trades needed for promotion (default 100)
    min_pf        : minimum aggregate profit factor
    min_exp       : minimum aggregate expectancy
    max_dd        : maximum worst-asset drawdown
    min_assets    : minimum assets that individually pass quality
    bars_per_year : trading bars per calendar year (252 for D1)
    """

    def __init__(
        self,
        min_trades:    int   = PORTFOLIO_MIN_TRADES,
        min_pf:        float = PORTFOLIO_MIN_PF,
        min_exp:       float = PORTFOLIO_MIN_EXP,
        max_dd:        float = PORTFOLIO_MAX_DD,
        min_assets:    int   = PORTFOLIO_MIN_ASSETS,
        bars_per_year: int   = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.min_trades    = min_trades
        self.min_pf        = min_pf
        self.min_exp       = min_exp
        self.max_dd        = max_dd
        self.min_assets    = min_assets
        self.bars_per_year = bars_per_year

    def analyze(
        self,
        asset_results:  Dict[str, BacktestResults],
        candle_counts:  Dict[str, int],
    ) -> PortfolioDensityReport:
        """Generate the portfolio density report."""
        if not asset_results:
            return self._empty_report()

        trade_counts = {sym: bt.total_trades for sym, bt in asset_results.items()}
        total_trades = sum(trade_counts.values())

        # Use the average candle count for years estimate
        avg_candles    = sum(candle_counts.values()) / max(len(candle_counts), 1)
        trading_years  = max(avg_candles / self.bars_per_year, 0.001)
        trades_per_yr  = total_trades / trading_years
        monthly_opps   = trades_per_yr / 12

        # Aggregate quality (weighted average by trade count)
        aggregate_pf, aggregate_exp, worst_dd = self._aggregate_quality(asset_results)

        # Count assets that individually pass
        assets_passing = sum(
            1 for bt in asset_results.values()
            if bt.profit_factor >= self.min_pf
            and bt.expectancy > self.min_exp
            and bt.max_drawdown <= self.max_dd
            and bt.total_trades >= 10
        )

        # Promotion decision
        promoted, reason = self._evaluate_promotion(
            total_trades, aggregate_pf, aggregate_exp, worst_dd, assets_passing
        )

        # Notes & warnings
        notes, warnings = self._generate_notes(
            total_trades, aggregate_pf, aggregate_exp, worst_dd,
            assets_passing, trade_counts
        )

        logger.info(
            "portfolio_density: total_trades=%d pf=%.2f exp=%.2f dd=%.1f%% promoted=%s",
            total_trades, aggregate_pf, aggregate_exp, worst_dd, promoted,
        )

        return PortfolioDensityReport(
            asset_trade_counts   = trade_counts,
            total_trades         = total_trades,
            trading_years        = round(trading_years, 2),
            trades_per_year      = round(trades_per_yr, 1),
            monthly_opportunities = round(monthly_opps, 1),
            aggregate_pf         = round(aggregate_pf, 2),
            aggregate_expectancy  = round(aggregate_exp, 2),
            worst_drawdown       = round(worst_dd, 1),
            assets_passing       = assets_passing,
            meets_100_trades     = total_trades >= 100,
            meets_150_trades     = total_trades >= 150,
            meets_200_trades     = total_trades >= 200,
            promoted             = promoted,
            promotion_reason     = reason,
            notes                = notes,
            warnings             = warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_quality(
        self, asset_results: Dict[str, BacktestResults]
    ) -> tuple:
        """Return (weighted_pf, weighted_exp, worst_dd)."""
        total_trades = sum(bt.total_trades for bt in asset_results.values())
        if total_trades == 0:
            return 0.0, 0.0, 0.0

        w_pf  = sum(
            (bt.profit_factor if not math.isinf(bt.profit_factor) else 3.0)
            * bt.total_trades
            for bt in asset_results.values()
        ) / total_trades

        w_exp = sum(
            bt.expectancy * bt.total_trades
            for bt in asset_results.values()
        ) / total_trades

        worst_dd = max(bt.max_drawdown for bt in asset_results.values())

        return w_pf, w_exp, worst_dd

    def _evaluate_promotion(
        self,
        total_trades:  int,
        pf:            float,
        exp:           float,
        worst_dd:      float,
        assets_passing: int,
    ) -> tuple:
        if total_trades < self.min_trades:
            return False, (
                f"Insufficient aggregate trades: {total_trades} < {self.min_trades}. "
                "Expand asset universe or relax filter parameters."
            )
        if pf < self.min_pf:
            return False, f"Aggregate profit factor {pf:.2f} < {self.min_pf}"
        if exp <= self.min_exp:
            return False, f"Aggregate expectancy ${exp:.2f} ≤ $0"
        if worst_dd > self.max_dd:
            return False, f"Worst-asset drawdown {worst_dd:.1f}% > {self.max_dd:.1f}%"
        if assets_passing < self.min_assets:
            return False, (
                f"Only {assets_passing} asset(s) individually pass quality "
                f"(need ≥ {self.min_assets})"
            )
        return True, (
            f"Portfolio passes all criteria: {total_trades} trades, "
            f"PF={pf:.2f}, Exp=${exp:.2f}, DD={worst_dd:.1f}%"
        )

    @staticmethod
    def _generate_notes(total, pf, exp, dd, passing, counts):
        notes, warnings = [], []
        if total >= 100:
            notes.append(f"Portfolio generates {total} trades (≥ 100 threshold met)")
        if pf >= 1.5:
            notes.append(f"Aggregate profit factor {pf:.2f} meets minimum")
        if exp > 0:
            notes.append(f"Positive aggregate expectancy ${exp:.2f}/trade")
        low_assets = [sym for sym, n in counts.items() if n < 10]
        if low_assets:
            warnings.append(f"Low trade count on: {', '.join(low_assets)}")
        if dd > 10:
            warnings.append(f"Worst-asset drawdown {dd:.1f}% is elevated")
        return notes, warnings

    def _empty_report(self) -> PortfolioDensityReport:
        return PortfolioDensityReport(
            asset_trade_counts={}, total_trades=0, trading_years=0.0,
            trades_per_year=0.0, monthly_opportunities=0.0,
            aggregate_pf=0.0, aggregate_expectancy=0.0, worst_drawdown=0.0,
            assets_passing=0, meets_100_trades=False, meets_150_trades=False,
            meets_200_trades=False, promoted=False,
            promotion_reason="No asset results provided",
            notes=[], warnings=["No data for portfolio analysis"],
        )
