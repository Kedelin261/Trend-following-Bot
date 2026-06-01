"""Module 6 — Profit Concentration Analysis.

Answers: Where do the profits really come from?

Determines what percentage of gross profit is concentrated in:
  - Top N assets
  - Top N regimes
  - Top N volatility states
  - Top N quality score bands

Produces an 80/20 concentration report: what fraction of trades
generates 80%+ of total profits.

No execution code. No broker code. No live trading.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.backtest.models import BacktestTrade

MIN_SAMPLE = 50


@dataclass
class ConcentrationEntry:
    """A single dimension's contribution to gross profit."""
    dimension:    str    # e.g. "ASSET", "REGIME", "VOLATILITY", "QUALITY"
    label:        str    # e.g. "SPY", "EXPANSION", "NORMAL_VOL", "80-89"
    gross_profit: float
    trade_count:  int
    pct_of_total: float  # this entry's % of total gross profit


@dataclass
class ProfitConcentrationReport:
    """Top-contributor summary across all analysis dimensions."""
    total_gross_profit:    float
    total_winning_trades:  int

    top_assets:            List[ConcentrationEntry]  = field(default_factory=list)
    top_regimes:           List[ConcentrationEntry]  = field(default_factory=list)
    top_volatility:        List[ConcentrationEntry]  = field(default_factory=list)
    top_quality:           List[ConcentrationEntry]  = field(default_factory=list)

    # 80/20 summary: what % of categories generate 80% of profits
    asset_80_pct_threshold:  float = 0.0
    regime_80_pct_threshold: float = 0.0
    vol_80_pct_threshold:    float = 0.0

    # label for the single best contributor in each dimension
    best_asset:      str = ""
    best_regime:     str = ""
    best_volatility: str = ""
    best_quality:    str = ""


class ProfitConcentrationAnalyzer:
    """Identifies where gross profits are concentrated.

    Accepts pre-computed per-dimension groupings from the other analyzers.

    Parameters
    ----------
    trades           : all closed trades (for totals)
    asset_results    : output of AssetContributionAnalyzer.analyze()
    regime_results   : output of RegimeContributionAnalyzer.analyze()
    volatility_results : output of VolatilityContributionAnalyzer.analyze()
    quality_results  : output of TradeQualityAnalyzer.analyze()
    """

    def __init__(
        self,
        trades,
        asset_results,
        regime_results,
        volatility_results,
        quality_results,
    ) -> None:
        self._trades     = trades
        self._assets     = asset_results
        self._regimes    = regime_results
        self._volatility = volatility_results
        self._quality    = quality_results

    def analyze(self) -> ProfitConcentrationReport:
        total_gp = sum(t.pnl for t in self._trades if t.is_win)
        total_wt = sum(1 for t in self._trades if t.is_win)

        report = ProfitConcentrationReport(
            total_gross_profit   = total_gp,
            total_winning_trades = total_wt,
        )

        # Assets
        asset_entries = [
            ConcentrationEntry(
                dimension    = "ASSET",
                label        = a.symbol,
                gross_profit = a.gross_profit,
                trade_count  = len([t for t in self._trades
                                    if t.symbol == a.symbol and t.is_win]),
                pct_of_total = (a.gross_profit / total_gp * 100.0)
                               if total_gp > 0 else 0.0,
            )
            for a in self._assets
            if a.gross_profit > 0
        ]
        asset_entries.sort(key=lambda e: e.gross_profit, reverse=True)
        report.top_assets = asset_entries
        if asset_entries:
            report.best_asset = asset_entries[0].label
            report.asset_80_pct_threshold = self._calc_80_threshold(asset_entries, total_gp)

        # Regimes
        regime_entries = [
            ConcentrationEntry(
                dimension    = "REGIME",
                label        = r.regime,
                gross_profit = r.gross_profit,
                trade_count  = len([t for t in self._trades if t.is_win]),
                pct_of_total = (r.gross_profit / total_gp * 100.0)
                               if total_gp > 0 else 0.0,
            )
            for r in self._regimes
            if r.gross_profit > 0
        ]
        regime_entries.sort(key=lambda e: e.gross_profit, reverse=True)
        report.top_regimes = regime_entries
        if regime_entries:
            report.best_regime = regime_entries[0].label
            report.regime_80_pct_threshold = self._calc_80_threshold(regime_entries, total_gp)

        # Volatility
        vol_entries = [
            ConcentrationEntry(
                dimension    = "VOLATILITY",
                label        = v.regime,
                gross_profit = v.gross_profit,
                trade_count  = len([t for t in self._trades if t.is_win]),
                pct_of_total = (v.gross_profit / total_gp * 100.0)
                               if total_gp > 0 else 0.0,
            )
            for v in self._volatility
            if v.gross_profit > 0
        ]
        vol_entries.sort(key=lambda e: e.gross_profit, reverse=True)
        report.top_volatility = vol_entries
        if vol_entries:
            report.best_volatility = vol_entries[0].label
            report.vol_80_pct_threshold = self._calc_80_threshold(vol_entries, total_gp)

        # Quality
        quality_entries = [
            ConcentrationEntry(
                dimension    = "QUALITY",
                label        = q.band,
                gross_profit = q.gross_profit,
                trade_count  = q.trades,
                pct_of_total = (q.gross_profit / total_gp * 100.0)
                               if total_gp > 0 else 0.0,
            )
            for q in self._quality
            if q.gross_profit > 0
        ]
        quality_entries.sort(key=lambda e: e.gross_profit, reverse=True)
        report.top_quality = quality_entries
        if quality_entries:
            report.best_quality = quality_entries[0].label

        return report

    @staticmethod
    def _calc_80_threshold(
        entries: List[ConcentrationEntry],
        total_gp: float,
    ) -> float:
        """Return cumulative % of categories needed to reach 80% of total profit."""
        if total_gp <= 0 or not entries:
            return 0.0
        cumulative = 0.0
        for i, e in enumerate(entries, 1):
            cumulative += e.gross_profit
            if cumulative / total_gp >= 0.80:
                return i / len(entries) * 100.0
        return 100.0
