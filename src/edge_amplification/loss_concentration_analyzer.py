"""Module 7 — Loss Concentration Analysis.

Answers: Where do the losses really come from?

Determines what percentage of gross loss is concentrated in:
  - Worst assets
  - Worst regimes
  - Worst volatility states
  - Worst quality score bands

Symmetric to ProfitConcentrationAnalyzer but focused on loss drivers.

No execution code. No broker code. No live trading.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

from src.backtest.models import BacktestTrade


@dataclass
class LossConcentrationEntry:
    """A single dimension's contribution to gross loss."""
    dimension:   str
    label:       str
    gross_loss:  float   # positive dollar amount
    trade_count: int
    pct_of_total: float  # this entry's % of total gross loss


@dataclass
class LossConcentrationReport:
    """Worst-contributor summary across all analysis dimensions."""
    total_gross_loss:     float
    total_losing_trades:  int

    worst_assets:         List[LossConcentrationEntry] = field(default_factory=list)
    worst_regimes:        List[LossConcentrationEntry] = field(default_factory=list)
    worst_volatility:     List[LossConcentrationEntry] = field(default_factory=list)
    worst_quality:        List[LossConcentrationEntry] = field(default_factory=list)

    worst_asset:      str = ""
    worst_regime:     str = ""
    worst_volatility_label: str = ""
    worst_quality:    str = ""


class LossConcentrationAnalyzer:
    """Identifies where gross losses are concentrated.

    Parameters
    ----------
    trades             : all closed trades (for totals)
    asset_results      : from AssetContributionAnalyzer.analyze()
    regime_results     : from RegimeContributionAnalyzer.analyze()
    volatility_results : from VolatilityContributionAnalyzer.analyze()
    quality_results    : from TradeQualityAnalyzer.analyze()
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

    def analyze(self) -> LossConcentrationReport:
        total_gl = abs(sum(t.pnl for t in self._trades if t.is_loss))
        total_lt = sum(1 for t in self._trades if t.is_loss)

        report = LossConcentrationReport(
            total_gross_loss    = total_gl,
            total_losing_trades = total_lt,
        )

        # Assets
        asset_entries = [
            LossConcentrationEntry(
                dimension    = "ASSET",
                label        = a.symbol,
                gross_loss   = a.gross_loss,
                trade_count  = len([t for t in self._trades
                                    if t.symbol == a.symbol and t.is_loss]),
                pct_of_total = (a.gross_loss / total_gl * 100.0)
                               if total_gl > 0 else 0.0,
            )
            for a in self._assets
            if a.gross_loss > 0
        ]
        asset_entries.sort(key=lambda e: e.gross_loss, reverse=True)
        report.worst_assets = asset_entries
        if asset_entries:
            report.worst_asset = asset_entries[0].label

        # Regimes
        regime_entries = [
            LossConcentrationEntry(
                dimension    = "REGIME",
                label        = r.regime,
                gross_loss   = r.gross_loss,
                trade_count  = len([t for t in self._trades if t.is_loss]),
                pct_of_total = (r.gross_loss / total_gl * 100.0)
                               if total_gl > 0 else 0.0,
            )
            for r in self._regimes
            if r.gross_loss > 0
        ]
        regime_entries.sort(key=lambda e: e.gross_loss, reverse=True)
        report.worst_regimes = regime_entries
        if regime_entries:
            report.worst_regime = regime_entries[0].label

        # Volatility
        vol_entries = [
            LossConcentrationEntry(
                dimension    = "VOLATILITY",
                label        = v.regime,
                gross_loss   = v.gross_loss,
                trade_count  = len([t for t in self._trades if t.is_loss]),
                pct_of_total = (v.gross_loss / total_gl * 100.0)
                               if total_gl > 0 else 0.0,
            )
            for v in self._volatility
            if v.gross_loss > 0
        ]
        vol_entries.sort(key=lambda e: e.gross_loss, reverse=True)
        report.worst_volatility = vol_entries
        if vol_entries:
            report.worst_volatility_label = vol_entries[0].label

        # Quality
        quality_entries = [
            LossConcentrationEntry(
                dimension    = "QUALITY",
                label        = q.band,
                gross_loss   = q.gross_loss,
                trade_count  = q.trades,
                pct_of_total = (q.gross_loss / total_gl * 100.0)
                               if total_gl > 0 else 0.0,
            )
            for q in self._quality
            if q.gross_loss > 0
        ]
        quality_entries.sort(key=lambda e: e.gross_loss, reverse=True)
        report.worst_quality = quality_entries
        if quality_entries:
            report.worst_quality = quality_entries[0].label

        return report
