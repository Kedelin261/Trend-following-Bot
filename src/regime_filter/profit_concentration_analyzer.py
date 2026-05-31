"""Profit concentration analyzer — where profits and losses come from.

Computes per-regime profit and loss concentration before and after
applying a filter, answering:
  'What % of profits does this filter preserve?'
  'What % of losses does this filter eliminate?'

High profit-preservation + high loss-elimination = ideal filter.

No broker code.  No API calls.  Pure analytics.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime.market_regime_classifier import MarketRegime
from src.regime_filter.filter_profiles import FilterProfile
from src.regime_filter.regime_filter_engine import RegimeFilterEngine


@dataclass
class RegimeConcentration:
    """Profit/loss contribution of one regime bucket."""

    regime:               str
    trade_count:          int
    gross_pnl:            float
    gross_wins:           float
    gross_losses:         float
    profit_contribution:  float   # fraction of total gross wins
    loss_contribution:    float   # fraction of total gross losses
    net_contribution:     float   # fraction of total net PnL

    @property
    def is_net_positive(self) -> bool:
        return self.gross_pnl > 0


@dataclass
class FilterConcentrationResult:
    """Profit/loss analysis of trades before and after applying a filter."""

    filter_profile:            FilterProfile
    regime_concentrations:     Dict[str, RegimeConcentration]   # by market regime

    # Aggregate impact of applying the filter
    profits_preserved_pct:     float   # % of baseline profits retained after filter
    losses_eliminated_pct:     float   # % of baseline losses removed by filter
    net_pnl_before:            float
    net_pnl_after:             float
    net_pnl_change_pct:        float

    @property
    def is_beneficial(self) -> bool:
        """Beneficial = preserves more profit than it removes loss."""
        return self.profits_preserved_pct > self.losses_eliminated_pct


class ProfitConcentrationAnalyzer:
    """Measures where profits and losses originate per market regime."""

    def __init__(self) -> None:
        self._engine = RegimeFilterEngine()

    def analyze(
        self,
        labelled_trades: List[LabelledTrade],
        profile:         FilterProfile,
    ) -> FilterConcentrationResult:
        """Compute profit/loss concentration before and after applying *profile*."""
        # Before filter
        all_trades = [lt.trade for lt in labelled_trades]
        total_wins_before   = sum(t.pnl for t in all_trades if t.is_win)
        total_losses_before = abs(sum(t.pnl for t in all_trades if t.is_loss))
        net_before          = sum(t.pnl for t in all_trades)

        # Per-regime concentration (grouped by market regime)
        regime_buckets: Dict[str, List] = {}
        for lt in labelled_trades:
            key = lt.market_regime.value
            regime_buckets.setdefault(key, []).append(lt.trade)

        concentrations: Dict[str, RegimeConcentration] = {}
        for regime, trades in regime_buckets.items():
            g_wins   = sum(t.pnl for t in trades if t.is_win)
            g_losses = abs(sum(t.pnl for t in trades if t.is_loss))
            net_pnl  = sum(t.pnl for t in trades)
            concentrations[regime] = RegimeConcentration(
                regime               = regime,
                trade_count          = len(trades),
                gross_pnl            = net_pnl,
                gross_wins           = g_wins,
                gross_losses         = g_losses,
                profit_contribution  = g_wins   / total_wins_before   if total_wins_before   > 0 else 0.0,
                loss_contribution    = g_losses / total_losses_before if total_losses_before > 0 else 0.0,
                net_contribution     = net_pnl  / abs(net_before)     if net_before != 0     else 0.0,
            )

        # After filter
        passing    = self._engine.apply_to_trades(labelled_trades, profile)
        pass_trades = [lt.trade for lt in passing]
        wins_after   = sum(t.pnl for t in pass_trades if t.is_win)
        losses_after = abs(sum(t.pnl for t in pass_trades if t.is_loss))
        net_after    = sum(t.pnl for t in pass_trades)

        profits_pres = wins_after   / total_wins_before   if total_wins_before   > 0 else 0.0
        losses_elim  = 1.0 - (losses_after / total_losses_before) if total_losses_before > 0 else 0.0
        net_chg_pct  = ((net_after - net_before) / abs(net_before) * 100) if net_before != 0 else 0.0

        return FilterConcentrationResult(
            filter_profile         = profile,
            regime_concentrations  = concentrations,
            profits_preserved_pct  = profits_pres,
            losses_eliminated_pct  = max(0.0, losses_elim),
            net_pnl_before         = net_before,
            net_pnl_after          = net_after,
            net_pnl_change_pct     = net_chg_pct,
        )
