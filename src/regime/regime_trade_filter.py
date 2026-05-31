"""Regime trade filter — simulates hypothetical regime-based trade gates.

THIS IS RESEARCH ONLY.  No strategy parameters are changed.
The filter simulates 'what would have happened if we only took trades
in regime X' by retroactively excluding trades taken in other regimes.

Pre-configured filter simulations:
  strong_bull_only  : only STRONG_BULL trades
  bull_and_above    : STRONG_BULL + BULL
  avoid_bear        : exclude BEAR + STRONG_BEAR
  avoid_crash       : exclude CRASH drawdown environment
  avoid_extreme_vol : exclude EXTREME_VOL
  avoid_crisis      : exclude CRISIS macro regime
  avoid_contraction : exclude CONTRACTION macro regime

No broker code. No API calls. Pure post-hoc analysis.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime

logger = logging.getLogger(__name__)

QUALITY_MIN_PF  = 1.50
QUALITY_MIN_EXP = 0.0


@dataclass
class FilterConfig:
    """Defines one hypothetical regime filter for simulation."""

    name:                    str
    description:             str
    allowed_market_regimes:  Optional[Set[MarketRegime]]  = None  # None = no restriction
    blocked_market_regimes:  Optional[Set[MarketRegime]]  = None
    blocked_drawdown_envs:   Optional[Set[DrawdownEnvironment]] = None
    blocked_volatility:      Optional[Set[VolatilityRegime]]    = None
    blocked_macro_regimes:   Optional[Set[MacroRegime]]         = None


@dataclass
class FilterSimulationResult:
    """Result of applying one hypothetical filter to the trade history."""

    filter_name:            str
    filter_description:     str
    original_trades:        int
    filtered_trades:        int
    trades_removed:         int
    projected_pf:           float
    projected_expectancy:   float
    projected_net_pnl:      float
    improves_quality:       bool   # True when projected PF ≥ 1.5 AND exp > 0
    note:                   str = ""

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.projected_pf) else f"{self.projected_pf:.2f}"

    @property
    def retention_pct(self) -> float:
        return self.filtered_trades / self.original_trades if self.original_trades > 0 else 0.0


# Pre-defined research filter configurations
DEFAULT_FILTERS: List[FilterConfig] = [
    FilterConfig(
        name="strong_bull_only",
        description="Only trade in STRONG_BULL market regime",
        allowed_market_regimes={MarketRegime.STRONG_BULL},
    ),
    FilterConfig(
        name="bull_and_above",
        description="Trade in BULL and STRONG_BULL regimes only",
        allowed_market_regimes={MarketRegime.STRONG_BULL, MarketRegime.BULL},
    ),
    FilterConfig(
        name="avoid_bear",
        description="Avoid BEAR and STRONG_BEAR regimes",
        blocked_market_regimes={MarketRegime.BEAR, MarketRegime.STRONG_BEAR},
    ),
    FilterConfig(
        name="avoid_crash",
        description="Avoid CRASH drawdown environment",
        blocked_drawdown_envs={DrawdownEnvironment.CRASH},
    ),
    FilterConfig(
        name="avoid_bear_and_crash",
        description="Avoid BEAR regimes AND CRASH environment",
        blocked_market_regimes={MarketRegime.BEAR, MarketRegime.STRONG_BEAR},
        blocked_drawdown_envs={DrawdownEnvironment.CRASH},
    ),
    FilterConfig(
        name="avoid_extreme_vol",
        description="Avoid EXTREME_VOL volatility environment",
        blocked_volatility={VolatilityRegime.EXTREME_VOL},
    ),
    FilterConfig(
        name="avoid_crisis",
        description="Avoid CRISIS macro regime",
        blocked_macro_regimes={MacroRegime.CRISIS},
    ),
    FilterConfig(
        name="avoid_contraction",
        description="Avoid CONTRACTION macro regime",
        blocked_macro_regimes={MacroRegime.CONTRACTION},
    ),
    FilterConfig(
        name="expansion_only",
        description="Only trade in EXPANSION macro regime",
        allowed_market_regimes={MarketRegime.STRONG_BULL, MarketRegime.BULL},
        blocked_macro_regimes={MacroRegime.CONTRACTION, MacroRegime.CRISIS},
    ),
]


class RegimeTradeFilter:
    """Simulates regime-based trade filters on historical trade data."""

    def simulate(
        self,
        labelled_trades: List[LabelledTrade],
        config:          FilterConfig,
    ) -> FilterSimulationResult:
        """Apply *config* to the trade list and compute projected metrics."""
        passing = [lt for lt in labelled_trades if self._passes(lt, config)]
        trades  = [lt.trade for lt in passing]
        total   = len(labelled_trades)

        pf, exp = self._quality(trades)
        net_pnl = sum(t.pnl for t in trades)

        improves = (
            (pf >= QUALITY_MIN_PF or math.isinf(pf))
            and exp > QUALITY_MIN_EXP
        )
        note = (
            f"Retains {len(trades)}/{total} trades ({len(trades)/total*100:.0f}%)"
            if total > 0 else "No trades to filter"
        )

        return FilterSimulationResult(
            filter_name          = config.name,
            filter_description   = config.description,
            original_trades      = total,
            filtered_trades      = len(trades),
            trades_removed       = total - len(trades),
            projected_pf         = pf,
            projected_expectancy = exp,
            projected_net_pnl    = net_pnl,
            improves_quality     = improves,
            note                 = note,
        )

    def simulate_all(
        self,
        labelled_trades: List[LabelledTrade],
        filters:         List[FilterConfig] = None,
    ) -> List[FilterSimulationResult]:
        """Run all filters and return results sorted by projected expectancy."""
        configs  = filters or DEFAULT_FILTERS
        results  = [self.simulate(labelled_trades, c) for c in configs]
        return sorted(results, key=lambda r: r.projected_expectancy, reverse=True)

    def best_filter(
        self, results: List[FilterSimulationResult]
    ) -> Optional[FilterSimulationResult]:
        """Return the filter with highest projected expectancy that improves quality."""
        improving = [r for r in results if r.improves_quality and r.filtered_trades >= 5]
        return max(improving, key=lambda r: r.projected_expectancy) if improving else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _passes(lt: LabelledTrade, config: FilterConfig) -> bool:
        # Allowed market regimes (whitelist)
        if config.allowed_market_regimes and \
           lt.market_regime not in config.allowed_market_regimes:
            return False
        # Blocked market regimes (blacklist)
        if config.blocked_market_regimes and \
           lt.market_regime in config.blocked_market_regimes:
            return False
        # Blocked drawdown environments
        if config.blocked_drawdown_envs and \
           lt.drawdown_env in config.blocked_drawdown_envs:
            return False
        # Blocked volatility regimes
        if config.blocked_volatility and \
           lt.volatility_regime in config.blocked_volatility:
            return False
        # Blocked macro regimes
        if config.blocked_macro_regimes and \
           lt.macro_regime in config.blocked_macro_regimes:
            return False
        return True

    @staticmethod
    def _quality(trades):
        if not trades:
            return 0.0, 0.0
        wins   = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]
        gross_wins   = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        exp = sum(t.pnl for t in trades) / len(trades)
        return pf, exp
