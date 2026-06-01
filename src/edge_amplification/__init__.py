"""Edge Amplification Research Package — Phase 5.3.

Identifies which trades, assets, regimes, and volatility environments
create or destroy the strategy's Profit Factor.

Research only. No execution. No broker code. No live trading.
No entry-logic modifications. No signal-engine changes.
"""

from src.edge_amplification.asset_contribution_analyzer import AssetContributionAnalyzer
from src.edge_amplification.regime_contribution_analyzer import RegimeContributionAnalyzer
from src.edge_amplification.volatility_contribution_analyzer import VolatilityContributionAnalyzer
from src.edge_amplification.trade_quality_analyzer import TradeQualityAnalyzer
from src.edge_amplification.holding_period_analyzer import HoldingPeriodAnalyzer
from src.edge_amplification.profit_concentration_analyzer import ProfitConcentrationAnalyzer
from src.edge_amplification.loss_concentration_analyzer import LossConcentrationAnalyzer
from src.edge_amplification.amplification_research_engine import AmplificationResearchEngine
from src.edge_amplification.amplification_report import AmplificationReport

__all__ = [
    "AssetContributionAnalyzer",
    "RegimeContributionAnalyzer",
    "VolatilityContributionAnalyzer",
    "TradeQualityAnalyzer",
    "HoldingPeriodAnalyzer",
    "ProfitConcentrationAnalyzer",
    "LossConcentrationAnalyzer",
    "AmplificationResearchEngine",
    "AmplificationReport",
]
