"""Phase 6.0A — Discovery Diagnostics package.

Instruments the three zero-trade strategy families to answer WHY
they produced zero trades during Phase 6.0 validation.

Exports
-------
RejectionCounter          shared per-rule rejection tracking
RejectionSummary          aggregated multi-asset summary
TrendPersistenceDiagnostics
BreakoutContinuationDiagnostics
VolatilityTransitionDiagnostics
DiagnosticReport          formats and prints the full validation report
"""

from src.edge_discovery.diagnostics.rejection_analyzer import (
    RejectionCounter,
    RejectionSummary,
)
from src.edge_discovery.diagnostics.trend_persistence_diagnostics import (
    TrendPersistenceDiagnostics,
)
from src.edge_discovery.diagnostics.breakout_continuation_diagnostics import (
    BreakoutContinuationDiagnostics,
)
from src.edge_discovery.diagnostics.volatility_transition_diagnostics import (
    VolatilityTransitionDiagnostics,
)
from src.edge_discovery.diagnostics.diagnostic_report import DiagnosticReport

__all__ = [
    "RejectionCounter",
    "RejectionSummary",
    "TrendPersistenceDiagnostics",
    "BreakoutContinuationDiagnostics",
    "VolatilityTransitionDiagnostics",
    "DiagnosticReport",
]
