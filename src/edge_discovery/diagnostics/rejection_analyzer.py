"""RejectionAnalyzer — Phase 6.0A shared rejection tracking infrastructure.

Provides two core data structures used by all three family diagnostics:

RejectionCounter
    Tracks per-rule rejection counts, first/last occurrence bar indices,
    and the raw total of candidates evaluated.

RejectionSummary
    Aggregated view across multiple assets — merges per-asset counters
    into a single summary with percentages and most-common-rejection
    identification.

Design constraints
------------------
- Pure data structures; no strategy logic lives here
- All mutation via explicit methods (not direct field access)
- Thread-safe enough for sequential single-threaded use
- Zero external dependencies beyond stdlib dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-asset rejection counter
# ---------------------------------------------------------------------------

@dataclass
class RejectionCounter:
    """Tracks rejection statistics for one strategy family on one asset.

    Attributes
    ----------
    family_name     : strategy family identifier string
    asset_symbol    : e.g. "SPY"
    total_bars      : total bars evaluated after min_candles warmup
    candidates      : bars that passed all upstream warmup checks and entered
                      rule evaluation (i.e. bars where rule-1 was tested)
    signals         : bars where ALL rules passed → a LONG signal was emitted
    rejections      : total bars rejected by any rule (= candidates - signals)
    rule_counts     : rule_label → rejection count
    rule_first_bar  : rule_label → first bar index (0-based) where rejection occurred
    rule_last_bar   : rule_label → last bar index where rejection occurred
    signal_bars     : list of bar indices (0-based) where a signal fired
    first_signal_bar: first bar index with a valid signal (None if none)
    last_signal_bar : last bar index with a valid signal (None if none)
    """

    family_name:      str
    asset_symbol:     str
    total_bars:       int                    = 0
    candidates:       int                    = 0
    signals:          int                    = 0
    rejections:       int                    = 0
    rule_counts:      Dict[str, int]         = field(default_factory=dict)
    rule_first_bar:   Dict[str, int]         = field(default_factory=dict)
    rule_last_bar:    Dict[str, int]         = field(default_factory=dict)
    signal_bars:      List[int]              = field(default_factory=list)
    first_signal_bar: Optional[int]          = None
    last_signal_bar:  Optional[int]          = None

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def record_candidate(self) -> None:
        """Increment candidate count (bar passed warmup, entered rule eval)."""
        self.candidates += 1

    def record_rejection(self, rule_label: str, bar_index: int) -> None:
        """Record that rule_label rejected this bar.

        Parameters
        ----------
        rule_label : human-readable rule identifier, e.g. "R1:consecutive<20"
        bar_index  : 0-based bar index within the candle array
        """
        self.rejections += 1
        self.rule_counts[rule_label]  = self.rule_counts.get(rule_label, 0) + 1

        if rule_label not in self.rule_first_bar:
            self.rule_first_bar[rule_label] = bar_index
        self.rule_last_bar[rule_label] = bar_index

    def record_signal(self, bar_index: int) -> None:
        """Record that a LONG signal fired at bar_index."""
        self.signals += 1
        self.signal_bars.append(bar_index)
        if self.first_signal_bar is None:
            self.first_signal_bar = bar_index
        self.last_signal_bar = bar_index

    def record_bar_evaluated(self) -> None:
        """Increment total_bars counter."""
        self.total_bars += 1

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def most_common_rejection(self) -> Optional[Tuple[str, int]]:
        """Return (rule_label, count) for the most-rejected rule, or None."""
        if not self.rule_counts:
            return None
        label = max(self.rule_counts, key=lambda k: self.rule_counts[k])
        return label, self.rule_counts[label]

    @property
    def signal_rate(self) -> float:
        """Fraction of candidates that produced a signal."""
        return self.signals / self.candidates if self.candidates > 0 else 0.0

    @property
    def rejection_pct(self) -> Dict[str, float]:
        """Per-rule rejection as percentage of total candidates."""
        if self.candidates == 0:
            return {k: 0.0 for k in self.rule_counts}
        return {k: v / self.candidates * 100.0 for k, v in self.rule_counts.items()}


# ---------------------------------------------------------------------------
# Multi-asset aggregated summary
# ---------------------------------------------------------------------------

@dataclass
class RejectionSummary:
    """Aggregated rejection statistics across all assets for one family.

    Built by passing a list of per-asset RejectionCounters to the
    class-method constructor.

    Attributes
    ----------
    family_name          : strategy family identifier string
    total_bars           : sum of per-asset total_bars
    total_candidates     : sum of per-asset candidates
    total_signals        : sum of per-asset signals
    total_rejections     : sum of per-asset rejections
    rule_totals          : rule_label → total rejections across all assets
    rule_pct             : rule_label → rejection % of total candidates
    most_common_rejection: (rule_label, count) for the dominant rule
    first_signal_asset   : asset with the earliest signal (or None)
    first_signal_bar     : earliest signal bar index across all assets (or None)
    any_signals          : True if any asset produced ≥1 signal
    per_asset            : original per-asset RejectionCounter list
    """

    family_name:           str
    total_bars:            int
    total_candidates:      int
    total_signals:         int
    total_rejections:      int
    rule_totals:           Dict[str, int]
    rule_pct:              Dict[str, float]
    most_common_rejection: Optional[Tuple[str, int]]
    first_signal_asset:    Optional[str]
    first_signal_bar:      Optional[int]
    any_signals:           bool
    per_asset:             List[RejectionCounter]

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_counters(
        cls,
        family_name: str,
        counters:    List[RejectionCounter],
    ) -> "RejectionSummary":
        """Aggregate a list of per-asset RejectionCounters.

        Parameters
        ----------
        family_name : strategy family name
        counters    : one counter per asset
        """
        total_bars       = sum(c.total_bars   for c in counters)
        total_candidates = sum(c.candidates   for c in counters)
        total_signals    = sum(c.signals      for c in counters)
        total_rejections = sum(c.rejections   for c in counters)

        # Aggregate rule counts
        rule_totals: Dict[str, int] = {}
        for c in counters:
            for rule, cnt in c.rule_counts.items():
                rule_totals[rule] = rule_totals.get(rule, 0) + cnt

        # Percentages
        rule_pct: Dict[str, float] = {}
        if total_candidates > 0:
            for rule, cnt in rule_totals.items():
                rule_pct[rule] = cnt / total_candidates * 100.0
        else:
            rule_pct = {k: 0.0 for k in rule_totals}

        # Most common rejection
        most_common: Optional[Tuple[str, int]] = None
        if rule_totals:
            top_rule = max(rule_totals, key=lambda k: rule_totals[k])
            most_common = (top_rule, rule_totals[top_rule])

        # First signal across all assets
        first_signal_bar:   Optional[int] = None
        first_signal_asset: Optional[str] = None
        for c in counters:
            if c.first_signal_bar is not None:
                if first_signal_bar is None or c.first_signal_bar < first_signal_bar:
                    first_signal_bar   = c.first_signal_bar
                    first_signal_asset = c.asset_symbol

        any_signals = total_signals > 0

        return cls(
            family_name=family_name,
            total_bars=total_bars,
            total_candidates=total_candidates,
            total_signals=total_signals,
            total_rejections=total_rejections,
            rule_totals=rule_totals,
            rule_pct=rule_pct,
            most_common_rejection=most_common,
            first_signal_asset=first_signal_asset,
            first_signal_bar=first_signal_bar,
            any_signals=any_signals,
            per_asset=counters,
        )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def conclusion(self) -> str:
        """Return standardised conclusion string.

        Three possible outcomes:
        NO_OPPORTUNITIES     : genuine structural impossibility
        OVERLY_RESTRICTIVE   : conditions are tight but not impossible
        IMPLEMENTATION_DEFECT: signals exist but trades do not appear
        """
        # This is set by the family diagnostic after its analysis;
        # default to a placeholder that diagnostics will override.
        return getattr(self, "_conclusion", "PENDING_ANALYSIS")

    def set_conclusion(self, conclusion: str) -> None:
        """Set the conclusion string from outside."""
        self._conclusion = conclusion  # type: ignore[attr-defined]
