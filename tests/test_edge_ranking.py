"""Tests for EdgeRanking."""
import pytest
from src.edge_lab.edge_profile import EdgeProfile
from src.edge_lab.edge_ranking import EdgeRanking

def _backtest_result(pf=1.7, exp=30.0, dd=6.0, trades=55):
    from src.backtest.models import BacktestResults
    wins = int(trades * 0.65)
    return BacktestResults(
        symbol="SPY", timeframe="D1",
        starting_balance=10_000, ending_balance=10_000 + exp * trades,
        net_profit=exp * trades, total_trades=trades,
        winning_trades=wins, losing_trades=trades - wins,
        win_rate=wins / trades, profit_factor=pf, expectancy=exp,
        max_drawdown=dd, sharpe_ratio=1.3,
        average_win=80.0, average_loss=50.0,
        largest_win=200.0, largest_loss=80.0,
        equity_curve=[], trades=[],
    )


def _profile(name, trades=110, pf=1.7, exp=30.0, dd=6.0, robustness="ROBUST"):
    # Build asset_results so assets_passing works
    per_asset = trades // 2
    ar = {
        "SPY": _backtest_result(pf=pf, exp=exp, dd=dd, trades=per_asset),
        "DIA": _backtest_result(pf=pf, exp=exp, dd=dd, trades=trades - per_asset),
    }
    p = EdgeProfile(strategy_name=name, description=name,
                    total_trades=trades, winning_trades=int(trades*0.65),
                    losing_trades=int(trades*0.35), win_rate=0.65,
                    profit_factor=pf, expectancy=exp, max_drawdown=dd,
                    sharpe_ratio=1.3, net_pnl=exp*trades,
                    asset_results=ar)
    p.robustness_rating = robustness
    return p

class TestEdgeRanking:
    def test_score_in_range(self):
        r = EdgeRanking()
        p = _profile("A")
        score = r.score(p)
        assert 0.0 <= score <= 100.0
    def test_higher_pf_scores_higher(self):
        r = EdgeRanking()
        low  = _profile("low",  pf=1.3)
        high = _profile("high", pf=2.5)
        assert r.score(high) > r.score(low)
    def test_promote_passes_all_criteria(self):
        r = EdgeRanking()
        p = _profile("Good", trades=110, pf=1.7, exp=30.0, dd=6.0, robustness="ROBUST")
        cand, _ = r.evaluate_promotion(p)
        assert cand is True
    def test_promote_fails_insufficient_trades(self):
        r = EdgeRanking()
        p = _profile("Bad", trades=20, pf=1.7)
        cand, reason = r.evaluate_promotion(p)
        assert cand is False
        assert "trades" in reason
    def test_rank_sorts_descending(self):
        r = EdgeRanking()
        profiles = [_profile("A", pf=1.3), _profile("B", pf=2.0), _profile("C", pf=1.6)]
        ranked = r.rank(profiles)
        assert ranked[0].profit_factor >= ranked[1].profit_factor
