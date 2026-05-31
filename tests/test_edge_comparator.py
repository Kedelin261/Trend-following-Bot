"""Tests for EdgeComparator."""
import pytest
from src.edge_lab.edge_comparator import EdgeComparator
from src.edge_lab.edge_profile import EdgeProfile

def _profile(name, trades=50, pf=1.5, exp=25.0, dd=7.0, score=60.0):
    from src.backtest.models import BacktestResults
    p = EdgeProfile(strategy_name=name, description=name,
                    total_trades=trades, winning_trades=int(trades*0.6),
                    losing_trades=int(trades*0.4), win_rate=0.6,
                    profit_factor=pf, expectancy=exp, max_drawdown=dd,
                    sharpe_ratio=1.2, net_pnl=exp*trades)
    p.edge_score = score
    return p

class TestEdgeComparator:
    def test_returns_comparison(self):
        comp = EdgeComparator()
        result = comp.compare(_profile("A", pf=1.8, exp=35.0, score=70),
                              _profile("B", pf=1.3, exp=15.0, score=50))
        assert result.challenger == "A"
        assert result.challenger_wins is True
    def test_pf_delta(self):
        comp = EdgeComparator()
        result = comp.compare(_profile("A", pf=1.8), _profile("B", pf=1.3))
        assert result.pf_delta == pytest.approx(0.5, abs=0.01)
    def test_exp_delta(self):
        comp = EdgeComparator()
        result = comp.compare(_profile("A", exp=30.0), _profile("B", exp=10.0))
        assert result.exp_delta == pytest.approx(20.0)
