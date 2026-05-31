"""Tests for scalability_analyzer."""
from src.edge_validation.scalability_analyzer import analyze_scalability
from src.edge_validation.history_expansion import HistorySliceResult
from src.edge_lab.edge_profile import EdgeProfile

def _slice(bars, trades, pf=1.6, exp=25.0, dd=5.0):
    p = EdgeProfile(strategy_name="X", description="X",
                    total_trades=trades, winning_trades=int(trades*0.6),
                    losing_trades=int(trades*0.4), win_rate=0.6,
                    profit_factor=pf, expectancy=exp, max_drawdown=dd,
                    sharpe_ratio=1.0, net_pnl=exp*trades)
    p.robustness_rating = "ROBUST"
    return HistorySliceResult(candle_count=bars, actual_bars=bars, profile=p)

def test_growing_trades_marks_scalable():
    slices = [_slice(500, 10), _slice(1000, 20), _slice(2000, 40)]
    r = analyze_scalability("X", slices)
    assert r.scales_trade_count is True

def test_stable_pf_marked():
    slices = [_slice(500, 10, pf=1.6), _slice(1000, 20, pf=1.5)]
    r = analyze_scalability("X", slices)
    assert r.pf_stable is True

def test_insufficient_slices():
    r = analyze_scalability("X", [_slice(500, 10)])
    assert r.is_scalable is False
