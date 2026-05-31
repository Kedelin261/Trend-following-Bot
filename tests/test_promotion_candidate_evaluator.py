"""Tests for promotion_candidate_evaluator."""
from src.edge_validation.promotion_candidate_evaluator import PromotionCandidateEvaluator
from src.edge_validation.strategy_survivability import SurvivabilityResult
from src.edge_lab.edge_profile import EdgeProfile
from src.backtest.models import BacktestResults

def _bt(pf=1.7, exp=30.0, dd=5.0, trades=55):
    w = int(trades * 0.65)
    return BacktestResults("SPY","D1",10_000,10_000+exp*trades,exp*trades,
                            trades,w,trades-w,w/trades,pf,exp,dd,1.3,
                            80.0,50.0,200.0,80.0,[],[])

def _profile(trades=110, pf=1.7, exp=30.0, dd=5.0, robustness="ROBUST"):
    per = trades // 2
    ar = {"SPY":_bt(pf,exp,dd,per), "DIA":_bt(pf,exp,dd,trades-per)}
    p = EdgeProfile("X","X",trades,int(trades*.65),int(trades*.35),0.65,pf,exp,dd,1.3,exp*trades,asset_results=ar)
    p.robustness_rating = robustness
    return p

def _surv(survives=True):
    return SurvivabilityResult("X",survives,True,True,True,[],["pass"])

ev = PromotionCandidateEvaluator()

def test_all_criteria_pass():
    r = ev.evaluate(_profile(), _surv(), history_bars=3000)
    assert r.is_candidate is True

def test_fails_insufficient_history():
    r = ev.evaluate(_profile(), _surv(), history_bars=1000)
    assert r.is_candidate is False

def test_fails_low_trades():
    r = ev.evaluate(_profile(trades=50), _surv(), history_bars=3000)
    assert r.is_candidate is False

def test_fails_unstable():
    r = ev.evaluate(_profile(robustness="UNSTABLE"), _surv(), history_bars=3000)
    assert r.is_candidate is False
