"""Tests for strategy_survivability."""
from src.edge_validation.strategy_survivability import evaluate_survivability, SurvivabilityResult
from src.edge_validation.history_expansion import HistorySliceResult
from src.edge_validation.asset_expansion import AssetExpansionResult, AssetContribution
from src.edge_validation.robustness_validator import RobustnessValidationResult
from src.edge_lab.edge_profile import EdgeProfile
from src.edge_lab.edge_stability import StabilityWindow

def _make_profile(pf=1.7, exp=30.0, dd=5.0, trades=50):
    p = EdgeProfile("X","X", trades, int(trades*0.6), int(trades*0.4), 0.6,
                    pf, exp, dd, 1.0, exp*trades)
    p.robustness_rating = "ROBUST"
    return p

def _make_slice(bars, pf=1.7, exp=30.0):
    return HistorySliceResult(bars, bars, _make_profile(pf=pf, exp=exp))

def _make_asset_result(pf=1.7, exp=30.0):
    p = _make_profile(pf=pf, exp=exp, trades=100)
    contribs = [AssetContribution("SPY",50,pf,exp,5.0,True),
                AssetContribution("VOO",50,pf,exp,5.0,True)]
    return AssetExpansionResult("X", contribs, p)

def _make_robustness(rating="ROBUST"):
    w = [StabilityWindow("early",100,1.6,25.0,True),
         StabilityWindow("middle",100,1.5,20.0,True),
         StabilityWindow("recent",100,1.7,30.0,True)]
    return RobustnessValidationResult("X", rating, 3 if rating=="ROBUST" else 1, w, "")

def test_survives_when_all_pass():
    slices = [_make_slice(3000)]
    r = evaluate_survivability("X", slices, _make_asset_result(), _make_robustness("ROBUST"))
    assert r.survives is True
    assert r.verdict == "SURVIVES"

def test_fails_when_unstable():
    slices = [_make_slice(3000)]
    r = evaluate_survivability("X", slices, _make_asset_result(), _make_robustness("UNSTABLE"))
    assert r.survives is False
    assert r.verdict == "FAILS"
