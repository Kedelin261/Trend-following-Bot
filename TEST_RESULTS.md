# Test Results

Generated: 2026-05-31

## Summary

| Metric | Result |
|--------|--------|
| Total collected | 1,242 |
| **Passed** | **1,217** |
| Skipped | 8 |
| Deselected (integration) | 17 |
| Failed | **0** |

## Run Command

```bash
pytest tests/ -m "not integration" -q
```

## Skipped Tests

The 8 skipped tests require a live broker connection (IBKR or MT5) and are marked `@pytest.mark.integration`. They pass when the broker is available.

To run integration tests:
```bash
pytest tests/ -q   # requires live IBKR at 127.0.0.1:7497
```

## Test Coverage by Module

| Module | Test File |
|--------|-----------|
| ATR Calculator | test_atr_calculator.py |
| Signal Engine | test_signal_engine.py |
| Trend Detector | test_trend_detector.py |
| Support/Resistance | test_support_resistance.py |
| Breakout Detector | test_breakout_detector.py |
| Volume Confirmation | test_volume_confirmation.py |
| Signal Scorer | test_signal_scorer.py |
| Candle Store | test_candle_store.py |
| Symbol Registry | test_symbol_registry.py |
| Timeframe Registry | test_timeframe_registry.py |
| Market Data Contract | test_market_data_provider_contract.py |
| MT5 Provider | test_mt5_provider.py |
| IBKR Provider | test_ibkr_provider.py |
| Trade Simulator | test_trade_simulator.py |
| Portfolio | test_portfolio.py |
| Performance Metrics | test_performance_metrics.py |
| Equity Curve | test_equity_curve.py |
| Walk Forward | test_walk_forward.py |
| Backtest Engine | test_backtest_engine.py |
| Market Regime (research) | test_market_regime.py |
| ADX Filter | test_adx_filter.py |
| Trend Quality | test_trend_quality.py |
| Volatility Filter | test_volatility_filter.py |
| Breakout Quality | test_breakout_quality.py |
| Benchmark Engine | test_benchmark_engine.py |
| Parameter Sweep | test_parameter_sweep.py |
| Multi-Asset Runner | test_multi_asset_runner.py |
| Strategy Analyzer | test_strategy_analyzer.py |
| Research Engine | test_research_engine.py |
| Strategy V2 | test_strategy_v2.py |
| Market Regime Filter | test_market_regime_filter.py |
| ADX Trade Filter | test_adx_trade_filter.py |
| Volatility Trade Filter | test_volatility_trade_filter.py |
| Breakout Quality Filter | test_breakout_quality_filter.py |
| Asset Selector | test_asset_selector.py |
| Strategy Comparator | test_strategy_comparator.py |
| Refinement Engine | test_refinement_engine.py |
| ADX Density | test_adx_density_research.py |
| Volatility Density | test_volatility_density_research.py |
| Breakout Density | test_breakout_density_research.py |
| Asset Expansion (density) | test_asset_expansion.py |
| Signal Density | test_signal_density_analyzer.py |
| Portfolio Density | test_portfolio_density_analyzer.py |
| Density Engine | test_density_engine.py |
| Timeframe Profile | test_timeframe_profile.py |
| Timeframe Backtester | test_timeframe_backtester.py |
| Timeframe Comparator | test_timeframe_comparator.py |
| Multi-TF Research | test_multi_timeframe_research.py |
| Signal Overlap | test_signal_overlap_analyzer.py |
| Opportunity Analyzer | test_opportunity_analyzer.py |
| Timeframe Engine | test_timeframe_engine.py |
| History Expansion (promo) | test_history_expansion.py |
| Asset Validation | test_asset_validation.py |
| Robustness Checker | test_robustness_checker.py |
| Promotion Validator | test_promotion_validator.py |
| Final Recommendation | test_final_recommendation.py |
| Promotion Engine | test_promotion_engine.py |
| Market Regime Classifier | test_market_regime_classifier.py |
| Trend Regime Detector | test_trend_regime_detector.py |
| Volatility Regime | test_volatility_regime_detector.py |
| Drawdown Env Detector | test_drawdown_environment_detector.py |
| Macro Regime Detector | test_macro_regime_detector.py |
| Regime Performance | test_regime_performance_analyzer.py |
| Regime Trade Filter | test_regime_trade_filter.py |
| Regime Stability Engine | test_regime_stability_engine.py |
| Filter Profiles | test_filter_profiles.py |
| Regime Filter Engine | test_regime_filter_engine.py |
| Filter Backtester | test_filter_backtester.py |
| Filter Comparator | test_filter_comparator.py |
| Regime Exclusion | test_regime_exclusion_research.py |
| Profit Concentration | test_profit_concentration_analyzer.py |
| Promotion Readiness | test_promotion_readiness_analyzer.py |
| Robustness Improvement | test_robustness_improvement_analyzer.py |
| Breakout Strategy | test_breakout_strategy.py |
| Pullback Strategy | test_pullback_strategy.py |
| Donchian Strategy | test_donchian_strategy.py |
| Volatility Expansion | test_volatility_expansion_strategy.py |
| Momentum Rotation | test_momentum_rotation_strategy.py |
| Edge Engine | test_edge_engine.py |
| Edge Comparator | test_edge_comparator.py |
| Edge Ranking | test_edge_ranking.py |
| Edge Stability | test_edge_stability.py |
| History Expansion (5.1) | test_history_expansion_v51.py |
| Asset Expansion (5.1) | test_asset_expansion_v51.py |
| Scalability Analyzer | test_scalability_analyzer.py |
| Robustness Validator (5.1) | test_robustness_validator_v51.py |
| Strategy Survivability | test_strategy_survivability.py |
| Promotion Candidate | test_promotion_candidate_evaluator.py |
| Validation Engine | test_validation_engine.py |
