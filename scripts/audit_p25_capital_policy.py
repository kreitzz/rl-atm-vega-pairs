"""Independent prior-close state and action verification for capital sizing."""
import json
import argparse
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts import p25_capital_policy as policy
from scripts import run_p25_capital_policy as run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pair-budget-mode',choices=['fixed','coupled'],default='fixed')
    args=p.parse_args()
    run.configure_budget_mode(args.pair_budget_mode)
    tests = unittest.defaultTestLoader.loadTestsFromNames([
        'tests.test_p25_capital_policy', 'tests.test_uniform_smile_risk'])
    result = unittest.TextTestRunner().run(tests)
    assert result.wasSuccessful()
    rows = pd.read_parquet(run.ALLOCATIONS)
    features = rows.set_index(['date', 'pair', 'wing'])
    models = {pd.Timestamp(p.stem): json.loads(p.read_text())
              for p in (run.OUT / 'models').glob('*.json') if not p.stem.endswith('_episodes')}
    fit_dates = pd.DatetimeIndex(sorted(models))
    summaries = []; effects = []
    for folder in sorted((run.OUT / 'replays').iterdir()):
        nav = pd.read_parquet(folder / 'daily_nav.parquet')
        decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        assert len(decisions) == 608
        keys = pd.MultiIndex.from_arrays([decisions.signal_date, decisions.pair, decisions.wing])
        matched = features.reindex(keys)
        assert matched.learned_level.notna().all()
        assert np.array_equal(matched.learned_level.to_numpy(), decisions.level.to_numpy())
        prior = nav.shift(1).reindex(pd.DatetimeIndex(decisions.entry_date))
        checks = dict(prior_equity=prior.nav, prior_cash=prior.cash,
            reserve_utilization=prior.capital_reserve / (.8 * prior.nav),
            stress_utilization=prior.stress_loss / (.05 * prior.nav),
            hedge_utilization=prior.gross_hedge_notional / (2 * prior.nav))
        relative = nav.nav / nav.cash_benchmark
        checks['excess_drawdown_fraction'] = (1 - relative / relative.cummax()).clip(lower=0).shift(1).reindex(pd.DatetimeIndex(decisions.entry_date))
        max_error = 0.
        for field, expected in checks.items():
            error = float(np.max(abs(decisions[field].to_numpy() - expected.to_numpy())))
            tolerance = 1e-7 if field in ['prior_equity', 'prior_cash'] else 1e-10
            assert error <= tolerance, (folder.name, field, error)
            max_error = max(max_error, error)
        if folder.name == 'capital_policy':
            for n, decision in decisions.iterrows():
                row = matched.iloc[n]
                context = decision.to_dict()
                fit = fit_dates[fit_dates.searchsorted(decision.signal_date, side='right') - 1]
                model = models[fit]
                assert fit <= decision.signal_date
                if model['last_training_outcome'] is not None:
                    assert pd.Timestamp(model['last_training_outcome']) < fit
                weights = np.array(model['weights'])
                expected = policy.scale(weights, row, context)
                assert expected == decision.trade_scale
                # A frozen-signal-only counterfactual changes no trades here;
                # it measures whether portfolio state affects the action.
                signal_only = weights.copy(); signal_only[2:] = 0.
                alternative = policy.scale(signal_only, row, context)
                effects.append(dict(signal_date=str(decision.signal_date.date()), pair=decision.pair,
                    fit=str(fit.date()), status=decision.status, actual_scale=expected,
                    signal_only_scale=alternative, capital_state_changes_action=expected != alternative,
                    stress_usage=decision.stress_utilization,
                    reserve_usage=decision.reserve_utilization,
                    excess_drawdown_fraction=decision.excess_drawdown_fraction))
        else:
            multiplier = float(folder.name.removeprefix('constant'))
            assert decisions.trade_scale.eq(multiplier).all()
        summaries.append(dict(case=folder.name, decisions=len(decisions),
                              all_prior_states_verified=True, maximum_state_error=max_error))
    effects = pd.DataFrame(effects)
    assert len(effects) == 608 and len(summaries) == 5
    effects.to_csv(run.REPORT / 'capital_state_action_effect.csv', index=False)
    details = dict(status='complete', unit_tests=result.testsRun, cases=summaries,
        all_five_books_states_match_previous_close=True,
        all_608_learned_actions_recomputed=True,
        capital_state_changes_proposed_actions=int(effects.capital_state_changes_action.sum()),
        capital_state_changes_entered_actions=int(effects.loc[effects.status.eq('entered'), 'capital_state_changes_action'].sum()),
        latest_training_outcome_precedes_each_fit=True,
        earliest_trained_fit=str(min(fit for fit, model in models.items() if model['status'] == 'trained').date()))
    (run.REPORT / 'state_audit.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps(details, indent=2), flush=True)
    # A cheap mechanism check: keep trained weights/fits, hide account-state
    # features, and replay. This is an inference ablation, not a separately
    # trained inventory-blind control or proof of causal economic value.
    run.base.cached.OUT = run.base.TRANSFER
    market = run.QuietMarket()
    calendar = market.rates.index[market.rates.index >= '2018-01-01']
    cfg = replace(run.base.config(), option_commission=.03, capital=run.CAPITAL,
                  max_gross_vega=200_000)

    def signal_only(row, context):
        fit = fit_dates[fit_dates.searchsorted(pd.Timestamp(row['date']), side='right')-1]
        weights = np.array(models[fit]['weights']); weights[2:] = 0.
        return policy.scale(weights, row, context)

    nav, trades, decisions, ledger = run.replay(market, rows, calendar, cfg,
                                               run.PAIR_BUDGET, signal_only)
    audit = run.base.verify(nav, trades, decisions, rows, cfg)
    record = dict(case='capital_state_hidden_at_inference', **run.metrics(nav, run.CAPITAL),
                  entries=audit['entries'], active_pairs=audit['active_pairs'],
                  average_reserve_fraction=float((nav.capital_reserve/nav.nav).mean()),
                  independent_ledger_error=audit['independent_ledger_error'], **ledger)
    folder = run.OUT / 'ablations' / 'signal_only'; folder.mkdir(parents=True, exist_ok=True)
    nav.to_parquet(folder/'daily_nav.parquet'); trades.to_csv(folder/'trades.csv', index=False)
    decisions.to_csv(folder/'decisions.csv', index=False)
    (folder/'status.json').write_text(json.dumps(record, indent=2)+'\n')
    observed = json.loads((run.OUT/'replays/capital_policy/status.json').read_text())
    pd.DataFrame([observed, record]).to_csv(run.REPORT/'capital_state_ablation.csv', index=False)
    pd.DataFrame(run.annual_metrics(nav, record['case'], run.CAPITAL)).to_csv(
        run.REPORT/'capital_state_ablation_annual.csv', index=False)
    print(f'Capital state hidden at inference: Sharpe {record["annualized_excess_sharpe"]:.6f}, '
          f'cash-relative CAGR {record["relative_to_cash_cagr_pct"]:.4f}%', flush=True)


if __name__ == '__main__':
    main()
