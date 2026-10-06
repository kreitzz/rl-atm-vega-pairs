"""Bounded walk-forward sequential direct-policy search in the actual ledger.

An outer capital policy observes inventory from its own trajectory. Every
candidate is evaluated with the same shared cash, costs, constraints and daily
hedges as deployment. Frozen monthly signal sizing is a common input, not a
new signal search. Semiannual overlay fits keep this first test inexpensive.
"""
from dataclasses import asdict, replace
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import unittest

import numpy as np
import pandas as pd

from scripts import run_p25_action_windows as base
from scripts import p25_capital_policy as policy
from scripts.run_p25_capital_sensitivity import metrics, annual_metrics

OUT = Path('artifacts/contracts/p25_capital_policy')
REPORT = Path('reports/p25_capital_policy')
ALLOCATIONS = Path('artifacts/contracts/p25_feature_search/replays/variance_premium/scored_allocations.parquet')
REFERENCE = Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned')
CAPITAL = 10_000_000
PAIR_BUDGET = 4_000_000
BUDGET_MODE = 'fixed'


def configure_budget_mode(mode):
    global OUT, REPORT, PAIR_BUDGET, BUDGET_MODE
    if mode not in ['fixed', 'coupled']:
        raise ValueError('Unknown pair-budget convention')
    BUDGET_MODE = mode
    if mode == 'coupled':
        OUT = Path('artifacts/contracts/p25_capital_policy_matched')
        REPORT = Path('reports/p25_capital_policy_matched')
        PAIR_BUDGET = 1_000_000


class QuietMarket(base.Market):
    def rate(self, date):
        return float(self.rates.loc[date])


def replay(market, rows, cal, cfg, pair_capital, callback):
    saved = base.engine.book_risk
    try:
        base.engine.book_risk = base.book_risk
        result = base.engine.simulate(market, rows, cal, cfg, pair_capital,
            policy=lambda row, context: int(row['learned_level']),
            scale_policy=callback, corporate_events=base.UNIFORM_CORPORATE_EVENTS,
            wing_capital=pair_capital,scale_pair_budget=BUDGET_MODE == 'coupled')
    finally:
        base.engine.book_risk = saved
    base.mark_book(result[0], market.rates, cfg.capital)
    return result


def train(market, rows, calendar, cfg, generations, population):
    paths = OUT / 'models'; paths.mkdir(parents=True, exist_ok=True)
    fits = list(pd.date_range('2018-01-01', '2025-01-01', freq='6MS'))
    logs = []
    warm = policy.constant_weights(1.)
    for cutoff in fits:
        stamp = str(cutoff.date()); model_path = paths / f'{stamp}.json'
        if model_path.exists():
            model = json.loads(model_path.read_text()); warm = np.array(model['weights'])
            logs.append(model); continue
        start = cutoff - pd.DateOffset(months=18)
        selected = rows.loc[rows.date.ge(start) & rows.date.lt(cutoff) & rows.outcome_end.lt(cutoff)]
        cal = calendar[(calendar >= start) & (calendar < cutoff)]
        weights = policy.constant_weights(1.)
        observations = []
        if start < calendar[0]-pd.Timedelta(days=7) or len(selected) < 16:
            status = 'constant1_fallback_insufficient_completed_history'
        else:
            assert selected.outcome_end.max() < cutoff and cal[-1] < cutoff
            status = 'trained'
            cache = {}

            def evaluate(w):
                key = tuple(np.round(w, 8))
                if key in cache:
                    return cache[key]
                nav, trades, decisions, ledger = replay(
                    market, selected, cal, cfg, PAIR_BUDGET,
                    lambda row, context: policy.scale(w, row, context))
                value = base.sharpe(nav.excess_return.iloc[1:])
                record = dict(fit=stamp, evaluation=len(cache), score=value,
                    weights=list(map(float, w)), entries=int(decisions.status.eq('entered').sum()),
                    account_breach_days=int(nav.post_trade_risk_breach.sum()),
                    max_ledger_error=ledger['max_ledger_error'])
                observations.append(record); cache[key] = value
                del nav, trades, decisions
                return value

            # Constant scales and the previous fit are anchors, not hindsight
            # controls chosen on the future evaluation period.
            anchors = [policy.constant_weights(x) for x in policy.SCALES] + [warm]
            scores = [evaluate(w) for w in anchors]
            best = int(np.argmax(scores)); weights = anchors[best].copy()
            best_score = scores[best]
            mean = weights.copy(); sd = np.ones(len(policy.FEATURES)) * .8
            rng = np.random.default_rng(2026 + cutoff.year * 12 + cutoff.month)
            for generation in range(generations):
                candidates = np.clip(rng.normal(mean, sd, size=(population, len(mean))), -3, 3)
                candidates[0] = weights
                values = np.array([evaluate(w) for w in candidates])
                best = int(np.argmax(values))
                if values[best] > best_score + 1e-9:
                    best_score = float(values[best]); weights = candidates[best].copy()
                elite = candidates[np.argsort(values)[-max(2, population // 4):]]
                mean = .3 * mean + .7 * elite.mean(axis=0)
                sd = np.maximum(.15, .3 * sd + .7 * elite.std(axis=0))
                print(f'Capital fit {stamp}: generation {generation+1}/{generations}, '
                      f'{len(cache)} exact episodes, train Sharpe {best_score:.4f}', flush=True)
                base.write_json(OUT / 'progress.json', dict(stage='train', fit=stamp,
                                generation=generation+1, training_episodes=len(cache),
                                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            pd.DataFrame(observations).to_json(paths / f'{stamp}_episodes.json', orient='records', indent=2)
        model = dict(fit=stamp, status=status, weights=weights.tolist(),
                     features=policy.FEATURES, training_rows=len(selected),
                     training_start=str(start.date()),
                     last_training_outcome=str(selected.outcome_end.max().date()) if len(selected) else None,
                     latest_training_session=str(cal[-1].date()) if len(cal) else None,
                     evaluations=len(observations), generations=generations, population=population,
                     train_sharpe=max((r['score'] for r in observations), default=None))
        base.write_json(model_path, model)
        logs.append(model); warm = weights.copy()
        print(f'Capital model {stamp}: {status}, {len(selected)} completed proposals', flush=True)
        gc.collect()
    pd.DataFrame(logs).to_json(REPORT / 'fits.json', orient='records', indent=2)
    return {pd.Timestamp(model['fit']): model for model in logs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--pair-budget-mode',choices=['fixed','coupled'],default='fixed')
    args = parser.parse_args()
    configure_budget_mode(args.pair_budget_mode)
    if args.generations < 1 or args.population < 4:
        raise ValueError('Positive generations and population >=4 required')
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    tests = unittest.defaultTestLoader.loadTestsFromName('tests.test_p25_capital_policy')
    assert unittest.TextTestRunner().run(tests).wasSuccessful()
    sources = [Path(__file__), Path('scripts/p25_capital_policy.py'),
               Path('scripts/uniform_smile_engine.py'), Path('scripts/uniform_smile_risk.py'),
               ALLOCATIONS, REFERENCE / 'daily_nav.parquet']
    sources += sorted((base.TRANSFER / 'market').glob('*.parquet'))
    hashes = {str(p): base.digest(p) for p in sources}
    base.write_json(OUT / 'inputs.json', dict(source_hashes=hashes, capital=CAPITAL,
        pair_and_wing_reference_budget=PAIR_BUDGET,
        pair_budget_mode=BUDGET_MODE,
        allocation='Frozen monthly signal policy plus sequential capital-scale policy',
        leverage_scales=policy.SCALES.tolist(), capital_refit_months=6, training_months=18,
        training_data_start='2018-01-02', insufficient_history_fallback=1.,
        optimizer='CEM derivative-free episodic policy search in exact shared ledger',
        objective='Annualized daily cash-excess Sharpe on completed past episodes',
        generations=args.generations, population=args.population,
        no_critic_or_neural_network=True, no_fresh_holdout=True,
        broker_margin_model=False, forced_liquidation=False, market_impact_modeled=False))
    base.cached.OUT = base.TRANSFER
    market = QuietMarket()
    calendar = market.rates.index[market.rates.index >= '2018-01-01']
    rows = pd.read_parquet(ALLOCATIONS)
    assert len(rows) == 608
    cfg = replace(base.config(), option_commission=.03, capital=CAPITAL, max_gross_vega=200_000)
    # Verify that the new optional hook and prior-close state leave the old
    # cash ledger unchanged when the policy always requests 1x.
    nav, trades, decisions, ledger = replay(market, rows, calendar, cfg, 1_000_000,
                                           lambda row, context: 1.)
    pd.testing.assert_frame_equal(nav, pd.read_parquet(REFERENCE / 'daily_nav.parquet'),
                                  rtol=1e-12, atol=1e-6)
    base.verify(nav, trades, decisions, rows, cfg)
    expected_prior = nav.nav.shift(1).reindex(pd.DatetimeIndex(decisions.entry_date)).to_numpy()
    np.testing.assert_allclose(decisions.prior_equity.to_numpy(), expected_prior, atol=1e-7, rtol=0)
    base.write_json(REPORT / 'hook_audit.json', dict(original_10m_nav_reproduced=True,
                    all_608_prior_equity_states_verified=True, ledger=ledger))
    print('Capital hook: original 10m NAV reproduced, all states use prior-close equity', flush=True)
    del nav, trades, decisions
    # Expiries are contract metadata observed at the signal close. They bound
    # past completed episodes; no future outcome is supplied to the policy.
    ends = []
    for row in rows.itertuples():
        quotes = [market.quote(row.date, int(getattr(row, f'optionid_{leg}'))) for leg in ['a', 'b']]
        ends.append(max(q.expiry for q in quotes) if all(q is not None for q in quotes)
                    else row.date + pd.Timedelta(days=60))
    rows['outcome_end'] = pd.to_datetime(ends)
    models = train(market, rows, calendar, cfg, args.generations, args.population)
    fit_dates = pd.DatetimeIndex(sorted(models))

    def learned(row, context):
        fit = fit_dates[fit_dates.searchsorted(pd.Timestamp(row['date']), side='right')-1]
        return policy.scale(models[fit]['weights'], row, context)

    cases = [('capital_policy', learned)] + [(f'constant{x:g}',
             lambda row, context, multiplier=float(x): multiplier) for x in policy.SCALES]
    comparison, annual = [], []
    for case, callback in cases:
        folder = OUT / 'replays' / case; folder.mkdir(parents=True, exist_ok=True)
        nav, trades, decisions, ledger = replay(market, rows, calendar, cfg, PAIR_BUDGET, callback)
        audit = base.verify(nav, trades, decisions, rows, cfg)
        if BUDGET_MODE == 'coupled' and case in ['constant1','constant2','constant4']:
            multiplier=int(case.removeprefix('constant'))
            old=Path('artifacts/contracts/p25_trade_scale/replays')/(
                'baseline' if multiplier==1 else f'size{multiplier}_budget{multiplier}')
            pd.testing.assert_frame_equal(nav,pd.read_parquet(old/'daily_nav.parquet'),rtol=1e-12,atol=1e-6)
        record = dict(case=case, **metrics(nav, CAPITAL), entries=audit['entries'],
            active_pairs=audit['active_pairs'],
            average_reserve_fraction=float((nav.capital_reserve / nav.nav).mean()),
            independent_ledger_error=audit['independent_ledger_error'], **ledger)
        comparison.append(record); annual.extend(annual_metrics(nav, case, CAPITAL))
        nav.to_parquet(folder / 'daily_nav.parquet'); trades.to_csv(folder / 'trades.csv', index=False)
        decisions.to_csv(folder / 'decisions.csv', index=False)
        base.write_json(folder / 'status.json', dict(status='complete', **record))
        print(f'{case}: Sharpe {record["annualized_excess_sharpe"]:.6f}; '
              f'cash-relative CAGR {record["relative_to_cash_cagr_pct"]:.4f}%; '
              f'{record["entries"]} entries; reserve {record["average_reserve_fraction"]:.2%}', flush=True)
        pd.DataFrame(comparison).to_csv(REPORT / 'comparison.csv', index=False)
        pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
        del nav, trades, decisions
        gc.collect()
    for fit, model in models.items():
        if model['last_training_outcome']:
            assert pd.Timestamp(model['last_training_outcome']) < fit
    for path, expected in hashes.items():
        assert base.digest(Path(path)) == expected, path
    base.write_json(REPORT / 'audit.json', dict(complete=True, models=len(models), replays=len(cases),
        all_training_outcomes_before_fit=True, original_10m_nav_reproduced=True,
        previous_close_capital_state_verified=True, all_cash_ledgers_verified=True,
        all_proposals_retained=True, equal_constraints_for_learned_and_constant_policies=True,
        pair_and_wing_reference_budget=PAIR_BUDGET, funded_equity=CAPITAL,
        pair_budget_mode=BUDGET_MODE,
        constant_1_2_4_reproduce_original_scale_cases=BUDGET_MODE == 'coupled',
        broker_margin_model=False, forced_liquidation=False, market_impact_modeled=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    protocol = (
        'First sequential capital-policy pilot: frozen monthly signal/sizing expert, '
        'new capital overlay refitted semiannually on preceding18months of completed '
        'proposals. All training calendar sessions and outcomes precede each fit. '
        'First18months use1x because earlier portfolio episodes were not added.\n'
        'Nine-parameter policy observes prior-close equity, reserve/stress/hedge usage, '
        'name inventory, pair stress, cash-excess drawdown and cash borrowing. '
        'Expert sizing is a state, not a calibrated confidence probability. '
        'Actions0.5/1/2/4 multiply the frozen50/75/100 expert size; no new skip gate. '
        'Agent transitions depend on its own inventory and cash, including risk '
        'admission, subsequent trading capacity, hedge costs and financing. '
        'CEM is derivative-free episodic policy search, not actor-critic/NN learning.\n'
        'Same10m equity and the same pair-budget convention for all four constant '
        'controls and the learned capital overlay. Reference budgets are '
        'research risk allowances, not allocated cash. Global risk limits remain '
        '5% stress/80% reserve/2x stock hedge/200k gross vega.\n'
        'Reserve is an internal research proxy, not broker margin. Held positions '
        'can breach limits after entry; no forced liquidation, assignment, quote '
        'depth or nonlinear market impact. Previously inspected history and '
        'retrospective roster/formation conditioning; no new holdout.\n\n'
    )
    if BUDGET_MODE == 'fixed':
        protocol+='Fixed mode: all policies use a4m per-pair/wing risk reference independent of trade size.\n'
    else:
        protocol+=('Coupled mode: per-pair/wing requested risk reference is1m times the '
            'current requested scale(.5/1/2/4). Entry checks use that requested budget; '
            'held-book pair-breach monitoring uses the maximum approved scale among '
            'still-active cohorts. All1x/2x/4x constant cash books reproduce the '
            'previous matched size/budget cases exactly.\n')
    (REPORT / 'findings.txt').write_text(protocol + pd.DataFrame(comparison).to_string(index=False) + '\n')


if __name__ == '__main__':
    main()
