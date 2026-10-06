"""Broaden native labels, add long RV, and regularize long-expiry controllers.

Preserves the first search. No future payoff/quote-path filter; additional
labels come from independent minimum-size executions of all remaining
proposals. Final portfolios still execute every original proposal through
the same risk/eligibility ledger. One guarded worker only.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_long_tenor_policy_search as base

PREVIOUS = base.OUT_BASE
OUT = Path('artifacts/contracts/long_tenor_full_universe')
REPORT = Path('reports/long_tenor_full_universe')
ORIGINAL_FEATURES = base.native_features
SLOW_RV = {}


def add_slow_features(values, date, secids, direction, expiry, rv):
    if values is None:
        return None
    values = dict(values)
    ia = values['iv_mean_native'] + direction*values['iv_gap_native']/2
    ib = values['iv_mean_native'] - direction*values['iv_gap_native']/2
    variances = [SLOW_RV.get((date, int(sid)), np.nan) for sid in secids]
    known = all(np.isfinite(variances))
    variances = [v if np.isfinite(v) else rv[date, int(sid)][1]
                 for v, sid in zip(variances, secids)]
    values['carry252'] = direction*((variances[0]-ia*ia)/(2*ia)-(variances[1]-ib*ib)/(2*ib))
    # An explicit research forecast blending recent and slow variance. The
    # slow weight falls as the remaining deterministic expiry approaches.
    dte = max(0, (expiry-date).days)
    slow_weight = dte/(dte+90)
    values['carry_forecast'] = (slow_weight*values['carry252'] +
        (1-slow_weight)*(.4*values['carry20']+.6*values['carry60']))
    values['carry252_available'] = float(known)
    return values


def slow_features(date, held, rv):
    values = ORIGINAL_FEATURES(date, held, rv)
    if values is None:
        return None
    return add_slow_features(values, date, [p['secid'] for p in held],
                             np.sign(held[0]['quantity']), max(p['expiry'] for p in held), rv)


def compact_tree(x, y, min_leaf, weights=None):
    low, high = np.quantile(y, [.01, .99])
    return DecisionTreeRegressor(max_leaf_nodes=2, min_samples_leaf=min_leaf,
        random_state=2026).fit(x, np.clip(y, low, high), sample_weight=weights)


def setup(instrument):
    global SLOW_RV
    base.INSTRUMENT = instrument
    base.OUT, base.REPORT = OUT/instrument, REPORT/instrument
    base.OUT.mkdir(parents=True, exist_ok=True)
    base.REPORT.mkdir(parents=True, exist_ok=True)
    base.native_features = slow_features
    base.tree_for = compact_tree
    base.ENTRY_FEATURES = ['carry_forecast','carry252','carry252_available','term_gap',
                           'iv_gap_native','iv_mean_native','correlation','vix']
    base.MANAGEMENT_FEATURES = ['carry_forecast','carry252','carry252_available',
                               'gap_convergence','age_fraction','delta_gap']
    base.configure()
    base.protected()
    market = base.market_for(180, 'slow_rv_inputs')
    close = (market.spots*market.adjustments).where(market.spots.gt(0) & market.adjustments.gt(0))
    r = np.log(close/close.shift(1))
    variance = 252*r.pow(2).ewm(halflife=252, min_periods=126, adjust=False).mean()
    SLOW_RV = {(date, int(sid)):float(value) for sid in variance for date, value in variance[sid].items()}


def observer_for(market, rv, records):
    def observer(date, held):
        features = slow_features(date, held, rv)
        for p in held:
            q = p['quote']
            raw = market.quote(date, p['optionid'])
            executable = raw is not None and base.valid_quote(raw) and base.valid_greeks(raw) and raw.bid > 0
            record = dict(date=date, cohort=p['cohort'], secid=p['secid'], quantity=p['quantity'],
                option_value=p['quantity']*100*(q['bid']+q['ask'])/2,
                hedge=-p['quantity']*100*q['delta'], gross_vega=abs(p['quantity'])*100*q['vega'],
                exit_executable=executable, state_known=features is not None,
                split_seen=abs(p['quantity']) != abs(p['initial_quantity']))
            record.update(features or {field:0. for field in base.MANAGEMENT_FEATURES})
            records.append(record)
        return 1.
    return observer


def augment_episode(ep, raw, row, rv):
    ep = dict(ep)
    ep['entry_features'] = row[base.ENTRY_FEATURES].to_numpy(float)
    states = np.zeros((len(ep['dates']), len(base.MANAGEMENT_FEATURES)))
    dated = raw.drop_duplicates('date').set_index('date')
    for j, date in enumerate(ep['dates']):
        if not ep['state_known'][j]:
            continue
        values = dated.loc[date].to_dict()
        values = add_slow_features(values, date, [row.secid_a,row.secid_b],
                                   row.direction, row.known_contract_expiry, rv)
        states[j] = [values[field] for field in base.MANAGEMENT_FEATURES]
    ep['states'] = states
    return ep


def prepare(tenor):
    folder = base.OUT/f'{tenor}D'
    if (folder/'prepared.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    prior = PREVIOUS/base.INSTRUMENT/f'{tenor}D'
    packed = joblib.load(prior/'episodes.joblib')
    cal = packed['calendar']
    market = base.market_for(tenor, f'{tenor}D_all_native_labels')
    rv = base.load_rv()
    rows = base.entry_frame(pd.read_parquet(prior/'signal_rows.parquet'), market, rv)
    rows.to_parquet(folder/'signal_rows.parquet', index=False)
    raw = pd.read_parquet(prior/'held_states.parquet')
    observed = {key:g for key, g in raw.groupby('cohort', sort=False)}
    row_by_cohort = {f'{r.date.date()}|{r.pair}|{r.wing}':r for _, r in rows.iterrows()}
    episodes = [augment_episode(ep, observed[ep['cohort']], row_by_cohort[ep['cohort']], rv)
                for ep in packed['episodes']]
    original_count = len(episodes)
    present = {ep['cohort'] for ep in episodes}
    labels_dir = folder/'additional_labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    for index, row in rows.iterrows():
        cohort = f'{row.date.date()}|{row.pair}|{row.wing}'
        note = labels_dir/f'{index}.json'
        episode_file = labels_dir/f'{index}.joblib'
        if cohort in present:
            checks.append(dict(cohort=cohort, status='original_executed_training_episode'))
            continue
        if note.exists():
            checks.append(json.loads(note.read_text()))
            if episode_file.exists():
                episodes.append(joblib.load(episode_file))
            continue
        if not row.native_features_available or pd.isna(row.known_contract_expiry):
            record = dict(cohort=cohort, status='missing_signal_contract_or_current_feature_inputs')
            checks.append(record)
            base.dump(note, record)
            continue
        terminal = int(cal.searchsorted(row.known_contract_expiry))
        if terminal >= len(cal):
            raise AssertionError('Original common calendar did not provide runoff')
        history = cal[cal.get_loc(row.date):terminal+1]
        one = pd.DataFrame([row]).assign(learned_level=2)
        recorded = []
        market.variant = f'{tenor}D_extra_label_{index}'
        market.reported_year = 0
        # Suppress per-episode year logging; still guarded and checkpointed.
        previous_rate = market.rate
        market.rate = lambda date:float(market.rates.loc[date])
        try:
            nav, trades, decisions, ledger = base.LONG.replay(one, market, history, tenor,
                observer_for(market, rv, recorded))
        finally:
            market.rate = previous_rate
        record = dict(cohort=cohort, status=decisions.status.iloc[0],
                      max_ledger_error=ledger['independent_ledger_error'])
        if record['status'] == 'entered':
            ep = base.shadow_episode(cohort, pd.DataFrame(recorded), trades, market, cal,
                                     one.iloc[0], base.source.configuration(tenor))
            joblib.dump(ep, episode_file)
            episodes.append(ep)
        base.dump(note, record)
        checks.append(record)
        if len(checks) % 20 == 0:
            print(f'{base.INSTRUMENT}/{tenor}D: checked {len(checks)}/529 proposals; {len(episodes)} native labels', flush=True)
            base.progress('full_universe_labels', tenor=tenor, checked=len(checks), episodes=len(episodes))
        del nav, trades, decisions, recorded
        gc.collect()
    assert len(checks) == len(rows) == 529
    assert len({ep['cohort'] for ep in episodes}) == len(episodes)
    # Additional research turnover penalty applies solely to learned elective
    # closes in training. Real execution fees/spreads remain exactly unchanged.
    for ep in episodes:
        ep['exit_cost_training'] = ep['exit_cost_training']*2
    joblib.dump(dict(episodes=episodes, calendar=cal), folder/'episodes.joblib')
    pd.DataFrame(checks).to_csv(base.REPORT/f'{tenor}D_label_coverage.csv', index=False)
    base.dump(folder/'prepared.json', dict(tenor=tenor, instrument=base.INSTRUMENT,
        proposals=529, original_training_episodes=original_count, full_universe_episodes=len(episodes),
        additional_isolated_episodes=len(episodes)-original_count, no_future_path_or_payoff_filter=True,
        long_rv_half_life=252, minimum_rv_observations=126, missing_long_rv_fallback='RV60 with availability flag',
        maximum_controller_leaves=2, training_elective_exit_cost_multiplier=2,
        original_costs_and_risk_limits_unchanged=True))
    # Reuse exact fixed controls and call baseline without numerical reruns.
    for case in ['fixed_50_hold','fixed_100_hold'] + (['transferred_hold'] if base.INSTRUMENT == 'C75' else []):
        destination = folder/'replays'/case
        destination.mkdir(parents=True, exist_ok=True)
        for path in (prior/'replays'/case).iterdir():
            target = destination/path.name
            if not target.exists():
                os.link(path, target)
    print(f'{base.INSTRUMENT}/{tenor}D: expanded {original_count} to {len(episodes)} executed native training opportunities', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['sweep','prepare','fit','replay','summarize'], default='sweep')
    p.add_argument('--instrument', choices=['P25','C75'])
    p.add_argument('--tenor', type=int, choices=base.TENORS)
    args = p.parse_args()
    for instrument in ([args.instrument] if args.instrument else ['P25','C75']):
        setup(instrument)
        for tenor in ([args.tenor] if args.tenor else base.TENORS):
            if args.phase in ['sweep','prepare']:
                prepare(tenor)
            if args.phase in ['sweep','fit']:
                base.fit(tenor, 48)
            if args.phase in ['sweep','replay']:
                base.replay(tenor, 48)
        if args.phase in ['sweep','summarize']:
            base.summarize()
            base.dump(base.REPORT/'extension_method.json', dict(
                all_original_proposals_checked=529, labels='All baseline labels plus isolated minimum-size executions of remaining proposals; original scheduled expiry purge retained',
                rv_horizon=252, controller_leaves=2, training_turnover_multiplier=2,
                final_costs_and_risk_limits='Unchanged', training_window_months=48,
                prior_first_search_preserved=True))
            audit = json.loads((base.REPORT/'audit.json').read_text())
            audit['source_sha256_extension'] = base.digest(__file__)
            audit['full_universe_label_checks'] = True
            audit['actual_policy_tree_leaves'] = 2
            base.dump(base.REPORT/'audit.json', audit)
        base.protected()
        gc.collect()


if __name__ == '__main__':
    main()
