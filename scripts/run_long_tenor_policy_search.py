"""Native long-tenor sizing and closed-loop exit policy search.

Compact episodic direct policy optimization, matching the checkpoint's
approach; not a neural Q-learning claim. Shadow search is followed by the
unchanged executable shared-cash ledger. All original artifacts are protected.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
from itertools import product
import json
from pathlib import Path
import resource
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_p25_long_expiries as source
from scripts.audit_long_expiry_parity import BoundMarket, OUT as PARITY
from scripts.option_flex.contract_engine import valid_quote, valid_greeks
from scripts.run_p25_capital_sensitivity import metrics, annual_metrics

OUT_BASE = Path('artifacts/contracts/long_tenor_policy_search')
REPORT_BASE = Path('reports/long_tenor_policy_search')
INSTRUMENT = 'P25'
OUT = OUT_BASE / INSTRUMENT
REPORT = REPORT_BASE / INSTRUMENT
TENORS = [180, 365]
WINDOWS = [24, 48]
ENTRY_FEATURES = ['carry20', 'carry60', 'iv_mean_native', 'iv_gap_native',
                  'term_gap', 'rv_iv_ratio', 'delta_gap', 'beta', 'correlation', 'vix']
MANAGEMENT_FEATURES = ['carry20', 'carry60', 'iv_mean_native', 'gap_convergence',
                       'rv_iv_ratio', 'delta_gap', 'delta_mean', 'age_fraction']
LONG = source.long


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def progress(stage, **kw):
    dump(OUT / 'progress.json', dict(stage=stage,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **kw))


def configure():
    LONG.configuration = source.configuration
    if not any(e.get('reason') == source.GE_HEALTHCARE_EVENT['reason']
               for e in LONG.base.UNIFORM_CORPORATE_EVENTS):
        LONG.base.UNIFORM_CORPORATE_EVENTS = [*LONG.base.UNIFORM_CORPORATE_EVENTS,
                                            source.GE_HEALTHCARE_EVENT]


def market_for(tenor, name):
    root = source.OUT / 'P' / f'{tenor}D' if INSTRUMENT == 'P25' else PARITY / f'{tenor}D'
    market = BoundMarket(root)
    market.variant, market.started, market.reported_year = name, time.monotonic(), None
    return market


def load_rv():
    frame = pd.read_parquet(source.PREVIOUS / 'rv_states.parquet')
    return {(pd.Timestamp(r.date), int(r.secid)): (float(r.rv20), float(r.rv60))
            for r in frame.itertuples()}


def native_features(date, held, rv):
    """Only the current completed close and entry facts; never future marks."""
    if len(held) != 2:
        return None
    a, b = held
    ia, ib = float(a['quote']['bs_iv']), float(b['quote']['bs_iv'])
    va, vb = rv.get((date, int(a['secid']))), rv.get((date, int(b['secid'])))
    if va is None or vb is None or not np.isfinite([ia, ib, *va, *vb]).all() or min(ia, ib) <= 0:
        return None
    direction = np.sign(a['quantity'])
    edge = [direction * ((va[k]-ia*ia)/(2*ia) - (vb[k]-ib*ib)/(2*ib)) for k in [0, 1]]
    initial_gap = float(a.get('entry_iv', ia)) - float(b.get('entry_iv', ib))
    entry_date = max(a.get('entry_date', date), b.get('entry_date', date))
    expiry = max(a['expiry'], b['expiry'])
    return dict(carry20=edge[0], carry60=edge[1], iv_mean_native=(ia+ib)/2,
        iv_gap_native=direction*(ia-ib), gap_convergence=direction*((ia-ib)-initial_gap),
        rv_iv_ratio=(va[0]/ia**2+vb[0]/ib**2)/2,
        delta_gap=abs(abs(a['quote']['delta'])-abs(b['quote']['delta'])),
        delta_mean=(abs(a['quote']['delta'])+abs(b['quote']['delta']))/2,
        age_fraction=(date-entry_date).days/max(1, (expiry-entry_date).days))


def entry_frame(rows, market, rv):
    output = rows.copy()
    for field in ENTRY_FEATURES:
        if field not in ['beta', 'correlation', 'vix']:
            output[field] = 0.
    output['native_features_available'] = False
    output['known_contract_expiry'] = pd.NaT
    for index, row in rows.iterrows():
        quotes = [market.quote(row.date, int(oid)) if np.isfinite(oid) else None
                  for oid in [row.optionid_a, row.optionid_b]]
        if any(q is None for q in quotes):
            continue
        held = [dict(secid=sid, quantity=row.direction*(1 if k == 0 else -1),
                     expiry=q.expiry, quote=q)
                for k, (sid, q) in enumerate(zip([row.secid_a, row.secid_b], quotes))]
        values = native_features(row.date, held, rv)
        output.loc[index, 'known_contract_expiry'] = max(q.expiry for q in quotes)
        if values is None:
            continue
        for field in ENTRY_FEATURES:
            if field in values:
                output.loc[index, field] = values[field]
        output.loc[index, 'term_gap'] = values['iv_gap_native'] - row.oriented_iv_gap
        output.loc[index, 'native_features_available'] = True
    assert np.isfinite(output[ENTRY_FEATURES]).all().all()
    return output


def shadow_episode(cohort, observations, fills, market, cal, row, cfg):
    """Marginal daily excess P&L with isolated hedge costs, not shared risk gates.

    Cash interest cancels against excess-return financing algebraically. The
    portfolio holds positive cash; account borrowing spreads are not imposed
    independently on each sleeve. Stock borrow/hedge netting is deliberately
    deferred to actual execution replays.
    """
    start, end = fills.date.min(), fills.date.max()
    dates = cal[(cal >= start) & (cal <= end)]
    n = len(dates)
    sids = [int(row.secid_a), int(row.secid_b)]
    live = {pd.Timestamp(d): g for d, g in observations.groupby('date', sort=False)}
    pos, value, hedge, spot, adjust, vegas = [np.zeros((n, 2)) for _ in range(6)]
    can_exit = np.zeros(n, bool)
    states = np.zeros((n, len(MANAGEMENT_FEATURES)))
    state_known = np.zeros(n, bool)
    split_seen = np.zeros(n, bool)
    for j, date in enumerate(dates):
        g = live.get(date)
        for k, sid in enumerate(sids):
            spot[j, k], adjust[j, k] = market.stock(date, sid)
        if g is None:
            continue
        for r in g.itertuples():
            k = sids.index(int(r.secid))
            pos[j, k], value[j, k], hedge[j, k], vegas[j, k] = r.quantity, r.option_value, r.hedge, r.gross_vega
        can_exit[j] = len(g) == 2 and bool(g.exit_executable.all())
        state_known[j] = len(g) == 2 and bool(g.state_known.all())
        split_seen[j] = bool(g.split_seen.any())
        if state_known[j]:
            states[j] = g[MANAGEMENT_FEATURES].iloc[0].to_numpy(float)
    cashflow = (-fills.quantity*100*fills.price-fills.commission-fills.execution_penalty).groupby(fills.date).sum().reindex(dates, fill_value=0).to_numpy()
    paid = fills.reason.isin(['entry', 'announced_corporate_action_exit'])
    contract_fees = fills.loc[paid].assign(count=lambda g:g.quantity.abs()).groupby('date')['count'].sum().reindex(dates, fill_value=0).to_numpy()
    cashflow_training = cashflow - contract_fees*(.65-.03)
    v = value.sum(axis=1)
    opt = cashflow + np.diff(np.r_[0., v])
    opt_training = cashflow_training + np.diff(np.r_[0., v])
    hprev = np.vstack([np.zeros(2), hedge[:-1]])
    sprev = np.vstack([spot[0], spot[:-1]])
    aprev = np.vstack([adjust[0], adjust[:-1]])
    ratio = adjust/aprev
    hprev_adjusted = hprev*ratio
    gross_hedge = (hprev_adjusted*(spot-sprev/ratio)).sum(axis=1)
    elapsed = dates.to_series().diff().dt.days.fillna(0).to_numpy()/365
    rates = market.rates.reindex(dates).shift(1).fillna(0).to_numpy()
    financing = rates*elapsed*(np.r_[0., v[:-1]]+(hprev*sprev).sum(axis=1))
    borrow = np.maximum(-hprev, 0)*sprev*cfg.stock_borrow_rate*elapsed[:, None]
    dividends = np.zeros(n)
    for event in market.divs.loc[market.divs.secid.isin(sids)].itertuples():
        kdate = cal.searchsorted(event.ex_date)
        if kdate >= len(cal):
            continue
        date = cal[kdate]
        if date in dates:
            j = dates.get_loc(date)
            dividends[j] += hprev_adjusted[j, sids.index(int(event.secid))]*event.amount
    gross = opt + gross_hedge + dividends - borrow.sum(axis=1) - financing
    gross_training = opt_training + gross_hedge + dividends - borrow.sum(axis=1) - financing
    hedge_cost = (np.abs(hedge-hprev_adjusted)*spot).sum(axis=1)*cfg.stock_cost_bps/10000
    entry_vega = float(vegas[0].mean())
    assert entry_vega > 0 and np.isfinite(gross_training).all()
    # Each state instruction is filled at the next close, subject to quotes.
    return dict(cohort=cohort, date=row.date, expiry=row.known_contract_expiry,
        pair=row.pair, dates=dates, positions=cal.get_indexer(dates),
        entry_features=row[ENTRY_FEATURES].to_numpy(float),
        original_level=int(row.learned_level), entry_vega=entry_vega,
        state_known=state_known, states=states, split_seen=split_seen,
        can_exit=can_exit, gross_training=gross_training,
        reward_training=gross_training-hedge_cost,
        reward_actual=gross-hedge_cost, hprev=hprev_adjusted, hedge=hedge, spot=spot,
        exit_cost_training=vegas.sum(axis=1)*cfg.option_penalty_vol_points + np.abs(pos).sum(axis=1)*.65,
        option_pnl=opt, hedge_gross_pnl=gross_hedge, dividends=dividends,
        isolated_hedge_cost=hedge_cost, isolated_borrow=borrow.sum(axis=1))


def prepare(tenor):
    folder = OUT / f'{tenor}D'
    if (folder / 'prepared.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    market = market_for(tenor, f'{tenor}D_training_observer')
    rv = load_rv()
    rows = pd.read_parquet(source.OUT / 'P' / f'{tenor}D/scored_allocations.parquet')
    if INSTRUMENT == 'C75':
        counterpart = pd.read_parquet(PARITY / 'same_strike_quotes.parquet')
        counterpart = counterpart.loc[counterpart.tenor.eq(tenor)]
        for side in ['a', 'b']:
            keys = rows[['date', f'secid_{side}', f'optionid_{side}']].rename(
                columns={f'secid_{side}':'secid', f'optionid_{side}':'put_optionid'})
            matched = keys.merge(counterpart[['date','secid','put_optionid','optionid']],
                on=['date','secid','put_optionid'], how='left', validate='many_to_one')
            assert len(matched) == len(rows)
            rows[f'optionid_{side}'] = matched.optionid.to_numpy()
        rows['wing'] = 'C75_same_strike'
    rows = entry_frame(rows, market, rv)
    rows.to_parquet(folder / 'signal_rows.parquet', index=False)
    cal = market.rates.index[market.rates.index >= '2018-01-01']
    observed = []
    def observer(date, held):
        features = native_features(date, held, rv)
        for p in held:
            q = p['quote']
            raw = market.quote(date, p['optionid'])
            executable = raw is not None and valid_quote(raw) and valid_greeks(raw) and raw.bid > 0
            record = dict(date=date, cohort=p['cohort'], secid=p['secid'], quantity=p['quantity'],
                option_value=p['quantity']*100*(q['bid']+q['ask'])/2,
                hedge=-p['quantity']*100*q['delta'], gross_vega=abs(p['quantity'])*100*q['vega'],
                exit_executable=executable, state_known=features is not None,
                split_seen=abs(p['quantity']) != abs(p['initial_quantity']))
            record.update(features or {field:0. for field in MANAGEMENT_FEATURES})
            observed.append(record)
        return 1.
    nav, fills, decisions, ledger = LONG.replay(rows, market, cal, tenor, observer)
    if INSTRUMENT == 'P25':
        original = pd.read_parquet(source.OUT / 'P/replays' / f'{tenor}D_hold/daily_nav.parquet')
        pd.testing.assert_frame_equal(nav, original, rtol=1e-12, atol=1e-6)
    else:
        original = nav
        destination = folder / 'replays/transferred_hold'
        destination.mkdir(parents=True, exist_ok=True)
        nav.to_parquet(destination / 'daily_nav.parquet')
        fills.to_csv(destination / 'trades.csv', index=False)
        decisions.to_csv(destination / 'decisions.csv', index=False)
        rows.to_parquet(destination / 'scored_allocations.parquet', index=False)
        ledger.pop('management_log', None)
        dump(destination / 'status.json', dict(case='transferred_hold', tenor=tenor,
            **metrics(nav, 10_000_000), entries=int(decisions.status.eq('entered').sum()),
            proposals=len(rows), trims=0, exits=0, **ledger))
        pd.DataFrame(annual_metrics(nav, f'{tenor}D_transferred_hold', 10_000_000)).to_csv(
            destination / 'annual.csv', index=False)
    raw = pd.DataFrame(observed)
    raw.to_parquet(folder / 'held_states.parquet', index=False)
    entries = fills.loc[fills.reason.eq('entry')]
    groups = {key:g for key, g in raw.groupby('cohort', sort=False)}
    row_lookup = {f'{r.date.date()}|{r.pair}|{r.wing}': rows.loc[i]
                  for i, r in rows.iterrows()}
    episodes = []
    for cohort, trades in fills.groupby('cohort', sort=False):
        if cohort not in set(entries.cohort):
            continue
        episodes.append(shadow_episode(cohort, groups[cohort], trades, market, cal,
                                        row_lookup[cohort], source.configuration(tenor)))
    # Independently extracted option marks, hedges and dividends must reproduce
    # all baseline gross daily ledger components; only unnetted costs differ.
    recon = np.zeros((len(cal), 3))
    for ep in episodes:
        np.add.at(recon, ep['positions'], np.stack([ep['option_pnl'], ep['hedge_gross_pnl'], ep['dividends']], axis=1))
    want = original[['option_pnl','hedge_gross_pnl','dividend_cash']].to_numpy()
    np.testing.assert_allclose(recon, want, rtol=1e-10, atol=1e-6)
    isolated_cost = sum(e['isolated_hedge_cost'].sum() for e in episodes)
    isolated_borrow = sum(e['isolated_borrow'].sum() for e in episodes)
    assert isolated_cost >= original.hedge_cost.sum()-1e-6
    assert isolated_borrow >= original.stock_borrow_cost.sum()-1e-6
    joblib.dump(dict(episodes=episodes, calendar=cal), folder / 'episodes.joblib')
    dump(folder / 'prepared.json', dict(tenor=tenor, proposals=len(rows), episodes=len(episodes),
         observer_nav_reproduced=INSTRUMENT == 'P25', gross_components_reproduced=True,
         instrument=INSTRUMENT,
         isolated_hedge_cost=isolated_cost, shared_hedge_cost=float(original.hedge_cost.sum()),
         isolated_borrow=isolated_borrow, shared_borrow=float(original.stock_borrow_cost.sum()),
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(f'{tenor}D: {len(episodes)} native episodes; baseline NAV and gross components reproduced', flush=True)
    progress('prepared', tenor=tenor, episodes=len(episodes))


def completed_episodes(episodes, cutoff, months):
    start = cutoff-pd.DateOffset(months=months)
    use = [ep for ep in episodes if ep['date'] >= start and ep['date'] < cutoff and ep['expiry'] < cutoff]
    assert all(ep['expiry'] < cutoff for ep in use)
    return use


def daily_sharpes(book):
    means, sd = book.mean(axis=0), book.std(axis=0, ddof=1)
    return np.divide(np.sqrt(252)*means, sd, out=np.full(len(means), -np.inf), where=sd > 1e-10)


def action_grid(leaves, levels):
    return np.array(list(product(levels, repeat=len(leaves))), dtype=float)


def tree_for(x, y, min_leaf, weights=None):
    low, high = np.quantile(y, [.01, .99])
    return DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=min_leaf,
         random_state=2026).fit(x, np.clip(y, low, high), sample_weight=weights)


def optimize_sizes(episodes, positions, history_len):
    x = np.stack([e['entry_features'] for e in episodes])
    y = np.array([e['reward_training'].sum()/e['entry_vega'] for e in episodes])
    tree = tree_for(x, y, max(8, len(episodes)//4))
    leaves = np.unique(tree.apply(x))
    states = np.searchsorted(leaves, tree.apply(x))
    grid = action_grid(leaves, [2, 3, 4])
    book = np.zeros((history_len, len(grid)))
    original = np.zeros(history_len)
    for ep, state in zip(episodes, states):
        idx = positions[ep['positions']]
        np.add.at(book, idx, ep['reward_training'][:, None]*grid[:, state][None, :]/ep['original_level'])
        np.add.at(original, idx, ep['reward_training'])
    scores = daily_sharpes(book)
    best = int(np.argmax(scores))
    return dict(tree=tree, leaves=leaves, choices=grid[best].astype(int)), dict(
        shadow_training_sharpe=float(scores[best]),
        shadow_original_sharpe=float(daily_sharpes(original[:, None])[0]),
        states=len(leaves), choices=grid[best].astype(int).tolist())


def simulate_shadow_management(ep, grid, state_ids, initial_scales, policy_enabled=True):
    """Vectorized irreversible weekly policy: hold, half, close, next-close fill.

    Fractional shadow quantities and unnetted hedge costs serve only training.
    Final evaluation uses integer matching and the actual shared ledger.
    """
    candidates = len(grid)
    n = len(ep['dates'])
    retained = np.ones(candidates)
    out = np.zeros((n, candidates))
    for j in range(n):
        previous = retained.copy()
        if policy_enabled and j > 0 and (j-1) >= 5 and (j-1) % 5 == 0 and ep['state_known'][j-1] and ep['can_exit'][j]:
            target = grid[:, state_ids[j-1]]
            # Existing engine cannot half-trim adjusted original counts after
            # a split. Continue/close remain available; no entry is removed.
            if ep['split_seen'][j-1]:
                target = np.where(target == .5, 1., target)
            retained = np.minimum(retained, target)
        changes = retained[None, :]*ep['hedge'][j, :, None] - previous[None, :]*ep['hprev'][j, :, None]
        hedge_cost = (np.abs(changes)*ep['spot'][j, :, None]).sum(axis=0)*.0001
        closing = (previous-retained)*ep['exit_cost_training'][j]
        out[j] = (previous*ep['gross_training'][j]-hedge_cost-closing)*initial_scales
    return out


def optimize_management(episodes, positions, history_len, sizing=None):
    arrays, targets, weights = [], [], []
    for ep in episodes:
        take = np.flatnonzero(ep['state_known'] & (np.arange(len(ep['dates'])) >= 5) & (np.arange(len(ep['dates'])) % 5 == 0))
        if len(take):
            remaining = np.cumsum(ep['reward_training'][::-1])[::-1]-ep['reward_training']
            arrays.append(ep['states'][take])
            targets.append(remaining[take]/ep['entry_vega'])
            weights.append(np.full(len(take), 1/len(take)))
    if not arrays or sum(map(len, arrays)) < 100:
        return None, dict(states=0, choices=[1.], status='insufficient_management_states')
    x, y, w = np.vstack(arrays), np.concatenate(targets), np.concatenate(weights)
    tree = tree_for(x, y, max(40, len(x)//12), w)
    leaves = np.unique(tree.apply(x))
    grid = action_grid(leaves, [0., .5, 1.])
    book = np.zeros((history_len, len(grid)))
    for ep in episodes:
        states = np.searchsorted(leaves, tree.apply(ep['states']))
        scale = 1.
        if sizing is not None:
            leaf = sizing['tree'].apply(ep['entry_features'][None, :])[0]
            scale = sizing['choices'][np.searchsorted(sizing['leaves'], leaf)]/ep['original_level']
        reward = simulate_shadow_management(ep, grid, states, scale)
        np.add.at(book, positions[ep['positions']], reward)
    scores = daily_sharpes(book)
    best = int(np.argmax(scores))
    hold = int(np.flatnonzero((grid == 1.).all(axis=1))[0])
    return dict(tree=tree, leaves=leaves, choices=grid[best]), dict(
        shadow_training_sharpe=float(scores[best]), shadow_hold_sharpe=float(scores[hold]),
        states=len(leaves), choices=grid[best].tolist(), training_states=len(x))


def fit(tenor, window):
    packed = joblib.load(OUT / f'{tenor}D/episodes.joblib')
    episodes, cal = packed['episodes'], packed['calendar']
    for cutoff in pd.date_range('2018-01-01', cal[-1], freq='QS-JAN'):
        folder = OUT / f'{tenor}D/models/{window}M'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = str(cutoff.date())
        note = folder / f'{stamp}.json'
        if note.exists():
            continue
        use = completed_episodes(episodes, cutoff, window)
        record = dict(tenor=tenor, window=window, fit=stamp, training_episodes=len(use),
                      last_training_expiry=str(max(e['expiry'] for e in use).date()) if use else None,
                      fitted=False, fallback='original_30D_sizes_and_hold')
        models = dict(sizing=None, management=None, joint_management=None)
        if len(use) >= 16:
            first = min(e['positions'][0] for e in use)
            last = int(cal.searchsorted(cutoff))
            mapping = np.arange(len(cal))-first
            models['sizing'], record['sizing'] = optimize_sizes(use, mapping, last-first)
            models['management'], record['management'] = optimize_management(use, mapping, last-first)
            models['joint_management'], record['joint_management'] = optimize_management(use, mapping, last-first, models['sizing'])
            record['fitted'] = True
        joblib.dump(models, folder / f'{stamp}.joblib')
        dump(note, record)
        print(f'Native fits {tenor}D/{window}M {stamp}: {len(use)} completed episodes; {record.get("sizing", {}).get("choices", [])}', flush=True)
        progress('fit', tenor=tenor, window=window, cutoff=stamp)
        gc.collect()


def load_schedule(tenor, window):
    paths = sorted((OUT / f'{tenor}D/models/{window}M').glob('*.joblib'))
    return {pd.Timestamp(p.stem):joblib.load(p) for p in paths}


def model_at(schedule, date):
    return schedule[max(stamp for stamp in schedule if stamp <= date)]


def sized_rows(rows, schedule):
    result = rows.copy()
    for i, row in result.iterrows():
        model = model_at(schedule, row.date)['sizing']
        if model is None or not row.native_features_available:
            continue
        leaf = model['tree'].apply(row[ENTRY_FEATURES].to_numpy(float)[None, :])[0]
        result.loc[i, 'learned_level'] = model['choices'][np.searchsorted(model['leaves'], leaf)]
    assert result.learned_level.isin([2, 3, 4]).all()
    return result


def management_callback(schedule, rv, cal, joint=False):
    positions = {date:i for i, date in enumerate(cal)}
    def callback(date, held):
        model = model_at(schedule, date)['joint_management' if joint else 'management']
        if model is None:
            return 1.
        if len(held) != 2:
            return 0.
        age = positions[date]-positions[held[0]['entry_date']]
        if age < 5 or age % 5:
            return 1.
        features = native_features(date, held, rv)
        if features is None:
            return 1.
        leaf = model['tree'].apply(np.array([features[f] for f in MANAGEMENT_FEATURES])[None, :])[0]
        target = float(model['choices'][np.searchsorted(model['leaves'], leaf)])
        if target == .5 and any(abs(p['quantity']) != abs(p['initial_quantity'])*p['management_fraction'] for p in held):
            # Conservatively defer unavailable fractional trims after splits;
            # integer flooring differences are still handled by the ledger.
            return 1.
        return target
    return callback


def save_replay(tenor, case, rows, callback=None):
    folder = OUT / f'{tenor}D/replays' / case
    if (folder / 'status.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    market = market_for(tenor, f'{tenor}D_{case}')
    cal = market.rates.index[market.rates.index >= '2018-01-01']
    nav, trades, decisions, audit = LONG.replay(rows, market, cal, tenor, callback)
    assert len(decisions) == len(rows) == 529
    if len(trades):
        entry = trades.loc[trades.reason.eq('entry')]
        assert entry.groupby('cohort').secid.nunique().eq(2).all()
        managed = trades.loc[trades.reason.isin(['learned_policy_trim', 'learned_policy_exit'])]
        if len(managed):
            assert (pd.to_datetime(managed.exit_signal_date) < managed.date).all()
    log = audit.pop('management_log', [])
    status = dict(case=case, tenor=tenor, **metrics(nav, 10_000_000),
        entries=int(decisions.status.eq('entered').sum()), proposals=len(rows),
        trims=int(trades.reason.eq('learned_policy_trim').sum()),
        exits=int(trades.reason.eq('learned_policy_exit').sum()), **audit)
    nav.to_parquet(folder / 'daily_nav.parquet')
    trades.to_csv(folder / 'trades.csv', index=False)
    decisions.to_csv(folder / 'decisions.csv', index=False)
    rows.to_parquet(folder / 'scored_allocations.parquet', index=False)
    dump(folder / 'management_log.json', log)
    dump(folder / 'status.json', status)
    pd.DataFrame(annual_metrics(nav, f'{tenor}D_{case}', 10_000_000)).to_csv(folder / 'annual.csv', index=False)
    print(f'{tenor}D {case}: excess Sharpe {status["annualized_excess_sharpe"]:.4f}, {status["entries"]} entries, {status["trims"]} trim / {status["exits"]} exit fills', flush=True)
    progress('replay', tenor=tenor, case=case, sharpe=status['annualized_excess_sharpe'])


def replay(tenor, window):
    rv = load_rv()
    rows = pd.read_parquet(OUT / f'{tenor}D/signal_rows.parquet')
    cal = pd.read_parquet(source.OUT / f'P/{tenor}D/market/rates.parquet').index
    schedule = load_schedule(tenor, window)
    sized = sized_rows(rows, schedule)
    save_replay(tenor, f'native_size_{window}M', sized)
    save_replay(tenor, f'native_manage_{window}M', rows,
                management_callback(schedule, rv, cal))
    save_replay(tenor, f'native_joint_{window}M', sized,
                management_callback(schedule, rv, cal, joint=True))


def controls(tenor):
    rows = pd.read_parquet(OUT / f'{tenor}D/signal_rows.parquet')
    for level in [2, 4]:
        save_replay(tenor, f'fixed_{level*25}_hold', rows.assign(learned_level=level))


def protected():
    if (OUT / 'protected_sources.json').exists():
        sources = json.loads((OUT / 'protected_sources.json').read_text())
        assert all(digest(path) == value for path, value in sources.items())
        return sources
    sources = {}
    for manifest in ['reports/p25_long_expiries/source_manifest.json',
                     'reports/p25_long_management/source_manifest.json',
                     'reports/long_expiry_parity/source_manifest.json']:
        for path, want in json.loads(Path(manifest).read_text()).items():
            assert digest(path) == want, path
            sources[path] = want
    for root in [source.OUT / 'P/replays', Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned'),
                 Path('artifacts/contracts/p25_long_management/replays/30D_manage_36M'),
                 PARITY / 'replays']:
        for path in root.rglob('*'):
            if path.is_file():
                sources[str(path)] = digest(path)
    dump(OUT / 'protected_sources.json', sources)
    return sources


def summarize():
    records, years, periods = [], [], []
    for tenor in TENORS:
        folders = [source.OUT / 'P/replays' / f'{tenor}D_{mode}' for mode in ['hold', 'manage_36M']] if INSTRUMENT == 'P25' else []
        folders += sorted((OUT / f'{tenor}D/replays').glob('*'))
        for folder in folders:
            if not (folder / 'status.json').exists():
                continue
            record = json.loads((folder / 'status.json').read_text())
            record['source_folder'] = str(folder)
            records.append(record)
            years.append(pd.read_csv(folder / 'annual.csv').assign(tenor=tenor, variant=folder.name))
            nav = pd.read_parquet(folder / 'daily_nav.parquet')
            for label, start, end in [('2018-2020', '2018-01-01', '2020-12-31'),
                                      ('2021-2023', '2021-01-01', '2023-12-31'),
                                      ('2024-partial2025', '2024-01-01', '2025-08-29'),
                                      ('2021-partial2025', '2021-01-01', '2025-08-29')]:
                returns = nav.excess_return.iloc[1:].loc[start:end]
                periods.append(dict(tenor=tenor, case=folder.name, period=label,
                    annualized_excess_sharpe=LONG.base.sharpe(returns)))
    result = pd.DataFrame(records)
    result.to_csv(REPORT / 'comparison.csv', index=False)
    pd.concat(years, ignore_index=True).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(periods).to_csv(REPORT / 'periods.csv', index=False)
    fits = [json.loads(p.read_text()) for p in OUT.glob('*D/models/*/*.json')]
    assert all(r['last_training_expiry'] is None or pd.Timestamp(r['last_training_expiry']) < pd.Timestamp(r['fit']) for r in fits)
    dump(REPORT / 'fits.json', fits)
    sources = protected()
    own = [Path(__file__), Path('tests/test_long_tenor_policy_search.py'),
           Path('scripts/run_signal_discovery_guarded.py'),
           Path('scripts/run_long_tenor_full_universe.py'),
           Path('scripts/run_long_tenor_flexible_actions.py')]
    own += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.joblib','.parquet']]
    dump(REPORT / 'source_manifest.json', {str(p):digest(p) for p in own})
    family = OUT.parent.name
    broad = family in ['long_tenor_full_universe', 'long_tenor_flexible_actions']
    flexible = family == 'long_tenor_flexible_actions'
    leaves = 2 if family == 'long_tenor_full_universe' else 4
    method = (
        'Same 529 frozen pair/date/direction proposals, native selected contracts, costs and risk limits. '
        + ('Entry ladder 0/25/50/75/100%, including an explicit learned zero-size action; fixed 50/100% controls retain the identical learned zero masks. '
           if flexible else 'Entry ladder 50/75/100%; no learned entry veto or leverage increase. ')
        + 'Cold start keeps original entry sizes and holds. Quarterly refits require original scheduled expiry before the fit, with minimum 16 completed episodes. '
        + ('48-month fits use baseline labels plus isolated minimum-size executions of every remaining native executable proposal. Inputs include a 252-session EWMA variance forecast, with explicit RV60 fallback when unavailable, term spread, IV spread/level, correlation and VIX. Shadow elective-exit costs are doubled during training only. '
           if broad else '24/48-month fits use original baseline-executed episodes. Inputs include native IV, RV20/60 carry, term spread, relative delta, beta/correlation and VIX. ')
        + f'Direct episodic policy search uses up to {leaves} tree leaves and exhaustive discrete action-map optimization. '
        + 'Management observes current features every five sessions from age five; irreversible hold/half/close of both legs, filled at the next observed session. '
    )
    dump(REPORT / 'audit.json', dict(status='complete', new_replays=sum(result.source_folder.str.startswith(str(OUT))),
        original_sources_unchanged=True, protected_sources=len(sources),
        completed_original_expiry_purge=True, causal_next_session_fills=True,
        one_worker=True, cpu_core_fraction=.5, rss_limit_bytes=3*1024**3,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        algorithm=f'Episodic direct policy search over at most {leaves}-leaf controllers, exhaustive action optimization; not neural Q-learning',
        shadow_training='Fractional isolated hedges; actual evaluation integer shared ledger'))
    (REPORT / 'findings.txt').write_text(
        f'NATIVE 180D / 365D {INSTRUMENT} SIZING AND MANAGEMENT\n\n'
        +method+'\n'
        'This extends the checkpoint policy-search method, not neural Q-learning. Shadow training uses fractional quantities and isolated stock hedge/borrow costs; option/hedge/dividend gross paths independently reconcile the unchanged baseline. All final results use the executable integer shared-cash ledger with original admission/risk limits and independently checked cash flows. Training commission .65; actual .03, 0.1 vol point options, 1bp stocks. All 529 proposals remain explicit.\n'
        'C75 uses actual same-strike/expiry call counterparts of P25 selections, approximately 75-delta at entry; calls use their native fitted Greeks with integer vega matching. Same 529 proposals, not the previously accepted-put-only locked parity subset. Call training and baselines separate.\n'
        'Missing history, American exercise, future dividend forecasts and financing approximation limitations remain. No new entry universe, untouched holdout or guaranteed improvement. All variants and loss periods retained.\n\n'
        +result[['tenor','case','annualized_excess_sharpe','entries','net_excess_pnl']].to_string(index=False)+'\n')
    print(result[['tenor','case','annualized_excess_sharpe','entries','net_excess_pnl']].to_string(index=False), flush=True)


def main():
    global OUT, REPORT, INSTRUMENT
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['test','prepare','fit','replay','controls','summarize','sweep'], required=True)
    p.add_argument('--tenor', type=int, choices=TENORS)
    p.add_argument('--window', type=int, choices=WINDOWS)
    p.add_argument('--instrument', choices=['P25', 'C75'], default='P25')
    args = p.parse_args()
    INSTRUMENT = args.instrument
    OUT, REPORT = OUT_BASE / INSTRUMENT, REPORT_BASE / INSTRUMENT
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    configure()
    protected()
    if args.phase == 'sweep':
        for instrument in ['P25', 'C75']:
            INSTRUMENT = instrument
            OUT, REPORT = OUT_BASE / instrument, REPORT_BASE / instrument
            OUT.mkdir(parents=True, exist_ok=True)
            REPORT.mkdir(parents=True, exist_ok=True)
            protected()
            for tenor in TENORS:
                prepare(tenor)
                for window in WINDOWS:
                    fit(tenor, window)
                    replay(tenor, window)
                    gc.collect()
                controls(tenor)
                gc.collect()
            summarize()
    elif args.phase == 'test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_long_tenor_policy_search','tests.test_p25_management'])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        dump(REPORT / 'tests.json', dict(tests_run=result.testsRun, successful=result.wasSuccessful()))
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.phase == 'prepare':
        prepare(args.tenor)
    elif args.phase == 'fit':
        fit(args.tenor, args.window)
    elif args.phase == 'replay':
        replay(args.tenor, args.window)
    elif args.phase == 'controls':
        controls(args.tenor)
    elif args.phase == 'summarize':
        summarize()
    protected()
    dump(OUT / f'resources_{args.phase}_{args.tenor}_{args.window}.json', dict(
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_core_fraction=.5, workers=1, rss_limit_bytes=3*1024**3))


if __name__ == '__main__':
    main()
