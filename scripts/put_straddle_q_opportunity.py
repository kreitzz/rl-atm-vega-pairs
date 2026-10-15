"""Forecast-informed fitted Q allocation with exact additive decision explanations.

Keep both entry bandits, funded-index execution, costs, timing and risk guards
unchanged. Replace the allocator's market representation. No new raw downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import put_straddle_q_frontier as core

CONFIG = Path('config/put_straddle_q_opportunity.json')
SPEC = json.loads(CONFIG.read_text())
SOURCE = Path(SPEC['source_root'])
OUT, REPORT = Path(SPEC['output_root']), Path(SPEC['report_root'])
SIGNALS = [Path('artifacts/contracts/straddle_feature_transfer/proposals.parquet'),
           Path('artifacts/contracts/p25_feature_search/signal_rows.parquet')]
ECONOMIC = ['pair_rank_strength', 'pair_vrp_gap', 'atm_iv_rv_gap', 'atm_completed_premium']
RISK = ['log_vol_ratio', 'correlation63', 'vix']
FORECAST = ['forecast_put_daily_bps', 'forecast_straddle_daily_bps']
GROUPS = dict(economic=ECONOMIC+RISK, forecast=FORECAST+RISK,
              combined=ECONOMIC+FORECAST+RISK)
ACTION = np.column_stack([np.ones(core.A), core.LEVELS, core.LEVELS[:, 0]*core.LEVELS[:, 1], core.LEVELS[:, 1]**2])
BASE_MODEL = core.model_for
GROUP = 'economic'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, default=str)+'\n')
    pending.replace(path)


def fingerprint():
    files = [CONFIG, Path(__file__), Path(core.__file__), Path('scripts/atm_p25_allocation_q.py'),
             Path('scripts/atm_p25_opportunity_policy.py'), SOURCE/'prepared.joblib',
             Path('data/raw/VIX_History.csv'), *SIGNALS]
    files += [SOURCE/f'transitions_{cap}.npz' for cap in SPEC['drawdown_budgets']]
    value = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = OUT/'sources.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Sources changed; use a new study root'
    else:
        dump(path, value)


def forecast_one(x, y, index):
    """Predict the next weekly average daily return using completed weeks only."""
    first = max(0, index-SPEC['forecast_training_weeks'])
    xx, yy = x[first:index], y[first:index]
    if len(xx) < SPEC['forecast_minimum_weeks']:
        return np.zeros(2), len(xx)
    center = xx.mean(axis=0); scale = np.maximum(xx.std(axis=0), 1e-6)
    design = np.column_stack([np.ones(len(xx)), np.clip((xx-center)/scale, -5, 5)])
    weights = np.exp2(-np.arange(len(xx)-1, -1, -1)/SPEC['forecast_half_life_weeks'])
    weights *= len(weights)/weights.sum()
    penalty = SPEC['forecast_ridge_alpha']*np.eye(design.shape[1]); penalty[0, 0] = .01
    beta = np.linalg.solve(design.T@(weights[:, None]*design)+penalty, design.T@(weights[:, None]*yy))
    return np.r_[1., np.clip((x[index]-center)/scale, -5, 5)]@beta, len(xx)


def prepare():
    import joblib
    from scripts.atm_p25_opportunity_policy import opportunity_features
    fingerprint()
    if (OUT/'prepared.joblib').exists():
        return
    data = joblib.load(SOURCE/'prepared.joblib')
    known = core.old.market_features(data['pair'], data['straddle'])
    a = pd.read_parquet(SIGNALS[0], columns=['date', 'iv', 'rv21', 'residual_memory'], use_threads=False)
    p = pd.read_parquet(SIGNALS[1], columns=['date', 'rank_strength', 'oriented_vrp_gap'], use_threads=False)
    opportunity = opportunity_features(data['pair'].index, a, p)
    frame = opportunity[ECONOMIC].copy()
    for field in RISK:
        frame[field] = known[field]
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    labels, weekly = [], []
    for m in data['weeks']:
        prior = int(m['indices'][0])-1
        if prior < 0:
            weekly.append(np.zeros(len(ECONOMIC+RISK)))
        else:
            weekly.append(frame.iloc[prior].to_numpy(float))
            for field in ['atm_source_date', 'pair_source_date']:
                stamp = opportunity[field].iloc[prior]
                assert pd.isna(stamp) or stamp <= m['feature_date']
        labels.append([1e4*data[name].excess_return.iloc[m['indices']].mean() for name in ['pair', 'straddle']])
    weekly = np.array(weekly); labels = np.array(labels)
    rows = []
    for i, m in enumerate(data['weeks']):
        pred, n = forecast_one(weekly, labels, i)
        assert all(w['end'] < m['cutoff'] for w in data['weeks'][:i])
        raw = dict(zip(ECONOMIC+RISK, weekly[i])); raw.update(zip(FORECAST, pred))
        m['markets'] = {group: np.array([raw[k] for k in names]) for group, names in GROUPS.items()}
        rows.append(dict(date=m['cutoff'], feature_date=m['feature_date'],
                         last_label_end=data['weeks'][i-1]['end'] if i else None,
                         forecast_training_weeks=n, realized_put_daily_bps=labels[i, 0],
                         realized_straddle_daily_bps=labels[i, 1], **raw))
    from scripts.research_atomic_io import atomic_dump
    atomic_dump(data, OUT/'prepared.joblib', joblib.dump)
    pd.DataFrame(rows).to_csv(REPORT/'weekly_features.csv', index=False)
    dump(REPORT/'protocol.json', dict(configuration=SPEC, features=GROUPS,
        timing='Every state uses the previous close. Forecast labels are completed weekly daily-average excess returns; forecasts fit only weeks ending before the next decision. Risk inputs are prior-close trailing63-session volatility ratio and correlation, not learned volatility forecasts.',
        reward=core.CFG['reward'], drawdown=core.CFG['drawdown'],
        interpretation='Feature availability and interpretation inherit the retained proposal archives; implied-minus-forecast variance is not a clean variance swap risk premium. No future option outcome is used as an input.',
        declaration_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest()))
    print(f'Prepared {len(rows)} weekly observations', flush=True)


def load():
    import joblib
    data = joblib.load(OUT/'prepared.joblib')
    for m in data['weeks']:
        m['market'] = m['markets'][GROUP]
    return data


def load_transitions(cap):
    with np.load(SOURCE/f'transitions_{cap}.npz') as archive:
        return {key: archive[key] for key in archive.files}


def features(data, week, states):
    week, states = np.asarray(week), np.asarray(states)
    if '_market' not in data:
        data['_market'] = np.stack([m['market'] for m in data['weeks']])
        data['_before'] = np.stack([m['before'] for m in data['weeks']])
    return np.column_stack([data['_market'][week], core.GRID[(states//core.A)%core.D],
                            data['_before'][week, states%core.A, :2]])


def standardize(values, center, scale):
    return np.column_stack([np.ones(len(values)), np.clip((values-center)/scale, -5, 5)])


def label(window, preference, gamma):
    return f'{GROUP}_w{window}_risk{preference:g}_g{gamma:g}'


def fit(data, trans, index, window, preference, gamma, cap):
    """Stream sufficient statistics once; fit all declared rewards/discounts."""
    cutoff = data['weeks'][index]['cutoff']
    train = np.array([i for i, m in enumerate(data['weeks'][:index])
        if m['end'] < cutoff and m['cutoff'] >= cutoff-pd.Timedelta(weeks=window)], int)
    if len(train) < SPEC['minimum_training_weeks']:
        return None
    rows = trans['row'][train]; width = rows.shape[1]
    dim = len(GROUPS[GROUP])+3
    total, total2 = np.zeros(dim), np.zeros(dim)
    for j, i in enumerate(train):
        z = features(data, np.full(width, i), rows[j]//core.A)
        total += z.sum(axis=0); total2 += (z*z).sum(axis=0)
    center = total/(len(train)*width)
    scale = np.maximum(np.sqrt(np.maximum(total2/(len(train)*width)-center*center, 0)), .1)
    dim += 1; coefficients = 5*dim
    following = trans['following'][train]
    unique, inverse = np.unique(following+np.arange(len(train))[:, None]*core.NS, return_inverse=True)
    inverse = inverse.ravel()
    projection = np.zeros((coefficients, len(unique)))
    gram = np.zeros((coefficients, coefficients)); reward_sums = np.zeros((coefficients, 3))
    for j, i in enumerate(train):
        z = standardize(features(data, np.full(width, i), rows[j]//core.A), center, scale)
        x = (ACTION[rows[j]%core.A, :, None]*z[:, None, :]).reshape(width, coefficients)
        gram += x.T@x/width
        reward_sums += x.T@np.column_stack([trans['profit'][i], trans['variance'][i], trans['stop_penalty'][i]])/width
        active = ~trans['done'][i]
        successors = inverse[j*width:(j+1)*width][active]
        np.add.at(projection, (np.arange(coefficients)[:, None], successors[None, :]), x[active].T/width)
    penalty = SPEC['ridge_alpha']*np.eye(coefficients); penalty[0, 0] = .01
    inverse_gram = np.linalg.inv(gram+penalty)
    next_week = train[unique//core.NS]+1; next_states = unique%core.NS
    nz = standardize(features(data, next_week, next_states), center, scale)
    legal = core.legal_actions(SPEC['drawdown_budgets'][cap])[(next_states//core.A)%core.D]
    last = max(data['weeks'][i]['end'] for i in train)
    for risk in SPEC['risk_preferences']:
        reward = reward_sums[:, 0]-risk*reward_sums[:, 1]-reward_sums[:, 2]
        immediate = inverse_gram@reward
        beta = immediate.copy()
        for g in [0., SPEC['discount']]:
            if g:
                for _ in range(SPEC['bellman_iterations']-1):
                    values = nz@beta.reshape(5, dim).T@ACTION.T
                    continuation = np.where(legal, values, -1e12).max(axis=1)
                    beta = inverse_gram@(reward+g*(projection@continuation))
            assert np.isfinite(beta).all() and last < cutoff
            model = dict(beta=beta.tolist(), immediate_beta=immediate.tolist(), center=center.tolist(), scale=scale.tolist(),
                cutoff=str(cutoff.date()), last_training_end=str(last.date()), training_weeks=len(train),
                window=window, preference=risk, gamma=g, cap=cap, group=GROUP,
                features=GROUPS[GROUP]+['drawdown_ratio_bucket', 'prior_put_exposure', 'prior_straddle_exposure'],
                critic_coefficients=coefficients, effective_training_weeks=len(train))
            dump(OUT/'models'/cap/label(window, risk, g)/(str(cutoff.to_period('M'))+'.json'), model)
    path = OUT/'models'/cap/label(window, preference, gamma)/(str(cutoff.to_period('M'))+'.json')
    print(json.dumps(dict(phase='fit', cap=cap, group=GROUP, month=str(cutoff.to_period('M')), window=window, weeks=len(train))), flush=True)
    return json.loads(path.read_text())


def model_for(data, trans, index, window, preference, gamma, cap):
    model = BASE_MODEL(data, trans, index, window, preference, gamma, cap)
    if model is None:
        return None
    z = standardize(features(data, np.full(core.NS, index), np.arange(core.NS)), np.array(model['center']), np.array(model['scale']))
    q = z@np.array(model['beta']).reshape(5, z.shape[1]).T@ACTION.T
    return dict(model, q=q.tolist())


def configure(group='economic'):
    global GROUP
    GROUP = group
    core.CFG = dict(core.CFG, **SPEC)
    core.OUT, core.REPORT = OUT, REPORT
    core.load, core.load_transitions = load, load_transitions
    core.fit, core.model_for, core.label = fit, model_for, label
    core.dump, core.sources = dump, fingerprint


def explain(model, raw, allowed, chosen=None):
    """Exact score-gap attribution; it explains the critic, not economic causality."""
    center, scale = np.array(model['center']), np.array(model['scale'])
    z = standardize(np.atleast_2d(raw), center, scale)[0]
    beta = np.array(model['beta']).reshape(5, len(z))
    immediate = np.array(model['immediate_beta']).reshape(5, len(z))
    scores = z@beta.T@ACTION.T
    order = np.array([1, 0, 5, 2, 3, 4, 6, 7])
    order = order[np.asarray(allowed, bool)[order]]
    winner = int(order[np.argmax(scores[order])])
    chosen = winner if chosen is None else int(chosen)
    alternatives = order[order != chosen]
    runner = int(alternatives[np.argmax(scores[alternatives])]) if len(alternatives) else chosen
    difference = ACTION[chosen]-ACTION[runner]
    terms = z*(difference@beta)
    near = float(z@(difference@immediate))
    gap = float(scores[chosen]-scores[runner])
    np.testing.assert_allclose(terms.sum(), gap, atol=1e-9, rtol=1e-9)
    flips = []
    for j in range(len(raw)):
        median = z.copy(); median[j+1] = 0.
        perturbed = median@beta.T@ACTION.T
        flips.append(int(order[np.argmax(perturbed[order])]) != winner)
    return dict(winner=winner, chosen=chosen, runner=runner, gap=gap,
        immediate_gap=near, continuation_gap=gap-near, scores=scores.tolist(),
        contributions=dict(zip(['intercept']+model['features'], terms.tolist())),
        center_ablation_flips=dict(zip(model['features'], flips)))


def validation(cap, group):
    configure(group); fingerprint()
    data = load(); trans = load_transitions(cap); records = []
    for window in SPEC['training_weeks']:
        for risk in SPEC['risk_preferences']:
            case = label(window, risk, SPEC['discount'])
            book, _ = core.replay(data, trans, cap, case, SPEC['validation_end'], risk, window, SPEC['discount'])
            records.append(dict(cap=cap, group=group, preference=risk, case=case, window=window,
                                **core.score(book, '2018-01-01', SPEC['validation_end'], risk)))
            pd.DataFrame(records).to_csv(REPORT/f'validation_{cap}_{group}.csv', index=False)
            print(json.dumps(records[-1]), flush=True)


def select():
    records = pd.concat([pd.read_csv(REPORT/f'validation_{c}_{g}.csv')
        for c in SPEC['drawdown_budgets'] for g in SPEC['groups']], ignore_index=True)
    assert len(records) == len(SPEC['drawdown_budgets'])*len(SPEC['groups'])*len(SPEC['training_weeks'])*len(SPEC['risk_preferences'])
    selected = []
    for (cap, risk), group in records.groupby(['cap', 'preference']):
        best = group.sort_values(['utility', 'case'], ascending=[False, True]).iloc[0].to_dict()
        selected.append(best)
    records.to_csv(REPORT/'validation_candidates.csv', index=False)
    dump(REPORT/'selection.json', dict(selected=selected, rule=SPEC['selection']))


def evaluate(cap):
    fingerprint()
    for entry in json.loads((REPORT/'selection.json').read_text())['selected']:
        if entry['cap'] != cap:
            continue
        configure(entry['group']); data = load(); trans = load_transitions(cap)
        for gamma in [SPEC['discount'], 0.]:
            case = label(int(entry['window']), entry['preference'], gamma)
            book, _ = core.replay(data, trans, cap, case, SPEC['evaluation_end'], entry['preference'], int(entry['window']), gamma)
            if gamma:
                earlier = pd.read_parquet(OUT/'replays'/cap/case/'validation.parquet', use_threads=False)
                np.testing.assert_allclose(book.excess_return.loc[earlier.index], earlier.excess_return, atol=1e-14, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['test', 'prepare', 'validation', 'select', 'evaluate'])
    parser.add_argument('--cap', choices=list(SPEC['drawdown_budgets']))
    parser.add_argument('--group', choices=SPEC['groups'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'test':
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_q_opportunity')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    elif args.phase == 'prepare': prepare()
    elif args.phase == 'validation': validation(args.cap, args.group)
    elif args.phase == 'select': select()
    elif args.phase == 'evaluate': evaluate(args.cap)


if __name__ == '__main__': main()
