"""Sharpe-directed tabular policy search for the AMD/NVDA IV spread."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = np.array([-1, 0, 1])  # +1 = long AMD / short NVDA


def forward_rv(returns, entries):
    """Annualized RMS log returns in (entry, entry + 30 calendar days]."""
    returns = returns.sort_index()
    rows = []
    for date in entries:
        end = date + pd.Timedelta(days=30)
        if end > returns.index.max():
            continue
        window = returns.loc[(returns.index > date) & (returns.index <= end)]
        if len(window) < 15 or window.isna().any().any() or (window <= -1).any().any():
            continue
        rv = np.sqrt(252 * np.square(np.log1p(window)).mean()) * 100
        rows.append({'date': date, 'end': end, 'rv_amd': rv['AMD'], 'rv_nvda': rv['NVDA']})
    return pd.DataFrame(rows, columns=['date', 'end', 'rv_amd', 'rv_nvda']).set_index('date')


def load_data(root):
    om = root / 'optionmetrics'
    files = sorted((om / 'atm_iv_compact').rglob('*.parquet'))
    files += sorted((om / 'atm_implied_volatility').glob('*.parquet'))
    frames = [pd.read_parquet(f, columns=['secid', 'date', 'impl_volatility'],
              filters=[('secid', 'in', [101121., 108321.]), ('days', '=', 30),
                       ('delta', '=', 50), ('cp_flag', '=', 'C')]) for f in files]
    iv = pd.concat([f for f in frames if len(f)]).drop_duplicates()
    iv['ticker'] = iv.secid.map({101121.: 'AMD', 108321.: 'NVDA'})
    iv = iv.pivot(index='date', columns='ticker', values='impl_volatility')
    iv = (iv.where(iv > 0).dropna() * 100).rename(columns={'AMD': 'iv_amd', 'NVDA': 'iv_nvda'})
    frames = []
    for f in sorted((root / 'crsp_underlying_daily').glob('*.parquet')):
        # Some downloaded year partitions are empty with null-typed identifiers.
        d = pd.read_parquet(f, columns=['permno', 'date', 'retx'])
        frames.append(d[d.permno.isin([61241, 86580])])
    ret = pd.concat([f for f in frames if len(f)]).drop_duplicates()
    ret['ticker'] = ret.permno.map({61241: 'AMD', 86580: 'NVDA'})
    ret = ret.pivot(index='date', columns='ticker', values='retx').sort_index()
    data = iv.join(forward_rv(ret, iv.index), how='inner').sort_index()
    data['spread'] = data.iv_amd - data.iv_nvda
    data['long_reward'] = (data.rv_amd - data.iv_amd) - (data.rv_nvda - data.iv_nvda)
    return data


def split_data(data):
    train = data[(data.index < '2017-01-01') & (data.end < '2017-01-01')]
    val = data[(data.index >= '2017-01-01') & (data.index < '2021-01-01') & (data.end < '2021-01-01')]
    test = data[data.index >= '2021-01-01']
    return train, val, test


def states(spread, edges):
    return np.searchsorted(edges, spread, side='right')


def payoff_sharpe(reward):
    """Unannualized mean / sample SD; all-flat has score zero by convention."""
    reward = np.asarray(reward, dtype=float)
    sd = reward.std(ddof=1)
    return float(reward.mean() / sd) if sd > 1e-12 else 0.0


def net_rewards(actions, reward, cost):
    return actions * reward - cost * (actions != 0)


def policy_scores(policies, state, reward, nstates, cost=0.5):
    """Exact full-episode Sharpe using per-state sufficient statistics."""
    n = len(reward)
    sums = np.bincount(state, weights=reward, minlength=nstates)
    squares = np.bincount(state, weights=reward ** 2, minlength=nstates)
    counts = np.bincount(state, minlength=nstates)
    traded = policies ** 2
    mean = (policies @ sums - cost * (traded @ counts)) / n
    second = traded @ squares - 2 * cost * (policies @ sums) + cost ** 2 * (traded @ counts)
    var = np.maximum((second - n * mean ** 2) / (n - 1), 0)
    return np.divide(mean, np.sqrt(var), out=np.zeros_like(mean), where=var > 1e-24)


def fit_policy(state, reward, nstates, seed, cost=0.5, iterations=80, population=256):
    """Cross-entropy episodic policy search maximizing training payoff Sharpe.

    An episode replays the training history; its terminal objective is Sharpe.
    Candidate policies map each IV-spread bucket to a fixed action. No future
    outcomes enter their observations. This is direct policy search, not TD Q-learning.
    """
    rng = np.random.default_rng(seed)
    probabilities = np.full((nstates, 3), 1 / 3)
    best = np.zeros(nstates, dtype=int)
    best_score = 0.0
    for _ in range(iterations):
        draws = rng.random((population, nstates, 1))
        indices = (draws > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2)
        indices = np.minimum(indices, 2)
        policies = ACTIONS[indices]
        policies[0] = best
        scores = policy_scores(policies, state, reward, nstates, cost)
        winner = int(scores.argmax())
        if scores[winner] > best_score:
            best_score = float(scores[winner])
            best = policies[winner].copy()
        elite = policies[np.argsort(scores)[-max(2, population // 10):]]
        frequencies = np.stack([(elite == a).mean(axis=0) for a in ACTIONS], axis=1)
        probabilities = .3 * probabilities + .7 * frequencies
        probabilities = .97 * probabilities + .03 / 3
    return best, best_score


def interval(x, seed=2026, repeats=2000, block=63, statistic=np.mean):
    """Circular moving-block bootstrap preserving local outcome dependence."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = []
    for _ in range(repeats):
        starts = rng.integers(n, size=int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        means.append(statistic(x[idx]))
    return np.quantile(means, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/sharpe_pass'))
    parser.add_argument('--cost', type=float, default=0.5,
                        help='All-in round-trip pair cost, in vol points per $1 vega per leg.')
    args = parser.parse_args()
    if args.cost < 0:
        parser.error('--cost must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.source_root)
    train, val, test = split_data(data)
    assert len(train) and len(val) and len(test)
    assert train.end.max() < val.index.min() and val.end.max() < test.index.min()
    candidates = []
    models = {}
    for bins in [5, 10, 20]:
        edges = np.unique(np.quantile(train.spread, np.arange(1, bins) / bins))
        fits = [fit_policy(states(train.spread, edges), train.long_reward.to_numpy(), len(edges)+1, seed, cost=args.cost)
                for seed in range(5)]
        policy, train_score = max(fits, key=lambda fit: fit[1])
        validation = net_rewards(policy[states(val.spread, edges)], val.long_reward.to_numpy(), args.cost)
        candidates.append({'bins': bins, 'train_sharpe': train_score,
                           'validation_sharpe': payoff_sharpe(validation)})
        models[bins] = (edges, policy, fits)
    selected = max(candidates, key=lambda x: x['validation_sharpe'])['bins']
    edges, learned_policy, fits = models[selected]
    test_states = states(test.spread, edges)
    actions = learned_policy[test_states]
    constant = int(max(ACTIONS, key=lambda a: payoff_sharpe(net_rewards(a, train.long_reward.to_numpy(), args.cost))))
    strategies = {'sharpe_policy': actions, 'flat': np.zeros(len(test), dtype=int),
                  'train_best_constant': np.full(len(test), constant),
                  'always_long_amd': np.ones(len(test), dtype=int),
                  'always_long_nvda': -np.ones(len(test), dtype=int),
                  'short_higher_iv': -np.sign(test.spread.to_numpy()).astype(int)}
    scores = []
    for name, action in strategies.items():
        reward = net_rewards(action, test.long_reward.to_numpy(), args.cost)
        lo, hi = interval(reward)
        scores.append({'strategy': name, 'mean_reward': float(reward.mean()),
                       'ci_low': lo, 'ci_high': hi, 'payoff_sharpe': payoff_sharpe(reward),
                       'sharpe_ci': interval(reward, statistic=payoff_sharpe), 'trade_fraction': float(np.mean(action != 0)),
                       'long_amd_fraction': float(np.mean(action == 1)),
                       'win_fraction_trades': float(np.mean(reward[action != 0] > 0)) if np.any(action != 0) else None})
    out = test.copy()
    out['action'] = actions
    out['gross_reward'] = actions * out.long_reward
    out['cost'] = args.cost * (actions != 0)
    out['reward'] = net_rewards(actions, out.long_reward, args.cost)
    out.to_csv(args.output/'test_predictions.csv')
    data.to_parquet(args.output/'dataset.parquet')
    pd.DataFrame(scores).to_csv(args.output/'scores.csv', index=False)
    annual = out.groupby(out.index.year).agg(entries=('reward','size'), mean_reward=('reward','mean'), long_amd_fraction=('action',lambda x: (x == 1).mean()))
    annual.to_csv(args.output/'annual.csv')
    policy = pd.DataFrame({'lower': np.r_[-np.inf, edges], 'upper': np.r_[edges, np.inf],
                           'action': learned_policy})
    policy.to_csv(args.output/'policy.csv', index=False)
    diff = out.reward.to_numpy() - net_rewards(constant, out.long_reward.to_numpy(), args.cost)
    summary = {'source_root': str(args.source_root), 'candidates': candidates, 'selected_bins': selected,
               'splits': {name: {'count':len(d), 'first':str(d.index.min().date()), 'last':str(d.index.max().date()), 'last_outcome':str(d.end.max().date())} for name,d in [('train',train),('validation',val),('test',test)]},
               'scores': scores, 'increment_vs_train_constant': {'mean':float(diff.mean()), 'ci':interval(diff)},
               'seed_test_sharpes': [payoff_sharpe(net_rewards(m[test_states],test.long_reward.to_numpy(), args.cost)) for m, _ in fits],
               'pair_cost_sensitivity_fixed_policy': {str(cost): {'mean':float(net_rewards(actions,out.long_reward,cost).mean()), 'sharpe':payoff_sharpe(net_rewards(actions,out.long_reward,cost))} for cost in [0,.25,.5,1]},
               'assumptions': {'dollar_vega_per_leg':1, 'cost_training':args.cost, 'horizon_calendar_days':30, 'rv_annualization':252, 'sharpe_annualized':False, 'bootstrap_block_entries':63, 'bootstrap_repeats':2000, 'iterations':80, 'population':256, 'objective':'unannualized payoff mean / sample SD; flat entries included', 'seeds':list(range(5))}}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(annual.to_string())
    print(policy.to_string(index=False))


if __name__ == '__main__':
    main()
