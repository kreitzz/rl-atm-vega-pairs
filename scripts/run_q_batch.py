"""Deterministic, exhaustive fitted tabular Q iteration on the existing proxy market.

Run: conda run -n xfund python -m scripts.run_q_batch --pair AMD NVDA
All feature transforms and empirical transitions are fitted on training only.
Counterfactual positions improve coverage, not the number of independent dates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.tree import DecisionTreeRegressor

from scripts.q_panel import HOLD, build_pair, load_raw, full_hold_payoff
from scripts.run_q_learning import (Market, STAY, LONG, SHORT, EXIT, FEATURES,
                                    daily_sharpe, folds, summarize)

VARIANTS = {
    'batch_coarse': ('original', False, False),
    'batch_age': ('original', True, False),
    'economic': ('economic', True, False),
    'economic_earnings': ('earnings', True, False),
    'economic_entry': ('economic', True, True),
    'spread_only': ('spread', True, False),
}


def slices(d):
    boundaries = [0, len(d)]
    if 'calendar_position' in d:
        boundaries = np.r_[0, np.flatnonzero(np.diff(d.calendar_position)!=1)+1, len(d)]
    return [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]


def respect_gaps(p, a, b):
    """Do not treat feature-filtered or missing-IV dates as adjacent trading days."""
    _, returns, _ = load_raw()
    calendar = returns.index  # includes dates with missing returns for either leg
    p = p.copy()
    p['calendar_position'] = calendar.get_indexer(p.index)
    for section in slices(p):
        block = p.iloc[section]
        p.loc[block.index, 'long_reward'] = full_hold_payoff(block)
        p.loc[block.index, 'end'] = pd.Series(block.index, index=block.index).shift(-HOLD)
    return p


def market(d):
    m = Market(d)
    m.frame = d
    return m


def features(d, mode):
    # Explicit allowlist: no future RV, next-day returns, rewards, or outcome dates.
    if mode == 'original':
        return d[FEATURES]
    if mode == 'spread':
        return d[['spread']]
    a, b = d.iv_amd, d.iv_nvda
    ra, rb = d.trailing_rv_amd, d.trailing_rv_nvda
    x = pd.DataFrame({'spread': a-b, 'iv_level': (a+b)/2,
                      'relative_carry': (ra**2-a**2)/(2*a) - (rb**2-b**2)/(2*b)}, index=d.index)
    if mode == 'earnings':
        x['earn_a'] = d.amd_earnings_days_capped
        x['earn_b'] = d.nvda_earnings_days_capped
    return x


class Encoder:
    def __init__(self, train, mode, exact_age, entry_context, leaves):
        self.mode, self.exact_age, self.entry_context = mode, exact_age, entry_context
        self.ages = HOLD-1 if exact_age else 2
        self.contexts = 2 if entry_context else 1
        self.width = 1 + 2*self.ages*self.contexts
        self.tree = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=252, random_state=2026)
        self.tree.fit(features(train, mode), train.long_reward)
        self.ids = np.unique(self.tree.apply(features(train, mode)))
        self.nstates = len(self.ids)*self.width

    def leaves(self, d):
        return np.searchsorted(self.ids, self.tree.apply(features(d, self.mode)))

    def state(self, leaf, pos=0, age=0, context=0):
        bucket = np.maximum(np.asarray(age)-1, 0) if self.exact_age else (np.asarray(age)>10).astype(int)
        slot = 1 + (((np.asarray(pos)<0).astype(int)*self.ages + bucket)*self.contexts + context)
        return np.asarray(leaf)*self.width + np.where(np.asarray(pos)==0, 0, slot)

    def context(self, market, t, e, pos):
        if not self.entry_context:
            return np.zeros_like(np.asarray(t), dtype=int)
        # Observable mark displacement, not future profit or accumulated sunk carry.
        return (pos*((market.ia[t]-market.ib[t]) - (market.ia[e]-market.ib[e])) < 0).astype(int)


def transitions(m, leaf, enc):
    """Enumerate every feasible (date, entry age, direction, legal action).

    Forced expiry is charged at the next close, then the next decision is flat.
    At the final close an open trade is liquidated and ALL bootstrapping stops.
    """
    if hasattr(m, 'frame') and len(slices(m.frame)) > 1:
        chunks = [transitions(Market(m.frame.iloc[s]), leaf[s], enc) for s in slices(m.frame)]
        return tuple(np.concatenate([c[i] for c in chunks]) for i in range(4))
    rows = []
    for pos in (0, 1, -1):
        for age in ((0,) if pos == 0 else range(1, HOLD)):
            t = np.arange(age, m.n-1)
            e = t-age
            state = enc.state(leaf[t], pos, age, enc.context(m, t, e, pos))
            for action in ((STAY, LONG, SHORT) if pos == 0 else (STAY, EXIT)):
                r = np.zeros(len(t))
                np_, na, ne = pos, age+1, e
                if pos == 0 and action == STAY:
                    np_, na = 0, 0
                elif action in (LONG, SHORT):
                    np_, na, ne = (1 if action == LONG else -1), 1, t
                    r = m.step(t, t, np_) - m.half[t]
                elif action == EXIT:
                    np_, na = 0, 0
                    r = -m.half[t]
                else:
                    r = m.step(t, e, pos)
                if np_ != 0 and na >= HOLD:
                    r = r-m.half[t+1]
                    np_, na = 0, 0
                terminal = t+1 == m.n-1
                if np_ != 0:
                    r = r - terminal*m.half[t+1]
                nxt = enc.state(leaf[t+1], np_, na, enc.context(m, t+1, ne, np_))
                rows.append((state*4+action, r, nxt, ~terminal))
    return tuple(np.concatenate([row[i] for row in rows]) for i in range(4))


def fit_q(m, leaf, enc, gamma, tol=1e-9, max_iter=2000):
    sa, r, nxt, live = transitions(m, leaf, enc)
    n = enc.nstates
    counts = np.bincount(sa, minlength=n*4)
    denom = np.maximum(counts, 1)
    reward = np.bincount(sa, weights=r, minlength=n*4)/denom
    transition = coo_matrix((live/denom[sa], (sa, nxt)), shape=(n*4, n)).tocsr()
    legal = np.zeros((n, 4), bool)
    flat = np.arange(n) % enc.width == 0
    legal[flat, :3] = True
    legal[~flat, STAY] = True
    legal[~flat, EXIT] = True
    # Unobserved inventory states close; unobserved flat states stay flat.
    available = legal & (counts.reshape(n, 4)>0)
    empty = ~available.any(axis=1)
    available[empty & flat, STAY] = True
    available[empty & ~flat, EXIT] = True
    q = np.where(available, 0., -np.inf)
    for iteration in range(max_iter):
        update = (reward + gamma*(transition @ q.max(axis=1))).reshape(n, 4)
        residual = np.max(np.abs(update[available]-q[available]))
        q = np.where(available, update, -np.inf)
        if residual < tol:
            break
    if residual >= tol:
        raise RuntimeError(f'Bellman iteration did not converge: {residual}')
    return q, {'iterations': iteration+1, 'residual': float(residual),
               'unvisited_states': int(empty.sum()), 'states': n,
               'median_state_action_rows': float(np.median(counts[counts>0]))}


def evaluate(d, enc, q):
    sections = slices(d)
    if len(sections)>1:
        chunks = [evaluate(d.iloc[s], enc, q) for s in sections]
        return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks]), sum(c[2] for c in chunks)
    m, leaf = Market(d), enc.leaves(d)
    pnl, invested = np.zeros(m.n), np.zeros(m.n, bool)
    pos, e, trades = 0, 0, 0
    for t in range(m.n-1):
        if pos and t-e >= HOLD:
            pnl[t] -= m.half[t]
            pos = 0
        state = enc.state(leaf[t], pos, t-e if pos else 0, enc.context(m, t, e, pos))
        action = int(np.argmax(q[state]))
        if pos == 0 and action in (LONG, SHORT):
            pos, e = (1 if action == LONG else -1), t
            pnl[t] -= m.half[t]
            trades += 1
        elif pos and action == EXIT:
            pnl[t] -= m.half[t]
            pos = 0
        if pos:
            pnl[t] += m.step(t, e, pos)
            invested[t] = True
    if pos:
        pnl[-1] -= m.half[-1]
    return pnl, invested, trades


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair', nargs=2, default=['AMD', 'NVDA'])
    ap.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=[v for v in VARIANTS if v != 'spread_only'])
    ap.add_argument('--out', type=Path)
    ap.add_argument('--respect-gaps', action='store_true', default=True, help='Split missing-date gaps (default)')
    ap.add_argument('--legacy-gaps', action='store_false', dest='respect_gaps', help='Reproduce the existing panel treatment, including compressed missing dates')
    args = ap.parse_args()
    if args.out is None:
        args.out = Path('artifacts/q_batch_strict' if args.respect_gaps else 'artifacts/q_batch')
    out = args.out / '_'.join(args.pair)
    out.mkdir(parents=True, exist_ok=True)
    panel = build_pair(*args.pair)
    if args.respect_gaps:
        panel = respect_gaps(panel, *args.pair)
    (out/'run_config.json').write_text(json.dumps({'pair': args.pair, 'variants': args.variants,
                                                'respect_gaps': args.respect_gaps}, indent=2))
    summaries, selections, grid = [], [], []
    parts = {v: [] for v in args.variants}
    parts['validation_selected'] = []
    years = []
    for year, tr, va, te in folds(panel):
        years.append(year)
        trn = tr.loc[tr.index < va.index.min()-pd.Timedelta(days=45)]
        fold_best = None
        for variant in args.variants:
            best = None
            for leaves in (4, 8):
                enc = Encoder(trn, *VARIANTS[variant], leaves)
                for gamma in (.9, .98):
                    q, diag = fit_q(market(trn), enc.leaves(trn), enc, gamma)
                    vp, vi, vt = evaluate(va, enc, q)
                    score = daily_sharpe(vp)
                    cfg = dict(fold=year, variant=variant, leaves=leaves, gamma=gamma,
                               val_sharpe=score, val_trades=vt, **diag)
                    grid.append(cfg)
                    if best is None or score > best[0]:
                        best = (score, enc, q, cfg)
            score, enc, q, cfg = best
            # Persist the selected model before evaluating test. No test-driven selection.
            import pickle
            with (out / f'{variant}_{year}.pkl').open('wb') as f:
                pickle.dump({'tree': enc.tree, 'leaf_ids': enc.ids, 'q': q, 'config': cfg,
                             'mode': enc.mode, 'exact_age': enc.exact_age,
                             'entry_context': enc.entry_context, 'width': enc.width}, f)
            tp, inv, nt = evaluate(te, enc, q)
            frame = pd.DataFrame({'pnl': tp, 'invested': inv}, index=te.index)
            parts[variant].append((frame, nt))
            selections.append({**cfg, 'test_sharpe': daily_sharpe(tp)})
            if fold_best is None or score > fold_best[0]:
                fold_best = (score, variant, frame, nt, cfg)
            print(year, variant, f'val={score:.3f} test={daily_sharpe(tp):.3f} trades={nt}', flush=True)
        _, variant, frame, nt, cfg = fold_best
        parts['validation_selected'].append((frame, nt))
        selections.append({**cfg, 'variant': 'validation_selected', 'selected_variant': variant,
                           'test_sharpe': daily_sharpe(frame.pnl)})
    for variant, chunks in parts.items():
        d = pd.concat([c[0] for c in chunks])
        d.to_csv(out / f'daily_{variant}.csv', index_label='date')
        summaries.append(summarize(variant, d.index, d.pnl, d.invested.mean(), sum(c[1] for c in chunks), years))
    pd.DataFrame(grid).to_csv(out/'validation_grid.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'selected_configs.csv', index=False)
    (out/'summary.json').write_text(json.dumps(summaries, indent=2))
    print(pd.DataFrame(summaries)[['model', 'sharpe', 'ann_pnl', 'max_dd', 'round_trips']].to_string(index=False))


if __name__ == '__main__':
    main()
