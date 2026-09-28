"""Why is the Q-learner not doing better? Diagnostics on the spread-feature Q-learners.

Re-runs the same walk-forward fits as run_q_learning.py (same folds, grid, seeds), but keeps
everything needed to inspect them: a trade log, per-seed Q-tables, the whole validation grid,
and state-visit counts. It reuses the ideas of the repo's existing diagnostics
(diagnose_independent_pairs.py, test_direction_controls.py, report_positions.py) on this
daily-position format, and adds checks specific to Q-learning.

  1 overfitting        train vs validation vs test Sharpe per fold
  2 data per state     how many training days fall in each state the policy acts in
  3 gross vs cost      per-trade edge before and after costs
  4 exits              agent's early exits vs holding the same trade to day 21
  5 seed stability     test Sharpe of each seed's Q-table vs the averaged table
  6 selection          does validation Sharpe rank the grid the same way as test Sharpe?
  7 attribution        P&L by year, by direction, by leg, by entry state
  8 direction control  same trades with random long/short, 5,000 draws

Usage:  python -m scripts.diagnose_q_learning --pair JPM BAC
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.q_panel import HOLD, build_pair, leg_step
from scripts.run_q_learning import (ALLOWED_FLAT, ALLOWED_POS, EXIT, LONG, SHORT, STAY, Market, daily_sharpe,
                                    fit_tree, folds, pstate, q_decider, simulate, train_q)

SEEDS = (0, 1, 2, 3, 4)


def simulate_log(m, decide, leaf=None):
    """Same accounting as run_q_learning.simulate, plus a trade log with leg split and exit reason."""
    pnl = np.zeros(m.n); pos = 0; e = -1; cum = 0.0; trades = []; cur = None
    def close(t, reason):
        cur.update(exit=t, reason=reason, exit_cost=m.half[t]); trades.append(cur)
    for t in range(m.n - 1):
        if pos != 0 and t - e >= HOLD:
            pnl[t] -= m.half[t]; close(t, 'matured'); pos = 0
        a = decide(t, pos, t - e if pos else 0, cum)
        if pos == 0 and a in (LONG, SHORT):
            pos = 1 if a == LONG else -1; e = t; pnl[t] -= m.half[t]; cum = 0.0
            cur = dict(entry=t, direction=pos, gross=0.0, leg_a=0.0, leg_b=0.0, entry_cost=m.half[t],
                       leaf=int(leaf[t]) if leaf is not None else -1)
        elif pos != 0 and a == EXIT:
            pnl[t] -= m.half[t]; close(t, 'agent_exit'); pos = 0
        if pos != 0:
            k = t - e
            la = pos * leg_step(m.ia[e], m.ia[t], m.ia[t+1], m.la[t], k)
            lb = -pos * leg_step(m.ib[e], m.ib[t], m.ib[t+1], m.lb[t], k)
            pnl[t] += la + lb; cum += la + lb
            cur['gross'] += la + lb; cur['leg_a'] += la; cur['leg_b'] += lb
    if pos != 0:
        pnl[m.n-1] -= m.half[m.n-1]; close(m.n-1, 'fold_end')
    return pnl, pd.DataFrame(trades)


def held_gross(m, entry, direction):
    """Gross P&L of the same trade held to day 21 (or the fold end)."""
    end = min(entry + HOLD, m.n - 1)
    return sum(m.step(t, entry, direction) for t in range(entry, end)), end


def state_visits(m, leaf, Q, use_pnl):
    """States the greedy policy occupies on the training path, with how many days each."""
    W = 9 if use_pnl else 5
    counts = np.zeros(len(Q), int)
    dec = q_decider(Q, leaf, use_pnl)
    pos = 0; e = -1; cum = 0.0
    for t in range(m.n - 1):
        if pos != 0 and t - e >= HOLD:
            pos = 0
        counts[leaf[t] * W + pstate(pos, t - e if pos else 0, cum, use_pnl)] += 1
        a = dec(t, pos, t - e if pos else 0, cum)
        if pos == 0 and a in (LONG, SHORT):
            pos = 1 if a == LONG else -1; e = t; cum = 0.0
        elif pos != 0 and a == EXIT:
            pos = 0
        if pos != 0:
            cum += m.step(t, e, pos)
    return counts


def diagnose_fold(tr, va, te, use_pnl):
    trn = tr.loc[tr.index < va.index.min() - pd.Timedelta(days=45)]
    grid = []; best = None
    for max_leaves in (4, 8):
        for min_leaf in (126, 252):
            to_state, nleaf = fit_tree(tr, max_leaves, min_leaf)
            for gamma in (.9, .98):
                Qs = [train_q(Market(trn), to_state(trn), nleaf, gamma, s, use_pnl=use_pnl) for s in SEEDS]
                Q = np.mean(Qs, axis=0)
                run = lambda d, QQ=Q: daily_sharpe(simulate(Market(d), q_decider(QQ, to_state(d), use_pnl))[0])
                row = {'leaves': max_leaves, 'min_leaf': min_leaf, 'gamma': gamma, 'nleaf': nleaf,
                       'train_sharpe': run(trn), 'val_sharpe': run(va), 'test_sharpe': run(te)}
                grid.append(row)
                if best is None or row['val_sharpe'] > best[0]['val_sharpe']:
                    best = (row, Q, Qs, to_state, nleaf)
    row, Q, Qs, to_state, nleaf = best
    m = Market(te); leaf_te = to_state(te)
    pnl, trades = simulate_log(m, q_decider(Q, leaf_te, use_pnl), leaf_te)
    seed_sh = [daily_sharpe(simulate(m, q_decider(q, leaf_te, use_pnl))[0]) for q in Qs]
    visits = state_visits(Market(trn), to_state(trn), Q, use_pnl)
    if len(trades):
        held = [held_gross(m, r.entry, r.direction) for r in trades.itertuples()]
        trades['held_gross'] = [h[0] for h in held]
        trades['held_exit_cost'] = [m.half[h[1]] for h in held]
        trades['days'] = trades.exit - trades.entry
        trades['date'] = te.index[trades.entry]
    return dict(selected=row, grid=pd.DataFrame(grid), pnl=pd.Series(pnl, index=te.index), trades=trades,
                seed_sharpes=seed_sh, visits=visits, W=9 if use_pnl else 5)


def summarize(name, res, rng):
    out = {}
    # 1 overfitting
    out['folds'] = pd.DataFrame([{'fold': y, **{k: r['selected'][k] for k in ('nleaf', 'gamma', 'train_sharpe', 'val_sharpe', 'test_sharpe')}}
                                 for y, r in res.items()])
    # 2 data per state (training days in each state the policy occupies)
    v = np.concatenate([r['visits'][r['visits'] > 0] for r in res.values()])
    total_states = sum(len(r['visits']) for r in res.values())
    out['visits'] = {'states_total': int(total_states), 'states_never_visited': int(total_states - len(v)),
                     'median_days_per_visited_state': float(np.median(v)),
                     'share_visited_states_under_30_days': float((v < 30).mean())}
    T = pd.concat([r['trades'].assign(fold=y) for y, r in res.items() if len(r['trades'])], ignore_index=True)
    T['cost'] = T.entry_cost + T.exit_cost
    T['net'] = T.gross - T.cost
    # 3 gross vs cost
    out['edge'] = {'trades': len(T), 'mean_gross': T.gross.mean(), 'mean_cost': T.cost.mean(), 'mean_net': T.net.mean(),
                   'win_rate_gross': (T.gross > 0).mean(), 'win_rate_net': (T.net > 0).mean(),
                   'median_days_held': float(T.days.median())}
    # 4 exits
    early = T[T.reason == 'agent_exit']
    held_net = early.held_gross - early.entry_cost - early.held_exit_cost
    out['exits'] = {'share_trades_exited_early': len(early) / len(T),
                    'early_exit_mean_net': float(early.net.mean()) if len(early) else np.nan,
                    'same_trades_held_mean_net': float(held_net.mean()) if len(early) else np.nan,
                    'early_exit_better_share': float((early.net > held_net).mean()) if len(early) else np.nan,
                    'early_exits_while_losing_share': float((early.gross < 0).mean()) if len(early) else np.nan,
                    'total_gain_from_early_exits': float((early.net - held_net).sum()) if len(early) else 0.0}
    # 5 seeds
    out['seeds'] = pd.DataFrame([{'fold': y, 'ensemble': r['selected']['test_sharpe'], 'seed_min': min(r['seed_sharpes']),
                                  'seed_max': max(r['seed_sharpes']), 'seed_sd': float(np.std(r['seed_sharpes']))}
                                 for y, r in res.items()])
    # 6 selection: rank agreement between validation and test over the 8-config grid
    sel = []
    for y, r in res.items():
        g = r['grid']
        rho = spearmanr(g.val_sharpe, g.test_sharpe)[0] if g.test_sharpe.nunique() > 1 and g.val_sharpe.nunique() > 1 else np.nan
        sel.append({'fold': y, 'rank_corr_val_test': rho, 'selected_test': r['selected']['test_sharpe'],
                    'grid_median_test': g.test_sharpe.median(), 'grid_best_test': g.test_sharpe.max()})
    out['selection'] = pd.DataFrame(sel)
    # 7 attribution
    P = pd.concat([r['pnl'] for r in res.values()])
    out['by_year'] = P.groupby(P.index.year).sum()
    out['by_direction'] = T.groupby(T.direction.map({1: 'long A / short B', -1: 'short A / long B'})).agg(
        trades=('net', 'size'), gross=('gross', 'sum'), cost=('cost', 'sum'), net=('net', 'sum'), leg_a=('leg_a', 'sum'), leg_b=('leg_b', 'sum'))
    out['by_reason'] = T.groupby('reason').agg(trades=('net', 'size'), mean_gross=('gross', 'mean'), mean_net=('net', 'mean'), days=('days', 'mean'))
    # 8 direction control: same dates, durations and costs; direction random
    signs = rng.choice([-1.0, 1.0], size=(5000, len(T)))
    rand_net = (signs * T.gross.to_numpy()[None, :] - T.cost.to_numpy()[None, :]).sum(axis=1)
    out['direction_control'] = {'actual_net': float(T.net.sum()), 'random_median_net': float(np.median(rand_net)),
                                'percentile_of_actual': float((rand_net < T.net.sum()).mean()),
                                'hit_rate': float((T.gross > 0).mean())}
    return out, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, required=True)
    ap.add_argument('--out', type=Path, default=Path('artifacts/q_diagnostics'))
    args = ap.parse_args()
    a, b = args.pair; name = f'{a}_{b}'
    out = args.out / name; out.mkdir(parents=True, exist_ok=True)
    p = build_pair(a, b)
    rng = np.random.default_rng(2026)
    result = {}
    for model, use_pnl in (('q_hybrid', False), ('q_hybrid_pnl', True)):
        res = {}
        for y, tr, va, te in folds(p):
            res[y] = diagnose_fold(tr, va, te, use_pnl)
            print(name, model, y, round(res[y]['selected']['test_sharpe'], 2), flush=True)
        s, T = summarize(model, res, rng)
        T.to_csv(out / f'trades_{model}.csv', index=False)
        pd.concat([r['grid'].assign(fold=y) for y, r in res.items()]).to_csv(out / f'grid_{model}.csv', index=False)
        result[model] = s
        for k, v in s.items():
            print(f'\n--- {model}: {k}')
            print(v.round(3).to_string() if hasattr(v, 'round') and not isinstance(v, dict) else json.dumps(v, indent=1, default=float))
    ser = {m: {k: (v.to_dict(orient='records') if isinstance(v, pd.DataFrame) else
                   {str(i): float(x) for i, x in v.items()} if isinstance(v, pd.Series) else v)
               for k, v in s.items()} for m, s in result.items()}
    (out / 'summary.json').write_text(json.dumps(ser, indent=1, default=float))


if __name__ == '__main__':
    main()
