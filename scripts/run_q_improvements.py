"""Step-by-step fixes for the AMD/NVDA Q-learner (Q + open-trade P&L state), from q_diagnostics.md.

Each step adds one change on top of the changes kept so far. A change is kept only if it raises
the mean validation Sharpe across folds (test results are never used to decide). Pass the kept
changes with --keep, e.g. --add seeds20 --keep min_hold. Changes:
  0 baseline      Q + P&L as in run_q_learning.py (5 seeds, validation grid, age x P&L states)
  1 stop_loss     forced exit once the open trade is down more than L vol points,
                  L = 0.5 x std of the 21-day payoff in that fold's training data
  2 seeds20       20 seeds averaged instead of 5
  3 fewer_states  drop the early/late age split: flat, long win/lose, short win/lose
  4 fixed_config  no per-fold selection: 8 leaves, min leaf 252, gamma 0.9 in every fold
  5 min_hold      no voluntary exit before day 5, and entries trained with 1.5x cost
                  (evaluation always uses the true cost)
The stop-loss, holding limit and cost multiplier apply in training and testing alike.

Usage:  python -m scripts.run_q_improvements --pair AMD NVDA --add seeds20 --keep ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.q_panel import HOLD, build_pair
from scripts.run_q_learning import (EARLY, EXIT, LONG, SHORT, STAY, Market, daily_sharpe, fit_tree, folds, summarize)

STEPS = ['baseline', 'stop_loss', 'seeds20', 'fewer_states', 'fixed_config', 'min_hold']


class Spec:
    def __init__(self, changes):
        c = set(changes)
        self.stop = 'stop_loss' in c
        self.seeds = tuple(range(20 if 'seeds20' in c else 5))
        self.use_age = 'fewer_states' not in c
        self.fixed = 'fixed_config' in c
        self.min_hold = 5 if 'min_hold' in c else 0
        self.cost_mult = 1.5 if 'min_hold' in c else 1.0
        self.W = 9 if self.use_age else 5

    def pstate(self, pos, age, cum):
        if pos == 0:
            return 0
        d = 0 if pos > 0 else 1
        lose = 1 if cum < 0 else 0
        if self.use_age:
            return 1 + ((d * 2 + (1 if age > EARLY else 0)) * 2 + lose)
        return 1 + d * 2 + lose


def options(spec, pos, age):
    if pos == 0:
        return (STAY, LONG, SHORT)
    return (STAY,) if age < spec.min_hold else (STAY, EXIT)


def train_q(spec, m, leaf, nleaf, gamma, seed, stop_l, alpha=.05, eps=.1, epochs=20):
    rng = np.random.default_rng(seed)
    W = spec.W; Q = np.zeros((nleaf * W, 4)); n = m.n
    entry_half = m.half * spec.cost_mult
    for _ in range(epochs):
        pos, e, cum = 0, -1, 0.0
        for t in range(n - 1):
            age = t - e if pos else 0
            s = leaf[t] * W + spec.pstate(pos, age, cum)
            opts = options(spec, pos, age); res = {}
            for a in opts:
                if a == STAY and pos == 0:
                    r, np_, na, nc = 0.0, 0, 0, 0.0
                elif a in (LONG, SHORT):
                    q = 1 if a == LONG else -1
                    g = m.step(t, t, q); r = -entry_half[t] + g; np_, na, nc = q, 1, g
                elif a == EXIT:
                    r, np_, na, nc = -m.half[t], 0, 0, 0.0
                else:
                    g = m.step(t, e, pos); r = g; np_, na, nc = pos, age + 1, cum + g
                if np_ != 0 and (na >= HOLD or (spec.stop and nc < -stop_l)):   # matured or stopped -> closed next day
                    r -= m.half[t+1]; np_, na, nc = 0, 0, 0.0
                if t + 1 == n - 1:
                    target = r
                else:
                    s2 = leaf[t+1] * W + spec.pstate(np_, na, nc)
                    target = r + gamma * Q[s2, list(options(spec, np_, na))].max()
                Q[s, a] += alpha * (target - Q[s, a])
                res[a] = (np_, na, nc)
            a = opts[rng.integers(len(opts))] if rng.random() < eps else opts[int(np.argmax(Q[s, list(opts)]))]
            np_, na, cum = res[a]
            if np_ != 0 and na == 1:
                e = t
            pos = np_
    return Q


def simulate(spec, m, Q, leaf, stop_l):
    pnl = np.zeros(m.n); pos = 0; e = -1; cum = 0.0; trades = 0; inv = np.zeros(m.n, bool)
    for t in range(m.n - 1):
        if pos != 0 and (t - e >= HOLD or (spec.stop and cum < -stop_l)):
            pnl[t] -= m.half[t]; pos = 0
        age = t - e if pos else 0
        opts = options(spec, pos, age)
        a = opts[int(np.argmax(Q[leaf[t] * spec.W + spec.pstate(pos, age, cum), list(opts)]))]
        if pos == 0 and a in (LONG, SHORT):
            pos = 1 if a == LONG else -1; e = t; pnl[t] -= m.half[t]; trades += 1; cum = 0.0
        elif pos != 0 and a == EXIT:
            pnl[t] -= m.half[t]; pos = 0
        if pos != 0:
            g = m.step(t, e, pos); pnl[t] += g; cum += g; inv[t] = True
    if pos != 0:
        pnl[m.n-1] -= m.half[m.n-1]
    return pnl, trades, inv


def run_fold(spec, tr, va, te):
    trn = tr.loc[tr.index < va.index.min() - pd.Timedelta(days=45)]
    stop_l = 0.5 * trn.long_reward.std()
    grid = [(8, 252, .9)] if spec.fixed else [(l, m, g) for l in (4, 8) for m in (126, 252) for g in (.9, .98)]
    best = None
    for max_leaves, min_leaf, gamma in grid:
        to_state, nleaf = fit_tree(tr, max_leaves, min_leaf)
        Q = np.mean([train_q(spec, Market(trn), to_state(trn), nleaf, gamma, s, stop_l) for s in spec.seeds], axis=0)
        vs = daily_sharpe(simulate(spec, Market(va), Q, to_state(va), stop_l)[0])
        if best is None or vs > best[0]:
            best = (vs, Q, to_state, dict(leaves=max_leaves, min_leaf=min_leaf, gamma=gamma, nleaf=nleaf, stop_loss=stop_l))
    vs, Q, to_state, cfg = best
    pnl, trades, inv = simulate(spec, Market(te), Q, to_state(te), stop_l)
    return pnl, trades, inv, {**cfg, 'val_sharpe': vs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['AMD', 'NVDA'])
    ap.add_argument('--add', choices=STEPS, required=True, help='change tested in this run (baseline = none)')
    ap.add_argument('--keep', nargs='*', default=[], choices=STEPS[1:], help='changes already kept')
    ap.add_argument('--out', type=Path, default=Path('artifacts/q_improvements'))
    args = ap.parse_args()
    changes = [c for c in args.keep + [args.add] if c != 'baseline']
    step = '+'.join(changes) or 'baseline'; spec = Spec(changes)
    out = args.out / f'{args.pair[0]}_{args.pair[1]}'; out.mkdir(parents=True, exist_ok=True)
    p = build_pair(*args.pair)
    parts = []; cfgs = []; tr_total = 0; inv_all = []
    for y, tr, va, te in folds(p):
        pnl, trades, inv, cfg = run_fold(spec, tr, va, te)
        parts.append(pd.Series(pnl, index=te.index)); inv_all.append(inv); tr_total += trades
        cfgs.append({'fold': y, **cfg, 'test_sharpe': daily_sharpe(pnl)})
        print(step, y, round(cfgs[-1]['val_sharpe'], 2), round(cfgs[-1]['test_sharpe'], 2), flush=True)
    P = pd.concat(parts)
    s = summarize(step, P.index, P.to_numpy(), np.concatenate(inv_all).mean(), tr_total, [c['fold'] for c in cfgs])
    s['mean_val_sharpe'] = float(np.mean([c['val_sharpe'] for c in cfgs]))
    (out / f'{step}.json').write_text(json.dumps({'summary': s, 'folds': cfgs}, indent=1, default=float))
    print(json.dumps({k: v for k, v in s.items() if k != 'fold_sharpes'}, default=float))


if __name__ == '__main__':
    main()
