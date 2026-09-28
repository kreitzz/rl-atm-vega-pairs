"""Tabular Q-learning for ATM vega pairs, compared with the repo's current model.

Models (same data, costs, walk-forward folds, daily marked P&L):
  q_hybrid   tree leaf x position x holding-age state, Q-learning on daily P&L
  paper      replication of the stat-arb paper: 4-step spread-move state (256) x position,
             reward A*(theta - X) - c|A|, trained on simulated OU paths
  threshold  distance rule: short the rich leg when |z| > 1, exit when z crosses 0
  current    repo method: tree + cross-entropy Sharpe search, 9-size ladder, 30-day cohorts

Usage:  python -m scripts.run_q_learning --pair JPM BAC
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.q_panel import HOLD, build_pair, leg_step
from scripts.run_first_pass import fit_policy, net_rewards, payoff_sharpe

FEATURE_SETS = {
    'spread': ['spread', 'amd_earnings_days_capped', 'nvda_earnings_days_capped', 'trailing_rv_amd', 'trailing_rv_nvda'],
    'ratio': ['iv_ratio', 'amd_earnings_days_capped', 'nvda_earnings_days_capped', 'trailing_rv_amd', 'trailing_rv_nvda'],
}
FEATURES = FEATURE_SETS['spread']      # tree inputs; switched by --features
LADDER = np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
EARLY = 10                        # holding-age bucket boundary (days)
STAY, LONG, SHORT, EXIT = 0, 1, 2, 3


# ---------------------------------------------------------------- folds
def folds(p):
    out = []
    for y in range(2010, 2025, 2):
        vs, ts, te = pd.Timestamp(y-2, 1, 1), pd.Timestamp(y, 1, 1), pd.Timestamp(y+2, 1, 1)
        tr = p[(p.index < vs) & (p.end < vs)]
        va = p[(p.index >= vs) & (p.index < ts)]
        te_ = p[(p.index >= ts) & (p.index < te)]
        if len(te_):
            out.append((y, tr, va, te_))
    return out


# ---------------------------------------------------------------- market arrays
class Market:
    def __init__(self, d):
        self.ia = d.iv_amd.to_numpy(float); self.ib = d.iv_nvda.to_numpy(float)
        self.la = d.lr_amd_next.to_numpy(float); self.lb = d.lr_nvda_next.to_numpy(float)
        self.half = d.cost.to_numpy(float) / 2       # one side of the round trip
        self.n = len(d)

    def step(self, t, e, pos):
        """Pair P&L from close t to t+1 for a position entered at close e (age t-e)."""
        k = t - e
        return pos * (leg_step(self.ia[e], self.ia[t], self.ia[t+1], self.la[t], k)
                      - leg_step(self.ib[e], self.ib[t], self.ib[t+1], self.lb[t], k))


def simulate(m, decide):
    """Run a policy day by day. decide(t, pos, age) -> STAY/LONG/SHORT/EXIT.
    Returns daily P&L (booked on day t for the move t->t+1, costs on trade day) and trade count."""
    pnl = np.zeros(m.n); pos = 0; e = -1; trades = 0; cum = 0.0
    for t in range(m.n - 1):
        if pos != 0 and t - e >= HOLD:                     # forced exit at expiry of the 30-day option
            pnl[t] -= m.half[t]; pos = 0
        a = decide(t, pos, t - e if pos else 0, cum)
        if pos == 0 and a in (LONG, SHORT):
            pos = 1 if a == LONG else -1; e = t; pnl[t] -= m.half[t]; trades += 1; cum = 0.0
        elif pos != 0 and a == EXIT:
            pnl[t] -= m.half[t]; pos = 0
        if pos != 0:
            g = m.step(t, e, pos); pnl[t] += g; cum += g
    if pos != 0:
        pnl[m.n-1] -= m.half[m.n-1]
    return pnl, trades


def daily_sharpe(x):
    x = np.asarray(x, float); sd = x.std(ddof=1)
    return float(np.sqrt(252) * x.mean() / sd) if sd > 1e-12 else 0.0


# ---------------------------------------------------------------- hybrid Q-learner
def pstate(pos, age, cum=0.0, use_pnl=False):
    """Position slot within a leaf. Base (5): flat, long early/late, short early/late.
    With use_pnl (9): each open slot also splits on whether the open trade is winning or losing."""
    if pos == 0:
        return 0
    base = (0 if pos > 0 else 1) * 2 + (1 if age > EARLY else 0)
    return 1 + (base * 2 + (1 if cum < 0 else 0) if use_pnl else base)


ALLOWED_FLAT = (STAY, LONG, SHORT)
ALLOWED_POS = (STAY, EXIT)


def train_q(m, leaf, nleaf, gamma, seed, alpha=.05, eps=.1, epochs=20, use_pnl=False):
    """Q-learning with epsilon-greedy exploration. Because the agent's trades do not move the market,
    every action available in the visited state gets its exact one-step backup (all-action update);
    the epsilon-greedy choice decides which path the agent follows."""
    rng = np.random.default_rng(seed)
    W = 9 if use_pnl else 5
    Q = np.zeros((nleaf * W, 4))
    n = m.n
    for _ in range(epochs):
        pos, e, cum = 0, -1, 0.0
        for t in range(n - 1):
            age = t - e if pos else 0
            s = leaf[t] * W + pstate(pos, age, cum, use_pnl)
            last = t + 1 == n - 1
            nxt = leaf[t+1] * W
            best_next = {}
            options = ALLOWED_FLAT if pos == 0 else ALLOWED_POS
            for a in options:
                if a == STAY and pos == 0:
                    r, np_, na, nc = 0.0, 0, 0, 0.0
                elif a in (LONG, SHORT):
                    q = 1 if a == LONG else -1
                    g = m.step(t, t, q); r = -m.half[t] + g; np_, na, nc = q, 1, g
                elif a == EXIT:
                    r, np_, na, nc = -m.half[t], 0, 0, 0.0
                else:                                                    # hold
                    g = m.step(t, e, pos); r = g; np_, na, nc = pos, age + 1, cum + g
                if np_ != 0 and na >= HOLD:                              # matures -> closed next day
                    r -= m.half[t+1]; np_, na, nc = 0, 0, 0.0
                if last:
                    target = r
                else:
                    s2 = nxt + pstate(np_, na, nc, use_pnl)
                    v = Q[s2, :3].max() if np_ == 0 else Q[s2, [STAY, EXIT]].max()
                    target = r + gamma * v
                Q[s, a] += alpha * (target - Q[s, a])
                best_next[a] = (np_, na, nc)
            if rng.random() < eps:
                a = options[rng.integers(len(options))]
            else:
                a = options[int(np.argmax(Q[s, list(options)]))]
            np_, na, cum = best_next[a]
            if np_ != 0 and na == 1:
                e = t
            pos = np_
            if pos == 0:
                e = -1
    return Q


def q_decider(Q, leaf, use_pnl=False):
    W = 9 if use_pnl else 5
    def decide(t, pos, age, cum=0.0):
        s = leaf[t] * W + pstate(pos, age, cum, use_pnl)
        opts = ALLOWED_FLAT if pos == 0 else ALLOWED_POS
        return opts[int(np.argmax(Q[s, list(opts)]))]
    return decide


def fit_tree(tr, max_leaves, min_leaf):
    tree = DecisionTreeRegressor(max_leaf_nodes=max_leaves, min_samples_leaf=min_leaf, random_state=2026)
    tree.fit(tr[FEATURES], tr.long_reward)
    ids = np.unique(tree.apply(tr[FEATURES]))
    to_state = lambda d: np.searchsorted(ids, tree.apply(d[FEATURES]))
    return to_state, len(ids)


def run_hybrid(tr, va, te, seeds=(0, 1, 2, 3, 4), use_pnl=False):
    trn = tr.loc[tr.index < va.index.min() - pd.Timedelta(days=45)]   # purge: training path ends before validation
    best = None; grid = []
    for max_leaves in (4, 8):
        for min_leaf in (126, 252):
            to_state, nleaf = fit_tree(tr, max_leaves, min_leaf)
            for gamma in (.9, .98):
                Qs = [train_q(Market(trn), to_state(trn), nleaf, gamma, s, use_pnl=use_pnl) for s in seeds]
                Q = np.mean(Qs, axis=0)
                vp, _ = simulate(Market(va), q_decider(Q, to_state(va), use_pnl))
                sc = daily_sharpe(vp)
                grid.append({'leaves': max_leaves, 'min_leaf': min_leaf, 'gamma': gamma, 'nleaf': nleaf, 'val_sharpe': sc})
                if best is None or sc > best[0]:
                    best = (sc, Q, to_state, grid[-1])
    sc, Q, to_state, cfg = best
    return q_decider(Q, to_state(te), use_pnl), cfg, grid


# ---------------------------------------------------------------- paper replication
def paper_codes(spread, k):
    d = np.diff(spread, prepend=spread[0])
    c = np.where(d > k, 3, np.where(d >= 0, 2, np.where(d > -k, 1, 0)))   # +2,+1,-1,-2 -> 3,2,1,0
    s = np.zeros(len(c), int)
    for lag in range(4):
        s = s * 4 + np.roll(c, lag)
    s[:4] = 0
    return s                                       # 0..255


def run_paper(tr, te, seed=0, episodes=10, alpha=.1, gamma=.99, eps=.1):
    x = tr.spread.to_numpy(float)
    theta, k = x.mean(), np.median(np.abs(np.diff(x)))
    c = tr.cost.mean() / 2
    b, a = np.polyfit(x[:-1], x[1:], 1)                     # AR(1) = discretised OU fit
    sig = np.std(x[1:] - (a + b * x[:-1]))
    rng = np.random.default_rng(seed)
    Q = np.zeros((256 * 3, 3))                              # positions flat/long/short; trades -1/0/+1
    trade_ok = {0: (0, 1, 2), 1: (0, 1), -1: (1, 2)}       # action index: 0 sell, 1 nothing, 2 buy
    for _ in range(episodes):
        path = np.empty(len(x)); path[0] = x[0]
        for t in range(1, len(x)):
            path[t] = a + b * path[t-1] + sig * rng.standard_normal()
        st = paper_codes(path, k); pos = 0
        for t in range(4, len(x) - 1):
            s = st[t] * 3 + (pos % 3)
            ok = trade_ok[pos]
            act = ok[rng.integers(len(ok))] if rng.random() < eps else ok[int(np.argmax(Q[s, list(ok)]))]
            A = act - 1
            r = A * (theta - path[t]) - c * abs(A)
            pos2 = pos + A
            s2 = st[t+1] * 3 + (pos2 % 3)
            Q[s, act] += alpha * (r + gamma * Q[s2, list(trade_ok[pos2])].max() - Q[s, act])
            pos = pos2
    codes = paper_codes(te.spread.to_numpy(float), k)

    def decide(t, pos, age, cum=0.0):
        s = codes[t] * 3 + (pos % 3)
        ok = trade_ok[pos]
        A = ok[int(np.argmax(Q[s, list(ok)]))] - 1
        if pos == 0:
            return LONG if A > 0 else SHORT if A < 0 else STAY
        return EXIT if A != 0 else STAY
    return decide


# ---------------------------------------------------------------- threshold rule
def run_threshold(tr, te, z_in=1.0):
    mu, sd = tr.spread.mean(), tr.spread.std()
    z = ((te.spread - mu) / sd).to_numpy()

    def decide(t, pos, age, cum=0.0):
        if pos == 0:
            return SHORT if z[t] > z_in else LONG if z[t] < -z_in else STAY
        return EXIT if (pos > 0 and z[t] >= 0) or (pos < 0 and z[t] <= 0) else STAY
    return decide


# ---------------------------------------------------------------- current repo method
def run_current(tr, va, te):
    best = None
    for max_leaves in (4, 8, 16):
        for min_leaf in (126, 252):
            to_state, nleaf = fit_tree(tr, max_leaves, min_leaf)
            st = to_state(tr); y = tr.long_reward.to_numpy(); c = tr.cost.to_numpy()
            fits = [fit_policy(st, y, nleaf, s, cost=c) for s in range(5)]
            pol, _ = max(fits, key=lambda f: f[1])
            fits = [fit_policy(st, y, nleaf, s, cost=c, action_values=LADDER, initial_policy=pol) for s in range(5)]
            pol, _ = max(fits, key=lambda f: f[1])
            vv = va[va.end < te.index.min()]
            sc = payoff_sharpe(net_rewards(pol[to_state(vv)], vv.long_reward.to_numpy(), vv.cost.to_numpy()))
            if best is None or sc > best[0]:
                best = (sc, pol, to_state, {'leaves': max_leaves, 'min_leaf': min_leaf, 'nleaf': nleaf, 'val_sharpe': sc})
    _, pol, to_state, cfg = best
    return pol[to_state(te)], cfg


def cohort_pnl(m, sizes):
    """Daily marked P&L of opening a 1/HOLD-sized 30-day cohort each day (overlapping book, max 1 unit)."""
    pnl = np.zeros(m.n)
    for e in range(m.n - 1):
        q = sizes[e] / HOLD
        if q == 0:
            continue
        pnl[e] -= abs(q) * m.half[e]
        end = min(e + HOLD, m.n - 1)
        for t in range(e, end):
            pnl[t] += m.step(t, e, 1.0) * q
        pnl[end] -= abs(q) * m.half[end]
    return pnl


# ---------------------------------------------------------------- main
def summarize(name, dates, pnl, pos_days, trades, folds_idx):
    s = pd.Series(pnl, index=dates)
    cum = s.cumsum()
    rng = np.random.default_rng(2026); n = len(s); boots = []
    for _ in range(1000):
        st = rng.integers(n, size=n // 63 + 1)
        idx = ((st[:, None] + np.arange(63)) % n).ravel()[:n]
        boots.append(daily_sharpe(s.values[idx]))
    return {'model': name, 'sharpe': daily_sharpe(s), 'sharpe_ci': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            'ann_pnl': float(s.mean() * 252), 'ann_vol': float(s.std() * np.sqrt(252)),
            'max_dd': float((cum - cum.cummax()).min()), 'days_invested': float(pos_days), 'round_trips': int(trades),
            'fold_sharpes': {str(y): daily_sharpe(s[(s.index >= pd.Timestamp(y, 1, 1)) & (s.index < pd.Timestamp(y+2, 1, 1))]) for y in folds_idx}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, required=True)
    ap.add_argument('--features', choices=sorted(FEATURE_SETS), default='spread',
                    help="Tree inputs: 'spread' (original) or 'ratio' (IV ratio replaces IV spread).")
    ap.add_argument('--out', type=Path, help='Defaults to artifacts/q_learning (spread) or artifacts/q_learning_ratio.')
    args = ap.parse_args()
    global FEATURES
    FEATURES = FEATURE_SETS[args.features]
    if args.out is None:
        args.out = Path('artifacts/q_learning' if args.features == 'spread' else f'artifacts/q_learning_{args.features}')
    a, b = args.pair
    name = f'{a}_{b}'
    out = args.out / name; out.mkdir(parents=True, exist_ok=True)
    p = build_pair(a, b)
    rows = {k: [] for k in ('q_hybrid', 'q_hybrid_pnl', 'paper', 'threshold', 'current')}
    configs = []
    for y, tr, va, te in folds(p):
        m = Market(te)
        dec, cfg, grid = run_hybrid(tr, va, te)
        dec2, cfg2, grid2 = run_hybrid(tr, va, te, use_pnl=True)
        configs += [{'fold': y, 'model': 'q_hybrid', **cfg}, {'fold': y, 'model': 'q_hybrid_pnl', **cfg2}]
        pd.concat([pd.DataFrame(grid).assign(model='q_hybrid'), pd.DataFrame(grid2).assign(model='q_hybrid_pnl')]).assign(fold=y).to_csv(out / f'grid_{y}.csv', index=False)
        for key, d in (('q_hybrid', dec), ('q_hybrid_pnl', dec2), ('paper', run_paper(tr, te)), ('threshold', run_threshold(tr, te))):
            pos_log = []
            def logged(t, pos, age, cum, d=d):
                act = d(t, pos, age, cum); pos_log.append(pos != 0 or act in (LONG, SHORT)); return act
            pnl, trades = simulate(m, logged)
            rows[key].append(pd.DataFrame({'pnl': pnl, 'invested': pos_log + [False]}, index=te.index).assign(trades=trades))
        sizes, ccfg = run_current(tr, va, te)
        configs.append({'fold': y, 'model': 'current', **ccfg})
        rows['current'].append(pd.DataFrame({'pnl': cohort_pnl(m, sizes), 'invested': sizes != 0}, index=te.index).assign(trades=int((sizes != 0).sum())))
        print(y, {k: round(daily_sharpe(v[-1].pnl), 2) for k, v in rows.items()}, flush=True)
    fold_years = [f[0] for f in folds(p)]
    summary = []
    for k, parts in rows.items():
        d = pd.concat(parts)
        d[['pnl', 'invested']].to_csv(out / f'daily_{k}.csv')
        trades = sum(int(x.trades.iloc[0]) for x in parts)
        summary.append(summarize(k, d.index, d.pnl.to_numpy(), d.invested.mean(), trades, fold_years))
    pd.DataFrame(configs).to_csv(out / 'selected_configs.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(summary).drop(columns=['fold_sharpes']).to_string())
    print(pd.DataFrame({s['model']: s['fold_sharpes'] for s in summary}).round(2).to_string())


if __name__ == '__main__':
    main()
