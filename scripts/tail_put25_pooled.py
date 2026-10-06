"""Shared tail allocation policy; optional features never veto an observation."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.tail_put25_policy import CODES
from scripts.tail_put25_rule_sizing import SIZE_CODES

CORE = ['iv_mean', 'oriented_iv_gap', 'rank_strength']
OPTIONAL = ['oriented_vrp_gap', 'rv_mean', 'beta', 'correlation', 'dte_mean',
            'oriented_dte_gap', 'commission_vol_points', 'vix',
            'profit_91d', 'count_91d', 'profit_365d', 'count_365d']
FEATURES = CORE + OPTIONAL + [c + '_available' for c in OPTIONAL]


def state(selection_a, selection_b, returns, screen, calendar, memory, vix, fee):
    a = selection_a.set_index('date').reindex(calendar)
    b = selection_b.set_index('date').reindex(calendar)
    ia, ib = a.bs_iv.where(a.bs_iv.gt(0)), b.bs_iv.where(b.bs_iv.gt(0))
    gap = ia - ib
    rank = gap.rolling(252, min_periods=63).rank(pct=True)
    direction = pd.Series(np.where(rank.le(.2), 1, np.where(rank.gt(.8), -1, 0)), index=calendar)
    mean = (ia + ib) / 2
    out = pd.DataFrame(index=calendar)
    out['observed'] = np.isfinite(ia) & np.isfinite(ib)
    out['eligible'] = screen.eligible.fillna(False).astype(bool)
    out['direction'] = direction
    out['rank'] = rank
    out['iv_mean'] = mean
    # Orient all directional inputs toward the proposed long minus short.
    out['oriented_iv_gap'] = direction * gap / mean
    out['rank_strength'] = (rank - .5).abs() * 2
    out.loc[rank.isna(), 'rank_strength'] = 0.
    rv2 = returns.reindex(calendar).pow(2).rolling(20, min_periods=10).mean() * 252
    out['oriented_vrp_gap'] = direction * ((ia**2 - rv2.iloc[:, 0]) - (ib**2 - rv2.iloc[:, 1])) / mean**2
    out['rv_mean'] = np.sqrt(rv2).mean(axis=1, skipna=False)
    out['beta'] = screen.beta
    out['correlation'] = screen.correlation
    da = (pd.to_datetime(a.expiry) - calendar.to_series()).dt.days
    db = (pd.to_datetime(b.expiry) - calendar.to_series()).dt.days
    out['dte_mean'] = (da + db) / 2
    out['oriented_dte_gap'] = direction * (da - db)
    out['commission_vol_points'] = fee / (100 * a.bs_vega) + fee / (100 * b.bs_vega)
    out['vix'] = vix.reindex(calendar) / 100
    for days in [91, 365]:
        for prefix in ['profit', 'count']:
            col = f'{prefix}_{days}d'
            out[col] = np.where(direction.gt(0), memory[f'{prefix}_long_{days}d'], memory[f'{prefix}_short_{days}d'])
            if prefix == 'profit':
                out[col] /= 100  # $ per one vol-point of vega; fixed scale.
                out[col] = out[col].where(out[f'count_{days}d'].gt(0)) if f'count_{days}d' in out else out[col]
        out[f'profit_{days}d'] = out[f'profit_{days}d'].where(out[f'count_{days}d'].gt(0))
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in OPTIONAL:
        out[col + '_available'] = np.isfinite(out[col]).astype(float)
        out[col] = out[col].fillna(0.)
    assert np.isfinite(out.loc[out.observed, FEATURES]).all().all()
    return out


def rule_codes(panel):
    """Core spread signal and previously requested beta screen; no memory veto."""
    good = panel.observed & panel.eligible
    return pd.Series(np.where(good & panel.direction.gt(0), CODES[-1],
                    np.where(good & panel.direction.lt(0), CODES[1], 0)),
                    index=panel.index, name='code', dtype=int)


def completed_mask(dates, ends, cutoff, years=5):
    dates, ends, cutoff = pd.DatetimeIndex(dates), pd.DatetimeIndex(ends), pd.Timestamp(cutoff)
    return (dates >= cutoff - pd.DateOffset(years=years)) & (dates < cutoff) & (ends < cutoff)


def optimize_book(book, leaves, tree):
    """Reference cross-entropy direct-Sharpe search on five allocation actions."""
    ns, na = len(leaves), 5
    mean = book.mean(axis=0)
    cov = np.cov(book, rowvar=False, ddof=1)
    rng = np.random.default_rng(2026)
    prob = np.full((ns, na), 1 / na)
    best, best_score = np.zeros(ns, int), 0.
    population = 192
    for _ in range(50):
        u = rng.random((population, ns))
        choice = (u[:, :, None] > prob.cumsum(axis=1)[None, :, :]).sum(axis=2).clip(max=na - 1)
        choice[0] = best
        choice[1:na + 1] = np.arange(na)[:, None]
        cols = choice + np.arange(ns)[None, :] * na
        m = mean[cols].sum(axis=1)
        v = cov[cols[:, :, None], cols[:, None, :]].sum(axis=(1, 2))
        score = np.sqrt(252) * np.divide(m, np.sqrt(np.maximum(v, 0)), out=np.zeros_like(m), where=v > 1e-18)
        winner = score.argmax()
        if score[winner] > best_score:
            best, best_score = choice[winner].copy(), float(score[winner])
        elite = choice[np.argsort(score)[-19:]]
        frequency = np.stack([(elite == a).mean(axis=0) for a in range(na)], axis=1)
        prob = .97 * (.3 * prob + .7 * frequency) + .03 / na
    return (tree, leaves, np.array(SIZE_CODES)[best]), dict(training_shadow_daily_sharpe=best_score)


def aggregate_chunk(book, states, labels, positions, first):
    """Add a pair's paths to the shared daily book before computing covariance."""
    days = positions[:, None] + 1 + np.arange(labels.shape[2])[None, :] - first
    valid = (days >= 0) & (days < len(book))
    for leaf in np.unique(states):
        rows = np.flatnonzero(states == leaf)
        for action in range(5):
            np.add.at(book[:, int(leaf) * 5 + action], days[rows][valid[rows]], labels[rows, action][valid[rows]])


def partition(x, target):
    tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=63, random_state=2026).fit(x, target)
    return tree, np.unique(tree.apply(x))
