"""Signal-close, outcome-independent inputs for the three fixed RL ablations."""
import numpy as np
import pandas as pd

GROUPS = {
    'rl_downside': ['downside_fraction_gap', 'downside_fraction_available'],
    'rl_iv_factor': ['iv_factor_reversion_gap', 'iv_factor_reversion_available'],
}
GROUPS['rl_both'] = GROUPS['rl_downside'] + GROUPS['rl_iv_factor']
ADDED = GROUPS['rl_both']


def downside_share(prices, calendar):
    s = prices.set_index('date').reindex(calendar)
    close = (s.close * s.cfadj).where(s.close.gt(0) & s.cfadj.gt(0))
    ret = np.log(close / close.shift(1))
    square = ret ** 2
    total = square.rolling(21, min_periods=21).sum().where(lambda x: x > 0)
    return square.where(ret < 0, 0.).where(ret.notna()).rolling(21, min_periods=21).sum() / total


def iv_factor_scores(iv):
    """Replicate the mined signal using all observable IV names, no labels."""
    iv = iv.where(iv > 0)
    changes = iv.diff()
    result = pd.DataFrame(index=iv.index, columns=iv.columns, dtype=float)
    for sid in iv:
        change = changes[sid]
        factor = changes.drop(columns=sid).median(axis=1)
        old = factor.shift(1).rolling(252, min_periods=126)
        variance = old.var().where(lambda x: x > 1e-10)
        beta = change.shift(1).rolling(252, min_periods=126).cov(factor.shift(1)) / variance
        residual = change - beta * factor
        scale = change.rolling(252, min_periods=126).std().where(lambda x: x > 0)
        result[sid] = -residual.rolling(21, min_periods=21).sum() / scale
    return result.replace([np.inf, -np.inf], np.nan)


def collect(definitions, source, stocks, calendar):
    marks = []
    for d in definitions:
        path = source / d['pair'] / 'states.parquet'
        if not path.exists():
            continue
        p = pd.read_parquet(path, columns=['iv_amd', 'iv_nvda'])
        for sid, suffix in zip(d['secids'], ['amd', 'nvda']):
            marks.append(pd.DataFrame(dict(date=p.index, secid=sid, iv=p['iv_' + suffix].to_numpy() / 100)))
    marks = pd.concat(marks, ignore_index=True)
    spread = marks.groupby(['date', 'secid']).iv.agg(['min', 'max'])
    assert (spread['max'] - spread['min']).dropna().abs().le(1e-10).all()
    iv = marks.drop_duplicates(['date', 'secid']).pivot(index='date', columns='secid', values='iv').reindex(calendar)
    factor = iv_factor_scores(iv)
    legs = {}
    for sid in iv:
        legs[int(sid)] = pd.DataFrame({
            'downside_fraction': downside_share(stocks[stocks.secid.eq(sid)], calendar),
            'iv_factor_reversion': factor[sid],
        }, index=calendar)
    return legs


def add_signals(panel, first, second):
    result = panel.copy()
    for signal in ['downside_fraction', 'iv_factor_reversion']:
        a = first[signal].reindex(panel.index)
        b = second[signal].reindex(panel.index)
        gap = a - b
        available = np.isfinite(a) & np.isfinite(b) & np.isfinite(gap)
        result[signal + '_gap'] = gap.where(available, 0.).astype(float)
        result[signal + '_available'] = available.astype(float)
    pd.testing.assert_frame_equal(result[panel.columns], panel)
    return result
