"""One fixed put-tail silo, using the unchanged reference Sharpe optimizer."""
import numpy as np
import pandas as pd

from scripts.curve_trading.config import COMBOS
from scripts.curve_trading.policy import ATM, fit, predict

COMBO = (30, 'P', 10, 10)
COMBO_INDEX = COMBOS.index(COMBO)
CODES = [0] + list(range(1 + COMBO_INDEX * 8, 1 + (COMBO_INDEX + 1) * 8))
ALIASES = [0] + list(range(1 + ATM * 8, 1 + (ATM + 1) * 8))
FEATURES = ['wing_iv_gap', 'wing_iv_gap_pct']


def states(selection_a, selection_b, panel, calendar):
    def iv(selection):
        s = selection.set_index('date').bs_iv.reindex(calendar)
        return s.where(np.isfinite(s) & s.gt(0))
    spread = iv(selection_a) - iv(selection_b)
    # Match the exploration's observable pair-state calendar, before eligibility
    # or payoff filtering. Do not add history on dates lacking a pair state.
    rank = spread.reindex(panel.index).reindex(calendar).rolling(252, min_periods=63).rank(pct=True)
    result = pd.DataFrame({'wing_iv_gap': spread, 'wing_iv_gap_pct': rank}).reindex(panel.index)
    result['eligible'] = panel.eligible.astype(bool)
    result['feature_available'] = np.isfinite(result[FEATURES]).all(axis=1)
    return result


def fit_tail(x, daily, positions, ends):
    # The reference kernel uses ATM code IDs to locate directional targets.
    # Only IDs are aliased inside fitting; labels and executed contracts are puts.
    model, details = fit(x, daily, positions, ends, ALIASES)
    tree, leaves, policy = model
    lookup = dict(zip(ALIASES, CODES))
    return (tree, leaves, np.array([lookup[int(c)] for c in policy])), details


def permitted(codes, dates, state, gates, available):
    codes = np.asarray(codes, dtype=int).copy()
    if not np.isin(codes, CODES).all():
        raise ValueError('Only flat and the put10 silo actions are allowed')
    rows = state.index.get_indexer(dates)
    assert (rows >= 0).all()
    common = state.eligible.to_numpy(bool)[rows] & state.feature_available.to_numpy(bool)[rows] & available[rows]
    for code in np.unique(codes):
        if not code:
            continue
        side = int(CODES.index(int(code)) >= 5)
        okay = common & gates[rows, side]
        codes[(codes == code) & ~okay] = 0
    return pd.Series(codes, index=dates, name='code', dtype=int)


def actions(model, dates, state, gates, available):
    dates = pd.DatetimeIndex(dates)
    out = pd.Series(0, index=dates, name='code', dtype=int)
    valid = state.loc[dates, 'feature_available']
    if valid.any():
        out.loc[valid] = predict(model, state.loc[dates[valid], FEATURES])
    return permitted(out, dates, state, gates, available)


def simple_rule(dates, state, gates, available):
    rank = state.loc[dates, 'wing_iv_gap_pct']
    out = np.where(rank.le(.2), CODES[-1], np.where(rank.gt(.8), CODES[1], 0))
    return permitted(out, dates, state, gates, available)
