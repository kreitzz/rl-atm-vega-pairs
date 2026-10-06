"""RL allocation overlay on the frozen put25 rule; never choose direction."""
import numpy as np
import pandas as pd

from scripts.tail_put25_atm_features import states as atm_states, columns as atm_columns
from scripts.tail_put25_policy import CODES, simple_rule
from scripts.curve_trading.policy import ATM, fit, predict

# The reference kernel needs full negative/positive alias slots for its target.
# Its negative alias is the zero-P&L action here; positive aliases are sizes of
# the already signed rule. This leaves exactly five distinct allocations.
SHORT_ALIAS = 1 + ATM * 8
SIZE_CODES = [SHORT_ALIAS, *range(SHORT_ALIAS + 4, SHORT_ALIAS + 8)]


def columns(panel, employee=False, rank=False):
    return atm_columns(panel, employee=employee) + ['wing_iv_gap_pct', 'spread_reversion_available']


def states(a, b, panel, calendar, memory, fee):
    result = atm_states(a, b, panel, calendar, memory, fee)
    result['spread_reversion_available'] = np.isfinite(result.spread_reversion).astype(float)
    result['spread_reversion'] = result.spread_reversion.where(result.spread_reversion_available.eq(1), 0.)
    # An unavailable optional reversion input cannot veto a known rule signal.
    result['feature_available'] = np.isfinite(result[['wing_iv_gap', 'wing_iv_gap_pct']]).all(axis=1)
    assert np.isfinite(result.loc[result.feature_available, columns(result)]).all().all()
    return result


def sizing_labels(daily, rule):
    """Integer-contract outcomes for 0/25/50/75/100% of the fixed direction."""
    daily = np.asarray(daily)
    rule = np.asarray(rule)
    assert np.isin(rule, [0, CODES[1], CODES[-1]]).all()
    result = np.zeros((len(rule), 5, daily.shape[-1]), dtype=daily.dtype)
    for level in range(1, 5):
        short, long = rule == CODES[1], rule == CODES[-1]
        result[short, level] = daily[short, 5-level]
        result[long, level] = daily[long, 4+level]
    return result


def fit_sizer(x, labels, positions, ends):
    return fit(x, labels, positions, ends, SIZE_CODES)


def translate(size_codes, rule):
    size_codes, rule = np.asarray(size_codes), np.asarray(rule)
    assert np.isin(size_codes, SIZE_CODES).all()
    assert np.isin(rule, [0, CODES[1], CODES[-1]]).all()
    result = np.zeros(len(rule), dtype=int)
    for level, alias in enumerate(SIZE_CODES[1:], 1):
        result[(size_codes == alias) & (rule == CODES[1])] = CODES[5-level]
        result[(size_codes == alias) & (rule == CODES[-1])] = CODES[4+level]
    return result


def actions(model, dates, state, gates, available, feature_columns):
    dates = pd.DatetimeIndex(dates)
    rule = simple_rule(dates, state, gates, available)
    out = rule.copy()
    good = state.loc[dates, 'feature_available']
    if good.any():
        out.loc[good] = translate(predict(model, state.loc[dates[good], feature_columns]), rule.loc[good])
    return out
