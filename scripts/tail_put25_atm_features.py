"""Transfer the frozen ATM state definitions to a single 25-delta put silo."""
import numpy as np
import pandas as pd

from scripts.run_atm65_ablation import features as atm_features
from scripts.run_atm65_employee_sentiment import ADDED as EMPLOYEE_ADDED
from scripts.tail_put25_policy import states as spread_states, permitted, fit_tail
from scripts.curve_trading.policy import predict


def columns(panel, employee=False, rank=False):
    result = atm_features(panel) + (EMPLOYEE_ADDED if employee else [])
    assert len(result) == (49 if employee else 47)
    return result + (['wing_iv_gap_pct'] if rank else [])


def states(selection_a, selection_b, panel, calendar, memory, fee):
    """Signal-close values only; no future quote/status filtering or filling.

    Common stock, earnings, VIX, surface and review context is inherited.
    Contract IV, reversion, costs, DTE, moneyness and completed-trade memory
    are recomputed for the puts. Original ATM outcome memory is overwritten.
    """
    result = panel.copy()
    tail = spread_states(selection_a, selection_b, panel, calendar)
    for suffix, side, sel in [('amd', 'a', selection_a), ('nvda', 'b', selection_b)]:
        q = sel.set_index('date').reindex(panel.index)
        result[f'iv_{suffix}'] = q.bs_iv.where(q.bs_iv.gt(0)) * 100
        result[f'dte_{side}'] = (q.expiry - q.index.to_series()).dt.days
        result[f'moneyness_{side}'] = q.strike / q.close - 1
        result[f'commission_{side}'] = fee / (100 * q.bs_vega.where(q.bs_vega.gt(0)))
    result['spread'] = result.iv_amd - result.iv_nvda
    result['rv_adjusted_value'] = ((result.trailing_rv_amd - result.iv_amd)
                                   - (result.trailing_rv_nvda - result.iv_nvda))
    # ATM add_features() runs on the trading calendar before incomplete state
    # rows are dropped. Preserve that ordering and the 252/126 warmup.
    spread = (selection_a.set_index('date').bs_iv.reindex(calendar)
              - selection_b.set_index('date').bs_iv.reindex(calendar)) * 100
    past = spread.shift(1).rolling(252, min_periods=126)
    result['spread_reversion'] = ((past.mean() - spread) / past.std().replace(0, np.nan)).reindex(panel.index)
    result['commission_vol_entry'] = result.commission_a + result.commission_b
    expected = [c for c in panel if c.startswith(('profit_', 'count_'))]
    assert set(expected) == set(memory.columns)
    result[memory.columns] = memory.reindex(panel.index)
    result[['wing_iv_gap', 'wing_iv_gap_pct']] = tail[['wing_iv_gap', 'wing_iv_gap_pct']]
    result['feature_available'] = np.isfinite(result[columns(result, rank=True)]).all(axis=1)
    # Eligibility is the original causal return-beta/correlation screen.
    np.testing.assert_array_equal(result.eligible, panel.eligible)
    return result


def actions(model, dates, state, gates, available, feature_columns):
    dates = pd.DatetimeIndex(dates)
    out = pd.Series(0, index=dates, name='code', dtype=int)
    good = state.loc[dates, 'feature_available']
    if good.any():
        out.loc[good] = predict(model, state.loc[dates[good], feature_columns])
    return permitted(out, dates, state, gates, available)
