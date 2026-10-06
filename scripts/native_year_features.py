"""Past-only forecasts and independent one-year relative-value signals."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORECAST_FEATURES = ['log_rv252', 'rv20_ratio', 'rv60_ratio', 'rv126_ratio', 'vol_of_vol']
ENTRY_FEATURES = ['edge_forecast', 'edge_slow', 'edge_z', 'z_available', 'iv_gap_relative',
                  'iv_mean', 'forecast_ratio_a', 'forecast_ratio_b', 'beta', 'correlation',
                  'momentum_gap', 'dte_gap', 'commission_vol_points', 'vix', 'native_iv_gap_z', 'native_iv_history_available']

# OCC published the corrected effective date on July 26. The stock vendor
# applied and withdrew a premature reverse split on July 26/27. Correct only
# observations at/after this public memo, never alter earlier observations.
VERIFIED_FACTOR_EVENTS = [dict(secid=111450, announced='2022-07-26', effective='2022-07-28',
    ratio=.1, reason='McEwen reverse split: premature vendor application/withdrawal',
    source='https://infomemo.theocc.com/infomemos?number=50780')]


def reconcile_adjustments(stocks):
    """Ignore factor resets contradicted by contemporaneous economic returns.

    A real split changes price return relative to the vendor's adjusted return.
    An adjustment-factor reset with identical price/total return supplies no
    evidence for changing a held share count. Use only that day's information.
    """
    out = stocks.sort_values(['secid', 'date']).copy()
    out['source_cfadj'] = out.cfadj
    notes = []
    for sid, idx in out.groupby('secid', sort=False).groups.items():
        g = out.loc[idx]
        ratio = g.source_cfadj/g.source_cfadj.shift(1)
        price_return = g.close/g.close.shift(1)-1
        changed = ratio.notna() & ~np.isclose(ratio, 1., rtol=1e-10, atol=1e-12)
        reset = changed & np.isfinite(g['return']) & np.isclose(price_return, g['return'], rtol=0, atol=1e-6)
        physical = ratio.where(changed & ~reset, 1.).fillna(1.)
        for event in VERIFIED_FACTOR_EVENTS:
            if int(sid) != event['secid']:
                continue
            announced, effective = pd.Timestamp(event['announced']), pd.Timestamp(event['effective'])
            affected = g.date.between(announced, effective)
            for index in g.index[affected]:
                desired = event['ratio'] if g.loc[index, 'date'] == effective else 1.
                if not np.isclose(physical.loc[index], desired):
                    notes.append(dict(date=str(g.loc[index, 'date'].date()), secid=int(sid),
                        source_ratio=float(ratio.loc[index]), source_daily_return=float(g.loc[index, 'return']),
                        raw_price_return=float(price_return.loc[index]),
                        interpretation=event['reason']+'; public memo already available',
                        source=event['source'], announced=event['announced'], effective=event['effective']))
                physical.loc[index] = desired
        out.loc[idx, 'cfadj'] = float(g.source_cfadj.iloc[0]) * physical.cumprod()
        for index in g.index[reset]:
            notes.append(dict(date=str(g.loc[index, 'date'].date()), secid=int(sid),
                source_ratio=float(ratio.loc[index]), source_daily_return=float(g.loc[index, 'return']),
                raw_price_return=float(price_return.loc[index]),
                interpretation='Factor reset without adjusted-return evidence of a share-count change; no future data used'))
    return out, notes


def variance_inputs(close, calendar):
    close = close.reindex(calendar).where(close.reindex(calendar).gt(0))
    r = np.log(close / close.shift(1))
    squared = r.pow(2)
    out = pd.DataFrame(index=calendar)
    for half_life in [20, 60, 126, 252]:
        out[f'rv{half_life}'] = 252 * squared.ewm(halflife=half_life, min_periods=126, adjust=False).mean()
    out['log_rv252'] = np.log(out.rv252)
    for span in [20, 60, 126]:
        out[f'rv{span}_ratio'] = np.log(out[f'rv{span}'] / out.rv252)
    out['vol_of_vol'] = squared.rolling(126, min_periods=126).std() / squared.rolling(126, min_periods=126).mean()
    out['momentum'] = close.shift(21) / close.shift(252) - 1
    # At t: mean of squared returns at t+1,...,t+252, not t itself.
    out['future_variance'] = 252 * squared.rolling(252, min_periods=252).mean().shift(-252)
    out['label_end'] = pd.Series(calendar, index=calendar).shift(-252)
    return out.replace([np.inf, -np.inf], np.nan), r


def completed_forecasts(rows, cutoff, years=8):
    return rows.loc[rows.date.lt(cutoff) & rows.date.ge(cutoff - pd.DateOffset(years=years)) &
                    rows.label_end.lt(cutoff) & rows.future_variance.gt(0) & rows.rv252.gt(0) &
                    np.isfinite(rows[FORECAST_FEATURES]).all(axis=1)].copy()


def forecast_rows(rows):
    out = rows.copy()
    out['forecast_variance'] = out.rv252
    out['forecast_fitted'] = False
    notes = []
    for period, idx in out.groupby(out.date.dt.to_period('Q')).groups.items():
        cutoff = period.start_time
        train = completed_forecasts(rows, cutoff)
        known = np.isfinite(out.loc[idx, FORECAST_FEATURES]).all(axis=1)
        take = np.asarray(idx)[known.to_numpy()]
        note = dict(fit=str(cutoff.date()), training_rows=len(train),
                    last_label_end=str(train.label_end.max().date()) if len(train) else None,
                    fitted=False, fallback='EWMA252 variance, no entry veto')
        if len(train) >= 100 and len(take):
            x = train[FORECAST_FEATURES].to_numpy(float)
            target = np.log(train.future_variance / train.rv252).to_numpy(float)
            lo, hi = np.quantile(target, [.01, .99])
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
            model.fit(x, np.clip(target, lo, hi))
            residual = target - model.predict(x)
            correction = float(np.clip(np.var(residual, ddof=1) / 2, 0, .5))
            prediction = np.clip(model.predict(out.loc[take, FORECAST_FEATURES].to_numpy(float)) + correction, lo, hi)
            out.loc[take, 'forecast_variance'] = out.loc[take, 'rv252'].to_numpy() * np.exp(prediction)
            out.loc[take, 'forecast_fitted'] = True
            note.update(fitted=True, residual_log_variance_correction=correction,
                        coefficients=model[-1].coef_.tolist(), intercept=float(model[-1].intercept_))
            assert train.label_end.max() < cutoff
        notes.append(note)
    return out, notes


def name_premiums(rows):
    out = rows.sort_values(['secid', 'date']).copy()
    out['carry_forecast'] = (out.forecast_variance - out.bs_iv.pow(2)) / (2 * out.bs_iv)
    out['carry_slow'] = (out.rv252 - out.bs_iv.pow(2)) / (2 * out.bs_iv)
    out['carry_z'] = np.nan
    for sid, idx in out.groupby('secid', sort=False).groups.items():
        known = np.asarray(idx)[np.isfinite(out.loc[idx, 'carry_forecast']).to_numpy()]
        carry = out.loc[known, 'carry_forecast']
        history = carry.shift(1).rolling(24, min_periods=12)
        out.loc[known, 'carry_z'] = (carry - history.mean()) / history.std().where(history.std().gt(1e-8))
    return out


def native_direction(edge):
    values = np.asarray(edge, float)
    return np.where(np.isfinite(values) & (values > 0), 1,
                    np.where(np.isfinite(values) & (values < 0), -1, 0)).astype(int)


def native_iv_history(rows):
    out = rows.copy()
    out['native_iv_gap_z'] = np.nan
    for pair, idx in out.groupby('pair', sort=False).groups.items():
        g = out.loc[idx].sort_values('date')
        known = g.loc[g.optionid_a.notna() & g.optionid_b.notna() & g.iv_mean.gt(0)]
        v = known.iv_gap_relative
        history = v.shift(1).rolling(24, min_periods=6)
        out.loc[known.index, 'native_iv_gap_z'] = (v-history.mean())/history.std().where(history.std().gt(1e-8))
    out['native_iv_history_available'] = out.native_iv_gap_z.notna().astype(float)
    edge = out.native_iv_gap_z.where(out.native_iv_history_available.eq(1), out.iv_gap_relative)
    out['iv_history_direction'] = native_direction(-edge)
    out['native_iv_gap_z'] = out.native_iv_gap_z.fillna(0.)
    return out
