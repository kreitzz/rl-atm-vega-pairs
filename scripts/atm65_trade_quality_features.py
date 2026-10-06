"""Causal name memory, pair IV regime and observed contract gamma geometry."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.option_flex.black_scholes import option_values
from scripts.curve_trading.config import DATA

LEG_SOURCE = Path('artifacts/contracts/atm_variance_mining/legs.parquet')
MEMORY = [f'name_{role}_{stat}_{side}' for side in ['a', 'b']
          for role in ['long', 'short'] for stat in ['mean365', 'uncertainty365']]
MEMORY += [f'name_blocks365_{side}' for side in ['a', 'b']]
MEMORY += [f'name_memory_{side}_available' for side in ['a', 'b']]
GROUPS = {
    'rl_name_memory': MEMORY,
    'rl_iv_corr_regime': ['iv_change_corr_history_z', 'iv_change_corr_history_available'],
    'rl_gamma_geometry': ['gamma_log_ratio_a_b', 'gamma_geometry_available'],
}
ADDED = sum(GROUPS.values(), [])


def nonoverlap_samples(frame):
    """Select with entry and scheduled end only; missing outcomes still occupy time."""
    keys = ['date', 'secid', 'optionid']
    if frame.duplicated(keys).any():
        for col in ['entry', 'end', 'long_net', 'short_net']:
            assert frame.groupby(keys)[col].nunique(dropna=False).le(1).all()
    frame = frame.drop_duplicates(keys).sort_values(['secid', 'date', 'optionid'])
    assert not frame.duplicated(['date', 'secid']).any()
    selected = []
    for sid, g in frame.groupby('secid'):
        previous_end = pd.Timestamp.min
        for row in g.itertuples():
            if pd.isna(row.end) or pd.isna(row.entry) or row.entry <= previous_end or row.end <= row.entry:
                continue
            selected.append(row.Index)
            previous_end = row.end
    return frame.loc[selected].copy()


def name_memory(samples, calendar):
    result = pd.DataFrame(0., index=calendar, columns=[
        'long_mean365', 'short_mean365', 'long_uncertainty365',
        'short_uncertainty365', 'blocks365', 'available'])
    # Strict completion-before-signal, not <=. Missing labels are not zeros.
    good = samples.dropna(subset=['end', 'long_net', 'short_net']).sort_values('end')
    ends = pd.DatetimeIndex(good.end).asi8
    times = pd.DatetimeIndex(calendar).asi8
    right = np.searchsorted(ends, times, side='left')
    left = np.searchsorted(ends, times-pd.Timedelta(days=365).value, side='left')
    count = right-left
    result['blocks365'] = count.astype(float)
    ready = count >= 4
    for role in ['long', 'short']:
        values = good[role+'_net'].to_numpy(float)
        sums = np.r_[0., np.cumsum(values)]
        squares = np.r_[0., np.cumsum(values**2)]
        total = sums[right]-sums[left]
        mean = np.divide(total, count, out=np.zeros(len(count)), where=count>0)
        ss = squares[right]-squares[left]-count*mean**2
        variance = np.divide(np.maximum(ss, 0), count-1, out=np.zeros(len(count)), where=count>1)
        uncertainty = np.sqrt(np.divide(variance, count, out=np.zeros(len(count)), where=count>0))
        result[role+'_mean365'] = np.where(ready, mean, 0.)
        result[role+'_uncertainty365'] = np.where(ready, uncertainty, 0.)
    result['available'] = ready.astype(float)
    return result


def gamma_per_vega(q, spots):
    dte = (pd.to_datetime(q.expiry)-pd.to_datetime(q.date)).dt.days.to_numpy()/365
    spot = np.asarray(spots, float)
    _, _, gamma, _ = option_values(spot-q.known_dividend_pv.to_numpy(), q.strike.to_numpy(),
        dte, q.bs_rate.to_numpy(), q.bs_iv.to_numpy(), q.cp_flag.to_numpy())
    return np.divide(gamma*spot**2, q.vega.to_numpy(), out=np.full(len(q), np.nan), where=q.vega.to_numpy()>0)


def collect(definitions, source, stocks, calendar):
    """Only cached single-name outcomes; no conditioning on pair success/eligibility."""
    raw = pd.read_parquet(LEG_SOURCE, columns=['date', 'secid', 'optionid', 'entry', 'end', 'long_net', 'short_net'])
    samples = nonoverlap_samples(raw)
    legs = {}
    for sid in sorted({sid for d in definitions for sid in d['secids']}):
        memory = name_memory(samples[samples.secid.eq(sid)], calendar)
        pieces = []
        for year in sorted(set(calendar.year)):
            selection = DATA/'selection/C'/str(year)/f'{sid}.parquet'
            quotes = DATA/'history/C'/str(year)/f'{sid}.parquet'
            if not selection.exists() or not quotes.exists(): continue
            s = pd.read_parquet(selection)
            s = s[s.tenor.eq(30)&s.bucket.eq('atm')&s.cp_flag.eq('C')][['date', 'optionid']]
            if s.empty: continue
            q = pd.read_parquet(quotes, filters=[('optionid', 'in', s.optionid.unique().tolist())],
                columns=['date', 'optionid', 'expiry', 'known_dividend_pv', 'strike', 'bs_rate', 'bs_iv', 'cp_flag', 'vega'])
            q = s.merge(q, on=['date', 'optionid'], how='left', validate='one_to_one')
            spot = stocks[stocks.secid.eq(sid)].set_index('date').close.reindex(pd.DatetimeIndex(q.date)).to_numpy()
            q['gamma_per_vega'] = gamma_per_vega(q, spot)
            pieces.append(q[['date', 'gamma_per_vega']])
        if pieces:
            geometry = pd.concat(pieces).set_index('date').gamma_per_vega
            assert not geometry.index.has_duplicates
            memory['gamma_per_vega'] = geometry.reindex(calendar)
        else: memory['gamma_per_vega'] = np.nan
        legs[sid] = memory
    return legs


def add_signals(panel, first, second):
    result = panel.copy()
    for side, leg in [('a', first), ('b', second)]:
        leg = leg.reindex(panel.index)
        for role in ['long', 'short']:
            for stat in ['mean365', 'uncertainty365']:
                result[f'name_{role}_{stat}_{side}'] = leg[f'{role}_{stat}'].fillna(0.)
        result[f'name_blocks365_{side}'] = leg.blocks365.fillna(0.)
        result[f'name_memory_{side}_available'] = leg.available.fillna(0.)
    # Use the identical pair-local estimator examined in the trade diagnostic.
    corr = panel.iv_amd.diff().rolling(126, min_periods=63).corr(panel.iv_nvda.diff())
    history = corr.shift(1).rolling(252, min_periods=126)
    z = (corr-history.mean())/history.std().where(lambda x:x>1e-8)
    result['iv_change_corr_history_z'] = z.replace([np.inf, -np.inf], np.nan).fillna(0.)
    result['iv_change_corr_history_available'] = np.isfinite(z).astype(float)
    ratio = first.gamma_per_vega.reindex(panel.index)/second.gamma_per_vega.reindex(panel.index)
    logratio = np.log(ratio.where(ratio>0)).replace([np.inf, -np.inf], np.nan)
    result['gamma_log_ratio_a_b'] = logratio.fillna(0.)
    result['gamma_geometry_available'] = logratio.notna().astype(float)
    pd.testing.assert_frame_equal(result[panel.columns], panel)
    assert np.isfinite(result[ADDED]).all().all()
    return result
