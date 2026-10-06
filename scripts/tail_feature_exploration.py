"""Causal option-smile and physical-return shape inputs for tail diagnostics."""
import numpy as np
import pandas as pd
from scripts.run_contract_tail_moments import trailing_moments

BUCKETS = ['atm', '25', '10']
COMBINATIONS = [(cp, bucket) for cp in ['C', 'P'] for bucket in BUCKETS]
FEATURES = ['wing_premium_gap', 'wing_premium_z_gap', 'wing_premium_change21_gap',
    'risk_reversal25_gap', 'risk_reversal25_z_gap', 'butterfly25_gap', 'butterfly25_z_gap',
    'tail_steepness_gap', 'wing_term_slope_gap', 'realized_skew63_gap', 'realized_kurt63_gap',
    'realized_skew252_gap', 'realized_kurt252_gap', 'semivariance_share63_gap',
    'jump_fraction63_gap', 'tail_es63_gap', 'wing_iv_gap', 'trailing_vrp_gap']
CONTROLS = ['wing_iv_gap', 'trailing_vrp_gap', 'beta', 'correlation', 'dte_gap', 'delta_gap', 'execution_cost_vol']


def past_z(series):
    past = series.shift(1).rolling(252, min_periods=126)
    return ((series-past.mean())/past.std().where(lambda x:x>1e-8)).replace([np.inf, -np.inf], np.nan)


def return_shape(stock, calendar):
    s = stock.set_index('date').reindex(calendar)
    adjusted = (s.close*s.cfadj).where(s.close.gt(0)&s.cfadj.gt(0))
    r = np.log(adjusted/adjusted.shift(1)); square = r**2
    out = pd.DataFrame(index=calendar)
    for window in [63, 252]:
        moments = trailing_moments(r, calendar, window)
        for moment in ['skew', 'kurt']:
            out[f'realized_{moment}{window}'] = moments[moment].where(moments.available.eq(1))
    total = square.rolling(63, min_periods=63).sum().where(lambda x:x>0)
    for side, condition in [('down', r<0), ('up', r>0)]:
        out[f'{side}_semivariance_share63'] = square.where(condition, 0.).where(r.notna()).rolling(63, min_periods=63).sum()/total
    old_variance = square.shift(1).rolling(63, min_periods=63).mean()
    out['jump_fraction63'] = square.where(square>9*old_variance, 0.).where(old_variance.notna()&r.notna()).rolling(63, min_periods=63).sum()/total
    windows = np.lib.stride_tricks.sliding_window_view(r.to_numpy(float), 63)
    valid = np.isfinite(windows).all(axis=1)
    where = np.flatnonzero(valid)+62; x = windows[valid]
    sd = x.std(axis=1, ddof=1)
    partition = np.partition(x, [6, 56], axis=1)
    for side, vals in [('down', -partition[:, :7].mean(axis=1)), ('up', partition[:, -7:].mean(axis=1))]:
        out[side+'_tail_es63'] = np.nan
        out.loc[calendar[where], side+'_tail_es63'] = np.divide(vals, sd, out=np.full(len(sd), np.nan), where=sd>1e-12)
    out['rv21'] = np.sqrt(252*square.rolling(21, min_periods=21).mean())
    return out.replace([np.inf, -np.inf], np.nan)


def surface_states(selection, calendar):
    """All selected wings share the known call-ATM expiry; no extrapolation."""
    frames = {}
    for tenor in [14, 30, 45]:
        for cp in ['C', 'P']:
            for bucket in BUCKETS:
                s = selection[selection.tenor.eq(tenor)&selection.cp_flag.eq(cp)&selection.bucket.eq(bucket)].set_index('date').reindex(calendar)
                assert not s.index.has_duplicates
                frames[tenor, cp, bucket] = s
    atm = frames[30, 'C', 'atm']; sigma = atm.bs_iv.where(atm.bs_iv.gt(0))
    def same_iv(cp, bucket):
        q = frames[30, cp, bucket]
        return q.bs_iv.where(q.expiry.eq(atm.expiry)&q.bs_iv.gt(0))
    call25, put25 = same_iv('C', '25'), same_iv('P', '25')
    rr = (call25-put25)/sigma
    bf = ((call25+put25)/2-sigma)/sigma
    result = {}
    for cp, bucket in COMBINATIONS:
        q = frames[30, cp, bucket]
        out = pd.DataFrame(index=calendar)
        out['wing_iv'] = same_iv(cp, bucket)
        out['wing_premium'] = out.wing_iv/sigma-1
        out['wing_premium_z'] = past_z(out.wing_premium)
        out['wing_premium_change21'] = out.wing_premium-out.wing_premium.shift(21)
        out['risk_reversal25'] = rr; out['risk_reversal25_z'] = past_z(rr)
        out['butterfly25'] = bf; out['butterfly25_z'] = past_z(bf)
        out['tail_steepness'] = (same_iv(cp, '10')-same_iv(cp, '25'))/sigma
        front, back = frames[14, cp, bucket], frames[45, cp, bucket]
        span = back.dte-front.dte
        bracket = front.expiry.lt(q.expiry)&back.expiry.gt(q.expiry)&span.gt(0)
        out['wing_term_slope'] = ((back.bs_iv-front.bs_iv)*30/span/sigma).where(bracket)
        out['expiry'] = q.expiry; out['optionid'] = q.optionid
        out['abs_delta'] = q.bs_delta.abs(); out['dte'] = q.dte
        out['quote_width_vol'] = (q.ask-q.bid)/q.bs_vega.where(q.bs_vega>0)
        # This is the modeled entry penalty and commission, not the full quoted width.
        out['execution_cost_vol'] = .1+.03/(100*q.bs_vega.where(q.bs_vega>0))
        result[cp, bucket] = out
    return result


def pair_features(first, second, shape_a, shape_b, panel, cp):
    out = pd.DataFrame(index=panel.index)
    for name in ['wing_premium', 'wing_premium_z', 'wing_premium_change21', 'risk_reversal25',
                 'risk_reversal25_z', 'butterfly25', 'butterfly25_z', 'tail_steepness', 'wing_term_slope', 'wing_iv']:
        out[name+'_gap'] = (first[name]-second[name]).reindex(panel.index)
    for name in ['realized_skew63', 'realized_kurt63', 'realized_skew252', 'realized_kurt252', 'jump_fraction63']:
        out[name+'_gap'] = (shape_a[name]-shape_b[name]).reindex(panel.index)
    side = 'down' if cp=='P' else 'up'
    for name in ['semivariance_share63', 'tail_es63']:
        out[name+'_gap'] = (shape_a[side+'_'+name]-shape_b[side+'_'+name]).reindex(panel.index)
    va = first.wing_iv**2-shape_a.rv21**2
    vb = second.wing_iv**2-shape_b.rv21**2
    out['trailing_vrp_gap'] = (va-vb).reindex(panel.index)
    out['beta'] = panel.beta; out['correlation'] = panel.correlation
    out['dte_gap'] = abs(first.dte-second.dte).reindex(panel.index)
    out['delta_gap'] = abs(first.abs_delta-second.abs_delta).reindex(panel.index)
    out['delta_error_max'] = np.nan  # Assigned using the selected bucket in the runner.
    out['execution_cost_vol'] = (first.execution_cost_vol+second.execution_cost_vol).reindex(panel.index)
    out['quote_width_vol_max'] = pd.concat([first.quote_width_vol, second.quote_width_vol], axis=1).max(axis=1, skipna=False).reindex(panel.index)
    return out


def scheduled_nonoverlap(dates, ends, known):
    chosen = np.zeros(len(dates), bool); last = pd.Timestamp.min
    for i, (date, end, observed) in enumerate(zip(dates, ends, known)):
        if observed and pd.notna(end) and date>last and end>date:
            chosen[i] = True; last = end
    return chosen


def ranked_statistics(frame, feature, controls=CONTROLS, min_observations=20):
    """Equal-pair centered ranks; partial is descriptive, not a trading forecast."""
    parts = []; pair_rows = []
    for pair, g in frame.groupby('pair'):
        g = g.dropna(subset=[feature, 'directional_edge'])
        if len(g)<min_observations or g[feature].nunique()<3 or g.directional_edge.nunique()<3: continue
        x = g[feature].rank(pct=True); y = g.directional_edge.rank(pct=True)
        h = pd.DataFrame(dict(x=x-x.mean(), y=y-y.mean(), w=1/len(g), date=g.date))
        denom = np.sqrt((h.x**2).sum()*(h.y**2).sum())
        pair_rows.append(dict(pair=pair, label=g.label.iloc[0], observations=len(g), spearman=float((h.x*h.y).sum()/denom)))
        parts.append(h)
    if not parts: return dict(rho=np.nan, partial=np.nan, pairs=0, ci_low=np.nan, ci_high=np.nan), pair_rows
    h = pd.concat(parts)
    def ratio(a,b,c): return a/np.sqrt(b*c) if b*c>1e-16 else np.nan
    rho = ratio((h.w*h.x*h.y).sum(), (h.w*h.x*h.x).sum(), (h.w*h.y*h.y).sum())
    # Joint quarter blocks preserve contemporaneous cross-pair/shared-name exposure.
    h['xy'] = h.w*h.x*h.y; h['xx'] = h.w*h.x*h.x; h['yy'] = h.w*h.y*h.y
    blocks = h.groupby(h.date.dt.to_period('Q'))[['xy','xx','yy']].sum().to_numpy()
    rng = np.random.default_rng(2026); draw = blocks[rng.integers(len(blocks), size=(500, len(blocks)))].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'): boot = draw[:,0]/np.sqrt(draw[:,1]*draw[:,2])
    lower, upper = np.nanquantile(boot, [.05, .95])
    partial_parts = []
    own_controls = [c for c in controls if c!=feature]
    for pair, g in frame.dropna(subset=[feature, 'directional_edge']+own_controls).groupby('pair'):
        if len(g)<40 or g[feature].nunique()<3: continue
        r = g[[feature, 'directional_edge']+own_controls].rank(pct=True)
        r -= r.mean(); x = r[feature].to_numpy(); y = r.directional_edge.to_numpy(); z = r[own_controls].to_numpy()
        x -= z@np.linalg.lstsq(z, x, rcond=1e-10)[0]; y -= z@np.linalg.lstsq(z, y, rcond=1e-10)[0]
        partial_parts.append(pd.DataFrame(dict(x=x, y=y, w=1/len(g))))
    partial = np.nan
    if partial_parts:
        r = pd.concat(partial_parts)
        partial = ratio((r.w*r.x*r.y).sum(), (r.w*r.x*r.x).sum(), (r.w*r.y*r.y).sum())
    return dict(rho=float(rho), partial=float(partial), pairs=len(parts), ci_low=float(lower), ci_high=float(upper)), pair_rows
