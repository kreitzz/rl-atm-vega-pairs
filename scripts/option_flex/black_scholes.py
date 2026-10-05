"""European call midpoint inversion; cash-dividend escrow approximation."""
import numpy as np
import pandas as pd
from scipy.special import ndtr


def call_values(spot, strike, time, rate, sigma):
    s, k, t, r, v = np.broadcast_arrays(spot, strike, time, rate, sigma)
    root = np.sqrt(t)
    d1 = (np.log(s/k)+(r+v*v/2)*t)/(v*root)
    d2 = d1-v*root
    density = np.exp(-d1*d1/2)/np.sqrt(2*np.pi)
    price = s*ndtr(d1)-k*np.exp(-r*t)*ndtr(d2)
    return price, ndtr(d1), density/(s*v*root), s*density*root/100


def invert_calls(spot, strike, time, rate, price):
    s, k, t, r, p = np.broadcast_arrays(spot, strike, time, rate, price)
    valid = np.isfinite(s+k+t+r+p) & (s > 0) & (k > 0) & (t > 0)
    lower = np.maximum(s-k*np.exp(-r*t), 0)
    valid &= (p > lower+1e-8) & (p < s-1e-8)
    # Explicit finite bracket: reject rather than clip extreme/unsolvable IV.
    lo, hi = np.full(s.shape, 1e-6), np.full(s.shape, 10.)
    with np.errstate(all='ignore'):
        valid &= (call_values(s,k,t,r,lo)[0] <= p) & (call_values(s,k,t,r,hi)[0] >= p)
        for _ in range(55):
            mid = (lo+hi)/2
            below = call_values(s,k,t,r,mid)[0] < p
            lo = np.where(below,mid,lo)
            hi = np.where(below,hi,mid)
        iv = np.where(valid,(lo+hi)/2,np.nan)
        fitted, delta, gamma, vega = call_values(s,k,t,r,iv)
    return iv, fitted, delta, gamma, vega


def option_values(spot,strike,time,rate,sigma,cp_flag='C'):
    price,delta,gamma,vega=call_values(spot,strike,time,rate,sigma)
    put=np.asarray(cp_flag)=='P'
    price=price-np.where(put,np.asarray(spot)-np.asarray(strike)*np.exp(-np.asarray(rate)*np.asarray(time)),0.)
    return price,delta-put,gamma,vega


def invert_options(spot,strike,time,rate,price,cp_flag):
    put=np.asarray(cp_flag)=='P'
    equivalent=np.asarray(price)+np.where(put,np.asarray(spot)-np.asarray(strike)*np.exp(-np.asarray(rate)*np.asarray(time)),0.)
    iv,_,_,_,_=invert_calls(spot,strike,time,rate,equivalent)
    fitted,delta,gamma,vega=option_values(spot,strike,time,rate,iv,cp_flag)
    return iv,fitted,delta,gamma,vega


def fit_quotes(quotes, stocks, zero, distributions):
    """Normalized quotes in; vendor fields preserved. Rates are % continuous.

    Only declared ordinary USD cash dividends enter PV. This does not forecast
    undeclared dividends or model American exercise. Vega is $/share/vol point.
    """
    q = quotes.copy().reset_index(drop=True)
    spot = stocks[['date','secid','close']].copy()
    spot['date'] = pd.to_datetime(spot.date)
    q = q.merge(spot, on=['date','secid'], how='left', validate='many_to_one')
    q['dte'] = (q.expiry-q.date).dt.days
    q['bs_rate'] = np.nan
    zero = zero.copy(); zero['date'] = pd.to_datetime(zero.date)
    zero = zero[zero.date.between(q.date.min()-pd.Timedelta(days=7), q.date.max())]
    curves = {d: g.sort_values('days') for d,g in zero.groupby('date')}
    dates = pd.DatetimeIndex(sorted(curves))
    for date, idx in q.groupby('date').groups.items():
        loc = dates.searchsorted(date, side='right')-1
        if loc < 0 or (date-dates[loc]).days > 7:
            continue
        curve = curves[dates[loc]]
        q.loc[idx,'bs_rate'] = np.interp(q.loc[idx,'dte'],curve.days,curve.rate)/100
    q['known_dividend_pv'] = 0.
    unsupported = np.zeros(len(q),dtype=bool)
    events = distributions[distributions.secid.isin(q.secid.unique())].copy()
    for c in ['ex_date','declare_date']:
        events[c] = pd.to_datetime(events[c])
    for e in events.itertuples():
        if (str(e.cancel_flag) != '0' or pd.isna(e.declare_date) or
                e.declare_date < pd.Timestamp('1950-01-01')):
            continue
        mask = (q.secid.eq(e.secid) & q.date.ge(e.declare_date) &
                q.date.lt(e.ex_date) & q.expiry.ge(e.ex_date)).to_numpy()
        if not mask.any():
            continue
        ordinary = (str(e.distr_type)=='1' and e.currency=='USD' and e.adj_factor==1
                    and str(e.approx_flag)=='0' and str(e.liquid_flag)=='0'
                    and np.isfinite(e.amount) and e.amount >= 0)
        if not ordinary:
            unsupported |= mask
            continue
        t = (e.ex_date-q.loc[mask,'date']).dt.days/365
        q.loc[mask,'known_dividend_pv'] += e.amount*np.exp(-q.loc[mask,'bs_rate']*t)
    effective = q.close-q.known_dividend_pv
    midpoint = (q.bid+q.ask)/2
    invalid = unsupported | q.bid.lt(0) | q.ask.lt(q.bid) | ~q.standard | q.multiplier.ne(100) | ~q.cp_flag.isin(['C','P'])
    values = invert_options(effective,q.strike,q.dte/365,q.bs_rate,midpoint.mask(invalid),q.cp_flag)
    for name, value in zip(['bs_iv','bs_price','bs_delta','bs_gamma','bs_vega'],values):
        q[name] = value
    q['bs_error'] = q.bs_price-midpoint
    q['bs_status'] = np.where(q.bs_iv.notna(),'ok','invalid_input_or_price_bounds')
    q.loc[unsupported,'bs_status'] = 'unsupported_distribution'
    return q
