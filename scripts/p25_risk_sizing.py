"""Causal directional risk inputs and a declared inventory sizing overlay."""
import numpy as np
import pandas as pd
from scripts.tail_put25_pooled import FEATURES

RV = ['long_rv_iv', 'short_rv_iv']
RELATION = ['iv_times_decorrelation', 'log_vol_mismatch']
ARMS = {'rv':RV, 'relationship':RELATION, 'both':RV+RELATION}


def risk_features(rows, iv_a, iv_b, rv_a, rv_b):
    """All series are aligned to the observed signal close, never the fill."""
    x = rows.copy()
    long_a = x.direction.to_numpy() > 0
    ra = np.asarray(rv_a)/np.asarray(iv_a)
    rb = np.asarray(rv_b)/np.asarray(iv_b)
    x['long_rv_iv'] = np.where(long_a,ra,rb)
    x['short_rv_iv'] = np.where(long_a,rb,ra)
    corr = x.correlation.where(x.correlation_available.eq(1))
    beta = x.beta.where(x.beta_available.eq(1))
    x['iv_times_decorrelation'] = x.iv_mean*(1-corr)
    x['log_vol_mismatch'] = np.abs(np.log((beta/corr).where((beta/corr)>0)))
    for name in RV+RELATION:
        x[name+'_available'] = np.isfinite(x[name]).astype(float)
        x[name] = x[name].where(np.isfinite(x[name]),0.)
    return x


def columns(arm):
    names = ARMS[arm]
    return FEATURES+names+[name+'_available' for name in names]


def inventory_context(row, names, prior_nav, cfg):
    budget = cfg.entry_vega_per_leg*max(0,prior_nav)/cfg.capital
    a,b = [float(names.get(int(row['secid_'+side]),0.)) for side in ['a','b']]
    return dict(name_a_prior_gross_vega=a,name_b_prior_gross_vega=b,
        name_inventory_units=max(a,b)/budget if budget>0 else 0.)


def inventory_level(level, context):
    """Nearest declared floor-ladder level after a gross-exposure adjustment.

    This is a fixed causal overlay, not a learned sequential inventory policy.
    Zero prior name exposure gives identity. No signal is changed to flat.
    """
    if level not in [2,3,4]:
        raise ValueError('Expected50/75/100 floor ladder')
    units = float(context['name_inventory_units'])
    if not np.isfinite(units) or units<0:
        raise ValueError('Invalid prior-close inventory')
    target = max(2.,level/np.sqrt(1+units))
    return int(np.array([2,3,4])[np.argmin(abs(np.array([2,3,4])-target))])
