"""Causal rolling AMD-on-NVDA daily return beta as a state input."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

if __package__:
    from .run_first_pass import load_return_history
else:
    from run_first_pass import load_return_history


def rolling_beta(returns, window):
    """OLS slope with intercept, using the latest complete window through entry."""
    returns=returns.sort_index().astype(float)
    good=returns[['AMD','NVDA']].notna().all(axis=1)
    y=returns.AMD.where(good)
    x=returns.NVDA.where(good)
    covariance=y.rolling(window,min_periods=window).cov(x,ddof=1)
    variance=x.rolling(window,min_periods=window).var(ddof=1)
    return covariance/variance.where(variance>1e-12)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--panel',type=Path,default=Path('artifacts/short_rv_momentum/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('artifacts/rolling_beta'))
    p.add_argument('--features-output',type=Path,default=Path('config/rolling_beta_features.json'))
    args=p.parse_args()
    panel=pd.read_parquet(args.panel).sort_index()
    returns=load_return_history(args.source_root)
    for window in [63,126]:
        panel=panel.join(rolling_beta(returns,window).rename(f'amd_on_nvda_beta_{window}'))
    base=['spread','amd_earnings_days_capped','nvda_earnings_days_capped','trailing_rv_amd','trailing_rv_nvda']
    features={'rv30cal':base,'beta63':base+['amd_on_nvda_beta_63'],
              'beta126':base+['amd_on_nvda_beta_126'],
              'beta63_126':base+['amd_on_nvda_beta_63','amd_on_nvda_beta_126']}
    required=sorted({f for fs in features.values() for f in fs})
    before=len(panel);panel=panel.dropna(subset=required)
    args.output.mkdir(parents=True,exist_ok=True)
    args.features_output.parent.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(args.output/'dataset.parquet')
    args.features_output.write_text(json.dumps(features,indent=2)+'\n')
    summary={'rows':len(panel),'extra_dropped':before-len(panel),'first':str(panel.index.min().date()),'last':str(panel.index.max().date()),
             'definition':'Cov(AMD retx, NVDA retx) / Var(NVDA retx); OLS slope with intercept',
             'windows':'63 and 126 consecutive trading-day returns through entry close',
             'direction':'AMD dependent variable, NVDA explanatory variable',
             'returns':'CRSP split-adjusted ex-distribution simple C2C returns',
             'missing':'Require every return in the window; variance <=1e-12 yields missing',
             'trading':'Beta is a state input only; initial vega remains balanced and both legs are assumed delta hedged'}
    (args.output/'feature_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
