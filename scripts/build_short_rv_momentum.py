"""Causal short-horizon realized volatility and standard 252/21 momentum."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

if __package__:
    from .run_first_pass import load_return_history
else:
    from run_first_pass import load_return_history


def short_rv(returns, window):
    squared=np.log1p(returns.astype(float).where(returns>-1))**2
    return 100*np.sqrt(252*squared.rolling(window,min_periods=window).mean())


def momentum_12_1(total_returns):
    # At entry t, compound r[t-251] through r[t-21]: exclude the latest 21 returns.
    logret=np.log1p(total_returns.astype(float).where(total_returns>-1))
    return 100*np.expm1(logret.rolling(231,min_periods=231).sum().shift(21))


def feature_sets():
    base=['spread','amd_earnings_days_capped','nvda_earnings_days_capped']
    result={}
    for horizon in ['30cal','5td','10td']:
        rv=['trailing_rv_amd','trailing_rv_nvda'] if horizon=='30cal' else [f'rv_{horizon}_amd',f'rv_{horizon}_nvda']
        for momentum in [False,True]:
            for vix in [False,True]:
                name=f'rv{horizon}'+('_momentum' if momentum else '')+('_vix' if vix else '')
                result[name]=base+rv+(['momentum_12_1_amd','momentum_12_1_nvda'] if momentum else [])+(['vix_previous_close'] if vix else [])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--panel',type=Path,default=Path('artifacts/rv_vix_pass/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('artifacts/short_rv_momentum'))
    p.add_argument('--features-output',type=Path,default=Path('config/short_rv_momentum_features.json'))
    args=p.parse_args()
    panel=pd.read_parquet(args.panel).sort_index()
    returns=load_return_history(args.source_root)
    total_returns=load_return_history(args.source_root,return_column='ret')
    for window in [5,10]:
        rv=short_rv(returns,window).rename(columns={t:f'rv_{window}td_{t.lower()}' for t in ['AMD','NVDA']})
        panel=panel.join(rv)
    momentum=momentum_12_1(total_returns).rename(columns={t:f'momentum_12_1_{t.lower()}' for t in ['AMD','NVDA']})
    panel=panel.join(momentum)
    features=feature_sets()
    required=sorted({f for fs in features.values() for f in fs})
    before=len(panel)
    panel=panel.dropna(subset=required)
    args.output.mkdir(parents=True,exist_ok=True)
    args.features_output.parent.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(args.output/'dataset.parquet')
    args.features_output.write_text(json.dumps(features,indent=2)+'\n')
    summary={'first':str(panel.index.min().date()),'last':str(panel.index.max().date()),'rows':len(panel),'dropped_for_warmup':before-len(panel),
             'short_rv':'5 or 10 consecutive trading-day C2C RMS log returns through entry, annualized sqrt(252)',
             'momentum':'total-return cumulative gain from 252 trading days ago to 21 trading days ago; 231 returns; percentage points',
             'comparison':'12 variants: 3 RV horizons x momentum yes/no x VIX yes/no; same complete panel for all',
             'source_root':str(args.source_root)}
    (args.output/'feature_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
