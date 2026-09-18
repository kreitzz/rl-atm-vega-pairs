"""Historical IV spread features, using strictly earlier observations for baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def spread_history_features(spread):
    spread=spread.sort_index().astype(float)
    prior=spread.shift(1)
    mean=prior.rolling(63,min_periods=63).mean()
    std=prior.rolling(63,min_periods=63).std(ddof=1)
    # Explicit floor avoids division by zero in a flat historical spread.
    z=(spread-mean)/std.clip(lower=.1)
    return pd.DataFrame({'spread_mean_63':mean,'spread_z_63':z,
                         'spread_change_21':spread-spread.shift(21)},index=spread.index)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--panel',type=Path,default=Path('artifacts/short_rv_momentum/dataset.parquet'))
    p.add_argument('--spread-history',type=Path,default=Path('artifacts/rv_vix_pass/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('artifacts/spread_history'))
    p.add_argument('--features-output',type=Path,default=Path('config/spread_history_features.json'))
    args=p.parse_args()
    panel=pd.read_parquet(args.panel).sort_index()
    history=pd.read_parquet(args.spread_history,columns=['spread'])
    panel=panel.join(spread_history_features(history.spread))
    base=['spread','amd_earnings_days_capped','nvda_earnings_days_capped','trailing_rv_amd','trailing_rv_nvda']
    features={'rv30cal':base,'spread_mean':base+['spread_mean_63'],
              'spread_zscore':base+['spread_z_63'],
              'spread_mean_zscore':base+['spread_mean_63','spread_z_63'],
              'spread_change':base+['spread_change_21'],
              'spread_all_history':base+['spread_mean_63','spread_z_63','spread_change_21']}
    required=sorted({c for fs in features.values() for c in fs})
    before=len(panel);panel=panel.dropna(subset=required)
    args.output.mkdir(parents=True,exist_ok=True)
    args.features_output.parent.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(args.output/'dataset.parquet')
    args.features_output.write_text(json.dumps(features,indent=2)+'\n')
    summary={'rows':len(panel),'extra_dropped':before-len(panel),'first':str(panel.index.min().date()),'last':str(panel.index.max().date()),
             'rolling_mean_std':'previous 63 available daily IV-spread observations, excluding entry; sample standard deviation',
             'zscore':'(current spread - lagged rolling mean) / max(lagged rolling SD, 0.1 vol points)',
             'change':'current spread minus spread 21 available observations earlier',
             'calendar':'windows count available observations; missing quotes are not interpolated',
             'base':'IV spread, known earnings countdowns, 30-calendar-day trailing RV per stock; same panel as preceding sizing run'}
    (args.output/'feature_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
