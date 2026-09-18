"""Attach earnings dates; actual countdowns require a known-calendar assumption."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def attach(panel, calendar):
    out = panel.copy().sort_index()
    for ticker in ['AMD', 'NVDA']:
        dates = pd.DatetimeIndex(calendar.loc[calendar.ticker.eq(ticker), 'earnings_date'].drop_duplicates().sort_values())
        if dates.empty:
            raise ValueError(f'No earnings dates for {ticker}')
        prefix = ticker.lower()
        entry = pd.DatetimeIndex(out.index)
        # Strictly earlier reports for lagged context: same-day release time is unknown.
        pos = dates.searchsorted(entry, side='left')
        prior = pd.Series([dates[i-1] if i else pd.NaT for i in pos], index=out.index)
        upcoming = pd.Series([dates[i] if i < len(dates) else pd.NaT for i in pos], index=out.index)
        out[f'{prefix}_last_earnings_date'] = prior
        out[f'{prefix}_days_since_earnings'] = (pd.Series(entry, index=out.index) - prior).dt.days
        # Historical-cadence forecast uses only reports strictly preceding entry.
        # No realized future RDQ is used to construct this state input.
        cadence = pd.Series(dates).diff().dt.days.rolling(8, min_periods=3).median().clip(70, 120)
        prior_cadence = pd.Series([cadence.iloc[i-1] if i else np.nan for i in pos], index=out.index)
        out[f'{prefix}_earnings_cadence_fallback'] = prior_cadence.isna()
        prior_cadence = prior_cadence.fillna(91.)
        out[f'{prefix}_estimated_days_until_earnings'] = (
            prior_cadence - out[f'{prefix}_days_since_earnings']
        ).clip(lower=0)
        out[f'{prefix}_same_day_earnings_time_unknown'] = entry.isin(dates)
        # Future dates require an explicit known-calendar assumption as features.
        out[f'{prefix}_next_earnings_date_ex_post'] = upcoming
        out[f'{prefix}_days_to_earnings_ex_post'] = (upcoming - pd.Series(entry,index=out.index)).dt.days
        within = (upcoming <= out.end).astype('boolean')
        within.loc[upcoming.isna()] = pd.NA
        out[f'{prefix}_earnings_in_hold_ex_post'] = within
    out['either_earnings_in_hold_ex_post'] = out.amd_earnings_in_hold_ex_post | out.nvda_earnings_in_hold_ex_post
    return out


def load_calendar(source_root):
    source = source_root/'compustat/underlying_quarterly_earnings.parquet'
    d = pd.read_parquet(source, columns=['permno','rdq','datadate'])
    d = d[d.permno.isin([61241,86580])].copy()
    d['ticker'] = d.permno.map({61241:'AMD',86580:'NVDA'})
    d = d.rename(columns={'rdq':'earnings_date','datadate':'fiscal_period_end'})
    d['earnings_date'] = pd.to_datetime(d.earnings_date)
    d['source'] = 'Compustat RDQ'
    supplement = Path(__file__).resolve().parents[1]/'config/earnings_supplement.csv'
    if supplement.exists():
        extra = pd.read_csv(supplement, parse_dates=['earnings_date','fiscal_period_end'])
        d = pd.concat([d,extra],ignore_index=True)
    return d.dropna(subset=['earnings_date']).drop_duplicates(['ticker','earnings_date']).sort_values(['ticker','earnings_date'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--panel', type=Path, default=Path('artifacts/sharpe_pass/dataset.parquet'))
    p.add_argument('--output', type=Path, default=Path('artifacts/earnings'))
    args = p.parse_args()
    source = args.source_root/'compustat/underlying_quarterly_earnings.parquet'
    d = load_calendar(args.source_root)
    args.output.mkdir(parents=True,exist_ok=True)
    d.to_csv(args.output/'amd_nvda_earnings_dates.csv',index=False)
    panel = attach(pd.read_parquet(args.panel),d)
    panel.to_parquet(args.output/'daily_panel_with_earnings.parquet')
    summary = {'source':str(source), 'date_definition':'Compustat RDQ, observed announcement date; no announcement-time or known-as-of field',
               'coverage':{t:{'first':str(g.earnings_date.min().date()),'last':str(g.earnings_date.max().date()),'announcements':len(g),
                              'largest_gap_days':int(g.earnings_date.diff().dt.days.max())} for t,g in d.groupby('ticker')},
               'panel_rows':len(panel),'rows_with_next_date_for_both':int(panel[['amd_next_earnings_date_ex_post','nvda_next_earnings_date_ex_post']].notna().all(axis=1).sum()),
               'same_day_timing':'Same-day announcement counted as possibly in holding window; pre/post-close timing unknown.',
               'model_inputs':'Estimated countdown uses prior cadence. Actual countdown may be used under the explicit assumption that dates within the holding window were known at entry.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
