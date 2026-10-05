"""Compact licensed pair data in mfe_wrds; train later in xfund without pyarrow.

Two passes: select contracts from each contemporaneous chain, then retain ALL
observed histories of those IDs, including invalid/missing-Greek observations.
The compact pickle is a trusted local artifact, never load third-party pickles.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings
from scripts.contract_q_engine import common_expiry_pair
from scripts.download_pair_contract_history import file_hash
from scripts.run_contract_pilot import normalize_quotes


def portable_table(frame):
    frame = frame.astype(object)
    frame.columns = pd.Index(frame.columns.to_numpy(dtype=object),dtype=object)
    return frame


def prepare(root, secids, output, settings=None):
    cfg = settings or Settings()
    requests = json.loads((root/'requests.json').read_text())
    for job in requests:
        if not (root/job['path']).exists():
            raise FileNotFoundError(f"Incomplete download: {job['path']}")
    stocks = pd.concat([pd.read_parquet(root/j['path']) for j in requests if j['kind']=='underlying'],ignore_index=True)
    stocks = stocks[stocks.secid.isin(secids)].copy()
    stocks.date = pd.to_datetime(stocks.date)
    stocks = stocks.sort_values(['secid','date'])
    for col in ('close','cfadj','return'):
        stocks[col] = pd.to_numeric(stocks[col],errors='coerce')
    if stocks.duplicated(['date','secid']).any():
        raise ValueError('Duplicate underlying observations')
    # Vendor underlying returns handle splits; no volatility from unadjusted price jumps.
    for sid,g in stocks.groupby('secid'):
        r = pd.Series(g['return'].to_numpy(float),index=pd.DatetimeIndex(g.date))
        squared = np.log1p(r.where(r>-1))**2
        expected = pd.Series(1.,index=r.index).rolling('30D').sum()
        count = squared.rolling('30D').count()
        rv = np.sqrt(squared.rolling('30D',min_periods=15).mean().where(count==expected)*252)*100
        stocks.loc[g.index,'rv'] = rv.to_numpy()
    returns = stocks.pivot(index='date',columns='secid',values='return')
    corr = returns[secids[0]].rolling(30,min_periods=15).corr(returns[secids[1]])
    stocks['corr'] = stocks.date.map(corr)
    rates = pd.read_parquet(root/'zero_curve.parquet')
    rates.date = pd.to_datetime(rates.date)
    rates['distance'] = abs(pd.to_numeric(rates.days)-30)
    rates = rates.sort_values(['date','distance','days']).drop_duplicates('date')
    rates['rate'] = pd.to_numeric(rates.rate)/100
    stocks = pd.merge_asof(stocks.sort_values('date'),rates[['date','rate']].sort_values('date'),
                          on='date',direction='backward',tolerance=pd.Timedelta(days=7))
    dividends = pd.read_parquet(root/'distributions.parquet')
    dividends = dividends[dividends.secid.isin(secids)&dividends.cancel_flag.astype(str).eq('0')].copy()
    for col in ('ex_date','declare_date'):
        dividends[col] = pd.to_datetime(dividends[col])
    dividends['amount'] = pd.to_numeric(dividends.amount)
    dividends['ordinary'] = (dividends.distr_type.astype(str).eq('1') &
        dividends.currency.eq('USD') & pd.to_numeric(dividends.adj_factor).eq(1) &
        dividends.approx_flag.astype(str).eq('0') & dividends.liquid_flag.astype(str).eq('0'))
    calendar = pd.DatetimeIndex(sorted(stocks.date.unique()))
    spot = stocks.pivot(index='date',columns='secid',values='close').reindex(columns=secids)
    paths = [root/j['path'] for j in requests if j['kind']=='quotes']
    paths.sort()
    chosen, selected_ids, carry_oi = {}, set(), {}
    # Keep lagged OI available across annual file boundaries.
    def normalized(path, previous):
        q = normalize_quotes(pd.read_parquet(path,filters=[('secid','in',secids)]))
        q = q.sort_values(['optionid','date'])
        q['known_open_interest'] = q.groupby('optionid').open_interest.shift(1)
        first = ~q.optionid.duplicated()
        q.loc[first,'known_open_interest'] = q.loc[first,'optionid'].map(previous)
        previous.update(q.groupby('optionid').tail(1).set_index('optionid').open_interest.to_dict())
        return q
    for path in paths:
        q = normalized(path,carry_oi)
        for date,chain in q.groupby('date'):
            t = calendar.get_indexer([date])[0]
            if t<0 or t+1>=len(calendar) or not np.isfinite(spot.loc[date]).all():
                continue
            pair = common_expiry_pair(chain,spot.loc[date].to_numpy(float),calendar[t+1],secids,cfg)
            chosen[date] = pair
            if pair is not None:
                selected_ids.update(pair)
        print(f'Selected using contemporaneous chains: {path.parent.name}',flush=True)
    retained, carry_oi = [], {}
    for path in paths:
        q = normalized(path,carry_oi)
        retained.append(q[q.optionid.isin(selected_ids)].copy())
    quotes = pd.concat(retained,ignore_index=True)
    # Object-backed tables avoid pickling ArrowStringArray dependencies into xfund.
    bundle = dict(quotes=portable_table(quotes),stocks=portable_table(stocks),calendar=calendar,secids=secids,
                  dividends=portable_table(dividends),candidates=chosen,settings=vars(cfg))
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.to_pickle(bundle,output,protocol=4)
    manifest = dict(format_version=1,secids=secids,start=str(calendar[0].date()),end=str(calendar[-1].date()),
                    sessions=len(calendar),quote_rows=len(quotes),selected_ids=len(selected_ids),
                    eligible_signal_dates=sum(v is not None for v in chosen.values()),
                    sha256=file_hash(output),settings=vars(cfg),
                    sources=[dict(path=str(root/j['path']),sha256=file_hash(root/j['path'])) for j in requests],
                    license='WRDS/OptionMetrics: licensed local data, no redistribution')
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('sources','settings')},indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('data/q_contracts/AMD_NVDA'))
    ap.add_argument('--secids',type=int,nargs=2,default=[101121,108321])
    ap.add_argument('--output',type=Path)
    args = ap.parse_args()
    prepare(args.root,args.secids,args.output or args.root/'prepared.pkl')


if __name__=='__main__':
    main()
