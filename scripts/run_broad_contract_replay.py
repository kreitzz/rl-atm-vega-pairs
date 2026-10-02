"""Replay every saved broad-universe policy on actual option contracts.

Historical screen membership is preserved. One fixed $1m sleeve per unique
pair; no unavailable or failed sleeve is silently removed from the universe.
"""
import argparse
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import re
import resource
import time

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, UnresolvedPosition, simulate
from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import audit, scores
from scripts.select_atm_contracts import load_calendar

ROOT=Path('data/option_contracts')
CACHE=Path('data/contract_history_shared')
MODELS=['rl_reference','rl_sized','nn_fixed']


def asset_id(asset):
    return int(re.fullmatch(r's(\d+)p\d+',asset).group(1))


def universe():
    screens=pd.read_csv('reports/broad_pairs/candidate_screens.csv')
    screens=screens[screens.year.isin([2018,2020,2022,2024])]
    definitions={p['pair']:p for p in json.loads(Path('artifacts/broad_pairs/pairs.json').read_text())}
    rows=[];seen=set()
    for key,g in screens.groupby('pair',sort=True):
        p=definitions[key];ids=[asset_id(p['asset_a']),asset_id(p['asset_b'])]
        canonical='__'.join(map(str,sorted(ids)));assert canonical not in seen;seen.add(canonical)
        paths=[Path('artifacts/broad_pairs/memory')/key/f'{year}.csv' for year in sorted(g.year.unique())]
        rows.append(dict(pair=canonical,label=p['label'],secids=ids,group='broad',source_pair=key,
                         industry=str(p['industry']),screen_years=sorted(map(int,g.year.unique())),
                         paths=[str(path) for path in paths if path.exists()],
                         columns=dict(rl_reference='rl_no_memory_action',rl_sized='rl_memory_action',nn_fixed='nn_memory_action')))
    for p in pd.read_csv('reports/contract_data/pairs.csv').itertuples():
        canonical='__'.join(map(str,sorted([p.secid_a,p.secid_b])))
        if canonical in seen:continue
        label=p.labels.split('|')[0];path=Path('artifacts/nn_rl_benchmark')/f'{label}.csv'
        if not path.exists():raise FileNotFoundError(path)
        rows.append(dict(pair=canonical,label=label,secids=[p.secid_a,p.secid_b],group='legacy_extra',
                         source_pair=label,industry='legacy',screen_years=[2018,2020,2022,2024],paths=[str(path)],
                         columns=dict(rl_reference='rl_base_sized_action',rl_sized='rl_both_sized_action',nn_fixed='nn_both_fixed_action')))
        seen.add(canonical)
    assert len(rows)==358 and sum(p['group']=='broad' for p in rows)==350
    return rows


class Market:
    def __init__(self,calendar):
        self.calendar=calendar
        zero=pd.read_parquet(ROOT/'zero_curve.parquet');zero['date']=pd.to_datetime(zero.date)
        zero['distance']=abs(zero.days-30)
        zero=zero.sort_values(['date','distance','days']).drop_duplicates('date').sort_values('date')
        zero['rate']/=100
        self.rates=pd.merge_asof(pd.DataFrame({'date':calendar}),zero[['date','rate']],on='date',direction='backward',tolerance=pd.Timedelta(days=7)).set_index('date').rate
        if self.rates.isna().any():raise ValueError('Missing financing rate')
        self.divs=pd.read_parquet(ROOT/'distributions.parquet')
        for col in ['ex_date','declare_date']:self.divs[col]=pd.to_datetime(self.divs[col])
        d=self.divs
        self.divs=d[d.cancel_flag.eq('0')&d.distr_type.eq('1')&d.currency.eq('USD')&d.adj_factor.eq(1)&d.approx_flag.eq('0')&d.liquid_flag.eq('0')]
        self.stocks=pd.concat([pd.read_parquet(ROOT/'underlying'/f'year={y}.parquet',columns=['date','secid','close','cfadj']) for y in sorted(set(calendar.year))],ignore_index=True)
        self.stocks['date']=pd.to_datetime(self.stocks.date)

    @lru_cache(maxsize=4)
    def asset(self,sid):
        qpaths=[CACHE/str(y)/f'{sid}.parquet' for y in sorted(set(self.calendar.year))]
        spaths=[CACHE/str(y)/f'{sid}_selection.parquet' for y in sorted(set(self.calendar.year))]
        qs=[pd.read_parquet(p) for p in qpaths if p.exists()]
        ss=[pd.read_parquet(p) for p in spaths if p.exists()]
        q=pd.concat(qs,ignore_index=True) if qs else pd.DataFrame(columns=['date','optionid','secid'])
        selection=pd.concat(ss,ignore_index=True) if ss else pd.DataFrame(columns=['date','secid','optionid'])
        stocks=self.stocks[self.stocks.secid.eq(sid)].set_index('date').reindex(self.calendar)
        stocks['secid']=sid;stocks['rate']=self.rates;stocks.index.name='date'
        return q,selection,stocks.reset_index()

    def pair(self,secids):
        a,b=[self.asset(s) for s in secids]
        return (pd.concat([a[0],b[0]],ignore_index=True),pd.concat([a[2],b[2]],ignore_index=True),
                pd.concat([a[1],b[1]],ignore_index=True),self.divs[self.divs.secid.isin(secids)])


def load_actions(definition,calendar):
    pieces=[]
    for path in definition['paths']:
        frame=pd.read_csv(path,index_col=0,parse_dates=[0])
        p=frame[list(definition['columns'].values())].rename(columns={v:k for k,v in definition['columns'].items()})
        if definition['group']=='broad':
            year=int(Path(path).stem)
            p=p[(p.index>=pd.Timestamp(year,1,1))&(p.index<pd.Timestamp(year+2,1,1))]
        pieces.append(p)
    if pieces:
        actions=pd.concat(pieces).sort_index()
        if actions.index.has_duplicates:raise ValueError('Overlapping saved policy folds')
    else:actions=pd.DataFrame(columns=MODELS,index=pd.DatetimeIndex([]),dtype=float)
    stop=calendar[-1]-pd.Timedelta(days=35)
    actions=actions[actions.index.isin(calendar)&(actions.index<=stop)]
    if not np.isfinite(actions.to_numpy(float)).all():raise ValueError('Invalid saved action')
    assert np.isin(actions.to_numpy(float),np.arange(-1,1.01,.25)).all()
    assert np.isin(actions.nn_fixed,[-1,0,1]).all()
    return actions


def collect_reports(definitions,status,output,report,calendar,cfg,rates):
    pd.DataFrame(status).to_csv(report/'status.csv',index=False)
    result=pd.DataFrame(status);summaries=[];annual=[]
    cash_return=rates.shift(1).fillna(0)*calendar.to_series().diff().dt.days.fillna(0)/365
    for group,roster in [('broad',[p for p in definitions if p['group']=='broad']),('all',definitions)]:
        for model in MODELS:
            members=result[result.pair.isin([p['pair'] for p in roster])&result.model.eq(model)]
            complete=members[members.status.eq('complete')]
            failures=members[members.status.ne('complete')]
            if len(members)!=len(roster) or len(failures):
                summaries.append(dict(group=group,model=model,status='incomplete',pairs=len(roster),completed_pairs=len(complete),failed_pairs=len(failures)))
                continue
            book=pd.DataFrame(0.,index=calendar,columns=['nav','cash_benchmark'])
            for row in complete.itertuples():
                nav=pd.read_parquet(output/row.pair/model/'daily_nav.parquet',columns=['nav','cash_benchmark'])
                assert nav.index.equals(calendar);book+=nav
            capital=len(roster)*cfg.capital
            book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-cash_return
            years=(calendar[-1]-calendar[0]).days/365.25
            summaries.append(dict(group=group,model=model,status='complete',pairs=len(roster),completed_pairs=len(complete),failed_pairs=0,
                                  **scores(book,capital),cagr=(book.nav.iloc[-1]/capital)**(1/years)-1))
            book.to_csv(output/f'{group}_{model}_portfolio.csv')
            for year,g in book.groupby(book.index.year):
                previous=capital if g.index[0]==calendar[0] else book.loc[:g.index[0]].nav.iloc[-2]
                annual.append(dict(group=group,model=model,year=int(year),total_return=g.nav.iloc[-1]/previous-1,
                                   annualized_excess_sharpe=np.sqrt(252)*g.excess_return.mean()/g.excess_return.std() if g.excess_return.std()>0 else np.nan))
    pd.DataFrame(summaries).to_csv(report/'portfolio.csv',index=False)
    pd.DataFrame(annual).to_csv(report/'annual.csv',index=False)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/contracts/broad_replay'))
    parser.add_argument('--report',type=Path,default=Path('reports/broad_contract_replay'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int,help='Smoke-test prefix only; not a full-universe result')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.report.mkdir(parents=True,exist_ok=True)
    definitions=universe()
    if args.limit:definitions=definitions[:args.limit]
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2018-01-01')&(calendar<='2025-08-29')]
    cfg=Settings(option_spread_fraction=0,option_penalty_vol_points=.1,risk_controls=True,
                 repair_crossed_held_quotes=True,support_integer_splits=True,allow_missing_flat_underlying=True)
    sources={path:digest(path) for p in definitions for path in p['paths']}
    fingerprint=dict(settings=asdict(cfg),definitions=definitions,signal_hashes=sources,
                     cache_manifest_sha256=digest(CACHE/'manifest.json'),
                     code_sha256={str(p):digest(p) for p in [Path(__file__),Path('scripts/contract_engine.py'),Path('scripts/contract_risk.py'),Path('scripts/run_selected_pair_backtest.py')]})
    metadata=args.output/'inputs.json'
    if args.resume and metadata.exists() and json.loads(metadata.read_text())!=fingerprint:raise ValueError('Resume inputs changed')
    metadata.write_text(json.dumps(fingerprint,indent=2)+'\n')
    (args.report/'universe.json').write_text(json.dumps(definitions,indent=2)+'\n')
    market=Market(calendar);status=[];start=time.monotonic()
    for n,definition in enumerate(definitions,1):
        key=definition['pair'];folder=args.output/key;folder.mkdir(parents=True,exist_ok=True)
        actions=load_actions(definition,calendar);actions.to_csv(folder/'actions.csv')
        q,stocks,selection,divs=market.pair(definition['secids'])
        for model in MODELS:
            dest=folder/model;dest.mkdir(exist_ok=True);path=dest/'status.json'
            if args.resume and path.exists():
                status.append(json.loads(path.read_text()));continue
            common=dict(pair=key,label=definition['label'],group=definition['group'],model=model,
                        saved_signals=len(actions),nonzero_signals=int(actions[model].ne(0).sum()),
                        screen_years=definition['screen_years'],no_fitted_policy=not bool(definition['paths']))
            try:
                nav,trades,decisions=simulate(q,stocks,actions[model],calendar,definition['secids'],cfg,divs,selection)
                cash_return=market.rates.shift(1).fillna(0)*calendar.to_series().diff().dt.days.fillna(0)/365
                nav['cash_benchmark']=cfg.capital*(1+cash_return).cumprod();nav['excess_return']=nav.return_-cash_return
                check=audit(nav,trades,stocks,cfg,definition['secids'])
                nav.to_parquet(dest/'daily_nav.parquet');trades.to_csv(dest/'trades.csv',index=False);decisions.to_csv(dest/'decisions.csv',index=False)
                row=dict(**common,status='complete',entries=int(decisions.status.eq('entered').sum()) if len(decisions) else 0,
                         **scores(nav,cfg.capital),**check)
            except (UnresolvedPosition,ValueError) as exc:
                row=dict(**common,status='incomplete',error=str(exc))
                if hasattr(exc,'partial_nav'):exc.partial_nav.to_parquet(dest/'partial_nav.parquet')
            path.write_text(json.dumps(row,indent=2)+'\n');status.append(row)
            print(f"{n}/{len(definitions)} {definition['label']} {model}: {row['status']} {row.get('error','')} {row.get('annualized_excess_sharpe','')}",flush=True)
        pd.DataFrame(status).to_csv(args.report/'status.csv',index=False)
        (args.report/'progress.json').write_text(json.dumps(dict(completed_pairs=n,total_pairs=len(definitions),
            elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    summary=collect_reports(definitions,status,args.output,args.report,calendar,cfg,market.rates)
    manifest=dict(settings=asdict(cfg),pairs=len(definitions),models=MODELS,start=str(calendar[0].date()),end=str(calendar[-1].date()),
                  policy='frozen saved proxy-trained policies; original screen membership; no new monthly refits or beta filter',
                  allocation='fixed equal $1m sleeves over unique-pair union; inactive and unfitted pairs stay in cash; no missing-pair reweighting',
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-start,
                  inputs_sha256=digest(metadata))
    (args.report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':main()
