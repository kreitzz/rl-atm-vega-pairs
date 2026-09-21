"""Point-in-history input/liquidity screen; no payoff ranking of candidate pairs."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.explore_universe_gap import correct_earnings


def forward_series(ret, days=30):
    """Match forward_rv with cumulative sums, retaining missing-return exclusions."""
    ret=ret.astype(float).sort_index();dates=ret.index;end=dates+pd.Timedelta(days=days)
    z=np.log1p(ret.where(ret>-1)).to_numpy();valid=np.isfinite(z)
    sums=np.r_[0.,np.cumsum(np.where(valid,z*z,0.))];counts=np.r_[0,np.cumsum(valid)]
    start=np.arange(len(ret))+1;stop=dates.searchsorted(end,side='right')
    n=stop-start;good=(counts[stop]-counts[start]==n)&(n>=15)&(end<=dates.max())
    value=np.sqrt(252*np.divide(sums[stop]-sums[start],n,out=np.zeros(len(ret)),where=n>0))*100
    value[~good]=np.nan
    return pd.Series(value,index=dates)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();root=args.source_root;out=Path('artifacts/broad_pairs');out.mkdir(exist_ok=True);(out/'assets').mkdir(exist_ok=True)
    reports=Path('reports/broad_pairs');reports.mkdir(exist_ok=True)
    config=json.loads(Path('config/broad_pair_research.json').read_text())
    om=root/'optionmetrics';master=pd.read_parquet(om/'security_master.parquet');names=pd.read_parquet(om/'security_name_history.parquet').sort_values('effect_date')
    links=pd.read_parquet(om/'crsp_optionmetrics_link.parquet');links=links[links.score.le(2)]
    etfs=json.loads(Path('config/etf_pairs.json').read_text())
    etf_ids={v['secid'] for v in etfs['assets'].values() if v['etf']}
    allowed=set(master.loc[master.issue_type.astype(str).eq('0'),'secid'])|etf_ids
    links=links[links.secid.isin(allowed)]
    metadata=[]
    for (secid,permno),g in links.groupby(['secid','permno']):
        if g.edate.max()<pd.Timestamp('2002-01-01'):continue
        history=names[names.secid.eq(secid)]
        if history.empty:continue
        latest=history.iloc[-1]
        metadata.append(dict(asset=f's{int(secid)}p{int(permno)}',secid=int(secid),permno=int(permno),ticker=str(latest.ticker),issuer=str(latest.issuer),etf=secid in etf_ids,
                             intervals=[(str(x.sdate.date()),str(x.edate.date())) for x in g.itertuples()]))
    if not args.resume:
        secids=sorted({x['secid'] for x in metadata});permnos=sorted({x['permno'] for x in metadata})
        files=sorted((om/'atm_iv_compact').rglob('*.parquet'))+sorted((om/'atm_implied_volatility').glob('*.parquet'))
        chunks=[pd.read_parquet(f,columns=['secid','date','impl_volatility'],filters=[('secid','in',secids),('days','=',30),('delta','=',50),('cp_flag','=','C')]) for f in files]
        iv=pd.concat(chunks).drop_duplicates(); iv=iv[iv.impl_volatility.gt(0)]
        if iv.duplicated(['secid','date']).any():raise ValueError('Conflicting IV records')
        ivs={int(k):g.set_index('date').impl_volatility.astype(float)*100 for k,g in iv.groupby('secid')};del chunks,iv
        chunks=[pd.read_parquet(f,columns=['permno','date','retx']) for f in sorted((root/'crsp_underlying_daily').glob('*.parquet'))]
        returns=pd.concat(chunks).drop_duplicates();calendar=pd.DatetimeIndex(returns.date.unique()).sort_values();calendar=calendar[calendar>=pd.Timestamp('2002-01-01')]
        returns=returns[returns.permno.isin(permnos)];rs={int(k):g.set_index('date').retx.astype(float) for k,g in returns.groupby('permno')};del chunks,returns
        pd.Series(calendar,name='date').to_csv(out/'calendar.csv',index=False)
        chunks=[pd.read_parquet(f,columns=['secid','date','volume','open_interest'],filters=[('cp_flag','=','C'),('secid','in',secids)]) for f in sorted((om/'option_activity').glob('*.parquet'))]
        activity=pd.concat(chunks).drop_duplicates();acts={int(k):g.set_index('date')[['volume','open_interest']].astype(float) for k,g in activity.groupby('secid')};del chunks,activity
        earnings=correct_earnings(pd.read_parquet(root/'compustat/underlying_quarterly_earnings.parquet'))
        supplement=pd.read_csv('config/earnings_supplement.csv',parse_dates=['earnings_date'])
        expected=pd.Series(1,index=calendar).rolling('30D',closed='right').sum()
        ready=[]
        for item in metadata:
            secid=item['secid'];permno=item['permno']
            if secid not in ivs or secid not in acts or permno not in rs:continue
            ret=rs[permno].reindex(calendar);valid=pd.Series(False,index=calendar)
            for start,end in item['intervals']:valid|=(calendar>=start)&(calendar<=end)
            # Never bridge another CRSP security or use returns outside its link period.
            ret=ret.where(valid)
            square=np.log1p(ret.where(ret>-1))**2
            rv=(252*square.rolling('30D',min_periods=15,closed='right').mean())**.5*100
            rv=rv.where(square.rolling('30D',closed='right').count().eq(expected))
            rv.loc[calendar<calendar.min()+pd.Timedelta(days=30)]=np.nan
            p=pd.DataFrame({'ret':ret,'trailing_rv':rv,'forward_rv':forward_series(ret)},index=calendar)
            p=p.join(ivs[secid].rename('iv')).join(acts[secid]);p.loc[~valid,'iv']=np.nan
            if item['etf']:
                spec=next(v for v in etfs['assets'].values() if v['secid']==secid and v['etf'])
                p.loc[p.index<pd.Timestamp(spec['history_start']),['iv','ret','trailing_rv','forward_rv']]=np.nan
                p['earnings']=31.
            else:
                dates=pd.to_datetime(earnings.loc[earnings.permno.eq(permno),'rdq']).dropna()
                dates=pd.DatetimeIndex(pd.concat([dates,supplement.loc[supplement.ticker.eq(item['ticker']),'earnings_date']]).unique()).sort_values()
                if len(dates)<20:continue
                positions=dates.searchsorted(p.index,side='left')
                nxt=pd.Series([dates[i] if i<len(dates) else pd.NaT for i in positions],index=p.index)
                p['earnings']=(nxt-pd.Series(p.index,index=p.index)).dt.days.clip(upper=31)
            p=p.loc[p.iv.notna()&(p.index<='2025-08-29')]
            if len(p)<504:continue
            p.to_parquet(out/'assets'/f"{item['asset']}.parquet");ready.append(item)
        metadata=ready;(out/'assets.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('Prepared',len(metadata),'linked assets',flush=True)
    else:metadata=json.loads((out/'assets.json').read_text())
    panels={x['asset']:pd.read_parquet(out/'assets'/f"{x['asset']}.parquet") for x in metadata}
    calendar=pd.DatetimeIndex(pd.read_csv(out/'calendar.csv',parse_dates=['date']).date)
    asset_info={x['asset']:x for x in metadata};screens=[];pair_defs={};asset_screens=[]
    for year in config['screen_years']:
        # Freeze input-only candidate formation before this fold's test starts.
        cutoff=pd.Timestamp(year,1,1);window=calendar[calendar<cutoff][-252:];eligible=[]
        for item in metadata:
            p=panels[item['asset']].reindex(window)
            if p[['iv','ret','trailing_rv','earnings','volume','open_interest']].notna().all(axis=1).sum()<config['minimum_observed_sessions']:continue
            volume=p.volume.median();oi=p.open_interest.median()
            if volume<config['minimum_median_call_volume'] or oi<config['minimum_median_call_open_interest']:continue
            h=names[names.secid.eq(item['secid'])&names.effect_date.lt(cutoff)]
            if h.empty or pd.isna(h.iloc[-1].sic):continue
            last=h.iloc[-1];sic=str(int(last.sic)).zfill(4);ticker=str(last.ticker)
            row=dict(asset=item['asset'],sic3=sic[:3],sic2=sic[:2],ticker=ticker,median_call_volume=volume,median_call_oi=oi,etf=item['etf'])
            eligible.append(row);asset_screens.append(dict(year=year,**row))
        e=pd.DataFrame(eligible);candidates=[]
        if e.empty:continue
        for industry,g in e[~e.etf].groupby('sic3'):
            g=g.sort_values(['median_call_oi','asset'],ascending=[False,True])
            # A company/share-class appears at most once in an industry screen.
            chosen=[];seen=set()
            for row in g.to_dict('records'):
                permno=asset_info[row['asset']]['permno']
                if permno in seen:continue
                seen.add(permno);chosen.append(row)
                if len(chosen)==config['maximum_names_per_historical_sic3']:break
            for a,b in combinations(chosen,2):
                ret=pd.concat([panels[a['asset']].ret.reindex(window),panels[b['asset']].ret.reindex(window)],axis=1).dropna()
                corr=ret.iloc[:,0].corr(ret.iloc[:,1]) if len(ret)>=200 else np.nan
                if not np.isfinite(corr) or corr<config['minimum_return_correlation']:continue
                a,b=sorted([a,b],key=lambda x:x['asset']);key=a['asset']+'__'+b['asset']
                candidates.append(dict(pair=key,asset_a=a['asset'],asset_b=b['asset'],label=a['ticker']+'/'+b['ticker'],industry=industry,division=a['sic2'],kind='stock_stock',correlation=corr,liquidity=min(a['median_call_oi'],b['median_call_oi'])))
        # Round-robin industries, highest input liquidity first within industry.
        c=pd.DataFrame(candidates)
        if len(c):
            c=c.sort_values(['industry','liquidity','pair'],ascending=[True,False,True]);c['rank']=c.groupby('industry').cumcount()
            candidates=c.sort_values(['rank','liquidity','pair'],ascending=[True,False,True]).head(config['maximum_stock_pairs_per_screen']).drop(columns='rank').to_dict('records')
        emap={r['asset']:r for r in eligible}
        for definition in etfs['pairs']:
            legs=[]
            for ticker in definition['pair'].split('_'):
                spec=etfs['assets'][ticker];key=f"s{spec['secid']}p{spec['permno']}"
                if key not in emap:break
                legs.append(emap[key])
            if len(legs)!=2:continue
            a,b=legs;ret=pd.concat([panels[a['asset']].ret.reindex(window),panels[b['asset']].ret.reindex(window)],axis=1).dropna()
            corr=ret.iloc[:,0].corr(ret.iloc[:,1]) if len(ret)>=200 else np.nan
            if not np.isfinite(corr) or corr<config['minimum_return_correlation']:continue
            key=a['asset']+'__'+b['asset'];candidates.append(dict(pair=key,asset_a=a['asset'],asset_b=b['asset'],label=definition['pair'].replace('_','/'),industry='ETF',division='ETF',kind=definition['kind'],correlation=corr,liquidity=min(a['median_call_oi'],b['median_call_oi'])))
        for c in candidates:
            screens.append(dict(year=year,**c));pair_defs[c['pair']]={k:v for k,v in c.items() if k not in ['correlation','liquidity']}
        print('Screen',year,'eligible names',len(e),'pairs',len(candidates),flush=True)
    pd.DataFrame(asset_screens).to_csv(reports/'asset_screens.csv',index=False);pd.DataFrame(screens).to_csv(reports/'candidate_screens.csv',index=False)
    (out/'pairs.json').write_text(json.dumps(list(pair_defs.values()),indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in x.items() if k!='intervals'} for x in metadata]).to_csv(reports/'assets.csv',index=False)
    print('Unique candidate pairs:',len(pair_defs),flush=True)


if __name__=='__main__':main()
