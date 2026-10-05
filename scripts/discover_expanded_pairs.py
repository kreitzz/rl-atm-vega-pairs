"""Uncapped historical option-universe discovery; no payoff/IV-level ranking."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.run_contract_walkforward import digest

SOURCE=Path('artifacts/broad_pairs')
OUT=Path('reports/expanded_pair_discovery')
NAMES=Path('/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds/optionmetrics/security_name_history.parquet')
RETURNS=NAMES.parent.parent/'crsp_underlying_daily'


def pair_statistics(returns,formation):
    """Only preceding 126 calendar-listed trading sessions; no missing returns."""
    history=returns.loc[returns.index<formation].tail(126)
    if len(history)<126:return pd.DataFrame(),pd.DataFrame()
    history=history.loc[:,history.notna().all()]
    covariance=history.cov()
    beta=covariance.div(pd.Series(np.diag(covariance),index=covariance.index),axis=1)
    return beta,history.corr()


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((SOURCE/'assets.json').read_text());info={x['asset']:x for x in meta}
    calendar=pd.DatetimeIndex(pd.read_csv(SOURCE/'calendar.csv',parse_dates=['date']).date)
    calendar=calendar[(calendar>='2016-01-01')&(calendar<='2025-08-29')]
    frames={}
    for x in meta:
        p=pd.read_parquet(SOURCE/'assets'/f'{x["asset"]}.parquet',columns=['ret','volume','open_interest'])
        frames[x['asset']]=p.reindex(calendar)
    return_files=[p for p in sorted(RETURNS.glob('year=*.parquet')) if 2016<=int(p.stem.split('=')[1])<=2025]
    raw=pd.concat([pd.read_parquet(p,columns=['date','permno','retx'],filters=[('permno','in',sorted({x['permno'] for x in meta}))]) for p in return_files],ignore_index=True).drop_duplicates()
    assert not raw.duplicated(['date','permno']).any()
    wide=raw.pivot(index='date',columns='permno',values='retx').reindex(calendar)
    series={}
    for x in meta:
        valid=pd.Series(False,index=calendar)
        for start,end in x['intervals']:valid|=(calendar>=start)&(calendar<=end)
        series[x['asset']]=wide.get(x['permno'],pd.Series(np.nan,index=calendar)).where(valid)
        frames[x['asset']]['ret']=series[x['asset']]
    returns=pd.DataFrame(series,index=calendar)
    del raw,wide,series
    names=pd.read_parquet(NAMES,columns=['secid','effect_date','ticker','sic','issuer'])
    names=names[names.secid.isin([x['secid'] for x in meta])].sort_values('effect_date')
    downloaded=set(pd.read_csv('reports/contract_data/underlyings.csv').secid)
    old={d['pair'] for d in json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())}
    rows=[];assets=[];counts=[]
    for formation in pd.date_range('2018-01-01','2025-07-01',freq='QS'):
        window=calendar[calendar<formation][-252:]
        historical=names[names.effect_date<formation].drop_duplicates('secid',keep='last').set_index('secid')
        eligible=[]
        for a,p in frames.items():
            x=info[a];past=p.reindex(window)
            if len(window)<252 or past.notna().all(axis=1).sum()<200:continue
            volume=past.volume.median();oi=past.open_interest.median()
            if volume<100 or oi<500 or x['secid'] not in historical.index:continue
            h=historical.loc[x['secid']]
            if pd.isna(h.sic) and not x['etf']:continue
            sic=str(int(h.sic)).zfill(4) if pd.notna(h.sic) else 'ETF'
            eligible.append(dict(asset=a,secid=x['secid'],permno=x['permno'],ticker=str(h.ticker),
                sic3=sic[:3],sic2=sic[:2],issuer=str(h.issuer).strip().upper(),etf=x['etf'],median_call_volume=volume,median_call_oi=oi))
        # Avoid pairing alternative links/share-class representations of one security.
        eligible=sorted(eligible,key=lambda x:(-x['median_call_oi'],x['asset']))
        unique=[];sids=set();permnos=set()
        for x in eligible:
            if x['secid'] in sids or x['permno'] in permnos:continue
            sids.add(x['secid']);permnos.add(x['permno']);unique.append(x)
        unique=sorted(unique,key=lambda x:x['secid'])
        lookup={x['asset']:x for x in unique}
        beta,corr=pair_statistics(returns[[x['asset'] for x in unique]],formation)
        if beta.empty:
            counts.append(dict(formation=formation,liquid_names=len(unique),complete_return_names=0,candidate_pairs=0))
            continue
        valid=(beta.ge(.85)&beta.le(1.15)&corr.ge(.5)).to_numpy()
        ii,jj=np.where(np.triu(valid,k=1));start=len(rows)
        for i,j in zip(ii,jj):
            a,b=lookup[beta.index[i]],lookup[beta.columns[j]]
            key=f'{a["secid"]}__{b["secid"]}'
            kind=('etf_pair' if a['etf'] or b['etf'] else
                  'same_issuer' if a['issuer']==b['issuer'] and a['issuer'] not in ['', 'NAN', 'NONE'] else
                  'same_sic3' if a['sic3']==b['sic3'] else 'same_sic2' if a['sic2']==b['sic2'] else 'cross_sector')
            rows.append(dict(formation=formation,pair=key,asset_a=a['asset'],asset_b=b['asset'],
                secid_a=a['secid'],secid_b=b['secid'],label=a['ticker']+'/'+b['ticker'],
                sic3_a=a['sic3'],sic3_b=b['sic3'],relationship=kind,beta=beta.iloc[i,j],correlation=corr.iloc[i,j],
                min_call_oi=min(a['median_call_oi'],b['median_call_oi']),min_call_volume=min(a['median_call_volume'],b['median_call_volume']),
                previously_tested=key in old,both_contracts_downloaded=a['secid'] in downloaded and b['secid'] in downloaded))
        assets.extend(dict(formation=formation,**x,complete_126_returns=x['asset'] in beta.index) for x in unique)
        counts.append(dict(formation=formation,liquid_names=len(unique),complete_return_names=len(beta),candidate_pairs=len(rows)-start))
        print(counts[-1],flush=True)
    candidates=pd.DataFrame(rows);candidates.to_csv(OUT/'candidate_screens.csv',index=False)
    pd.DataFrame(assets).to_csv(OUT/'asset_screens.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'formation_counts.csv',index=False)
    summaries=[]
    for pair,g in candidates.groupby('pair'):
        last=g.sort_values('formation').iloc[-1]
        summaries.append(dict(pair=pair,label=last.label,secid_a=last.secid_a,secid_b=last.secid_b,
            asset_a=last.asset_a,asset_b=last.asset_b,latest_relationship=last.relationship,
            formation_count=len(g),first_formation=g.formation.min(),last_formation=g.formation.max(),
            median_beta=g.beta.median(),median_correlation=g.correlation.median(),
            median_min_call_oi=g.min_call_oi.median(),previously_tested=last.previously_tested,
            both_contracts_downloaded=last.both_contracts_downloaded))
    summary=pd.DataFrame(summaries).sort_values(['formation_count','median_correlation','pair'],ascending=[False,False,True])
    summary.to_csv(OUT/'pairs.csv',index=False)
    new=summary[~summary.previously_tested]
    new[new.both_contracts_downloaded].to_csv(OUT/'new_pairs_cached_names.csv',index=False)
    new[~new.both_contracts_downloaded].to_csv(OUT/'new_pairs_needing_data.csv',index=False)
    missing=sorted((set(new.secid_a)|set(new.secid_b))-downloaded)
    pd.DataFrame([dict(secid=s,labels=';'.join(sorted({x['ticker'] for x in meta if x['secid']==s}))) for s in missing]).to_csv(OUT/'additional_underlyings.csv',index=False)
    stats=dict(prepared_assets=len(meta),formation_dates=len(counts),unique_candidates=len(summary),new_pairs=len(new),
        new_pairs_cached_names=int(new.both_contracts_downloaded.sum()),additional_underlyings=len(missing),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        selection='Quarterly discovery; previous 252 sessions liquidity/coverage, previous 126 complete-session return beta .85–1.15 and correlation >=.5; no industry cap, no pair count cap, no payoff or IV-level ranking',
        orientation='Lower SECID regressed on higher SECID; no orientation search',
        limitations='Prepared 1,922-security local universe is not the entire US option market; upstream standardized-IV availability and historical earnings coverage affect asset inclusion. Both downloaded names does not establish tradable pair/date coverage. Historical candidate membership must be respected in any future test.',
        code_sha256=digest(__file__),source_metadata_sha256=digest(SOURCE/'assets.json'),name_history_sha256=digest(NAMES),
        return_source_hashes={str(p):digest(p) for p in return_files})
    (OUT/'manifest.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
