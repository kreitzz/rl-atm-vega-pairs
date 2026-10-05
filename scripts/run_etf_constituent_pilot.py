"""Frozen 2018 ETF constituent roster: matched beta-gate ablation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.train_expiry_contracts import settings,FILTERS
from scripts.run_aligned_contract_learning import contract_states,load_labels,train_pair,mark_book,asset_paths,NEW_FEATURES,EXTRA,MODELS
from scripts.run_contract_surface_features import augment,FEATURES
from scripts.run_rv_vix_pass import join_vix
from scripts.contract_learning_labels import profitability_memory
from scripts.contract_engine import PreparedMarket,simulate,UnresolvedPosition
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_broad_contract_replay import Market,ROOT,CACHE
from scripts.select_atm_contracts import load_calendar
from scripts.run_contract_walkforward import digest

OUTPUT=Path('artifacts/contracts/etf_constituent_pilot')
REPORT=Path('reports/etf_constituent_pilot')
_market=None


def definitions():
    assets=json.loads(Path('artifacts/broad_pairs/assets.json').read_text())
    lookup={a['ticker']:a for a in assets}
    selected=[]
    for etf,names,snapshot,url in [
        ('OIH',['SLB','HAL','RIG','BKR'],'2018-03-31','https://www.sec.gov/Archives/edgar/data/1137360/000093041318001946/c91215_nq.htm'),
        ('SIL',['HL','AG','CDE'],'2018-07-31','https://www.sec.gov/Archives/edgar/data/1432353/000114420418051506/tv502272_nq.htm')]:
        for name in names:
            a,b=lookup[etf],lookup[name]
            selected.append(dict(pair=etf+'_'+name,label=etf+'/'+name,group='broad',secids=[a['secid'],b['secid']],
                source_pair=a['asset']+'__'+b['asset'],screen_years=list(range(2019,2026)),
                holdings_snapshot=snapshot,holdings_source=url))
    return selected


def eligibility(panel, variant):
    result=panel.correlation.ge(.5) & np.isfinite(panel.beta)
    if variant=='tight':result &= panel.beta.between(.85,1.15)
    elif variant!='no_beta_cutoff':raise ValueError(variant)
    return result


def worker(task):
    global _market
    d,sources,variant=task;folder=OUTPUT/variant/d['pair'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists():return json.loads((folder/'status.json').read_text())
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    cfg=settings();stop=cal[-1]-pd.Timedelta(days=cfg.max_dte+5);evaluation=cal[cal>='2019-01-01']
    if _market is None:_market=Market(cal)
    q,s,selection,divs=_market.pair(d['secids'])
    panel,screen=contract_states(d,selection,s,cal,cfg,beta_tolerance=.15)
    panel['eligible']=eligibility(panel,variant)
    screen['eligible']=eligibility(screen,variant)
    panel=panel[panel.index<=stop];panel['commission_vol_entry']=panel.pop('commission_vol_roundtrip')/2
    panel=augment(d,panel,cal);index=panel.index.copy()
    panel=join_vix(panel,pd.read_csv('data/raw/VIX_History.csv',parse_dates=['DATE']));assert panel.index.equals(index);panel.index=index
    assert panel.vix_previous_close.notna().all() and (panel.vix_date<panel.index).all()
    daily,ends,vegas,statuses=load_labels(d,panel,q,s,selection,cal,divs,cfg,folder,sources)
    ends=pd.DatetimeIndex(ends)
    reward=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
        out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
    panel=panel.join(profitability_memory(panel.index,ends,reward,vegas[:,[0,-1]]>0))
    panel.to_parquet(folder/'states.parquet');screen.to_csv(folder/'beta_screen.csv')
    pd.Series(statuses).value_counts().rename_axis('status').to_csv(folder/'label_status.csv')
    features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES+['vix_previous_close']
    print(f'{d["label"]}: {len(panel)} labels; {dict(pd.Series(statuses).value_counts())}',flush=True)
    actions,fits=train_pair(d,panel,daily,ends,cal,q,s,selection,divs,cfg,_market.rates,
        pd.Timestamp('2019-01-01'),stop,folder,feature_columns=features,memory_filters=FILTERS,recent_refit=True)
    actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
    prepared=PreparedMarket(q,s,selection);records=[]
    for model in MODELS:
        dest=folder/model;dest.mkdir(exist_ok=True);a=actions[model]
        assert panel.loc[a.index[a.ne(0)],'eligible'].all()
        row=dict(variant=variant,pair=d['pair'],label=d['label'],group=d['group'],model=model,nonzero_signals=int(a.ne(0).sum()))
        try:
            nav,t,dec=simulate(q,s,a,evaluation,d['secids'],cfg,divs,selection,prepared)
            mark_book(nav,_market.rates,cfg.capital);checks=audit(nav,t,s,cfg,d['secids'])
            assert t.empty or set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
            nav.to_parquet(dest/'daily_nav.parquet');t.to_csv(dest/'trades.csv',index=False);dec.to_csv(dest/'decisions.csv',index=False)
            row.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,**scores(nav,cfg.capital),**checks)
        except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
        row['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;records.append(row)
    (folder/'status.json').write_text(json.dumps(records,indent=2)+'\n');print(f'Completed {d["label"]}',flush=True)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['tight','no_beta_cutoff'],required=True)
    variant=parser.parse_args().variant
    output=OUTPUT/variant;report=REPORT/variant
    output.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    defs=definitions();(report/'universe.json').write_text(json.dumps(defs,indent=2)+'\n')
    sources=dict(cache=digest(CACHE/'manifest.json'),surface=digest('data/contract_surface/manifest.json'),vix=digest('data/raw/VIX_History.csv'))
    code=[__file__,'scripts/run_aligned_contract_learning.py','scripts/run_contract_walkforward.py','scripts/contract_engine.py',
        'scripts/batched_contract_labels.py','scripts/contract_learning_labels.py','scripts/train_expiry_contracts.py',
        'scripts/test_contract_entry_gates.py','scripts/run_contract_surface_features.py','scripts/run_rv_vix_pass.py']
    fingerprint=dict(settings=asdict(settings()),definitions=defs,sources=sources,filters=FILTERS,
        variant=variant,evaluation_start='2019-01-01',
        beta_screen=dict(enabled=variant=='tight',minimum=.85,maximum=1.15,window=126,correlation_floor=.5),
        code={p:digest(p) for p in code},
        assets={str(p):digest(p) for d in defs for p in asset_paths(d)})
    meta=output/'inputs.json'
    if meta.exists():assert json.loads(meta.read_text())==fingerprint,'Resume inputs changed'
    meta.write_text(json.dumps(fingerprint,indent=2)+'\n')
    records=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[(d,dict(sources,**{str(p):fingerprint['assets'][str(p)] for p in asset_paths(d)}),variant) for d in defs]
        for i,result in enumerate(pool.map(worker,tasks),1):
            records.extend(result);pd.DataFrame(records).to_csv(report/'status.csv',index=False)
            print(f'Finished {i}/{len(defs)} pairs',flush=True)
    (report/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,workers=2,
        max_worker_peak_rss_bytes=max(r['worker_peak_rss_bytes'] for r in records),inputs_sha256=digest(meta)),indent=2)+'\n')


if __name__=='__main__':main()
