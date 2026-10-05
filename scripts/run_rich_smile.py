"""Two predeclared smile-state additions; fixed 30D contracts and native control."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.build_rich_smile import OUTPUT as DATA, VALUES
from scripts.run_tenor_comparison import (OUTPUT as BASE, LABELBASE, ROOT, COMMON_RUNOFF, config,
    train_tenor, FILTERS, definitions, NEW_FEATURES, EXTRA, FEATURES)
from scripts.run_broad_contract_replay import Market
from scripts.select_atm_contracts import load_calendar
from scripts.contract_engine import PreparedMarket,simulate,UnresolvedPosition
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_contract_walkforward import digest

OUTPUT=Path('artifacts/contracts/rich_smile')
REPORT=Path('reports/rich_smile')
VARIANTS=['smile_levels','smile_levels_changes']
_market=None

@lru_cache(maxsize=4)
def asset(sid):
    p=DATA/'assets'/f'{sid}.parquet'
    return pd.read_parquet(p,columns=VALUES) if p.exists() else pd.DataFrame(columns=VALUES,index=pd.DatetimeIndex([]))

def augment_smile(panel,frames,calendar):
    additions={};levels=[];changes=[]
    for horizon in [0,5,21]:
        for value in VALUES:
            vectors=[]
            for side,frame in zip(['a','b'],frames):
                series=frame[value].reindex(calendar).replace([np.inf,-np.inf],np.nan)
                if horizon:series=series-series.shift(horizon)
                vectors.append(series.reindex(panel.index))
            stem=f'smile_{value}'+(f'_change{horizon}' if horizon else '')
            for side,series in [('a',vectors[0]),('b',vectors[1]),('difference',vectors[0]-vectors[1])]:
                name=f'{stem}_{side}'
                additions[name]=series.fillna(0.);additions[name+'_available']=series.notna().astype(float)
                (changes if horizon else levels).extend([name,name+'_available'])
    return pd.concat([panel,pd.DataFrame(additions,index=panel.index)],axis=1),levels,changes

def worker(d):
    global _market
    rows=[];todo=[]
    for variant in VARIANTS:
        dest=OUTPUT/d['pair']/variant;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'status.json').exists():rows.append(json.loads((dest/'status.json').read_text()))
        else:todo.append(variant)
    if not todo:return rows
    source=BASE/d['pair']/'30d';prior=json.loads((source/'status.json').read_text());assert prior['status']=='complete'
    if prior.get('structural_cash'):
        for variant in todo:
            row=dict(pair=d['pair'],label=d['label'],variant=variant,status='complete',entries=0,structural_cash=True,nav_source=prior['nav_source'])
            (OUTPUT/d['pair']/variant/'status.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        return rows
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    if _market is None:_market=Market(cal)
    cfg=config(30);q,s,selected,divs=_market.pair(d['secids'])
    original=pd.read_parquet(source/'states.parquet')
    panel,levels,changes=augment_smile(original,[asset(sid) for sid in d['secids']],cal)
    pd.testing.assert_frame_equal(original,panel[original.columns])
    saved=pd.read_parquet(LABELBASE/d['pair']/'states.parquet')
    with np.load(LABELBASE/d['pair']/'labels.npz',allow_pickle=False) as z:
        idx=saved.index.get_indexer(panel.index);assert (idx>=0).all()
        daily=z['daily'][idx];ends=pd.DatetimeIndex(z['ends'][idx])
    features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES+['vix_previous_close']
    for variant in todo:
        start=time.monotonic();folder=OUTPUT/d['pair']/variant
        extra=levels+(changes if variant=='smile_levels_changes' else [])
        actions,fits=train_tenor(d,panel,daily,ends,cal,q,s,selected,divs,cfg,_market.rates,
            pd.Timestamp('2018-01-01'),cal[-1]-pd.Timedelta(days=COMMON_RUNOFF),folder,
            feature_columns=features+extra,memory_filters=FILTERS,recent_refit=True)
        actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
        row=dict(pair=d['pair'],label=d['label'],variant=variant,features=len(features+extra),nonzero_signals=int(actions.rl_sized.ne(0).sum()))
        try:
            nav,t,dec=simulate(q,s,actions.rl_sized,cal[cal>='2018-01-01'],d['secids'],cfg,divs,selected,PreparedMarket(q,s,selected))
            mark_book(nav,_market.rates,cfg.capital);checks=audit(nav,t,s,cfg,d['secids'])
            nav.to_parquet(folder/'daily_nav.parquet');t.to_csv(folder/'trades.csv',index=False);dec.to_csv(folder/'decisions.csv',index=False)
            row.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,**scores(nav,cfg.capital),**checks)
        except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
        row['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (folder/'status.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(f'{d["label"]} {variant}: {row["status"]}, {row.get("entries",0)} entries, {time.monotonic()-start:.1f}s',flush=True)
    return rows

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);defs=definitions()
    previous=json.loads((BASE/'inputs.json').read_text())
    for group in ['code','sources']:
        for p,h in previous[group].items():assert digest(p)==h,p
    code=[__file__,'scripts/build_rich_smile.py','scripts/run_tenor_comparison.py']
    inputs=dict(variants=VARIANTS,base=digest(BASE/'inputs.json'),smile=digest(DATA/'manifest.json'),
        code={p:digest(p) for p in code},reference=digest('config/current_contract_reference.json'),
        base_files={str(p):digest(p) for d in defs for p in (BASE/d['pair']/'30d').glob('*') if p.is_file()})
    p=OUTPUT/'inputs.json'
    if p.exists():assert json.loads(p.read_text())==inputs,'Run inputs changed'
    else:p.write_text(json.dumps(inputs,indent=2)+'\n')
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(worker,defs),1):
            rows.extend(result);pd.DataFrame(rows).to_csv(REPORT/'status.csv',index=False)
            print(f'{i}/{len(defs)} pairs finished',flush=True)
    (REPORT/'manifest.json').write_text(json.dumps(dict(pairs=len(defs),elapsed_seconds=time.monotonic()-start,workers=2),indent=2)+'\n')

if __name__=='__main__':main()
