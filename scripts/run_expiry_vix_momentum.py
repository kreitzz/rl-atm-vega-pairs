"""VIX, 12–1 total-return momentum and combined expiry-policy ablations."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.build_short_rv_momentum import momentum_12_1
from scripts.run_rv_vix_pass import join_vix
from scripts.run_aligned_contract_learning import asset_paths,train_pair,mark_book,NEW_FEATURES,EXTRA,MODELS
from scripts.run_contract_surface_features import FEATURES
from scripts.train_expiry_contracts import settings,FILTERS
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.contract_engine import PreparedMarket,simulate,UnresolvedPosition
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar

BASE=Path('artifacts/contracts/expiry_trained_beta_085_115')
OUTPUT=Path('artifacts/contracts/expiry_vix_momentum')
REPORT=Path('reports/expiry_vix_momentum')
RAW=Path('/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds/crsp_underlying_daily')
MOMENTUM=['momentum_12_1_a','momentum_12_1_b','momentum_12_1_a_available','momentum_12_1_b_available']
VARIANTS={'vix':['vix_previous_close'],'momentum':MOMENTUM,'both':['vix_previous_close']+MOMENTUM}
_market=None


def add_states(panel,vix,momentum):
    result=join_vix(panel,vix)
    assert result.index.equals(panel.index)
    result.index=panel.index.copy()
    assert result.vix_previous_close.notna().all()
    assert (result.vix_date<result.index).all()
    for side,series in zip(['a','b'],momentum):
        name=f'momentum_12_1_{side}'
        value=series.reindex(result.index).replace([np.inf,-np.inf],np.nan)
        result[name]=value.fillna(0.)
        result[name+'_available']=value.notna().astype(float)
    return result


def prepare_momentum(definitions):
    paths=sorted({p for d in definitions for p in asset_paths(d)})
    permnos=sorted({int(p.stem.split('p')[1]) for p in paths})
    files=[p for p in sorted(RAW.glob('year=*.parquet')) if int(p.stem.split('=')[1])<=2025]
    frames=[pd.read_parquet(p,columns=['date','permno','ret'],filters=[('permno','in',permnos)]) for p in files]
    returns=pd.concat(frames,ignore_index=True).drop_duplicates()
    assert not returns.duplicated(['date','permno']).any()
    wide=returns.pivot(index='date',columns='permno',values='ret').sort_index()
    calendar=pd.DatetimeIndex(pd.read_csv('artifacts/broad_pairs/calendar.csv',parse_dates=['date']).date)
    wide=wide.reindex(calendar)
    meta={r['asset']:r for r in json.loads(Path('artifacts/broad_pairs/assets.json').read_text())}
    dest=OUTPUT/'momentum';dest.mkdir(parents=True,exist_ok=True)
    for path in paths:
        item=meta[path.stem];r=wide[int(item['permno'])].copy()
        valid=pd.Series(False,index=calendar)
        for start,end in item['intervals']:valid|=(calendar>=start)&(calendar<=end)
        momentum_12_1(r.where(valid)).rename('momentum').to_frame().to_parquet(dest/path.name)
    return {str(p):digest(p) for p in files}


def worker(definition):
    global _market
    pair=definition['pair'];source=BASE/pair
    if all((OUTPUT/v/pair/'status.json').exists() for v in VARIANTS):
        return [r for v in VARIANTS for r in json.loads((OUTPUT/v/pair/'status.json').read_text())]
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    if _market is None:_market=Market(cal)
    cfg=settings();evaluation=cal[cal>='2018-01-01'];stop=cal[-1]-pd.Timedelta(days=cfg.max_dte+5)
    q,s,selection,divs=_market.pair(definition['secids']);prepared=PreparedMarket(q,s,selection)
    baseline=pd.read_parquet(source/'states.parquet')
    vix=pd.read_csv('data/raw/VIX_History.csv',parse_dates=['DATE'])
    mom=[pd.read_parquet(OUTPUT/'momentum'/p.name).momentum for p in asset_paths(definition)]
    panel=add_states(baseline,vix,mom)
    pd.testing.assert_frame_equal(panel[baseline.columns],baseline)
    with np.load(source/'labels.npz',allow_pickle=False) as z:
        daily=z['daily'];ends=pd.DatetimeIndex(z['ends'])
    assert len(daily)==len(panel)
    features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES
    records=[]
    for variant,extra in VARIANTS.items():
        folder=OUTPUT/variant/pair;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():
            records.extend(json.loads((folder/'status.json').read_text()));continue
        panel.to_parquet(folder/'states.parquet')
        print(f'{definition["label"]} {variant}: fitting {len(panel)} rows',flush=True)
        actions,fits=train_pair(definition,panel,daily,ends,cal,q,s,selection,divs,cfg,_market.rates,
            pd.Timestamp('2018-01-01'),stop,folder,feature_columns=features+extra,memory_filters=FILTERS,recent_refit=True)
        actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
        own=[]
        for model in MODELS:
            a=actions[model];assert panel.loc[a.index[a.ne(0)],'eligible'].all()
            dest=folder/model;dest.mkdir(exist_ok=True)
            row=dict(pair=pair,label=definition['label'],group=definition['group'],variant=variant,model=model,
                nonzero_signals=int(a.ne(0).sum()))
            try:
                nav,t,dec=simulate(q,s,a,evaluation,definition['secids'],cfg,divs,selection,prepared)
                mark_book(nav,_market.rates,cfg.capital);checks=audit(nav,t,s,cfg,definition['secids'])
                assert t.empty or set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                nav.to_parquet(dest/'daily_nav.parquet');t.to_csv(dest/'trades.csv',index=False);dec.to_csv(dest/'decisions.csv',index=False)
                row.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,**scores(nav,cfg.capital),**checks)
            except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
            row['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            own.append(row)
        (folder/'status.json').write_text(json.dumps(own,indent=2)+'\n');records.extend(own)
        print(f'Completed {definition["label"]} {variant}',flush=True)
    return records


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((BASE/'inputs.json').read_text());definitions=inputs['definitions']
    code=[__file__,'scripts/build_short_rv_momentum.py','scripts/run_rv_vix_pass.py',
        'scripts/run_aligned_contract_learning.py','scripts/run_contract_walkforward.py','scripts/contract_engine.py',
        'scripts/train_expiry_contracts.py','scripts/test_contract_entry_gates.py']
    fingerprint=dict(base_inputs=digest(BASE/'inputs.json'),variants=VARIANTS,settings=inputs['settings'],
        vix=digest('data/raw/VIX_History.csv'),code={p:digest(p) for p in code},
        base_files={str(BASE/d['pair']/f):digest(BASE/d['pair']/f) for d in definitions for f in ['states.parquet','labels.npz']})
    meta=OUTPUT/'inputs.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        assert {k:v for k,v in old.items() if k!='return_files'}==fingerprint,'Resume inputs changed'
        for p,h in old['return_files'].items():assert digest(p)==h
    else:
        fingerprint['return_files']=prepare_momentum(definitions)
        meta.write_text(json.dumps(fingerprint,indent=2)+'\n')
    started=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(worker,definitions),1):
            records.extend(result);pd.DataFrame(records).to_csv(REPORT/'status.csv',index=False)
            print(f'Finished {i}/{len(definitions)} pairs, all three variants',flush=True)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,workers=2,
        max_worker_peak_rss_bytes=max(r['worker_peak_rss_bytes'] for r in records),inputs_sha256=digest(meta)),indent=2)+'\n')


if __name__=='__main__':main()
