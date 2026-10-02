"""Retrain the full 0.226 reference roster with call/put choice on either leg."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.download_matched_puts import definitions
from scripts.option_flex.market import FlexMarket
from scripts.option_flex.policy import ACTIONS,COMBOS,FLAT,fit,predict,gate,Replay
from scripts.option_flex.contract_learning_labels import ShadowLabels,profitability_memory,LADDER
from scripts.option_flex.batched_contract_labels import build_labels
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.run_aligned_contract_learning import NEW_FEATURES,EXTRA,screen_dates,mark_book
from scripts.run_contract_surface_features import FEATURES
from scripts.run_broad_contract_replay import ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.train_expiry_contracts import settings
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_contract_walkforward import digest

BASE=Path('artifacts/contracts/expiry_vix_momentum/vix')
LABELBASE=Path('artifacts/contracts/expiry_trained_beta_085_115')
OUTPUT=Path('artifacts/contracts/option_flex')
REPORT=Path('reports/option_flex')
_market=None


def labels_and_states(d,panel,q,s,selections,div,cal,cfg,folder):
    saved=folder/'labels.npz'
    if saved.exists():
        with np.load(saved,allow_pickle=False) as z:daily=z['daily'];ends=pd.DatetimeIndex(z['ends'])
        return pd.read_parquet(folder/'states.parquet'),daily,ends
    parts={};ends={};all_status={};label_details={}
    for combo in COMBOS:
        started=time.monotonic()
        selected=selections[combo]
        panel[f'{combo}_available']=panel.index.isin(selected.groupby('date').secid.nunique().loc[lambda x:x.eq(2)].index)
        if combo=='CC':
            with np.load(LABELBASE/d['pair']/'labels.npz',allow_pickle=False) as z:
                data=z['daily'];end=pd.DatetimeIndex(z['ends']);vegas=z['vegas'];status=z['statuses']
            assert len(data)==len(panel)
        else:
            # Each trajectory uses only its selected pair of option types.
            ids=set(selected.optionid);own=q[q.optionid.isin(ids)]
            maker=ShadowLabels(own,s,selected,cal,d['secids'],cfg,div)
            data,end_index,vegas,status=build_labels(maker,cal.get_indexer(panel.index),batch_size=128)
            end=cal[end_index]
        reward=np.divide(data[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
            out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
        memory=profitability_memory(panel.index,end,reward,vegas[:,[0,-1]]>0).add_prefix(combo+'_')
        panel=panel.join(memory);parts[combo]=data;ends[combo]=end.values;all_status[combo]=pd.Series(status).value_counts().to_dict()
        label_details.update({combo+'_ends':end.values,combo+'_vegas':vegas,combo+'_statuses':status})
        print(f'{d["label"]}: {combo} labels ready in {time.monotonic()-started:.1f}s',flush=True)
    daily=np.stack([parts[c][:,int(np.flatnonzero(np.isclose(LADDER,a))[0])] for c,a in ACTIONS],axis=1)
    common_end=pd.DatetimeIndex(np.stack(list(ends.values())).max(axis=0))
    panel.to_parquet(folder/'states.parquet');np.savez_compressed(saved,daily=daily,ends=common_end.values,**label_details)
    (folder/'label_status.json').write_text(json.dumps(all_status,indent=2,default=int)+'\n')
    return panel,daily,common_end


def train(d,panel,daily,ends,cal,replay,rates,folder,cfg):
    stop=cal[-1]-pd.Timedelta(days=cfg.max_dte+5);start=pd.Timestamp('2018-01-01')
    wanted=(panel.index>=start)&(panel.index<=stop)&screen_dates(d,panel.index)
    actions=pd.Series(FLAT,index=panel.index[wanted],dtype=int);logs=[]
    features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES+['vix_previous_close']
    finite=np.isfinite(daily).all(axis=(1,2))
    for month in pd.period_range(start,stop,freq='M'):
        if month.month==1:print(f'{d["label"]}: monthly training {month.year}',flush=True)
        cutoff=month.start_time;test=(actions.index>=cutoff)&(actions.index<(month+1).start_time)
        if not test.any():continue
        vs=cutoff-pd.DateOffset(years=1);vc=cal[(cal>=vs)&(cal<cutoff)]
        tr=(panel.index>=vs-pd.DateOffset(years=5))&(ends<vs)&finite
        row=dict(fit=str(cutoff.date()),model='rl_flex',training_rows=int(tr.sum()))
        if tr.sum()<504 or len(vc)<200:logs.append(dict(row,status='insufficient_training'));continue
        va=(panel.index>=vc[0])&(panel.index<=vc[-1]-pd.Timedelta(days=cfg.max_dte+5))
        if not va.any():continue
        potential=any(gate(pd.Series(k,index=panel.index[va]),panel).ne(FLAT).any() for k in range(len(ACTIONS)) if k!=FLAT)
        if not potential:logs.append(dict(row,status='flat_no_validation_opportunities'));continue
        data=daily[tr].copy();data[~panel.loc[tr,'eligible'].to_numpy(bool)]=0
        model,details=fit(panel.loc[tr,features],data,cal.get_indexer(panel.index[tr]),cal.get_indexer(ends[tr]))
        proposal=gate(pd.Series(predict(model,panel.loc[va,features]),index=panel.index[va]),panel)
        error=None;score=None
        try:
            nav,t,_=replay.run(proposal,vc);mark_book(nav,rates,cfg.capital);audit(nav,t,replay.s,cfg,d['secids'])
            score=scores(nav,cfg.capital)['annualized_excess_sharpe']
        except (UnresolvedPosition,ValueError) as exc:error=str(exc)
        permitted=score is not None and score>0
        if permitted:
            recent=(panel.index>=cutoff-pd.DateOffset(years=5))&(ends<cutoff)&finite
            assert recent.sum()>=504 and ends[recent].max()<cutoff
            data=daily[recent].copy();data[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
            model,detail=fit(panel.loc[recent,features],data,cal.get_indexer(panel.index[recent]),cal.get_indexer(ends[recent]))
            idx=actions.index[test];actions.loc[idx]=gate(pd.Series(predict(model,panel.loc[idx,features]),index=idx),panel)
            details.update(last_recent_outcome=str(ends[recent].max().date()),recent_training_rows=int(recent.sum()))
        logs.append(dict(row,status='enabled' if permitted else 'flat_validation_gate',validation_sharpe=score,validation_error=error,
            last_training_outcome=str(ends[tr].max().date()),validation_start=str(vc[0].date()),**details))
        pd.DataFrame(logs).to_csv(folder/'fits.csv',index=False);actions.to_csv(folder/'actions.partial.csv')
    return actions,pd.DataFrame(logs)


def worker(d):
    global _market
    folder=OUTPUT/d['pair'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists():return json.loads((folder/'status.json').read_text())
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')];cfg=settings()
    panel=pd.read_parquet(BASE/d['pair']/'states.parquet')
    wanted=(panel.index>='2018-01-01')&screen_dates(d,panel.index)
    if not panel.loc[wanted,'eligible'].any():
        # A computational shortcut only: these capital sleeves stay in the universe.
        # No option-type choice can bypass unchanged beta/correlation eligibility.
        nav=pd.read_parquet(BASE/d['pair']/'rl_sized/daily_nav.parquet')
        np.testing.assert_allclose(nav.nav,nav.cash_benchmark,rtol=0,atol=1e-7)
        nav.to_parquet(folder/'daily_nav.parquet');panel.to_parquet(folder/'states.parquet')
        actions=pd.Series(FLAT,index=panel.index[wanted],name='action_code');actions.to_csv(folder/'actions.csv')
        pd.DataFrame(dict(signal_date=actions.index,status='flat',action=0.)).to_csv(folder/'decisions.csv',index=False)
        pd.DataFrame(columns=['date','cohort','secid','optionid','quantity','price','commission','execution_penalty','reason','cp_flag']).to_csv(folder/'trades.csv',index=False)
        pd.DataFrame([dict(model='rl_flex',status='structurally_flat')]).to_csv(folder/'fits.csv',index=False)
        result=dict(pair=d['pair'],label=d['label'],calls_control='verified_cash_only',nonzero_signals=0,
            status='complete',entries=0,**scores(nav,cfg.capital),worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (folder/'status.json').write_text(json.dumps(result,indent=2)+'\n');return result
    if _market is None:_market=FlexMarket(cal)
    print(f'{d["label"]}: loading matched calls and puts',flush=True)
    q,s,selections,div=_market.flex_pair(d['secids']);replay=Replay(q,s,selections,d['secids'],div,cfg)
    # Replay the frozen calls-only actions with this engine as a regression control.
    original=pd.read_csv(BASE/d['pair']/'actions.csv',index_col=0,parse_dates=[0]).rl_sized
    control=pd.Series([ACTIONS.index(('CC',float(a))) for a in original],index=original.index)
    nav,t,_=replay.run(control,cal[cal>='2018-01-01']);mark_book(nav,_market.rates,cfg.capital)
    expected=pd.read_parquet(BASE/d['pair']/'rl_sized/daily_nav.parquet')
    np.testing.assert_allclose(nav.nav,expected.nav,rtol=0,atol=1e-7)
    panel,daily,ends=labels_and_states(d,panel,q,s,selections,div,cal,cfg,folder)
    print(f'{d["label"]}: {len(panel)} rows, 33 actions',flush=True)
    actions,fits=train(d,panel,daily,ends,cal,replay,_market.rates,folder,cfg)
    actions.rename('action_code').to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
    result=dict(pair=d['pair'],label=d['label'],calls_control='identical',nonzero_signals=int(actions.ne(FLAT).sum()))
    try:
        nav,t,dec=replay.run(actions,cal[cal>='2018-01-01']);mark_book(nav,_market.rates,cfg.capital)
        checks=audit(nav,t,s,cfg,d['secids'])
        if len(t):
            cp=q[['optionid','cp_flag']].drop_duplicates();assert not cp.optionid.duplicated().any();t=t.merge(cp,on='optionid',validate='many_to_one')
        nav.to_parquet(folder/'daily_nav.parquet');t.to_csv(folder/'trades.csv',index=False);dec.to_csv(folder/'decisions.csv',index=False)
        result.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,**scores(nav,cfg.capital),**checks)
    except (UnresolvedPosition,ValueError) as exc:result.update(status='incomplete',error=str(exc))
    result['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (folder/'status.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path('data/call_put/manifest.json').read_text());assert manifest['completed']==manifest['requested']
    defs=definitions();cfg=settings()
    from dataclasses import asdict
    fingerprint=dict(definitions=defs,settings=asdict(cfg),actions=ACTIONS,put_manifest=digest('data/call_put/manifest.json'),
        reference=digest('config/current_contract_reference.json'),
        call_cache=digest('data/contract_history_shared/manifest.json'),
        reference_files={str(p):digest(p) for d in defs for p in [BASE/d['pair']/'states.parquet',BASE/d['pair']/'actions.csv',
            BASE/d['pair']/'rl_sized/daily_nav.parquet',LABELBASE/d['pair']/'labels.npz']},
        code={str(p):digest(p) for p in [Path(__file__),*Path('scripts/option_flex').glob('*.py'),
            Path('scripts/run_aligned_contract_learning.py'),Path('scripts/run_broad_contract_replay.py'),
            Path('scripts/train_expiry_contracts.py'),Path('scripts/run_selected_pair_backtest.py')]})
    path=OUTPUT/'inputs.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(fingerprint)),'Inputs changed'
    path.write_text(json.dumps(fingerprint,indent=2)+'\n');start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(worker,defs[:args.limit]),1):
            records.append(row);pd.DataFrame(records).to_csv(REPORT/'status.csv',index=False)
            print(f'{i}/{len(defs)} {row["label"]}: {row["status"]}',flush=True)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-start,pairs=len(records),workers=2),indent=2)+'\n')


if __name__=='__main__':main()
