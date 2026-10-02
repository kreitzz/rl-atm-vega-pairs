"""Monthly tenor comparison: same reference RL state, memory gate and portfolio engine."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.prepare_tenor_comparison import OUTPUT as DATA, TENORS, config
from scripts.run_option_flex import BASE, LABELBASE, definitions
from scripts.run_aligned_contract_learning import (contract_states, asset_paths, load_labels, mark_book, screen_dates,
    completed_training_mask, validate_policy, NEW_FEATURES, EXTRA)
from scripts.run_contract_surface_features import augment, FEATURES
from scripts.run_contract_walkforward import digest, fit_nn, fit_rl, predict
from scripts.contract_learning_labels import profitability_memory, beta_screen
from scripts.contract_engine import PreparedMarket, simulate, UnresolvedPosition
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_selected_pair_backtest import audit, scores
from scripts.run_rv_vix_pass import join_vix

OUTPUT=Path('artifacts/contracts/tenor_comparison')
REPORT=Path('reports/tenor_comparison')
COMMON_RUNOFF=60
MODELS=['rl_sized']
FILTERS={'rl_sized':'memory_both'}
_market=None

class TenorMarket(Market):
    @lru_cache(maxsize=4)
    def tenor_asset(self,sid):
        paths=[DATA/'history'/str(y)/f'{sid}.parquet' for y in sorted(set(self.calendar.year))]
        pieces=[pd.read_parquet(p) for p in paths if p.exists()]
        return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['date','optionid','secid'])

    @lru_cache(maxsize=8)
    def tenor_selection(self,sid,tenor):
        paths=[DATA/f'{tenor}d'/str(y)/f'{sid}_selection.parquet' for y in sorted(set(self.calendar.year))]
        pieces=[pd.read_parquet(p) for p in paths if p.exists()]
        return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['date','optionid','secid','expiry','strike','bs_vega'])

    def for_tenor(self,secids,tenor):
        if tenor==30:return self.pair(secids)
        q=pd.concat([self.tenor_asset(sid) for sid in secids],ignore_index=True)
        selected=pd.concat([self.tenor_selection(sid,tenor) for sid in secids],ignore_index=True)
        stocks=[]
        for sid in secids:
            s=self.stocks[self.stocks.secid.eq(sid)].set_index('date').reindex(self.calendar)
            s['secid']=sid;s['rate']=self.rates;s.index.name='date';stocks.append(s.reset_index())
        return q,pd.concat(stocks,ignore_index=True),selected,self.divs[self.divs.secid.isin(secids)]


def train_tenor(definition,panel,daily,ends,calendar,q,stocks,selection,divs,cfg,rates,start,stop,folder,
               feature_columns=None,memory_filters=None,recent_refit=False):
    wanted=(panel.index>=start)&(panel.index<=stop)&screen_dates(definition,panel.index)
    actions=pd.DataFrame(0.,index=panel.index[wanted],columns=MODELS)
    memory=[c for c in panel if c.startswith(('profit_','count_'))]
    features=NEW_FEATURES+EXTRA+memory if feature_columns is None else feature_columns
    prepared=PreparedMarket(q,stocks,selection)
    rows=[]
    for month in pd.period_range(start,stop,freq='M'):
        cutoff=month.start_time
        test=(actions.index>=cutoff)&(actions.index<(month+1).start_time)
        if not test.any():continue
        validation_start=cutoff-pd.DateOffset(years=1)
        train=completed_training_mask(panel,ends,daily,validation_start)
        val_cal=calendar[(calendar>=validation_start)&(calendar<cutoff)]
        if train.sum()<504 or len(val_cal)<200:
            rows.append(dict(fit=str(cutoff.date()),model='both',status='insufficient_training',training_rows=int(train.sum())))
            continue
        # All validation positions close before the current prediction month.
        runoff=COMMON_RUNOFF
        val=(panel.index>=val_cal[0])&(panel.index<=val_cal[-1]-pd.Timedelta(days=runoff))
        if not val.any():continue
        assert ends[train].max()<validation_start and val_cal[-1]<cutoff
        x=panel.loc[train,features]
        # Beta eligibility is the same in training proposals, validation and deployment.
        labels=daily[train].copy();labels[~panel.loc[train,'eligible'].to_numpy(bool)]=0
        positions=calendar.get_indexer(panel.index[train])
        end_positions=calendar.get_indexer(ends[train])
        for name in MODELS:
            neural=name=='nn_fixed'
            if memory_filters:
                from scripts.test_contract_entry_gates import gate_actions
                eligibility=panel.loc[val,'eligible'].astype(float)
                possible=any(gate_actions(eligibility*direction,panel,memory_filters[name]).ne(0).any()
                             for direction in [-1,1])
                if not possible:
                    rows.append(dict(fit=str(cutoff.date()),model=name,status='flat_no_validation_opportunities',
                        training_rows=int(train.sum()),last_training_outcome=str(ends[train].max().date()),
                        validation_start=str(val_cal[0].date()),validation_end=str(val_cal[-1].date()),
                        validation_sharpe=None,validation_error=None))
                    continue
            model,details=fit_nn(x,labels) if neural else fit_rl(x,labels,positions,end_positions)
            proposal=predict(model,panel.loc[val,features],neural)*panel.loc[val,'eligible'].to_numpy()
            proposal=pd.Series(proposal,index=panel.index[val])
            if memory_filters:
                from scripts.test_contract_entry_gates import gate_actions
                proposal=gate_actions(proposal,panel,memory_filters[name])
            permitted,score,error=validate_policy(proposal,q,stocks,selection,divs,
                                                 definition['secids'],val_cal,cfg,rates,prepared)
            dates=actions.index[test]
            if permitted:
                if recent_refit:
                    use=(panel.index>=cutoff-pd.DateOffset(years=5))&(ends<cutoff)&np.isfinite(daily).all(axis=(1,2))
                    if use.sum()<504:
                        rows.append(dict(fit=str(cutoff.date()),model=name,status='insufficient_recent_training',
                            training_rows=int(train.sum()),recent_training_rows=int(use.sum()),validation_sharpe=score))
                        continue
                    assert ends[use].max()<cutoff
                    fresh=daily[use].copy();fresh[~panel.loc[use,'eligible'].to_numpy(bool)]=0
                    model,refit_details=(fit_nn(panel.loc[use,features],fresh) if neural else
                        fit_rl(panel.loc[use,features],fresh,calendar.get_indexer(panel.index[use]),calendar.get_indexer(ends[use])))
                    details.update(recent_training_rows=int(use.sum()),last_recent_outcome=str(ends[use].max().date()),
                                   **{'recent_'+k:v for k,v in refit_details.items()})
                deployed=pd.Series(predict(model,panel.loc[dates,features],neural)*panel.loc[dates,'eligible'].to_numpy(),index=dates)
                if memory_filters:deployed=gate_actions(deployed,panel,memory_filters[name])
                actions.loc[dates,name]=deployed
            rows.append(dict(fit=str(cutoff.date()),model=name,status='enabled' if permitted else 'flat_validation_gate',
                             training_rows=int(train.sum()),last_training_outcome=str(ends[train].max().date()),
                             validation_start=str(val_cal[0].date()),validation_end=str(val_cal[-1].date()),
                             validation_sharpe=score,validation_error=error,**details))
        pd.DataFrame(rows).to_csv(folder/'fits.csv',index=False)
        actions.to_csv(folder/'actions.partial.csv')
        if month.month in [1,7]:print(f"  {definition['label']} fit {month}",flush=True)
    assert np.isin(actions.rl_sized,np.arange(-1,1.01,.25)).all()
    return actions,pd.DataFrame(rows)


def worker(d):
    global _market
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    stop=cal[-1]-pd.Timedelta(days=COMMON_RUNOFF)
    returns=pd.concat([pd.read_parquet(p,columns=['ret']).ret.rename(i) for i,p in enumerate(asset_paths(d))],axis=1)
    screen=beta_screen(returns,cal,beta_tolerance=.15)
    wanted=(cal>='2018-01-01')&(cal<=stop)&screen_dates(d,cal)
    structural=not screen.loc[wanted,'eligible'].any()
    results=[]
    for tenor in TENORS:
        folder=OUTPUT/d['pair']/f'{tenor}d';folder.mkdir(parents=True,exist_ok=True)
        status=folder/'status.json'
        if status.exists():results.append(json.loads(status.read_text()));continue
        row=dict(pair=d['pair'],label=d['label'],tenor=tenor)
        if structural:
            nav=pd.read_parquet(BASE/d['pair']/'rl_sized/daily_nav.parquet')
            np.testing.assert_allclose(nav.nav,nav.cash_benchmark,rtol=0,atol=1e-7)
            row.update(status='complete',entries=0,structural_cash=True,nav_source=str(BASE/d['pair']/'rl_sized/daily_nav.parquet'))
            status.write_text(json.dumps(row,indent=2)+'\n');results.append(row);continue
        if _market is None:_market=TenorMarket(cal)
        started=time.monotonic();cfg=config(tenor)
        q,s,selected,divs=_market.for_tenor(d['secids'],tenor)
        if tenor==30:
            panel=pd.read_parquet(BASE/d['pair']/'states.parquet')
            with np.load(LABELBASE/d['pair']/'labels.npz',allow_pickle=False) as z:
                daily=z['daily'];ends=pd.DatetimeIndex(z['ends'])
            assert len(panel)==len(daily)
            keep=panel.index<=stop;panel=panel.loc[keep].copy();daily=daily[keep];ends=ends[keep]
            # Cached native calls labels: no arbitrary matched-put availability filter.
        else:
            panel,own_screen=contract_states(d,selected,s,cal,cfg,beta_tolerance=.15)
            panel=panel[panel.index<=stop]
            panel['commission_vol_entry']=panel.pop('commission_vol_roundtrip')/2
            panel=augment(d,panel,cal)
            sources=dict(tenor_manifest=digest(DATA/'manifest.json'),**{str(p):digest(p) for p in asset_paths(d)})
            daily,ends,vegas,statuses=load_labels(d,panel,q,s,selected,cal,divs,cfg,folder,sources)
            ends=pd.DatetimeIndex(ends)
            rewards=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
                out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
            panel=panel.join(profitability_memory(panel.index,ends,rewards,vegas[:,[0,-1]]>0))
            panel=join_vix(panel,pd.read_csv('data/raw/VIX_History.csv',parse_dates=['DATE']))
            assert panel.vix_previous_close.notna().all() and (panel.vix_date<panel.index).all()
            pd.Series(statuses).value_counts().rename_axis('status').to_csv(folder/'label_status.csv')
        panel.to_parquet(folder/'states.parquet')
        features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
        features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES+['vix_previous_close']
        print(f'{d["label"]} {tenor}d: {len(panel)} state rows; training',flush=True)
        actions,fits=train_tenor(d,panel,daily,ends,cal,q,s,selected,divs,cfg,_market.rates,
            pd.Timestamp('2018-01-01'),stop,folder,feature_columns=features,memory_filters=FILTERS,recent_refit=True)
        actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
        row.update(state_rows=len(panel),nonzero_signals=int(actions.rl_sized.ne(0).sum()))
        try:
            nav,trades,dec=simulate(q,s,actions.rl_sized,cal[cal>='2018-01-01'],d['secids'],cfg,divs,selected,PreparedMarket(q,s,selected))
            mark_book(nav,_market.rates,cfg.capital);checks=audit(nav,trades,s,cfg,d['secids'])
            assert trades.empty or set(trades.reason).issubset({'entry','split_adjustment','expiry_settlement'})
            nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);dec.to_csv(folder/'decisions.csv',index=False)
            row.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,**scores(nav,cfg.capital),**checks)
        except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
        row['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        status.write_text(json.dumps(row,indent=2)+'\n');results.append(row)
        print(f'{d["label"]} {tenor}d: {row["status"]}, {row.get("entries",0)} entries, {time.monotonic()-started:.1f}s',flush=True)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    defs=definitions()
    code=[__file__,'scripts/prepare_tenor_comparison.py','scripts/run_aligned_contract_learning.py',
        'scripts/contract_engine.py','scripts/contract_learning_labels.py','scripts/batched_contract_labels.py',
        'scripts/run_contract_walkforward.py','scripts/run_contract_surface_features.py','scripts/run_nn_rl_benchmark.py',
        'scripts/test_contract_entry_gates.py','scripts/contract_risk.py','scripts/run_rv_vix_pass.py']
    inputs=dict(definitions=defs,settings={t:asdict(config(t)) for t in TENORS},common_runoff=COMMON_RUNOFF,
        code={p:digest(p) for p in code},tenor_manifest=digest(DATA/'manifest.json'),reference=digest('config/current_contract_reference.json'),
        sources={str(p):digest(p) for p in [Path('data/contract_history_shared/manifest.json'),Path('data/contract_surface/manifest.json'),Path('data/raw/VIX_History.csv'),
            *{p for d in defs for p in asset_paths(d)},*[p for d in defs for p in [BASE/d['pair']/'states.parquet',LABELBASE/d['pair']/'labels.npz']]]})
    path=OUTPUT/'inputs.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(inputs)),'Inputs changed'
    else:path.write_text(json.dumps(inputs,indent=2)+'\n')
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(worker,defs[:args.limit]),1):
            rows.extend(result);pd.DataFrame(rows).to_csv(REPORT/'status.csv',index=False)
            print(f'{i}/{len(defs)} pairs finished',flush=True)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,pairs=len(rows)//3,workers=2),indent=2)+'\n')

if __name__=='__main__':main()
