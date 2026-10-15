"""Three-input offline contextual sizing bandit for the ATM call experiment.

Actual-contract unit ledgers supply counterfactual action rewards. Cash is
replayed separately by action, including integer counts and nonlinear financing.
Training samples save computation; they never restrict daily funded proposals.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import research_atm_rv_persistence as base

CFG_PATH=Path('config/atm_rv_bandit.json')
CFG=json.loads(CFG_PATH.read_text())
OUT=Path('artifacts/contracts/atm_rv_bandit')
REPORT=Path('reports/atm_rv_bandit')
FEATURES=CFG['features']
ACTIONS=np.array(CFG['actions'],int)
CAPITAL=base.CFG['account_capital']
PATH_LENGTH=36
EXECUTION=['date','pair','wing','secid_a','secid_b','optionid_a','optionid_b','direction','base_level','priority']


def configure(path):
    """Separate experiment roots keep previous ladders and fitted policies intact."""
    global CFG_PATH,CFG,OUT,REPORT,FEATURES,ACTIONS
    CFG_PATH=Path(path);CFG=json.loads(CFG_PATH.read_text())
    OUT=Path(CFG.get('output_root','artifacts/contracts/atm_rv_bandit'))
    REPORT=Path(CFG.get('report_root','reports/atm_rv_bandit'))
    FEATURES=CFG['features'];ACTIONS=np.array(CFG['actions'],int)
    assert len(ACTIONS)==3 and 4 in ACTIONS


def controls():return CFG.get('fixed_controls',['fixed100','fixed50','fixed75'])


def replay_folder(case):
    return Path(CFG.get('reuse_replays',{}).get(case,str(OUT/'replays'/case)))


def dump(path,value):base.dump(path,value)


def progress(phase,**details):
    record=dict(phase=phase,time=time.time(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**details)
    dump(OUT/'progress.json',record)
    print(json.dumps(record,default=str),flush=True)


def events():
    from scripts.run_year_cheap_vega import EXTRA_CORPORATE_EVENTS
    return (json.loads(Path('config/p25_submission.json').read_text())['corporate_events']
            +EXTRA_CORPORATE_EVENTS+base.CFG['corporate_events']+base.CFG['listing_risk_events'])


def with_vix(frame):
    """Signal-close VIX is available before next-session execution."""
    vix=pd.read_csv('data/raw/VIX_History.csv',usecols=['DATE','CLOSE'],parse_dates=['DATE'])
    vix=vix.rename(columns={'DATE':'vix_date','CLOSE':'vix'}).sort_values('vix_date')
    result=pd.merge_asof(frame.sort_values(['date','secid']),vix,left_on='date',right_on='vix_date',
        direction='backward',allow_exact_matches=True,tolerance=pd.Timedelta(days=7))
    assert result.loc[result.vix.notna(),'vix_date'].le(result.loc[result.vix.notna(),'date']).all()
    return result


def choose_fraction(frame,score):
    """Rank only information known at signal close, with deterministic ID ties."""
    fraction=CFG.get('selection_fraction',base.CFG['selection_fraction'])
    if not 0<fraction<=1:raise ValueError('Selection fraction must be in (0,1]')
    pieces=[]
    for _,g in frame.groupby('date',sort=False):
        if len(g)<base.CFG['minimum_cross_section']:continue
        count=max(1,int(np.floor(len(g)*fraction)))
        pieces.append(g.sort_values([score,'secid'],kind='stable').iloc[:count])
    return pd.concat(pieces,ignore_index=True) if pieces else frame.iloc[:0].copy()


def sample():
    """Choose monthly samples using current features only, before reading payoffs."""
    marker=OUT/'sample.parquet'
    if marker.exists() and (OUT/'proposals.parquet').exists() and (REPORT/'protocol.json').exists():return
    cal=base.calendar()
    dates=pd.DatetimeIndex(pd.Series(cal,index=cal).groupby(cal.to_period('M')).first())
    pieces=[]
    for path in sorted((base.OUT/'panel').glob('*.parquet')):
        frame=pd.read_parquet(path,columns=['date','entry_date','expiry','secid','optionid','iv','rv21','residual_memory'],use_threads=False)
        frame=frame.loc[frame.date.isin(dates)&frame.date.le(base.CFG['last_signal'])]
        pieces.append(frame.dropna(subset=['iv','rv21','residual_memory']))
    pool=with_vix(pd.concat(pieces,ignore_index=True)).dropna(subset=FEATURES)
    chosen=choose_fraction(pool,'residual_memory')
    samples=[]
    for date,g in chosen.groupby('date',sort=True):
        g=g.sort_values(['residual_memory','secid'],kind='stable')
        slots=np.unique(np.linspace(0,len(g)-1,min(len(g),CFG['training_sample_per_month'])).round().astype(int))
        samples.append(g.iloc[slots])
    rows=pd.concat(samples,ignore_index=True).sort_values(['secid','date']).reset_index(drop=True)
    rows['sample_id']=np.arange(len(rows))
    rows.to_parquet(marker,index=False)
    # Every sizing policy receives the same daily signal-ranked proposals.
    if CFG.get('selection_fraction',.2)==.2:
        proposals=pd.read_parquet(base.OUT/'residual_memory_proposals.parquet',use_threads=False)
    else:
        pieces=[]
        for path in sorted((base.OUT/'forecast_months').glob('????-??/predictions.parquet')):
            available=pd.read_parquet(path,columns=['date','entry_date','expiry','secid','optionid','iv','rv21','residual_memory'],use_threads=False)
            pieces.append(choose_fraction(available,'residual_memory'))
        proposals=pd.concat(pieces,ignore_index=True)
        proposals['pair']='single_'+proposals.secid.astype(str)
        proposals['secid_a']=proposals.secid;proposals['secid_b']=proposals.secid
        proposals['optionid_a']=proposals.optionid;proposals['optionid_b']=np.nan
        proposals['wing']='ATM_C';proposals['direction']=-1;proposals['base_level']=4
        proposals['priority']=-proposals.residual_memory
        old=pd.read_parquet(base.OUT/'residual_memory_proposals.parquet',columns=['date','secid','optionid'],use_threads=False)
        assert pd.MultiIndex.from_frame(proposals[['date','secid','optionid']]).isin(pd.MultiIndex.from_frame(old)).all()
        del old,pieces,available
    proposals=with_vix(proposals)
    proposals.to_parquet(OUT/'proposals.parquet',index=False)
    dump(REPORT/'protocol.json',dict(specification=CFG,base_specification=base.CFG,
        proposals=len(proposals),sample_trades=len(rows),sample_names=int(rows.secid.nunique()),
        selection_uses_future_outcomes=False,vix='Same signal-close CBOE close, known before next-session fills.',
        costs=base.execution_settings(),universe=f"Original retrospective cached universe; bottom {100*CFG.get('selection_fraction',.2):g}% of daily residual-memory ranks.",
        resources='One worker, 20% of one core, 256 MiB resident/compressed stop; Brave kept open.'))
    progress('sample_complete',samples=len(rows),names=int(rows.secid.nunique()),proposals=len(proposals))


class SmallMarket:
    """One name's sampled contracts only; stream the original quote file."""
    def __init__(self,sid,rows):
        from scripts.option_flex.contract_engine import MarketRow
        self.sid=sid
        self.rates=pd.read_parquet(base.MARKET_SOURCE/'rates.parquet',use_threads=False).rate
        self.stocks=pd.read_parquet(base.SOURCE/f'{sid}_stocks.parquet',columns=['date','close','cfadj'],use_threads=False).set_index('date')
        for repair in base.CFG['observed_stock_repairs']:
            if repair['secid']==sid:
                self.stocks.loc[pd.Timestamp(repair['date']),['close','cfadj']]=[repair['close'],repair['cfadj']]
        self.divs=pd.read_parquet(base.MARKET_SOURCE/'dividends.parquet',filters=[('secid','==',int(sid))],use_threads=False)
        wanted=set(map(int,rows.optionid));self.quotes={}
        for batch in pq.ParquetFile(base.SOURCE/f'{sid}_quotes.parquet').iter_batches(batch_size=4096,columns=base.QUOTE_COLUMNS,use_threads=False):
            frame=batch.to_pandas()
            frame=frame.loc[frame.optionid.isin(wanted)&frame.date.ge(rows.date.min())]
            for record in frame.to_dict('records'):
                self.quotes[(record['date'],int(record['optionid']))]=MarketRow(record)
    def stock(self,date,sid):
        if sid!=self.sid:raise ValueError('Single-name shadow cannot request another stock')
        if date not in self.stocks.index:return np.nan,np.nan
        return tuple(self.stocks.loc[date,['close','cfadj']])
    def quote(self,date,oid):return self.quotes.get((date,int(oid)))
    def rate(self,date):return float(self.rates.at[date])
    def dividend_pv(self,date,sid,expiry,rate):
        div=self.divs.loc[self.divs.declare_date.le(date)&self.divs.ex_date.gt(date)&self.divs.ex_date.le(expiry)]
        return float((div.amount*np.exp(-rate*(div.ex_date-date).dt.days/365)).sum())


def scaled_cash_paths(nav,counts,rates,capital=CAPITAL,borrowing_spread=.01):
    """Exact cash recursion for integer multiples of an independent unit ledger.

    Costs, dividends, hedge turnover, and option flows scale by contract count.
    Interest is recalculated on each action's actual cash, including debit spread.
    Initial stock/option inventories are zero; rates accrue from prior close.
    """
    elapsed=nav.index.to_series().diff().dt.days.fillna(0).to_numpy()/365
    previous_rates=rates.reindex(nav.index).shift(1).fillna(0).to_numpy()
    cash=np.full(len(counts),float(capital));previous_nav=cash.copy();benchmark=float(capital)
    unit_flow=nav.cash.diff().fillna(nav.cash.iloc[0]-capital).to_numpy()-nav.interest.to_numpy()
    inventories=(nav.option_value+nav.stock_value).to_numpy()
    result=np.zeros((len(counts),len(nav)))
    for t in range(len(nav)):
        old_benchmark=benchmark
        benchmark*=1+previous_rates[t]*elapsed[t]
        cash+=cash*(previous_rates[t]+borrowing_spread*(cash<0))*elapsed[t]+counts*unit_flow[t]
        equity=cash+counts*inventories[t]
        result[:,t]=equity-previous_nav-(benchmark-old_benchmark)
        previous_nav=equity
    return result


def labels():
    from scripts import atm_rv_persistence_engine as engine
    from scripts.option_flex.contract_engine import Settings
    cfg=Settings(**base.execution_settings())
    # A one-contract ledger has no competing trades. Funded multi-trade
    # capacity constraints enter the later portfolio replay.
    unit_cfg=cfg
    frame=pd.read_parquet(OUT/'sample.parquet',use_threads=False)
    cal=base.calendar();corporate=events()
    engine.book_risk=base.sparse_book_risk
    total=valid=0
    for sid,rows in frame.groupby('secid',sort=True):
        path=OUT/'labels'/f'{sid}.parquet';array_path=path.with_suffix('.npz')
        if path.exists() and array_path.exists():
            meta=pd.read_parquet(path,columns=['finite_label'],use_threads=False)
            total+=len(meta);valid+=int(meta.finite_label.sum());continue
        market=SmallMarket(int(sid),rows)
        records=[];paths=np.zeros((len(rows),len(ACTIONS),PATH_LENGTH))
        for local,row in enumerate(rows.itertuples()):
            record=dict(sample_id=row.sample_id,secid=int(sid),date=row.date,expiry=row.expiry,
                residual_memory=row.residual_memory,iv=row.iv,vix=row.vix,label_row=local,
                finite_label=False,end=pd.NaT,entry_position=-1,full_profit=np.nan)
            try:
                start=cal.get_loc(row.date);stop=cal.searchsorted(row.expiry,side='right')
                if stop>=len(cal):raise ValueError('Insufficient label runoff')
                trading=cal[start:stop+1]
                proposal=pd.DataFrame([dict(date=row.date,pair=f'single_{sid}',wing='ATM_C',secid_a=sid,
                    secid_b=sid,optionid_a=row.optionid,optionid_b=np.nan,direction=-1,base_level=4)])
                cohort=f'{row.date.date()}|single_{sid}|ATM_C'
                nav,trades,decisions,audit=engine.simulate(market,proposal,trading,unit_cfg,CAPITAL,
                    standalone=True,corporate_events=corporate,max_stale_quote_sessions=2,
                    locked_entry_counts={cohort:(1,)})
                if not len(decisions) or decisions.status.iloc[0]!='entered':
                    record['reason']='no_executable_entry'
                else:
                    first=decisions.entry_date.iloc[0];end=trades.date.max()
                    live=market.quote(first,int(row.optionid));vega=100*float(live.vega)
                    counts=np.floor(base.CFG['entry_vega']*ACTIONS/4/vega).astype(int)
                    reward=scaled_cash_paths(nav,counts,market.rates,borrowing_spread=cfg.borrowing_spread)
                    keep=(nav.index>=first)&(nav.index<=end)
                    reward=reward[:,keep]
                    if reward.shape[1]>PATH_LENGTH:raise ValueError('Shadow path exceeds storage bound')
                    paths[local,:,:reward.shape[1]]=reward
                    record.update(finite_label=True,end=end,entry_position=int(cal.get_loc(first)),
                        full_profit=float(reward[np.flatnonzero(ACTIONS==4)[0]].sum()),counts=counts.tolist(),length=reward.shape[1],reason='ok')
                    assert audit['max_ledger_error']<1e-5
            except Exception as exc:
                # A missing eventual payoff never deletes a daily live proposal.
                # It simply supplies no training reward once it is in the past.
                record['reason']=f'{type(exc).__name__}: {exc}'
            records.append(record)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(array_path,paths=paths)
        pd.DataFrame(records).to_parquet(path,index=False)
        total+=len(records);valid+=sum(r['finite_label'] for r in records)
        progress('labels',names_done=len(list(path.parent.glob('*.npz'))),samples_done=total,valid_labels=valid,secid=int(sid))
        del market,records,paths
        gc.collect()
    meta=pd.concat([pd.read_parquet(p,use_threads=False) for p in sorted((OUT/'labels').glob('*.parquet'))],ignore_index=True)
    meta.to_parquet(OUT/'training.parquet',index=False)
    meta.groupby(['finite_label','reason']).size().rename('count').to_csv(REPORT/'label_coverage.csv')
    reuse=CFG.get('sampling_source')
    if reuse:
        previous=pd.read_parquet(Path(reuse)/'training.parquet',use_threads=False).sort_values('sample_id')
        current=meta.sort_values('sample_id')
        np.testing.assert_array_equal(current.sample_id,previous.sample_id)
        np.testing.assert_array_equal(current.finite_label,previous.finite_label)
        np.testing.assert_allclose(current.full_profit,previous.full_profit,atol=1e-8,rtol=0,equal_nan=True)
        for path in sorted((OUT/'labels').glob('*.npz')):
            with np.load(path) as new_paths, np.load(Path(reuse)/'labels'/path.name) as old_paths:
                np.testing.assert_allclose(new_paths['paths'][:,0],old_paths['paths'][:,0],atol=1e-8,rtol=0)
                np.testing.assert_allclose(new_paths['paths'][:,1],old_paths['paths'][:,2],atol=1e-8,rtol=0)
        dump(REPORT/'training_path_audit.json',dict(samples=len(meta),unchanged_50_and_100_percent_paths=True,
            full_profit_target_remains_100_percent=True,sampling_source=reuse))
    progress('labels_complete',samples=len(meta),finite=int(meta.finite_label.sum()))


def completed_rows(meta,cutoff,months):
    cutoff=pd.Timestamp(cutoff)
    return meta.loc[meta.finite_label & meta.date.ge(cutoff-pd.DateOffset(months=months))
        & meta.date.lt(cutoff) & meta.end.lt(cutoff)].reset_index(drop=True)


def contexts(values,cuts):
    values=np.asarray(values,float)
    return ((values>np.asarray(cuts))*np.array([1,2,4])).sum(axis=1).astype(int)


def best_actions(book,counts):
    """Exhaust the <=3^8 maps; constants are explicit controls and tie-breaks."""
    free=np.flatnonzero(counts>=CFG['minimum_cell_samples'])
    full_index=int(np.flatnonzero(ACTIONS==4)[0])
    choices=np.full((3+3**len(free),8),full_index,dtype=int)
    choices[:3]=np.array([full_index]+[i for i in range(3) if i!=full_index])[:,None]
    choices[3:,free]=np.array(list(itertools.product(range(3),repeat=len(free))),int)
    columns=choices+np.arange(8)[None,:]*3
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    means=mean[columns].sum(axis=1)
    variance=cov[columns[:,:,None],columns[:,None,:]].sum(axis=(1,2))
    scores=np.sqrt(252)*np.divide(means,np.sqrt(np.maximum(0,variance)),out=np.zeros(len(means)),where=variance>1e-18)
    winner=int(np.argmax(scores))
    return ACTIONS[choices[winner]],float(scores[winner]),float(scores[0]),len(free)


def reward_partition(values,target,min_leaf,max_leaves=8):
    """Small CART partition, fitted only to completed full-size trade profits.

    Enumerate nine within-training quantile splits per feature; greedily split
    the leaf with greatest squared-error reduction. No external tree runtime.
    """
    values=np.asarray(values,float);target=np.asarray(target,float)
    leaves=[dict(indices=np.arange(len(target)),conditions=[])]
    while len(leaves)<max_leaves:
        best=None
        for leaf_id,leaf in enumerate(leaves):
            idx=leaf['indices'];y=target[idx]
            if len(idx)<2*min_leaf:continue
            parent=float(np.sum((y-y.mean())**2))
            for feature in range(values.shape[1]):
                for threshold in np.unique(np.quantile(values[idx,feature],np.arange(.1,1.,.1))):
                    left=values[idx,feature]<=threshold
                    if min(left.sum(),(~left).sum())<min_leaf:continue
                    a,b=y[left],y[~left]
                    gain=parent-float(np.sum((a-a.mean())**2)+np.sum((b-b.mean())**2))
                    if gain>1e-9 and (best is None or gain>best[0]):
                        best=(gain,leaf_id,feature,float(threshold),idx[left],idx[~left])
        if best is None:break
        _,leaf_id,feature,threshold,a,b=best
        conditions=leaves[leaf_id]['conditions']
        leaves[leaf_id]=dict(indices=a,conditions=conditions+[[feature,threshold,False]])
        leaves.append(dict(indices=b,conditions=conditions+[[feature,threshold,True]]))
    return [leaf['conditions'] for leaf in leaves]


def tree_contexts(values,leaves):
    values=np.asarray(values,float);states=np.zeros(len(values),int)
    assigned=np.zeros(len(values),bool)
    for state,conditions in enumerate(leaves):
        keep=np.ones(len(values),bool)
        for feature,threshold,above in conditions:
            keep &= (values[:,feature]>threshold) if above else (values[:,feature]<=threshold)
        states[keep]=state;assigned|=keep
    assert assigned[np.isfinite(values).all(axis=1)].all()
    return states


def fit(cutoff,months,meta,cal,partition='median'):
    rows=completed_rows(meta,cutoff,months)
    model=dict(cutoff=str(pd.Timestamp(cutoff).date()),months=months,features=FEATURES,
        cuts=[0.,0.,0.],choices=[4]*8,training_samples=len(rows),last_completed=None,partition=partition)
    if len(rows)<CFG['minimum_training_samples']:return model
    model['last_completed']=str(rows.end.max().date())
    assert rows.end.lt(cutoff).all()
    cuts=rows[FEATURES].median().to_numpy()
    if partition=='reward_tree':
        model['tree']=reward_partition(rows[FEATURES],rows.full_profit,
            max(12,min(63,len(rows)//8)))
        states=tree_contexts(rows[FEATURES],model['tree'])
    else:states=contexts(rows[FEATURES],cuts)
    occupancy=np.bincount(states,minlength=8)
    first=cal.searchsorted(pd.Timestamp(cutoff)-pd.DateOffset(months=months))
    stop=cal.searchsorted(cutoff)
    book=np.zeros((stop-first,8*len(ACTIONS)))
    for sid,indices in rows.groupby('secid').groups.items():
        with np.load(OUT/'labels'/f'{sid}.npz') as archive:
            paths=archive['paths'][rows.loc[indices,'label_row'].to_numpy(int)]
        days=rows.loc[indices,'entry_position'].to_numpy(int)[:,None]+np.arange(PATH_LENGTH)[None,:]-first
        valid=(days>=0)&(days<len(book))
        for state in np.unique(states[indices]):
            selected=np.flatnonzero(states[indices]==state)
            for action in range(len(ACTIONS)):
                np.add.at(book[:,state*3+action],days[selected][valid[selected]],paths[selected,action][valid[selected]])
    choices,score,baseline,nfree=best_actions(book,occupancy)
    model.update(cuts=cuts.tolist(),choices=choices.tolist(),occupancy=occupancy.tolist(),
        training_shadow_sharpe=score,full_shadow_sharpe=baseline,free_context_cells=nfree)
    return model


def predict(model,frame):
    values=frame[FEATURES].to_numpy(float)
    states=tree_contexts(values,model['tree']) if 'tree' in model else contexts(values,model['cuts'])
    levels=np.asarray(model['choices'])[states]
    levels[~np.isfinite(values).all(axis=1)]=4
    return levels


def month_schedule(rows,start,stop):
    """Include the preceding close's orders; retain only this replay chunk."""
    selected=rows.loc[rows.date.ge(start)&rows.date.lt(stop)]
    return {pd.Timestamp(d):g.sort_values(['priority','pair'],ascending=[False,True],kind='stable')
            for d,g in selected.groupby('date',sort=False)}


def allocations(case,stage):
    rows=pd.read_parquet(OUT/'proposals.parquet',use_threads=False)
    if stage=='validation':rows=rows.loc[rows.date.le(CFG['validation_end'])].copy()
    if case.startswith('fixed'):
        rows['learned_level']=int(case.removeprefix('fixed'))//25
    else:
        months=int(case.removeprefix('bandit').removeprefix('tree'))
        meta=pd.read_parquet(OUT/'training.parquet',use_threads=False)
        cal=base.calendar();parts=[]
        for month,g in rows.groupby(rows.date.dt.to_period('M'),sort=True):
            path=OUT/'models'/case/f'{month}.json'
            if path.exists():model=json.loads(path.read_text())
            else:
                model=fit(month.start_time,months,meta,cal,
                    partition='reward_tree' if case.startswith('tree') else 'median');dump(path,model)
            assert model['last_completed'] is None or pd.Timestamp(model['last_completed'])<month.start_time
            g=g.copy();g['learned_level']=predict(model,g);parts.append(g)
        rows=pd.concat(parts,ignore_index=True)
    rows.to_parquet(OUT/'allocations'/f'{case}_{stage}.parquet',index=False)
    progress('allocations',case=case,stage=stage,proposals=len(rows),sizes=rows.learned_level.value_counts().to_dict())


def replay(case,stage):
    import joblib
    from scripts import atm_rv_persistence_engine as engine
    from scripts.option_flex.contract_engine import Settings
    from scripts.cheap_vega_runtime import mark_book,metrics,annual_metrics
    cfg=Settings(**base.execution_settings())
    # Full proposal roster keeps resume inventory keys identical across the split.
    rows=pd.read_parquet(OUT/'allocations'/f'{case}_{stage}.parquet',columns=EXECUTION+['learned_level'],use_threads=False)
    universe=pd.read_parquet(OUT/'proposals.parquet',columns=['secid_a','optionid_a'],use_threads=False)
    market=base.DiskMarket(universe.secid_a.unique(),universe.optionid_a)
    del universe
    # To preserve roster keys in validation, keep all proposals but stop the clock.
    # Test rows are saved before validation even though future sizes are never read.
    if stage=='validation':
        full=pd.read_parquet(OUT/'proposals.parquet',columns=EXECUTION,use_threads=False)
        full['learned_level']=4
        levels=rows.set_index(['date','pair']).learned_level
        index=pd.MultiIndex.from_frame(full[['date','pair']])
        full['learned_level']=levels.reindex(index,fill_value=4).to_numpy()
        rows=full
    for col in ['pair','wing']:rows[col]=rows[col].astype('category')
    cal=market.rates.index[(market.rates.index>=CFG['validation_start'])&(market.rates.index<=CFG['test_end'])]
    end=pd.Timestamp(CFG['validation_end'] if stage=='validation' else CFG['test_end'])
    folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
    engine.book_risk=base.sparse_book_risk
    state=None
    try:
        for month in pd.period_range(cal[0],end,freq='M'):
            dest=folder/'months'/str(month);marker=dest/'checkpoint.joblib'
            if marker.exists():state=joblib.load(marker);continue
            dest.mkdir(parents=True,exist_ok=True)
            finish=min(month.end_time,end)
            schedule=month_schedule(rows,cal[0] if state is None else state['previous_date'],finish)
            nav,trades,decisions,audit=engine.simulate(market,rows,cal,cfg,base.CFG['per_name_reference_capital'],
                standalone=True,corporate_events=events(),max_stale_quote_sessions=2,
                policy=(lambda row,context:4) if CFG.get('use_scale_hook') else (lambda row,context:int(row['learned_level'])),
                scale_policy=(lambda row,context:float(row['learned_level'])/4) if CFG.get('use_scale_hook') else None,
                scale_pair_budget=False,resume_state=state,
                through_date=finish,prepared_schedule=schedule)
            state=audit.pop('resume_state')
            nav.to_parquet(dest/'daily_nav.parquet');trades.to_pickle(dest/'trades.pkl');decisions.to_pickle(dest/'decisions.pkl')
            joblib.dump(state,marker)
            progress('replay_month',case=case,stage=stage,month=str(month),entries=int(decisions.status.eq('entered').sum()) if len(decisions) else 0)
        months=sorted(p.parent for p in (folder/'months').glob('????-??/checkpoint.joblib'))
        del rows,schedule
        market.days.clear();market.allowed.clear();gc.collect()
        nav=pd.concat([pd.read_parquet(p/'daily_nav.parquet',use_threads=False) for p in months])
        mark_book(nav,market.rates,cfg.capital)
        nav.to_parquet(folder/f'{stage}_nav.parquet')
        decisions=pd.concat([pd.read_pickle(p/'decisions.pkl') for p in months],ignore_index=True)
        decisions.to_csv(folder/f'{stage}_decisions.csv',index=False)
        dump(folder/f'{stage}_status.json',dict(case=case,stage=stage,status='complete',
            entries=int(decisions.status.eq('entered').sum()),**metrics(nav,cfg.capital),max_ledger_error=state['max_ledger_error']))
        pd.DataFrame(annual_metrics(nav,case,cfg.capital)).to_csv(folder/f'{stage}_annual.csv',index=False)
        if stage=='test':assert not state['positions'] and abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    finally:market.conn.close()


def select():
    from paper_code.metrics import sharpe
    path=REPORT/'selection.json'
    if path.exists():return
    candidates=[]
    for case in CFG['validation_candidates']:
        months=int(case.removeprefix('bandit').removeprefix('tree'))
        nav=pd.read_parquet(OUT/'replays'/case/'validation_nav.parquet',use_threads=False)
        assert nav.index.max()<=pd.Timestamp(CFG['validation_end'])
        candidates.append(dict(case=case,months=months,validation_sharpe=sharpe(nav.excess_return.iloc[1:])))
    ranked=sorted(candidates,key=lambda row:(-row['validation_sharpe'],-row['months'],row['case']))
    dump(path,dict(selected=ranked[0],candidates=ranked,selected_using_validation_only=True,
        previously_examined_dates=True,configuration_hash=hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()))
    progress('selection',**ranked[0])


def signal_report():
    """Compare constant full-size signal pools before interpreting sizing gains."""
    from paper_code.metrics import period_metrics
    records=[];entered={}
    for fraction,folder,stem in [(20,base.OUT/'replays'/'residual_memory','daily'),
                                 (int(100*CFG.get('selection_fraction',.2)),replay_folder('fixed100'),'test')]:
        nav=pd.read_parquet(folder/(stem+'_nav.parquet'),use_threads=False)
        path=folder/('decisions.csv' if stem=='daily' else stem+'_decisions.csv')
        decisions=pd.read_csv(path,usecols=['signal_date','entry_date','pair','status'],parse_dates=['signal_date','entry_date'])
        entered[fraction]=set(zip(decisions.loc[decisions.status.eq('entered'),'signal_date'],decisions.loc[decisions.status.eq('entered'),'pair']))
        for label,start,stop in [('validation',CFG['validation_start'],CFG['validation_end']),('test',CFG['test_start'],CFG['test_end']),('full',CFG['validation_start'],CFG['test_end'])]:
            records.append(dict(bottom_percent=fraction,period=label,**period_metrics(nav,decisions,start,stop)))
    pd.DataFrame(records).to_csv(REPORT/'signal_comparison.csv',index=False)
    if len(entered)==2:
        a,b=entered.values()
        dump(REPORT/'signal_entry_overlap.json',dict(shared=len(a&b),old_only=len(a-b),new_only=len(b-a)))
    print(pd.DataFrame(records)[['bottom_percent','period','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)


def report():
    from paper_code.metrics import period_metrics,paired_interval
    selected=json.loads((REPORT/'selection.json').read_text())['selected']['case']
    results=[];yearly=[];returns={}
    baseline=base.OUT/'replays'/'residual_memory'
    for case in controls()+[selected]:
        if case=='fixed100' and not CFG.get('use_local_fixed100'):
            nav=pd.read_parquet(baseline/'daily_nav.parquet',use_threads=False)
            decisions=pd.read_csv(baseline/'decisions.csv',usecols=['entry_date','status'],parse_dates=['entry_date'])
        else:
            folder=replay_folder(case)
            if not (folder/'test_nav.parquet').exists():continue
            nav=pd.read_parquet(folder/'test_nav.parquet',use_threads=False)
            decisions=pd.read_csv(folder/'test_decisions.csv',usecols=['entry_date','status'],parse_dates=['entry_date'])
        returns[case]=nav.excess_return
        for label,start,stop in [('validation','2018-01-01',CFG['validation_end']),('test',CFG['test_start'],CFG['test_end']),('full','2018-01-01',CFG['test_end'])]:
            results.append(dict(case=case,period=label,**period_metrics(nav,decisions,start,stop)))
        for year in range(2018,2026):
            yearly.append(dict(case=case,year=year,**period_metrics(nav,decisions,f'{year}-01-01',min(f'{year}-12-31',CFG['test_end']))))
    pd.DataFrame(results).to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(yearly).to_csv(REPORT/'annual.csv',index=False)
    intervals=[]
    for period,start,stop in [('validation','2018-01-01',CFG['validation_end']),('test',CFG['test_start'],CFG['test_end'])]:
        for control in controls():
            if control not in returns:continue
            a=returns[selected].loc[start:stop];b=returns[control].loc[start:stop]
            if period=='validation':a,b=a.iloc[1:],b.iloc[1:]
            intervals.append(dict(period=period,control=control,**paired_interval(a,b,block=63,draws=1000)))
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_comparisons.csv',index=False)
    logs=[]
    for path in sorted((OUT/'models'/selected).glob('*.json')):
        model=json.loads(path.read_text());logs.append({k:v for k,v in model.items() if k not in ['cuts','choices','features','occupancy']})
    pd.DataFrame(logs).to_csv(REPORT/'monthly_training.csv',index=False)
    dump(REPORT/'validation_candidates.json',json.loads((REPORT/'selection.json').read_text()))
    protocol=json.loads((REPORT/'protocol.json').read_text())
    protocol['final_specification']=CFG
    protocol['selection']=json.loads((REPORT/'selection.json').read_text())
    protocol['policy_training_note']='Independent action-specific shadow cash paths; funded inventory/admission effects are evaluated during replay, not optimized directly during training.'
    dump(REPORT/'protocol.json',protocol)
    dump(REPORT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),CFG_PATH,base.CFG_PATH,Path('scripts/atm_rv_persistence_engine.py'),Path('data/raw/VIX_History.csv')]})
    progress('report_complete',selected=selected,models_complete=len(returns))


def audit():
    """Reconcile saved fills independently; verify timing and proposal parity."""
    selected=json.loads((REPORT/'selection.json').read_text())['selected']['case']
    checks=[];admissions=[];sizes=[]
    cal=base.calendar()
    for case in [c for c in controls() if c!='fixed100' or CFG.get('use_local_fixed100')]+[selected]:
        folder=replay_folder(case)
        nav=pd.read_parquet(folder/'test_nav.parquet',use_threads=False)
        paths=sorted((folder/'months').glob('????-??/trades.pkl'))
        trades=pd.concat([pd.read_pickle(p) for p in paths],ignore_index=True)
        decisions=pd.read_csv(folder/'test_decisions.csv',parse_dates=['signal_date','entry_date'])
        assert len(decisions)==json.loads((REPORT/'protocol.json').read_text())['proposals']
        assert decisions.entry_date.eq(cal[cal.get_indexer(decisions.signal_date)+1]).all()
        effective_levels=(decisions.action.abs()*4).round().astype(int)
        assert effective_levels.loc[decisions.status.eq('entered')].isin(ACTIONS).all()
        decisions['level']=effective_levels
        assert trades.loc[trades.reason.eq('entry'),'quantity'].lt(0).all()
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
        flow=(-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
        expected=(flow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl
            -nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash)
        error=float((expected-nav.pnl).abs().max());assert error<1e-5
        charged=trades.reason.isin(['entry','announced_corporate_action_exit'])
        assert (trades.loc[charged,'commission']-.03*trades.loc[charged,'quantity'].abs()).abs().max()<1e-8
        assert trades.loc[~charged,'commission'].eq(0).all()
        if case not in CFG.get('reuse_replays',{}):trades.to_csv(folder/'trades.csv',index=False)
        checks.append(dict(case=case,independent_ledger_error=error,terminal_inventory_flat=True,
            proposals=len(decisions),entries=int(decisions.status.eq('entered').sum()),next_session_execution=True))
        for status,count in decisions.status.value_counts().items():
            admissions.append(dict(case=case,status=status,count=int(count)))
        for period,start,stop in [('validation','2018-01-01',CFG['validation_end']),('test',CFG['test_start'],CFG['test_end'])]:
            part=decisions.loc[decisions.entry_date.between(start,stop)]
            for status,g in [('proposed',part),('entered',part.loc[part.status.eq('entered')])]:
                for level,count in g.level.value_counts().items():
                    sizes.append(dict(case=case,period=period,population=status,size_fraction=float(level)/4,count=int(count)))
        del nav,trades,decisions,expected,flow
        gc.collect()
    for path in (OUT/'models'/selected).glob('*.json'):
        model=json.loads(path.read_text())
        assert model['last_completed'] is None or pd.Timestamp(model['last_completed'])<pd.Timestamp(model['cutoff'])
    dump(REPORT/'ledger_audit.json',checks)
    pd.DataFrame(admissions).to_csv(REPORT/'admissions.csv',index=False)
    pd.DataFrame(sizes).to_csv(REPORT/'sizes.csv',index=False)
    progress('audit_complete',portfolios=len(checks))


def charts():
    """Render from small saved tables in a fresh worker, reusing the font cache."""
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(base.OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=json.loads((REPORT/'selection.json').read_text())['selected']['case']
    names={'fixed100':'Fixed 100%','fixed75':'Fixed 75%','fixed50':'Fixed 50%','fixed200':'Fixed 200%',selected:'Contextual bandit'}
    results=pd.read_csv(REPORT/'results.csv')
    cases=controls()+[selected]
    fig,ax=plt.subplots(figsize=(8,4))
    for i,case in enumerate(cases):
        part=results.loc[results.case.eq(case)].set_index('period')
        ax.bar(np.arange(2)+(i-1.5)*.18,part.loc[['validation','test'],'annualized_excess_sharpe'],
            width=.18,label=names[case])
    ax.set_xticks([0,1],['2018–2021 validation','2022–August 2025 test'])
    ax.set(ylabel='Annualized excess Sharpe',title='Same ATM call proposals, costs and risk limits')
    ax.axhline(0,color='grey',lw=.7);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(REPORT/'validation_test_sharpe.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for case in cases:
        path=base.OUT/'replays'/'residual_memory'/'daily_nav.parquet' if case=='fixed100' and not CFG.get('use_local_fixed100') else replay_folder(case)/'test_nav.parquet'
        nav=pd.read_parquet(path,columns=['nav','cash_benchmark'],use_threads=False)
        ax.plot(nav.index,100*(nav.nav/nav.cash_benchmark-1),label=names[case],lw=1.2)
    ax.axvline(pd.Timestamp(CFG['test_start']),color='grey',ls='--',lw=.9)
    ax.set(ylabel='Account growth relative to cash (%)',title='Continuing accounts across the validation/test boundary')
    ax.grid(alpha=.15);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(REPORT/'cash_relative_growth.png',dpi=140);plt.close(fig)
    # A concrete, reproducible view of the final fitted sizing rule.
    model_path=sorted((OUT/'models'/selected).glob('*.json'))[-1]
    model=json.loads(model_path.read_text());rules=[]
    for leaf,conditions in enumerate(model.get('tree',[])):
        text=[]
        for feature,threshold,above in conditions:
            unit=100 if feature in [0,1] else 1
            label=['past-year RV−IV (vol points)','ATM IV (vol points)','VIX'][feature]
            text.append(f"{label} {'>' if above else '<='} {threshold*unit:.2f}")
        rules.append(dict(fit=model['cutoff'],leaf=leaf,conditions=' AND '.join(text),
            size_fraction=model['choices'][leaf]/4,training_samples=model['occupancy'][leaf]))
    pd.DataFrame(rules).to_csv(REPORT/'latest_policy_rules.csv',index=False)
    lines=['ATM RV-IV CONTEXTUAL SIZING BANDIT','',
        'Inputs: trailing-year completed RV-minus-IV memory, signal-close ATM IV, and signal-close VIX.',
        f"Daily proposal pool: bottom {100*CFG.get('selection_fraction',.2):g}% of past-year residual memory, identical across sizing policies. All entries short approximately 30-day ATM calls and delta hedge daily.",
        f"Sizing actions: {', '.join(str(int(a*25))+'%' for a in ACTIONS)} of the original $450-vega reference per trade. Account and execution assumptions unchanged.",
        f"Candidates compared on 2018-2021 validation: {', '.join(CFG['validation_candidates'])}.",
        'Configuration and selection history are recorded in protocol.json; previous ladders remain in separate experiment folders.',
        f'Selected candidate: {selected}. Monthly fits use only outcomes completed before month start; the selected procedure continues to refit during evaluation.',
        'Training uses a bounded sample of up to twelve trade proposals at the first trading signal date of each month. This limits temporal/VIX coverage and is a prototype, not an exhaustive use of all training opportunities.',
        'Unit actual-contract cash ledgers generate integer-count action payoffs; cash is separately replayed by action including financing. Funded capacity effects enter only during portfolio replay.',
        '2018-2021 is called validation; 2022-August2025 is called test. These historical dates were examined in previous research, so this is not an untouched holdout.',
        '', 'ANNUALIZED EXCESS SHARPE / ANNUALIZED GROWTH RELATIVE TO CASH / ENTRIES']
    for case in cases:
        for period in ['validation','test','full']:
            row=results.loc[results.case.eq(case)&results.period.eq(period)].iloc[0]
            lines.append(f"{names[case]}, {period}: Sharpe {row.annualized_excess_sharpe:.3f}; cash-relative growth {row.annualized_growth_relative_to_cash_pct:.3f}%; {row.entries} entries.")
    lines += ['', 'Interpretation: compare both validation and test. The difference from fixed full size cannot by itself isolate predictive sizing from changed inventory/admission; fixed-size controls use the same action ladder.',
        'The bootstrap intervals are conditional on these fitted/selected policies, using paired 63-session circular blocks and 1,000 draws. They do not correct research selection.',
        '', 'Files: results.csv, annual.csv, validation_candidates.json, monthly_training.csv, sizes.csv,',
        'sharpe_comparisons.csv, ledger_audit.json, admissions.csv, latest_policy_rules.csv, protocol.json, source_hashes.json.',
        f'Daily funded ledgers, monthly checkpoints, fitted policies and action-specific training paths: {OUT}.',
        '', f'Reproduce with scripts.research_atm_rv_bandit --config {CFG_PATH} under scripts.run_signal_discovery_guarded:',
        'phases sample, labels, allocate/replay (each validation candidate), select, allocate/replay (selected case and fixed-size controls, stage test), report, audit, charts.',
        'Use one worker, 20% one-core duty, 256 MiB RSS/compressed stop, 30% minimum headroom, 256 MiB swap-growth stop. Monthly checkpoints permit safe resumption.']
    (REPORT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    progress('charts_complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(CFG_PATH))
    parser.add_argument('--phase',choices=['sample','labels','allocate','replay','select','report','audit','charts','test','signal_report'],required=True)
    parser.add_argument('--case',default='bandit18',choices=['bandit18','bandit60','tree18','tree60','fixed50','fixed75','fixed100','fixed200'])
    parser.add_argument('--stage',default='validation',choices=['validation','test'])
    args=parser.parse_args()
    configure(args.config)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);(OUT/'allocations').mkdir(exist_ok=True)
    if args.stage=='test' and args.phase in ['allocate','replay'] and args.case.startswith(('bandit','tree')):
        lock=json.loads((REPORT/'selection.json').read_text())
        assert lock['configuration_hash']==hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()
        if args.case.startswith(('bandit','tree')):assert args.case==lock['selected']['case']
    def tests():
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_atm_rv_bandit'))
        if not result.wasSuccessful():raise SystemExit(1)
    dict(sample=sample,labels=labels,allocate=lambda:allocations(args.case,args.stage),
        replay=lambda:replay(args.case,args.stage),select=select,report=report,audit=audit,charts=charts,test=tests,signal_report=signal_report)[args.phase]()


if __name__=='__main__':main()
