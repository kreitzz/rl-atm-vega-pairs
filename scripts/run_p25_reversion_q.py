"""Paper-inspired options timing experiment, isolated from prior checkpoints."""
import argparse
from pathlib import Path
import resource
import json
import hashlib
import gc
import time
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
import pandas as pd
import joblib

from scripts import p25_reversion_q as learning

ROOT = Path('artifacts/contracts/uniform_smile')
OUT = Path('artifacts/contracts/p25_reversion_q')
REPORT = Path('reports/p25_reversion_q')
META = Path('artifacts/contracts/p25_feature_search/training_rows.parquet')
FROZEN = Path('artifacts/contracts/p25_feature_search/replays/variance_premium/scored_allocations.parquet')
WIDTH = 57
RISK_PENALTY = .25
MODEL_VERSION = 3
EXECUTION_VERSION = 3


def model_folder():
    return OUT/('models_v3' if RISK_PENALTY==.25 else 'models_v3_risk0')


def result_name(case):
    return case+'_v3'+('' if RISK_PENALTY==.25 else '_risk0')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def progress(stage, **details):
    dump(OUT/'progress.json', dict(stage=stage, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **details))


def protect():
    path=OUT/'protected_sources.json'
    if path.exists():
        sources=json.loads(path.read_text())
        for name,want in sources.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want, name
        return
    sources={}
    paths=[META,FROZEN,Path('config/p25_variance_premium_candidate.json')]
    for case in ['10m_fixed50','10m_fixed100','10m_learned']:
        paths += list((Path('artifacts/contracts/p25_capital_sensitivity/replays')/case).glob('*'))
    for p in paths:
        if p.is_file():sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(path,sources)


class AssetCache:
    def __init__(self, cal):
        self.cal=cal
        self.cache=OrderedDict()
    def get(self, sid):
        if sid not in self.cache:
            folder=ROOT/'market/assets'
            columns=['date','optionid','expiry','strike','bid','ask','delta','vega','bs_iv','bs_rate','known_dividend_pv','cfadj','bs_status']
            q=pd.read_parquet(folder/f'{sid}_quotes.parquet',columns=columns)
            q=q.set_index(['date','optionid']).sort_index()
            s=pd.read_parquet(folder/f'{sid}_stocks.parquet',columns=['date','close','cfadj']).set_index('date').reindex(self.cal)
            self.cache[sid]=(q,s)
            while len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        return self.cache[sid]


def exit_adjustments(rows, cache, cal):
    """Extra cost of paired next-close liquidation versus that close's hold.

    Existing shadow labels already contain the hold hedge rebalance. Replace
    that stock turnover cost with the cost of unwinding the prior hedge.
    Option midpoint liquidation itself changes cash/marks but not NAV.
    """
    from scripts.option_flex.black_scholes import option_values
    n=len(rows)
    positions=rows.position.to_numpy(int)[:,None]+1+np.arange(WIDTH)[None,:]
    dates=cal[np.minimum(positions.ravel(),len(cal)-1)]
    parts=[]
    for side in ['a','b']:
        sid=int(rows['secid_'+side].iloc[0])
        q,s=cache.get(sid)
        ids=np.repeat(rows['optionid_'+side].to_numpy(int),WIDTH)
        keys=pd.MultiIndex.from_arrays([dates,ids])
        gathered=q.reindex(keys)
        arr={f:gathered[f].to_numpy().reshape(n,WIDTH) for f in ['strike','bid','ask','delta','vega','bs_iv','bs_rate','known_dividend_pv','cfadj']}
        expiry=gathered.expiry.to_numpy().reshape(n,WIDTH)
        unsupported=gathered.bs_status.eq('unsupported_distribution').to_numpy().reshape(n,WIDTH)
        spot=s.close.to_numpy()[np.minimum(positions,len(cal)-1)]
        adj=s.cfadj.to_numpy()[np.minimum(positions,len(cal)-1)]
        factor=adj/adj[:,[0]]
        terminal=positions >= cal.searchsorted(expiry,side='right')-1
        delta=np.asarray(arr['delta'],float).copy()
        valid_greek=np.isfinite(delta+arr['vega'])&(delta>=-1)&(delta<=0)&(arr['vega']>=0)
        missing=~valid_greek&~terminal
        if missing.any():
            iv=pd.DataFrame(arr['bs_iv']).where(lambda x:x>0).ffill(axis=1).to_numpy()
            days=(expiry-dates.values.reshape(n,WIDTH))/np.timedelta64(1,'D')
            good=missing&np.isfinite(iv+arr['bs_rate']+arr['known_dividend_pv'])&(spot>arr['known_dividend_pv'])&(days>0)
            if good.any():
                _,dd,_,_=option_values((spot-arr['known_dividend_pv'])[good],arr['strike'][good],days[good]/365,
                    arr['bs_rate'][good],iv[good],'P')
                delta[good]=dd
        delta[terminal]=0.
        delta=np.nan_to_num(delta)
        # Per-initial-contract hedges include verified split factors already
        # admitted by the original finite-label audit.
        hedge=-factor*100*delta
        old=np.column_stack([np.zeros(n),hedge[:,:-1]])
        ratio=adj/np.column_stack([adj[:,0],adj[:,:-1]])
        old*=ratio
        correction=(np.abs(old)-np.abs(hedge-old))*spot*.0001
        valid_quote=np.isfinite(arr['bid']+arr['ask']+arr['strike']+arr['cfadj'])&(arr['bid']>0)&(arr['ask']>=arr['bid'])
        executable=valid_quote&valid_greek&~terminal&~unsupported&(positions<len(cal))
        counts=np.floor((450*learning.LEVELS[:3]/4)[None,:]/(100*arr['vega'][:,[0]]))
        extra=factor*(100*np.nan_to_num(arr['vega'])*.1+.65)+correction
        parts.append((counts,extra,executable,100*arr['vega'][:,0]))
    valid=np.ones((n,3),bool)
    vegas=[p[0]*p[3][:,None] for p in parts]
    for counts,_,_,v in parts:valid&=(counts>=1)&np.isfinite(counts)&(v[:,None]>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        valid&=abs(vegas[0]-vegas[1])/np.maximum(vegas[0],vegas[1])<=.1
    extra=sum(np.nan_to_num(p[0])[:,:,None]*np.nan_to_num(p[1])[:,None,:] for p in parts)/450
    executable=parts[0][2]&parts[1][2]
    return np.tile(extra,(1,2,1)).astype(np.float32),executable,np.tile(valid,(1,2))


def prepare():
    if (OUT/'prepared.json').exists():return
    rows=pd.read_parquet(META).reset_index(drop=True)
    cal=pd.read_parquet(ROOT/'market/rates.parquet').index
    assert np.array_equal(cal.get_indexer(pd.DatetimeIndex(rows.date)),rows.position.to_numpy())
    shape=(len(rows),6,WIDTH)
    arrays={
        'reward':np.lib.format.open_memmap(OUT/'reward.npy',mode='w+',dtype='float32',shape=shape),
        'exit_adjustment':np.lib.format.open_memmap(OUT/'exit_adjustment.npy',mode='w+',dtype='float32',shape=shape),
        'base':np.lib.format.open_memmap(OUT/'base.npy',mode='w+',dtype='int32',shape=(len(rows),WIDTH)),
        'can_exit':np.lib.format.open_memmap(OUT/'can_exit.npy',mode='w+',dtype='bool',shape=(len(rows),WIDTH)),
        'entry_valid':np.lib.format.open_memmap(OUT/'entry_valid.npy',mode='w+',dtype='bool',shape=(len(rows),6)),
        'entry_base':np.lib.format.open_memmap(OUT/'entry_base.npy',mode='w+',dtype='int32',shape=(len(rows),))}
    assets=sorted(set(rows.secid_a)|set(rows.secid_b))
    marks={}
    for sid in assets:
        s=pd.read_parquet(ROOT/'market/assets'/f'{int(sid)}_selection.parquet',columns=['date','tenor','cp_flag','bucket','bs_iv'])
        marks[int(sid)]=s.loc[s.tenor.eq(30)&s.cp_flag.eq('P')&s.bucket.eq('25')].set_index('date').bs_iv.reindex(cal)
    cache=AssetCache(cal)
    coverage=[]
    for number,(pair,idx) in enumerate(rows.groupby('pair',sort=False).groups.items(),1):
        idx=np.asarray(idx);g=rows.loc[idx]
        a,b=int(g.secid_a.iloc[0]),int(g.secid_b.iloc[0])
        states=learning.spread_states(marks[a]-marks[b])
        states.index.name='date'
        states.to_parquet(OUT/'pair_states'/f'{pair}.parquet')
        p=g.position.to_numpy(int)
        arrays['entry_base'][idx]=states.base_state.to_numpy()[p]
        future=p[:,None]+1+np.arange(WIDTH)[None,:]
        arrays['base'][idx]=states.base_state.to_numpy()[np.minimum(future,len(cal)-1)]
        disk=np.load(ROOT/'pairs'/pair/'P25_labels.npy',mmap_mode='r')
        labels=disk[g.label_row.to_numpy(int)]
        arrays['reward'][idx]=labels[:,learning.LABEL_CODES]
        extra,executable,valid=exit_adjustments(g,cache,cal)
        arrays['exit_adjustment'][idx]=extra
        arrays['can_exit'][idx]=executable
        arrays['entry_valid'][idx]=valid
        coverage.append(dict(pair=pair,candidates=len(g),finite_training_paths=int(g.finite_label.sum()),
            signal_eligible=int(g.signal_eligible.sum()),reversion_history_available=int(states.reversion_days.notna().sum())))
        if number%10==0:
            print(f'Prepared {number}/125 pairs',flush=True)
            progress('prepare',pairs=number)
        del labels,disk,extra,states
    for v in arrays.values():v.flush()
    rows[['date','end','position','pair','wing','secid_a','secid_b','optionid_a','optionid_b','rule_direction','signal_eligible','finite_label']].to_parquet(OUT/'rows.parquet',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    dump(OUT/'prepared.json',dict(candidates=len(rows),pairs=rows.pair.nunique(),assets=len(assets),
        finite_training_paths=int(rows.finite_label.sum()),width=WIDTH,feature_future_data=False,
        new_entry_gates=False,training_path_costs='Original .65 option commission, .1 vol point, 1bp stock hedge',
        full_universe='All cached beta/correlation/formation-context candidates; rank/profit gates removed'))
    progress('prepared')


def open_arrays():
    return {f:np.load(OUT/f'{f}.npy',mmap_mode='r') for f in ['reward','exit_adjustment','base','can_exit','entry_valid','entry_base']}


def fit():
    rows=pd.read_parquet(OUT/'rows.parquet')
    cal=pd.read_parquet(ROOT/'market/rates.parquet').index
    arrays=open_arrays()
    logs=[]
    for cut in pd.date_range('2018-01-01','2025-08-01',freq='MS'):
        stamp=str(cut.date());file=model_folder()/f'{stamp}.joblib';note=file.with_suffix('.json')
        if note.exists():logs.append(json.loads(note.read_text()));continue
        idx=learning.causal_rows(rows,cut,18)
        record=dict(fit=stamp,training_cohorts=len(idx),model_version=MODEL_VERSION,last_scheduled_expiry=str(rows.end.iloc[idx].max()) if len(idx) else None)
        if len(idx)<32:
            model=None
            record.update(status='cold_start_original_rank_fixed100_hold',transitions=0)
        elif (OUT/('models_v2' if RISK_PENALTY==.25 else 'models_v2_risk0')/f'{stamp}.joblib').exists():
            source=OUT/('models_v2' if RISK_PENALTY==.25 else 'models_v2_risk0')/f'{stamp}.joblib'
            old=json.loads(source.with_suffix('.json').read_text())
            assert old['training_cohorts']==len(idx)
            model=joblib.load(source)
            assert model is not None
            original_hash=hashlib.sha256(model['hold_q'].tobytes()+model['close_q'].tobytes()).hexdigest()
            base=arrays['base'][idx,0]
            learning.fit_entry_head(model,arrays['entry_base'][idx],arrays['reward'][idx,:,0],
                learning.held_state(base[:,None],0,np.arange(6)[None,:]),arrays['entry_valid'][idx])
            assert hashlib.sha256(model['hold_q'].tobytes()+model['close_q'].tobytes()).hexdigest()==original_hash
            record.update(status='fitted',transitions=old['transitions'],risk_penalty=RISK_PENALTY,
                reward_scale=model['reward_scale'],iterations=40,held_bellman_tables_unchanged=True,
                entry_rejection_outcomes_included=True,held_table_sha256=original_hash)
        else:
            lengths=cal.get_indexer(pd.DatetimeIndex(rows.end.iloc[idx]))-rows.position.iloc[idx].to_numpy()
            base=arrays['base'][idx]
            reward=arrays['reward'][idx]
            extra=arrays['exit_adjustment'][idx]
            valid=arrays['entry_valid'][idx]
            j=np.arange(WIDTH-1)[None,None,:]
            use=(j < lengths[:,None,None]-1)&valid[:,:,None]
            use=np.broadcast_to(use,(len(idx),6,WIDTH-1)).copy()
            held=learning.held_state(base[:,:-1,None].transpose(0,2,1),j,np.arange(6)[None,:,None])
            following=learning.held_state(base[:,1:,None].transpose(0,2,1),j+1,np.arange(6)[None,:,None])
            terminal=np.broadcast_to(j+1>=lengths[:,None,None]-1,use.shape)
            can_exit=np.broadcast_to(arrays['can_exit'][idx,1:][:,None,:],use.shape)
            r=reward[:,:,1:][use].astype(float)
            close=(reward[:,:,1:]-extra[:,:,1:])[use].astype(float)
            assert np.isfinite(r).all() and np.isfinite(close).all()
            model=learning.fit_q(arrays['entry_base'][idx],held[use],following[use],r,close,
                can_exit[use],terminal[use],reward[:,:,0],learning.held_state(base[:,[0]],0,np.arange(6)[None,:]),valid,risk_penalty=RISK_PENALTY)
            record.update(status='fitted',transitions=len(r),risk_penalty=RISK_PENALTY,reward_scale=model['reward_scale'],iterations=40)
            del held,following,reward,extra,base,use,r,close
        file.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,file)
        dump(note,record);logs.append(record)
        if cut.month in [1,4,7,10]:
            print(f'Q fit {stamp}: {record["training_cohorts"]} completed cohorts, {record["transitions"]} transitions',flush=True)
        progress('fit',cutoff=stamp,completed=len(logs))
        gc.collect()
    pd.DataFrame(logs).to_csv(REPORT/('fits_v3.csv' if RISK_PENALTY==.25 else 'fits_v3_risk0.csv'),index=False)


def scored(scope,sized):
    rows=pd.read_parquet(OUT/'rows.parquet')
    rows=rows.loc[rows.date.ge('2018-01-01')&rows.date.le('2025-06-30')].copy()
    if scope=='matched':
        frozen=pd.read_parquet(FROZEN,columns=['date','pair'])
        rows=rows.merge(frozen,on=['date','pair'],how='inner',validate='one_to_one')
        assert len(rows)==608
    states=open_arrays()['entry_base']
    original=pd.read_parquet(OUT/'rows.parquet')[['date','pair']].reset_index().rename(columns={'index':'array_row'})
    rows=rows.merge(original,on=['date','pair'],validate='one_to_one')
    rows['base_level']=4;rows['direction']=1;rows['learned_level']=0
    for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
        model=joblib.load(model_folder()/f'{month.start_time.date()}.joblib')
        if model is None:
            direction=rows.loc[idx,'rule_direction'].to_numpy(int)
            level=np.where(direction!=0,4,0)
            direction=np.where(direction==0,1,direction)
        else:
            direction,level=learning.entry_actions(model,states[rows.loc[idx,'array_row'].to_numpy(int)],sized=sized)
        rows.loc[idx,'direction']=direction;rows.loc[idx,'learned_level']=level
    return rows.sort_values(['date','pair'],kind='stable').reset_index(drop=True)


def management(rows,cal):
    states={pair:pd.read_parquet(OUT/'pair_states'/f'{pair}.parquet').base_state.reindex(cal) for pair in rows.pair.unique()}
    levels={f'{r.date.date()}|{r.pair}|{r.wing}':int(r.learned_level) for r in rows.itertuples()}
    dates={d:i for i,d in enumerate(cal)}
    current_month=None;model=None
    def callback(date,held):
        nonlocal model,current_month
        month=date.to_period('M')
        if month!=current_month:
            model=joblib.load(model_folder()/f'{month.start_time.date()}.joblib')
            current_month=month
        if model is None or len(held)!=2:return 1.
        # The shadow close action requires both contracts to survive to its
        # fill close. Their scheduled exit dates are already known today.
        if not paired_exit_survives(date,held,cal):return 1.
        pair=held[0]['pair'];cohort=held[0]['cohort']
        # The engine preserves leg order as A then B, including paired exits.
        direction=int(np.sign(held[0]['quantity']))
        level=levels[cohort]
        action=(level-2)+(3 if direction>0 else 0)
        age=dates[date]-dates[held[0]['entry_date']]
        state=int(learning.held_state(int(states[pair].loc[date]),age,action))
        return 0. if learning.should_close(model,[state])[0] else 1.
    return callback


def paired_exit_survives(date,held,cal):
    position=cal.get_loc(date)
    if len(held)!=2 or position+1>=len(cal):return False
    return all(p['exit_date']>cal[position+1] for p in held)


def replay(case):
    from scripts import run_p25_long_management as ledger
    from scripts import uniform_smile_data as cached
    from scripts.run_p25_signal_transfer import Market
    from scripts.run_p25_capital_sensitivity import metrics,annual_metrics
    name=result_name(case)
    folder=OUT/'replays'/name
    if (folder/'status.json').exists():return
    scope='matched' if case.startswith('matched') else 'universe'
    sized='_sized' in case
    managed='_managed' in case
    frozen_entry=case.startswith('matched_frozen_sized_q_managed')
    rows=pd.read_parquet(FROZEN) if frozen_entry else scored(scope,sized)
    locked=None
    if case.endswith('_locked'):
        reference=Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned')
        accepted=pd.read_csv(reference/'decisions.csv',parse_dates=['signal_date']).query("status=='entered'")
        rows=rows.merge(accepted[['signal_date','pair']].rename(columns={'signal_date':'date'}),on=['date','pair'],validate='one_to_one')
        entry=pd.read_csv(reference/'trades.csv').query("reason=='entry'")
        groups={cohort:g.set_index('secid') for cohort,g in entry.groupby('cohort')}
        locked={f'{r.date.date()}|{r.pair}|{r.wing}':tuple(groups[f'{r.date.date()}|{r.pair}|{r.wing}'].loc[[r.secid_a,r.secid_b],'quantity'].abs()) for r in rows.itertuples()}
    cached.OUT=ROOT
    market=Market();market.variant=name;market.started=time.monotonic();market.reported_year=None
    cal=market.rates.index[market.rates.index>='2018-01-01']
    callback=management(rows,cal) if managed else None
    print(f'Replay {case}: {len(rows)} candidates, {int(rows.learned_level.gt(0).sum())} requested entries',flush=True)
    nav,trades,decisions,audit=ledger.replay(rows,market,cal,30,callback,locked_counts=locked)
    logs=audit.pop('management_log',[])
    record=dict(case=name,scope=scope,sized=sized,managed=managed,risk_penalty=RISK_PENALTY,model_version=MODEL_VERSION,execution_version=EXECUTION_VERSION,
        status='complete',**metrics(nav,10_000_000),entries=int(decisions.status.eq('entered').sum()),proposals=len(rows),
        learned_skips=int(rows.learned_level.eq(0).sum()),
        closed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique()) if len(trades) else 0,
        active_pairs=int(decisions.loc[decisions.status.eq('entered'),'pair'].nunique()),
        **audit)
    folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'config.json',asdict(ledger.configuration(30)))
    rows.to_parquet(folder/'scored_allocations.parquet',index=False)
    nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
    pd.DataFrame(annual_metrics(nav,name,10_000_000)).to_csv(folder/'annual.csv',index=False)
    dump(folder/'management_log.json',logs);dump(folder/'status.json',record)
    print(f'{name}: Sharpe {record["annualized_excess_sharpe"]:.6f}, {record["entries"]} entries, {record["closed_cohorts"]} paired exits',flush=True)
    progress('replay',case=case)


def anchor():
    """Prove the broader quote cache preserves the retained funded ledger."""
    if (REPORT/'anchor.json').exists():return
    from scripts import run_p25_long_management as ledger
    from scripts import uniform_smile_data as cached
    from scripts.run_p25_signal_transfer import Market
    cached.OUT=ROOT
    market=Market();market.variant='reversion_q_original_anchor';market.started=time.monotonic();market.reported_year=None
    rows=pd.read_parquet(FROZEN)
    cal=market.rates.index[market.rates.index>='2018-01-01']
    nav,trades,decisions,audit=ledger.replay(rows,market,cal,30)
    expected=Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned')
    pd.testing.assert_frame_equal(nav,pd.read_parquet(expected/'daily_nav.parquet'),rtol=1e-12,atol=1e-6)
    old=pd.read_csv(expected/'trades.csv',parse_dates=['date'])
    pd.testing.assert_frame_equal(trades.reset_index(drop=True),old.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-7)
    old=pd.read_csv(expected/'decisions.csv',parse_dates=['signal_date','entry_date'])
    pd.testing.assert_frame_equal(decisions[old.columns].reset_index(drop=True),old.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-7)
    dump(REPORT/'anchor.json',dict(retained_nav_trades_decisions_reproduced=True,**audit))
    print('Original 1.003 Sharpe ledger reproduced with broader quote cache',flush=True)


CASES=['matched_q_fixed_managed','matched_q_sized_managed','universe_q_fixed_managed',
       'universe_q_sized_managed','universe_q_fixed_hold','universe_q_sized_hold',
       'matched_frozen_sized_q_managed','matched_frozen_sized_q_managed_locked']


def summarize():
    records=[];years=[];preliminary=[]
    for folder in sorted((OUT/'replays').iterdir()):
        if not (folder/'status.json').exists():continue
        record=json.loads((folder/'status.json').read_text())
        if record.get('model_version',1)!=MODEL_VERSION or record.get('execution_version',1)!=EXECUTION_VERSION:
            preliminary.append(record);continue
        records.append(record)
        years.append(pd.read_csv(folder/'annual.csv'))
    for case in ['10m_fixed100','10m_learned']:
        folder=Path('artifacts/contracts/p25_capital_sensitivity/replays')/case
        r=json.loads((folder/'status.json').read_text());r.update(case='frozen_'+case)
        records.append(r)
    result=pd.DataFrame(records)
    result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(preliminary).to_csv(REPORT/'preliminary_comparison.csv',index=False)
    if years:pd.concat(years,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    columns=[c for c in ['case','annualized_excess_sharpe','entries','proposals','learned_skips','closed_cohorts','net_excess_pnl'] if c in result]
    print(result[columns].to_string(index=False),flush=True)
    note=('PAPER-INSPIRED ACTUAL-CONTRACT MEAN-REVERSION Q EXPERIMENT\n\n'
        'Retained 30D 25-delta puts; all 36,327 cached candidate states across 125 pairs and 161 assets. '
        'Evaluation candidate universe has 17,967 pair/date states. Original beta .85–1.15, correlation >=.5, '
        'and retrospective formation context retained. Old rank and profitability entry vetoes removed. '
        'Missing future paths exclude training rewards only, never evaluation candidates. Matched runs retain all original 608 proposals.\n\n'
        'Four recent IV-spread moves, binned by strictly prior change volatility, encode 256 patterns plus missing history. '
        'Add trailing relative-spread valuation and an empirical excursion-reversion bucket. Excursions use a prior mean '
        'and 2-SD threshold; completed, failed at 63 sessions, and ongoing/censored excursions are explicit. '
        'This causal excursion diagnostic is not an exact replication of the paper local-extreme EMRT estimator. '
        'No hedge-coefficient search; actual quantities remain entry-vega matched. Selected-bucket spread paths are signals; '
        'held contracts retain their own actual price paths, strikes and expiries.\n\n'
        'Monthly 18M fits require scheduled expiry strictly before fit; minimum 32 completed cohorts. '
        'Pooled tabular fitted Q, 40 Bellman iterations, gamma .99 per daily step. Held state includes age, direction, '
        'and initial size. Actions hold or paired close; entry chooses flat or either direction, at 100% for fixed '
        'or 50/75/100% for sized. Flat has zero episodic value; no modeled continuation into a future new trade. '
        'Actual historical daily option/hedge PnL shadow labels, not an oracle equilibrium reward or synthetic OU paths. '
        'Initial declared mean/variance utility r-.25*r*r/(2*training-only reward std), a risk proxy rather than exact Sharpe. '
        'Follow-up risk0 uses plain actual PnL to test whether the added utility penalty suppresses entries. '
        'A further management-only control preserves all original entry directions and learned sizes. '
        'Its locked-inventory attribution control retains the original 212 accepted dates and initial integer counts, '
        'still subject to risk checks; those original execution counts are a counterfactual control, not a live predictor. '
        'Final model version3 includes failed exit attempts as continuing held transitions and unfilled '
        'integer-size entries as zero-exposure outcomes, rather than conditioning entry estimates on successful fills. '
        'Final execution version3 avoids an elective order when a known leg expiry would precede its next-session fill, '
        'matching the paired shadow-close transition. '
        'Initial executable-only close estimates and their replays are retained separately as preliminary diagnostics. '
        'Categorical pooling handles sparse patterns without activating/deactivating individual pairs. '
        'Cold start uses the original extreme-rank rule at full size and holds; no profitability veto.\n\n'
        'Daily management decides after close, fills next session only with executable quotes. Both legs close together '
        'when both survive. Unequal expiries retain original own-expiry runoff; no continuous vega rebalancing, reversals '
        'of existing inventory, trims or rolls. Independent fixed-integer shadow cohorts approximate training; final '
        'evaluation is the existing reconciled shared-cash/shared-stock-hedge account with NAV-scaled integer counts '
        'and unchanged original risk limits. Option commission .65 in training, .03 in evaluation; .1 vol point entry '
        'and elective exit penalty, stock hedging 1bp, floating cash and original borrow/dividend conventions.\n\n'
        '$10m equity, Jan2018–Aug2025, 2025 partial. Annualized daily cash-excess Sharpe, sqrt252. '
        'All evaluation dates already explored; no untouched holdout. Sequential mean-reversion timing is exploratory '
        'and cannot be described as an improvement until the matched and broad comparisons support it. '
        'Previous checkpoints preserved. One worker, 50% of one core, 3GiB RSS stop.\n\n'+result[columns].to_string(index=False)+'\n')
    (REPORT/'findings.txt').write_text(note)
    fits=[json.loads(p.read_text()) for p in OUT.glob('models*/*.json')]
    assert all(r['last_scheduled_expiry'] is None or pd.Timestamp(r['last_scheduled_expiry'])<pd.Timestamp(r['fit']) for r in fits)
    dump(REPORT/'audit.json',dict(status='complete' if all((OUT/'replays'/result_name(c)/'status.json').exists() for c in CASES) else 'partial',
        completed_cases=[r['case'] for r in records],prior_checkpoint_unchanged=True,
        fit_cutoffs_verified=len(fits),original_anchor_verified=(REPORT/'anchor.json').exists(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def validate():
    """Audit final policy outputs and decompose financial costs, not refit."""
    from scripts.run_p25_capital_sensitivity import metrics
    records=[];admissions=[];holding=[]
    cal=pd.read_parquet(ROOT/'market/rates.parquet').index
    market_rows=pd.read_parquet(OUT/'rows.parquet')
    expected=market_rows.loc[market_rows.date.ge('2018-01-01')&market_rows.date.le('2025-06-30'),['date','pair']]
    for folder in sorted((OUT/'replays').iterdir()):
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if status.get('model_version')!=MODEL_VERSION or status.get('execution_version')!=EXECUTION_VERSION:continue
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        rows=pd.read_parquet(folder/'scored_allocations.parquet')
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        assert len(decisions)==len(rows)==status['proposals']
        assert len(nav)==len(cal[cal>='2018-01-01'])
        assert nav.index.equals(cal[cal>='2018-01-01'])
        np.testing.assert_allclose(metrics(nav,10_000_000)['annualized_excess_sharpe'],status['annualized_excess_sharpe'],atol=1e-12)
        if folder.name.startswith('universe'):
            pd.testing.assert_frame_equal(rows[['date','pair']].sort_values(['date','pair']).reset_index(drop=True),
                expected.sort_values(['date','pair']).reset_index(drop=True))
        entered=decisions.status.eq('entered')
        if entered.any():
            a,b=decisions.loc[entered,'vega_a'],decisions.loc[entered,'vega_b']
            assert ((a-b).abs()/np.maximum(a,b)<=.1+1e-12).all()
        if len(trades):
            np.testing.assert_allclose(trades.commission,trades.quantity.abs()*.03*trades.reason.isin(['entry','announced_corporate_action_exit','learned_policy_exit','learned_policy_trim']),atol=1e-10)
            assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
            exits=trades.reason.eq('learned_policy_exit')
            if exits.any():assert (pd.to_datetime(trades.loc[exits,'exit_signal_date'])<trades.loc[exits,'date']).all()
            first=trades.loc[trades.reason.eq('entry')].groupby('cohort').date.min()
            last=trades.groupby('cohort').date.max().reindex(first.index)
            durations=(last-first).dt.days
            mean_days=float(durations.mean())
            median_days=float(durations.median())
        else:mean_days=median_days=0
        commissions=float(trades.commission.sum()) if len(trades) else 0
        penalties=float(trades.execution_penalty.sum()) if len(trades) else 0
        # option_pnl includes option execution cash flows and is already net
        # of commissions/penalties. Add them back to obtain gross midpoint PnL.
        gross=float((nav.option_pnl+nav.hedge_gross_pnl+nav.dividend_cash).sum())+commissions+penalties
        stock=float(nav.hedge_cost.sum());borrow=float(nav.stock_borrow_cost.sum())
        excess_interest=float((nav.interest-nav.cash_benchmark.diff().fillna(0)).sum())
        excess=gross-commissions-penalties-stock-borrow+excess_interest
        np.testing.assert_allclose(excess,status['net_excess_pnl'],atol=1e-6,rtol=0)
        records.append(dict(case=folder.name,gross_hedged_pnl=gross,option_commissions=commissions,
            option_penalty=penalties,stock_execution=stock,stock_borrow=borrow,excess_financing=excess_interest,
            net_excess_pnl=excess,mean_holding_calendar_days=mean_days,median_holding_calendar_days=median_days))
        for state,count in decisions.status.value_counts().items():admissions.append(dict(case=folder.name,status=state,count=int(count)))
        holding.append(dict(case=folder.name,entries=int(entered.sum()),mean_holding_calendar_days=mean_days,median_holding_calendar_days=median_days))
    pd.DataFrame(records).to_csv(REPORT/'pnl_components.csv',index=False)
    pd.DataFrame(admissions).to_csv(REPORT/'admissions.csv',index=False)
    pd.DataFrame(holding).to_csv(REPORT/'holding.csv',index=False)
    protect()
    audit=json.loads((REPORT/'audit.json').read_text())
    audit.update(executable_replays_verified=len(records),all_universe_candidates_preserved=True,
        option_costs_verified=True,entry_vega_tolerance_verified=True,next_session_exits_verified=True,
        pnl_components_reconcile=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(REPORT/'audit.json',audit)
    paths=[Path(__file__),Path('scripts/p25_reversion_q.py'),Path('tests/test_p25_reversion_q.py'),
        Path('config/p25_reversion_q_research.json'),Path('config/p25_reversion_q_followup.json')]
    paths+=list(OUT.glob('models_v3*/*.joblib'))+list(OUT.glob('replays/*_v3*/daily_nav.parquet'))
    paths+=list(OUT.glob('*.npy'))+[OUT/'rows.parquet']+list((OUT/'pair_states').glob('*.parquet'))
    dump(REPORT/'source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(f'Validated {len(records)} final replays: candidate coverage, costs, vega matching, next-session exits and PnL reconcile',flush=True)


def inspect():
    rows = pd.read_parquet('artifacts/contracts/p25_feature_search/training_rows.parquet')
    print(json.dumps(dict(rows=len(rows), pairs=rows.pair.nunique(), assets=len(set(rows.secid_a)|set(rows.secid_b)),
        start=str(rows.date.min()), end=str(rows.date.max()), finite_labels=int(rows.finite_label.sum()),
        original_signal_eligible=int(rows.signal_eligible.sum()),
        evaluation_candidates=int(((rows.date>='2018-01-01')&(rows.date<='2025-06-30')).sum()),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2))
    print(rows[['date','pair','end','label_row','finite_label','signal_eligible']].head().to_string(index=False))
    for pair in [rows.pair.iloc[0], '101121__108321']:
        a=np.load(ROOT/'pairs'/pair/'P25_labels.npy',mmap_mode='r')
        print(pair,a.shape)


def main():
    global RISK_PENALTY
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',default='inspect',choices=['inspect','prepare','fit','replay','summarize','all','test','anchor','validate'])
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--risk-penalty',type=float,choices=[0.,.25],default=.25)
    parser.add_argument('--both-rewards',action='store_true')
    args=parser.parse_args()
    RISK_PENALTY=args.risk_penalty
    if args.phase=='inspect':inspect();return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    (OUT/'pair_states').mkdir(exist_ok=True)
    protect()
    if args.both_rewards:
        if args.phase!='all':raise ValueError('--both-rewards requires --phase all')
        prepare();anchor()
        for penalty in [.25,0.]:
            RISK_PENALTY=penalty
            fit()
            for case in CASES:replay(case)
            summarize()
        validate();protect();progress('complete',reward_penalties=[.25,0.])
        return
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_p25_reversion_q'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    if args.phase in ['prepare','all']:prepare()
    if args.phase in ['fit','all']:fit()
    if args.phase in ['anchor','all']:anchor()
    if args.phase in ['replay','all']:
        anchor()
        for case in ([args.case] if args.case else CASES):replay(case)
    if args.phase in ['summarize','all']:summarize()
    if args.phase=='validate':validate()
    protect()
    progress('phase_complete',phase=args.phase)


if __name__=='__main__':main()
