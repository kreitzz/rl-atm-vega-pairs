"""Controlled one-year hold/prune experiment; one guarded worker, cached data only."""
import argparse
from pathlib import Path
import json
import gc
import time
import resource
import hashlib
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from scripts import run_native_year_entry as native
from scripts import run_long_tenor_policy_search as shadow
from scripts import year_reversion_management as qlearn
from scripts.audit_long_expiry_parity import BoundMarket

OUT=Path('artifacts/contracts/year_reversion_management')
REPORT=Path('reports/year_reversion_management')
ANCHORS={'learned':'forecast_rl_sized','fixed50':'forecast_variance_fixed50'}


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str))


def protect():
    path=OUT/'protected_sources.json'
    if path.exists():
        files=json.loads(path.read_text())
        for name,want in files.items():assert shadow.digest(name)==want,name
    else:
        paths=[Path('reports/native_year_entry/comparison.csv'),Path('config/p25_variance_premium_candidate.json')]
        for name in ANCHORS.values():paths+=list((native.OUT/'replays'/name).glob('*'))
        paths+=list(Path('slides').glob('*.pptx'))
        dump(path,{str(p):shadow.digest(p) for p in paths if p.is_file()})


def progress(stage,**details):
    path=OUT/'progress.json'
    prior=json.loads(path.read_text()).get('peak_rss_bytes',0) if path.exists() else 0
    dump(path,dict(stage=stage,peak_rss_bytes=max(prior,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**details))


def setup():
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    native.configure_native(); protect()
    # Shadow labels need only our compact state columns, not old entry states.
    shadow.ENTRY_FEATURES=[]
    shadow.MANAGEMENT_FEATURES=['paper_base','carry_bin','age_bin']


def input_variance(market):
    close=(market.spots*market.adjustments).where(market.spots.gt(0)&market.adjustments.gt(0))
    r=np.log(close/close.shift(1))
    result={}
    for half,minobs in [(20,20),(60,60),(252,126)]:
        result[half]=252*r.pow(2).ewm(halflife=half,min_periods=minobs,adjust=False).mean()
    return result


def state_values(date,held,history,variance):
    if len(held)!=2:return None
    a,b=held
    ivs=[]
    for p in held:
        quote=p['quote']
        value=float(quote['bs_iv'])
        ivs.append(value if quote.get('bs_status')!='causal_held_mark' and value>0 else np.nan)
    values=history.update(ivs[0]-ivs[1])
    estimates=[]
    expiry=max(p['expiry'] for p in held)
    slow=max(0,(expiry-date).days)/(max(0,(expiry-date).days)+90)
    for p in held:
        sid=int(p['secid'])
        recent=[float(variance[k].at[date,sid]) for k in [20,60,252]]
        estimates.append(slow*recent[2]+(1-slow)*(.4*recent[0]+.6*recent[1]))
    carry=np.nan
    if np.isfinite([*ivs,*estimates]).all():
        carry=np.sign(a['quantity'])*((estimates[0]-ivs[0]**2)/(2*ivs[0])-(estimates[1]-ivs[1]**2)/(2*ivs[1]))
    values['carry_bin']=3 if not np.isfinite(carry) else 0 if carry<-.01 else 2 if carry>.01 else 1
    entry=max(p['entry_date'] for p in held)
    age=(date-entry).days/max(1,(expiry-entry).days)
    values['age_bin']=0 if age<.25 else 1 if age<.5 else 2
    values['carry']=carry
    return values


def replay_book(rows,market,cal,callback,locked=None):
    return native.old.long.replay(rows,market,cal,365,callback,locked_counts=locked)


def observe(market,variance,records):
    histories={}
    def callback(date,held):
        key=held[0]['cohort']
        if len(held)==2:
            hist=histories.setdefault(key,qlearn.SpreadHistory())
            values=state_values(date,held,hist,variance)
        else:values=None
        for p in held:
            raw=market.quote(date,p['optionid']); quote=p['quote']
            executable=raw is not None and shadow.valid_quote(raw) and shadow.valid_greeks(raw) and raw.bid>0
            record=dict(date=date,cohort=key,secid=p['secid'],quantity=p['quantity'],initial_quantity=p['initial_quantity'],
                option_value=p['quantity']*100*(quote['bid']+quote['ask'])/2,
                hedge=-p['quantity']*100*quote['delta'],gross_vega=abs(p['quantity'])*100*quote['vega'],
                exit_executable=executable,state_known=values is not None,split_seen=False,
                **(values or dict(paper_base=256*9+5,carry_bin=3,age_bin=2)))
            records.append(record)
        return 1.
    return callback


def prepare():
    marker=OUT/'prepared.json'
    if marker.exists():return
    market=BoundMarket(native.OUT); variance=input_variance(market)
    cal=market.rates.index[market.rates.index>='2018-01-01']
    rows=pd.read_parquet(native.OUT/'replays/forecast_variance_fixed50/scored_allocations.parquet')
    rows=rows.sort_values(['date','pair']).reset_index(drop=True)
    last_year=None
    def rate(date):
        nonlocal last_year
        if last_year!=date.year:
            last_year=date.year
            print(f'Training observation {date.date()}',flush=True)
            progress('observe',date=str(date.date()))
        return float(market.rates.loc[date])
    market.rate=rate
    observed=[]
    print('Observe audited fixed-50 forecast portfolio, 2018–2025',flush=True)
    nav,fills,decisions,audit=replay_book(rows,market,cal,observe(market,variance,observed))
    pd.testing.assert_frame_equal(nav,pd.read_parquet(native.OUT/'replays/forecast_variance_fixed50/daily_nav.parquet'),rtol=1e-12,atol=1e-6)
    raw=pd.DataFrame(observed); raw.to_parquet(OUT/'held_states.parquet',index=False)
    fills.to_csv(OUT/'training_fills.csv',index=False)
    decisions.to_csv(REPORT/'training_admissions.csv',index=False)
    lookup={f'{r.date.date()}|{r.pair}|{r.wing}':rows.loc[i] for i,r in rows.iterrows()}
    groups={key:g for key,g in raw.groupby('cohort',sort=False)}
    episodes=[]; records=[]; recon=np.zeros((len(cal),3)); coverage=[]
    for number,(cohort,trades) in enumerate(fills.groupby('cohort',sort=False)):
        if not trades.reason.eq('entry').any():continue
        group=groups[cohort]
        ep=shadow.shadow_episode(cohort,group,trades,market,cal,lookup[cohort],native.old.configuration(365))
        ep['can_trim']=np.zeros(len(ep['dates']),bool)
        for date,g in group.groupby('date',sort=False):
            if len(g)!=2:continue
            counts=g.initial_quantity.abs().to_numpy()
            remain=np.floor(counts*.5)
            fractions=remain/counts
            ep['can_trim'][ep['dates'].get_loc(date)]=bool((remain>=1).all() and (max(fractions)-min(fractions)<=.1))
        # The observer includes unavailable features explicitly, without dropping dates.
        np.add.at(recon,ep['positions'],np.stack([ep['option_pnl'],ep['hedge_gross_pnl'],ep['dividends']],axis=1))
        records.extend(qlearn.transitions(ep,len(episodes)))
        episodes.append(ep)
        known=group.drop_duplicates('date')
        coverage.append(dict(cohort=cohort,pair=ep['pair'],entry=str(ep['dates'][0].date()),
            daily_states=len(known),move_history_available=int(known.get('path_code',pd.Series(dtype=float)).lt(256).sum()),
            completed_reversion_available=int(known.get('reversion_days',pd.Series(dtype=float)).notna().sum())))
        if len(episodes)%100==0:print(f'Prepared {len(episodes)} held-contract episodes',flush=True)
    np.testing.assert_allclose(recon,nav[['option_pnl','hedge_gross_pnl','dividend_cash']],rtol=1e-10,atol=1e-6)
    joblib.dump(dict(episodes=episodes,calendar=cal),OUT/'episodes.joblib')
    arrays=qlearn.pack(records)
    np.savez(OUT/'transitions.npz',**arrays)
    pd.DataFrame(coverage).to_csv(REPORT/'state_coverage.csv',index=False)
    dump(marker,dict(training_portfolio='Audited2018-forward forecast fixed50, all eligible monthly proposals; actual execution/risk acceptance only',
         candidate_dates=len(rows),episodes=len(episodes),action_transitions=len(records),
         gross_components_reconciled=True,training_commission=.65,actual_commission=.03,
         earliest_management_training='2018-01-01; pre2018 unsupported corporate-action adjustment; cold start holds',
         training_shadow='Fractional marginal hedge/borrow labels; integer trim feasibility retained; shared netting evaluated in final account',
         no_future_filter_on_evaluation=True))
    progress('prepared',episodes=len(episodes),transitions=len(records))


def fit():
    with np.load(OUT/'transitions.npz') as disk:data={k:disk[k] for k in disk.files}
    records=[]
    for cutoff in pd.date_range('2018-01-01','2025-07-01',freq='QS-JAN'):
        use=qlearn.causal_prefix(data,cutoff)
        count=len(np.unique(use['cohort']))
        for variant in qlearn.VARIANTS:
            path=OUT/'models'/variant/f'{cutoff.date()}.joblib'
            note=path.with_suffix('.json')
            if note.exists():records.append(json.loads(note.read_text()));continue
            model=qlearn.fit_q(use,variant) if count>=16 and len(use['reward'])>=640 else None
            path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,path)
            record=dict(fit=str(cutoff.date()),variant=variant,training_cohorts=count,
                transitions=len(use['reward']),last_training_step=str(pd.Timestamp(int(use['end'].max())).date()) if len(use['end']) else None,
                window_months=48,fitted=model is not None)
            dump(note,record); records.append(record)
        print(f'Quarterly Q fit {cutoff.date()}: {count} past cohorts, {len(use["reward"])} transitions',flush=True)
        progress('fit',cutoff=str(cutoff.date())); gc.collect()
    pd.DataFrame(records).to_csv(REPORT/'fits.csv',index=False)


def callback(schedule,variance,cal,decision_records=None):
    histories={}; positions={d:i for i,d in enumerate(cal)}
    def policy(date,held):
        if len(held)!=2:return 1.
        key=held[0]['cohort']; hist=histories.setdefault(key,qlearn.SpreadHistory())
        values=state_values(date,held,hist,variance)
        age=positions[date]-positions[held[0]['entry_date']]
        current=min(p['management_fraction'] for p in held)
        if age<5 or age%5 or positions[date]+1>=len(cal):return current
        # Expiry is known today: never request a paired order that would fill
        # after one leg has already run off.
        if cal[positions[date]+1]>=min(p['exit_date'] for p in held):return current
        fitted=schedule[max(k for k in schedule if k<=date)]
        target=current
        if fitted is not None:
            state=np.array([[values[k] for k in ['paper_base','carry_bin','age_bin']]])
            target=float(qlearn.choose(fitted,state,np.array([current]))[0])
        if decision_records is not None:
            decision_records.append(dict(date=date,cohort=key,current=current,target=target,**values))
        return target
    return policy


def save_replay(case,entry_policy,variant=None,locked=False):
    folder=OUT/'replays'/case
    if (folder/'status.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    root=native.OUT/'replays'/ANCHORS[entry_policy]
    rows=pd.read_parquet(root/'scored_allocations.parquet')
    locked_counts=None
    if locked:
        fills=pd.read_csv(root/'trades.csv'); entry=fills[fills.reason.eq('entry')]
        groups={key:g.set_index('secid') for key,g in entry.groupby('cohort')}
        keys=rows.apply(lambda r:f'{r.date.date()}|{r.pair}|{r.wing}',axis=1)
        rows=rows.loc[keys.isin(groups)].copy()
        locked_counts={f'{r.date.date()}|{r.pair}|{r.wing}':tuple(groups[f'{r.date.date()}|{r.pair}|{r.wing}'].loc[[r.secid_a,r.secid_b],'quantity'].abs()) for r in rows.itertuples()}
    market=BoundMarket(native.OUT); cal=market.rates.index[market.rates.index>='2018-01-01']
    market.rate=lambda date:float(market.rates.loc[date])
    record=[]
    if variant is None:hook=lambda date,held:1.
    else:
        schedule={pd.Timestamp(p.stem):joblib.load(p) for p in sorted((OUT/'models'/variant).glob('*.joblib'))}
        hook=callback(schedule,input_variance(market),cal,record)
    print(f'Replay {case}',flush=True)
    nav,trades,decisions,audit=replay_book(rows,market,cal,hook,locked_counts)
    if variant is None:
        expected=pd.read_parquet(root/'daily_nav.parquet')
        pd.testing.assert_frame_equal(nav,expected,rtol=1e-12,atol=1e-6)
        old=pd.read_csv(root/'trades.csv')
        pd.testing.assert_frame_equal(trades.reset_index(drop=True).assign(date=lambda x:x.date.astype(str)),old,check_dtype=False,rtol=1e-12,atol=1e-6)
    log=audit.pop('management_log',[])
    dump(folder/'management_log.json',log)
    nav.to_parquet(folder/'daily_nav.parquet'); trades.to_csv(folder/'trades.csv',index=False)
    decisions.to_csv(folder/'decisions.csv',index=False); rows.to_parquet(folder/'scored_allocations.parquet',index=False)
    pd.DataFrame(record).to_parquet(folder/'management_decisions.parquet',index=False)
    entered=decisions.loc[decisions.status.eq('entered')]
    if locked_counts is not None:
        grouped={key:g.set_index('secid') for key,g in trades.loc[trades.reason.eq('entry')].groupby('cohort')}
        for r in rows.itertuples():
            key=f'{r.date.date()}|{r.pair}|{r.wing}'
            if key in grouped:
                assert tuple(grouped[key].loc[[r.secid_a,r.secid_b],'quantity'].abs())==locked_counts[key]
    status=dict(case=case,entry_policy=entry_policy,state=variant or 'expiry_hold',locked_inventory=locked,
        **shadow.metrics(nav,10_000_000),entries=len(entered),proposals=len(rows),active_pairs=int(entered.pair.nunique()),
        trimmed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_trim'),'cohort'].nunique()),
        closed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique()),**audit)
    dump(folder/'status.json',status)
    pd.DataFrame(shadow.annual_metrics(nav,case,10_000_000)).to_csv(folder/'annual.csv',index=False)
    print(f'{case}: excess Sharpe {status["annualized_excess_sharpe"]:.4f}; {len(entered)} entries; {status["trimmed_cohorts"]} trims / {status["closed_cohorts"]} closes',flush=True)
    progress('replay',case=case,sharpe=status['annualized_excess_sharpe'])


def replay():
    # Hold hooks must reproduce saved NAV before accepting management results.
    for entry in ANCHORS:save_replay(entry+'_hold',entry)
    save_replay('learned_hold_locked','learned',locked=True)
    for entry in ANCHORS:
        for variant in qlearn.VARIANTS:save_replay(entry+'_'+variant,entry,variant)
    for variant in qlearn.VARIANTS:save_replay('learned_'+variant+'_locked','learned',variant,True)


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.report_long_tenor_search import paired_bootstrap
    result=[]; annual=[]; paired=[]; components=[]; admissions=[]
    for folder in sorted((OUT/'replays').glob('*')):
        if not (folder/'status.json').exists():continue
        status=json.loads((folder/'status.json').read_text()); result.append(status)
        annual.append(pd.read_csv(folder/'annual.csv'))
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        np.testing.assert_allclose(shadow.metrics(nav,10_000_000)['annualized_excess_sharpe'],status['annualized_excess_sharpe'])
        trades=pd.read_csv(folder/'trades.csv')
        decisions=pd.read_csv(folder/'decisions.csv')
        scored=pd.read_parquet(folder/'scored_allocations.parquet')
        assert len(decisions)==len(scored)==status['proposals']
        if not status['locked_inventory']:
            pd.testing.assert_frame_equal(scored,pd.read_parquet(native.OUT/'replays'/ANCHORS[status['entry_policy']]/'scored_allocations.parquet'))
        entered=decisions.loc[decisions.status.eq('entered')]
        if len(entered):
            assert ((entered.vega_a-entered.vega_b).abs()/np.maximum(entered.vega_a,entered.vega_b)<=.1+1e-12).all()
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        paid=trades.reason.isin(['entry','learned_policy_trim','learned_policy_exit','announced_corporate_action_exit'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],trades.loc[paid,'quantity'].abs()*.03,atol=1e-10)
        managed=trades.reason.isin(['learned_policy_trim','learned_policy_exit'])
        if managed.any():
            assert (pd.to_datetime(trades.loc[managed,'exit_signal_date'])<pd.to_datetime(trades.loc[managed,'date'])).all()
            assert trades.loc[managed].groupby(['cohort','date']).secid.nunique().eq(2).all()
        assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
        commissions=float(trades.commission.sum()); penalties=float(trades.execution_penalty.sum())
        gross=float((nav.option_pnl+nav.hedge_gross_pnl+nav.dividend_cash).sum())+commissions+penalties
        excess_interest=float((nav.interest-nav.cash_benchmark.diff().fillna(0)).sum())
        excess=gross-commissions-penalties-nav.hedge_cost.sum()-nav.stock_borrow_cost.sum()+excess_interest
        np.testing.assert_allclose(excess,status['net_excess_pnl'],atol=1e-6,rtol=0)
        dated=trades.assign(date=pd.to_datetime(trades.date))
        first=dated.loc[dated.reason.eq('entry')].groupby('cohort').date.min()
        last=dated.groupby('cohort').date.max().reindex(first.index)
        duration=(last-first).dt.days
        components.append(dict(case=folder.name,gross_hedged_pnl=gross,option_commissions=commissions,
            option_penalty=penalties,stock_execution=float(nav.hedge_cost.sum()),stock_borrow=float(nav.stock_borrow_cost.sum()),
            excess_financing=excess_interest,net_excess_pnl=excess,mean_holding_days=float(duration.mean()),median_holding_days=float(duration.median())))
        for name,count in decisions.status.value_counts().items():admissions.append(dict(case=folder.name,status=name,count=int(count)))
        if status['state']!='expiry_hold':
            baseline=pd.read_parquet(OUT/'replays'/(status['entry_policy']+'_hold')/'daily_nav.parquet')
            gain=status['annualized_excess_sharpe']-shadow.metrics(baseline,10_000_000)['annualized_excess_sharpe']
            lo,hi=paired_bootstrap(nav.excess_return.iloc[1:],baseline.excess_return.iloc[1:],replicates=1000)
            paired.append(dict(case=folder.name,sharpe_gain=gain,unadjusted_95_low=lo,unadjusted_95_high=hi))
    df=pd.DataFrame(result); df.to_csv(REPORT/'comparison.csv',index=False)
    assert len(df)==12,'All predeclared live and locked controls must finish'
    annual_table=pd.concat(annual,ignore_index=True)
    annual_table.to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(components).to_csv(REPORT/'pnl_components.csv',index=False)
    pd.DataFrame(admissions).to_csv(REPORT/'admissions.csv',index=False)
    pd.DataFrame(paired).to_csv(REPORT/'paired_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,entry in zip(axes,ANCHORS):
        d=df[df.entry_policy.eq(entry)&~df.locked_inventory].set_index('case').loc[[entry+'_hold']+[entry+'_'+v for v in qlearn.VARIANTS]]
        ax.bar(['Hold','Carry Q','Paper Q','Both Q'],d.annualized_excess_sharpe,color=['#66717e','#337e65','#d48236','#2463a6'])
        for i,v in enumerate(d.annualized_excess_sharpe):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,5 if v>=0 else -12),textcoords='offset points',ha='center')
        ax.axhline(0,color='#888',lw=.7); ax.spines[['top','right']].set_visible(False)
        ax.set_title('Learned entry sizing' if entry=='learned' else 'Fixed 50% entry')
        ax.set_ylabel('Annualized cash-excess Sharpe')
    fig.tight_layout(); fig.savefig(REPORT/'comparison.png',dpi=180); fig.savefig(REPORT/'comparison.svg'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for case,color,label in [('learned_hold','#66717e','Hold'),('learned_carry','#337e65','Carry Q'),
                             ('learned_paper','#d48236','Paper Q'),('learned_combined','#2463a6','Both Q')]:
        rows=annual_table.loc[annual_table.case.eq(case)].sort_values('year')
        for ax,field in zip(axes,['annualized_excess_sharpe','return_above_cash_pp']):
            ax.plot(rows.year,rows[field],marker='o',markersize=3,color=color,label=label,lw=1.4)
    for ax,label in zip(axes,['Annualized excess Sharpe','Return above cash (percentage points)']):
        ax.set_ylabel(label); ax.axhline(0,color='#888',lw=.7); ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(range(2018,2026),[str(y)+('*' if y==2025 else '') for y in range(2018,2026)],rotation=40)
    axes[0].legend(frameon=False,fontsize=8); axes[0].set_title('Learned entry sizing: annual management results')
    axes[1].set_title('*2025 partial through August 29')
    fig.tight_layout(); fig.savefig(REPORT/'annual.png',dpi=180); fig.savefig(REPORT/'annual.svg'); plt.close(fig)
    cols=['case','annualized_excess_sharpe','entries','trimmed_cohorts','closed_cohorts','net_excess_pnl']
    text=('ONE-YEAR PAPER-INSPIRED HOLD/PRUNE EXPERIMENT\n\n'
      'Approximately365D P25 puts, existing native variance-forecast directions and entry policies frozen. $10m funded shared ledger;2018-Aug2025. Three predeclared management representations: carry-only, paper-inspired recent held-IV spread movements/reversion history, both. Quarterly48-month prefix training;24 pooled Bellman iterations;discount .98/five sessions;hold/half/close, weekly next-session fills. No new entry or pair veto. No re-expansion. Known expiry prevents one-leg elective orders. Daily stock delta hedges and initial entry-vega matching; later Greeks can drift.\n'
      'Paper states use the actual held-contract IV spread, not a constant-maturity index. Four moves are standardized by prior126-session change volatility with20 observations;prior126-session valuation needs63 observations. Excursions begin beyond2SD of the prior mean;completed,failed after63sessions,and censored observations retained. Sparse/missing states pool toward last-move/valuation/reversion/age states; no individual pair activation. Carry is a remaining-DTE-weighted recent/slow realized-variance proxy.\n'
      'Training episodes come from audited2018-forward fixed50 forecast-account admissions; pre2018 reconstruction hit an unsupported2016 corporate-action adjustment and is not used. Cold start holds. Every training transition ends strictly before its fit, so completed prefixes of still-open trades are usable. Shadow rewards charge .65 commissions and fractional marginal hedge/borrow costs, with observed next-fill and integer-trim feasibility. Fractional trim shadow rewards approximate rounded real exposures; final replays use actual integer counts, shared hedge netting/cash,and .03 commissions. Same .1vol-point/leg elective-entry/exit penalty,1bp stock hedges,financing and risk limits.\n'
      'Locked original learned-entry cohorts/counts are counterfactual attribution controls, not live predictors; risk checks remain. Plain PnL utility is not direct portfolio-Sharpe optimization. This is a paper-inspired adaptation, not EMRT coefficient optimization or synthetic/oracle-mean replication. Prior checkpoint NAV/trades preserved and hold hooks reproduced exactly. All history has been researched; no fresh holdout or selection-adjusted significance.\n\n'+df[cols].to_string(index=False)+'\n')
    (REPORT/'findings.txt').write_text(text)
    protect()
    fits=pd.read_csv(REPORT/'fits.csv'); fitted=fits.loc[fits.last_training_step.notna()]
    assert (pd.to_datetime(fitted.last_training_step)<pd.to_datetime(fitted.fit)).all()
    files=[Path(__file__),Path('scripts/year_reversion_management.py'),Path('config/year_reversion_management.json'),OUT/'transitions.npz',OUT/'held_states.parquet']
    files+=list((OUT/'models').glob('*/*.joblib'))
    dump(REPORT/'source_manifest.json',{str(p):shadow.digest(p) for p in files})
    dump(REPORT/'audit.json',dict(status='complete',replays=len(df),original_hold_nav_trades_reproduced=True,
         prior_sources_unchanged=True,training_cutoffs_verified=True,option_commissions_verified=True,
         position_roundtrips_verified=True,paired_next_session_management_verified=True,
         entry_vega_tolerance_verified=True,original_proposals_preserved=True,pnl_components_reconcile=True,
         peak_rss_bytes=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,json.loads((OUT/'progress.json').read_text()).get('peak_rss_bytes',0)),cpu_core_fraction=.5,workers=1))
    print(df[cols].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['prepare','fit','replay','summarize','all'],default='all')
    args=parser.parse_args(); setup()
    for stage,fn in [('prepare',prepare),('fit',fit),('replay',replay),('summarize',summarize)]:
        if args.phase in [stage,'all']:fn()
    protect()


if __name__=='__main__':main()
