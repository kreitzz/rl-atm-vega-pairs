"""Separate entry-bandit and capital-Q effects using the same two signals.

One guarded worker per entry family. Rebuild strategy-dependent state inputs
and resizing costs, refit Q on that family's returns, and verify both endpoints.
The fixed-full/50:50 comparison is primary; all fixed mixes and sizes are saved.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import signal_allocation_variance as learner
from scripts import signal_capital_allocation as core

CONFIG=Path('config/signal_learning_factorial.json')
CFG=json.loads(CONFIG.read_text())
ROOT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
COLUMNS=['nav','return_','excess_return','cash_benchmark','capital_reserve','gross_vega','gross_hedge_notional']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.partial.json')
    pending.write_text(json.dumps(value,indent=2,default=str)+'\n');pending.replace(path)


def load(entry):
    import joblib
    return joblib.load(ROOT/entry/'prepared.joblib')


def prepare(entry):
    import joblib
    target=ROOT/entry/'prepared.joblib'
    source_files=[CONFIG,Path(CFG['reference_root'])/'prepared.joblib',Path('data/raw/VIX_History.csv')]
    for folder in CFG['entries'][entry].values():source_files.extend([Path(folder)/'daily_nav.parquet',Path(folder)/'trades.csv'])
    signatures={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest=ROOT/entry/'sources.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signatures,'Inputs changed; use a new study directory'
    else:save(manifest,signatures)
    if target.exists():return
    reference=joblib.load(Path(CFG['reference_root'])/'prepared.joblib')
    books={key:pd.read_parquet(Path(folder)/'daily_nav.parquet',columns=COLUMNS,use_threads=False) for key,folder in CFG['entries'][entry].items()}
    pair,straddle=books['put'],books['straddle']
    assert pair.index.equals(straddle.index) and pair.index.equals(reference['pair'].index)
    np.testing.assert_allclose(pair.nav.iloc[0],CFG['capital'],atol=1e-6)
    np.testing.assert_allclose(straddle.nav.iloc[0],CFG['capital'],atol=1e-6)
    cash=core.aligned_cash(pair,straddle)
    np.testing.assert_allclose(cash,reference['cash'],atol=1e-14,rtol=0)
    if entry!='bandit':
        put_reference=pd.read_csv('reports/p25_validation_split/periods.csv')
        straddle_reference=pd.read_csv('reports/straddle_policy_refinement/comparison.csv')
        for period,start,end in core.PERIODS:
            put_period={'evaluation':'retrospective_holdout','full':'full_history'}.get(period,period)
            expected_put=put_reference.loc[(put_reference['case']=='screened_'+entry)&(put_reference.period==put_period)].iloc[0]
            expected_straddle=straddle_reference.loc[(straddle_reference['case']==entry)&(straddle_reference.period==period)].iloc[0]
            for book,expected in [(pair,expected_put),(straddle,expected_straddle)]:
                observed=core.stats(book.excess_return,cash,start,end)['annualized_excess_sharpe']
                assert abs(observed-expected.annualized_excess_sharpe)<1e-8,(entry,period,observed,expected.annualized_excess_sharpe)
    costs=[]
    for key,book in [('put',pair),('straddle',straddle)]:
        trades=pd.read_csv(Path(CFG['entries'][entry][key])/'trades.csv',usecols=['date','cohort','optionid','quantity'],parse_dates=['date'])
        counts=core.engine.contract_counts(trades,book.index)
        costs.append((.1*book.gross_vega+.0001*book.gross_hedge_notional+.03*counts)/book.nav)
    rates=np.column_stack(costs)
    known=core.engine.market_features(pair,straddle)
    relative=(known.atm_sharpe63-known.pair_sharpe63).replace([np.inf,-np.inf],np.nan).fillna(0).clip(-5,5)
    gross=np.column_stack([1+pair.return_,1+straddle.return_,1+cash])
    periods=[];before=core.TARGETS.copy()
    for original in reference['periods']:
        indices=original['indices'];prior=int(indices[0])-1
        market=original['market'].copy()
        # Economic opportunity inputs stay identical. Rebuild risk/performance
        # inputs from the strategy indices actually available in this family.
        if prior>=0:
            assert pair.index[prior]==original['feature_date']<original['cutoff']
            market[4:7]=known[['log_vol_ratio','correlation63','vix']].replace([np.inf,-np.inf],np.nan).fillna(0).iloc[prior].to_numpy()
            market[7]=relative.iloc[prior]
        rate=rates[indices[0]];week_gross=gross[indices];week_cash=cash.iloc[indices].to_numpy()
        paths=np.stack([[core.engine.month_path(week_gross,week_cash,rate,p,a)[0] for a in core.TARGETS] for p in before])
        after=np.stack([core.engine.drift(a,week_gross[1:]) for a in core.TARGETS])
        m=dict(cutoff=original['cutoff'],end=original['end'],feature_date=original['feature_date'],market=market,
               indices=indices,gross=week_gross,cash=week_cash,costs=rate,before=before.copy(),after=after,paths=paths)
        if entry=='bandit':
            for key in ['market','gross','cash','costs','before','after','paths']:
                np.testing.assert_allclose(m[key],original[key],atol=1e-12,rtol=0,err_msg=key)
        periods.append(m);before=after
    data=dict(pair=pair,straddle=straddle,cash=cash,periods=periods)
    pending=target.with_suffix('.partial.joblib');joblib.dump(data,pending);pending.replace(target)
    # Reuse only the existing bandit-family models. Every fixed-entry family
    # trains its own new Q models on its own completed reward paths.
    if entry=='bandit':
        for path in Path(CFG['bandit_models']).glob('*.json'):
            dest=ROOT/entry/'models'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    save(REPORT/entry/'preparation.json',dict(rows=len(pair),weekly_periods=len(periods),features=core.FEATURES+['previous_straddle_weight'],
        economic_features_identical=True,strategy_risk_features_recomputed=True,resizing_costs_recomputed=True,
        bandit_reference_reproduced=entry=='bandit',source_hashes=signatures))
    print('Prepared '+entry,flush=True)


def replay(entry):
    prepare(entry);data=load(entry);periods=data['periods']
    learner.CFG=dict(learner.CFG,**CFG);learner.ROOT=ROOT/entry;learner.REPORT=REPORT/entry
    risks=np.asarray(CFG['risk_preferences']);n=len(risks)
    model=None;month=None;previous=np.full(n,3,int)
    paths=[[] for _ in risks];actions=[[] for _ in risks];timing=[]
    for index,m in enumerate(periods):
        key=str(m['cutoff'].to_period('M'))
        if key!=month or model is None:model=learner.fit(data,index);month=key
        if model:assert pd.Timestamp(model['last_training_end'])<m['cutoff']
        for j,risk in enumerate(risks):
            action=3
            if model:
                state=np.r_[m['market'],m['before'][previous[j],1]]
                score=core.design(np.tile(state,(core.N,1)),np.arange(core.N),np.asarray(model['center']),np.asarray(model['scale']))@np.asarray(model['beta'])[:,j]
                action=int(score.argmax())
            paths[j].append(m['paths'][previous[j],action]);actions[j].append(action);previous[j]=action
        timing.append(dict(date=m['cutoff'],last_training_end=model['last_training_end'] if model else None))
        if index%100==0:print(json.dumps(dict(entry=entry,week=index)),flush=True)
    cash=np.concatenate([m['cash'] for m in periods]);indices=np.concatenate([m['indices'] for m in periods]);dates=data['pair'].index[indices]
    records=[];annual=[];decision_rows=[];endpoints=[]
    def record(path,allocator,case,risk=None,weight=None):
        excess=np.concatenate(path);nav=CFG['capital']*np.cumprod(1+excess+cash)
        book=pd.DataFrame(dict(excess_return=excess,cash_return=cash,nav=nav),index=dates)
        folder=ROOT/entry/'replays';folder.mkdir(parents=True,exist_ok=True)
        pending=folder/(case+'.partial.parquet');book.to_parquet(pending);pending.replace(folder/(case+'.parquet'))
        meta=dict(entry_policy=entry,allocator=allocator,case=case,variance_penalty=risk,straddle_target=weight)
        for period,start,end in core.PERIODS:records.append(dict(**meta,period=period,**core.metrics(book,start,end)))
        for year in range(2018,2026):annual.append(dict(**meta,year=year,**core.metrics(book,f'{year}-01-01',min(f'{year}-12-31',CFG['evaluation_end']))))
        return book
    for j,risk in enumerate(risks):
        record(paths[j],'Q',f'q_lambda{risk:g}',risk=float(risk))
        for m,action,t in zip(periods,actions[j],timing):decision_rows.append(dict(entry_policy=entry,variance_penalty=risk,date=m['cutoff'],straddle_weight=float(core.SHARES[action]),last_training_end=t['last_training_end']))
    for action,weight in enumerate(core.SHARES):
        book=record([m['paths'][action,action] for m in periods],'fixed',f'fixed_{weight:g}',weight=float(weight))
        if weight in [0,1]:
            original=data['pair' if weight==0 else 'straddle']
            np.testing.assert_allclose(book.excess_return,original.excess_return,atol=1e-12,rtol=0)
            np.testing.assert_allclose(book.nav,original.nav,atol=1e-5,rtol=0)
            endpoints.append(dict(straddle_target=float(weight),source_nav_and_returns_reproduced=True))
    frame=pd.DataFrame(records)
    if entry=='bandit':
        refq=pd.read_csv('reports/signal_allocation_variance/results.csv')
        reff=pd.read_csv('reports/signal_allocation_variance/fixed_results.csv')
        for r in frame.itertuples():
            expected=(refq.loc[(refq.variance_penalty==r.variance_penalty)&(refq.period==r.period)] if r.allocator=='Q'
                      else reff.loc[(reff.straddle_target==r.straddle_target)&(reff.period==r.period)]).iloc[0]
            assert abs(r.annualized_excess_sharpe-expected.annualized_excess_sharpe)<1e-8
            assert abs(r.profit_above_cash-expected.profit_above_cash)<1e-5
    folder=REPORT/entry;folder.mkdir(parents=True,exist_ok=True)
    frame.to_csv(folder/'results.csv',index=False);pd.DataFrame(annual).to_csv(folder/'annual.csv',index=False)
    pd.DataFrame(decision_rows).to_csv(folder/'decisions.csv',index=False)
    save(folder/'verification.json',dict(entry=entry,endpoints=endpoints,monthly_model_count=len(list((ROOT/entry/'models').glob('*.json'))),
        training_timing_verified=True,bandit_results_reproduced=entry=='bandit',fresh_family_specific_Q=entry!='bandit'))
    print(frame.loc[frame.period.eq('evaluation')].to_string(index=False),flush=True)


def analyze():
    available=[entry for entry in CFG['entries'] if (REPORT/entry/'results.csv').exists()]
    assert CFG['primary_fixed_entry'] in available and 'bandit' in available
    frames=[pd.read_csv(REPORT/entry/'results.csv') for entry in available]
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(REPORT/'all_results.csv',index=False)
    primary=CFG['primary_fixed_entry'];weight=CFG['fixed_allocation_straddle_weight'];rows=[];effects=[]
    for period,_,_ in core.PERIODS:
        for risk in CFG['risk_preferences']:
            cells={}
            for entry,entry_label in [(primary,'Fixed'),('bandit','Bandit')]:
                for allocator in ['fixed','Q']:
                    mask=frame.entry_policy.eq(entry)&frame.allocator.eq(allocator)&frame.period.eq(period)
                    mask &= frame.straddle_target.eq(weight) if allocator=='fixed' else frame.variance_penalty.eq(risk)
                    r=frame.loc[mask].iloc[0].to_dict();r.update(cell=entry_label+' entry / '+('fixed' if allocator=='fixed' else 'Q')+' allocation',comparison_risk=risk)
                    rows.append(r);cells[(entry_label,allocator)]=r
            for metric in ['annualized_excess_sharpe','profit_above_cash']:
                ff=cells['Fixed','fixed'][metric];bf=cells['Bandit','fixed'][metric]
                fq=cells['Fixed','Q'][metric];bq=cells['Bandit','Q'][metric]
                effects.append(dict(period=period,variance_penalty=risk,metric=metric,bandit_effect_fixed_allocation=bf-ff,Q_effect_fixed_entries=fq-ff,
                    Q_effect_bandit_entries=bq-bf,bandit_effect_Q_allocation=bq-fq,interaction=bq-bf-fq+ff))
    four=pd.DataFrame(rows);four.to_csv(REPORT/'four_way.csv',index=False);pd.DataFrame(effects).to_csv(REPORT/'effects.csv',index=False)
    sensitivity=frame.loc[frame.allocator.eq('Q')|frame.straddle_target.eq(weight)].copy()
    sensitivity.to_csv(REPORT/'sizing_sensitivity.csv',index=False)
    save(REPORT/'protocol.json',dict(configuration=CFG,primary='Fixed full entries and fixed 50/50 capital allocation. No later-period optimization of either fixed control.',
        Q='Same original nine feature definitions, seven capital splits, lambda grid, gamma, ridge and training windows. Refit independently for each fixed-entry family.',
        feature_control='Four economic opportunity features shared. Relative volatility, correlation and relative trailing Sharpe recomputed from that family of strategy indices, using only prior-close data.',
        scope='Fractional strategy-index allocation above independently funded option ledgers. The entry intervention includes admission/inventory effects. No claim of a shared integer-contract portfolio.',
        interaction='Sharpe difference-of-differences is descriptive, not an additive causal attribution.',
        completed_entry_families=available,paper_unchanged=True))
    print(four.loc[(four.period!='full')&(four.comparison_risk==CFG['reference_risk_preference']),['cell','period','annualized_excess_sharpe','profit_above_cash']].to_string(index=False),flush=True)


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.read_csv(REPORT/'four_way.csv');frame=frame.loc[frame.comparison_risk.eq(CFG['reference_risk_preference'])]
    labels=['Fixed entry\nFixed allocation','Bandit entry\nFixed allocation','Fixed entry\nQ allocation','Bandit entry\nQ allocation']
    cells=['Fixed entry / fixed allocation','Bandit entry / fixed allocation','Fixed entry / Q allocation','Bandit entry / Q allocation']
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.21,top=.78,wspace=.15)
    colors=['#9ba5af','#56a4b0','#40678e','#087fa4']
    for ax,period,title in zip(axes,['validation','evaluation'],['2018–2021 validation','2022–August 2025']):
        part=frame.loc[frame.period.eq(period)].set_index('cell').loc[cells]
        values=part.annualized_excess_sharpe.to_numpy()
        ax.bar(np.arange(4),values,color=colors,width=.65)
        for i,(v,p) in enumerate(zip(values,part.profit_above_cash)):
            ax.annotate(f'{v:.3f}',(i,v),xytext=(0,7 if v>=0 else -15),textcoords='offset points',ha='center',weight='bold',fontsize=10)
            ax.text(i,.025,f'${p/1e6:.3f}m',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='white',weight='bold')
        ax.set_xticks(np.arange(4),labels,fontsize=8.5);ax.set_title(title,loc='left',fontsize=12,pad=14)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Annualized excess Sharpe');axes[0].margins(y=.22)
    fig.text(.08,.945,'Separating entry sizing from capital allocation',fontsize=17,weight='bold')
    fig.text(.08,.88,'Fixed entries: 100% size · Fixed capital: 50/50 · Q: original state, γ = 0.99, λ = 0.20',fontsize=10,color='#516473')
    fig.text(.08,.05,'Dollar labels show cumulative profit above cash. Both learning layers retain the same proposals, costs and calendar.',fontsize=9,color='#657681')
    for ext in ['png','pdf','svg']:fig.savefig(REPORT/('four_way.'+ext),dpi=120,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['prepare','run','analyze','charts','sensitivities','summarize']);p.add_argument('--entry',choices=list(CFG['entries']))
    args=p.parse_args()
    if args.phase in ['prepare','run']:
        if args.entry is None:p.error('--entry is required')
        (prepare if args.phase=='prepare' else replay)(args.entry)
    elif args.phase=='sensitivities':
        for entry in ['fixed75','fixed50']:replay(entry)
    elif args.phase=='summarize':analyze();charts()
    else:globals()[args.phase]()
