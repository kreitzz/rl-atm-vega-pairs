"""Causal additive sizing: matched linear, spline mains, and selected interactions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[k]='1'
import argparse
import gc
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_representation as base
from scripts.run_p25_threshold_search import ThresholdMarket
from scripts.diagnose_p25_pca import training_window
from scripts.p25_additive_sizing import fit_model
from scripts.analyze_p25_opportunity_filter import analyze,paired_sharpe_interval

ROOT=base.ROOT
OUT=ROOT/'artifacts/contracts/p25_additive_sizing'
REPORT=ROOT/'reports/p25_additive_sizing'
OLD=ROOT/'artifacts/contracts/p25_strategy_audit/sizing/replays'
KINDS={'linear_score':'linear','additive_main':'gam','additive_pairs':'gam_interactions'}
CASES=['tree_cem','fixed100']+list(KINDS)
ALPHAS=[.1,1.,10.]
QUOTES=ROOT/'artifacts/contracts/p25_opportunity_filter/quotes.sqlite'
CONFIG=base.BUNDLE/'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json'


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():raise InterruptedError('Additive sizing STOP requested')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    features=[f for f in json.loads(base.CONFIG.read_text())['screened']['inputs'] if not f.endswith('_available')]
    files=[base.SOURCE,base.CONFIG,CONFIG,QUOTES,ROOT/'config/p25_submission.json',
        ROOT/'scripts/run_p25_additive_sizing.py',ROOT/'scripts/p25_additive_sizing.py',
        ROOT/'scripts/uniform_smile_engine.py',ROOT/'scripts/uniform_smile_risk.py']
    for case in ['tree_cem','fixed100']:
        files += [OLD/case/name for name in ['daily_nav.parquet','trades.csv','decisions.csv','scored_allocations.parquet']]
    protocol=dict(cases=CASES,features=features,alpha_multipliers=ALPHAS,
        interactions=[['rank_strength','correlation'],['rank_strength','vix'],['profit_365d','count_365d']],
        model='Linear ridge, degree2 spline additive mains (3 training quantile knots), and same mains plus3 residualized tensor interactions; ridge n*alpha; linear spline extrapolation.',
        outer='90 monthly fits Jan2018-Jun2025; same18month signal window with signal/end strictly before fit, original qualified completed rows.',
        inner='Two chronological3month validation folds in preceding6months; each uses its own18month completed training window. Validation outcomes must be completed before outer fit; min32train and8validation. Require both folds elsealpha1. Choose pooled validation MSE; exact ties prefer stronger penalty.',
        size_head='All3 new models use their outer-training score terciles: below first50%, middle75%, above second100%; right-sided ties. Smooth score, discrete sizes. No CEM for these models.',
        controls='Original tree/CEM and fixed100% archived replays. Additive-vs-linear isolates function family under matched head; versus tree compares whole pipelines.',
        isolation='Same608proposals, original20%opportunityrank/profit/count gates, contracts,10mcapital,costs,riskrules,dailyhedges,andcontinuousholdings. No threshold-search5%rule combined.',
        target='Original fullsize standalone lifetime dollar profit, including legacy.65optioncommission; funded replays use.03commission. This objective mismatch remains common across regression targets.',
        stability='On actual proposals perturb each economic feature alone by+/-5% outertrain populationSD. Record sizing changes; synthetic local sensitivity, not plausible joint scenarios. Compare adjacent fits on shared historical training rows. Originaltree evaluated too.',
        evaluation='2018-21 earlier,2022-Aug29,2025 later; already researched history, not untouchedholdout. No automatic strategy promotion.',
        inference='Paired circular63/126session blocks2000replications, conditional exploratory intervals without multiple-research adjustment.',
        resources=dict(threads=1,duty=.2,rss_mib=384),hashes={str(p.relative_to(ROOT)):base.digest(p) for p in files})
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol changed')
    base.write(path,protocol)
    columns=json.loads(base.CONFIG.read_text())['screened']['inputs']+['date','end','pair','signal_eligible','finite_label','target']
    meta=pd.read_parquet(base.SOURCE,columns=columns)
    signals=pd.read_parquet(OLD/'tree_cem/scored_allocations.parquet')
    missing=[f for f in json.loads(base.CONFIG.read_text())['screened']['inputs'] if f not in signals]
    if missing:signals=signals.merge(meta[['date','pair']+missing],on=['date','pair'],validate='one_to_one')
    assert len(signals)==608 and not signals.duplicated(['date','pair']).any()
    for case in ['tree_cem','fixed100']:
        dest=OUT/'replays'/case;dest.mkdir(parents=True,exist_ok=True)
        for name in ['daily_nav.parquet','trades.csv','decisions.csv','scored_allocations.parquet']:
            shutil.copyfile(OLD/case/name,dest/name)
    return meta,signals,features,protocol


def choose_alpha(meta,features,kind,cutoff,budget):
    folds=[];details=[]
    for offset in [6,3]:
        start=cutoff-pd.DateOffset(months=offset);stop=start+pd.DateOffset(months=3)
        train=training_window(meta,start)
        valid=meta.loc[meta.signal_eligible&meta.finite_label&meta.date.ge(start)&meta.date.lt(stop)&meta.end.lt(cutoff)]
        usable=len(train)>=32 and len(valid)>=8
        details.append(dict(start=str(start.date()),stop=str(stop.date()),training_rows=len(train),validation_rows=len(valid),usable=usable,
            latest_train_end=str(train.end.max()) if len(train) else None,latest_validation_end=str(valid.end.max()) if len(valid) else None))
        if usable:folds.append((train,valid))
    scores=[]
    if len(folds)==2:
        for alpha in ALPHAS:
            squared=[]
            for train,valid in folds:
                model=fit_model(train,features,kind=kind,alpha_multiplier=alpha)
                squared.extend((model.predict(valid)-valid.target.to_numpy())**2)
                budget.rest()
            scores.append(dict(alpha_multiplier=alpha,mse=float(np.mean(squared)),observations=len(squared)))
        chosen=min(scores,key=lambda r:(r['mse'],-r['alpha_multiplier']))['alpha_multiplier']
        status='two_fold_selected'
    else:chosen=1.;status='insufficient_inner_data_fallback'
    return dict(alpha_multiplier=chosen,cv_status=status,folds=details,scores=scores)


def perturb(case,stamp,model,group,train,features,predict_sizes):
    output=[]
    if not len(group):return output
    original=np.asarray(predict_sizes(model,group),float)
    for feature in features:
        sd=float(train[feature].std(ddof=0))
        if not np.isfinite(sd) or sd<1e-12:continue
        changed=0;total=0.
        for sign in [-1,1]:
            altered=group.copy();altered[feature]+=sign*.05*sd
            diff=abs(np.asarray(predict_sizes(model,altered),float)-original)/4
            changed+=int((diff>0).sum());total+=float(diff.sum())
        output.append(dict(case=case,fit=stamp,feature=feature,n=2*len(group),changed=changed,total_abs_fraction_change=total))
    return output


def train(meta,signals,features,budget):
    import io,zipfile
    manifest=json.loads((ROOT/'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    tree_inputs=json.loads(base.CONFIG.read_text())['screened']['inputs']
    fits=[];perturbations=[];stability=[];allocations=[];contributions=[];previous={}
    signature=base.digest(REPORT/'protocol.json')
    with zipfile.ZipFile(base.ARCHIVE) as archive:
        for month in pd.period_range('2018-01','2025-06',freq='M'):
            cutoff=month.start_time;stamp=str(cutoff.date())
            rows=training_window(meta,cutoff).reset_index(drop=True)
            group=signals.loc[signals.date.dt.to_period('M').eq(month)].copy()
            tree=joblib.load(io.BytesIO(base.archive_read(archive,base.MODEL_ROOT+stamp+'.joblib',manifest)))
            def tree_sizes(model,frame):
                own=frame.copy()
                for f in tree_inputs:
                    if f not in own:raise ValueError(f'Missing original tree input: {f}')
                action=dict(zip(model['leaves'],model['choices']))
                return np.array([action[leaf] for leaf in model['tree'].apply(own[tree_inputs])])
            if len(group):np.testing.assert_array_equal(tree_sizes(tree,group),group.learned_level)
            for case in ['tree_cem']+list(KINDS):
                if case=='tree_cem':
                    model=tree;size_fn=tree_sizes
                else:
                    folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
                    file=folder/f'{stamp}.joblib';audit_path=folder/f'{stamp}.json'
                    if file.exists():
                        model=joblib.load(file);audit=json.loads(audit_path.read_text())
                        assert audit['protocol_sha256']==signature
                    else:
                        selected=choose_alpha(meta,features,KINDS[case],cutoff,budget)
                        model=fit_model(rows,features,kind=KINDS[case],alpha_multiplier=selected['alpha_multiplier'])
                        audit=dict(case=case,fit=stamp,n_train=len(rows),latest_train_end=str(rows.end.max()),
                            protocol_sha256=signature,**selected)
                        joblib.dump(model,file);base.write(audit_path,audit)
                    size_fn=lambda m,f:m.size_codes(f)
                    record={k:v for k,v in audit.items() if k not in ['folds','scores','protocol_sha256']}
                    # Model-specific design diagnostics are saved without relying on internals.
                    record['terms']=len(model.contributions(rows.iloc[:1]).columns)-1
                    record['basis_columns']=model.metadata['basis_columns']
                    fits.append(record)
                    if len(group):
                        own=group[['date','pair']].copy();own['case']=case
                        own['prediction']=model.predict(group);own['learned_level']=model.size_codes(group)
                        allocations.append(own)
                        c=model.contributions(group).reset_index(drop=True)
                        c.insert(0,'case',case);c.insert(0,'pair',group.pair.to_numpy());c.insert(0,'date',group.date.to_numpy())
                        contributions.append(c)
                perturbations+=perturb(case,stamp,model,group,rows,features,size_fn)
                if case in previous:
                    old,oldrows=previous[case]
                    keys=pd.MultiIndex.from_frame(rows[['date','pair']]);oldkeys=pd.MultiIndex.from_frame(oldrows[['date','pair']])
                    common=rows.loc[keys.isin(oldkeys)]
                    if len(common):
                        diff=abs(np.asarray(size_fn(model,common))-np.asarray(size_fn(old,common)))/4
                        stability.append(dict(case=case,fit=stamp,n_common=len(common),changed=int((diff>0).sum()),total_abs_fraction_change=float(diff.sum())))
                previous[case]=(model,rows)
                budget.rest()
            print(f'{stamp}: {len(rows)} training rows; {len(group)} proposals; 3 additive/linear fits saved',flush=True)
            gc.collect();budget.rest()
    pd.DataFrame(fits).to_csv(REPORT/'fits.csv',index=False)
    pd.DataFrame(perturbations).to_csv(REPORT/'perturbation.csv',index=False)
    pd.DataFrame(stability).to_csv(REPORT/'refit_stability.csv',index=False)
    scores=pd.concat(allocations,ignore_index=True)
    scores.to_parquet(OUT/'allocations.parquet',index=False)
    pd.concat(contributions,ignore_index=True).to_parquet(OUT/'contributions.parquet',index=False)
    for case in KINDS:
        assert len(scores.loc[scores.case.eq(case)])==608


def replay(signals,budget):
    scores=pd.read_parquet(OUT/'allocations.parquet');cal=pd.read_parquet(base.MARKET/'rates.parquet').index
    cfg=base.Settings(**json.loads(CONFIG.read_text()));checkpoint=json.loads((ROOT/'config/p25_submission.json').read_text())
    for case in KINDS:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),allocations_sha256=base.digest(OUT/'allocations.parquet'))
        if (folder/'status.json').exists():
            status=json.loads((folder/'status.json').read_text());assert all(status[k]==v for k,v in signature.items());continue
        own=scores.loc[scores.case.eq(case),['date','pair','learned_level']]
        frame=signals.drop(columns='learned_level').merge(own,on=['date','pair'],validate='one_to_one')
        assert len(frame)==608 and frame.learned_level.isin([2,3,4]).all()
        frame.to_parquet(folder/'scored_allocations.parquet',index=False)
        market=ThresholdMarket(budget,frame,QUOTES);original=base.engine.book_risk
        print(f'Starting funded replay: {case}',flush=True)
        try:
            base.engine.book_risk=base.book_risk
            nav,trades,decisions,ledger=base.engine.simulate(market,frame,cal[cal>='2018-01-01'],cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=checkpoint['corporate_events'],wing_capital=1_000_000)
        finally:base.engine.book_risk=original
        base.mark_book(nav,market.rates,cfg.capital);audit=base.verify_ledger(nav,trades,decisions,frame)
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
        base.write(folder/'status.json',dict(status='complete',**signature,**audit,**ledger))
        market.db.close();del market,nav,trades,decisions
        gc.collect();budget.rest()


def report(protocol,budget):
    analyze({case:OUT/'replays'/case for case in CASES},REPORT/'funded',reference='tree_cem',rest=budget.rest)
    records=[]
    for candidate,reference in [('additive_main','linear_score'),('additive_pairs','additive_main')]:
        a=pd.read_parquet(OUT/'replays'/candidate/'daily_nav.parquet').excess_return
        b=pd.read_parquet(OUT/'replays'/reference/'daily_nav.parquet').excess_return
        for period,start,end in [('validation','2018-01-01','2021-12-31'),('evaluation','2022-01-01','2025-08-29')]:
            x,y=a.loc[start:end],b.loc[start:end]
            if x.index[0]==a.index[0]:x,y=x.iloc[1:],y.iloc[1:]
            for block in [63,126]:records.append(dict(case=candidate,reference=reference,period=period,**paired_sharpe_interval(x,y,block=block,rest=budget.rest)))
    pd.DataFrame(records).to_csv(REPORT/'matched_intervals.csv',index=False)
    for p,expected in protocol['hashes'].items():assert base.digest(ROOT/p)==expected
    base.write(REPORT/'completion.json',dict(status='complete',monthly_models=270,proposals_per_case=608,
        new_funded_replays=3,peak_rss_mib=budget.peak,protected_sources_unchanged=True))
    print(pd.read_csv(REPORT/'funded/periods.csv').to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','train','replay','report'],default='all')
    args=parser.parse_args();budget=Budget();meta,signals,features,protocol=prepare();budget.rest()
    if args.phase in ['all','train']:train(meta,signals,features,budget)
    if args.phase in ['all','replay']:replay(signals,budget)
    if args.phase in ['all','report']:report(protocol,budget)


if __name__=='__main__':main()
