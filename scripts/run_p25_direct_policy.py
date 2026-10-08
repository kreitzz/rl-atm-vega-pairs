"""Matched-objective direct sizing experiment. Serial, 20% duty, 384 MiB RSS.

Run with .venv-pca312: python -m scripts.run_p25_direct_policy.
Original artifacts are read-only; all outputs use p25_direct_policy directories.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import gc
import io
import json
import shutil
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_representation as base
from scripts.diagnose_p25_pca import training_window
from scripts.p25_direct_policy import make_policy, shadow_scorer, fit_direct
from scripts.p25_representation import state_book, fit_menu
from scripts.run_p25_additive_sizing import perturb
from scripts.run_p25_threshold_search import ThresholdMarket
from scripts.analyze_p25_opportunity_filter import analyze, paired_sharpe_interval

ROOT=base.ROOT
OUT=ROOT/'artifacts/contracts/p25_direct_policy'
REPORT=ROOT/'reports/p25_direct_policy'
OLD=ROOT/'artifacts/contracts/p25_strategy_audit/sizing/replays'
PRIOR=ROOT/'artifacts/contracts/p25_additive_sizing/replays'
KINDS={'direct_linear':'linear','direct_additive':'gam'}
CONTROLS={'tree_cem':OLD/'tree_cem','fixed100':OLD/'fixed100',
          'regression_linear':PRIOR/'linear_score','regression_additive':PRIOR/'additive_main'}
QUOTES=ROOT/'artifacts/contracts/p25_opportunity_filter/quotes.sqlite'
CONFIG=base.BUNDLE/'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json'
FILES=['daily_nav.parquet','trades.csv','decisions.csv','scored_allocations.parquet']
# Predeclared optimization diagnostics, not alternative policies selected on OOS results.
SEED_MONTHS={'2018-01-01','2019-07-01','2021-01-01','2022-07-01','2024-01-01','2025-06-01'}


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():raise InterruptedError('Direct policy STOP requested')


def load_inputs():
    inputs=json.loads(base.CONFIG.read_text())['screened']['inputs']
    cols=inputs+['date','end','pair','signal_eligible','finite_label','target','position','label_row','direction']
    meta=pd.read_parquet(base.SOURCE,columns=cols,use_threads=False)
    signals=pd.read_parquet(OLD/'tree_cem/scored_allocations.parquet',use_threads=False)
    missing=[f for f in inputs if f not in signals]
    if missing:signals=signals.merge(meta[['date','pair']+missing],on=['date','pair'],validate='one_to_one')
    assert len(signals)==608 and not signals.duplicated(['date','pair']).any()
    flags=[f for f in inputs if f.endswith('_available')]
    assert signals[flags].eq(1).all().all()
    cal=pd.read_parquet(base.MARKET/'rates.parquet',use_threads=False).index
    manifest=json.loads((ROOT/'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    return meta,signals,[f for f in inputs if f not in flags],inputs,cal,manifest


def prepare(meta,features,budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    files=[base.SOURCE,base.CONFIG,base.ARCHIVE,CONFIG,QUOTES,base.MARKET/'rates.parquet',ROOT/'config/p25_submission.json']
    files += [ROOT/'scripts'/name for name in ['p25_direct_policy.py','run_p25_direct_policy.py',
        'p25_additive_sizing.py','p25_representation.py','diagnose_p25_pca.py','uniform_smile_engine.py','uniform_smile_risk.py']]
    files += [folder/name for folder in CONTROLS.values() for name in FILES]
    used=meta.loc[meta.signal_eligible&meta.finite_label&meta.date.ge('2016-07-01')&meta.date.lt('2025-06-01')]
    files += [base.LABELS/pair/'P25_labels.npy' for pair in sorted(used.pair.unique())]
    hashes={}
    for p in files:
        hashes[str(p.relative_to(ROOT))]=base.digest(p);budget.rest()
    protocol=dict(cases=list(CONTROLS)+list(KINDS),features=features,
        outer='90 monthly fits Jan 2018-Jun 2025; same 18-month eligible completed rows; date/end strictly before fit.',
        objective='Unpenalized annualized sample Sharpe of summed original directional action-specific daily label paths, including zero calendar days. Exact same shadow objective as retained tree CEM.',
        cost_scope='Original legacy training paths remain unchanged for all policies; funded costs and risk rules also identical. Existing training-versus-funded cost and admission mismatch remains.',
        basis='Same 15 economic variables. Linear centered/unit-SD terms; additive degree-2 splines with 3 training quantile knots, centered/unit group RMS. No interactions. Linear extrapolation.',
        action='Dimensionless additive score: below -0.5 => 50%, [-0.5,0.5) => 75%, >=0.5 => 100%. Intercept and coefficient scale learned directly; no payoff-regression target or score terciles.',
        optimizer=dict(iterations=50,population=192,evaluations=9600,seed=2026,elites=19,
            coefficient_bounds=[-3,3],smoothing=[.3,.7],sd_floor_fraction=.03,
            start='Zero mean; intercept SD 1; each feature basis coefficient SD 1/sqrt(15). Full-size best anchor plus all three constants injected each iteration.',
            caveat='New policies use diagonal Gaussian CEM; original uses categorical CEM over leaf actions. Equal candidate counts do not imply equal optimization difficulty, effective capacity or convergence.'),
        complexity='Fixed low-degree bases and coefficient bounds only; no additive penalty changes the common Sharpe objective. No hyperparameter or model-family selection from evaluation outcomes.',
        controls='Archived tree and fixed100 plus prior regression linear/additive for descriptive pipeline comparisons. Reproduce every original monthly tree CEM choice and objective.',
        execution='Same 608 original proposals and 20% gate; same contracts, 10m funded account, current costs, hedges, admission, continuing inventory.',
        seeds=dict(months=sorted(SEED_MONTHS),extra_seeds=[2027,2028],scope='Training-only optimization sensitivity; never used to choose deployed seed or estimate OOS seed robustness.'),
        stability='Same +/-5% training-SD one-variable perturbations and adjacent-fit common training examples as prior experiment. Recommendations, not held-position turnover.',
        inference='Paired circular blocks 63/126 sessions, 2000 draws, seed 2026. Conditional on fitted policies; no correction for repeated research.',
        historical_status='2018-21 earlier; 2022-Aug29 2025 later. Both already inspected; no untouched holdout and no automatic promotion.',
        resources=dict(threads=1,duty=.2,rss_limit_mib=384),hashes=hashes)
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen direct-policy protocol changed')
    if not path.exists():base.write(path,protocol)
    for case,src in CONTROLS.items():
        dest=OUT/'replays'/case;dest.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            if not (dest/name).exists():shutil.copyfile(src/name,dest/name)
            assert base.digest(dest/name)==base.digest(src/name)
    return protocol


def train(meta,signals,features,inputs,cal,manifest,budget,limit=None):
    fits=[];allocations=[];contributions=[];perturbations=[];refits=[];seeds=[];previous={}
    signature=base.digest(REPORT/'protocol.json')
    months=pd.period_range('2018-01','2025-06',freq='M')
    if limit:months=months[:limit]
    with zipfile.ZipFile(base.ARCHIVE) as archive:
        for month in months:
            begun=time.monotonic();cutoff=month.start_time;stamp=str(cutoff.date())
            rows=training_window(meta,cutoff).reset_index(drop=True)
            group=signals.loc[signals.date.dt.to_period('M').eq(month)].copy()
            reference=joblib.load(io.BytesIO(base.archive_read(archive,base.MODEL_ROOT+stamp+'.joblib',manifest)))
            scorer=shadow_scorer(rows,cal,base.LABELS)
            leaves=reference['tree'].apply(rows[inputs]);states=np.searchsorted(reference['leaves'],leaves)
            book=state_book(rows,states,len(reference['leaves']),cal,base.LABELS)
            choices,tree_detail=fit_menu(book)
            np.testing.assert_array_equal(choices,reference['choices'])
            tree_sizes=choices[states]
            np.testing.assert_allclose(scorer.score(tree_sizes)[0],tree_detail['training_shadow_daily_sharpe'],rtol=1e-10,atol=1e-10)
            fits.append(dict(case='tree_cem',fit=stamp,training_rows=len(rows),latest_train_end=str(rows.end.max()),
                parameters=len(choices),evaluations=9600,seed=2026,exact_archived_choices=True,**tree_detail))
            for case,kind in KINDS.items():
                folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{stamp}.joblib';audit_path=folder/f'{stamp}.json'
                if path.exists():
                    model=joblib.load(path);audit=json.loads(audit_path.read_text())
                    assert audit['protocol_sha256']==signature
                else:
                    model,details=fit_direct(make_policy(rows,features,kind),rows,scorer,rest=budget.rest)
                    audit=dict(case=case,fit=stamp,training_rows=len(rows),latest_train_end=str(rows.end.max()),
                        protocol_sha256=signature,**details)
                    joblib.dump(model,path);base.write(audit_path,audit)
                fits.append({k:v for k,v in audit.items() if k not in ['trace','protocol_sha256']})
                if stamp in SEED_MONTHS:
                    for seed in [2027,2028]:
                        sp=folder/f'{stamp}_seed{seed}.json'
                        if sp.exists():
                            diag=json.loads(sp.read_text());assert diag['protocol_sha256']==signature
                        else:
                            alt,detail=fit_direct(make_policy(rows,features,kind),rows,scorer,seed=seed,rest=budget.rest)
                            diag=dict(case=case,fit=stamp,protocol_sha256=signature,
                                assignment_disagreement=float(np.mean(alt.size_codes(rows)!=model.size_codes(rows))),
                                primary_shadow_sharpe=audit['training_shadow_daily_sharpe'],**detail)
                            base.write(sp,diag)
                        seeds.append({k:v for k,v in diag.items() if k not in ['trace','protocol_sha256']})
                if len(group):
                    own=group[['date','pair']].copy();own['case']=case;own['score']=model.predict(group)
                    own['learned_level']=model.size_codes(group);allocations.append(own)
                    c=model.contributions(group).reset_index(drop=True)
                    c.insert(0,'case',case);c.insert(0,'pair',group.pair.to_numpy());c.insert(0,'date',group.date.to_numpy())
                    contributions.append(c)
                perturbations+=perturb(case,stamp,model,group,rows,features,lambda m,f:m.size_codes(f))
                if case in previous:
                    old,oldrows=previous[case]
                    keys=pd.MultiIndex.from_frame(rows[['date','pair']]);oldkeys=pd.MultiIndex.from_frame(oldrows[['date','pair']])
                    common=rows.loc[keys.isin(oldkeys)]
                    diff=abs(model.size_codes(common)-old.size_codes(common))/4
                    refits.append(dict(case=case,fit=stamp,n_common=len(common),changed=int((diff>0).sum()),total_abs_fraction_change=float(diff.sum())))
                previous[case]=(model,rows)
                budget.rest()
            base.write(OUT/'progress.json',dict(stage='training',fit=stamp,models_complete=2*len(fits)//3,peak_rss_mib=budget.peak))
            print(f'{stamp}: {len(rows)} completed rows; original choices reproduced; two direct policies saved ({time.monotonic()-begun:.1f}s)',flush=True)
            del scorer,book;gc.collect();budget.rest()
    pd.DataFrame(fits).to_csv(REPORT/'fits.csv',index=False)
    pd.DataFrame(seeds).to_csv(REPORT/'seed_diagnostics.csv',index=False)
    pd.DataFrame(perturbations).to_csv(REPORT/'perturbation.csv',index=False)
    pd.DataFrame(refits).to_csv(REPORT/'refit_stability.csv',index=False)
    pd.concat(allocations,ignore_index=True).to_parquet(OUT/'allocations.parquet',index=False)
    pd.concat(contributions,ignore_index=True).to_parquet(OUT/'contributions.parquet',index=False)


def replay(signals,cal,budget):
    scores=pd.read_parquet(OUT/'allocations.parquet',use_threads=False)
    cfg=base.Settings(**json.loads(CONFIG.read_text()));checkpoint=json.loads((ROOT/'config/p25_submission.json').read_text())
    for case in KINDS:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),allocations_sha256=base.digest(OUT/'allocations.parquet'))
        if (folder/'status.json').exists():
            status=json.loads((folder/'status.json').read_text());assert all(status[k]==v for k,v in signature.items());continue
        own=scores.loc[scores.case.eq(case),['date','pair','learned_level']]
        assert len(own)==608
        frame=signals.drop(columns='learned_level').merge(own,on=['date','pair'],validate='one_to_one')
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
        market.db.close();del market,nav,trades,decisions;gc.collect();budget.rest()


def report(protocol,budget):
    cases=list(CONTROLS)+list(KINDS)
    periods=analyze({case:OUT/'replays'/case for case in cases},REPORT/'funded',reference='tree_cem',rest=budget.rest)
    records=[]
    for candidate,reference in [('direct_additive','direct_linear'),('direct_linear','regression_linear'),('direct_additive','regression_additive')]:
        a=pd.read_parquet(OUT/'replays'/candidate/'daily_nav.parquet',use_threads=False).excess_return
        b=pd.read_parquet(OUT/'replays'/reference/'daily_nav.parquet',use_threads=False).excess_return
        for period,start,end in [('validation','2018-01-01','2021-12-31'),('evaluation','2022-01-01','2025-08-29')]:
            x,y=a.loc[start:end],b.loc[start:end]
            if x.index[0]==a.index[0]:x,y=x.iloc[1:],y.iloc[1:]
            for block in [63,126]:records.append(dict(case=candidate,reference=reference,period=period,**paired_sharpe_interval(x,y,block=block,rest=budget.rest)))
    pd.DataFrame(records).to_csv(REPORT/'matched_intervals.csv',index=False)
    for p,expected in protocol['hashes'].items():
        assert base.digest(ROOT/p)==expected;budget.rest()
    base.write(REPORT/'completion.json',dict(status='complete',primary_monthly_models=180,tree_controls_reproduced=90,
        additional_training_seed_fits=24,new_funded_replays=2,proposals_per_case=608,peak_rss_mib=budget.peak,protected_sources_unchanged=True))
    print(periods.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','train','replay','report'],default='all')
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.limit and args.phase!='train':raise ValueError('--limit is training-only')
    budget=Budget();meta,signals,features,inputs,cal,manifest=load_inputs();protocol=prepare(meta,features,budget)
    if args.phase in ['all','train']:train(meta,signals,features,inputs,cal,manifest,budget,args.limit)
    if args.phase in ['all','replay']:replay(signals,cal,budget)
    if args.phase in ['all','report']:report(protocol,budget)


if __name__=='__main__':main()
