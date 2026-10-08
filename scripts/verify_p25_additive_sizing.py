"""Read-only verification of causal additive sizing and saved funded replays."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'
import io
import json
from pathlib import Path
import zipfile
import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_representation as base
from scripts.diagnose_p25_pca import training_window
from scripts.submission_market import verify_ledger

ROOT=base.ROOT
OUT=ROOT/'artifacts/contracts/p25_additive_sizing'
REPORT=ROOT/'reports/p25_additive_sizing'
OLD=ROOT/'artifacts/contracts/p25_strategy_audit/sizing/replays'
NEW={'linear_score':'linear','additive_main':'gam','additive_pairs':'gam_interactions'}


def check_cv(audit,meta,cutoff):
    usable=[]
    for offset,stored in zip([6,3],audit['folds']):
        start=cutoff-pd.DateOffset(months=offset);stop=start+pd.DateOffset(months=3)
        train=training_window(meta,start)
        valid=meta.loc[meta.signal_eligible&meta.finite_label&meta.date.ge(start)&meta.date.lt(stop)&meta.end.lt(cutoff)]
        assert pd.Timestamp(stored['start'])==start and pd.Timestamp(stored['stop'])==stop
        assert len(train)==stored['training_rows'] and len(valid)==stored['validation_rows']
        enough=len(train)>=32 and len(valid)>=8
        assert stored['usable']==enough
        if len(train):
            assert train.date.lt(start).all() and train.end.lt(start).all()
            assert pd.Timestamp(stored['latest_train_end'])==train.end.max()
        if len(valid):
            assert valid.end.lt(cutoff).all()
            assert pd.Timestamp(stored['latest_validation_end'])==valid.end.max()
        if enough:usable.append((train,valid))
    assert len(audit['folds'])==2
    if len(usable)==2:
        assert audit['cv_status']=='two_fold_selected'
        scores=audit['scores']
        assert {r['alpha_multiplier'] for r in scores}=={.1,1.,10.}
        for row in scores:
            assert np.isfinite(row['mse']) and row['mse']>=0
            assert row['observations']==sum(len(valid) for _,valid in usable)
        expected=min(scores,key=lambda r:(r['mse'],-r['alpha_multiplier']))['alpha_multiplier']
        assert audit['alpha_multiplier']==expected
    else:
        assert audit['cv_status']=='insufficient_inner_data_fallback'
        assert audit['alpha_multiplier']==1. and not audit['scores']


def tree_codes(model,frame,tree_inputs):
    assert set(tree_inputs).issubset(frame.columns)
    choices=dict(zip(model['leaves'],model['choices']))
    return np.array([choices[leaf] for leaf in model['tree'].apply(frame[tree_inputs])])


def model_checks(model,rows,features):
    prediction=model.predict(rows)
    contributions=model.contributions(rows)
    np.testing.assert_allclose(contributions.sum(axis=1),prediction,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(model.training_score_edges,np.quantile(prediction,[1/3,2/3]),rtol=0,atol=1e-8)
    expected=np.searchsorted(model.training_score_edges,prediction,side='right')+2
    np.testing.assert_array_equal(model.size_codes(rows),expected)
    assert model.metadata['training_rows']==len(rows)
    assert model.metadata['basis_columns']==model.design(rows).shape[1]
    assert model.features==features
    groups=model.term_bases(rows)
    for feature in features:
        term=model.mains[feature]
        values=rows[feature].to_numpy(float)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(term.knots,np.unique(np.quantile(values,[0,.5,1])),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.training_mean[feature],values.mean(),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.training_sd[feature],values.std(ddof=0),rtol=0,atol=1e-12)
    for name,basis in groups.items():
        np.testing.assert_allclose(basis.mean(axis=0),0.,rtol=0,atol=1e-8)
        term=model.interactions[name] if name in model.interactions else model.mains[name]
        norm=np.sqrt(np.mean(np.sum(basis**2,axis=1)))
        np.testing.assert_allclose(norm,0. if term.constant else 1.,rtol=0,atol=1e-8)
        if name in model.interactions and not term.constant:
            local=np.column_stack([np.ones(len(rows)),groups[term.first],groups[term.second]])
            error=np.max(abs(local.T@basis))/len(rows)
            assert error<1e-7,(name,error)
    return dict(basis_columns=model.metadata['basis_columns'],terms=len(groups),
        collapsed_terciles=bool(model.training_score_edges[0]==model.training_score_edges[1]))


def verify(rest=lambda:None):
    protocol=json.loads((REPORT/'protocol.json').read_text());signature=base.digest(REPORT/'protocol.json')
    for path,expected in protocol['hashes'].items():
        assert base.digest(ROOT/path)==expected,f'Protected source changed: {path}'
        rest()
    features=protocol['features'];tree_inputs=json.loads(base.CONFIG.read_text())['screened']['inputs']
    flags=[f for f in tree_inputs if f.endswith('_available')]
    assert not set(features)&set(flags)
    meta=pd.read_parquet(base.SOURCE,columns=tree_inputs+['date','end','pair','signal_eligible','finite_label','target'],use_threads=False)
    signals=pd.read_parquet(OLD/'tree_cem/scored_allocations.parquet',use_threads=False)
    missing=[f for f in tree_inputs if f not in signals]
    if missing:signals=signals.merge(meta[['date','pair']+missing],on=['date','pair'],validate='one_to_one')
    assert len(signals)==608 and signals[flags].eq(1).all().all()
    allocations=pd.read_parquet(OUT/'allocations.parquet',use_threads=False)
    contribution_table=pd.read_parquet(OUT/'contributions.parquet',use_threads=False)
    assert len(allocations)==608*3
    assert not allocations.duplicated(['case','date','pair']).any()
    manifest=json.loads((ROOT/'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    fits=[];perturbations=[];refits=[];previous={}
    with zipfile.ZipFile(base.ARCHIVE) as archive:
        for month in pd.period_range('2018-01','2025-06',freq='M'):
            cutoff=month.start_time;stamp=str(cutoff.date())
            rows=training_window(meta,cutoff).reset_index(drop=True)
            assert rows[flags].eq(1).all().all()
            assert rows.date.lt(cutoff).all() and rows.end.lt(cutoff).all()
            group=signals.loc[signals.date.dt.to_period('M').eq(month)].copy()
            tree=joblib.load(io.BytesIO(base.archive_read(archive,base.MODEL_ROOT+stamp+'.joblib',manifest)))
            for case in ['tree_cem']+list(NEW):
                if case=='tree_cem':
                    model=tree
                    codes=lambda m,f:tree_codes(m,f,tree_inputs)
                    if len(group):np.testing.assert_array_equal(codes(model,group),group.learned_level)
                else:
                    folder=OUT/'models'/case
                    audit=json.loads((folder/f'{stamp}.json').read_text())
                    model=joblib.load(folder/f'{stamp}.joblib')
                    assert audit['protocol_sha256']==signature and audit['case']==case
                    assert audit['fit']==stamp and audit['n_train']==len(rows)
                    assert pd.Timestamp(audit['latest_train_end'])==rows.end.max()
                    check_cv(audit,meta,cutoff)
                    assert model.kind==NEW[case]
                    assert model.metadata['alpha_multiplier']==audit['alpha_multiplier']
                    np.testing.assert_allclose(model.ridge.alpha,len(rows)*audit['alpha_multiplier'],rtol=0,atol=1e-12)
                    details=model_checks(model,rows,features)
                    codes=lambda m,f:m.size_codes(f)
                    fits.append(dict(case=case,fit=stamp,n_train=len(rows),alpha=audit['alpha_multiplier'],cv_status=audit['cv_status'],**details))
                    if len(group):
                        keys=pd.MultiIndex.from_frame(group[['date','pair']])
                        saved=allocations.loc[allocations.case.eq(case)].set_index(['date','pair']).reindex(keys)
                        np.testing.assert_allclose(model.predict(group),saved.prediction,rtol=0,atol=1e-8)
                        np.testing.assert_array_equal(codes(model,group),saved.learned_level)
                        c=model.contributions(group)
                        saved_c=contribution_table.loc[contribution_table.case.eq(case)].set_index(['date','pair']).reindex(keys)
                        np.testing.assert_allclose(c.to_numpy(),saved_c[c.columns].to_numpy(),rtol=0,atol=1e-8)
                        np.testing.assert_allclose(saved_c[c.columns].sum(axis=1),saved.prediction,rtol=1e-10,atol=1e-8)
                if len(group):
                    original=np.asarray(codes(model,group),float)
                    for feature in features:
                        sd=float(rows[feature].std(ddof=0))
                        if not np.isfinite(sd) or sd<1e-12:continue
                        changed=0;amount=0.
                        for sign in [-1,1]:
                            altered=group.copy();altered[feature]+=sign*.05*sd
                            diff=abs(np.asarray(codes(model,altered),float)-original)/4
                            changed+=int((diff>0).sum());amount+=float(diff.sum())
                        perturbations.append(dict(case=case,fit=stamp,feature=feature,n=2*len(group),changed=changed,total_abs_fraction_change=amount))
                if case in previous:
                    old,oldrows=previous[case]
                    keys=pd.MultiIndex.from_frame(rows[['date','pair']]);oldkeys=pd.MultiIndex.from_frame(oldrows[['date','pair']])
                    common=rows.loc[keys.isin(oldkeys)]
                    if len(common):
                        diff=abs(np.asarray(codes(model,common))-np.asarray(codes(old,common)))/4
                        refits.append(dict(case=case,fit=stamp,n_common=len(common),changed=int((diff>0).sum()),total_abs_fraction_change=float(diff.sum())))
                previous[case]=(model,rows)
                rest()
    assert len(fits)==270
    for name,expected,keys in [('perturbation',perturbations,['case','fit','feature']),('refit_stability',refits,['case','fit'])]:
        actual=pd.read_csv(REPORT/f'{name}.csv').sort_values(keys).reset_index(drop=True)
        expected=pd.DataFrame(expected).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-10)
    p=pd.DataFrame(perturbations)
    denominators=p.pivot(index=['fit','feature'],columns='case',values='n')
    assert denominators.notna().all().all() and denominators.nunique(axis=1).eq(1).all()
    r=pd.DataFrame(refits).pivot(index='fit',columns='case',values='n_common')
    assert r.notna().all().all() and r.nunique(axis=1).eq(1).all()
    ledger=[];calendar=None
    for case in ['tree_cem','fixed100']+list(NEW):
        folder=OUT/'replays'/case
        frame=pd.read_parquet(folder/'scored_allocations.parquet',use_threads=False)
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        if calendar is None:calendar=nav.index
        else:assert nav.index.equals(calendar)
        assert len(frame)==len(decisions)==608
        expected=signals[['date','pair','direction','optionid_a','optionid_b']].sort_values(['date','pair']).reset_index(drop=True)
        actual=frame[expected.columns].sort_values(['date','pair']).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
        audit=verify_ledger(nav,trades,decisions,frame)
        if case in NEW:
            status=json.loads((folder/'status.json').read_text())
            assert status['status']=='complete' and status['protocol_sha256']==signature
            assert status['allocations_sha256']==base.digest(OUT/'allocations.parquet')
            desired=allocations.loc[allocations.case.eq(case)].set_index(['date','pair']).learned_level
            keys=pd.MultiIndex.from_frame(frame[['date','pair']])
            np.testing.assert_array_equal(frame.learned_level,desired.reindex(keys).to_numpy())
            actionkeys=pd.MultiIndex.from_frame(decisions[['signal_date','pair']])
            np.testing.assert_array_equal(decisions.level,desired.reindex(actionkeys).to_numpy())
        else:
            for name in ['daily_nav.parquet','trades.csv','decisions.csv','scored_allocations.parquet']:
                assert base.digest(folder/name)==base.digest(OLD/case/name)
        assert decisions.entry_date.gt(decisions.signal_date).all()
        ledger.append(dict(case=case,**audit))
        rest()
    pd.DataFrame(fits).to_csv(REPORT/'verification_models.csv',index=False)
    pd.DataFrame(ledger).to_csv(REPORT/'verification_ledgers.csv',index=False)
    result=dict(status='passed',monthly_models=270,proposals_per_new_case=608,
        outer_and_inner_window_chronology_verified=True,
        alpha_choice_verified_against_saved_fold_scores=True,
        cv_scores_refitted_for_verification=False,
        all_saved_predictions_contributions_and_sizes_reproduced=True,
        training_transforms_terciles_and_interaction_orthogonality_verified=True,
        perturbations_and_refit_changes_reproduced=True,stability_denominators_identical=True,
        original_tree_sizes_reproduced=True,training_and_signal_availability_flags_all_one=True,
        archived_controls_identical=True,all_funded_ledgers_reconciled=True,
        protected_hashes_unchanged=True,protocol_sha256=signature,
        max_independent_ledger_error=max(a['independent_ledger_error'] for a in ledger))
    (REPORT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    budget=base.Budget()
    print(json.dumps(verify(rest=budget.rest),indent=2),flush=True)
