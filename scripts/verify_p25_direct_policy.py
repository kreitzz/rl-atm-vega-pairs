"""Read-only reconstruction of direct policies, shadow objectives and funded books."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
import io
import json
import zipfile
import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_direct_policy as run
from scripts.diagnose_p25_pca import training_window
from scripts.p25_direct_policy import make_policy,shadow_scorer,fit_direct,size_codes
from scripts.p25_representation import state_book
from scripts.run_p25_additive_sizing import perturb
from scripts.submission_market import verify_ledger


def verify(budget):
    meta,signals,features,inputs,cal,manifest=run.load_inputs()
    protocol=json.loads((run.REPORT/'protocol.json').read_text());signature=run.base.digest(run.REPORT/'protocol.json')
    for p,expected in protocol['hashes'].items():
        assert run.base.digest(run.ROOT/p)==expected,f'Source changed: {p}';budget.rest()
    allocations=pd.read_parquet(run.OUT/'allocations.parquet',use_threads=False)
    contributions=pd.read_parquet(run.OUT/'contributions.parquet',use_threads=False)
    assert len(allocations)==1216 and not allocations.duplicated(['case','date','pair']).any()
    fits=pd.read_csv(run.REPORT/'fits.csv')
    assert len(fits)==270 and fits.groupby('case').size().eq(90).all()
    audits=[];refits=[];perturbations=[];previous={};seed_count=0;reproduced=0
    with zipfile.ZipFile(run.base.ARCHIVE) as archive:
        for month in pd.period_range('2018-01','2025-06',freq='M'):
            cutoff=month.start_time;stamp=str(cutoff.date())
            rows=training_window(meta,cutoff).reset_index(drop=True)
            assert rows.date.lt(cutoff).all() and rows.end.lt(cutoff).all()
            assert rows.date.ge(cutoff-pd.DateOffset(months=18)).all()
            group=signals.loc[signals.date.dt.to_period('M').eq(month)].copy()
            reference=joblib.load(io.BytesIO(run.base.archive_read(archive,run.base.MODEL_ROOT+stamp+'.joblib',manifest)))
            tree_actions=dict(zip(reference['leaves'],reference['choices']))
            tree_codes=np.array([tree_actions[i] for i in reference['tree'].apply(rows[inputs])])
            if len(group):
                expected=np.array([tree_actions[i] for i in reference['tree'].apply(group[inputs])])
                np.testing.assert_array_equal(expected,group.learned_level)
            # Independent state aggregation checks the selected actions, not sparse scoring internals.
            def score_actions(actions):
                book=state_book(rows,actions-2,3,cal,run.base.LABELS)
                daily=book[:,2]+book[:,8]+book[:,14]
                variance=daily.var(ddof=1)
                return float(np.sqrt(252)*daily.mean()/np.sqrt(variance)) if variance>1e-18 else 0.
            tree_row=fits.loc[fits.case.eq('tree_cem')&fits.fit.eq(stamp)].iloc[0]
            np.testing.assert_allclose(score_actions(tree_codes),tree_row.training_shadow_daily_sharpe,atol=1e-10,rtol=1e-10)
            for case,kind in run.KINDS.items():
                folder=run.OUT/'models'/case
                model=joblib.load(folder/f'{stamp}.joblib');audit=json.loads((folder/f'{stamp}.json').read_text())
                assert audit['protocol_sha256']==signature and audit['evaluations']==9600 and audit['seed']==2026
                assert audit['training_rows']==len(rows) and pd.Timestamp(audit['latest_train_end'])==rows.end.max()
                assert model.features==features and model.kind==kind and np.max(abs(model.coefficients))<=3.
                recreated=make_policy(rows,features,kind)
                np.testing.assert_allclose(model.design(rows),recreated.design(rows),rtol=0,atol=1e-12)
                for feature in features:
                    np.testing.assert_array_equal(model.mains[feature].knots,recreated.mains[feature].knots)
                    basis=model.mains[feature].transform(rows)
                    np.testing.assert_allclose(basis.mean(axis=0),0.,atol=1e-10)
                    norm=np.sqrt(np.mean(np.sum(basis*basis,axis=1)))
                    np.testing.assert_allclose(norm,0. if model.mains[feature].constant else 1.,atol=1e-10)
                selected=model.size_codes(rows)
                independent=score_actions(selected)
                np.testing.assert_allclose(independent,audit['training_shadow_daily_sharpe'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(score_actions(np.full(len(rows),4)),audit['baseline_shadow_daily_sharpe'],atol=1e-10,rtol=1e-10)
                trace=audit['trace'];assert len(trace)==50
                assert all(x['iteration']==i+1 for i,x in enumerate(trace))
                assert np.all(np.diff([x['best_score'] for x in trace])>=0)
                assert audit['training_shadow_daily_sharpe']>=audit['baseline_shadow_daily_sharpe']-1e-12
                if stamp in ['2018-01-01','2025-06-01']:
                    again,detail=fit_direct(recreated,rows,shadow_scorer(rows,cal,run.base.LABELS),rest=budget.rest)
                    np.testing.assert_array_equal(again.coefficients,model.coefficients)
                    assert detail['trace']==trace;reproduced+=1
                if stamp in run.SEED_MONTHS:
                    for seed in [2027,2028]:
                        diag=json.loads((folder/f'{stamp}_seed{seed}.json').read_text())
                        assert diag['protocol_sha256']==signature and diag['seed']==seed and diag['evaluations']==9600
                        assert 0<=diag['assignment_disagreement']<=1;seed_count+=1
                if len(group):
                    keys=pd.MultiIndex.from_frame(group[['date','pair']])
                    own=allocations.loc[allocations.case.eq(case)].set_index(['date','pair']).reindex(keys)
                    np.testing.assert_allclose(model.predict(group),own.score,rtol=0,atol=1e-12)
                    np.testing.assert_array_equal(size_codes(own.score),own.learned_level)
                    c=contributions.loc[contributions.case.eq(case)].set_index(['date','pair']).reindex(keys)
                    actual=model.contributions(group)
                    np.testing.assert_allclose(c[actual.columns].to_numpy(),actual.to_numpy(),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(c[actual.columns].sum(axis=1),own.score,rtol=0,atol=1e-10)
                perturbations+=perturb(case,stamp,model,group,rows,features,lambda m,f:m.size_codes(f))
                if case in previous:
                    old,oldrows=previous[case]
                    keys=pd.MultiIndex.from_frame(rows[['date','pair']]);oldkeys=pd.MultiIndex.from_frame(oldrows[['date','pair']])
                    common=rows.loc[keys.isin(oldkeys)]
                    diff=abs(model.size_codes(common)-old.size_codes(common))/4
                    refits.append(dict(case=case,fit=stamp,n_common=len(common),changed=int((diff>0).sum()),total_abs_fraction_change=float(diff.sum())))
                previous[case]=(model,rows)
                audits.append(dict(case=case,fit=stamp,training_rows=len(rows),independent_shadow_sharpe=independent,
                    half=int((selected==2).sum()),three_quarter=int((selected==3).sum()),full=int((selected==4).sum())))
                budget.rest()
            if month.month in [1,7]:print(f'Verified monthly models through {stamp}',flush=True)
    for name,data,keys in [('perturbation',perturbations,['case','fit','feature']),('refit_stability',refits,['case','fit'])]:
        actual=pd.read_csv(run.REPORT/f'{name}.csv').sort_values(keys).reset_index(drop=True)
        expected=pd.DataFrame(data).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=0,atol=1e-10)
    ledgers=[];calendar=None
    for case in list(run.CONTROLS)+list(run.KINDS):
        folder=run.OUT/'replays'/case
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        frame=pd.read_parquet(folder/'scored_allocations.parquet',use_threads=False)
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        if calendar is None:calendar=nav.index
        else:assert nav.index.equals(calendar)
        assert len(frame)==len(decisions)==608 and decisions.entry_date.gt(decisions.signal_date).all()
        identity=['date','pair','direction','optionid_a','optionid_b']
        pd.testing.assert_frame_equal(frame[identity].sort_values(['date','pair']).reset_index(drop=True),
            signals[identity].sort_values(['date','pair']).reset_index(drop=True),check_dtype=False)
        result=verify_ledger(nav,trades,decisions,frame);ledgers.append(dict(case=case,**result))
        if case in run.CONTROLS:
            for name in run.FILES:assert run.base.digest(folder/name)==run.base.digest(run.CONTROLS[case]/name)
        else:
            status=json.loads((folder/'status.json').read_text());assert status['status']=='complete'
            assert status['protocol_sha256']==signature and status['allocations_sha256']==run.base.digest(run.OUT/'allocations.parquet')
            desired=allocations.loc[allocations.case.eq(case)].set_index(['date','pair']).learned_level
            np.testing.assert_array_equal(frame.learned_level,desired.reindex(pd.MultiIndex.from_frame(frame[['date','pair']])).to_numpy())
            np.testing.assert_array_equal(decisions.level,desired.reindex(pd.MultiIndex.from_frame(decisions[['signal_date','pair']])).to_numpy())
        budget.rest()
    assert len(audits)==180 and seed_count==24 and reproduced==4
    pd.DataFrame(audits).to_csv(run.REPORT/'verification_models.csv',index=False)
    pd.DataFrame(ledgers).to_csv(run.REPORT/'verification_ledgers.csv',index=False)
    result=dict(status='passed',monthly_direct_models=180,original_tree_models=90,additional_seed_records=24,
        full_searches_independently_repeated=4,all_training_objectives_independently_reaggregated=True,
        training_only_transforms_and_completed_outcomes_verified=True,all_saved_predictions_and_contributions_reproduced=True,
        all_stability_metrics_reproduced=True,archived_controls_identical=True,all_six_funded_ledgers_reconciled=True,
        max_independent_ledger_error=max(x['independent_ledger_error'] for x in ledgers),
        peak_rss_mib=budget.peak,protected_hashes_unchanged=True,protocol_sha256=signature)
    run.base.write(run.REPORT/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':verify(run.Budget())
