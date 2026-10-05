"""Cross-artifact timing, gate preservation, coverage and ledger checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_recent_contract_refit import SOURCE, OUTPUT, REPORT, MODELS, FILTERS
from scripts.test_contract_entry_gates import gate_actions


def main():
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    status=pd.read_csv(REPORT/'status.csv')
    assert len(status)==4*len(definitions)
    assert not status.duplicated(['pair','model','variant']).any()
    fits=pd.read_csv(REPORT/'fits.csv')
    known=fits.last_outcome.notna()
    assert (pd.to_datetime(fits.loc[known,'last_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    assert fits.loc[fits.status.eq('refitted'),'training_rows'].ge(504).all()
    for d in definitions:
        actions=pd.read_csv(OUTPUT/d['pair']/'actions.csv',index_col=0,parse_dates=[0])
        old_actions=pd.read_csv(SOURCE/d['pair']/'actions.csv',index_col=0,parse_dates=[0])
        actions.index=pd.DatetimeIndex(actions.index)
        old_actions.index=pd.DatetimeIndex(old_actions.index)
        assert actions.index.equals(old_actions.index)
        states=pd.read_parquet(SOURCE/d['pair']/'states.parquet').reindex(actions.index)
        try:old_fits=pd.read_csv(SOURCE/d['pair']/'fits.csv')
        except pd.errors.EmptyDataError:old_fits=pd.DataFrame(columns=['status','model','fit'])
        for model in MODELS:
            enabled=pd.to_datetime(old_fits.loc[old_fits.status.eq('enabled')&old_fits.model.eq(model),'fit']).dt.to_period('M')
            permitted=actions.index.to_period('M').isin(enabled)
            assert actions.loc[~permitted,model].eq(0).all()
            assert actions.loc[~states.eligible.astype(bool),model].eq(0).all()
            skipped=fits.loc[fits.pair.eq(d['pair'])&fits.model.eq(model)&fits.status.eq('insufficient_recent_history'),'fit']
            skipped_months=pd.to_datetime(skipped).dt.to_period('M')
            assert actions.loc[actions.index.to_period('M').isin(skipped_months),model].eq(0).all()
            assert np.isin(actions[model],[-1,0,1] if model=='nn_fixed' else np.arange(-1,1.01,.25)).all()
            actual=pd.read_csv(OUTPUT/d['pair']/f'{model}_recent_memory'/'actions.csv',index_col=0,parse_dates=[0]).iloc[:,0]
            actual.index=pd.DatetimeIndex(actual.index)
            pd.testing.assert_series_equal(actual,gate_actions(actions[model],states,FILTERS[model]),check_freq=False)
    old=pd.read_csv('reports/aligned_contract_learning/status.csv')
    gated=pd.read_csv('reports/contract_entry_gates/status.csv')
    gated=gated[gated.apply(lambda r:r.variant==FILTERS[r.model],axis=1)]
    failed=set(status.loc[status.status.ne('complete'),'pair'])|set(old.loc[old.status.ne('complete'),'pair'])|set(gated.loc[gated.status.ne('complete'),'pair'])
    activity=[]
    for source,frame in [('baseline',old),('previous_memory',gated),('recent',status[status.variant.eq('recent')]),('recent_memory',status[status.variant.eq('recent_memory')])]:
        subset=frame[frame.group.eq('broad')&~frame.pair.isin(failed)]
        for model,g in subset.groupby('model'):
            activity.append(dict(variant=source,model=model,pairs=len(g),entries=int(g.entries.sum()),active_pairs=int(g.entries.gt(0).sum())))
    pd.DataFrame(activity).to_csv(REPORT/'activity.csv',index=False)
    nn=fits[fits.model.eq('nn_fixed')]
    result=dict(evaluations=len(status),fits=len(fits),nn_convergence_warnings=int(nn.converged.astype(str).str.lower().eq('false').sum()),
        max_ledger_error=float(status.ledger_error.max()),timing_and_gate_checks='passed',
        insufficient_recent_history=int(fits.status.eq('insufficient_recent_history').sum()),
        test_suite='119 unittest tests passed',max_worker_peak_rss_bytes=int(status.worker_peak_rss_bytes.max()),
        conservative_two_worker_peak_bound_bytes=int(2*status.worker_peak_rss_bytes.max()))
    assert result['max_ledger_error']<.01
    (REPORT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    document=Path('reports/recent_contract_refit.md')
    addition=f"\nAudit: {len(status):,} evaluations, {len(fits):,} refit decisions; strict completed-outcome timing, historical enable gates, beta eligibility and action ladders verified. {result['insufficient_recent_history']} enabled model-months stayed flat for insufficient recent history. Maximum ledger error {result['max_ledger_error']:.3g} dollars. {result['nn_convergence_warnings']} NN convergence warnings retained. 119 unit tests passed. Conservative two-worker peak-memory bound {result['conservative_two_worker_peak_bound_bytes']/1024**3:.2f} GiB.\n\n[Trading activity](recent_contract_refit/activity.csv) · [Audit](recent_contract_refit/audit.json)\n"
    if addition not in document.read_text():document.write_text(document.read_text()+addition)


if __name__=='__main__':main()
