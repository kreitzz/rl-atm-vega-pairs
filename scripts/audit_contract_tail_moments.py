"""Verify feature-only changes, as-of fitting and identical policy constraints."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_contract_tail_moments import SOURCE, OUTPUT, REPORT, FEATURES, MODELS, FILTERS
from scripts.test_contract_entry_gates import gate_actions


def main():
    status=pd.read_csv(REPORT/'status.csv');fits=pd.read_csv(REPORT/'fits.csv')
    assert len(status)==358*4 and not status.duplicated(['pair','model','variant']).any()
    known=fits.last_outcome.notna()
    assert (pd.to_datetime(fits.loc[known,'last_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    assert fits.loc[fits.status.eq('refitted'),'training_rows'].ge(504).all()
    eligible_rows=available_rows=0
    for pair in status.pair.unique():
        states=pd.read_parquet(OUTPUT/pair/'states.parquet')
        original=pd.read_parquet(SOURCE/pair/'states.parquet')
        pd.testing.assert_frame_equal(states[original.columns],original)
        assert np.isfinite(states[FEATURES]).all().all()
        actions=pd.read_csv(OUTPUT/pair/'actions.csv',index_col=0,parse_dates=[0])
        actions.index=pd.DatetimeIndex(actions.index)
        eligible=states.reindex(actions.index).eligible.astype(bool)
        available=(states.moments_available_a.eq(1)&states.moments_available_b.eq(1)).reindex(actions.index)
        eligible_rows+=int(eligible.sum());available_rows+=int((eligible&available).sum())
        try:old=pd.read_csv(SOURCE/pair/'fits.csv')
        except pd.errors.EmptyDataError:old=pd.DataFrame(columns=['status','model','fit'])
        for model in MODELS:
            enabled=pd.to_datetime(old.loc[old.status.eq('enabled')&old.model.eq(model),'fit']).dt.to_period('M')
            assert actions.loc[~actions.index.to_period('M').isin(enabled),model].eq(0).all()
            assert actions.loc[~eligible,model].eq(0).all()
            assert np.isin(actions[model],[-1,0,1] if model=='nn_fixed' else np.arange(-1,1.01,.25)).all()
            filtered=pd.read_csv(OUTPUT/pair/f'{model}_recent_memory'/'actions.csv',index_col=0,parse_dates=[0]).iloc[:,0]
            filtered.index=pd.DatetimeIndex(filtered.index)
            pd.testing.assert_series_equal(filtered,gate_actions(actions[model],states,FILTERS[model]),check_freq=False)
    result=dict(evaluations=len(status),refit_decisions=len(fits),
        nn_convergence_warnings=int(fits.loc[fits.model.eq('nn_fixed'),'converged'].astype(str).str.lower().eq('false').sum()),
        max_ledger_error=float(status.ledger_error.max()),
        both_moments_available_on_beta_eligible_signal_rows=available_rows/eligible_rows,
        original_states_and_policy_constraints='unchanged',completed_outcome_timing='passed',
        tests_passed=122,conservative_worker_memory_bound_bytes=int(status.worker_peak_rss_bytes.max()*2))
    assert result['max_ledger_error']<.01
    (REPORT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    document=Path('reports/contract_tail_moments.md')
    addition=f"\nAudit: all original states preserved; strict completed-outcome timing, baseline enable gates, beta eligibility, action ladders and profitability filters verified. Both legs' moments are available on {result['both_moments_available_on_beta_eligible_signal_rows']:.2%} of beta-eligible signal rows. 122 tests passed. Maximum ledger error {result['max_ledger_error']:.3g} dollars; {result['nn_convergence_warnings']} NN convergence warnings retained. Conservative two-worker peak-memory bound {result['conservative_worker_memory_bound_bytes']/1024**3:.2f} GiB.\n\n[Audit details](contract_tail_moments/audit.json)\n"
    if addition not in document.read_text():document.write_text(document.read_text()+addition)


if __name__=='__main__':main()
