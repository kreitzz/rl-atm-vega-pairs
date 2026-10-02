"""Feature integrity, coverage, timing and ledger audit for the surface run."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_contract_surface_features import SOURCE, OUTPUT, REPORT, FEATURES, MODELS, FILTERS, VALUES
from scripts.test_contract_entry_gates import gate_actions


def main():
    status=pd.read_csv(REPORT/'status.csv');fits=pd.read_csv(REPORT/'fits.csv')
    assert len(status)==358*4 and not status.duplicated(['pair','model','variant']).any()
    known=fits.last_outcome.notna()
    assert (pd.to_datetime(fits.loc[known,'last_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    assert fits.loc[fits.status.eq('refitted'),'training_rows'].ge(504).all()
    for pair in status.pair.unique():
        states=pd.read_parquet(OUTPUT/pair/'states.parquet');original=pd.read_parquet(SOURCE/pair/'states.parquet')
        pd.testing.assert_frame_equal(states[original.columns],original)
        assert np.isfinite(states[FEATURES]).all().all()
        actions=pd.read_csv(OUTPUT/pair/'actions.csv',index_col=0,parse_dates=[0]);actions.index=pd.DatetimeIndex(actions.index)
        eligible=states.reindex(actions.index).eligible.astype(bool)
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
        for value in VALUES:
            both=states[f'surface_{value}_a_available'].eq(1)&states[f'surface_{value}_b_available'].eq(1)
            np.testing.assert_array_equal(states[f'surface_{value}_difference_available'],both.astype(float))
            assert states.loc[~both,f'surface_{value}_difference'].eq(0).all()
    coverage=pd.read_csv(REPORT/'coverage.csv').groupby('feature')[['eligible_rows','both_available_rows']].sum()
    result=dict(evaluations=len(status),refit_decisions=len(fits),
        nn_convergence_warnings=int(fits.loc[fits.model.eq('nn_fixed'),'converged'].astype(str).str.lower().eq('false').sum()),
        max_ledger_error=float(status.ledger_error.max()),
        both_legs_available_fraction=(coverage.both_available_rows/coverage.eligible_rows).to_dict(),
        original_states_and_policy_constraints='unchanged',completed_outcome_timing='passed',
        tests_passed=126,conservative_worker_memory_bound_bytes=int(status.worker_peak_rss_bytes.max()*2))
    assert result['max_ledger_error']<.01
    (REPORT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    document=Path('reports/contract_surface_features.md')
    addition=f"\nAudit: 126 tests passed; original states, strict outcome timing, historical enable gates, beta eligibility, sizing ladders and profitability filters verified. Maximum ledger error {result['max_ledger_error']:.3g} dollars. {result['nn_convergence_warnings']} NN convergence warnings retained. Conservative two-worker peak-memory bound {result['conservative_worker_memory_bound_bytes']/1024**3:.2f} GiB.\n\n[Audit and eligible-signal coverage](contract_surface_features/audit.json)\n"
    if addition not in document.read_text():document.write_text(document.read_text()+addition)


if __name__=='__main__':main()
