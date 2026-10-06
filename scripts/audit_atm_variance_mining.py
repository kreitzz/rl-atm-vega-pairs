"""Timing audit and incremental signal diagnostics against the current states."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.mine_atm_variance_signals import OUT,REPORT,SIGNALS,EMPLOYEE
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import features,write_json


def partial_rank(g,signal,controls):
    """Descriptive conditional correlation; never used to choose live actions."""
    g=g.dropna(subset=[signal,'directional_edge']).copy()
    g=g[g.groupby('pair').pair.transform('size')>=20]
    if len(g)<100:return np.nan
    cols=[signal,'directional_edge',*controls]
    ranks=g.groupby('pair')[cols].rank(pct=True)
    ranks-=ranks.groupby(g.pair).transform('mean')
    x=ranks[controls].fillna(0.).to_numpy();y=ranks[[signal,'directional_edge']].to_numpy()
    weight=1/g.groupby('pair').pair.transform('size').to_numpy();root=np.sqrt(weight)
    beta=np.linalg.lstsq(x*root[:,None],y*root[:,None],rcond=1e-8)[0]
    residual=y-x@beta;a,b=residual.T
    den=np.sqrt(np.sum(weight*a*a)*np.sum(weight*b*b))
    return float(np.sum(weight*a*b)/den) if den>1e-14 else np.nan


def main():
    manifest=json.loads((OUT/'inputs.json').read_text())
    assert digest('scripts/mine_atm_variance_signals.py')==manifest['code_sha256']
    assert digest(EMPLOYEE/'inputs.json')==manifest['reference_inputs_sha256']
    for path,sha in manifest['source_states'].items():assert digest(path)==sha,path
    frozen=json.loads(Path('config/atm30_fee065_employee_2026-10-01_checkpoint.json').read_text())
    assert digest(frozen['archive'])==frozen['archive_sha256']
    for path,sha in frozen['files'].items():assert digest(path)==sha,path
    fits=pd.read_csv(REPORT/'fit_audit.csv')
    assert (pd.to_datetime(fits.last_training_end)<pd.to_datetime(fits.cutoff)).all()
    assert (pd.to_datetime(fits.cutoff)<=pd.to_datetime(fits.first_test_date)).all()
    legs=pd.read_parquet(OUT/'legs.parquet')
    assert not legs.duplicated(['date','secid']).any()
    assert legs.iv.gt(0).all() and legs.optionid.notna().all()
    unresolved=legs.long_net.isna()|legs.short_net.isna()
    forecasts=pd.read_parquet(OUT/'forecasts.parquet')
    assert not forecasts.duplicated(['date','secid']).any()
    ready=forecasts.expanded_forecast.notna()
    assert forecasts.loc[ready,'expanded_forecast'].gt(0).all()
    # Missing future outcomes MUST be retained in the signal/peer universe.
    assert (ready&(forecasts.long_net.isna()|forecasts.short_net.isna())).any()
    data=pd.read_parquet(OUT/'pair_signals.parquet')
    assert not data.duplicated(['date','pair']).any()
    assert (data.end>data.date).all()
    controls=features(data);assert len(controls)==47
    rows=[];concentration=[]
    screen=pd.read_csv(REPORT/'signals.csv')
    for signal in SIGNALS:
        early=data[(data.date<'2022-01-01')&(data.end<'2022-01-01')]
        later=data[data.date>='2022-01-01']
        for period,g in [('early_completed',early),('later',later)]:
            rows.append(dict(signal=signal,period=period,partial_within_pair_rank_corr=partial_rank(g,signal,controls),
                controls=len(controls),observations=int(g[signal].notna().sum())))
        threshold=screen[(screen.signal==signal)&screen.period.eq('later')&screen['sample'].eq('all')].early_fixed_extreme_threshold.iloc[0]
        h=later.dropna(subset=[signal]).copy();h=h[h[signal].abs()>=threshold]
        h['selected_net']=np.where(h[signal]>0,h.long_net,h.short_net)
        if len(h):
            summed=h.groupby('pair').selected_net.agg(['sum','size'])
            total=summed['sum'].sum();count=summed['size'].sum()
            loo=(total-summed['sum'])/(count-summed['size']).replace(0,np.nan)
            winning=summed['sum'].clip(lower=0).sort_values(ascending=False)
            worst=summed['sum'].idxmin();best=summed['sum'].idxmax()
            labels=h.groupby('pair').label.first()
            concentration.append(dict(signal=signal,extreme_trades=len(h),active_pairs=len(summed),
                mean_net=float(h.selected_net.mean()),median_net=float(h.selected_net.median()),
                top_positive_pair_share=float(winning.iloc[0]/winning.sum()) if winning.sum()>0 else None,
                top_three_positive_pair_share=float(winning.iloc[:3].sum()/winning.sum()) if winning.sum()>0 else None,
                worst_leave_one_pair_out_mean=float(loo.min()),best_pair=labels[best],worst_pair=labels[worst]))
    partial=pd.DataFrame(rows);partial.to_csv(REPORT/'incremental_rank.csv',index=False)
    concentration=pd.DataFrame(concentration);concentration.to_csv(REPORT/'concentration.csv',index=False)
    stability=pd.read_csv(REPORT/'stability.csv').merge(partial[partial.period.eq('later')][['signal','partial_within_pair_rank_corr']],on='signal')
    stability=stability.merge(concentration[['signal','active_pairs','top_positive_pair_share','worst_leave_one_pair_out_mean']],on='signal',how='left')
    stability.to_csv(REPORT/'shortlist_diagnostics.csv',index=False)
    coverage=pd.read_csv(REPORT/'coverage.csv');assert len(coverage)==345
    report=dict(source_states_verified=len(manifest['source_states']),frozen_files_verified=len(frozen['files']),
        frozen_archive_verified=True,purged_quarterly_fits=len(fits),unique_signal_ready_legs=len(legs),
        signal_ready_legs_with_unavailable_payoff=int(unresolved.sum()),
        unavailable_payoffs_retained_in_forecast_factor_universe=True,
        resolved_pair_cohorts=len(data),pairs_with_evaluable_cohorts=int(data.pair.nunique()),
        requested_sleeves=345,sleeves_with_state_inputs=int(coverage.state_rows.gt(0).sum()),
        unresolved_pair_outcomes=int(coverage.unresolved_rows.sum()),
        current_cost_commission=.03,rl_refits=0,nn_refits=0,
        source_and_output_sha256={str(p):digest(p) for p in [OUT/'legs.parquet',OUT/'outcomes.parquet',OUT/'forecasts.parquet',OUT/'pair_signals.parquet',Path(__file__)]},
        diagnostic_note='Partial rank correlations regress within-pair ranks on the 47 existing common numeric states, in the evaluation sample. This is conditional association, not an executable residual forecast or a policy backtest.')
    write_json(REPORT/'audit.json',report)
    notes=REPORT/'summary.txt'
    notes.write_text(notes.read_text()+'\nIncremental evidence and concentration:\n'+stability.to_string(index=False)+'\n\nVerification:\n'+json.dumps(report,indent=2)+'\n')
    print(stability.to_string(index=False),flush=True);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
