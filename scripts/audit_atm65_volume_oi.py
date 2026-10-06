"""Check the frozen reference, training cutoffs, states, and signal use."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_atm65_volume_oi import OUT,REPORT,EMPLOYEE,MODELS,baseline_actions,CHECKPOINT
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.atm65_volume_oi_features import ADDED


def quarter_differences():
    rows=[]
    for period in ['full','recent']:
        base=pd.read_csv(REPORT/f'all_rl_baseline_{period}_portfolio.csv',index_col=0,parse_dates=[0])
        candidate=pd.read_csv(REPORT/f'all_rl_volume_oi_{period}_portfolio.csv',index_col=0,parse_dates=[0])
        assert base.index.equals(candidate.index)
        h=pd.DataFrame(dict(a=base.excess_return.iloc[1:],b=candidate.excess_return.iloc[1:]))
        h['n']=1.;h['aa']=h.a*h.a;h['bb']=h.b*h.b
        blocks=h.groupby(h.index.to_period('Q'))[['n','a','aa','b','bb']].sum().to_numpy()
        rng=np.random.default_rng(2026);draws=rng.integers(len(blocks),size=(2000,len(blocks)))
        n,a,aa,b,bb=blocks[draws].sum(axis=1).T
        def ratio(s,ss):
            sd=np.sqrt(np.maximum((ss-s*s/n)/(n-1),0))
            return np.sqrt(252)*np.divide(s/n,sd,out=np.full(len(n),np.nan),where=sd>1e-12)
        difference=ratio(b,bb)-ratio(a,aa)
        rows.append(dict(period=period,resamples=2000,
            sharpe_difference=float(np.sqrt(252)*(h.b.mean()/h.b.std()-h.a.mean()/h.a.std())),
            difference_ci90_low=float(np.nanquantile(difference,.05)),
            difference_ci90_high=float(np.nanquantile(difference,.95))))
    pd.DataFrame(rows).to_csv(REPORT/'paired_quarter_differences.csv',index=False)
    return rows


def main():
    manifest=json.loads((OUT/'inputs.json').read_text())
    for path,sha in manifest['code_and_sources'].items():assert digest(path)==sha,f'Source changed: {path}'
    reference=json.loads(CHECKPOINT.read_text())
    assert digest(reference['archive'])==reference['archive_sha256']
    for path,sha in reference['files'].items():assert digest(path)==sha,f'Frozen source changed: {path}'
    coverage=pd.read_csv(REPORT/'coverage.csv').set_index('pair');rows=[];fits=[]
    for definition in manifest['definitions']:
        pair=definition['pair'];c=coverage.loc[pair]
        if not c.retrain:continue
        folder=OUT/pair;panel=pd.read_parquet(folder/'states.parquet')
        original=pd.read_parquet(EMPLOYEE/pair/'states.parquet')
        pd.testing.assert_frame_equal(panel[original.columns],original)
        assert np.isfinite(panel[ADDED]).all().all()
        assert panel[ADDED[1]].isin([0.,1.]).all()
        assert panel.loc[panel[ADDED[1]].eq(0),ADDED[0]].eq(0).all()
        table=pd.read_csv(folder/'fits.csv');check_fits(table)
        for model in MODELS:assert table[model+'_baseline_reproduced'].all()
        fits.append(table.assign(pair=pair))
        for model in MODELS:
            old=read_actions(baseline_actions(pair,model,bool(c.employee_features)))
            new=read_actions(folder/model/'actions.csv');assert old.index.equals(new.index)
            row=dict(pair=pair,label=definition['label'],model=model,changed_recommendations=int(old.ne(new).sum()),
                old_nonzero=int(old.ne(0).sum()),new_nonzero=int(new.ne(0).sum()),
                entries_with_signal=0,entries_without_signal=0)
            path=folder/model/'full/decisions.csv'
            if path.exists():
                decisions=pd.read_csv(path,parse_dates=['signal_date','entry_date'])
                entered=decisions[decisions.status.eq('entered')]
                assert (entered.entry_date>entered.signal_date).all()
                valid=panel.reindex(pd.DatetimeIndex(entered.signal_date))[ADDED[1]].eq(1.)
                row.update(entries_with_signal=int(valid.sum()),entries_without_signal=int((~valid).sum()))
            rows.append(row)
    changes=pd.DataFrame(rows);changes.to_csv(REPORT/'policy_changes.csv',index=False)
    fit_table=pd.concat(fits,ignore_index=True)
    accounting=json.loads((REPORT/'accounting_coverage.json').read_text())
    assert accounting['common_complete_pairs']==345 and not accounting['unresolved'],accounting
    report=dict(source_files_verified=len(manifest['code_and_sources']),
        frozen_source_files_verified=len(reference['files']),baseline_archive_sha256_verified=True,
        retrained_pairs=int(coverage.retrain.sum()),monthly_controls_verified=len(fit_table),
        original_state_columns_unchanged=True,training_outcomes_strictly_purged=True,accounting=accounting,
        entry_coverage=changes.groupby('model')[['entries_with_signal','entries_without_signal']].sum().to_dict('index'),
        enabled_rl_refits=int(fit_table.rl_volume_oi_status.eq('enabled').sum()),
        enabled_refits_using_value=int(fit_table.rl_volume_oi_refit_volume_oi_value_splits.fillna(0).gt(0).sum()),
        enabled_refits_using_availability=int(fit_table.rl_volume_oi_refit_volume_oi_missing_splits.fillna(0).gt(0).sum()),
        paired_quarter_differences=quarter_differences(),
        uncertainty_note='Paired calendar-quarter resampling of the inspected realized portfolio paths; descriptive, no policy refitting or model-selection correction.')
    write_json(REPORT/'audit.json',report)
    pairs=pd.read_csv(REPORT/'pairs.csv');pairs=pairs[(pairs.universe=='all')&(pairs.period=='full')]
    base=pairs[pairs.model.eq('rl_baseline')].set_index('pair')
    candidate=pairs[pairs.model.eq('rl_volume_oi')].set_index('pair')
    delta=pd.DataFrame(dict(label=candidate.label,pnl_change=candidate.net_excess_pnl-base.net_excess_pnl,
        entry_change=candidate.entries-base.entries)).reset_index()
    delta.to_csv(REPORT/'pair_changes.csv',index=False)
    path=REPORT/'summary.txt';path.write_text(path.read_text()+'\nVerification:\n'+json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print(delta.sort_values('pnl_change').iloc[np.r_[0:5,-5:0]].to_string(index=False),flush=True)


if __name__=='__main__':main()
