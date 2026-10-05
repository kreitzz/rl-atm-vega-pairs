"""Verify unchanged best checkpoint and attribute transcript-trial changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_atm65_transcripts import OUT,REPORT,EMPLOYEE,EMPLOYEE_REPORT,MODELS,baseline_actions
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.atm65_transcript_features import ADDED


def main():
    manifest=json.loads((OUT/'inputs.json').read_text())
    for p,h in manifest['code_and_sources'].items():assert digest(p)==h,f'Source changed: {p}'
    checkpoint=json.loads(Path(manifest['baseline_checkpoint']).read_text())
    assert digest(checkpoint['archive'])==checkpoint['archive_sha256']
    coverage=pd.read_csv(REPORT/'coverage.csv').set_index('pair');rows=[];fits=[]
    for d in manifest['definitions']:
        c=coverage.loc[d['pair']]
        if not c.retrain:continue
        folder=OUT/d['pair'];panel=pd.read_parquet(folder/'states.parquet')
        original=pd.read_parquet(EMPLOYEE/d['pair']/'states.parquet')
        pd.testing.assert_frame_equal(panel[original.columns],original)
        table=pd.read_csv(folder/'fits.csv');check_fits(table)
        for m in MODELS:assert table[m+'_baseline_reproduced'].all()
        fits.append(table.assign(pair=d['pair']))
        for m in MODELS:
            old=read_actions(baseline_actions(d['pair'],m,bool(c.employee_features)))
            new=read_actions(folder/m/'actions.csv');assert old.index.equals(new.index)
            row=dict(pair=d['pair'],label=d['label'],model=m,changed_recommendations=int(old.ne(new).sum()),
                     old_nonzero=int(old.ne(0).sum()),new_nonzero=int(new.ne(0).sum()),
                     entries_with_transcripts=0,entries_without_transcripts=0)
            path=folder/m/'full/decisions.csv'
            if path.exists():
                decisions=pd.read_csv(path,parse_dates=['signal_date','entry_date'])
                entered=decisions[decisions.status.eq('entered')]
                assert (entered.entry_date>entered.signal_date).all()
                state=panel.reindex(pd.DatetimeIndex(entered.signal_date))
                valid=state[[x for x in ADDED if x.endswith('_available')]].gt(0).any(axis=1)
                row.update(entries_with_transcripts=int(valid.sum()),entries_without_transcripts=int((~valid).sum()))
            rows.append(row)
    changes=pd.DataFrame(rows);changes.to_csv(REPORT/'policy_changes.csv',index=False)
    all_fits=pd.concat(fits,ignore_index=True)
    accounting=json.loads((REPORT/'accounting_coverage.json').read_text())
    report=dict(source_files_verified=len(manifest['code_and_sources']),baseline_archive_sha256_verified=True,
        retrained_pairs=int(coverage.retrain.sum()),monthly_controls_verified=len(all_fits),
        original_state_columns_unchanged=True,training_outcomes_strictly_purged=True,accounting=accounting,
        entry_coverage=changes.groupby('model')[['entries_with_transcripts','entries_without_transcripts']].sum().to_dict('index'),
        enabled_rl_refits=int(all_fits.rl_transcripts_status.eq('enabled').sum()),
        enabled_rl_refits_using_transcript_values=int(all_fits.rl_transcripts_refit_transcript_value_splits.fillna(0).gt(0).sum()),
        enabled_rl_refits_using_missingness=int(all_fits.rl_transcripts_refit_transcript_missing_splits.fillna(0).gt(0).sum()))
    write_json(REPORT/'audit.json',report)
    pairs=pd.read_csv(REPORT/'pairs.csv');pairs=pairs[(pairs.universe=='all')&(pairs.period=='full')];deltas=[]
    for kind in ['rl','nn']:
        b=pairs[pairs.model.eq(kind+'_baseline')].set_index('pair');n=pairs[pairs.model.eq(kind+'_transcripts')].set_index('pair')
        own=pd.DataFrame({'label':n.label,'pnl_change':n.net_excess_pnl-b.net_excess_pnl,'entry_change':n.entries-b.entries}).reset_index()
        own['model']=kind;deltas.append(own)
    delta=pd.concat(deltas,ignore_index=True);delta.to_csv(REPORT/'pair_changes.csv',index=False)
    path=REPORT.with_suffix('.md');text=path.read_text().split('\n## Verification and trade coverage')[0]
    lines=['','## Verification and trade coverage','',
        f'{len(all_fits)} monthly control rows reproduced current-reference actions. All snapshotted sources and the best-checkpoint archive passed SHA-256 verification. Original state values and completed-outcome cutoffs are unchanged.','',
        '| Candidate on refitted sleeves | Entries with transcripts | Entries without transcripts |','|---|---:|---:|']
    for m,counts in report['entry_coverage'].items():lines.append(f'| {m} | {counts["entries_with_transcripts"]} | {counts["entries_without_transcripts"]} |')
    lines+=['',f'Of {report["enabled_rl_refits"]} enabled RL recent refits, {report["enabled_rl_refits_using_transcript_values"]} split on a transcript value and {report["enabled_rl_refits_using_missingness"]} split on transcript availability. These counts can overlap. Adding features can also change original-feature partitions and validation decisions.','',
        '[Policy changes](atm65_transcripts/policy_changes.csv) · [Pair P&L changes](atm65_transcripts/pair_changes.csv) · [Audit](atm65_transcripts/audit.json)','']
    path.write_text(text+'\n'.join(lines));print(json.dumps(report,indent=2),flush=True)
    print(delta[delta.model.eq('rl')].sort_values('pnl_change').iloc[np.r_[0:4,-4:0]].to_string(index=False),flush=True)


if __name__=='__main__':main()
