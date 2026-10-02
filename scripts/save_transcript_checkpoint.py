"""Promote the transcript trial only if full-period RL Sharpe improves."""
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import pandas as pd
from scripts.run_atm65_transcripts import OUT,REPORT,CHECKPOINT,EMPLOYEE,employee_roster,baseline_actions,baseline_results
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json

NAME='atm30_fee065_employee_transcripts_2026-10-01'
CONFIG=Path('config')/(NAME+'_checkpoint.json')
ARCHIVE=Path('artifacts/contracts/checkpoints')/(NAME+'.tar')


def main():
    comparison=pd.read_csv(REPORT/'comparison.csv')
    full=comparison[(comparison.universe=='all')&(comparison.period=='full')].set_index('model')
    accounting=json.loads((REPORT/'accounting_coverage.json').read_text())
    if accounting['common_complete_pairs']!=345 or accounting['unresolved']:
        raise ValueError('Promotion requires all 345 matched sleeves to complete')
    if full.loc['rl_transcripts','annualized_excess_sharpe']<=full.loc['rl_baseline','annualized_excess_sharpe']:
        print('Transcript trial does not improve full-period Sharpe. Employee-sentiment reference retained.',flush=True);return
    if CONFIG.exists():
        manifest=json.loads(CONFIG.read_text());assert digest(ARCHIVE)==manifest['archive_sha256']
        write_json(Path('config/current_contract_reference.json'),{k:v for k,v in manifest.items() if k!='files'});return
    assert json.loads((REPORT/'audit.json').read_text())['baseline_archive_sha256_verified']
    trial=json.loads((OUT/'inputs.json').read_text());prior=json.loads(CHECKPOINT.read_text())
    for p,h in trial['code_and_sources'].items():assert digest(p)==h,p
    policy_manifest=json.loads(Path(prior['policy_sources']).read_text())
    policies=copy.deepcopy(policy_manifest['policies']);coverage=pd.read_csv(REPORT/'coverage.csv').set_index('pair')
    previous=OUT/'previous_reference.json';previous.write_text(Path('config/current_contract_reference.json').read_text())
    files={Path(p) for p in prior['files']};files.update(Path(p) for p in trial['code_and_sources'])
    files.update(p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file())
    files.update([REPORT.with_suffix('.md'),Path(__file__),CHECKPOINT])
    for policy in policies:
        changed=bool(coverage.loc[policy['pair'],'retrain']);policy['transcript_features']=changed
        if changed:
            for kind,model in [('rl_sized','rl_transcripts'),('nn_fixed','nn_transcripts')]:
                policy['rl_actions' if kind=='rl_sized' else 'nn_actions']=str(OUT/policy['pair']/model/'actions.csv')
                for period in ['full','recent']:policy['ledgers'][kind+'_'+period]=str(OUT/policy['pair']/model/period)
    write_json(OUT/'checkpoint_policies.json',dict(checkpoint=NAME,policies=policies));files.add(OUT/'checkpoint_policies.json')
    files=sorted({p.relative_to(Path.cwd()) if p.is_absolute() else p for p in files})
    hashes={str(p):digest(p) for p in files};temporary=ARCHIVE.with_suffix('.tmp')
    with tarfile.open(temporary,'w') as tar:
        for p in files:tar.add(p,arcname=str(p),recursive=False)
    with tarfile.open(temporary,'r|') as tar:
        count=0
        for member in tar:
            stream=tar.extractfile(member);h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==hashes[member.name],member.name
            count+=1
    assert count==len(files);temporary.replace(ARCHIVE)
    results=comparison[(comparison.universe=='all')&comparison.model.isin(['rl_transcripts','nn_transcripts'])]
    manifest=dict(checkpoint=NAME,saved_date='2026-10-01',model='rl_sized_employee_transcripts',benchmark='nn_fixed_employee_transcripts',
        training_hurdle_per_contract=.65,execution_commission_per_contract=.03,archive=str(ARCHIVE),archive_sha256=digest(ARCHIVE),
        archive_bytes=ARCHIVE.stat().st_size,report=str(REPORT.with_suffix('.md')),manifest=str(CONFIG),common_complete_pairs=345,
        predecessor=str(CHECKPOINT),policy_sources=str(OUT/'checkpoint_policies.json'),matched_results=results.to_dict('records'),
        execution_settings=trial['actual'],selection='Current best full-period annualized excess Sharpe; transcript family added to the employee-sentiment reference.',
        preservation=prior['preservation'],verification=f'{count} archive members verified against SHA-256 of source files.',
        data_timing_note='Backward availability lags enforced; archived historical vendor versions and company mappings are not fully verified.',files=hashes)
    write_json(CONFIG,manifest);write_json(Path('config/current_contract_reference.json'),{k:v for k,v in manifest.items() if k!='files'})
    print(results[['model','period','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)
    print(f'Saved {NAME}: {count} verified files, {ARCHIVE.stat().st_size/2**20:.1f} MiB',flush=True)


if __name__=='__main__':main()
