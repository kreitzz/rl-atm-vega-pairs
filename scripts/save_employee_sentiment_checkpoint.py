"""Freeze the new best reference without replacing its predecessor archive."""
import hashlib
import json
from pathlib import Path
import tarfile
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import NN, CURVE, FROZEN, write_json
from scripts.run_atm65_employee_sentiment import OUT, REPORT
import pandas as pd

NAME = 'atm30_fee065_employee_2026-10-01'
CONFIG = Path('config')/(NAME+'_checkpoint.json')
ARCHIVE = Path('artifacts/contracts/checkpoints')/(NAME+'.tar')


def main():
    if CONFIG.exists():
        manifest = json.loads(CONFIG.read_text())
        assert digest(ARCHIVE) == manifest['archive_sha256']
        write_json(Path('config/current_contract_reference.json'), {k:v for k,v in manifest.items() if k != 'files'})
        print(f'{NAME}: verified and selected', flush=True); return
    inputs = json.loads((OUT/'inputs.json').read_text())
    for p,h in inputs['code_and_sources'].items(): assert digest(p) == h, p
    audit = json.loads((REPORT/'audit.json').read_text())
    assert audit['accounting']['common_complete_pairs'] == 345 and not audit['accounting']['unresolved']
    previous = OUT/'previous_reference.json'
    if not previous.exists(): previous.write_text(Path('config/current_contract_reference.json').read_text())
    coverage = pd.read_csv(REPORT/'coverage.csv').set_index('pair')
    files = {Path(p) for p in inputs['code_and_sources']}
    files.update(p for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file())
    files.add(REPORT.with_suffix('.md')); files.add(Path(__file__))
    policies = []
    for d in inputs['definitions']:
        changed = bool(coverage.loc[d['pair'],'retrain'])
        rl = OUT/d['pair']/'rl_sentiment' if changed else CURVE/d['pair']/'atm_control'
        nn = OUT/d['pair']/'nn_sentiment' if changed else NN/d['pair']/'fee_065'
        policy = dict(pair=d['pair'], label=d['label'], employee_features=changed,
                      rl_actions=str(rl/'actions.csv') if (rl/'actions.csv').exists() else None,
                      nn_actions=str(nn/'actions.csv') if (nn/'actions.csv').exists() else None, ledgers={})
        for model, base in [('rl_sized',rl),('nn_fixed',nn)]:
            if (base/'actions.csv').exists(): files.add(base/'actions.csv')
            for period in ['full','recent']:
                folder = base/period if changed or model == 'nn_fixed' else FROZEN/d['pair']/(period+'_003')
                status = json.loads((folder/'status.json').read_text()); assert status['status'] == 'complete'
                if not (base/'actions.csv').exists(): assert status['entries'] == 0 and status.get('cash_only')
                policy['ledgers'][model+'_'+period] = str(folder)
                files.update(p for p in folder.rglob('*') if p.is_file())
                if status.get('nav_source'): files.add(Path(status['nav_source']))
        policies.append(policy)
    write_json(OUT/'checkpoint_policies.json', dict(checkpoint=NAME, policies=policies))
    files.add(OUT/'checkpoint_policies.json')
    files = sorted({p.relative_to(Path.cwd()) if p.is_absolute() else p for p in files})
    hashes = {str(p):digest(p) for p in files}
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    temporary = ARCHIVE.with_suffix('.tmp')
    with tarfile.open(temporary,'w') as tar:
        for p in files: tar.add(p,arcname=str(p),recursive=False)
    with tarfile.open(temporary,'r|') as tar:
        count = 0
        for member in tar:
            stream = tar.extractfile(member); h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
            assert h.hexdigest() == hashes[member.name], member.name
            count += 1
    assert count == len(hashes); temporary.replace(ARCHIVE)
    comparison = pd.read_csv(REPORT/'comparison.csv')
    results = comparison[(comparison.universe == 'all') & comparison.model.isin(['rl_sentiment','nn_sentiment'])]
    manifest = dict(checkpoint=NAME,saved_date='2026-10-01',model='rl_sized_employee_sentiment',benchmark='nn_fixed_employee_sentiment',
        training_hurdle_per_contract=.65,execution_commission_per_contract=.03,
        archive=str(ARCHIVE),archive_sha256=digest(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,
        report=str(REPORT.with_suffix('.md')),manifest=str(CONFIG),common_complete_pairs=345,
        predecessor='config/atm30_fee065_checkpoint.json',policy_sources=str(OUT/'checkpoint_policies.json'),
        matched_results=results.to_dict('records'),execution_settings=inputs['actual'],
        selection='Current best full-period annualized excess Sharpe; employee leadership gap and availability added on 28 covered sleeves.',
        preservation='Frozen states, historical actions, fit logs, gates, ledgers, reports and code. Prior checkpoint retained. Fitted monthly model objects were not serialized; raw market data remain at original input paths.',
        verification=f'{count} archive members verified against SHA-256 of source files.',
        data_timing_note='Backward availability lags enforced; archived historical vendor versions and company mappings are not fully verified.',files=hashes)
    write_json(CONFIG,manifest)
    write_json(Path('config/current_contract_reference.json'),{k:v for k,v in manifest.items() if k != 'files'})
    print(results[['model','period','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)
    print(f'Saved {NAME}: {count} verified files, {ARCHIVE.stat().st_size/2**20:.1f} MiB',flush=True)


if __name__ == '__main__': main()
