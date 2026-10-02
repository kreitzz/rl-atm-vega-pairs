"""Checkpointed 90D experiment; one capped child, pauses between every job."""
import argparse
import fcntl
import hashlib
import json
import shutil
import signal
from pathlib import Path
import time
from scripts.limited_research_job import run,DUTY_CYCLE

DATA=Path('data/curve90');OUTPUT=Path('artifacts/contracts/curve90')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    parser=argparse.ArgumentParser();parser.add_argument('--max-jobs',type=int,default=1)
    args=parser.parse_args();DATA.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    defs=json.loads(Path('artifacts/contracts/curve_trading/inputs.json').read_text())['definitions']
    jobs=json.loads(Path('data/curve_trading/puts_raw/requests.json').read_text())
    code=[*Path('scripts/curve90').glob('*.py'),*[Path('scripts')/p for p in ['run_curve90.py','prepare_curve90.py','download_curve90.py',
        'finish_curve90.py','limited_research_job.py','check_curve90.py','report_curve90.py','prepare_curve_contracts.py',
        'select_atm_contracts.py','run_contract_pilot.py']],Path('scripts/option_flex/contract_engine.py'),
        Path('scripts/option_flex/contract_risk.py'),Path('scripts/option_flex/black_scholes.py')]
    signature=dict(code={str(p):digest(p) for p in code},definitions=defs,jobs=jobs,target=90,dte_window=[75,105],runoff=110,
        duty_cycle=DUTY_CYCLE,workers=1,max_rss_gib=3,pause_seconds=1,baseline=digest('artifacts/contracts/curve_trading/inputs.json'))
    frozen=OUTPUT/'inputs.json'
    if frozen.exists():assert json.loads(frozen.read_text())==signature,'90D inputs changed'
    else:frozen.write_text(json.dumps(signature,indent=2)+'\n')
    todo=[]
    pilot=next(d for d in defs if d['label']=='AMD/NVDA')
    pilot_ids=set(pilot['secids'])
    first=[i for i,j in enumerate(jobs) if pilot_ids.intersection(j['secids'])]
    # Include every partition sharing one of those securities, across years,
    # so current/prior-year selection dependencies are ready before histories.
    while True:
        expanded={sid for i in first for sid in jobs[i]['secids']}
        closure=[i for i,j in enumerate(jobs) if expanded.intersection(j['secids'])]
        if closure==first:break
        first=closure
    rest=[i for i in range(len(jobs)) if i not in first]
    def prepare_partitions(indexes):
        for i in indexes:
            j=jobs[i]
            if not (DATA/'raw'/j['path']).with_suffix('.json').exists():todo.append(('download',i))
            if not (DATA/'metadata/select'/str(j['year'])/(Path(j['path']).stem+'.json')).exists():todo.append(('select',i))
        for i in indexes:
            j=jobs[i]
            if not (DATA/'metadata/history'/str(j['year'])/(Path(j['path']).stem+'.json')).exists():todo.append(('history',i))
    prepare_partitions(first)
    if not (OUTPUT/'reconciliation.json').exists():todo.append(('check',None))
    if any(not (OUTPUT/pilot['pair']/v/'status.json').exists() for v in ['atm_control','curve']):todo.append(('train',pilot['pair']))
    if not (OUTPUT/'memory_equivalence.json').exists():todo.append(('verify',pilot['pair']))
    prepare_partitions(rest)
    for d in [d for d in defs if d['pair']!=pilot['pair']]:
        if any(not (OUTPUT/d['pair']/v/'status.json').exists() for v in ['atm_control','curve']):todo.append(('train',d['pair']))
    if not (OUTPUT/'task_status.json').exists():todo.append(('report',None))
    for count,(stage,key) in enumerate(todo[:args.max_jobs],1):
        if shutil.disk_usage(DATA).free<10*1024**3:raise RuntimeError('Less than 10 GiB disk free; stopped before next job')
        if stage=='download':module,arguments='scripts.download_curve90',['--index',str(key)]
        elif stage in ['select','history']:module,arguments='scripts.prepare_curve90',['--stage',stage,'--index',str(key)]
        elif stage=='train':module,arguments='scripts.run_curve90',['--pair',key]
        elif stage=='verify':module,arguments='scripts.run_curve90',['--pair',key,'--verify-existing']
        else:module,arguments='scripts.'+('check_curve90' if stage=='check' else 'report_curve90'),[]
        status=dict(stage=stage,key=key,completed_this_batch=count-1,remaining_at_start=len(todo),status='running')
        (OUTPUT/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
        print(f'{stage} {key}: starting capped job',flush=True)
        try:run(module,arguments,DATA/'STOP',timeout=1800)
        except BaseException as exc:
            status.update(status='stopped',error=str(exc));(OUTPUT/'progress.json').write_text(json.dumps(status,indent=2)+'\n');raise
        status.update(status='checkpointed',completed_this_batch=count)
        (OUTPUT/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
        time.sleep(1)
    lock.close()

if __name__=='__main__':main()
