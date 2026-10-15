"""Resume the declared study, using ONE guarded numerical subprocess at a time.

Standard library only. Keeps readable status, per-job logs and completed-job
markers. STOP halts both this supervisor and its guarded worker. A lock prevents
accidentally starting two research workers. No app closing or memory purging.
"""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/straddle_overnight_study.json'
CFG=json.loads(CONFIG.read_text())
OUT=ROOT/CFG['output_root'];REPORT=ROOT/CFG['report_root']
PYTHON='/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/.venv/bin/python'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def bandit_specs():
    rows=[f'{f}_tree4_w{w}' for f in CFG['feature_groups'] for w in CFG['training_windows_months']]
    return rows+[f'market_{k}_w{w}' for k in ['tree8','ridge'] for w in CFG['training_windows_months']]


def q_specs():
    q=CFG['q_study'];rows=[]
    for t,f,s,w,a,g in itertools.product(q['tasks'],q['frequencies'],q['feature_groups'],
        q['training_windows_months'],q['ridge_alphas'],q['discounts']):
        frequency='monthly' if f=='M' else 'weekly'
        name=f'{t}_{frequency}_{s}_w{w}_a{a:g}_g{g:g}'.replace('.','p')
        rows.append(dict(task=t,frequency=f,case=name))
    return rows


def retryable_failure(text):
    """Only explicit resource failures qualify for automatic same-limit retry."""
    return any(s in text for s in ['MemoryError: System swap increased',
        'MemoryError: System memory pressure','LowMemoryHeadroom:',
        'TimeoutError: Job wall-clock limit reached'])


class Supervisor:
    def __init__(self,hours,deadline=None):
        self.started=time.time();self.deadline=min(self.started+hours*3600,deadline or float('inf'));self.completed=0
        (OUT/'jobs').mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)

    def job(self,key,module,args):
        marker=OUT/'jobs'/f'{key}.json'
        command=[PYTHON,'-m','scripts.run_signal_discovery_guarded','--module',module,
            '--timeout','1800','--duty-cycle','0.2','--rss-limit-mib','256',
            '--warning-rss-limit-mib','256','--min-free-percent','30','--warning-min-free-percent','30',
            '--max-swap-growth-mib','256','--pause-low-headroom-seconds','60',
            '--stop-file',str(OUT/'STOP'),'--']+args
        signature=hashlib.sha256(json.dumps(command).encode()).hexdigest()
        if marker.exists():
            assert json.loads(marker.read_text())['signature']==signature
            return
        attempt=0;stalled=0
        while stalled<12:
            attempt+=1
            if (OUT/'STOP').exists():raise InterruptedError('STOP requested')
            if time.time()>=self.deadline:raise TimeoutError('Eight-hour study budget reached; checkpoints retained')
            log=OUT/'logs'/f'{key}.{int(self.started)}.{attempt}.log'
            # Completed months/models are durable progress even when a later
            # memory stop prevents the whole requested batch from finishing.
            before_progress=sum(1 for _ in OUT.glob('cases/*/replays/tree18/months/*/checkpoint.joblib'))+sum(1 for _ in OUT.glob('q/models/*/*.json'))
            status=dict(pid=os.getpid(),state='running',job=key,module=module,arguments=args,
                attempt=attempt,started=self.started,updated=time.time(),completed_this_session=self.completed,
                deadline=self.deadline,worker_limits=CFG['resource_limits'],log=str(log.relative_to(ROOT)))
            dump(OUT/'status.json',status);print(json.dumps(status),flush=True)
            with log.open('w') as output:
                process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if (OUT/'STOP').exists() or time.time()>=self.deadline:
                        # The guard reads STOP and terminates its OWN worker.
                        (OUT/'STOP').touch();process.wait(timeout=15)
                        raise InterruptedError('STOP or wall-clock budget reached; checkpoints retained')
                    time.sleep(1)
            text=log.read_text()
            if process.returncode==0:
                dump(marker,dict(signature=signature,finished=time.time(),log=str(log.relative_to(ROOT)),attempt=attempt))
                self.completed+=1;return
            # Match actual resource failures, not 'headroom' appearing in a
            # traceback's function signature (which previously masked EOFError).
            retryable=retryable_failure(text)
            if not retryable:
                raise RuntimeError(f'{key} failed; inspect {log.relative_to(ROOT)}\n'+text[-1600:])
            after_progress=sum(1 for _ in OUT.glob('cases/*/replays/tree18/months/*/checkpoint.joblib'))+sum(1 for _ in OUT.glob('q/models/*/*.json'))
            stalled=0 if after_progress>before_progress else stalled+1
            dump(OUT/'status.json',dict(status,state='waiting_for_memory',updated=time.time(),last_exit=process.returncode))
            # Deliberately no faster/larger fallback when the computer is busy.
            for _ in range(30):
                if (OUT/'STOP').exists():raise InterruptedError('STOP requested')
                time.sleep(1)
        raise RuntimeError(f'{key}: repeated memory stops; preserve checkpoints and stop rather than weaken guard')


def collect_csv(paths,destination):
    rows=[];fields=[]
    for p in sorted(paths):
        with p.open(newline='') as stream:
            for row in csv.DictReader(stream):
                rows.append(row)
                for field in row:
                    if field not in fields:fields.append(field)
    if rows:
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('w',newline='') as output:
            w=csv.DictWriter(output,fields);w.writeheader();w.writerows(rows)


def consolidate():
    collect_csv((REPORT/'cases').glob('*/results.csv'),REPORT/'results.csv')
    collect_csv((REPORT/'cases').glob('*/annual.csv'),REPORT/'annual.csv')
    collect_csv((REPORT/'q/cases').glob('*/*/results.csv'),REPORT/'q/results.csv')
    collect_csv((REPORT/'q/cases').glob('*/*/annual.csv'),REPORT/'q/annual.csv')
    lines=['OVERNIGHT STUDY: recorded results, including failures',
        'Validation: 2018-2021. Later evaluation: 2022-Aug2025. These dates were already inspected.',
        'Primary selections use validation only within this new study; never replace them with the best later winner.',
        'Entry sizing uses actual funded option replay. Q allocation uses fractional strategy indices, not shared option inventory.','']
    for label,path,period in [('ENTRY BANDITS',REPORT/'results.csv','evaluation'),('Q ALLOCATION',REPORT/'q/results.csv','test')]:
        lines.append(label)
        if not path.exists():lines.append('Not completed.');continue
        with path.open() as stream:rows=list(csv.DictReader(stream))
        grouped={}
        for r in rows:grouped.setdefault((r.get('task',''),r.get('frequency',''),r['case']),{})[r['period']]=r
        for key,values in grouped.items():
            v=values.get('validation',{});e=values.get(period,{})
            lines.append(' / '.join(x for x in key if x)+': validation '+v.get('annualized_excess_sharpe','?')+'; later '+e.get('annualized_excess_sharpe','?'))
        lines.append('')
    (REPORT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')


def run(supervisor):
    b='scripts.straddle_overnight_study';q='scripts.straddle_overnight_q';job=supervisor.job
    job('bandit_test',b,['--phase','test']);job('features',b,['--phase','features','--limit','250'])
    job('assemble',b,['--phase','assemble']);job('q_test',q,['--phase','test'])
    # Alternate families so the Q work begins before the longer funded sweep.
    for t,f in itertools.product(CFG['q_study']['tasks'],CFG['q_study']['frequencies']):
        for c in ['constant_1','constant_0','constant_2','rule','random_7','random_19','random_41']:
            job(f'q_val_{t}_{f}_{c}',q,['--phase','validation','--task',t,'--frequency',f,'--case',c,'--control',c])
    queue=q_specs();cursor=0
    for name in bandit_specs():
        job('fit_'+name,b,['--phase','fit','--case',name])
        for chunk in range(6):
            job(f'val_{name}_{chunk}',b,['--phase','book','--case',name,'--months','8'])
            # 96 Q procedures spread across 72 bandit chunks, sequentially.
            take=2 if cursor<24 else 1
            for _ in range(take):
                if cursor>=len(queue):break
                item=queue[cursor];cursor+=1
                job('q_val_'+item['case'],q,['--phase','validation','--case',item['case']])
        if name=='core_tree4_w36':job('replication',b,['--phase','replication'])
    while cursor<len(queue):
        item=queue[cursor];cursor+=1
        job('q_val_'+item['case'],q,['--phase','validation','--case',item['case']])
    job('bandit_select',b,['--phase','select']);job('q_select',q,['--phase','select'])
    chosen=json.loads((REPORT/'selection.json').read_text())['later_cases']
    for name in chosen:
        job('later_fit_'+name,b,['--phase','fit','--case',name,'--later'])
        for chunk in range(6):job(f'later_{name}_{chunk}',b,['--phase','book','--case',name,'--later','--months','8'])
        job('analyze_'+name,b,['--phase','analyze','--case',name]);consolidate()
    qjobs=json.loads((REPORT/'q/selection.json').read_text())['later_jobs']
    # Materialize matched references before per-case inference.
    order=sorted(range(len(qjobs)),key=lambda i:(qjobs[i]['case']!='constant_1',i))
    for i in order:job('q_later_'+str(i),q,['--phase','evaluate','--job-index',str(i)])
    for item in qjobs:
        job('q_analyze_'+item['task']+'_'+item['frequency']+'_'+item['case'],q,
            ['--phase','analyze','--task',item['task'],'--frequency',item['frequency'],'--case',item['case']])
        consolidate()
    consolidate()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hours',type=float,default=8)
    p.add_argument('--detach',action='store_true');p.add_argument('--deadline',type=float);a=p.parse_args()
    if not 0<a.hours<=8:p.error('--hours must be greater than zero and at most eight')
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (OUT/'supervisor.log').open('a') as output:
            command=[sys.executable,'-m','scripts.launch_straddle_overnight','--hours',str(a.hours)]
            if a.deadline is not None:command+=['--deadline',str(a.deadline)]
            child=subprocess.Popen(command,
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,maximum_hours=a.hours,status=str(OUT/'status.json'),
            log=str(OUT/'supervisor.log'),limits=CFG['resource_limits'])),flush=True)
        return
    with (OUT/'runner.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('The study supervisor is already running')
        supervisor=Supervisor(a.hours,a.deadline)
        try:
            run(supervisor)
            dump(OUT/'status.json',dict(state='complete',finished=time.time(),completed_this_session=supervisor.completed,report=str(REPORT/'FINDINGS.txt')))
        except BaseException as e:
            consolidate();dump(OUT/'status.json',dict(state='stopped',updated=time.time(),error=str(e),completed_this_session=supervisor.completed))
            raise


if __name__=='__main__':main()
