"""Three bounded independent workers; immutable per-worker resume inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from scripts.run_broad_contract_replay import universe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3,choices=[1,2,3])
    args=parser.parse_args()
    root=Path('artifacts/contracts/aligned_learning');report=Path('reports/aligned_contract_learning')
    root.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    definitions=universe();groups=[definitions[i::args.workers] for i in range(args.workers)]
    spec=dict(workers=args.workers,groups=[[d['pair'] for d in group] for group in groups],
              runner_sha256=hashlib.sha256(Path('scripts/run_aligned_contract_learning.py').read_bytes()).hexdigest())
    path=root/'batch_inputs.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Batch inputs changed')
    path.write_text(json.dumps(spec,indent=2)+'\n')
    (report/'universe.json').write_text(json.dumps(definitions,indent=2)+'\n')
    processes=[];logs=[];started=time.monotonic()
    try:
        for i,group in enumerate(groups):
            folder=root/f'worker_{i}';folder.mkdir(exist_ok=True)
            log=(folder/'run.log').open('a');logs.append(log)
            command=[sys.executable,'-m','scripts.run_aligned_contract_learning','--output',str(folder),
                     '--report',str(folder/'report'),'--resume','--pairs',*[d['pair'] for d in group]]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            processes.append(process)
        (root/'processes.json').write_text(json.dumps(dict(coordinator=os.getpid(),workers=[p.pid for p in processes]))+'\n')
        while True:
            records=[];peaks=[]
            for i,group in enumerate(groups):
                folder=root/f'worker_{i}'
                for d in group:
                    status=folder/d['pair']/'status.json'
                    if status.exists():
                        try:records.extend(json.loads(status.read_text()))
                        except json.JSONDecodeError:pass
                progress=folder/'report/progress.json'
                if progress.exists():
                    try:peaks.append(json.loads(progress.read_text())['peak_rss_bytes'])
                    except json.JSONDecodeError:pass
            pd.DataFrame(records).to_csv(report/'status.csv',index=False)
            info=dict(completed_pairs=len(records)//2,total_pairs=len(definitions),workers=args.workers,
                      elapsed_seconds=time.monotonic()-started,worker_peak_rss_sum_bytes=sum(peaks),
                      status='running' if any(p.poll() is None for p in processes) else 'finished')
            (report/'progress.json').write_text(json.dumps(info,indent=2)+'\n')
            print(json.dumps(info),flush=True)
            failures=[i for i,p in enumerate(processes) if p.poll() not in [None,0]]
            if failures:raise RuntimeError(f'Worker failure; inspect worker run.log: {failures}')
            if sum(peaks)>3*1024**3:raise MemoryError('Combined worker high-water RSS exceeded 3 GiB guard')
            if all(p.poll()==0 for p in processes):break
            time.sleep(30)
        if len(records)!=2*len(definitions):raise AssertionError('Missing completed pair attempts')
        sources=[];manifests=[]
        for i,group in enumerate(groups):
            folder=root/f'worker_{i}'
            metadata=folder/'inputs.json'
            sources.append(dict(path=str(metadata),sha256=hashlib.sha256(metadata.read_bytes()).hexdigest()))
            manifests.append(json.loads((folder/'report/manifest.json').read_text()))
            for d in group:
                link=root/d['pair']
                if not link.exists():link.symlink_to(Path(f'worker_{i}')/d['pair'],target_is_directory=True)
        manifest=dict(manifests[0],pairs=len(definitions),workers=args.workers,worker_inputs=sources,
                      peak_rss_bytes=max(m['peak_rss_bytes'] for m in manifests),
                      worker_peak_rss_sum_bytes=sum(m['peak_rss_bytes'] for m in manifests),
                      elapsed_seconds=time.monotonic()-started,inputs_sha256=None)
        (report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        subprocess.run([sys.executable,'-m','scripts.report_aligned_contract_learning'],check=True)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()
        for log in logs:log.close()


if __name__=='__main__':main()
