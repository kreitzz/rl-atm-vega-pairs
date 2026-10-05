"""Resume local preparation and evaluation when the bounded download is ready."""
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from scripts.curve_trading.config import DATA,OUTPUT,REPORT
from scripts.run_contract_walkforward import digest

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    environment=REPORT/'environment.json'
    if not environment.exists():
        environment.write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','pyarrow']},
            blas_threads=1,training_workers=2),indent=2)+'\n')
    runtime=OUTPUT/'runtime_dependencies.json'
    sources=['scripts/option_flex/contract_risk.py',str(environment),'scripts/check_curve_labels.py']
    signature={p:digest(p) for p in sources}
    if runtime.exists():assert json.loads(runtime.read_text())==signature,'Runtime dependencies changed'
    else:runtime.write_text(json.dumps(signature,indent=2)+'\n')
    jobs=json.loads((DATA/'puts_raw/requests.json').read_text())
    while True:
        ready=sum((DATA/'puts_raw'/j['path']).exists() and (DATA/'puts_raw'/j['path']).with_suffix('.json').exists() for j in jobs)
        if ready==len(jobs):break
        print(f'Waiting for put partitions: {ready}/{len(jobs)}',flush=True);time.sleep(30)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for arguments in [
        ['scripts.prepare_curve_contracts','--cp','C','--stage','select'],
        ['scripts.prepare_curve_contracts','--cp','C','--stage','history'],
        ['scripts.prepare_curve_contracts','--cp','P','--stage','select'],
        ['scripts.prepare_curve_contracts','--cp','P','--stage','history'],
        ['scripts.check_curve_labels'],
        ['scripts.run_curve_trading','--workers','2'],
        ['scripts.report_curve_trading'],
    ]:
        print('Starting '+' '.join(arguments),flush=True)
        subprocess.run([sys.executable,'-m',*arguments],check=True,env=env)

if __name__=='__main__':main()
