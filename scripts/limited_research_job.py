"""Run a single research process group with a strict wall-clock duty cycle.

No worker pool. STOP file and parent interruption terminate only this child
process group. One BLAS/Arrow thread, low scheduling priority, 50% duty cycle.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DUTY_CYCLE = .5
PERIOD_SECONDS = .2

def run(module,args,stop_file,timeout=1800):
    allowed={'scripts.download_curve90','scripts.prepare_curve90','scripts.run_curve90','scripts.check_curve90','scripts.report_curve90'}
    if module not in allowed:raise ValueError('Unapproved research module')
    if stop_file.exists():raise InterruptedError('STOP file present')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',ARROW_NUM_THREADS='1')
    env['RESEARCH_SUPERVISOR_PID']=str(os.getpid())
    bootstrap="""import os,runpy,sys,threading,time
os.nice(15)
def watchdog():
    while True:
        if os.getppid()!=int(os.environ['RESEARCH_SUPERVISOR_PID']):os._exit(143)
        time.sleep(.2)
threading.Thread(target=watchdog,daemon=True).start()
import pyarrow
pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1)
m=sys.argv.pop(1);runpy.run_module(m,run_name='__main__')
"""
    child=subprocess.Popen([sys.executable,'-c',bootstrap,module,*args],env=env,start_new_session=True)
    started=time.monotonic();last_check=started
    def send(sig):
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:child.wait()
        except PermissionError as exc:
            # macOS may return EPERM when the child exits between poll and
            # killpg. Suppress it only after waitpid confirms termination.
            # waitpid can lag the kernel's process-group teardown. Allow a
            # bounded reap window; a live inaccessible child is still fatal.
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:raise exc
    try:
        while child.poll() is None:
            if stop_file.exists():raise InterruptedError('STOP file present')
            if time.monotonic()-started>timeout:raise TimeoutError('Job wall-clock limit reached; checkpoint retained')
            # Cap the whole process group, not merely the Python parent.
            send(signal.SIGCONT);time.sleep(PERIOD_SECONDS * DUTY_CYCLE)
            if child.poll() is not None:break
            send(signal.SIGSTOP);time.sleep(PERIOD_SECONDS * (1-DUTY_CYCLE))
            if time.monotonic()-last_check>=10 and child.poll() is None:
                last_check=time.monotonic()
                snapshot=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
                if snapshot.returncode and child.poll() is None:raise RuntimeError('Cannot monitor worker memory; stopped safely')
                rss=snapshot.stdout.strip()
                if rss and int(rss)>3*1024*1024:raise MemoryError('Worker exceeded 3 GiB RSS; stopped safely')
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,module)
    finally:
        if child.poll() is None:
            send(signal.SIGCONT);send(signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def main():
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    run(a.module,a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments,Path('data/curve90/STOP'),a.timeout)

if __name__=='__main__':main()
