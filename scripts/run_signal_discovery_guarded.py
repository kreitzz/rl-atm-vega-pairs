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
    allowed={'scripts.discover_contract_signals','scripts.forecast_contract_variance','scripts.run_vega_portfolio','scripts.report_vega_portfolio','scripts.run_portfolio_rl','scripts.report_portfolio_rl','scripts.replay_frozen_atm_commission','scripts.retrain_atm_commission','scripts.audit_atm_commission_training','scripts.run_atm_cost_hurdles','scripts.audit_atm_cost_hurdles'}
    allowed.update({'scripts.run_atm_higher_cost_hurdles', 'scripts.audit_atm_higher_cost_hurdles',
                    'scripts.run_atm_nn_cost_hurdles', 'scripts.audit_atm_nn_cost_hurdles'})
    allowed.update({'scripts.continue_atm_nn_cost_hurdles', 'scripts.audit_atm_nn_cost_hurdles_complete'})
    allowed.add('scripts.report_atm_nn_full_roster')
    allowed.update({'scripts.run_atm65_ablation', 'scripts.run_atm65_ablation_audit'})
    allowed.add('scripts.diagnose_atm65_loss_years')
    allowed.update({'scripts.run_atm65_signal_mining', 'scripts.audit_atm65_signal_mining'})
    allowed.add('scripts.report_atm65_signal_concentration')
    allowed.update({'scripts.run_curve_management', 'scripts.audit_curve_management'})
    allowed.add('scripts.test_wrds_pair_signals')
    allowed.add('scripts.test_wrds_alternative_pairs')
    allowed.add('scripts.run_atm65_employee_sentiment')
    allowed.add('scripts.audit_atm65_employee_sentiment')
    allowed.add('scripts.save_employee_sentiment_checkpoint')
    allowed.add('scripts.run_atm65_transcripts')
    allowed.add('scripts.audit_atm65_transcripts')
    allowed.add('scripts.save_transcript_checkpoint')
    allowed.add('scripts.run_atm65_transcript_selector')
    allowed.add('scripts.run_atm65_rl_transcripts')
    allowed.add('scripts.export_current_rl_inputs')
    allowed.add('scripts.profile_vol_alternative_data')
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
    run(a.module,a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments,Path('artifacts/contracts/signal_discovery/STOP'),a.timeout)

if __name__=='__main__':main()
