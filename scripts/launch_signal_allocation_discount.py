"""Serial small-worker comparison of the two discount factors."""
import argparse
import fcntl
import json
import time
from pathlib import Path
from scripts import launch_straddle_overnight as guarded


def main():
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=2.);a=p.parse_args()
    cfg=json.loads(Path('config/signal_allocation_discount.json').read_text());guarded.CFG=cfg
    guarded.OUT=Path(cfg['output_root']).resolve();guarded.OUT.mkdir(parents=True,exist_ok=True)
    lock=(guarded.OUT/'supervisor.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    supervisor=guarded.Supervisor(a.hours)
    try:
        for phase in ['fixed','validation','select','evaluate','analyze']:
            supervisor.job(phase,'scripts.signal_allocation_discount',['--phase',phase])
        guarded.dump(guarded.OUT/'status.json',dict(state='complete',jobs=supervisor.completed,finished=time.time()))
    except BaseException as exc:
        guarded.dump(guarded.OUT/'status.json',dict(state='stopped',error=repr(exc),finished=time.time()));raise


if __name__=='__main__':main()
