"""Calibrate Sharpe rewards around a training-only portfolio reference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import signal_allocation_objectives as parent

CONFIG=Path('config/signal_allocation_policy_reward.json');DECLARED=json.loads(CONFIG.read_text())
CFG=dict(parent.CFG,**DECLARED);ROOT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
BASE_FIT=parent.fit;BASE_REWARD=parent.reward_parts
REFERENCE=None;POSITION=0


def choose_reference(data,index):
    cutoff=data['periods'][index]['cutoff']
    train=[m for m in data['periods'][:index] if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=CFG['training_weeks'])]
    if len(train)<CFG['minimum_training_weeks']:return None
    paths=np.stack([np.concatenate([m['paths'][a,a] for m in train]) for a in range(parent.N)])
    scores=np.sqrt(252)*paths.mean(axis=1)/np.maximum(paths.std(axis=1,ddof=1),1e-8)
    action=int(scores.argmax());ref=paths[action]
    return dict(action=action,scores=scores,total=float(ref.sum()),squares=float(ref@ref),count=len(ref),
        weeks=[m['paths'][action,action] for m in train],last=max(m['end'] for m in train))


def reward_parts(paths,cash,reference,total,squares,count):
    global POSITION
    ref=REFERENCE['weeks'][POSITION];POSITION+=1
    return BASE_REWARD(paths,cash,ref,REFERENCE['total'],REFERENCE['squares'],REFERENCE['count'])


def fit(data,index):
    global REFERENCE,POSITION
    REFERENCE=choose_reference(data,index);POSITION=0
    if REFERENCE is None:return None
    assert REFERENCE['last']<data['periods'][index]['cutoff']
    result=BASE_FIT(data,index);assert POSITION==len(REFERENCE['weeks'])
    month=str(data['periods'][index]['cutoff'].to_period('M'))
    for obj in parent.objectives():
        for gamma in [CFG['discount'],0.]:
            path=ROOT/'models'/parent.name(obj['key'],gamma)/(month+'.json')
            value=json.loads(path.read_text());value.update(reference_action=REFERENCE['action'],
                training_reference_sharpes=REFERENCE['scores'].tolist(),reference_kind='training-only highest-Sharpe constant capital mix',
                reference_sharpe=float(REFERENCE['scores'][REFERENCE['action']]),reference_mean=REFERENCE['total']/REFERENCE['count'],
                reference_sessions=REFERENCE['count'])
            parent.base.dump(path,value)
    return result


def sources():
    paths=[CONFIG,Path(__file__),Path(parent.__file__),Path(parent.base.__file__),parent.SOURCE/'prepared.joblib',parent.SOURCE/'sources.json']
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};path=ROOT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:parent.base.dump(path,record)


def configure():
    parent.CONFIG=CONFIG;parent.CFG=CFG;parent.ROOT,parent.REPORT=ROOT,REPORT
    parent.reward_parts=reward_parts;parent.fit=fit;parent.sources=sources;parent.configure()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['test','fixed','validation','select','evaluate','analyze','charts']);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);configure();sources()
    if a.phase=='test':
        import unittest
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_signal_allocation_policy_reward')).wasSuccessful():raise SystemExit(1)
    elif a.phase=='fixed':parent.base.fixed()
    elif a.phase=='validation':parent.run(CFG['validation_end'])
    elif a.phase=='evaluate':parent.run(CFG['evaluation_end'])
    else:getattr(parent,a.phase)()


if __name__=='__main__':main()
