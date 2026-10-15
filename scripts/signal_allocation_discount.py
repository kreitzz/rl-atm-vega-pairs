"""Matched discount0.5/0.99 rerun; no other economic or reward change."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from scripts import signal_capital_allocation as base

CONFIG=Path('config/signal_allocation_discount.json');DECLARED=json.loads(CONFIG.read_text())
CFG=dict(base.CFG,**DECLARED);ROOT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
SOURCE=Path(CFG['source_root']);BASE_FIT=base.fit


def sources():
    paths=[CONFIG,Path(__file__),Path(base.__file__),SOURCE/'prepared.joblib',SOURCE/'sources.json']
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    path=ROOT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:base.dump(path,record)


def load():
    import joblib
    return joblib.load(SOURCE/'prepared.joblib')


def fit(data,index):
    fitted=None
    try:
        for gamma in CFG['discounts']:
            base.CFG['discount']=gamma
            fitted=BASE_FIT(data,index)
    finally:base.CFG['discount']=DECLARED['discount']
    return fitted


def configure():
    base.CFG=CFG;base.ROOT,base.REPORT=ROOT,REPORT;base.load,base.fit,base.sources=load,fit,sources


def run(stop):
    data=load();records=[]
    assert (REPORT/'endpoint_verification.json').exists()
    for gamma in CFG['discounts']+[0.]:
        for risk in CFG['risk_preferences']:
            name=base.case(risk,gamma);book=base.replay(data,name,stop,risk,gamma)
            for period,start,end in base.PERIODS:
                if pd.Timestamp(start)>pd.Timestamp(stop):continue
                records.append(dict(kind='q' if gamma else 'myopic',case=name,period=period,risk=risk,gamma=gamma,
                                    **base.metrics(book,start,min(end,stop))))
    pd.DataFrame(records).to_csv(REPORT/('validation.csv' if stop==CFG['validation_end'] else 'q_results.csv'),index=False)


def select():
    frame=pd.read_csv(REPORT/'validation.csv');records=[]
    for gamma in CFG['discounts']:
        r=frame.loc[frame.gamma.eq(gamma)&frame.period.eq('validation')].sort_values(
            ['annualized_excess_sharpe','case'],ascending=[False,True]).iloc[0]
        records.append(dict(gamma=gamma,case=r['case'],risk=float(r.risk)))
    value=dict(selected=records,rule='Highest validation Sharpe separately per discount; no later replacement')
    path=REPORT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:base.dump(path,value)


def analyze():
    base.analyze_backup=None
    frame=pd.concat([pd.read_csv(REPORT/'fixed_results.csv'),pd.read_csv(REPORT/'q_results.csv')],ignore_index=True)
    frame.to_csv(REPORT/'results.csv',index=False);timing=[]
    for path in ROOT.glob('models/*/*.json'):
        m=json.loads(path.read_text());assert pd.Timestamp(m['last_training_end'])<pd.Timestamp(m['cutoff'])
        timing.append(dict(path=str(path),cutoff=m['cutoff'],last_training_end=m['last_training_end']))
    pd.DataFrame(timing).to_csv(REPORT/'model_timing.csv',index=False)
    chosen=json.loads((REPORT/'selection.json').read_text())['selected']
    frame.loc[frame.case.isin([r['case'] for r in chosen])].to_csv(REPORT/'validation_selected_results.csv',index=False)
    # Every nonendpoint benchmark must also match the preceding corrected study.
    old=pd.read_csv('reports/signal_capital_allocation/fixed_results.csv')
    pd.testing.assert_frame_equal(pd.read_csv(REPORT/'fixed_results.csv'),old)
    base.dump(REPORT/'verification.json',dict(fixed_results_unchanged=True,model_timing_checks=len(timing),
        discounts=CFG['discounts'],bellman_updates=CFG['iterations']))
    (REPORT/'FINDINGS.txt').write_text(DECLARED['scope']+'\n\n'+frame.loc[frame.kind.eq('q')&frame.period.ne('full')].to_string(index=False)+'\n')
    print(frame.loc[frame.kind.eq('q')&frame.period.eq('evaluation')].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['fixed','validation','select','evaluate','analyze']);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);configure();sources()
    if a.phase=='fixed':base.fixed()
    elif a.phase=='validation':run(CFG['validation_end'])
    elif a.phase=='evaluate':run(CFG['evaluation_end'])
    else:globals()[a.phase]()


if __name__=='__main__':main()
