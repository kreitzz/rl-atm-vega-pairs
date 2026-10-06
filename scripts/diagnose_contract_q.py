"""Audit the original contract Q and training-only round trips; no model selection.

Run in xfund. Hindsight outcomes are diagnostics, never a tradable strategy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, UnresolvedPosition
from scripts.contract_q_engine import LONG, SHORT, WAIT, EXIT
from scripts.run_contract_q import FittedQ, market_slice


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), p10=float(np.quantile(a,.1)),
                median=float(np.median(a)), p90=float(np.quantile(a,.9)),
                positive_fraction=float((a>0).mean())) if len(a) else dict(n=0)


def main():
    out=Path('artifacts/contract_q/diagnosis')
    out.mkdir(parents=True,exist_ok=True)
    bundle=pd.read_pickle('data/q_contracts/AMD_NVDA/prepared.pkl')
    cfg=Settings(**bundle['settings'])
    result={}
    for label,start,end in [('train','2010-01-01','2023-01-01'),
                            ('validation','2023-01-01','2024-01-01')]:
        m=market_slice(bundle,start,end,cfg)
        idx=[t for t in range(m.n-1) if LONG in m.legal(t,None)]
        x=np.array([m.features(t,None) for t in idx])
        result[label]={}
        for name in ['q_downside_0','q_downside_0.1']:
            model=FittedQ.load(Path('artifacts/contract_q/AMD_NVDA/2024')/f'{name}.pkl')
            q=model.predict(x)
            frame=pd.DataFrame(q,columns=['wait','long','short','exit'],index=m.dates[idx])
            frame['entry_advantage']=q[:,1:3].max(axis=1)-q[:,0]
            frame.to_csv(out/f'{label}_{name}_flat_values.csv')
            result[label][name]=dict(eligible_days=len(idx),
                greedy_entry_days=int((frame.entry_advantage>0).sum()),
                entry_advantage_bps=distribution(frame.entry_advantage),
                wait_value_bps=distribution(q[:,0]),
                best_entry_value_bps=distribution(q[:,1:3].max(axis=1)),
                sweep_changes=model.history)
        if label!='train':
            continue
        # Fixed every-tenth-session schedule, independent of prices and outcomes.
        paths=[]
        statuses={}
        for t in range(0,m.n-32,10):
            if LONG not in m.legal(t,None):
                continue
            for action in (LONG,SHORT):
                book,reward,info=m.step(t,None,action)
                statuses[info['status']]=statuses.get(info['status'],0)+1
                if book is None:
                    continue
                pnl=reward
                costs=info['option_cost']+info['hedge_cost']
                entry=m.dates[t+1]
                for j in range(t+1,min(t+32,m.n-1)):
                    try:
                        _,er,ei=m.step(j,book,EXIT)
                        close_quotes=[m.quote(j+1,oid) for oid in book.optionids]
                        paths.append(dict(signal_date=str(m.dates[t].date()),
                            entry_date=str(entry.date()),exit_date=str(m.dates[j+1].date()),
                            direction=action,holding_sessions=j-t,
                            net_dollars=pnl+er,execution_cost_dollars=costs+ei['option_cost']+ei['hedge_cost'],
                            before_execution_cost_dollars=pnl+er+costs+ei['option_cost']+ei['hedge_cost'],
                            optionid_a=book.optionids[0],optionid_b=book.optionids[1],
                            entry_iv_a=book.entry_quotes[0]['impl_volatility']*100,
                            entry_iv_b=book.entry_quotes[1]['impl_volatility']*100,
                            exit_iv_a=close_quotes[0]['impl_volatility']*100,
                            exit_iv_b=close_quotes[1]['impl_volatility']*100,
                            entry_spot_a=m.spots[t+1,0],entry_spot_b=m.spots[t+1,1],
                            exit_spot_a=m.spots[j+1,0],exit_spot_b=m.spots[j+1,1]))
                        book,hr,hi=m.step(j,book,WAIT)
                        pnl+=hr
                        costs+=hi['option_cost']+hi['hedge_cost']
                        if book is None:
                            break
                    except UnresolvedPosition as exc:
                        raise RuntimeError('Do not silently drop unresolved diagnostic paths') from exc
        frame=pd.DataFrame(paths)
        frame.to_csv(out/'training_round_trips.csv',index=False)
        last=frame.groupby(['signal_date','direction']).tail(1)
        result['sampled_training_paths']=dict(entry_statuses=statuses,
            completed_tickets=len(last),fixed_hold_net=distribution(last.net_dollars),
            fixed_hold_before_execution_cost=distribution(last.before_execution_cost_dollars),
            fixed_hold_execution_cost=distribution(last.execution_cost_dollars),
            horizons={str(h):distribution(frame.loc[frame.holding_sessions.eq(h),'net_dollars']) for h in [1,5,10,20]},
            hindsight_best_exit=distribution(frame.groupby(['signal_date','direction']).net_dollars.max()),
            hindsight_best_direction_fixed_hold=distribution(last.groupby('signal_date').net_dollars.max()))
        result['sampled_training_paths']['annual_mean_execution_cost_dollars']={
            str(year):float(value) for year,value in
            last.groupby(pd.to_datetime(last.signal_date).dt.year).execution_cost_dollars.mean().items()}
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
