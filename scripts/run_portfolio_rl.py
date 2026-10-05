"""Bounded monthly portfolio RL sizing experiment with exact cash-ledger replay."""
from pathlib import Path
import json
import resource
import warnings
import numpy as np
import pandas as pd
from scripts import run_vega_portfolio as engine
from scripts.discover_contract_signals import digest
from scripts.portfolio_rl_policy import LADDER,FEATURES,completed_mask,state_features,fit_predict

OUT=Path('artifacts/contracts/portfolio_rl');REPORT=Path('reports/portfolio_rl')


def episode_data(data,start,end):
    return (data[0][(data[0]>=start)&(data[0]<=end)],*data[1:5],{start:data[5][start]},*data[6:])


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);(OUT/'episodes').mkdir(exist_ok=True)
    engine.OUT=OUT;engine.REPORT=REPORT
    warnings.filterwarnings('ignore',message='Unrecognized options detected')
    sources=[Path(__file__),Path('scripts/portfolio_rl_policy.py'),Path('scripts/run_vega_portfolio.py'),Path('scripts/portfolio_vega.py'),Path('scripts/forecast_contract_variance.py'),Path('artifacts/contracts/variance_forecast/legs.parquet')]
    signature={str(p):digest(p) for p in sources}
    manifest=OUT/'inputs.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature,'Source changed; preserve prior checkpoints separately'
    else:manifest.write_text(json.dumps(signature,indent=2)+'\n')
    (OUT/'status.json').write_text(json.dumps(dict(status='running',stage='setup'))+'\n')
    data=engine.setup();dates=sorted(data[5]);paths={};rows=[]
    cal=data[0]
    for i,date in enumerate(dates):
        end=dates[i+1] if i+1<len(dates) else cal[-1]
        signal=data[1].index[data[1].index.get_loc(date)-1]
        rows.append(dict(date=date,signal_date=signal,end=end,**state_features(data[5][date])))
        episode=episode_data(data,date,end);values=[]
        for scale in LADDER:
            file=OUT/'episodes'/f'{date.date()}_{scale:.2f}.csv'
            if file.exists():f=pd.read_csv(file,index_col=0,parse_dates=[0])
            else:
                f=engine.run('forecast',episode,entry_scales={date:scale},quiet=True,persist=False)
                f.to_csv(file)
            assert f.index.equals(episode[0]) and f.names.iloc[-1]==0
            assert f.ledger_residual.abs().max()<1e-5
            value=(f.nav-f.cash_benchmark).diff();value.iloc[0]=f.nav.iloc[0]-f.cash_benchmark.iloc[0]
            values.append(value.to_numpy()/engine.CAPITAL)
        paths[date]=(episode[0],np.stack(values,axis=1))
        (OUT/'status.json').write_text(json.dumps(dict(status='running',stage='episodes',completed=i+1,total=len(dates)))+'\n')
        print(f'Episodes {i+1}/{len(dates)} complete: {date.date()}',flush=True)
    states=pd.DataFrame(rows)
    for i,row in states.iterrows():
        prior=states[(states.end<row.signal_date)&(states.date<row.date)]
        for n in [6,12]:
            chosen=prior.tail(n)
            states.loc[i,f'memory_{n}m']=np.mean([paths[d][1][:,-1].sum() for d in chosen.date]) if len(chosen) else np.nan
    states.to_csv(REPORT/'states.csv',index=False)
    decisions=[];rl={};nn={}
    for _,row in states[states.date>='2022-01-01'].iterrows():
        train=states.loc[completed_mask(states,row.signal_date)]
        assert len(train)>=18 and train.end.max()<row.signal_date
        test=states.loc[[row.name]]
        r,n,fit=fit_predict(train,test,paths);rl[row.date]=r;nn[row.date]=n
        decisions.append(dict(date=row.date,signal_date=row.signal_date,last_training_end=train.end.max(),training_months=len(train),rl_scale=r,nn_scale=n,**fit))
        print(f'Fit {row.date.date()}: RL {r:.2f}, NN {n:.0f}, {len(train)} completed months',flush=True)
    pd.DataFrame(decisions).to_csv(REPORT/'decisions.csv',index=False)
    common=(cal[cal>='2022-01-01'],*data[1:5],{d:q for d,q in data[5].items() if d>=pd.Timestamp('2022-01-01')},*data[6:])
    results=[]
    variants=[('forecast','forecast',None),('equal_rank','equal_rank',None),('nn_fixed','forecast',nn),('rl_sized','forecast',rl)]
    for name,mode,actions in variants:
        (OUT/name).mkdir(exist_ok=True)
        engine.OUT=OUT/name
        checkpoint=OUT/name/mode/'nav.csv'
        if checkpoint.exists():
            f=pd.read_csv(checkpoint,index_col=0,parse_dates=[0])
            assert f.index.equals(common[0]) and f.names.iloc[-1]==0
            assert f.ledger_residual.abs().max()<1e-5
        else:f=engine.run(mode,common,entry_scales=actions,quiet=True)
        results.append(dict(model=name,**engine.metrics(f)))
        (OUT/'status.json').write_text(json.dumps(dict(status='running',stage='replay',completed_models=[r['model'] for r in results]))+'\n')
        print(f'Replay {name}: {results[-1]}',flush=True)
        engine.quotes.cache_clear()
    (OUT/'status.json').write_text(json.dumps(dict(status='complete',results=results,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')

if __name__=='__main__':main()
