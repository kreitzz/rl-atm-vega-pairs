"""One serial partition: 90D expiry anchors, delta selection, or held histories."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.curve90.config import DATA,config
from scripts.prepare_curve_contracts import choose,reduce_choices,HISTORY_COLUMNS
from scripts.run_contract_pilot import normalize_quotes
from scripts.option_flex.black_scholes import fit_quotes
from scripts.select_atm_contracts import select_candidates,load_calendar,SORT,ASC
from scripts.run_contract_walkforward import digest

ROOT=Path('data/option_contracts')
STREAM_BATCH_ROWS=32768

def jobs():return json.loads(Path('data/curve_trading/puts_raw/requests.json').read_text())

def prepare(index,stage):
    job=jobs()[index];year=job['year'];raw=DATA/'raw'/job['path']
    meta=DATA/'metadata'/stage/str(year)/(raw.stem+'.json');meta.parent.mkdir(parents=True,exist_ok=True)
    if meta.exists():
        old=json.loads(meta.read_text())
        for p,h in old['files'].items():assert digest(p)==h,p
        return
    stock=pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet');stock['date']=pd.to_datetime(stock.date)
    stock=stock[stock.secid.isin(job['secids'])]
    cal=load_calendar(ROOT);stock['entry_date']=stock.date.map(pd.Series(cal[1:].values,index=cal[:-1]))
    zero=pd.read_parquet(ROOT/'zero_curve.parquet');div=pd.read_parquet(ROOT/'distributions.parquet');files={}
    if stage=='select':
        best=pd.DataFrame()
        # Preserve the previous ATM-anchor convention: nearest eligible expiry, then strike.
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=STREAM_BATCH_ROWS,use_threads=False):
            q=normalize_quotes(batch.to_pandas());c=select_candidates(q,stock,replace(config(),max_moneyness_distance=.1))
            if len(c):best=pd.concat([best,c],ignore_index=True).sort_values(SORT,ascending=ASC).drop_duplicates(['secid','date'])
        if len(best):
            anchor=best[['date','secid','expiry','strike','optionid']].rename(columns={'strike':'anchor_strike','optionid':'anchor_id'}).assign(tenor=90)
            expiry=anchor[['date','secid','expiry']];choices={'C':pd.DataFrame(),'P':pd.DataFrame()}
            for batch in pq.ParquetFile(raw).iter_batches(batch_size=STREAM_BATCH_ROWS,use_threads=False):
                q=normalize_quotes(batch.to_pandas());q['relative_width']=(q.ask-q.bid)/((q.ask+q.bid)/2)
                q=q[q.standard&q.multiplier.eq(100)&q.bid.gt(0)&q.ask.ge(q.bid)&q.open_interest.ge(100)&q.relative_width.le(.25)]
                q=q.merge(expiry,on=['date','secid','expiry'],validate='many_to_one')
                if q.empty:continue
                q=fit_quotes(q,stock,zero,div)
                for cp in ['C','P']:
                    selected=choose(q[q.cp_flag.eq(cp)],anchor,cp)
                    if len(selected):choices[cp]=reduce_choices(pd.concat([choices[cp],selected],ignore_index=True))
            for cp,frame in choices.items():
                if frame.empty:continue
                for sid,g in frame.groupby('secid'):
                    p=DATA/'selection'/cp/str(year)/f'{int(sid)}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                    g.to_parquet(p,index=False);files[str(p)]=digest(p)
    else:
        # A selected 90D contract needs BOTH the new >60D quotes and cached <=60D quotes.
        wanted=set()
        for cp in ['C','P']:
            for yr in [year-1,year]:
                for sid in job['secids']:
                    p=DATA/'selection'/cp/str(yr)/f'{sid}.parquet'
                    if p.exists():wanted.update(pd.read_parquet(p,columns=['optionid']).optionid)
        sources=[raw,ROOT/job['path'],Path('data/curve_trading/puts_raw')/job['path']]
        writers={};paths={}
        try:
            if wanted:
                for source in sources:
                    for batch in pq.ParquetFile(source).iter_batches(batch_size=STREAM_BATCH_ROWS,use_threads=False):
                        frame=batch.to_pandas();frame=frame[frame.optionid.isin(wanted)]
                        if frame.empty:continue
                        q=fit_quotes(normalize_quotes(frame),stock,zero,div);q['delta']=q.bs_delta;q['vega']=q.bs_vega;q=q[HISTORY_COLUMNS]
                        for (cp,sid),g in q.groupby(['cp_flag','secid']):
                            key=(cp,int(sid));p=DATA/'history'/cp/str(year)/f'{int(sid)}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                            table=pa.Table.from_pandas(g,preserve_index=False)
                            if key not in writers:
                                writers[key]=pq.ParquetWriter(p.with_suffix('.partial'),table.schema,compression='zstd');paths[key]=p
                            writers[key].write_table(table)
        finally:
            for w in writers.values():w.close()
        for p in paths.values():p.with_suffix('.partial').replace(p);files[str(p)]=digest(p)
    meta.write_text(json.dumps(dict(stage=stage,index=index,files=files,source_hash=digest(raw)),indent=2)+'\n')
    print(f'{stage} {index+1}/{len(jobs())} {year}: {len(files)} files',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--stage',choices=['select','history'],required=True);a=p.parse_args()
    if a.stage=='history':
        own=jobs()[a.index]
        for j in jobs():
            if j['year'] in [own['year']-1,own['year']] and set(j['secids']).intersection(own['secids']):
                assert (DATA/'metadata/select'/str(j['year'])/(Path(j['path']).stem+'.json')).exists(),'Finish current/prior-year selections for these securities before histories'
    prepare(a.index,a.stage)

if __name__=='__main__':main()
