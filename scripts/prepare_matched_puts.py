"""Fit downloaded put partitions without changing the existing call cache."""
import json
from pathlib import Path
import pandas as pd
from scripts.download_pair_contract_history import file_hash
from scripts.run_contract_pilot import normalize_quotes
from scripts.option_flex.black_scholes import fit_quotes
from scripts.option_flex.contract_engine import valid_quote
from scripts.train_expiry_contracts import settings

ROOT=Path('data/call_put')


def main():
    zero=pd.read_parquet('data/option_contracts/zero_curve.parquet');divs=pd.read_parquet('data/option_contracts/distributions.parquet');cfg=settings()
    jobs=json.loads((ROOT/'requests.json').read_text());records=[]
    for job in jobs:
        source=ROOT/job['path'];meta=source.with_suffix('.json')
        if not meta.exists():continue
        dest=ROOT/'prepared'/source.name;dest.parent.mkdir(exist_ok=True);stamp=dest.with_suffix('.json')
        fingerprint=dict(raw_sha256=file_hash(source),code=file_hash(Path('scripts/option_flex/black_scholes.py')),
            prepare_code=file_hash(Path(__file__)),settings=str(cfg))
        if stamp.exists():
            old=json.loads(stamp.read_text());assert old['fingerprint']==fingerprint;records.append(old);continue
        raw=pd.read_parquet(source);stocks=pd.read_parquet(f"data/option_contracts/underlying/year={job['year']}.parquet")
        stocks['date']=pd.to_datetime(stocks.date);stocks=stocks[stocks.secid.isin(job['secids'])]
        if len(raw):
            q=normalize_quotes(raw);del raw
            q=pd.concat([fit_quotes(q.iloc[i:i+10000],stocks,zero,divs) for i in range(0,len(q),10000)],ignore_index=True)
            for col in ['delta','vega','impl_volatility']:q['vendor_'+col]=q[col]
            q['delta']=q.bs_delta;q['vega']=q.bs_vega;q['impl_volatility']=q.bs_iv
            for sid,g in q.groupby('secid'):
                folder=ROOT/'history'/str(job['year']);folder.mkdir(parents=True,exist_ok=True)
                g.to_parquet(folder/f'{int(sid)}.parquet',index=False)
                path=Path('data/contract_history_shared')/str(job['year'])/f'{int(sid)}_selection.parquet'
                if path.exists():
                    call=pd.read_parquet(path,columns=['date','secid','strike','expiry'])
                    match=g.merge(call,on=['date','secid','strike','expiry'],how='inner',validate='many_to_one')
                    if len(match):
                        match=match[match.apply(lambda r:valid_quote(r,True,cfg),axis=1)]
                        match=match.sort_values(['date','open_interest','optionid'],ascending=[True,False,True]).drop_duplicates('date')
                    match.to_parquet(folder/f'{int(sid)}_selection.parquet',index=False)
            n=len(q)
        else:n=0
        saved=dict(fingerprint=fingerprint,job=job,rows=n);stamp.write_text(json.dumps(saved,indent=2)+'\n');records.append(saved)
        print(f'Prepared {source.name}: {n:,} put marks',flush=True)
    (ROOT/'manifest.json').write_text(json.dumps(dict(completed=len(records),requested=len(jobs),partitions=records),indent=2)+'\n')


if __name__=='__main__':main()
