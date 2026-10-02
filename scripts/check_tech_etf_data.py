"""Resolve tech ETF security identifiers through the existing WRDS connection."""
import getpass
import os
import json
import argparse
import shutil
import pandas as pd
from pathlib import Path
from scripts.download_pair_contract_history import connection,download,quote_jobs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download-smh',action='store_true');parser.add_argument('--refresh-returns',action='store_true');args=parser.parse_args()
    output=Path('data/tech_etf');output.mkdir(parents=True,exist_ok=True)
    db=connection(os.getenv('WRDS_USERNAME',getpass.getuser()))
    if args.refresh_returns:
        permno=json.loads((output/'smh_asset.json').read_text())['permno']
        with db.cursor() as c:
            c.execute("select permno,dlycaldt as date,dlyretx as retx from crsp_a_stock.dsf_v2 where permno=%s and dlycaldt between '2025-01-01' and '2025-09-30' order by dlycaldt",(permno,))
            recent=pd.DataFrame(c.fetchall(),columns=[d.name for d in c.description])
        old=pd.read_parquet(output/'smh_returns.parquet')
        data=pd.concat([old,recent],ignore_index=True).drop_duplicates(['permno','date'],keep='last').sort_values('date')
        assert pd.to_datetime(data.date).max()>=pd.Timestamp('2025-08-29')
        data.to_parquet(output/'smh_returns.parquet',index=False)
        print(f'SMH CRSP returns: {len(data)} rows, {data.date.min()} to {data.date.max()}');return
    with db.cursor() as c:
        c.execute("select secid, effect_date, ticker, issuer from optionm.secnmd where ticker = ANY(%s) order by secid,effect_date",(['QQQ','SMH','SOXX','XLK'],))
        rows=[dict(zip([d.name for d in c.description],r)) for r in c.fetchall()]
    (output/'identifiers.json').write_text(json.dumps(rows,default=str,indent=2)+'\n')
    print(json.dumps(rows,default=str,indent=2))
    if not args.download_smh:return
    with db.cursor() as c:
        c.execute("select secid,count(*) from optionm.opprcd2024 where secid = ANY(%s) and date='2024-06-03' group by secid",([151482,151483,151720],))
        traded=c.fetchall();assert len(traded)==1 and int(traded[0][0])==151720,traded
        c.execute("select distinct permno,namedt,nameenddt,ticker,comnam from crsp.stocknames where ticker='SMH' order by namedt")
        names=pd.DataFrame(c.fetchall(),columns=[d.name for d in c.description]);names.to_csv(output/'crsp_names.csv',index=False)
        print(names.to_string(index=False),flush=True)
        current=names[pd.to_datetime(names.nameenddt)==pd.to_datetime(names.nameenddt).max()]
        assert current.permno.nunique()==1
        permno=int(current.permno.iloc[0])
        c.execute("select permno,date,retx from crsp.dsf where permno=%s and date between '2012-01-01' and '2025-09-30' order by date",(permno,))
        pd.DataFrame(c.fetchall(),columns=[d.name for d in c.description]).to_parquet(output/'smh_returns.parquet',index=False)
    db.commit()
    (output/'smh_asset.json').write_text(json.dumps(dict(ticker='SMH',secid=151720,permno=permno),indent=2)+'\n')
    root=output/'contracts';root.mkdir(exist_ok=True)
    assets=pd.DataFrame([dict(secid=151720,start='2012-01-01')])
    jobs=[j for j in quote_jobs(assets,1) if j.get('year',2012)>=2012 and j['kind']!='rates']
    (root/'requests.json').write_text(json.dumps(jobs,indent=2)+'\n')
    shutil.copyfile('data/option_contracts/zero_curve.parquet',root/'zero_curve.parquet')
    for i,job in enumerate(jobs,1):
        result=download(job,root,os.getenv('WRDS_USERNAME',getpass.getuser()))
        print(f'{i}/{len(jobs)} {job["path"]}: {result["rows"]:,} rows ({result["status"]})',flush=True)


if __name__=='__main__':main()
