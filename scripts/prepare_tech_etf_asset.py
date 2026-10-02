"""Prepare SMH close-to-close return states without standardized-IV masking."""
import json
from pathlib import Path
import pandas as pd
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT
from scripts.run_rv_vix_pass import trailing_rv
from scripts.run_contract_walkforward import digest


def main():
    root=Path('data/tech_etf');asset=json.loads((root/'smh_asset.json').read_text())
    raw=pd.read_parquet(root/'smh_returns.parquet');raw['date']=pd.to_datetime(raw.date)
    assert raw.permno.eq(asset['permno']).all() and not raw.date.duplicated().any()
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2012-01-01')&(calendar<='2025-09-30')]
    ret=raw.set_index('date').retx.astype(float).reindex(calendar)
    assert ret.notna().all()
    frame=pd.DataFrame({'ret':ret,'trailing_rv':trailing_rv(ret.to_frame('SMH')).SMH,'earnings':31.})
    path=Path('artifacts/broad_pairs/assets')/f"s{asset['secid']}p{asset['permno']}.parquet"
    if path.exists():pd.testing.assert_frame_equal(pd.read_parquet(path),frame)
    else:frame.to_parquet(path)
    metadata=dict(asset=asset,path=str(path),sha256=digest(path),rows=len(frame),source_sha256=digest(root/'smh_returns.parquet'),
        return_source='CRSP dsf retx through 2024; dsf_v2 dlyretx in 2025',standardized_iv_mask=False)
    (root/'asset_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(metadata)


if __name__=='__main__':main()
