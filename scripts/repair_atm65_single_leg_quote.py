"""Bounded vendor query for one missing mark; preserve fitted actions and caches."""
import getpass
import json
import os
from pathlib import Path
from dataclasses import replace
import pandas as pd
from scripts.download_pair_contract_history import connection,QUOTE_SCHEMA
from scripts.run_atm65_single_leg import OUT,REPORT,save_result,summary,employee_roster
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.single_leg_policy import Replay
from scripts.run_contract_pilot import normalize_quotes
from scripts.option_flex.black_scholes import fit_quotes
from scripts.prepare_curve_contracts import HISTORY_COLUMNS
from scripts.replay_frozen_atm_commission import load_pair
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.curve_trading.config import config
from scripts.run_contract_walkforward import digest


def main():
    folder=OUT/'quote_recovery';folder.mkdir(exist_ok=True)
    raw_path=folder/'132591447_wrds.parquet'
    query='SELECT '+','.join(QUOTE_SCHEMA.names)+' FROM optionm.opprcd2020 WHERE secid=%s AND optionid=%s AND date BETWEEN %s AND %s ORDER BY date LIMIT 100'
    parameters=[112463,132591447,'2020-02-01','2020-04-17']
    if not raw_path.exists():
        db=connection(os.getenv('WRDS_USERNAME',getpass.getuser()))
        with db.cursor() as cursor:
            cursor.execute("SET statement_timeout='30s'");cursor.execute(query,parameters)
            raw=pd.DataFrame(cursor.fetchall(),columns=[c.name for c in cursor.description])
        db.close();raw.to_parquet(raw_path,index=False)
    else:raw=pd.read_parquet(raw_path)
    missing_date=pd.Timestamp('2020-03-13')
    present=raw.date.astype(str).eq(str(missing_date.date())).any()
    record=dict(query=query,parameters=parameters,rows=len(raw),raw_sha256=digest(raw_path),
        missing_date_recovered=bool(present),fitted_actions_unchanged=True,original_market_caches_unchanged=True)
    if not present:
        write_json(folder/'manifest.json',record)
        print('Vendor has no quote on 2020-03-13. No held price was imputed.',flush=True);return
    manifest=json.loads((OUT/'inputs.json').read_text());defs=manifest['definitions']
    d=next(d for d in defs if d['pair']=='112463__161182')
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=Market(cal);q,s,selections,divs=load_pair(market,d['secids'])
    known=pd.MultiIndex.from_frame(q[['date','optionid']])
    normalized=normalize_quotes(raw)
    patch=normalized.loc[~pd.MultiIndex.from_frame(normalized[['date','optionid']]).isin(known)].copy()
    if patch.empty:raise ValueError('No missing marks to recover')
    fitted=fit_quotes(patch,s,pd.read_parquet(ROOT/'zero_curve.parquet'),pd.read_parquet(ROOT/'distributions.parquet'))
    fitted['delta']=fitted.bs_delta;fitted['vega']=fitted.bs_vega
    fitted=fitted[HISTORY_COLUMNS];fitted.to_parquet(folder/'fitted_patch.parquet',index=False)
    q=pd.concat([q,fitted],ignore_index=True);assert not q.duplicated(['date','optionid']).any()
    cfg=replace(config(),option_commission=.03)
    path=OUT/d['pair']/'rl_single_historical/actions.csv';action_sha=digest(path)
    actions=read_actions(path);replay=Replay(q,s,selections,d['secids'],divs,cfg)
    nav,trades,decisions=replay.run(actions,cal[cal>='2018-01-01']);mark_book(nav,market.rates,cfg.capital)
    dest=OUT/d['pair']/'rl_single_historical/full'
    write_json(folder/'original_incomplete_status.json',json.loads((dest/'status.json').read_text()))
    result=save_result(dest,nav,trades,decisions,cfg,s,d)
    result.update(training_hurdle=.65,execution_commission=.03,recovered_mark_source=str(folder/'fitted_patch.parquet'))
    write_json(dest/'status.json',result);assert digest(path)==action_sha
    record.update(fitted_patch_sha256=digest(folder/'fitted_patch.parquet'),actions_sha256=action_sha,
        replay_completed=True,recovered_marks=[dict(date=str(r.date.date()),optionid=int(r.optionid)) for r in fitted.itertuples()],
        recovery_code_sha256=digest(Path(__file__)),training_note='Models and their labels were frozen before the supplementary mark download; only the affected full-period realized replay was completed.')
    write_json(folder/'manifest.json',record)
    summary(defs,pd.read_csv(REPORT/'coverage.csv'),market,cfg,employee_roster())
    print('Recovered vendor marks and replayed the affected path without refitting.',flush=True)


if __name__=='__main__':main()
