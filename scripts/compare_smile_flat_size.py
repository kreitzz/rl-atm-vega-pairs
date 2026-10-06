"""Matched current individual sizing versus constant full-vega entry size."""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.joint_smile_data import SharedMarket,registry
from scripts.joint_smile_engine import simulate
from scripts.curve_trading.config import config
from scripts.run_broad_contract_replay import ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.research_opposite_wing_signals import CORPORATE_EVENTS

FROZEN=Path('artifacts/contracts/joint_smile')
OUT=Path('artifacts/contracts/smile_flat_size')
REPORT=Path('reports/smile_flat_size')


def main():
    parser=argparse.ArgumentParser();parser.parse_args()
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources=registry();sources.expected.update(json.loads(Path('reports/joint_smile/output_hashes.json').read_text()))
    code=[Path('scripts/compare_smile_flat_size.py'),Path('scripts/joint_smile_engine.py'),Path('scripts/joint_smile_data.py'),
          Path('scripts/option_flex/contract_engine.py'),Path('scripts/option_flex/contract_risk.py')]
    for p in code:sources.track(p)
    signals=pd.read_parquet(sources.track(FROZEN/'training_rows.parquet'))
    market=SharedMarket();cal=load_calendar(ROOT);cal=cal[(cal>='2018-01-01')&(cal<='2025-08-29')]
    for p in (FROZEN/'market').glob('*_quotes.parquet'):sources.track(p)
    for name in ['stocks','dividends','rates']:sources.track(FROZEN/'market'/f'{name}.parquet')
    cfg=replace(config(),option_commission=.03,capital=3_000_000*345,max_gross_vega=60_000*345)
    protocol=('Same 2,133 nonzero frozen ATM/P25/P75 proposal dates, directions and selected contracts across all345 roster pairs. '
              'Compare existing individual policy levels with level4 on every proposal: base $450 vega per leg, unchanged inherited NAV scaling. '
              'Frozen policy cash decisions remain cash in both arms; this isolates positive entry-size choices and does not measure the full value of learned selection/skips. '
              'Exactly the audited shared-book simulator and original $1m/20k gross-vega per-pair/per-wing risk reservations, $3m/60k pair budget and summed global limits. '
              'Risk and integer-matching rejections can change accepted trade counts; no hindsight requirement to match executed trades. '
              'Initial paired vega mismatch<=10%, daily delta hedge, own-expiry hold/corporate exits. '
              'Actual commission0.03, mid+.1volpoint perleg, stock1bp, financing/dividends/borrow unchanged. '
              'No refitting or parameter search; same inspected 2018-Aug2025 history. '
              'One worker, 50% of one core, one BLAS/Arrow thread, 3GiB RSS stop.')
    write_json(OUT/'inputs.json',dict(protocol=protocol,pairs=345,proposals=len(signals),fixed_entry_vega_base=450,actual_commission=.03))
    books={};rows=[];annual=[];wing=[];decision_counts=[];checks=[]
    for model in ['individual_sizing','fixed_full_vega']:
        if model=='individual_sizing':
            source=FROZEN/'shared_frozen_native_limits'
            nav=pd.read_parquet(sources.track(source/'daily_nav.parquet'))
            trades=pd.read_csv(sources.track(source/'trades.csv'),parse_dates=['date'])
            decisions=pd.read_csv(sources.track(source/'decisions.csv'),parse_dates=['signal_date','entry_date'])
        else:
            print('Replay fixed full-vega entries on the frozen three-policy signals',flush=True)
            nav,trades,decisions,internal=simulate(market,signals,cal,cfg,3_000_000,policy=lambda row,context:4,
                                                  corporate_events=CORPORATE_EVENTS,wing_capital=1_000_000)
            mark_book(nav,market.rates,cfg.capital)
            nav.to_parquet(OUT/'daily_nav.parquet');trades.to_csv(OUT/'trades.csv',index=False);decisions.to_csv(OUT/'decisions.csv',index=False)
        assert nav.index.equals(cal) and len(decisions)==len(signals)
        assert decisions.signal_date.tolist()==sorted(decisions.signal_date.tolist())
        assert model!='fixed_full_vega' or decisions.level.eq(4).all()
        cashflow=(-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(cal,fill_value=0)
        expected=cashflow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        error=float((expected-nav.pnl).abs().max());assert error<1e-5
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
        paid=trades.reason.isin(['entry','announced_corporate_action_exit'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],trades.loc[paid,'quantity'].abs()*.03,rtol=0,atol=1e-10)
        entered=decisions.loc[decisions.status.eq('entered')]
        assert ((entered.vega_a-entered.vega_b).abs()/entered[['vega_a','vega_b']].max(axis=1)).le(.10+1e-12).all()
        value=scores(nav,cfg.capital)
        if model=='individual_sizing':
            frozen_status=json.loads(sources.track(source/'status.json').read_text())
            np.testing.assert_allclose(value['annualized_excess_sharpe'],frozen_status['annualized_excess_sharpe'],rtol=0,atol=1e-12)
        row=dict(model=model,entries=len(entered),active_pairs=int(entered.pair.nunique()),
                 net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),commission=float(trades.commission.sum()),
                 option_cost=float(nav.option_execution_cost.sum()),hedge_cost=float(nav.hedge_cost.sum()),**value)
        rows.append(row);books[model]=nav
        checks.append(dict(model=model,independent_ledger_error=error,all_inventory_closed=True,entry_vega_neutrality_verified=True))
        for year,g in nav.groupby(nav.index.year):
            sd=g.excess_return.std();annual.append(dict(model=model,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None))
        for w,g in decisions.groupby('wing'):
            e=g.loc[g.status.eq('entered')];wing.append(dict(model=model,wing=w,proposals=len(g),entries=len(e),mean_entry_allocation=float(e.level.mean()/4)))
        for status,count in decisions.status.value_counts().items():decision_counts.append(dict(model=model,status=status,proposals=int(count)))
        print(f'{model}: annualized excess Sharpe {value["annualized_excess_sharpe"]:.6f}, {len(entered)} entries',flush=True)
    a=pd.read_csv(FROZEN/'shared_frozen_native_limits/decisions.csv',parse_dates=['signal_date'])
    b=pd.read_csv(OUT/'decisions.csv',parse_dates=['signal_date'])
    assert a[['signal_date','pair','wing','base_level']].equals(b[['signal_date','pair','wing','base_level']])
    result=pd.DataFrame(rows);result.to_csv(REPORT/'comparison.csv',index=False)
    years=pd.DataFrame(annual);years.to_csv(REPORT/'annual.csv',index=False)
    years.pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_comparison.csv')
    pd.DataFrame(wing).to_csv(REPORT/'wings.csv',index=False);pd.DataFrame(decision_counts).to_csv(REPORT/'decisions.csv',index=False)
    pd.DataFrame(checks).to_csv(REPORT/'ledger_audit.csv',index=False)
    for model,nav in books.items():nav.to_csv(REPORT/f'{model}_portfolio.csv')
    for path,value in sources.items():assert digest(path)==value,f'Frozen source changed: {path}'
    write_json(OUT/'sources.json',sources)
    record=dict(status='complete',pairs=345,proposals=len(signals),no_retraining=True,signal_dates_and_directions_preserved=True,
                inherited_cash_signals_preserved=True,actual_commission=.03,frozen_sources_verified=len(sources.verified),
                elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(REPORT/'audit.json',record);(REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+protocol+'\n\n'+json.dumps(record,indent=2)+'\n')
    write_json(REPORT/'output_hashes.json',{str(p):digest(p) for p in [*code,*[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']]})
    print(years.pivot(index='year',columns='model',values='annualized_excess_sharpe').to_string(),flush=True)


if __name__=='__main__':main()
