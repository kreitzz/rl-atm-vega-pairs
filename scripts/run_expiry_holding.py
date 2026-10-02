"""Frozen-policy lifecycle diagnostic, not a retrained maturity strategy."""
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.contract_engine import Settings, PreparedMarket, simulate, UnresolvedPosition
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit, scores
from scripts.run_contract_walkforward import digest

OUT=Path('artifacts/contracts/expiry_holding')
REPORT=Path('reports/expiry_holding')
POLICIES={
    'earlier_filtered_rl':('entry_gates','rl_sized_memory_both'),
    'surface_filtered_rl':('surface_features','rl_sized_recent_memory'),
    'surface_filtered_nn':('surface_features','nn_fixed_recent_memory'),
}


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    cfg=Settings(**json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())['settings'])
    cfg.hold_to_expiry=True
    cal=load_calendar(ROOT);cal=cal[(cal>='2018-01-01')&(cal<='2025-08-29')]
    market=Market(cal)
    roster=set(pd.read_csv('reports/current_contract_losses/pairs.csv').pair)
    definitions=[d for d in json.loads(Path('reports/aligned_contract_learning/universe.json').read_text()) if d['pair'] in roster and d['group']=='broad']
    status=[];books={};source_hashes={}
    for i,d in enumerate(definitions):
        prepared=None
        for name,(root,sub) in POLICIES.items():
            src=Path('artifacts/contracts')/root/d['pair']/sub
            baseline=pd.read_parquet(src/'daily_nav.parquet')
            dest=OUT/d['pair']/name;dest.mkdir(parents=True,exist_ok=True)
            old_dec=src/'decisions.csv'
            # Saved filtered actions exist for the surface experiments. Earlier
            # filtered signals are reconstructed from every recorded decision.
            if (src/'actions.csv').exists():
                path=src/'actions.csv';a=pd.read_csv(path,index_col=0,parse_dates=[0]).iloc[:,0]
            elif old_dec.exists() and old_dec.stat().st_size>2:
                path=old_dec;dec=pd.read_csv(path,parse_dates=['signal_date']);a=dec.set_index('signal_date').action
            else:
                path=None;a=pd.Series(dtype=float,index=pd.DatetimeIndex([]))
            source_hashes[str(src)]=dict(nav=digest(src/'daily_nav.parquet'),actions=digest(path) if path else None)
            row=dict(pair=d['pair'],label=d['label'],policy=name)
            try:
                if a.ne(0).any():
                    if prepared is None:
                        q,s,selection,divs=market.pair(d['secids']);prepared=PreparedMarket(q,s,selection)
                    nav,t,dec=simulate(q,s,a,cal,d['secids'],cfg,divs,selection,prepared)
                    mark_book(nav,market.rates,cfg.capital)
                    checks=audit(nav,t,s,cfg,d['secids'])
                    assert t.empty or set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                    t.to_csv(dest/'trades.csv',index=False);dec.to_csv(dest/'decisions.csv',index=False)
                    row.update(entries=int(dec.status.eq('entered').sum()),fallbacks=int(nav.greek_fallbacks.sum()),
                        risk_breach_days=int(nav.post_trade_risk_breach.sum()),**checks)
                else:
                    nav=baseline.copy();row.update(entries=0,fallbacks=0,risk_breach_days=0,ledger_error=0.)
                nav.to_parquet(dest/'daily_nav.parquet')
                row['status']='complete'
                books[(d['pair'],name)]=(baseline[['nav','cash_benchmark']],nav[['nav','cash_benchmark']])
            except (UnresolvedPosition,ValueError) as exc:
                row.update(status='incomplete',error=str(exc))
            status.append(row)
        if i%10==0:print(f'{i+1}/{len(definitions)} pairs; incomplete={sum(r["status"]!="complete" for r in status)}',flush=True)
    frame=pd.DataFrame(status);frame.to_csv(REPORT/'status.csv',index=False)
    failed=set(frame.loc[frame.status.ne('complete'),'pair']);members=[d['pair'] for d in definitions if d['pair'] not in failed]
    rows=[];annual=[]
    for name in POLICIES:
        for ix,variant in enumerate(['previous_exits','hold_to_expiry']):
            book=sum((books[(pair,name)][ix] for pair in members))
            book['return_']=book.nav.pct_change().fillna(book.nav.iloc[0]/(cfg.capital*len(members))-1)
            mark_book(book,market.rates,cfg.capital*len(members))
            book.to_csv(REPORT/f'{name}_{variant}.csv')
            rows.append(dict(policy=name,variant=variant,pairs=len(members),**scores(book,cfg.capital*len(members))))
            for year,g in book.groupby(book.index.year):
                ret=g.return_;ex=g.excess_return
                annual.append(dict(policy=name,variant=variant,year=year,return_percent=100*((1+ret).prod()-1),
                    annualized_excess_sharpe=np.sqrt(252)*ex.mean()/ex.std() if ex.std()>1e-12 else None))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    manifest=dict(settings=asdict(cfg),pairs_attempted=len(definitions),pairs_common=len(members),unresolved_pairs=sorted(failed),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sources=source_hashes,code={p:digest(p) for p in [__file__,'scripts/contract_engine.py','scripts/contract_risk.py']})
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Hold-to-expiry diagnostic','',
        'Frozen historical signals, sizing actions and profitability-memory gates. No model or reward labels have been retrained for expiry holding. Changed lifecycles affect available capital and accepted entries; this is not a matched-inventory comparison.','',
        'Each leg settles separately at intrinsic value on the last supplied trading session on or before its expiry. No option exit spread, commission or modeled exit penalty. Daily stock hedging and its transaction costs continue, including final hedge liquidation. Assumes cash settlement and no early exercise/assignment for these American calls. Different expiries can leave one leg outstanding.','',
        'Interim NAV still uses observed option midpoints. Daily BS delta uses current fitted IV; failed fits use the last valid IV with current spot, remaining time, rate and declared-dividend PV. Missing or invalid quotes and unsupported adjustments remain explicit incomplete histories. Risk limits apply to entry; existing risk breaches are reported rather than triggering sales. Insolvency still fails the simulation.','',
        f'{len(definitions)} pairs attempted; {len(members)} common complete pairs compared, including cash-only sleeves. {len(failed)} pairs excluded from both sides because at least one replay is unresolved; this ex-post common-roster selection can bias results.','',
        '| Policy | Lifecycle | Annualized excess Sharpe | Cumulative excess return |','|---|---|---:|---:|']
    for r in result.itertuples():lines.append(f'| {r.policy} | {r.variant} | {r.annualized_excess_sharpe:.3f} | {r.excess_return:.3%} |')
    lines+=['','2018–August 2025. Sharpe uses daily excess returns and √252. Returns use $1m starting capital per pair. All periods have previously been inspected.','',
        '[Annual results](expiry_holding/annual.csv) · [Status and fallback counts](expiry_holding/status.csv) · [Comparison](expiry_holding/comparison.csv)','']
    Path('reports/expiry_holding.md').write_text('\n'.join(lines))
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
