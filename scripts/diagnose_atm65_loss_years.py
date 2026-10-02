"""Saved-ledger year/leg attribution, no fitting or policy changes."""
from pathlib import Path
import json
import time
import resource
import numpy as np
import pandas as pd
from scripts.diagnose_current_contract_losses import attribute, complete_nav, read_csv, COMPONENTS
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar
from scripts.run_atm65_ablation import FROZEN, OUT

REPORT=Path('reports/atm65_loss_years')

def option_leg_marks(trades, sid, calendar):
    own=trades[trades.secid.eq(sid)]
    changes=own.pivot_table(index='date',columns='optionid',values='quantity',aggfunc='sum',fill_value=0)
    holdings=changes.reindex(calendar,fill_value=0).cumsum()
    rows,cols=np.where(holdings.to_numpy()!=0)
    quotes=[];sources=[]
    for year in sorted(set(calendar[rows].year)):
        p=Path('data/curve_trading/history/C')/str(year)/f'{sid}.parquet'
        quotes.append(pd.read_parquet(p,filters=[('optionid','in',list(holdings.columns.astype(int)))],
                        columns=['date','optionid','bid','ask']))
        sources.append(p)
    mark=np.zeros(len(calendar))
    if len(rows):
        q=pd.concat(quotes).set_index(['date','optionid'])
        wanted=pd.MultiIndex.from_arrays([calendar[rows],holdings.columns[cols]])
        midpoint=q.reindex(wanted)[['bid','ask']].mean(axis=1,skipna=False).to_numpy()
        assert np.isfinite(midpoint).all(), 'Missing held midpoint'
        np.add.at(mark,rows,holdings.to_numpy()[rows,cols]*100*midpoint)
    return pd.Series(mark,index=calendar),sources

def main():
    REPORT.mkdir(exist_ok=True);started=time.monotonic()
    inputs=json.loads((OUT/'inputs.json').read_text());defs=inputs['definitions'];cfg=inputs['actual']
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2018-01-01')&(calendar<='2025-08-29')]
    market=Market(calendar);stocks=market.stocks.set_index(['secid','date']).close.sort_index()
    daily=pd.DataFrame(0.,index=calendar,columns=COMPONENTS)
    pairs=[];legs=[];worst=[];entries=[];rows=[];source_files=[];errors=[];leg_error=[]
    for d in defs:
        folder=FROZEN/d['pair']/'full_003';status=json.loads((folder/'status.json').read_text())
        assert status['status']=='complete'
        if status.get('cash_only'): continue
        navpath=Path(status.get('nav_source',folder/'daily_nav.parquet'))
        nav=complete_nav(pd.read_parquet(navpath),cfg['capital'])
        trades=read_csv(folder/'trades.csv',['date']);dec=read_csv(folder/'decisions.csv',['signal_date','entry_date'])
        attrib,_,error=attribute(nav,trades,d['secids'],stocks,market.divs,cfg)
        daily+=attrib;errors.append(error);source_files.extend([folder/'status.json',navpath])
        if (folder/'trades.csv').exists(): source_files.append(folder/'trades.csv')
        entered=dec[dec.status.eq('entered')] if len(dec) else pd.DataFrame()
        for year,g in attrib.groupby(attrib.index.year):
            e=entered[entered.entry_date.dt.year.eq(year)] if len(entered) else entered
            totals=g.sum().to_dict()
            pairs.append(dict(pair=d['pair'],label=d['label'],year=year,entries=len(e),
                              gross_hedged_pnl=totals['option_mid_pnl']+totals['stock_hedge_gross_pnl']+totals['dividends'],**totals))
        for year in [2018,2025]:
            g=attrib[attrib.index.year==year].nsmallest(5,'net_excess_pnl').copy()
            g['date']=g.index;g['pair']=d['pair'];g['label']=d['label']
            worst.extend(g.to_dict('records'))
        if not len(trades): continue
        for e in entered.itertuples():
            entries.append(dict(pair=d['pair'],label=d['label'],entry_date=str(e.entry_date.date()),
                                signal_date=str(e.signal_date.date()),year=e.entry_date.year,action=e.action,
                                mean_entry_vega=(e.vega_a+e.vega_b)/2))
        # Reconstruct each leg's daily option marks from the exact cached held quotes.
        combined_marks=pd.Series(0.,index=calendar)
        for j,sid in enumerate(d['secids']):
            mark,sources=option_leg_marks(trades,sid,calendar);source_files.extend(sources)
            combined_marks+=mark
            own=trades[trades.secid.eq(sid)]
            option_cash=(-own.quantity*100*own.price).groupby(own.date).sum().reindex(calendar,fill_value=0)
            optpnl=option_cash+mark.diff().fillna(mark.iloc[0])
            h=nav[f'hedge_{sid}'];previous=h.shift(1).fillna(0);factor=nav[f'split_factor_{sid}']
            spot=stocks.xs(sid,level='secid').reindex(calendar)
            hedge=previous*(spot*factor-spot.shift(1)).fillna(0)
            hedge_cost=((h-previous*factor).abs()*spot).fillna(0)*cfg['stock_cost_bps']/10000
            elapsed=calendar.to_series().diff().dt.days.fillna(0)/365
            borrow=((-previous).clip(lower=0)*spot.shift(1)).fillna(0)*cfg['stock_borrow_rate']*elapsed
            dividend=pd.Series(0.,index=calendar)
            for e in market.divs[market.divs.secid.eq(sid)].itertuples():
                k=calendar.searchsorted(e.ex_date)
                if 0<k<len(calendar): dividend.iloc[k]+=previous.iloc[k]*factor.iloc[k]*e.amount
            commission=own.groupby('date').commission.sum().reindex(calendar,fill_value=0)
            penalty=own.groupby('date').execution_penalty.sum().reindex(calendar,fill_value=0)
            leg=pd.DataFrame(dict(option_mid_pnl=optpnl,stock_hedge_gross_pnl=hedge,dividends=dividend,
                option_commission=commission,option_penalty=penalty,stock_execution=hedge_cost,stock_borrow=borrow))
            leg['gross_hedged_pnl']=optpnl+hedge+dividend
            leg['net_before_shared_financing']=leg.gross_hedged_pnl-commission-penalty-hedge_cost-borrow
            entries_own=own[own.reason.eq('entry')]
            for year,g in leg.groupby(leg.index.year):
                e=entries_own[entries_own.date.dt.year.eq(year)]
                legs.append(dict(pair=d['pair'],label=d['label'],secid=sid,name=d['label'].split('/')[j],year=year,
                    long_entries=int(e.quantity.gt(0).sum()),short_entries=int(e.quantity.lt(0).sum()),**g.sum().to_dict()))
            rows.append(leg[['net_before_shared_financing']].rename(columns={'net_before_shared_financing':str(sid)}))
        np.testing.assert_allclose(combined_marks,nav.option_value,atol=1e-7,rtol=0)
        reconstructed=rows[-2].sum(axis=1)+rows[-1].sum(axis=1)+attrib.financing_vs_cash
        np.testing.assert_allclose(reconstructed,attrib.net_excess_pnl,atol=1e-7,rtol=0)
        leg_error.append(float(abs(reconstructed-attrib.net_excess_pnl).max()))
        if status['entries']: print(f'{d["label"]}: {status["entries"]} cohorts attributed',flush=True)
    # Independent check against the portfolio already reported to the user.
    portfolio=pd.read_csv('reports/atm65_ablation/rl_sized_full_portfolio.csv',index_col=0,parse_dates=[0])
    expected=portfolio.nav.diff().fillna(0)-portfolio.cash_benchmark.diff().fillna(0)
    np.testing.assert_allclose(daily.net_excess_pnl,expected,atol=1e-6,rtol=0)
    daily.to_csv(REPORT/'daily.csv')
    annual=daily.groupby(daily.index.year).sum();annual['gross_hedged_pnl']=annual.option_mid_pnl+annual.stock_hedge_gross_pnl+annual.dividends
    annual.to_csv(REPORT/'annual_components.csv')
    pairs=pd.DataFrame(pairs);pairs.to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(legs).to_csv(REPORT/'legs.csv',index=False)
    pd.DataFrame(entries).to_csv(REPORT/'entries.csv',index=False)
    pd.DataFrame(worst).sort_values('net_excess_pnl').groupby(pd.to_datetime(pd.DataFrame(worst).date).dt.year).head(10).to_csv(REPORT/'worst_pair_days.csv',index=False)
    lines=['# Why the $0.65 RL checkpoint loses in 2018 and 2025','',
        'Saved original RL-sized policy, the matched 345 pairs, actual $0.03 commission. No fitting, pricing, sizing or policy changes. Calendar-year P&L includes positions carried across year boundaries; shared financing stays separate from leg attribution. 2025 ends August 29.','',
        '| Year | Gross hedged P&L | Option commission | Option penalty | Stock execution | Stock borrow | Financing vs cash | Net excess P&L | New entries |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for year in [2018,2025]:
        g=annual.loc[year];e=sum(x['year']==year for x in entries)
        vals=[g[c] for c in ['gross_hedged_pnl','option_commission','option_penalty','stock_execution','stock_borrow','financing_vs_cash','net_excess_pnl']]
        lines.append(f'| {year} | '+' | '.join(f'${v:,.0f}' for v in vals)+f' | {e} |')
        own=pairs[pairs.year.eq(year)].sort_values('net_excess_pnl')
        print(f'\n{year} components:\n{g.to_string()}\nWorst pairs:\n{own[["label","entries","gross_hedged_pnl","net_excess_pnl"]].head(8).to_string(index=False)}',flush=True)
    lines+=['','Option and hedge P&L must be read together; gross excludes execution, stock borrow and shared financing. Net excess dollars are the annual change in portfolio NAV less the annual change in the cash benchmark. Cash earns interest even when the trading sleeve loses. Initial vega matching and daily delta hedging do not eliminate relative gamma/jump risk or later vega drift. Event causes require contract/underlying inspection; this ledger establishes where losses occurred.','',
        '[Annual components](atm65_loss_years/annual_components.csv) · [Pairs](atm65_loss_years/pairs.csv) · [Legs](atm65_loss_years/legs.csv) · [Worst days](atm65_loss_years/worst_pair_days.csv) · [Entries](atm65_loss_years/entries.csv)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    # Large quote partitions are referenced through the existing history manifest;
    # hash the saved books and code directly rather than rereading raw partitions.
    files={p for p in source_files if 'data/curve_trading/history' not in str(p)}
    files.update([Path(__file__),Path('scripts/diagnose_current_contract_losses.py'),Path('data/curve_trading/C_history_manifest.json'),OUT/'inputs.json'])
    manifest=dict(pairs=len(defs),original_policy_unchanged=True,largest_daily_ledger_error=max(errors),
        largest_leg_ledger_error=max(leg_error),largest_portfolio_error=float(abs(daily.net_excess_pnl-expected).max()),
        source_sha256={str(p):digest(p) for p in sorted(files)},elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (REPORT/'audit.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
