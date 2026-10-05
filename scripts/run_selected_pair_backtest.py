"""Replay frozen pair policies on observed quotes for selected ATM contracts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd
from scripts.audit_contract_history import iter_contract_batches
from scripts.black_scholes import fit_quotes
from scripts.contract_engine import Settings, simulate, UnresolvedPosition
from scripts.run_contract_pilot import normalize_quotes
from scripts.select_atm_contracts import load_calendar
from scripts.download_pair_contract_history import file_hash

PAIRS = {p['pair']: [n['secid'] for n in p['names'].values()]
         for p in json.loads(Path('config/universe_gap_pairs.json').read_text()) if p['original_pair']}
MODELS = {'rl_sized':'rl_both_sized_action', 'nn_fixed':'nn_both_fixed_action'}


def selected_year(root, year, secids):
    paths=sorted((root/str(year)).glob('*.parquet'))
    frames=[pd.read_parquet(p,filters=[('secid','in',secids)]) for p in paths]
    frames=[f for f in frames if len(f)]
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def prepare(root, selected_root, output, years):
    ids=[s for pair in PAIRS.values() for s in pair]
    zero=pd.read_parquet(root/'zero_curve.parquet')
    divs=pd.read_parquet(root/'distributions.parquet')
    for year in years:
        selected=selected_year(selected_root,year,ids)
        if selected.empty: raise ValueError(f'No selected contracts in {year}')
        stocks=pd.read_parquet(root/'underlying'/f'year={year}.parquet',filters=[('secid','in',ids)])
        stocks['date']=pd.to_datetime(stocks.date)
        wanted=set(selected.optionid)
        if (selected_root/str(year-1)).exists():
            prior=selected_year(selected_root,year-1,ids)
            if len(prior): wanted.update(prior.optionid)
        chunks=[];count=0
        for raw in iter_contract_batches(root,ids,f'{year}-01-01',f'{year}-12-31'):
            raw=raw[raw.optionid.isin(wanted)]
            if len(raw):
                count+=len(raw)
                if count>500000:raise MemoryError('Year exceeds bounded cache size')
                chunks.append(raw)
        raw=pd.concat(chunks,ignore_index=True);del chunks
        if len(raw)>500000:raise MemoryError('Year exceeds bounded cache size')
        q=normalize_quotes(raw);del raw
        fits=[]
        for offset in range(0,len(q),5000):
            fits.append(fit_quotes(q.iloc[offset:offset+5000],stocks,zero,divs))
        q=pd.concat(fits,ignore_index=True);del fits
        for name in ['delta','vega','impl_volatility']:q['vendor_'+name]=q[name]
        q['delta']=q.bs_delta;q['vega']=q.bs_vega;q['impl_volatility']=q.bs_iv
        for pair,secids in PAIRS.items():
            folder=output/'cache'/pair;folder.mkdir(parents=True,exist_ok=True)
            q[q.secid.isin(secids)].to_parquet(folder/f'{year}.parquet',index=False)
            selected[selected.secid.isin(secids)].to_parquet(folder/f'{year}_selection.parquet',index=False)
        print(f'Prepared {year}: {len(q):,} held-contract marks',flush=True)


def scores(nav, capital):
    excess=nav.excess_return.iloc[1:]
    return dict(total_return=float(nav.nav.iloc[-1]/capital-1),
                excess_return=float((nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])/capital),
                annualized_excess_sharpe=float(np.sqrt(252)*excess.mean()/excess.std()) if excess.std()>1e-12 else None,
                max_drawdown=float((nav.nav/nav.nav.cummax().clip(lower=capital)-1).min()))


def audit(nav,trades,stocks,cfg,secids):
    spots=stocks.pivot(index='date',columns='secid',values='close').reindex(nav.index)
    holdings=pd.DataFrame({s:nav[f'hedge_{s}'] for s in secids})
    factors=pd.DataFrame({s:nav.get(f'split_factor_{s}',pd.Series(1.,index=nav.index)) for s in secids})
    changes=spots*factors-spots.shift(1)
    prior=holdings.shift(1).fillna(0)
    if (changes.isna()&prior.ne(0)).any().any():raise AssertionError('Missing held stock price in audit')
    hedge_pnl=(prior*changes.fillna(0)).sum(axis=1)-nav.hedge_cost
    option_cash=(pd.Series(0.,index=nav.index) if trades.empty else
                 (-trades.quantity*100*trades.price-trades.commission-trades.get('execution_penalty',0.)).groupby(trades.date).sum().reindex(nav.index,fill_value=0))
    option_pnl=option_cash+nav.option_value.diff().fillna(nav.option_value)
    expected=option_pnl+hedge_pnl+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
    np.testing.assert_allclose(expected,nav.pnl,atol=1e-7,rtol=0)
    np.testing.assert_allclose(nav.nav,nav.cash+nav.stock_value+nav.option_value,atol=1e-7,rtol=0)
    assert trades.empty or trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
    return dict(option_pnl=float(option_pnl.sum()),hedge_pnl=float(hedge_pnl.sum()),ledger_error=float(abs(expected-nav.pnl).max()))


def run(root, output, years, repair, cache_root=None, report=None, selected_root=Path('data/atm_30d'), spread_fraction=1., risk_controls=False):
    cache_root=output/'cache' if cache_root is None else cache_root
    report=Path('reports/selected_pair_backtest') if report is None else report
    calendar=load_calendar(root)
    zero=pd.read_parquet(root/'zero_curve.parquet');zero['date']=pd.to_datetime(zero.date)
    zero['distance']=abs(zero.days-30)
    zero=zero.sort_values(['date','distance','days']).drop_duplicates('date').sort_values('date')
    zero['rate']=zero.rate/100
    divs=pd.read_parquet(root/'distributions.parquet')
    for col in ['ex_date','declare_date']:divs[col]=pd.to_datetime(divs[col])
    results=[]; navs={}; statuses=[]
    cfg=Settings(repair_crossed_held_quotes=repair,option_spread_fraction=spread_fraction,risk_controls=risk_controls)
    # Continuous history, plus separate 2025 pilot so a historical data failure
    # does not conceal whether the new implementation works in the pilot period.
    windows=[(str(years[0]) if len(years)==1 else 'history',pd.Timestamp(f'{min(years)}-01-01'),pd.Timestamp(f'{max(years)}-12-31'))]
    if min(years)<2025<=max(years):windows.append(('2025',pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31')))
    for pair,secids in PAIRS.items():
        source=Path('artifacts/nn_rl_benchmark')/f'{pair}.csv'
        signals=pd.read_csv(source,index_col=0,parse_dates=[0])
        # Signal files come from the same existing NN/RL experiment and orientation.
        q=pd.concat([pd.read_parquet(cache_root/pair/f'{y}.parquet') for y in years],ignore_index=True)
        selection=pd.concat([pd.read_parquet(cache_root/pair/f'{y}_selection.parquet') for y in years],ignore_index=True)
        stocks=pd.concat([pd.read_parquet(root/'underlying'/f'year={y}.parquet',filters=[('secid','in',secids)]) for y in years],ignore_index=True)
        stocks['date']=pd.to_datetime(stocks.date)
        stocks=pd.merge_asof(stocks.sort_values('date'),zero[['date','rate']],on='date',direction='backward',tolerance=pd.Timedelta(days=7))
        ordinary=divs[divs.secid.isin(secids)&divs.cancel_flag.eq('0')&divs.distr_type.eq('1')&divs.currency.eq('USD')&divs.adj_factor.eq(1)&divs.approx_flag.eq('0')&divs.liquid_flag.eq('0')].copy()
        for window,start,end in windows:
            cal=calendar[(calendar>=start)&(calendar<=end)]
            # Finish the 30-day holding period inside available data; no hidden truncation.
            stop=cal[-1]-pd.Timedelta(days=35)
            sig=signals[(signals.index>=cal[0])&(signals.index<=stop)]
            for model,col in {**MODELS,'higher_iv':'rl_both_sized_action'}.items():
                actions=sig[col].copy()
                if model=='higher_iv':
                    iv=selection.pivot(index='date',columns='secid',values='bs_iv').reindex(actions.index)
                    actions=actions.abs()*np.sign(iv[secids[1]]-iv[secids[0]])
                    actions=actions.fillna(0.)
                # Missing saved predictions mean no new signal, never removal of a NAV date.
                try:
                    nav,trades,decisions=simulate(q[q.date.between(cal[0],cal[-1])],stocks[stocks.date.isin(cal)],actions,cal,secids,cfg,ordinary,selection)
                    rates=stocks.groupby('date').rate.mean().reindex(nav.index).shift(1).fillna(0)
                    cash_return=rates*nav.index.to_series().diff().dt.days.fillna(0)/365
                    nav['cash_benchmark']=cfg.capital*(1+cash_return).cumprod()
                    nav['excess_return']=nav.return_-cash_return
                    check=audit(nav,trades,stocks,cfg,secids)
                    dest=output/window/pair/model;dest.mkdir(parents=True,exist_ok=True)
                    nav.to_csv(dest/'daily_nav.csv');trades.to_csv(dest/'trades.csv',index=False);decisions.to_csv(dest/'decisions.csv',index=False)
                    row=dict(window=window,pair=pair,model=model,status='complete',start=str(cal[0].date()),end=str(cal[-1].date()),
                             entries=int(decisions.status.eq('entered').sum()),signals=len(actions),
                             absent_signal_sessions=int(((cal>=cal[0])&(cal<=stop)).sum()-len(actions)),
                             repaired_quotes=int(nav.repaired_held_quotes.sum()),**scores(nav,cfg.capital),**check)
                    results.append(row);navs[(window,pair,model)]=nav
                    statuses.append(row)
                except UnresolvedPosition as exc:
                    status='capital_exhausted' if 'capital exhausted' in str(exc) else 'incomplete'
                    statuses.append(dict(window=window,pair=pair,model=model,status=status,error=str(exc),**getattr(exc,'diagnostics',{})))
                    if hasattr(exc,'partial_nav'):
                        dest=output/window/pair/model;dest.mkdir(parents=True,exist_ok=True)
                        exc.partial_nav.to_csv(dest/'partial_nav.csv')
                        exc.trades.to_csv(dest/'partial_trades.csv',index=False)
                print(json.dumps(statuses[-1]),flush=True)
                pd.DataFrame(statuses).to_csv(output/'status.csv',index=False)
    annual=[];portfolios=[]
    for (window,pair,model),nav in navs.items():
        for year,g in nav.groupby(nav.index.year):
            prior=cfg.capital if g.index[0]==nav.index[0] else nav.loc[:g.index[0]].nav.iloc[-2]
            excess=g.excess_return
            annual.append(dict(window=window,pair=pair,model=model,year=int(year),total_return=float(g.nav.iloc[-1]/prior-1),
                               annualized_excess_sharpe=float(np.sqrt(252)*excess.mean()/excess.std()) if excess.std()>0 else None))
    for window,_,_ in windows:
        for model in [*MODELS,'higher_iv']:
            members=[navs.get((window,pair,model)) for pair in PAIRS]
            if any(n is None for n in members):continue
            index=members[0].index
            assert all(n.index.equals(index) for n in members)
            book=pd.DataFrame(index=index)
            book['nav']=sum(n.nav for n in members)
            book['cash_benchmark']=sum(n.cash_benchmark for n in members)
            book['return_']=book.nav.pct_change().fillna(0)
            book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            portfolios.append(dict(window=window,model=model,**scores(book,4*cfg.capital)))
            book.to_csv(output/f'{window}_{model}_portfolio.csv')
    report.mkdir(parents=True,exist_ok=True)
    for name,rows in [('status',statuses),('annual',annual),('portfolio',portfolios)]:pd.DataFrame(rows).to_csv(report/f'{name}.csv',index=False)
    manifest=dict(settings=asdict(cfg),years=years,models=MODELS,pairs=PAIRS,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  signals=[dict(path=str(p),sha256=file_hash(p)) for p in [Path('artifacts/nn_rl_benchmark')/f'{pair}.csv' for pair in PAIRS]],
                  selected_manifest_hash=file_hash(selected_root/'manifest.json'),
                  cache_files=[dict(path=str(p),sha256=file_hash(p)) for pair in PAIRS for y in years
                               for p in [cache_root/pair/f'{y}.parquet',cache_root/pair/f'{y}_selection.parquet']])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/option_contracts'))
    p.add_argument('--selected',type=Path,default=Path('data/atm_30d'))
    p.add_argument('--output',type=Path)
    p.add_argument('--years',nargs='+',type=int,default=list(range(2010,2026)))
    p.add_argument('--reuse-cache',action='store_true')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--repair-crossed-held-quotes',action='store_true')
    p.add_argument('--option-spread-fraction',type=float,
                   help='Fraction of midpoint-to-quoted-side distance paid: 1=bid/ask, 0.5=halfway, 0=mid')
    p.add_argument('--risk-controls',action=argparse.BooleanOptionalAction,default=True,
                   help='Default on: NAV-scaled sizing, hedge/reserve/stress caps and whole-pair deleveraging')
    args=p.parse_args()
    if args.output is None:
        args.output=Path('artifacts/contracts/four_pairs_risk_controlled' if args.risk_controls else 'artifacts/contracts/four_pairs')
    if args.report is None and args.risk_controls:
        args.report=Path('reports/selected_pair_backtest/risk_controlled')
    if args.option_spread_fraction is None:
        args.option_spread_fraction=.5 if args.risk_controls else 1.
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.reuse_cache:prepare(args.root,args.selected,args.output,args.years)
    run(args.root,args.output,args.years,args.repair_crossed_held_quotes,args.cache_root,args.report,args.selected,args.option_spread_fraction,args.risk_controls)


if __name__=='__main__': main()
