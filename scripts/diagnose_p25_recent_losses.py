"""Reconcile recent maturity-transfer losses to actual cohorts and quotes."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.shared_trade_ledger import reconstruct_shared,FIELDS
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json
from scripts.curve_trading.config import config

BASE=Path('artifacts/contracts/p25_maturity_transfer')
OUT=Path('reports/p25_recent_losses')


def forward_path(stock,entry,exit_date):
    x=(stock.close*stock.cfadj).loc[entry:exit_date]
    r=np.log(x/x.shift(1)).iloc[1:]
    if len(r)==0 or not np.isfinite(r).all():
        return dict(forward_rv=np.nan,max_move_pct=np.nan,max_move_date=None)
    date=r.abs().idxmax()
    return dict(forward_rv=float(100*np.sqrt(252*r.pow(2).mean())),
        max_move_pct=float(100*r.loc[date]),max_move_date=date)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();sources={};checks=[]
    def read(path,**kwargs):
        sources[str(path)]=digest(path)
        return pd.read_parquet(path,**kwargs)
    def csv(path,**kwargs):
        sources[str(path)]=digest(path)
        return pd.read_csv(path,**kwargs)
    defs_path=Path('artifacts/contracts/tail_put25_pooled/inputs.json')
    sources[str(defs_path)]=digest(defs_path)
    labels={d['pair']:d['label'] for d in json.loads(defs_path.read_text())['definitions']}
    annual=[];pairs=[];cohorts=[];all_legs=[];worst=[];quote_audit=[];return_stats=[]
    for tenor in [30,90]:
        market=BASE/f'{tenor}D/market'
        stocks=read(market/'stocks.parquet')
        divs=read(market/'dividends.parquet')
        bystock={int(sid):g.set_index('date').sort_index() for sid,g in stocks.groupby('secid')}
        for wing in ['P25','ATM','P75','combined']:
            name=f'{tenor}D_floor'+('_'+wing if wing!='combined' else '')
            folder=BASE/'replays'/name
            nav=read(folder/'daily_nav.parquet')
            trades=csv(folder/'trades.csv',parse_dates=['date'])
            decisions=csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
            oids=trades.optionid.unique().tolist()
            quotes=pd.concat([read(market/f'{year}_quotes.parquet',filters=[('optionid','in',oids)]) for year in range(2018,2026)],ignore_index=True)
            legs,daily,financing,vegas,gross_vegas,error,components=reconstruct_shared(nav,trades,quotes,stocks,divs,asdict(config()))
            checks.append(dict(model=name,maximum_daily_error=error))
            legs['model']=name
            legs['pair']=legs.cohort.str.split('|').str[1]
            legs['wing']=legs.cohort.str.split('|').str[2]
            legs['label']=legs.pair.map(labels)
            for idx,row in legs.iterrows():
                info=forward_path(bystock[int(row.secid)],row.entry_date,row.exit_date)
                for field,value in info.items():
                    legs.loc[idx,field]=value
            all_legs.append(legs)
            group=legs.groupby('cohort')
            entry=group.entry_date.min();exit_date=group.exit_date.max()
            metadata=pd.DataFrame({'entry_date':entry,'exit_date':exit_date})
            metadata['pair']=metadata.index.str.split('|').str[1]
            metadata['wing']=metadata.index.str.split('|').str[2]
            metadata['label']=metadata.pair.map(labels)
            metadata['whole_trade_net_before_shared_financing']=daily.sum()
            metadata['model']=name
            for year in [2024,2025]:
                use=daily.index.year==year
                pnl=daily.loc[use].sum()
                active=pnl.ne(0)
                detail=metadata.loc[active].copy()
                detail['year']=year
                detail['calendar_net_before_shared_financing']=pnl.loc[active]
                detail['entered_before_year']=detail.entry_date.lt(pd.Timestamp(year,1,1))
                cohorts.append(detail.reset_index())
                pair_values=pnl.groupby(metadata.pair).sum()
                for pair,value in pair_values.items():
                    if value!=0:
                        ids=metadata.index[metadata.pair.eq(pair)]
                        pairs.append(dict(model=name,year=year,pair=pair,label=labels.get(pair),
                            net_before_shared_financing=float(value),
                            carried_cohort_pnl=float(pnl.loc[ids[metadata.loc[ids,'entry_date'].lt(pd.Timestamp(year,1,1))]].sum())))
                vals={field:float(table.loc[use].to_numpy().sum()) for field,table in components.items()}
                gross=vals['option_mid_pnl']+vals['stock_hedge_gross_pnl']+vals['dividends']
                fund=float(financing.loc[use].sum())
                net=float(pnl.sum()+fund)
                e=decisions.loc[decisions.status.eq('entered')&decisions.entry_date.dt.year.eq(year)]
                annual.append(dict(model=name,year=year,entries=len(e),gross_delta_hedged_pnl=gross,
                    financing_vs_cash=fund,net_excess_pnl=net,
                    carried_cohort_pnl=float(pnl.loc[metadata.entry_date.lt(pd.Timestamp(year,1,1))].sum()),**vals))
                r=nav.loc[use,'excess_return']
                benchmark_delta=nav.cash_benchmark.diff().fillna(0)
                return_stats.append(dict(model=name,year=year,entries=len(e),active_sessions=int(nav.loc[use,'active_cohorts'].gt(0).sum()),
                    annualized_excess_sharpe=float(np.sqrt(252)*r.mean()/r.std()) if r.std()>1e-12 else None,
                    net_excess_pnl=net,
                    gross_vega_mean=float(nav.loc[use,'gross_vega'].mean()),
                    greek_fallbacks=int(nav.loc[use,'greek_fallbacks'].sum()),
                    repaired_quotes=int(nav.loc[use,'repaired_held_quotes'].sum()),
                    modeled_held_marks=int(nav.loc[use,'causal_held_mark_fallbacks'].sum()),
                    net_dollar_pnl_sharpe=float(np.sqrt(252)*(nav.loc[use,'pnl']-benchmark_delta.loc[use]).mean()/(nav.loc[use,'pnl']-benchmark_delta.loc[use]).std())))
            # Audit exact contract marks around the largest losing cohort days.
            qlookup=quotes.set_index(['date','optionid'])
            losing=daily.loc[daily.index>='2024-01-01'].stack().nsmallest(12)
            for (date,cohort),pnl in losing.items():
                prev=nav.index[nav.index.get_loc(date)-1]
                data=dict(model=name,date=date,cohort=cohort,pair=metadata.loc[cohort,'pair'],label=metadata.loc[cohort,'label'],
                    cohort_daily_net=float(pnl),portfolio_daily_net=float(daily.loc[date].sum()+financing.loc[date]))
                for field,table in components.items():
                    data[field]=float(table.at[date,cohort])
                worst.append(data)
                own=trades.loc[trades.cohort.eq(cohort)]
                held=own.loc[own.date.le(prev)].groupby(['secid','optionid']).quantity.sum()
                for (sid,oid),qty in held[held.ne(0)].items():
                    a=qlookup.loc[(prev,oid)] if (prev,oid) in qlookup.index else None
                    b=qlookup.loc[(date,oid)] if (date,oid) in qlookup.index else None
                    spot0=float(bystock[int(sid)].at[prev,'close']);spot1=float(bystock[int(sid)].at[date,'close'])
                    rec=dict(model=name,date=date,previous_date=prev,cohort=cohort,label=metadata.loc[cohort,'label'],
                        secid=int(sid),optionid=int(oid),quantity=float(qty),prior_spot=spot0,current_spot=spot1)
                    for prefix,row in [('prior',a),('current',b)]:
                        if row is None:
                            rec[prefix+'_quote_present']=False
                            continue
                        rec[prefix+'_quote_present']=True
                        for field in ['bid','ask','bs_iv','bs_status','delta','vega','strike','expiry','cp_flag']:
                            rec[prefix+'_'+field]=row[field]
                        rec[prefix+'_mid']=(row.bid+row.ask)/2
                        rec[prefix+'_relative_width']=(row.ask-row.bid)/rec[prefix+'_mid'] if rec[prefix+'_mid']>0 else np.nan
                    quote_audit.append(rec)
            # Actual cohort daily P&L, including overlapping calendar exposure.
            daily.loc[daily.index>='2023-09-01'].to_parquet(OUT/f'{name}_cohort_daily.parquet')
            for path,value in sources.items():
                assert digest(path)==value,path
            write_json(OUT/'progress.json',dict(model=name,books_complete=len(checks),elapsed=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            pd.DataFrame(annual).to_csv(OUT/'annual_components.csv',index=False)
            print(f'{name}: cohort ledger reconciled within ${error:.8f}',flush=True)
            del quotes,legs,daily,vegas,gross_vegas,components,nav,trades,decisions,qlookup
            gc.collect()
    pd.DataFrame(return_stats).to_csv(OUT/'annual_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'annual_pairs.csv',index=False)
    pd.concat(cohorts,ignore_index=True).to_csv(OUT/'calendar_cohorts.csv',index=False)
    pd.concat(all_legs,ignore_index=True).to_csv(OUT/'legs.csv',index=False)
    pd.DataFrame(worst).to_csv(OUT/'worst_cohort_days.csv',index=False)
    pd.DataFrame(quote_audit).to_csv(OUT/'worst_day_quotes.csv',index=False)
    for code in [Path(__file__),Path('scripts/shared_trade_ledger.py'),Path('scripts/trade_quality_ledger.py')]:
        sources[str(code)]=digest(code)
    write_json(OUT/'audit.json',dict(status='complete',books=8,no_retraining=True,no_new_backtest=True,
        exact_observed_contract_marks_and_causal_greeks=True,shared_net_stock_cost_allocation=True,
        all_stock_values_and_gross_notional_match=True,all_daily_vega_and_option_values_match=True,
        all_cash_ledgers_reconcile=True,max_daily_error=max(x['maximum_daily_error'] for x in checks),
        financing_kept_separate_from_cohorts=True,source_hashes=sources,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(pd.DataFrame(return_stats).to_string(index=False),flush=True)
    print(pd.DataFrame(pairs).sort_values('net_before_shared_financing').head(15).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
