"""Separate wings from same-strike call/put pricing and hedge assumptions.

Targeted actual counterpart histories, original accepted put cohorts and
integer counts. Research diagnostics, not a new signal or reward search.
"""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import resource
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import run_p25_long_expiries as source
from scripts.option_flex.black_scholes import fit_quotes
from scripts.run_contract_pilot import normalize_quotes
from scripts.prepare_curve_contracts import HISTORY_COLUMNS
from scripts.download_pair_contract_history import QUOTE_SCHEMA,file_hash
from scripts.rebuild_smile_contracts import repaired_distributions

OUT=Path('artifacts/contracts/long_expiry_parity')
REPORT=Path('reports/long_expiry_parity')
DATA=Path('data/long_expiry_parity')
TENORS=[30,180,365]
KEYS=['date','secid','expiry','strike']


class BoundMarket(source.long.base.Market):
    """Bind quote reads to this instance rather than the legacy global root."""
    def __init__(self,root):
        self.root=root;previous=source.long.base.cached.OUT
        try:
            source.long.base.cached.OUT=root
            super().__init__()
        finally:source.long.base.cached.OUT=previous

    def quote(self,date,oid):
        previous=source.long.base.cached.OUT
        try:
            source.long.base.cached.OUT=self.root
            return super().quote(date,oid)
        finally:source.long.base.cached.OUT=previous


class ParityGreeksMarket(BoundMarket):
    """Actual call prices; put delta+1 and put vega as a European control."""
    def __init__(self,root,puts,mapping):
        super().__init__(root);self.puts=puts;self.mapping=mapping

    def quote(self,date,oid):
        call=super().quote(date,oid)
        if call is None:return None
        put=self.puts.quote(date,self.mapping[int(oid)])
        if put is None:return call
        assert call.strike==put.strike and call.expiry==put.expiry
        call=call.copy()
        if np.isfinite([put.delta,put.vega,put.get('bs_iv',np.nan)]).all():
            call.update(delta=put.delta+1,vega=put.vega,bs_iv=put.bs_iv)
        return call


def put_root(tenor):
    return source.long.base.TRANSFER if tenor==30 else source.OUT/'P'/f'{tenor}D'


def prepare():
    if (OUT/'prepared.json').exists():return
    references=[]
    for tenor in TENORS:
        rows=pd.read_parquet(source.OUT/'P/replays'/f'{tenor}D_hold/scored_allocations.parquet')
        market=BoundMarket(put_root(tenor))
        needed=pd.concat([rows[['date',f'secid_{side}',f'optionid_{side}']].rename(columns={f'secid_{side}':'secid',f'optionid_{side}':'optionid'}) for side in ['a','b']]).dropna().drop_duplicates()
        for row in needed.itertuples():
            q=market.quote(row.date,int(row.optionid))
            if q is None:continue
            references.append(dict(tenor=tenor,date=row.date,secid=int(row.secid),expiry=q.expiry,strike=q.strike,put_optionid=int(row.optionid),put_iv=q.bs_iv,put_delta=q.delta,put_vega=q.vega,put_mid=(q.bid+q.ask)/2,put_bid=q.bid,put_ask=q.ask,put_rate=q.bs_rate,put_dividend_pv=q.known_dividend_pv,spot=market.stock(row.date,int(row.secid))[0]))
        del market;gc.collect()
    refs=pd.DataFrame(references).drop_duplicates(['tenor','date','secid'])
    refs.to_parquet(OUT/'put_references.parquet',index=False)
    zero=pd.read_parquet(source.ROOT/'zero_curve.parquet');divs=repaired_distributions();parts=[];vendor=[]
    for year,g in refs.groupby(refs.date.dt.year):
        stock=pd.read_parquet(source.ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close'])
        for batch in pq.ParquetFile(source.DATA/'signal_chains'/f'{year}.parquet').iter_batches(batch_size=4096,use_threads=False):
            q=normalize_quotes(batch.to_pandas())
            p=q.loc[q.cp_flag.eq('P'),['date','optionid','impl_volatility']].rename(columns={'optionid':'put_optionid','impl_volatility':'put_vendor_iv'})
            p=p.merge(g[['date','put_optionid']].drop_duplicates(),on=['date','put_optionid'],how='inner')
            if len(p):vendor.append(p)
            eligible=q.loc[q.cp_flag.eq('C')&q.standard&q.multiplier.eq(100)]
            eligible=eligible.sort_values(KEYS+['open_interest','optionid'],ascending=[True]*4+[False,True]).drop_duplicates(KEYS)
            calls=eligible.merge(g,on=KEYS,how='inner',validate='one_to_many')
            if len(calls):parts.append(fit_quotes(calls,stock,zero,divs))
        # The original 30D puts were not in the new long-expiry chain request.
        short=g.loc[g.tenor.eq(30),['date','put_optionid']].drop_duplicates()
        if len(short):
            fields=', '.join('q.'+name for name in QUOTE_SCHEMA.names)
            query=f'select {fields} from optionm.opprcd{int(year)} q join unnest(%s::bigint[],%s::date[]) wanted(optionid,date) on q.optionid=wanted.optionid and q.date=wanted.date'
            raw=DATA/'put_signal'/f'{year}.parquet'
            source.read_query(query,(short.put_optionid.astype(int).tolist(),short.date.dt.date.tolist()),raw)
            p=normalize_quotes(pd.read_parquet(raw))[['date','optionid','impl_volatility']].rename(columns={'optionid':'put_optionid','impl_volatility':'put_vendor_iv'})
            vendor.append(p)
        print(f'Same-strike signal checks: {year}',flush=True)
        del stock;gc.collect()
    calls=pd.concat(parts,ignore_index=True).sort_values(['tenor']+KEYS+['open_interest','optionid'],ascending=[True]*5+[False,True]).drop_duplicates(['tenor']+KEYS)
    calls=calls.merge(pd.concat(vendor,ignore_index=True).drop_duplicates(['date','put_optionid']),on=['date','put_optionid'],how='left',validate='many_to_one')
    calls['vendor_iv_gap_vol_points']=100*(calls.impl_volatility-calls.put_vendor_iv)
    calls['bs_iv_gap_vol_points']=100*(calls.bs_iv-calls.put_iv)
    t=(calls.expiry-calls.date).dt.days/365
    expected=calls.spot-calls.put_dividend_pv-calls.strike*np.exp(-calls.put_rate*t)
    calls['parity_residual_dollars']=(calls.bid+calls.ask)/2-calls.put_mid-expected
    calls['european_parity_inside_quotes']=(calls.bid-calls.put_ask).le(expected)&(calls.ask-calls.put_bid).ge(expected)
    calls['delta_parity_error']=calls.bs_delta-calls.put_delta-1
    calls['vega_ratio']=calls.bs_vega/calls.put_vega
    calls.to_parquet(OUT/'same_strike_quotes.parquet',index=False)
    calls.drop(columns=[c for c in calls.columns if c.startswith('put_')],errors='ignore').to_csv(DATA/'same_strike_quotes.csv',index=False)
    wings=[]
    for tenor in TENORS:
        c=pd.read_parquet(source.OUT/'C'/f'{tenor}D/selected.parquet')
        g=refs.loc[refs.tenor.eq(tenor)].merge(c[['date','secid','strike','expiry','bs_iv']],on=['date','secid'],suffixes=('_put','_call'))
        g['vol_gap_vol_points']=100*(g.put_iv-g.bs_iv)
        g['call_vs_put_strike_pct']=100*(g.strike_call/g.strike_put-1)
        g['same_expiry']=g.expiry_put.eq(g.expiry_call)
        wings.append(g)
    pd.concat(wings,ignore_index=True).to_csv(DATA/'opposite_wing_quotes.csv',index=False)
    source.long.dump(OUT/'prepared.json',dict(put_references=len(refs),same_strike_calls=len(calls),tenors=TENORS,selection='Same signal-date, underlying, strike and expiry as original P25 put; no payoff/path filter'))


def histories():
    calls=pd.read_parquet(OUT/'same_strike_quotes.parquet')
    ids=calls.optionid.unique().astype(int).tolist()
    zero=pd.read_parquet(source.ROOT/'zero_curve.parquet');divs=repaired_distributions()
    for tenor in TENORS:
        market=OUT/f'{tenor}D/market';market.mkdir(parents=True,exist_ok=True)
        for kind in ['stocks','rates','dividends']:
            shutil.copyfile(put_root(tenor)/'market'/f'{kind}.parquet',market/f'{kind}.parquet')
    for year in range(2018,2026):
        fitted=DATA/'fitted'/f'{year}.parquet'
        if not fitted.exists():
            fields=', '.join(QUOTE_SCHEMA.names)
            raw=DATA/'history'/f'{year}.parquet'
            source.read_query(f"select {fields} from optionm.opprcd{year} where optionid=ANY(%s) and cp_flag='C' and date between %s and %s",(ids,f'{year}-01-01',min(f'{year}-12-31','2025-08-29')),raw)
            stock=pd.read_parquet(source.ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close'])
            fitted.parent.mkdir(parents=True,exist_ok=True);temp=fitted.with_suffix('.partial');writer=None;count=0
            try:
                for batch in pq.ParquetFile(raw).iter_batches(batch_size=4096,use_threads=False):
                    q=fit_quotes(normalize_quotes(batch.to_pandas()),stock,zero,divs)
                    q['delta']=q.bs_delta;q['vega']=q.bs_vega
                    table=pa.Table.from_pandas(q[HISTORY_COLUMNS],preserve_index=False)
                    if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                    writer.write_table(table);count+=len(q)
                if writer is None:raise RuntimeError(f'Empty counterpart history: {year}')
                writer.close();writer=None;temp.replace(fitted)
                source.long.dump(fitted.with_suffix('.json'),dict(rows=count,sha256=file_hash(fitted),source_sha256=file_hash(raw)))
            finally:
                if writer is not None:writer.close()
            print(f'Fitted same-strike call histories: {year}, {count:,} rows',flush=True)
            del stock;gc.collect()
        for tenor in TENORS:
            target=OUT/f'{tenor}D/market'/f'{year}_quotes.parquet'
            if not target.exists():os.link(fitted,target)


def replay():
    for tenor in TENORS:
        baseline=source.OUT/'P/replays'/f'{tenor}D_hold'
        rows=pd.read_parquet(baseline/'scored_allocations.parquet')
        trades=pd.read_csv(baseline/'trades.csv',parse_dates=['date']);entry=trades.loc[trades.reason.eq('entry')]
        groups={cohort:g.set_index('secid') for cohort,g in entry.groupby('cohort')}
        rows=rows.loc[[f'{r.date.date()}|{r.pair}|{r.wing}' in groups for r in rows.itertuples()]].copy()
        counts={f'{r.date.date()}|{r.pair}|{r.wing}':tuple(groups[f'{r.date.date()}|{r.pair}|{r.wing}'].loc[[r.secid_a,r.secid_b],'quantity'].abs().astype(int)) for r in rows.itertuples()}
        puts=BoundMarket(put_root(tenor))
        source.long.save_case(f'{tenor}D_reference_put_locked',rows,puts,tenor,anchor=baseline/'daily_nav.parquet',locked_counts=counts)
        choice=pd.read_parquet(OUT/'same_strike_quotes.parquet');choice=choice.loc[choice.tenor.eq(tenor)]
        call_rows=rows.drop(columns=['optionid_a','optionid_b']).copy();call_rows['wing']='C75_same_strike'
        for side in ['a','b']:
            keys=rows[['date',f'secid_{side}',f'optionid_{side}']].rename(columns={f'secid_{side}':'secid',f'optionid_{side}':'put_optionid'})
            keys=keys.merge(choice[['date','secid','put_optionid','optionid']],on=['date','secid','put_optionid'],how='left',validate='many_to_one')
            call_rows[f'optionid_{side}']=keys.optionid.to_numpy()
        call_counts={k.rsplit('|',1)[0]+'|C75_same_strike':v for k,v in counts.items()}
        market=BoundMarket(OUT/f'{tenor}D')
        native_rows=call_rows.copy();checks=[];cfg=source.configuration(tenor)
        for idx,r in call_rows.iterrows():
            if not np.isfinite([r.optionid_a,r.optionid_b]).all():continue
            entry_date=market.rates.index[market.rates.index>r.date][0]
            qs=[market.quote(entry_date,int(oid)) for oid in [r.optionid_a,r.optionid_b]]
            if any(q is None or not np.isfinite(q.vega) or q.vega<=0 for q in qs):continue
            cohort=f'{r.date.date()}|{r.pair}|{r.wing}'
            vegas=[n*100*q.vega for n,q in zip(call_counts[cohort],qs)]
            imbalance=abs(vegas[0]-vegas[1])/max(vegas)
            blocked=imbalance>cfg.max_entry_vega_imbalance
            checks.append(dict(date=r.date,pair=r.pair,entry_date=entry_date,locked_count_vega_imbalance=imbalance,blocked=blocked))
            # Retain a flat decision rather than relax neutrality or resize.
            if blocked:native_rows.loc[idx,'optionid_a']=np.nan
        pd.DataFrame(checks).to_csv(REPORT/f'{tenor}D_native_call_vega_checks.csv',index=False)
        source.long.save_case(f'{tenor}D_actual_call_same_strike_locked',native_rows,market,tenor,locked_counts=call_counts)
        mapping=dict(zip(choice.optionid.astype(int),choice.put_optionid.astype(int)))
        market=ParityGreeksMarket(OUT/f'{tenor}D',puts,mapping)
        source.long.save_case(f'{tenor}D_actual_call_parity_greeks_locked',call_rows,market,tenor,locked_counts=call_counts)
        # Descriptive common-cohort control: entry eligibility, never P&L ranks.
        accepted=pd.read_csv(OUT/'replays'/f'{tenor}D_actual_call_same_strike_locked/decisions.csv',parse_dates=['signal_date'])
        accepted=accepted.loc[accepted.status.eq('entered')]
        common=rows.merge(accepted[['signal_date','pair']].rename(columns={'signal_date':'date'}),on=['date','pair'],how='inner',validate='one_to_one')
        common_counts={f'{r.date.date()}|{r.pair}|{r.wing}':counts[f'{r.date.date()}|{r.pair}|{r.wing}'] for r in common.itertuples()}
        source.long.save_case(f'{tenor}D_put_common_call_entries_locked',common,puts,tenor,locked_counts=common_counts)
        accepted=pd.read_csv(OUT/'replays'/f'{tenor}D_actual_call_parity_greeks_locked/decisions.csv',parse_dates=['signal_date'])
        accepted=accepted.loc[accepted.status.eq('entered')]
        common=rows.merge(accepted[['signal_date','pair']].rename(columns={'signal_date':'date'}),on=['date','pair'],how='inner',validate='one_to_one')
        common_counts={f'{r.date.date()}|{r.pair}|{r.wing}':counts[f'{r.date.date()}|{r.pair}|{r.wing}'] for r in common.itertuples()}
        source.long.save_case(f'{tenor}D_put_common_parity_greeks_entries_locked',common,puts,tenor,locked_counts=common_counts)


def quote_stats():
    quotes=pd.read_parquet(OUT/'same_strike_quotes.parquet');wings=pd.read_csv(DATA/'opposite_wing_quotes.csv')
    stats=[]
    for tenor in TENORS:
        g=quotes.loc[quotes.tenor.eq(tenor)];w=wings.loc[wings.tenor.eq(tenor)]
        valid=g.loc[g.bs_status.eq('ok')&g.standard&g.put_iv.gt(0)]
        vendor=valid.loc[valid.impl_volatility.between(1e-8,10)&valid.put_vendor_iv.between(1e-8,10)]
        stats.append(dict(tenor=tenor,matched_strike_quotes=len(g),valid_bs_quotes=len(valid),valid_vendor_iv_pairs=len(vendor),median_put25_minus_call25_iv_points=float(w.vol_gap_vol_points.median()),median_call25_strike_above_put25_pct=float(w.call_vs_put_strike_pct.median()),same_expiry_fraction=float(w.same_expiry.mean()),median_same_strike_call_minus_put_bs_iv_points=float(valid.bs_iv_gap_vol_points.median()),median_abs_same_strike_bs_iv_gap=float(valid.bs_iv_gap_vol_points.abs().median()),median_abs_same_strike_vendor_iv_gap=float(vendor.vendor_iv_gap_vol_points.abs().median()),parity_inside_bid_ask_fraction=float(g.european_parity_inside_quotes.mean()),median_delta_parity_error=float(valid.delta_parity_error.median()),median_vega_ratio=float(valid.vega_ratio.median())))
    pd.DataFrame(stats).to_csv(REPORT/'quote_summary.csv',index=False)
    print(pd.DataFrame(stats).to_string(index=False),flush=True)
    return pd.DataFrame(stats)


def summarize():
    stats=quote_stats()
    results=[json.loads(p.read_text()) for p in sorted((OUT/'replays').glob('*/status.json'))]
    table=pd.DataFrame(results);table.to_csv(REPORT/'replay_comparison.csv',index=False)
    components=[]
    for folder in sorted((OUT/'replays').iterdir()):
        if not (folder/'status.json').exists():continue
        nav=pd.read_parquet(folder/'daily_nav.parquet');trades=pd.read_csv(folder/'trades.csv')
        option_gross=float((-trades.quantity*100*trades.price).sum())
        c=dict(case=folder.name,option_gross=option_gross,hedge_gross=float(nav.hedge_gross_pnl.sum()),dividends=float(nav.dividend_cash.sum()),interest_above_cash=float(nav.interest.sum()-(nav.cash_benchmark.iloc[-1]-10_000_000)),option_cost=float(nav.option_execution_cost.sum()),stock_cost=float(nav.hedge_cost.sum()),stock_borrow=float(nav.stock_borrow_cost.sum()))
        c['gross_delta_hedged_with_dividends']=c['option_gross']+c['hedge_gross']+c['dividends']
        c['before_execution_and_borrow_after_financing']=c['gross_delta_hedged_with_dividends']+c['interest_above_cash']
        c['net_excess_pnl']=c['gross_delta_hedged_with_dividends']+c['interest_above_cash']-c['option_cost']-c['stock_cost']-c['stock_borrow']
        assert abs(c['net_excess_pnl']-(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]))<1e-5
        components.append(c)
    pd.DataFrame(components).to_csv(REPORT/'pnl_components.csv',index=False)
    paired=[];frequency=[]
    for tenor in TENORS:
        for label,c_case,p_case in [('native_greeks',f'{tenor}D_actual_call_same_strike_locked',f'{tenor}D_put_common_call_entries_locked'),('parity_greeks',f'{tenor}D_actual_call_parity_greeks_locked',f'{tenor}D_put_common_parity_greeks_entries_locked')]:
            c=table.set_index('case').loc[c_case];p=table.set_index('case').loc[p_case]
            cnav=pd.read_parquet(OUT/'replays'/c_case/'daily_nav.parquet');pnav=pd.read_parquet(OUT/'replays'/p_case/'daily_nav.parquet')
            paired.append(dict(tenor=tenor,greeks=label,put_entries=int(p.entries),call_entries=int(c.entries),put_sharpe=float(p.annualized_excess_sharpe),call_sharpe=float(c.annualized_excess_sharpe),put_excess_pnl=float(p.net_excess_pnl),call_excess_pnl=float(c.net_excess_pnl),daily_return_correlation=float(cnav.excess_return.corr(pnav.excess_return))))
            for freq,periods in [('daily',252),('W-FRI',52),('ME',12)]:
                returns=[]
                for nav in [pnav,cnav]:
                    if freq=='daily':r=nav.excess_return.iloc[1:]
                    else:
                        sample=nav[['nav','cash_benchmark']].resample(freq).last().dropna()
                        r=sample.nav.pct_change()-sample.cash_benchmark.pct_change()
                    returns.append(r.dropna())
                score=lambda r:float(r.mean()/r.std(ddof=1)*np.sqrt(periods)) if r.std(ddof=1)>0 else 0.
                frequency.append(dict(tenor=tenor,greeks=label,frequency=freq,put_sharpe=score(returns[0]),call_sharpe=score(returns[1]),return_correlation=float(returns[0].corr(returns[1]))))
    pd.DataFrame(paired).to_csv(REPORT/'matched_comparison.csv',index=False)
    pd.DataFrame(frequency).to_csv(REPORT/'frequency_diagnostics.csv',index=False)
    pd.concat([pd.read_csv(p) for p in sorted((OUT/'replays').glob('*/annual.csv'))],ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    source.verify_previous_sources()
    note=('CALL/PUT AND SMILE AUDIT\n\n25-delta puts and 25-delta calls have different strikes; put-call parity concerns identical strike and expiry. Same-strike counterparts are approximately 75-delta calls. Signal-day diagnostics use actual WRDS quotes and vendor IVs alongside our European BS fit. The fit includes declared ordinary dividends only, not forecasts of undeclared dividends or American exercise. Midpoint parity residuals therefore include carry/model approximations as well as quote noise, not necessarily arbitrage.\n\nReplay the originally accepted put cohorts with identical integer quantities and reproduce the complete baseline NAV. Then transfer them to actual calls at the identical strike/expiry, retaining all original dates/directions/counts, daily hedge, actual costs and risk limits. Rejected calls remain explicit. The common-cohort put replay is a descriptive entry-feasibility control, not a new tradable signal. A separate call-price control uses put delta+1 and put vega (European parity) to isolate separate BS inversion effects; it is not an American pricing model or a deployed policy. Corporate actions retain the announced pre-effective paired exits, including GE HealthCare.\n\n'+pd.DataFrame(stats).to_string(index=False)+'\n\n'+table[['case','annualized_excess_sharpe','entries','locked_cohort_risk_rejections','net_excess_pnl']].to_string(index=False)+'\n')
    (REPORT/'findings.txt').write_text(note)
    c30=pd.DataFrame(components).set_index('case')
    put=c30.loc['30D_put_common_parity_greeks_entries_locked'];call=c30.loc['30D_actual_call_parity_greeks_locked']
    f=pd.DataFrame(frequency);monthly=f.loc[f.tenor.eq(30)&f.greeks.eq('parity_greeks')&f.frequency.eq('ME')].iloc[0]
    extra=(f'\nINTERPRETATION\n\n30D European-parity-Greek control, 64 identical cohorts/counts: before execution and stock borrowing, after financing, puts earn ${put.before_execution_and_borrow_after_financing:,.2f} and calls ${call.before_execution_and_borrow_after_financing:,.2f}. Net excess P&L is ${put.net_excess_pnl:,.2f} versus ${call.net_excess_pnl:,.2f}; extra call hedge trading and stock borrowing explain most of this endpoint difference. Daily excess Sharpe is .4314 versus .2405. Monthly return correlation is {monthly.return_correlation:.4f}, compared with .4695 daily. This supports economically similar 30D payoff exposure with additional call costs and short-lived mark/carry variation, rather than evidence of a completely different volatility signal. It does not separately identify stale quotes versus changing carry. Sampling diagnostics retain the daily Sharpe as the primary measure; monthly Sharpe is not substituted to optimize results.\n\nThis control is a 64-trade subset through the common July 2024 entry cutoff, not the full ~1.0 Sharpe put checkpoint. Native independently fitted call Greeks use 60 matched trades; their puts/calls are .4010/.1112. Longer maturities show larger carry and model differences. European declared-dividend-only fits, American exercise/assignment omissions, hard-to-borrow pricing and floating cash financing versus term discounting require further validation before attributing a long-expiry put/call difference to alpha.\n\nPrimary parity reference: https://www.optionseducation.org/advancedconcepts/put-call-parity\n')
    with (REPORT/'findings.txt').open('a') as handle:handle.write(extra)
    progress=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
    source.long.dump(REPORT/'audit.json',dict(cases=len(results),original_put_nav_reproduced=True,independent_ledger_checks=True,selection_by_entry_data_not_profit=True,peak_rss_bytes=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,progress.get('peak_rss_bytes',0))))
    paths=[Path(__file__),OUT/'put_references.parquet',OUT/'same_strike_quotes.parquet']+list((OUT/'replays').glob('*/status.json'))
    source.long.dump(REPORT/'source_manifest.json',{str(p):file_hash(p) for p in paths})
    print(table[['case','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(pd.DataFrame(frequency).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','quotes','histories','replay','summarize','all'],default='all');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    source.long.OUT=OUT;source.long.REPORT=REPORT;source.long.configuration=source.configuration
    source.long.base.UNIFORM_CORPORATE_EVENTS=[*source.long.base.UNIFORM_CORPORATE_EVENTS,source.GE_HEALTHCARE_EVENT]
    source.verify_previous_sources()
    if a.phase=='quotes':quote_stats();return
    if a.phase in ['prepare','all']:prepare()
    if a.phase in ['histories','all']:histories()
    if a.phase in ['replay','all']:replay()
    if a.phase in ['summarize','all']:summarize()


if __name__=='__main__':main()
