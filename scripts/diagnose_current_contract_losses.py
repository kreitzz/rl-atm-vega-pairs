"""Exact ledger attribution and fixed-inventory cost bridge; no policy tuning."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.reprice_contract_execution import reprice_cash
from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar

REPORT=Path('reports/current_contract_losses')
POLICIES={
    'earlier_filtered_rl':(Path('artifacts/contracts/entry_gates'),'rl_sized_memory_both','reports/contract_entry_gates/status.csv','rl_sized','memory_both'),
    'surface_rl':(Path('artifacts/contracts/surface_features'),'rl_sized_recent','reports/contract_surface_features/status.csv','rl_sized','recent'),
    'surface_filtered_rl':(Path('artifacts/contracts/surface_features'),'rl_sized_recent_memory','reports/contract_surface_features/status.csv','rl_sized','recent_memory'),
    'surface_nn':(Path('artifacts/contracts/surface_features'),'nn_fixed_recent','reports/contract_surface_features/status.csv','nn_fixed','recent'),
    'surface_filtered_nn':(Path('artifacts/contracts/surface_features'),'nn_fixed_recent_memory','reports/contract_surface_features/status.csv','nn_fixed','recent_memory'),
}
COMPONENTS=['option_mid_pnl','stock_hedge_gross_pnl','dividends','option_commission','option_penalty','stock_execution','stock_borrow','financing_vs_cash','net_excess_pnl']


def read_csv(path,dates=None):
    if not path.exists():return pd.DataFrame()
    try:return pd.read_csv(path,parse_dates=dates)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def complete_nav(nav,capital):
    """Cash-only analytical paths omit ledger fields; restore their exact zeros."""
    out=nav.copy()
    if 'pnl' not in out:out['pnl']=out.nav.diff().fillna(out.nav.iloc[0]-capital)
    if 'cash' not in out:out['cash']=out.nav
    if 'interest' not in out:out['interest']=out.pnl
    for col in ['option_value','stock_value','option_execution_cost','option_model_penalty','hedge_cost','stock_borrow_cost','dividend_cash','gross_vega','net_vega','active_cohorts','risk_exit_cohorts']:
        if col not in out:out[col]=0.
    return out


def attribute(nav,trades,secids,stocks,dividends,cfg):
    """Daily book components; complete-lifecycle per-underlying attribution."""
    idx=nav.index;elapsed=idx.to_series().diff().dt.days.fillna(0)/365
    cash_interest=nav.cash_benchmark.diff().fillna(nav.cash_benchmark.iloc[0]-cfg['capital'])
    result=pd.DataFrame(0.,index=idx,columns=COMPONENTS)
    legs=[];gross_stock=pd.Series(0.,index=idx);div_total=pd.Series(0.,index=idx)
    stock_cost_total=pd.Series(0.,index=idx);borrow_total=pd.Series(0.,index=idx)
    for sid in secids:
        h=nav.get(f'hedge_{sid}',pd.Series(0.,index=idx))
        prev=h.shift(1).fillna(0);factor=nav.get(f'split_factor_{sid}',pd.Series(1.,index=idx))
        spot=stocks.xs(sid,level='secid').reindex(idx) if sid in stocks.index.get_level_values('secid') else pd.Series(np.nan,index=idx)
        change=spot*factor-spot.shift(1)
        if (change.isna()&prev.ne(0)).any():raise AssertionError('Missing held stock price')
        hedge=prev*change.fillna(0)
        cost=((h-prev*factor).abs()*spot).fillna(0)*cfg['stock_cost_bps']/10000
        borrow=((-prev).clip(lower=0)*spot.shift(1)).fillna(0)*cfg['stock_borrow_rate']*elapsed
        dividend=pd.Series(0.,index=idx)
        for e in dividends[dividends.secid.eq(sid)].itertuples():
            j=idx.searchsorted(e.ex_date)
            if 0<j<len(idx):dividend.iloc[j]+=prev.iloc[j]*factor.iloc[j]*e.amount
        own=trades[trades.secid.eq(sid)] if len(trades) else pd.DataFrame()
        option_gross=float((-own.quantity*100*own.price).sum()) if len(own) else 0.
        commission=float(own.commission.sum()) if len(own) else 0.
        penalty=float(own.execution_penalty.sum()) if len(own) else 0.
        entries=own[own.reason.eq('entry')] if len(own) else own
        legs.append(dict(secid=sid,option_mid_pnl=option_gross,stock_hedge_gross_pnl=hedge.sum(),dividends=dividend.sum(),
            option_commission=commission,option_penalty=penalty,stock_execution=cost.sum(),stock_borrow=borrow.sum(),
            net_trading_before_financing=option_gross+hedge.sum()+dividend.sum()-commission-penalty-cost.sum()-borrow.sum(),
            long_option_entries=int(entries.quantity.gt(0).sum()) if len(entries) else 0,
            short_option_entries=int(entries.quantity.lt(0).sum()) if len(entries) else 0))
        gross_stock+=hedge;div_total+=dividend;stock_cost_total+=cost;borrow_total+=borrow
    option_cash=pd.Series(0.,index=idx)
    if len(trades):
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        option_cash=(-trades.quantity*100*trades.price).groupby(trades.date).sum().reindex(idx,fill_value=0)
        commissions=trades.groupby('date').commission.sum().reindex(idx,fill_value=0)
        penalties=trades.groupby('date').execution_penalty.sum().reindex(idx,fill_value=0)
        np.testing.assert_allclose(commissions+penalties,nav.option_execution_cost,atol=1e-7,rtol=0)
    result['option_mid_pnl']=option_cash+nav.option_value.diff().fillna(nav.option_value)
    result['stock_hedge_gross_pnl']=gross_stock
    result['dividends']=div_total
    result['option_commission']=nav.option_execution_cost-nav.option_model_penalty
    result['option_penalty']=nav.option_model_penalty
    result['stock_execution']=stock_cost_total;result['stock_borrow']=borrow_total
    result['financing_vs_cash']=nav.interest-cash_interest
    result['net_excess_pnl']=nav.pnl-cash_interest
    for computed,stored in [(div_total,nav.dividend_cash),(stock_cost_total,nav.hedge_cost),(borrow_total,nav.stock_borrow_cost)]:
        np.testing.assert_allclose(computed,stored,atol=1e-7,rtol=0)
    explained=result.option_mid_pnl+gross_stock+div_total-result.option_commission-result.option_penalty-stock_cost_total-borrow_total+result.financing_vs_cash
    np.testing.assert_allclose(explained,result.net_excess_pnl,atol=1e-7,rtol=0)
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-6
    np.testing.assert_allclose(sum(r['net_trading_before_financing'] for r in legs)+result.financing_vs_cash.sum(),result.net_excess_pnl.sum(),atol=1e-6,rtol=0)
    return result,legs,float(abs(explained-result.net_excess_pnl).max())


def main():
    REPORT.mkdir(exist_ok=True);started=time.monotonic()
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    definitions=[d for d in definitions if d['group']=='broad'];failed=set();source_status={}
    for name,(_,_,path,model,variant) in POLICIES.items():
        s=pd.read_csv(path);s=s[s.model.eq(model)&s.variant.eq(variant)&s.group.eq('broad')]
        source_status[name]=s;failed|=set(s.loc[s.status.ne('complete'),'pair'])
    definitions=[d for d in definitions if d['pair'] not in failed]
    cfg=json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())['settings']
    assert cfg['option_spread_fraction']==0
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2018-01-01')&(calendar<='2025-08-29')]
    market=Market(calendar);stocks=market.stocks.set_index(['secid','date']).close.sort_index()
    pair_rows=[];leg_rows=[];annual_rows=[];bridge_rows=[];decisions=[];exits=[];cohorts=[];daily_losses=[];aggregate_rows=[];hashes={};errors=[]
    for name,(root,subdir,_,_,_) in POLICIES.items():
        books={};components=pd.DataFrame(0.,index=calendar,columns=COMPONENTS)
        for d in definitions:
            folder=root/d['pair']/subdir
            for file in ['daily_nav.parquet','trades.csv','decisions.csv']:
                if (folder/file).exists():hashes[str(folder/file)]=digest(folder/file)
            nav=complete_nav(pd.read_parquet(folder/'daily_nav.parquet'),cfg['capital'])
            assert nav.index.equals(calendar)
            trades=read_csv(folder/'trades.csv',['date']);dec=read_csv(folder/'decisions.csv',['signal_date','entry_date'])
            attrib,legs,error=attribute(nav,trades,d['secids'],stocks,market.divs,cfg);errors.append(error);components+=attrib
            for leg in legs:leg_rows.append(dict(policy=name,pair=d['pair'],label=d['label'],**leg))
            entered=dec[dec.status.eq('entered')] if len(dec) else pd.DataFrame()
            entry_vega=float(((entered.vega_a+entered.vega_b)/2).sum()) if len(entered) else 0.
            totals=attrib.sum().to_dict();gross=totals['option_mid_pnl']+totals['stock_hedge_gross_pnl']+totals['dividends']
            pair_rows.append(dict(policy=name,pair=d['pair'],label=d['label'],industry=d['industry'],entries=len(entered),
                entry_vega_sum=entry_vega,gross_hedged_pnl=gross,**totals))
            worst=attrib.nsmallest(5,'net_excess_pnl').copy();worst['date']=worst.index;worst['policy']=name;worst['pair']=d['pair'];worst['label']=d['label'];worst['risk_exit_cohorts']=nav.risk_exit_cohorts.reindex(worst.index)
            daily_losses.extend(worst.reset_index(drop=True).to_dict('records'))
            if len(dec):
                for reason,count in dec.status.value_counts().items():decisions.append(dict(policy=name,reason=reason,count=int(count)))
                for limits,count in dec.loc[dec.status.eq('risk_limit_entry'),'risk_limits'].dropna().value_counts().items() if 'risk_limits' in dec else []:
                    decisions.append(dict(policy=name,reason='risk_components:'+limits,count=int(count)))
            if len(trades):
                closed=trades[~trades.reason.isin(['entry','split_adjustment'])]
                for cohort,t in trades.groupby('cohort'):
                    start=t.loc[t.reason.eq('entry'),'date'].min();end=t.date.max()
                    close=t[~t.reason.isin(['entry','split_adjustment'])]
                    reason='|'.join(sorted(close.reason.unique()))
                    e=entered[entered.signal_date.eq(pd.Timestamp(cohort))]
                    assert len(e)==1
                    cohorts.append(dict(policy=name,pair=d['pair'],label=d['label'],cohort=cohort,
                        entry_date=str(start.date()),exit_date=str(end.date()),holding_days=(end-start).days,exit_reason=reason,
                        action=float(e.action.iloc[0]),mean_initial_vega=float((e.vega_a.iloc[0]+e.vega_b.iloc[0])/2),
                        option_commission=float(t.commission.sum()),option_penalty=float(t.execution_penalty.sum())))
            stages={'net':np.zeros(len(nav)), 'no_option_penalty':nav.option_model_penalty,
                    'no_option_commission':nav.option_execution_cost-nav.option_model_penalty,
                    'no_option_costs':nav.option_execution_cost,
                    'no_execution_costs':nav.option_execution_cost+nav.hedge_cost,
                    'no_execution_or_stock_borrow':nav.option_execution_cost+nav.hedge_cost+nav.stock_borrow_cost}
            for stage,saving in stages.items():
                out=reprice_cash(nav,market.rates,saving,np.zeros(len(nav)),cfg['capital'],cfg['borrowing_spread'])
                if stage=='net':np.testing.assert_allclose(out.nav,nav.nav,atol=1e-7,rtol=0)
                if stage not in books:books[stage]=out[['nav','cash_benchmark']].copy()
                else:books[stage]+=out[['nav','cash_benchmark']]
        components.to_csv(REPORT/f'{name}_daily_attribution.csv')
        for year,g in components.groupby(components.index.year):annual_rows.append(dict(policy=name,year=year,**g.sum().to_dict()))
        aggregate_rows.append(dict(policy=name,pairs=len(definitions),**components.sum().to_dict()))
        for stage,book in books.items():
            book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT/f'{name}_{stage}_portfolio.csv')
            for period,mask in [('full',book.index.year>0),('2018_2021',book.index.year<2022),('2022_2025',book.index.year>=2022)]:
                g=book[mask];excess=g.excess_return
                capital=len(definitions)*cfg['capital']
                prior=capital if g.index[0]==book.index[0] else book.loc[:g.index[0]].nav.iloc[-2]
                previous_cash=capital if g.index[0]==book.index[0] else book.loc[:g.index[0]].cash_benchmark.iloc[-2]
                bridge_rows.append(dict(policy=name,stage=stage,period=period,pairs=len(definitions),
                    net_excess_dollars=(g.nav.iloc[-1]-prior)-(g.cash_benchmark.iloc[-1]-previous_cash),
                    annualized_excess_sharpe=np.sqrt(252)*excess.mean()/excess.std() if excess.std()>1e-12 else None))
        print(f'Attributed {name}',flush=True)
    pairs=pd.DataFrame(pair_rows);pairs.to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(leg_rows).to_csv(REPORT/'legs.csv',index=False)
    pd.DataFrame(annual_rows).to_csv(REPORT/'annual_components.csv',index=False)
    pd.DataFrame(aggregate_rows).to_csv(REPORT/'components.csv',index=False)
    bridge=pd.DataFrame(bridge_rows);bridge.to_csv(REPORT/'cost_bridge.csv',index=False)
    pd.DataFrame(decisions).groupby(['policy','reason'])['count'].sum().reset_index().to_csv(REPORT/'entry_decisions.csv',index=False)
    cohort=pd.DataFrame(cohorts);cohort.to_csv(REPORT/'cohorts.csv',index=False)
    cohort.groupby(['policy','exit_reason']).agg(cohorts=('cohort','size'),average_holding_days=('holding_days','mean')).reset_index().to_csv(REPORT/'exits.csv',index=False)
    pd.DataFrame(daily_losses).sort_values('net_excess_pnl').groupby('policy',sort=False).head(20).to_csv(REPORT/'worst_pair_days.csv',index=False)
    concentration=[]
    for name,g in pairs.groupby('policy'):
        active=g[g.entries.gt(0)];negative=active[active.net_excess_pnl.lt(0)].sort_values('net_excess_pnl');positive=active[active.net_excess_pnl.gt(0)]
        concentration.append(dict(policy=name,entries=int(active.entries.sum()),active_pairs=len(active),profitable_pairs=len(positive),losing_pairs=len(negative),
            gross_profitable_pairs=int(active.gross_hedged_pnl.gt(0).sum()),
            total_positive_dollars=positive.net_excess_pnl.sum(),total_negative_dollars=negative.net_excess_pnl.sum(),
            five_worst_share_of_losses=negative.net_excess_pnl.head(5).sum()/negative.net_excess_pnl.sum(),
            five_worst='; '.join(negative.label.head(5))))
    pd.DataFrame(concentration).to_csv(REPORT/'concentration.csv',index=False)
    manifest=dict(pairs=len(definitions),excluded_pairs=sorted(failed),settings=cfg,policy_folders={k:str(v[0]/v[1]) for k,v in POLICIES.items()},
        method='Exact daily ledger and terminal flat-book per-underlying attribution. Fixed trade/hedge inventories in cost bridge; financing recomputed, no resizing, selection or risk-exit changes. Gross hedged P&L excludes financing, costs and stock borrow; financing versus cash is separate.',
        max_daily_ledger_error=max(errors),source_sha256=hashes,code_sha256=digest(__file__),elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_report(pd.DataFrame(aggregate_rows),bridge,pairs,pd.DataFrame(concentration),len(definitions))


def write_report(components,bridge,pairs,concentration,n):
    focus=['earlier_filtered_rl','surface_filtered_rl','surface_filtered_nn']
    lines=['# What causes the current contract simulations to lose money?', '',
        f'All comparisons use the same {n} complete broad pairs, 2018–August 2025, at the existing $1 million budget per pair. Cash-only sleeves remain included; two unresolved histories are explicitly excluded. These are accounting diagnostics of saved policies, not a new strategy fit.', '',
        '## Exact P&L components', '',
        'Dollar amounts below are thousands of dollars across the entire configured portfolio and full period. Option and stock hedge P&L must be read together. Costs are shown as positive deductions. Financing is actual cash interest minus the interest on the matched idle-cash benchmark.', '',
        '| Policy | Option mid P&L | Stock hedge P&L | Dividends | Option commission | Option penalty | Stock execution | Stock borrow | Financing vs cash | Net excess |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in components[components.policy.isin(focus)].itertuples():
        vals=[getattr(r,c)/1000 for c in COMPONENTS]
        lines.append('| '+r.policy+' | '+' | '.join(f'{x:,.1f}' for x in vals)+' |')
    lines+=['','## Fixed-inventory cost bridge','','Remove one cost or a stated combination while keeping the observed trades, hedges and exits fixed. Recompute cash financing exactly. These are diagnostic counterfactuals, not achievable strategy results; lower fees could change sizing, eligibility and risk exits in a fresh simulation. Sharpe is annualized from daily cash-excess returns using √252.','',
        '| Policy | Cost treatment | Full-period Sharpe | 2022–2025 Sharpe | Full-period excess dollars |','|---|---|---:|---:|---:|']
    for name in focus:
        g=bridge[bridge.policy.eq(name)]
        for r in g[g.period.eq('full')].itertuples():
            later=g[g.period.eq('2022_2025')&g.stage.eq(r.stage)].annualized_excess_sharpe.iloc[0]
            lines.append(f'| {name} | {r.stage} | {r.annualized_excess_sharpe:.3f} | {later:.3f} | ${r.net_excess_dollars:,.0f} |')
    lines+=['','## Loss concentration','','| Policy | Entries | Active pairs | Profitable net | Profitable before costs/funding | Five worst pairs’ share of losses | Five worst pairs |','|---|---:|---:|---:|---:|---:|---|']
    for r in concentration[concentration.policy.isin(focus)].itertuples():
        lines.append(f'| {r.policy} | {r.entries} | {r.active_pairs} | {r.profitable_pairs} | {r.gross_profitable_pairs} | {r.five_worst_share_of_losses:.1%} | {r.five_worst} |')
    lines+=['','The per-underlying file reconciles each option leg with its aggregate stock hedge, option/stock costs, stock borrow and dividends over complete lifecycles. Shared portfolio financing stays separate; leg P&L is not presented as standalone strategy profitability. Long/short counts describe option entries, and a leg can trade in both directions.','',
        'Exit-reason and holding-period tables describe observed behavior. They do not establish that a risk limit caused a loss: testing different exits requires a separate replay. Worst pair-day rows are mark-to-market attribution, not an allocation to individual overlapping cohorts.','',
        '[Components](current_contract_losses/components.csv) · [Cost bridge](current_contract_losses/cost_bridge.csv) · [Pair P&L](current_contract_losses/pairs.csv) · [Underlying legs](current_contract_losses/legs.csv) · [Annual components](current_contract_losses/annual_components.csv) · [Exit reasons](current_contract_losses/exits.csv) · [Worst pair-days](current_contract_losses/worst_pair_days.csv) · [Entry decisions](current_contract_losses/entry_decisions.csv)','']
    Path('reports/current_contract_losses.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
