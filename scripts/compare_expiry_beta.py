"""Matched-roster comparison of retrained expiry policies under two beta gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPORT=Path('reports/expiry_trained_beta_085_115')
ROOTS={'0.75–1.25':Path('artifacts/contracts/expiry_trained'),
       '0.85–1.15':Path('artifacts/contracts/expiry_trained_beta_085_115')}


def main():
    old=pd.read_csv('reports/expiry_trained/status.csv')
    new=pd.read_csv(REPORT/'status.csv')
    prior=pd.read_csv('reports/expiry_holding/status.csv')
    failed=set(old.loc[old.status.ne('complete'),'pair'])|set(new.loc[new.status.ne('complete'),'pair'])
    failed|=set(prior.loc[prior.status.ne('complete'),'pair'])
    members=sorted(set(old.pair)&set(new.pair)&set(prior.pair)-failed)
    assert members
    (REPORT/'matched_pairs.json').write_text(json.dumps(members,indent=2)+'\n')
    results=[];annual=[];activity=[]
    for band,root in ROOTS.items():
        status=old if band=='0.75–1.25' else new
        for model in ['rl_sized','nn_fixed']:
            book=None;gross=option_cost=hedge_cost=borrow=0.
            for pair in members:
                nav=pd.read_parquet(root/pair/model/'daily_nav.parquet',columns=['nav','cash_benchmark','pnl','interest','stock_borrow_cost','option_execution_cost','hedge_cost'])
                gross+=(nav.pnl-nav.interest+nav.stock_borrow_cost+nav.option_execution_cost+nav.hedge_cost).sum()
                option_cost+=nav.option_execution_cost.sum();hedge_cost+=nav.hedge_cost.sum();borrow+=nav.stock_borrow_cost.sum()
                nav=nav[['nav','cash_benchmark']]
                if book is None:book=nav.copy()
                else:
                    assert book.index.equals(nav.index)
                    book+=nav
            r=book.nav.pct_change().fillna(0);cash=book.cash_benchmark.pct_change().fillna(0)
            excess=r-cash
            pnl=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
            book.assign(return_=r,excess_return=excess,excess_pnl=pnl).to_csv(REPORT/f'{model}_beta_{band.replace("–","_")}_portfolio.csv')
            for period,mask in [('full',np.ones(len(book),bool)),('2018–2021',book.index.year<2022),('2022–2025',book.index.year>=2022)]:
                e=excess[mask]
                if period!='2022–2025':e=e.iloc[1:]
                results.append(dict(beta_range=band,model=model,period=period,pairs=len(members),
                    return_percent=100*((1+r[mask]).prod()-1),
                    cash_return_percent=100*((1+cash[mask]).prod()-1),
                    excess_pnl=pnl[mask].sum(),annualized_excess_sharpe=np.sqrt(252)*e.mean()/e.std() if e.std()>1e-12 else np.nan))
            for year,g in book.groupby(book.index.year):
                e=excess.loc[g.index]
                annual.append(dict(beta_range=band,model=model,year=year,
                    return_percent=100*((1+r.loc[g.index]).prod()-1),
                    excess_pnl=pnl.loc[g.index].sum(),
                    annualized_excess_sharpe=np.sqrt(252)*e.mean()/e.std() if e.std()>1e-12 else np.nan))
            s=status[status.pair.isin(members)&status.model.eq(model)]
            activity.append(dict(beta_range=band,model=model,entries=int(s.entries.sum()),active_pairs=int(s.entries.gt(0).sum()),
                gross_pnl=gross,option_execution_cost=option_cost,hedge_execution_cost=hedge_cost,stock_borrow_cost=borrow))
    results=pd.DataFrame(results);annual=pd.DataFrame(annual);activity=pd.DataFrame(activity)
    results.to_csv(REPORT/'beta_comparison.csv',index=False)
    annual.to_csv(REPORT/'beta_annual.csv',index=False)
    activity.to_csv(REPORT/'beta_activity.csv',index=False)
    lines=['# Tighter pairwise beta: retraining and backtest','',
        f'Both ranges use the same {len(members)} complete broad pairs, including cash-only sleeves. Each pair starts with $1m. Evaluation: January 2018–August 2025. Both models were retrained monthly under their respective eligibility screens. Costs, expiry settlement, risk sizing, profitability gates and feature set are unchanged.','',
        '| Model | Beta range | Annualized excess Sharpe | Net excess P&L | Entries | Active pairs |',
        '|---|---|---:|---:|---:|---:|']
    for row in results[results.period.eq('full')].itertuples():
        a=activity[activity.beta_range.eq(row.beta_range)&activity.model.eq(row.model)].iloc[0]
        lines.append(f'| {row.model} | {row.beta_range} | {row.annualized_excess_sharpe:.3f} | ${row.excess_pnl:,.0f} | {a.entries} | {a.active_pairs} |')
    lines+=['','## Gross profit and costs','',
        '| Model | Beta range | Gross P&L | Option execution | Hedge execution | Stock borrow |',
        '|---|---|---:|---:|---:|---:|']
    for row in activity.itertuples():
        lines.append(f'| {row.model} | {row.beta_range} | ${row.gross_pnl:,.0f} | ${row.option_execution_cost:,.0f} | ${row.hedge_execution_cost:,.0f} | ${row.stock_borrow_cost:,.0f} |')
    lines+=['','Gross includes option payoffs, stock hedging and hedge dividends. Net excess P&L additionally reflects execution, stock borrow and financing relative to cash.','',
        '## Annualized excess Sharpe by year','',
        '| Year | RL old range | RL tighter range | NN old range | NN tighter range |',
        '|---|---:|---:|---:|---:|']
    yearly=annual.pivot(index='year',columns=['model','beta_range'],values='annualized_excess_sharpe')
    for year,row in yearly.iterrows():
        values=[row[(model,band)] for model in ['rl_sized','nn_fixed'] for band in ROOTS]
        lines.append(f'| {year} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    audit=json.loads((REPORT/'audit.json').read_text())
    manifest=json.loads((REPORT/'manifest.json').read_text())
    lines+=['',f'Audit passed for all {audit["pairs_checked"]} attempted pairs: label/memory timing, action ladders, beta eligibility, entry commissions and expiry settlement. Fit timing and daily ledger checks also pass in the training runner. NN convergence warnings remain: {audit["nn_validation_warnings"]}/{audit["nn_validation_fits"]} validation fits and {audit["nn_recent_warnings"]}/{audit["nn_recent_fits"]} recent refits. Fits are retained and flagged, not silently removed.',
        f'Runtime: {manifest["elapsed_seconds"]/60:.1f} minutes with two workers; largest reported worker peak RSS {manifest["max_worker_peak_rss_bytes"]/1024**3:.2f} GiB.','',
        'Sharpe uses daily returns in excess of the cash benchmark, annualized with √252. Total percentage returns in the CSV include interest on unused capital; net excess P&L removes that benchmark. 2025 is partial. Different trade sets and sizing mean this measures the full retrained policy change, not only rejected entries.','',
        'The tighter threshold was chosen after inspecting these years, so this is an exploratory comparison, not an untouched holdout. Complete-history selection remains a limitation. The monthly validation gate still validates the proposal before its recent-data refit, rather than the exact deployed weights.','',
        '[Period comparison](expiry_trained_beta_085_115/beta_comparison.csv) · [Annual returns and Sharpes](expiry_trained_beta_085_115/beta_annual.csv) · [Activity](expiry_trained_beta_085_115/beta_activity.csv) · [Audit](expiry_trained_beta_085_115/audit.json)','']
    Path('reports/expiry_beta_comparison.md').write_text('\n'.join(lines))
    print(results.to_string(index=False));print(activity.to_string(index=False))


if __name__=='__main__':main()
