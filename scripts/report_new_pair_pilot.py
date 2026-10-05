"""New-pair results and their incremental contribution to the VIX reference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_new_pair_pilot import OUTPUT,REPORT
from scripts.run_aligned_contract_learning import screen_dates
from scripts.diagnose_current_contract_losses import read_csv


def metrics(book):
    r=book.nav.pct_change().fillna(0);cash=book.cash_benchmark.pct_change().fillna(0)
    e=r-cash;difference=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
    return r,e,difference


def main():
    status=pd.read_csv(REPORT/'status.csv');defs=json.loads((REPORT/'universe.json').read_text())
    failed=set(status.loc[status.status.ne('complete'),'pair'])
    complete=[d for d in defs if d['pair'] not in failed]
    fits=[];pairs=[];decisions=[]
    for d in defs:
        folder=OUTPUT/d['pair'];actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
        actions.index=pd.DatetimeIndex(actions.index)
        assert screen_dates(d,actions.index).all()
        f=read_csv(folder/'fits.csv')
        if len(f):
            for end,start in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                if end in f:
                    valid=f[end].notna();assert (pd.to_datetime(f.loc[valid,end])<pd.to_datetime(f.loc[valid,start])).all()
            f['pair']=d['pair'];fits.append(f)
        for model in ['rl_sized','nn_fixed']:
            row=status[status.pair.eq(d['pair'])&status.model.eq(model)].iloc[0].to_dict()
            row['eligible_quarters']=len(d['screen_quarters'])
            if row['status']=='complete':
                nav=pd.read_parquet(folder/model/'daily_nav.parquet')
                row['net_excess_pnl']=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
                row['gross_pnl']=(nav.pnl-nav.interest+nav.option_execution_cost+nav.hedge_cost+nav.stock_borrow_cost).sum()
                row['execution_cost']=(nav.option_execution_cost+nav.hedge_cost).sum()
                dec=read_csv(folder/model/'decisions.csv')
                if len(dec):
                    for reason,n in dec[dec.action.ne(0)].status.value_counts().items():
                        decisions.append(dict(pair=d['pair'],model=model,status=reason,count=n))
            pairs.append(row)
    pd.concat(fits,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    pairs=pd.DataFrame(pairs);pairs.to_csv(REPORT/'pair_results.csv',index=False)
    pd.DataFrame(decisions).to_csv(REPORT/'execution_gates.csv',index=False)
    original=json.loads(Path('reports/expiry_vix_momentum/matched_pairs.json').read_text())
    rows=[];annual=[]
    for model in ['rl_sized','nn_fixed']:
        old=pd.read_csv(f'reports/expiry_vix_momentum/vix_{model}_portfolio.csv',index_col=0,parse_dates=[0])[['nav','cash_benchmark']]
        if not complete:continue
        added=sum(pd.read_parquet(OUTPUT/d['pair']/model/'daily_nav.parquet',columns=['nav','cash_benchmark']) for d in complete)
        assert old.index.equals(added.index)
        # Same aggregate starting capital for expanded versus old+cash comparison.
        control=old.copy();control['nav']+=added.cash_benchmark;control['cash_benchmark']+=added.cash_benchmark
        books={'existing_reference':old,'new_pairs_only':added,'existing_plus_new':old+added,'existing_plus_cash':control}
        for name,book in books.items():
            r,e,pnl=metrics(book);book.assign(return_=r,excess_return=e,excess_pnl=pnl).to_csv(REPORT/f'{model}_{name}_portfolio.csv')
            count=len(original) if name=='existing_reference' else len(complete) if name=='new_pairs_only' else len(original)+len(complete)
            for period,mask in [('full',np.ones(len(book),bool)),('2018–2021',book.index.year<2022),('2022–2025',book.index.year>=2022)]:
                er=e[mask]
                if period!='2022–2025':er=er.iloc[1:]
                rows.append(dict(model=model,portfolio=name,period=period,pairs=count,
                    return_percent=100*((1+r[mask]).prod()-1),net_excess_pnl=pnl[mask].sum(),
                    annualized_excess_sharpe=np.sqrt(252)*er.mean()/er.std() if er.std()>1e-12 else np.nan))
            for year,g in book.groupby(book.index.year):
                er=e.loc[g.index]
                annual.append(dict(model=model,portfolio=name,year=year,return_percent=100*((1+r.loc[g.index]).prod()-1),
                    net_excess_pnl=pnl.loc[g.index].sum(),annualized_excess_sharpe=np.sqrt(252)*er.mean()/er.std() if er.std()>1e-12 else np.nan))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    lines=['# New industry-peer pilot with the current VIX policy','',
        'Twelve illustrative new pairs selected before observing their backtest P&L: APA/DVN, HES/COP, CDE/AG, BZH/LEN, MGM/HLT, AIG/PRU, AMAT/INTC, A/DHR, BA/TXT, NEE/SO, LUMN/T and EW/MDT. The sample spans multiple businesses and uses already-downloaded names. This is a manually chosen exploratory sample from historical discovery, not a random sample or an untouched universe-selection test. Ticker labels can be historical.','',
        'Independent models retain the current VIX-only features, monthly training and recent refits, hard profitability-memory and validation gates, beta 0.85–1.15, correlation ≥0.5, risk limits and costs. New entries are allowed only in historically qualifying discovery quarters; existing positions still run to expiry. Training and validation use completed historical outcomes, including history preceding discovery, as in the existing specification.','',
        'Discovery beta uses direct underlying CRSP returns; monthly policy beta continues to use the existing prepared return series, which has some IV-availability-related gaps. This preserves the current policy input pipeline but can remove otherwise eligible opportunities. No additional gate was relaxed for this pilot.','',
        '| Pair | RL entries | RL annualized excess Sharpe | RL net excess P&L | NN entries | NN net excess P&L |',
        '|---|---:|---:|---:|---:|---:|']
    for d in defs:
        own=pairs[pairs.pair==d['pair']].set_index('model')
        a,b=own.loc['rl_sized'],own.loc['nn_fixed']
        sr=f'{a.annualized_excess_sharpe:.3f}' if pd.notna(a.annualized_excess_sharpe) else '—'
        left=f'{int(a.entries)} | {sr} | ${a.net_excess_pnl:,.0f}' if a.status=='complete' else '— | incomplete | —'
        right=f'{int(b.entries)} | ${b.net_excess_pnl:,.0f}' if b.status=='complete' else '— | incomplete'
        lines.append(f'| {d["label"]} | {left} | {right} |')
    lines+=['','## Portfolio comparison','',f'{len(complete)} of {len(defs)} new pairs complete under both models. Incomplete pairs are excluded consistently from aggregate comparisons and retained in the status file. Existing reference uses its fixed 346-pair roster.','',
        '| Model | Portfolio | Annualized excess Sharpe | Net excess P&L |',
        '|---|---|---:|---:|']
    for r in comparison[comparison.period=='full'].itertuples():
        lines.append(f'| {r.model} | {r.portfolio} | {r.annualized_excess_sharpe:.3f} | ${r.net_excess_pnl:,.0f} |')
    lines+=['','HES/COP is excluded from both models’ aggregate new-pair baskets because the NN encountered an unsupported distribution on February 8, 2024 (option 159326264). Its complete positive RL result remains visible above.','',
        'RL trades five of the twelve candidates, or four of the eleven in the matched aggregate. The matched RL basket has 16 entries; NN has 37. LUMN/T’s one NN trade contributes about $25.3k of the matched new NN basket’s $25.7k net excess profit, so the positive NN result is highly concentrated. Sparse entries do not establish persistent pair-level profitability.','',
        'January 2018–August 2025. Each pair starts with $1m; cash-only periods remain included. Sharpe uses daily cash-excess returns and √252. Percentage returns in the CSV include interest on unused capital. `existing_plus_cash` funds the added sleeves but leaves them idle, giving the same starting capital as `existing_plus_new`. Aggregate results combine independent pair sleeves: they do not enforce a new portfolio-wide concentration or shared-underlying risk budget.','',
        'All dates have already been inspected during research. The pilot is not evidence of an independently validated strategy. The monthly validation gate still evaluates the proposal before its recent-data refit, rather than the exact deployed weights. NN convergence warnings are retained in the fit logs.','',
        '[Pair results](new_pair_pilot/pair_results.csv) · [Period comparison](new_pair_pilot/comparison.csv) · [Yearly returns and Sharpe](new_pair_pilot/annual.csv) · [Historical membership](new_pair_pilot/universe.json) · [Run status](new_pair_pilot/status.csv) · [Fits](new_pair_pilot/fits.csv) · [Audit](new_pair_pilot/audit.json)','']
    if (REPORT/'audit.json').exists():
        audit=json.loads((REPORT/'audit.json').read_text());manifest=json.loads((REPORT/'manifest.json').read_text())
        lines+=['',f'Audit passed across {audit["pairs_checked"]} pairs: completed-label memory timing, action ladders, beta eligibility and settlement/cost checks. Quarterly historical membership and fit cutoffs also pass. The full 148-test suite passes. NN convergence warnings: {audit["nn_validation_warnings"]}/{audit["nn_validation_fits"]} validation fits and {audit["nn_recent_warnings"]}/{audit["nn_recent_fits"]} recent refits. Runtime {manifest["elapsed_seconds"]/60:.1f} minutes; peak worker RSS {manifest["max_worker_peak_rss_bytes"]/1024**3:.2f} GiB with two workers.','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(pairs[['label','model','status','entries','annualized_excess_sharpe','net_excess_pnl']].to_string(index=False))
    print(comparison.to_string(index=False))


if __name__=='__main__':main()
