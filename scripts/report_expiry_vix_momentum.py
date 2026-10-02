"""Audit macro-feature ablations and compare all variants on one roster."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_expiry_vix_momentum import BASE,OUTPUT,REPORT,VARIANTS,add_states
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import asset_paths


def main():
    inputs=json.loads((OUTPUT/'inputs.json').read_text())
    for p,h in inputs['code'].items():assert digest(p)==h,p
    for p,h in inputs['base_files'].items():assert digest(p)==h,p
    assert digest('data/raw/VIX_History.csv')==inputs['vix']
    definitions=json.loads((BASE/'inputs.json').read_text())['definitions']
    statuses=pd.read_csv(REPORT/'status.csv')
    assert len(statuses)==len(definitions)*len(VARIANTS)*2
    baseline_status=pd.read_csv('reports/expiry_trained_beta_085_115/status.csv')
    failed=set(statuses.loc[statuses.status.ne('complete'),'pair'])|set(baseline_status.loc[baseline_status.status.ne('complete'),'pair'])
    previous=json.loads(Path('reports/expiry_trained_beta_085_115/matched_pairs.json').read_text())
    members=sorted(set(previous)-failed)
    (REPORT/'matched_pairs.json').write_text(json.dumps(members,indent=2)+'\n')
    fits=[];coverage=[]
    vix=pd.read_csv('data/raw/VIX_History.csv',parse_dates=['DATE'])
    for d in definitions:
        base=pd.read_parquet(BASE/d['pair']/'states.parquet')
        momentum=[pd.read_parquet(OUTPUT/'momentum'/p.name).momentum for p in asset_paths(d)]
        expected=add_states(base,vix,momentum)
        coverage.append(dict(pair=d['pair'],rows=len(base),
            both_momentum_available=int((expected.momentum_12_1_a_available*expected.momentum_12_1_b_available).sum())))
        for variant in VARIANTS:
            folder=OUTPUT/variant/d['pair'];panel=pd.read_parquet(folder/'states.parquet')
            pd.testing.assert_frame_equal(panel,expected)
            actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
            assert np.isin(actions.rl_sized,np.arange(-1,1.01,.25)).all()
            assert np.isin(actions.nn_fixed,[-1,0,1]).all()
            for model in ['rl_sized','nn_fixed']:
                assert panel.loc[actions.index[actions[model].ne(0)],'eligible'].all()
            try:f=pd.read_csv(folder/'fits.csv')
            except pd.errors.EmptyDataError:continue
            for end,start in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                if end in f:
                    ok=f[end].notna();assert (pd.to_datetime(f.loc[ok,end])<pd.to_datetime(f.loc[ok,start])).all()
            f['pair']=d['pair'];f['variant']=variant;fits.append(f)
    fits=pd.concat(fits,ignore_index=True);fits.to_csv(REPORT/'fits.csv',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'feature_coverage.csv',index=False)
    nn=fits[fits.model.eq('nn_fixed')]
    audit=dict(pairs_checked=len(definitions),matched_pairs=len(members),
        nn_validation_fits=int(nn.converged.notna().sum()),nn_validation_warnings=int(nn.converged.eq(False).sum()),
        nn_recent_fits=int(nn.recent_converged.notna().sum()),nn_recent_warnings=int(nn.recent_converged.eq(False).sum()),
        momentum_coverage=sum(x['both_momentum_available'] for x in coverage)/sum(x['rows'] for x in coverage))
    (REPORT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    rows=[];annual=[];activity=[];pair_rows=[]
    roots={'baseline':BASE,**{v:OUTPUT/v for v in VARIANTS}}
    for variant,root in roots.items():
        for model in ['rl_sized','nn_fixed']:
            book=None;gross=execution=borrow=0.
            for pair in members:
                nav=pd.read_parquet(root/pair/model/'daily_nav.parquet')
                g=(nav.pnl-nav.interest+nav.stock_borrow_cost+nav.option_execution_cost+nav.hedge_cost).sum()
                gross+=g;execution+=(nav.option_execution_cost+nav.hedge_cost).sum();borrow+=nav.stock_borrow_cost.sum()
                pair_rows.append(dict(variant=variant,model=model,pair=pair,gross_pnl=g,
                    net_excess_pnl=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]))
                own=nav[['nav','cash_benchmark']]
                if book is None:book=own.copy()
                else:
                    assert book.index.equals(own.index);book+=own
            r=book.nav.pct_change().fillna(0);cash=book.cash_benchmark.pct_change().fillna(0);excess=r-cash
            pnl=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
            book.assign(return_=r,excess_return=excess,excess_pnl=pnl).to_csv(REPORT/f'{variant}_{model}_portfolio.csv')
            for period,mask in [('full',np.ones(len(book),bool)),('2018–2021',book.index.year<2022),('2022–2025',book.index.year>=2022)]:
                e=excess[mask]
                if period!='2022–2025':e=e.iloc[1:]
                rows.append(dict(variant=variant,model=model,period=period,pairs=len(members),
                    return_percent=100*((1+r[mask]).prod()-1),cash_return_percent=100*((1+cash[mask]).prod()-1),
                    excess_pnl=pnl[mask].sum(),annualized_excess_sharpe=np.sqrt(252)*e.mean()/e.std() if e.std()>1e-12 else np.nan))
            for year,g in book.groupby(book.index.year):
                e=excess.loc[g.index]
                annual.append(dict(variant=variant,model=model,year=year,
                    return_percent=100*((1+r.loc[g.index]).prod()-1),excess_pnl=pnl.loc[g.index].sum(),
                    annualized_excess_sharpe=np.sqrt(252)*e.mean()/e.std() if e.std()>1e-12 else np.nan))
            status=baseline_status if variant=='baseline' else statuses[statuses.variant.eq(variant)]
            active=status[status.pair.isin(members)&status.model.eq(model)]
            activity.append(dict(variant=variant,model=model,entries=int(active.entries.sum()),active_pairs=int(active.entries.gt(0).sum()),
                gross_pnl=gross,execution_cost=execution,stock_borrow_cost=borrow))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    yearly=pd.DataFrame(annual);yearly.to_csv(REPORT/'annual.csv',index=False)
    activity=pd.DataFrame(activity);activity.to_csv(REPORT/'activity.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT/'pair_attribution.csv',index=False)
    lines=['# VIX and momentum: expiry-policy ablation','',
        f'Comparison on {len(members)} common complete pairs, January 2018–August 2025. Each pair retains its $1m capital sleeve, including inactive sleeves. All variants use beta 0.85–1.15 and return correlation ≥0.5. Monthly training, validation, recent refits, profitability gates, costs and actual expiry accounting are unchanged. Both RL sizing and fixed-size NN are retrained for each feature set.','',
        'VIX is the strictly previous available close, at most seven calendar days stale. Momentum is each leg’s compounded CRSP total return from 252 trading days ago to 21 days ago (231 returns), in percentage points, respecting security-link intervals. Missing momentum is zero with a separate availability flag; no trading dates are dropped. No extra momentum-difference feature was added.','',
        '| Features added | RL annualized excess Sharpe | NN annualized excess Sharpe |',
        '|---|---:|---:|']
    table=comparison[comparison.period.eq('full')].pivot(index='variant',columns='model',values='annualized_excess_sharpe')
    for variant in roots:
        row=table.loc[variant];lines.append(f'| {variant} | {row.rl_sized:.3f} | {row.nn_fixed:.3f} |')
    lines+=['','## Annualized excess Sharpe by year','',
        '| Year | RL baseline | RL VIX | RL momentum | RL both | NN baseline | NN VIX | NN momentum | NN both |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    year_table=yearly.pivot(index='year',columns=['model','variant'],values='annualized_excess_sharpe')
    for year,row in year_table.iterrows():
        values=[row[(m,v)] for m in ['rl_sized','nn_fixed'] for v in roots]
        lines.append(f'| {year} | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    lines+=['','## Activity and gross edge','',
        '| Variant | Model | Entries | Active pairs | Gross P&L | Execution costs |',
        '|---|---|---:|---:|---:|---:|']
    for row in activity.itertuples():
        lines.append(f'| {row.variant} | {row.model} | {row.entries} | {row.active_pairs} | ${row.gross_pnl:,.0f} | ${row.execution_cost:,.0f} |')
    lines+=['','Daily cash-excess Sharpe is annualized using √252. Yearly percentage returns in the CSV include interest on unused capital; excess P&L removes the cash benchmark. 2025 ends in August. Gross P&L excludes execution, stock borrow and financing.','',
        'All periods have been inspected before; these are exploratory feature comparisons, not an untouched holdout. Overlapping trades and shared names are dependent. The validation gate still validates the earlier fitted proposal, not the exact recent-refit weights. Incomplete backtests are disclosed in status and excluded consistently from all variants.','',
        f'Audit: {audit["pairs_checked"]} pairs checked; original states, causal feature alignment, entry eligibility, action ladders and completed-outcome training cutoffs verified. Both momentum inputs available on {audit["momentum_coverage"]:.2%} of state rows. NN warnings: {audit["nn_validation_warnings"]}/{audit["nn_validation_fits"]} validation fits and {audit["nn_recent_warnings"]}/{audit["nn_recent_fits"]} recent fits. Daily ledger reconciliation runs inside every completed simulation.','',
        '[Period comparison](expiry_vix_momentum/comparison.csv) · [Annual percentage returns and Sharpes](expiry_vix_momentum/annual.csv) · [Activity](expiry_vix_momentum/activity.csv) · [Pair attribution](expiry_vix_momentum/pair_attribution.csv) · [Coverage and errors](expiry_vix_momentum/status.csv) · [Audit](expiry_vix_momentum/audit.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison.to_string(index=False));print(activity.to_string(index=False));print(audit)


if __name__=='__main__':main()
