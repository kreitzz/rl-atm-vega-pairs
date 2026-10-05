"""Audit the fixed two-variant smile experiment and compare with its exact 30D control."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_rich_smile import OUTPUT,REPORT,BASE,DATA,VARIANTS,definitions,asset,augment_smile,ROOT,COMMON_RUNOFF
from scripts.build_rich_smile import VALUES
from scripts.select_atm_contracts import load_calendar
from scripts.run_contract_walkforward import digest
from scripts.test_contract_entry_gates import gate_actions
from scripts.run_aligned_contract_learning import screen_dates
from scripts.report_option_flex_ablation import sharpe


def main():
    inputs=json.loads((OUTPUT/'inputs.json').read_text())
    for group in ['code','base_files']:
        for p,h in inputs[group].items():assert digest(p)==h,p
    assert digest(BASE/'inputs.json')==inputs['base']
    assert digest(DATA/'manifest.json')==inputs['smile']
    assert digest('config/current_contract_reference.json')==inputs['reference']
    old=json.loads((BASE/'inputs.json').read_text())
    for group in ['code','sources']:
        for p,h in old[group].items():assert digest(p)==h,p
    meta=json.loads((DATA/'manifest.json').read_text());assert digest(DATA/'inputs.json')==meta['inputs']
    for p,h in meta['assets'].items():assert digest(p)==h,p
    prep=json.loads((DATA/'inputs.json').read_text())
    for group in ['code','sources']:
        for p,h in prep[group].items():assert digest(p)==h,p
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    defs=definitions();records=[];failed=set();audits=[];coverage=[]
    for d in defs:
        basefolder=BASE/d['pair']/'30d';base=json.loads((basefolder/'status.json').read_text());assert base['status']=='complete'
        records.append(dict(base,variant='control'))
        if not base.get('structural_cash'):
            original=pd.read_parquet(basefolder/'states.parquet')
            panel,levels,changes=augment_smile(original,[asset(sid) for sid in d['secids']],cal)
            use=(panel.index>='2018-01-01')&panel.eligible&screen_dates(d,panel.index)
            for feature in levels+changes:
                if feature.endswith('_available'):
                    coverage.append(dict(pair=d['pair'],feature=feature,eligible_rows=int(use.sum()),available=int(panel.loc[use,feature].sum())))
        for variant in VARIANTS:
            folder=OUTPUT/d['pair']/variant;row=json.loads((folder/'status.json').read_text());records.append(row)
            if row['status']!='complete':failed.add(d['pair']);continue
            nav=pd.read_parquet(row.get('nav_source',folder/'daily_nav.parquet'))
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
            if row.get('structural_cash'):
                np.testing.assert_allclose(nav.nav,nav.cash_benchmark,atol=1e-7,rtol=0)
            else:
                a=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0]).rl_sized
                pd.testing.assert_series_equal(a,gate_actions(a,panel,'memory_both'))
                assert panel.loc[a.index[a.ne(0)],'eligible'].all()
                assert a.empty or a.index.max()<=cal[-1]-pd.Timedelta(days=COMMON_RUNOFF)
                try:f=pd.read_csv(folder/'fits.csv')
                except pd.errors.EmptyDataError:f=pd.DataFrame()
                for last,start in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                    if last in f:
                        use=f[last].notna();assert (pd.to_datetime(f.loc[use,last])<pd.to_datetime(f.loc[use,start])).all()
                try:dec=pd.read_csv(folder/'decisions.csv')
                except pd.errors.EmptyDataError:dec=pd.DataFrame(columns=['status'])
                entered=dec[dec.status.eq('entered')]
                if len(entered):
                    assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(.1+1e-12).all()
                    t=pd.read_csv(folder/'trades.csv');assert t.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
                    assert set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                    ex=t[t.reason.eq('expiry_settlement')];assert ex.commission.eq(0).all() and ex.execution_penalty.eq(0).all()
            audits.append(dict(pair=d['pair'],variant=variant,status='passed'))
    status=pd.DataFrame(records);status.to_csv(REPORT/'pair_status.csv',index=False)
    status[status.status.ne('complete')].to_csv(REPORT/'incomplete.csv',index=False)
    pd.DataFrame(audits).to_csv(REPORT/'audit.csv',index=False)
    c=pd.DataFrame(coverage);c.to_csv(REPORT/'coverage_by_pair.csv',index=False)
    c=c.drop(columns='pair').groupby('feature').sum();c['available_fraction']=c.available/c.eligible_rows;c.to_csv(REPORT/'coverage.csv')
    members=[d for d in defs if d['pair'] not in failed];summary=[];annual=[];pairs=[];components=[]
    for variant in ['control']+VARIANTS:
        own=status[status.variant.eq(variant)];book=None;entries=0;active=0
        for d in members:
            row=own[own.pair.eq(d['pair'])].iloc[0]
            folder=BASE/d['pair']/'30d' if variant=='control' else OUTPUT/d['pair']/variant
            path=row.nav_source if isinstance(row.get('nav_source'),str) else folder/'daily_nav.parquet'
            nav=pd.read_parquet(path);b=nav[['nav','cash_benchmark']]
            if book is None:book=b.copy()
            else:assert book.index.equals(b.index);book+=b
            entries+=row.entries;active+=int(row.entries>0)
            pnl=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
            pairs.append(dict(pair=d['pair'],label=d['label'],variant=variant,entries=row.entries,net_excess_pnl=pnl))
            def value(k):return float(row[k]) if k in row and pd.notna(row[k]) else 0.
            component=dict(variant=variant,pair=d['pair'],gross_option_pnl=value('option_pnl')+nav.option_execution_cost.sum(),
                gross_hedge_pnl=value('hedge_pnl')+nav.hedge_cost.sum(),option_cost=-nav.option_execution_cost.sum(),hedge_cost=-nav.hedge_cost.sum(),
                stock_borrow=-nav.stock_borrow_cost.sum(),dividends=nav.dividend_cash.sum(),
                financing_above_cash=nav.interest.sum()-(nav.cash_benchmark.iloc[-1]-nav.cash_benchmark.iloc[0]))
            np.testing.assert_allclose(sum(v for k,v in component.items() if k not in ['variant','pair']),pnl,rtol=0,atol=1e-5)
            components.append(component)
        r=book.nav.pct_change().fillna(0);e=r-book.cash_benchmark.pct_change().fillna(0)
        pnl=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
        book.assign(excess_return=e,excess_pnl=pnl).to_csv(REPORT/f'{variant}_portfolio.csv')
        if variant=='control' and not failed:
            prior=pd.read_csv('reports/tenor_comparison/30d_full_portfolio.csv',index_col=0,parse_dates=[0])
            np.testing.assert_allclose(book.nav,prior.nav,atol=1e-7,rtol=0)
        summary.append(dict(variant=variant,pairs=len(members),annualized_excess_sharpe=sharpe(e.iloc[1:]),entries=int(entries),active_pairs=active,
                            net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]))
        for year,g in e.groupby(e.index.year):annual.append(dict(variant=variant,year=year,annualized_excess_sharpe=sharpe(g),net_excess_pnl=pnl.loc[g.index].sum()))
    result=pd.DataFrame(summary);result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pair_results.csv',index=False)
    pd.DataFrame(components).drop(columns='pair').groupby('variant').sum().to_csv(REPORT/'pnl_components.csv')
    lines=['# Richer smile information: fixed 30D policy experiment','',f'{len(members)}/{len(defs)} pairs common to all completed books, January 2018–August 2025.', '',
        '| State | Annualized excess Sharpe | Entries | Active pairs | P&L above cash |','|---|---:|---:|---:|---:|']
    for r in result.itertuples():lines.append(f'| {r.variant} | {r.annualized_excess_sharpe:.3f} | {r.entries} | {r.active_pairs} | ${r.net_excess_pnl:,.0f} |')
    lines+=['',
        'Two predeclared additions to the full reference state: (1) richer smile levels; (2) those levels plus 5- and 21-session changes. Each preserves the original features, same 30D call selections, native call labels, full profitability gate, monthly refit, sizing ladder, beta/correlation screen, costs, risk limits and common 60-day runoff. The exact control is the preceding maturity experiment’s 30D policy. Nothing was tuned after viewing these new outcomes.', '',
        'For each underlying/date, fit eligible cached calls at the selected ATM contract’s exact expiry. Interpolate IV at BS call deltas 0.10, 0.25, 0.50, 0.75 and 0.90 only between observed fitted points whose delta gap is ≤0.15. No extrapolation. Standard contracts, positive bid, open interest ≥100 and relative quoted width ≤25%; rejected or missing points remain unavailable.', '',
        'Eight normalized shape descriptors: 10- and 25-delta downside/upside wing premiums relative to the selected ATM IV; 25-delta call-minus-put-equivalent risk reversal; 25-delta butterfly versus 50-delta IV; and downside/upside 10-versus-25-delta tail slopes. All are divided by current selected ATM IV. The downside measures are inferred from ITM calls under the existing European BS/dividend convention, not observed OTM put IVs. Each descriptor enters for both legs and their difference, with explicit availability flags. Changes use exact 5/21 trading-session lags on the market calendar; a missing endpoint stays unavailable. Zero-filled missing values always have a zero availability flag.', '',
        'The original depth-limited tree/CEM learner and search budget stay fixed. The extra inputs do not guarantee better partitions. Training and memory use only past completed outcomes. The inherited validation gate evaluates pre-refit weights, not the exact deployed refit. All dates have already been inspected in research; this is an adaptive comparison, not an untouched holdout. Initial vega matching can drift, each leg settles at its own expiry, and American early assignment is still omitted.', '',
        'Incomplete books, if any, are disclosed and their pairs excluded from every common comparison, including the control. No incomplete book is treated as cash. No reference policy is promoted automatically.', '',
        '[Annual results](rich_smile/annual.csv) · [Pair results](rich_smile/pair_results.csv) · [Feature coverage](rich_smile/coverage.csv) · [P&L components](rich_smile/pnl_components.csv) · [Audit](rich_smile/audit.csv) · [Incomplete books](rich_smile/incomplete.csv)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(result.to_string(index=False))
    (OUTPUT/'task_status.json').write_text(json.dumps(dict(status='complete',pairs=len(defs),common_complete=len(members),failed_pairs=sorted(failed),
        tests='4 feature tests passed; unchanged engine previously tested',reference_promoted=False),indent=2)+'\n')

if __name__=='__main__':main()
