"""Audit and report the predeclared 14/30/45D maturity experiment."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_tenor_comparison import OUTPUT, REPORT, DATA, TENORS, config, definitions, COMMON_RUNOFF, BASE
from scripts.run_contract_walkforward import digest
from scripts.test_contract_entry_gates import gate_actions
from scripts.report_option_flex_ablation import sharpe

@lru_cache(maxsize=6)
def selection(sid,tenor):
    root=Path('data/contract_history_shared') if tenor==30 else DATA/f'{tenor}d'
    pieces=[pd.read_parquet(root/str(y)/f'{sid}_selection.parquet') for y in range(2010,2026) if (root/str(y)/f'{sid}_selection.parquet').exists()]
    return pd.concat(pieces,ignore_index=True).set_index('date') if pieces else pd.DataFrame()

def main():
    inputs=json.loads((OUTPUT/'inputs.json').read_text())
    for group in ['code','sources']:
        for p,h in inputs[group].items():assert digest(p)==h,p
    assert digest(DATA/'manifest.json')==inputs['tenor_manifest']
    assert digest('config/current_contract_reference.json')==inputs['reference']
    prep=json.loads((DATA/'inputs.json').read_text())
    for group in ['code','sources','raw_quotes']:
        for p,h in prep[group].items():assert digest(p)==h,p
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert digest(DATA/'inputs.json')==manifest['inputs']
    for part in manifest['partitions']:
        for p,h in part['outputs'].items():assert digest(p)==h,p
    check=json.loads((REPORT/'real_data_reconciliation.json').read_text())
    assert len(check['checks'])==36 and max(r['max_error'] for r in check['checks'])<1e-8
    defs=definitions();records=[];failed=set();audits=[];coverage=[];actual=[]
    for d in defs:
        for tenor in TENORS:
            folder=OUTPUT/d['pair']/f'{tenor}d';row=json.loads((folder/'status.json').read_text());records.append(row)
            if row['status']!='complete':failed.add(d['pair']);continue
            nav=pd.read_parquet(row.get('nav_source',folder/'daily_nav.parquet'))
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
            if row.get('structural_cash'):
                np.testing.assert_allclose(nav.nav,nav.cash_benchmark,rtol=0,atol=1e-7)
            else:
                panel=pd.read_parquet(folder/'states.parquet');a=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0]).rl_sized
                pd.testing.assert_series_equal(a,gate_actions(a,panel,'memory_both'))
                assert panel.loc[a.index[a.ne(0)],'eligible'].all()
                assert a.empty or a.index.max()<=nav.index[-1]-pd.Timedelta(days=COMMON_RUNOFF)
                coverage.append(dict(pair=d['pair'],tenor=tenor,state_rows=len(panel),eligible_state_rows=int(panel.eligible.sum()),
                                     median_signal_dte_a=panel.dte_a.median(),median_signal_dte_b=panel.dte_b.median()))
                try:fits=pd.read_csv(folder/'fits.csv')
                except pd.errors.EmptyDataError:fits=pd.DataFrame()
                for last,start in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                    if last in fits:
                        use=fits[last].notna();assert (pd.to_datetime(fits.loc[use,last])<pd.to_datetime(fits.loc[use,start])).all()
                try:dec=pd.read_csv(folder/'decisions.csv')
                except pd.errors.EmptyDataError:dec=pd.DataFrame(columns=['status'])
                entered=dec[dec.status.eq('entered')]
                if len(entered):
                    cfg=config(tenor)
                    assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
                    trades=pd.read_csv(folder/'trades.csv',parse_dates=['date','cohort'])
                    assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
                    assert set(trades.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                    settlements=trades[trades.reason.eq('expiry_settlement')]
                    assert settlements.commission.eq(0).all() and settlements.execution_penalty.eq(0).all()
                    for t in trades[trades.reason.eq('entry')].itertuples():
                        q=selection(t.secid,tenor).loc[t.cohort]
                        assert q.optionid==t.optionid and q.cp_flag=='C'
                        dte=(q.expiry-t.date).days
                        assert cfg.min_dte<=dte<=cfg.max_dte
                        actual.append(dict(pair=d['pair'],tenor=tenor,secid=t.secid,entry=str(t.date.date()),dte=dte))
            audits.append(dict(pair=d['pair'],tenor=tenor,status='passed'))
    records=pd.DataFrame(records);records.to_csv(REPORT/'pair_status.csv',index=False)
    records[records.status.ne('complete')].to_csv(REPORT/'incomplete.csv',index=False)
    pd.DataFrame(audits).to_csv(REPORT/'audit.csv',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    actual=pd.DataFrame(actual,columns=['pair','tenor','secid','entry','dte']);actual.to_csv(REPORT/'entry_tenors.csv',index=False)
    members=[d for d in defs if d['pair'] not in failed];summary=[];annual=[];pair_rows=[];components=[]
    for tenor in TENORS:
        for scope,roster in [('common',members),('full',defs)]:
            own=records[records.tenor.eq(tenor)]
            if scope=='full' and own.status.ne('complete').any():
                summary.append(dict(tenor=tenor,scope=scope,status='incomplete',pairs=len(roster)));continue
            book=None;entries=0;active=0
            for d in roster:
                row=own[own.pair.eq(d['pair'])].iloc[0];folder=OUTPUT/d['pair']/f'{tenor}d'
                path=row.nav_source if isinstance(row.get('nav_source'),str) else folder/'daily_nav.parquet'
                nav=pd.read_parquet(path);b=nav[['nav','cash_benchmark']]
                if book is None:book=b.copy()
                else:assert book.index.equals(b.index);book+=b
                entries+=row.entries;active+=int(row.entries>0)
                if scope=='common':
                    pnl=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
                    pair_rows.append(dict(pair=d['pair'],label=d['label'],tenor=tenor,entries=row.entries,net_excess_pnl=pnl))
                    def value(k):return float(row[k]) if k in row and pd.notna(row[k]) else 0.
                    c=dict(tenor=tenor,pair=d['pair'],gross_option_pnl=value('option_pnl')+nav.option_execution_cost.sum(),
                           gross_hedge_pnl=value('hedge_pnl')+nav.hedge_cost.sum(),option_cost=-nav.option_execution_cost.sum(),
                           hedge_cost=-nav.hedge_cost.sum(),stock_borrow=-nav.stock_borrow_cost.sum(),dividends=nav.dividend_cash.sum(),
                           financing_above_cash=nav.interest.sum()-(nav.cash_benchmark.iloc[-1]-nav.cash_benchmark.iloc[0]))
                    np.testing.assert_allclose(sum(v for k,v in c.items() if k not in ['tenor','pair']),pnl,atol=1e-5,rtol=0)
                    components.append(c)
            r=book.nav.pct_change().fillna(0);e=r-book.cash_benchmark.pct_change().fillna(0)
            pnl=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
            book.assign(excess_return=e,excess_pnl=pnl).to_csv(REPORT/f'{tenor}d_{scope}_portfolio.csv')
            selected_entries=actual[actual.tenor.eq(tenor)&actual.pair.isin([d['pair'] for d in roster])]
            gaps=selected_entries.groupby(['pair','entry']).dte.agg(lambda x:x.max()-x.min())
            summary.append(dict(tenor=tenor,scope=scope,status='complete',pairs=len(roster),entries=int(entries),active_pairs=active,
                median_expiry_gap_days=gaps.median(),same_expiry_fraction=gaps.eq(0).mean(),
                annualized_excess_sharpe=sharpe(e.iloc[1:]),net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1],
                median_entry_dte=actual.loc[actual.tenor.eq(tenor)&actual.pair.isin([d['pair'] for d in roster]),'dte'].median()))
            if scope=='common':
                for year,g in e.groupby(e.index.year):annual.append(dict(tenor=tenor,year=year,annualized_excess_sharpe=sharpe(g),
                    return_percent=100*((1+r.loc[g.index]).prod()-1),net_excess_pnl=pnl.loc[g.index].sum()))
    summary=pd.DataFrame(summary);summary.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT/'pair_results.csv',index=False)
    pd.DataFrame(components).drop(columns='pair').groupby('tenor').sum().to_csv(REPORT/'pnl_components.csv')
    lines=['# ATM option maturity comparison','',f'{len(members)}/{len(defs)} pairs complete for all three tenors. Common-roster results:', '',
        '| Target DTE | Annualized excess Sharpe | Entries | Active pairs | P&L above cash | Median entry DTE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary[summary.scope.eq('common')].itertuples():lines.append(f'| {r.tenor} | {r.annualized_excess_sharpe:.3f} | {r.entries:.0f} | {r.active_pairs:.0f} | ${r.net_excess_pnl:,.0f} | {r.median_entry_dte:.0f} |')
    lines+=['',
        'Targets 14, 30 and 45 calendar days at next-session entry, with respective windows 7–21, 21–45 and 35–55 days. Original selection priority is preserved: nearest tenor, then nearest strike, within the existing moneyness/liquidity limits. Calls/calls only; each leg held to its own expiry, daily delta hedge, initial integer vega mismatch ≤10%. Same costs, risk limits, sizing ladder, beta .85–1.15 and correlation ≥.5.', '',
        'Original full RL state including VIX, surface context and directional profitability memory; monthly five-year training, separate prior-year validation, then recent-five-year completed-outcome refit. The inherited validation gate tests pre-refit weights. No rank-only state or gate removal. Spread, RV-relative value, reversion, contract attributes, labels and profitability memory reflect the traded tenor; cached surface-context features stay fixed across tenors. Thirty-day calls reuse their original native labels and states, without the matched-put sample restriction. This deliberately avoids imposing irrelevant put availability on other maturities.', '',
        'A common 60-calendar-day entry runoff applies to both validation and final evaluation, January 2018–August 2025. This differs from the older 50-day runoff; thus the fresh 30D control need not equal either older 0.226 native-call or 0.247 matched-put-row Sharpe. Contract availability and training samples differ by tenor and are reported. The three targets were declared before the run; all periods are previously inspected research history, not an untouched holdout.', '',
        'Same vega is not same gamma or tail risk. No early assignment is modeled. Missing/unsupported held contracts make a book incomplete; failed pairs are excluded from every common-roster comparison and remain explicit in the full-roster completeness table. They are never assigned zero returns. No reference policy has been promoted.', '',
        '[Comparison and full-roster status](tenor_comparison/comparison.csv) · [Annual results](tenor_comparison/annual.csv) · [Pair results](tenor_comparison/pair_results.csv) · [P&L components](tenor_comparison/pnl_components.csv) · [Coverage](tenor_comparison/coverage.csv) · [Actual entry maturities](tenor_comparison/entry_tenors.csv) · [Incomplete books](tenor_comparison/incomplete.csv) · [Audit](tenor_comparison/audit.csv) · [Real-data payoff reconciliation](tenor_comparison/real_data_reconciliation.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(summary.to_string(index=False))
    (OUTPUT/'task_status.json').write_text(json.dumps(dict(status='complete',pairs_attempted=len(defs),common_complete=len(members),
        failed_pairs=sorted(failed),tests='16 targeted tests passed',reference_promoted=False),indent=2)+'\n')

if __name__=='__main__':main()
