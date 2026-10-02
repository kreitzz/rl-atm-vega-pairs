"""Audit every sleeve and compare expanded contract choice on a common roster."""
import json
import numpy as np
import pandas as pd
from scripts.run_curve_trading import DATA,OUTPUT,REPORT,BASE,VARIANTS,definitions,config,COMBOS,ACTIONS,RUNOFF
from scripts.curve_trading.policy import ATM,gate
from scripts.run_contract_walkforward import digest
from scripts.report_option_flex_ablation import sharpe
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT

def main():
    inputs=json.loads((OUTPUT/'inputs.json').read_text())
    for group in ['code','sources']:
        for path,h in inputs[group].items():assert digest(path)==h,path
    for path,h in json.loads((OUTPUT/'runtime_dependencies.json').read_text()).items():assert digest(path)==h,path
    for cp in ['C','P']:
        for stage in ['select','history']:
            manifest=json.loads((DATA/f'{cp}_{stage}_manifest.json').read_text())
            assert manifest['inputs']==digest(DATA/'preparation_inputs.json')
            assert len(manifest['partitions'])==256
            for partition in manifest['partitions']:
                for path,h in partition['files'].items():assert digest(path)==h,path
    reconciliation=json.loads((REPORT/'real_data_reconciliation.json').read_text())
    assert len(reconciliation['checks'])==18 and sum(r['complete'] for r in reconciliation['checks'])>=36
    assert max(r['max_label_error'] for r in reconciliation['checks'] if r['max_label_error'] is not None)<1e-8
    assert len(reconciliation['frozen_controls'])==3 and max(r['max_nav_error'] for r in reconciliation['frozen_controls'])<1e-5
    defs=definitions();records=[];failed=set();mix=[];audits=[];cfg=config()
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    for d in defs:
        for variant in VARIANTS:
            folder=OUTPUT/d['pair']/variant;path=folder/'status.json'
            if not path.exists():raise ValueError(f'Unfinished pair: {path}')
            row=json.loads(path.read_text());records.append(row)
            if row['status']!='complete':failed.add(d['pair']);continue
            nav=pd.read_parquet(row.get('nav_source',folder/'daily_nav.parquet'))
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
            if row.get('structural_cash'):
                np.testing.assert_allclose(nav.nav,nav.cash_benchmark,atol=1e-7,rtol=0)
            else:
                panel=pd.read_parquet(folder.parent/'states.parquet')
                a=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0]).code.astype(int)
                with np.load(folder.parent/'gates.npz') as z:
                    np.testing.assert_array_equal(z['dates'],panel.index.values)
                    pd.testing.assert_series_equal(a,gate(a,a.index,panel,z['gates'],z['available']),check_names=False)
                    control_path=folder.parent/'gates_control.npz'
                    if control_path.exists():
                        with np.load(control_path) as control:
                            for key in ['gates','available','ends']:np.testing.assert_array_equal(z[key][ATM],control[key][ATM])
                assert a.empty or a.index.max()<=nav.index[-1]-pd.Timedelta(days=RUNOFF)
                try:fits=pd.read_csv(folder/'fits.csv')
                except pd.errors.EmptyDataError:fits=pd.DataFrame()
                for outcome,bound in [('last_training_outcome','validation_start'),('last_recent_outcome','fit')]:
                    if outcome in fits:
                        good=fits[outcome].notna();assert (pd.to_datetime(fits.loc[good,outcome])<pd.to_datetime(fits.loc[good,bound])).all()
                if variant=='curve':
                    try:rank=pd.read_csv(folder/'shortlists.csv')
                    except (pd.errors.EmptyDataError,FileNotFoundError):
                        assert not a.ne(0).any();rank=pd.DataFrame()
                    if len(rank):assert (pd.to_datetime(rank.ranking_cutoff)<=pd.to_datetime((rank.year-1).astype(str)+'-01-01')).all()
                    for date,code in a[a.ne(0)].items():
                        combo,_=ACTIONS[int(code)]
                        assert rank[rank.year.eq(date.year)&rank.combo.eq(combo)].selected.all()
                try:dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
                except pd.errors.EmptyDataError:dec=pd.DataFrame(columns=['status'])
                entered=dec[dec.status.eq('entered')]
                if len(entered):
                    assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
                    t=pd.read_csv(folder/'trades.csv',parse_dates=['date','cohort'])
                    assert t.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
                    assert set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                    exited=t[t.reason.eq('expiry_settlement')]
                    assert exited.commission.eq(0).all() and exited.execution_penalty.eq(0).all()
                    for trade in entered.itertuples():
                        code=int(a.loc[trade.signal_date]);combo,size=ACTIONS[code];tenor,cp,ba,bb=COMBOS[combo]
                        selections=[]
                        for sid,bucket,oid in zip(d['secids'],[ba,bb],[trade.optionid_a,trade.optionid_b]):
                            source=f'data/curve_trading/selection/{cp}/{trade.signal_date.year}/{sid}.parquet'
                            selected=pd.read_parquet(source)
                            selected=selected[selected.date.eq(trade.signal_date)&selected.tenor.eq(tenor)&selected.bucket.eq(str(bucket))]
                            assert len(selected)==1 and selected.optionid.iloc[0]==oid and selected.cp_flag.iloc[0]==cp
                            selections.append(selected.iloc[0])
                        mix.append(dict(pair=d['pair'],label=d['label'],variant=variant,date=str(trade.entry_date.date()),
                            tenor=tenor,cp=cp,bucket_a=ba,bucket_b=bb,size=size,vega_a=trade.vega_a,vega_b=trade.vega_b,
                            dte_a=(selections[0].expiry-trade.entry_date).days,dte_b=(selections[1].expiry-trade.entry_date).days,
                            absolute_delta_a=abs(selections[0].bs_delta),absolute_delta_b=abs(selections[1].bs_delta)))
            audits.append(dict(pair=d['pair'],variant=variant,status='passed'))
        row=json.loads((BASE/d['pair']/'30d/status.json').read_text());row['variant']='previous_atm_reference';records.append(row)
    status=pd.DataFrame(records);status.to_csv(REPORT/'pair_status.csv',index=False)
    status[status.status.ne('complete')].to_csv(REPORT/'incomplete.csv',index=False)
    pd.DataFrame(audits).to_csv(REPORT/'audit.csv',index=False)
    summary=[];annual=[];pairs=[];components=[]
    common=[d for d in defs if d['pair'] not in failed]
    for variant in ['previous_atm_reference',*VARIANTS]:
        own=status[status.variant.eq(variant)].set_index('pair')
        for scope,roster in [('common',common),('full',defs)]:
            if scope=='full' and own.status.ne('complete').any():
                summary.append(dict(variant=variant,scope=scope,status='incomplete',pairs=len(roster)));continue
            book=None;entries=0;active=0
            for d in roster:
                row=own.loc[d['pair']];folder=(BASE/d['pair']/'30d') if variant=='previous_atm_reference' else OUTPUT/d['pair']/variant
                path=row.nav_source if isinstance(row.get('nav_source'),str) else folder/'daily_nav.parquet'
                nav=pd.read_parquet(path);b=nav[['nav','cash_benchmark']]
                if book is None:book=b.copy()
                else:assert book.index.equals(b.index);book+=b
                entries+=row.entries;active+=int(row.entries>0)
                pnl=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
                if scope=='common':
                    pairs.append(dict(pair=d['pair'],label=d['label'],variant=variant,entries=int(row.entries),net_excess_pnl=pnl))
                    def value(k):return float(row[k]) if k in row and pd.notna(row[k]) else 0.
                    c=dict(variant=variant,pair=d['pair'],gross_option_pnl=value('option_pnl')+nav.option_execution_cost.sum(),
                        gross_hedge_pnl=value('hedge_pnl')+nav.hedge_cost.sum(),option_cost=-nav.option_execution_cost.sum(),
                        hedge_cost=-nav.hedge_cost.sum(),stock_borrow=-nav.stock_borrow_cost.sum(),dividends=nav.dividend_cash.sum(),
                        financing_above_cash=nav.interest.sum()-(nav.cash_benchmark.iloc[-1]-nav.cash_benchmark.iloc[0]))
                    np.testing.assert_allclose(sum(v for k,v in c.items() if k not in ['variant','pair']),pnl,atol=1e-5,rtol=0);components.append(c)
            r=book.nav.pct_change().fillna(0);e=r-book.cash_benchmark.pct_change().fillna(0)
            pnl=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
            book.assign(excess_return=e,excess_pnl=pnl).to_csv(REPORT/f'{variant}_{scope}_portfolio.csv')
            summary.append(dict(variant=variant,scope=scope,status='complete',pairs=len(roster),entries=int(entries),active_pairs=active,
                annualized_excess_sharpe=sharpe(e.iloc[1:]),net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]))
            if scope=='common':
                for year,g in e.groupby(e.index.year):annual.append(dict(variant=variant,year=year,annualized_excess_sharpe=sharpe(g),
                    return_percent=100*((1+r.loc[g.index]).prod()-1),
                    cash_return_percent=100*((1+book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod()-1),
                    excess_return_percent=100*((1+r.loc[g.index]).prod()-(1+book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod()),
                    net_excess_pnl=pnl.loc[g.index].sum()))
    summary=pd.DataFrame(summary);summary.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pair_results.csv',index=False)
    pd.DataFrame(components).drop(columns='pair').groupby('variant').sum().to_csv(REPORT/'pnl_components.csv')
    concentration=[];pair_table=pd.DataFrame(pairs)
    for variant in ['previous_atm_reference',*VARIANTS]:
        own=pair_table[pair_table.variant.eq(variant)].sort_values('net_excess_pnl',ascending=False)
        positive=own[own.net_excess_pnl>1e-7];total_positive=positive.net_excess_pnl.sum()
        row=dict(variant=variant,profitable_pairs=len(positive),losing_pairs=int(own.net_excess_pnl.lt(-1e-7).sum()),
                 top_three_share_of_positive_pnl=positive.head(3).net_excess_pnl.sum()/total_positive if total_positive else None)
        if len(positive):
            top=positive.iloc[0];folder=(BASE/top.pair/'30d') if variant=='previous_atm_reference' else OUTPUT/top.pair/variant
            individual=pd.read_parquet(folder/'daily_nav.parquet')
            portfolio=pd.read_csv(REPORT/f'{variant}_common_portfolio.csv',index_col=0,parse_dates=[0])
            without=portfolio.nav-individual.nav+individual.cash_benchmark
            e=without.pct_change().fillna(0)-portfolio.cash_benchmark.pct_change().fillna(0)
            row.update(largest_winner=top.label,largest_winner_excess_pnl=top.net_excess_pnl,
                       sharpe_replacing_largest_winner_with_cash=sharpe(e.iloc[1:]))
        concentration.append(row)
    pd.DataFrame(concentration).to_csv(REPORT/'concentration.csv',index=False)
    choice_table=pd.DataFrame(mix)
    if len(choice_table):
        choice_table['common_roster']=~choice_table.pair.isin(failed)
        choice_table.to_csv(REPORT/'contract_mix.csv',index=False)
        choice_table=choice_table[choice_table.common_roster]
    else:pd.DataFrame(columns=['pair','variant','tenor','cp','bucket_a','bucket_b','common_roster']).to_csv(REPORT/'contract_mix.csv',index=False)
    if len(choice_table):
        choice_table.groupby(['variant','tenor','cp'],dropna=False).size().rename('entries').to_csv(REPORT/'choice_summary.csv')
        choice_table.assign(different_buckets=choice_table.bucket_a.astype(str).ne(choice_table.bucket_b.astype(str)),
                            different_expiry=choice_table.dte_a.ne(choice_table.dte_b),
                            entry_vega_imbalance=abs(choice_table.vega_a-choice_table.vega_b)/choice_table[['vega_a','vega_b']].max(axis=1)).groupby('variant').agg(
            different_bucket_fraction=('different_buckets','mean'),different_expiry_fraction=('different_expiry','mean'),
            median_entry_vega_imbalance=('entry_vega_imbalance','median'),maximum_entry_vega_imbalance=('entry_vega_imbalance','max')).to_csv(REPORT/'choice_checks.csv')
    else:
        pd.DataFrame(columns=['variant','tenor','cp','entries']).to_csv(REPORT/'choice_summary.csv',index=False)
        pd.DataFrame(columns=['variant','different_bucket_fraction']).to_csv(REPORT/'choice_checks.csv',index=False)
    lines=['# Contract-curve choice experiment','',f'{len(common)}/{len(defs)} pairs complete across both new variants. Common-roster comparison:','',
           '| Policy | Annualized excess Sharpe | Entries | Active pairs | P&L above cash |','|---|---:|---:|---:|---:|']
    for r in summary[summary.scope.eq('common')].itertuples():lines.append(f'| {r.variant} | {r.annualized_excess_sharpe:.3f} | {r.entries:.0f} | {r.active_pairs:.0f} | ${r.net_excess_pnl:,.0f} |')
    lines += ['',
        'Actions: 14/30/45-day target expiries; calls/calls or puts/puts; each leg independently chooses exact original ATM or 10/25/50/75/90 absolute BS delta. Nearest eligible actual contract within 0.075 delta; no interpolation or extrapolation of traded contracts. Both legs share a target tenor, but their actual expiries may differ. Expiry is anchored to the prior tenor study’s selection, so this is a bounded grid, not every strike or every maturity. Exact ATM remains nested. Same standard-contract/OI/quote-width filters; the former 10% ATM moneyness limit is replaced by explicit delta selection.', '',
        'The expanded policy considers all 216 contract combinations. Each January, it retains exact ATM30 calls plus up to seven combinations ranked by their best full-size directional shadow daily Sharpe over five years strictly before the prior validation year. Ranking requires 21 memory-permitted eligible cohorts; it does not inspect validation or evaluation outcomes. Monthly eight-leaf direct Sharpe policy search chooses the contract combination and one of eight signed sizes or flat. This is direct policy search, not a Q-learning or actor-critic implementation. The ATM control uses the same generalized optimizer with only the original ATM30 call choice; its random-search action ordering differs from the older reference.', '',
        'Same native 30D reference state and observation calendar, including VIX, momentum, surface context and ATM profitability memory. No new smile-state variables are added here. The execution gate uses the selected combination’s own strictly completed 91/365-day directional profitability memory. Initial integer vega mismatch ≤10%, daily delta hedging, existing costs and portfolio risk limits, monthly beta .85–1.15 and correlation ≥.5. Option quantities are fixed after entry apart from supported splits; there is no daily option-vega rebalancing. Each leg settles at its own expiry. Early assignment is not modeled.', '',
        'Execution uses observed midpoint plus/minus a 0.1-vol-point entry penalty per leg, $0.65 per option contract, and 1 bp on stock hedge trades. It does not assume crossing the full quoted option spread. Annual return columns distinguish total return, the cash benchmark, and return above cash; Sharpe uses daily cash-excess returns times √252.', '',
        'Monthly five-year proposal fit, separate prior-year full-book validation, then recent-five-year completed-outcome refit if validation cash-excess Sharpe is positive. As in the inherited model, validation gates pre-refit weights. Training uses rows with complete labels across the current shortlist; unavailable contracts give known no-entry labels, while unresolved held paths are excluded only from past training, never used as deployment filters. Float32 temporary labels bound storage; all cash-flow calculations and portfolio replay use float64. Common 60-calendar-day entry runoff; January 2018–August 2025 evaluation, including previously inspected research history. This is not an untouched holdout.', '',
        'Failed sleeves remain explicitly incomplete; common-roster comparisons remove them from every policy, including the old reference. Full-roster results are withheld for incomplete variants. No reference model has been promoted.', '',
        'Concentration diagnostics include replacing the largest winning sleeve with cash after the fact. This is a sensitivity check, not an implementable pair-selection rule.', '',
        '[Annual results](curve_trading/annual.csv) · [Pair results](curve_trading/pair_results.csv) · [Concentration](curve_trading/concentration.csv) · [Contract choices](curve_trading/contract_mix.csv) · [Choice summary](curve_trading/choice_summary.csv) · [P&L components](curve_trading/pnl_components.csv) · [Incomplete sleeves](curve_trading/incomplete.csv) · [Audit](curve_trading/audit.csv) · [Real-data reconciliation](curve_trading/real_data_reconciliation.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(summary.to_string(index=False))
    (OUTPUT/'task_status.json').write_text(json.dumps(dict(status='complete',pairs_attempted=len(defs),common_complete=len(common),
        failed_pairs=sorted(failed),reference_promoted=False,report_code=digest(__file__),
        runtime_dependencies=digest(OUTPUT/'runtime_dependencies.json')),indent=2)+'\n')

if __name__=='__main__':main()
