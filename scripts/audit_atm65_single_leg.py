"""Audit the frozen control, live entry rules and outright-vega exposures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_atm65_single_leg import OUT,REPORT,EMPLOYEE,MODELS,CHECKPOINT,baseline_results
from scripts.run_atm65_ablation import write_json,read_actions,PERIODS
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.single_leg_policy import ADDED,single_conditions
from scripts.option_flex.single_leg_engine import WEIGHTS
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.curve_trading.config import config
from scripts.replay_frozen_atm_commission import cash_book
from dataclasses import replace


def uncertainty():
    rows=[]
    for period,_ in PERIODS:
        for model,control in [('rl_pair_vrp','rl_baseline'),('rl_single_historical','rl_pair_vrp'),('rl_single_absolute','rl_pair_vrp')]:
            a=pd.read_csv(REPORT/f'all_{control}_{period}_portfolio.csv',index_col=0,parse_dates=[0]).excess_return.iloc[1:]
            b=pd.read_csv(REPORT/f'all_{model}_{period}_portfolio.csv',index_col=0,parse_dates=[0]).excess_return.iloc[1:]
            assert a.index.equals(b.index)
            h=pd.DataFrame(dict(a=a,b=b,n=1.,aa=a*a,bb=b*b))
            blocks=h.groupby(h.index.to_period('Q'))[['n','a','aa','b','bb']].sum().to_numpy()
            draw=np.random.default_rng(2026).integers(len(blocks),size=(2000,len(blocks)))
            n,sa,aa,sb,bb=blocks[draw].sum(axis=1).T
            def ratio(s,ss):
                sd=np.sqrt(np.maximum((ss-s*s/n)/(n-1),0))
                return np.sqrt(252)*np.divide(s/n,sd,out=np.full(len(n),np.nan),where=sd>1e-12)
            delta=ratio(sb,bb)-ratio(sa,aa)
            rows.append(dict(universe='common_complete_344',model=model,control=control,period=period,
                sharpe_change=float(np.sqrt(252)*(b.mean()/b.std()-a.mean()/a.std())),
                paired_quarter_ci90_low=float(np.nanquantile(delta,.05)),paired_quarter_ci90_high=float(np.nanquantile(delta,.95))))
    pd.DataFrame(rows).to_csv(REPORT/'paired_quarter_differences.csv',index=False)
    return rows


def main():
    manifest=json.loads((OUT/'inputs.json').read_text())
    for path,sha in manifest['code_and_sources'].items():assert digest(path)==sha,f'Source changed: {path}'
    frozen=json.loads(CHECKPOINT.read_text())
    assert digest(frozen['archive'])==frozen['archive_sha256']
    for path,sha in frozen['files'].items():assert digest(path)==sha,f'Frozen source changed: {path}'
    coverage=pd.read_csv(REPORT/'coverage.csv').set_index('pair')
    accounting=json.loads((REPORT/'accounting_coverage.json').read_text())
    assert len(manifest['definitions'])==345
    assert accounting['common_complete_pairs']==344 and accounting['unresolved']==[
        dict(pair='112463__161182',model='rl_single_historical',period='full',error='2020-03-13: missing held option 132591447')],accounting
    recovery=json.loads((OUT/'quote_recovery/manifest.json').read_text())
    assert not recovery['missing_date_recovered']
    assert digest(OUT/'quote_recovery/132591447_wrds.parquet')==recovery['raw_sha256']
    cfg=replace(config(),option_commission=.03)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=Market(cal);fits=[];entry_rows=[];changes=[];exposures=[];primary=[];annual=[]
    for period,start in PERIODS:
        calendar=cal[cal>=start]
        books={m:pd.DataFrame(0.,index=calendar,columns=['net_vega','gross_vega','single_net_vega','single_gross_vega','pair_net_vega']) for m in ['rl_baseline',*MODELS]}
        counts={m:dict(paired_entries=0,single_long_entries=0,single_short_entries=0) for m in books}
        cash=cash_book(calendar,market.rates,cfg.capital)
        navs={m:cash[['nav','cash_benchmark']]*345 for m in books}
        complete={m:0 for m in books};active={m:0 for m in books}
        for d in manifest['definitions']:
            pair=d['pair'];c=coverage.loc[pair];panel=None
            if c.retrain:
                folder=OUT/pair;panel=pd.read_parquet(folder/'states.parquet')
                if period=='full':
                    original=pd.read_parquet(EMPLOYEE/pair/'states.parquet')
                    pd.testing.assert_frame_equal(panel[original.columns],original)
                    assert np.isfinite(panel[ADDED]).all().all()
                    for side in ['a','b']:
                        assert panel[f'vrp_z_{side}_available'].isin([0.,1.]).all()
                        assert panel.loc[panel[f'vrp_z_{side}_available'].eq(0),f'vrp_z_{side}'].eq(0).all()
                    table=pd.read_csv(folder/'fits.csv');check_fits(table);assert table.baseline_reproduced.all()
                    fits.append(table.assign(pair=pair))
                    reference=json.loads((folder/'reference_replay_audit.json').read_text())
                    assert reference['baseline_actions_reproduced'] and reference['full_and_recent_nav_reproduced']
                    for m in MODELS:
                        action=read_actions(folder/m/'actions.csv');assert action.isin(range(25)).all()
                        if m=='rl_pair_vrp':assert action.le(8).all()
                        else:
                            cheap,rich=single_conditions(panel,m.removeprefix('rl_single_'))
                            pos=panel.index.get_indexer(action.index);assert (pos>=0).all()
                            for code in range(9,25):
                                allowed=cheap if (code-9)%8>=4 else rich
                                assert not ((action.to_numpy()==code)&~allowed[pos]).any()
                        changes.append(dict(pair=pair,label=d['label'],model=m,nonzero_recommendations=int(action.ne(0).sum()),single_recommendations=int(action.ge(9).sum())))
            for m in books:
                folder=OUT/pair/m/period if c.retrain and m in MODELS else baseline_results(pair,m,period,bool(c.employee_features))
                status=json.loads((folder/'status.json').read_text())
                if status['status']!='complete':
                    assert pair=='112463__161182' and m=='rl_single_historical' and period=='full'
                    continue
                complete[m]+=1;active[m]+=int(status['entries']>0)
                if status.get('cash_only'):
                    assert status['entries']==0;continue
                nav=pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'));assert nav.index.equals(calendar)
                navs[m]+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                for col in books[m].columns:
                    if col in nav:books[m][col]+=nav[col]
                    elif col=='pair_net_vega':books[m][col]+=nav.net_vega
                if c.retrain and m in MODELS:
                    try:trades=pd.read_csv(folder/'trades.csv',parse_dates=['date','cohort'])
                    except pd.errors.EmptyDataError:trades=pd.DataFrame()
                    stock=market.stocks[market.stocks.secid.isin(d['secids'])].copy()
                    audit(nav,trades,stock,cfg,d['secids'])
                    decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
                    entered=decisions[decisions.status.eq('entered')];assert len(entered)==status['entries']
                    assert (entered.entry_date>entered.signal_date).all()
                    for e in entered.itertuples():
                        a,b=WEIGHTS[int(e.action)]
                        if e.entry_kind=='pair':
                            assert a and b and a==-b
                            counts[m]['paired_entries']+=1
                        else:
                            assert bool(a)^bool(b)
                            sign=int(np.sign(a or b));assert sign==e.entry_sign
                            cheap,rich=single_conditions(panel,m.removeprefix('rl_single_'))
                            position=panel.index.get_loc(e.signal_date);assert (cheap if sign>0 else rich)[position]
                            assert abs(e.proposed_net_vega)<=900*e.sizing_nav/cfg.capital+1e-8
                            paid=trades[trades.reason.eq('entry')&trades.cohort.eq(e.signal_date)]
                            assert len(paid)==1 and np.sign(paid.quantity.iloc[0])==sign
                            assert paid.secid.iloc[0]==d['secids'][0 if a else 1]
                            counts[m]['single_long_entries' if sign>0 else 'single_short_entries']+=1
                        entry_rows.append(dict(model=m,period=period,pair=pair,label=d['label'],signal_date=str(e.signal_date.date()),
                            entry_date=str(e.entry_date.date()),entry_kind=e.entry_kind,entry_sign=e.entry_sign,action=e.action))
                else:counts[m]['paired_entries']+=status['entries']
        for m,book in books.items():
            if complete[m]==345:book.to_csv(REPORT/f'exposure_{m}_{period}.csv')
            total=sum(counts[m].values())
            portfolio=navs[m]
            row=dict(model=m,period=period,requested_pairs=345,complete_pairs=complete[m],
                status='complete' if complete[m]==345 else 'incomplete',
                available_entries=total,entries=total if complete[m]==345 else None,
                active_pairs=active[m] if complete[m]==345 else None,
                annualized_excess_sharpe=None,net_excess_pnl=None)
            if complete[m]==345:
                portfolio['return_']=portfolio.nav.pct_change().fillna(0)
                portfolio['excess_return']=portfolio.return_-portfolio.cash_benchmark.pct_change().fillna(0)
                row.update(net_excess_pnl=float(portfolio.nav.iloc[-1]-portfolio.cash_benchmark.iloc[-1]),**scores(portfolio,cfg.capital*345))
                portfolio.to_csv(REPORT/f'roster_{m}_{period}_portfolio.csv')
                for year,g in portfolio.groupby(portfolio.index.year):
                    sd=g.excess_return.std()
                    annual.append(dict(model=m,period=period,year=year,return_percent=100*((1+g.return_).prod()-1),
                        annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None))
            primary.append(row)
            exposures.append(dict(model=m,period=period,complete_pairs=complete[m],requested_pairs=345,**counts[m],
                mean_net_vega=float(book.net_vega.mean()),maximum_absolute_net_vega=float(book.net_vega.abs().max()),
                mean_gross_vega=float(book.gross_vega.mean()),mean_single_net_vega=float(book.single_net_vega.mean()),
                mean_single_gross_vega=float(book.single_gross_vega.mean()),
                days_with_standalone_exposure=int(book.single_gross_vega.gt(1e-8).sum())))
    pd.DataFrame(exposures).to_csv(REPORT/'exposures.csv',index=False)
    pd.DataFrame(primary).to_csv(REPORT/'primary_comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'primary_annual.csv',index=False)
    baseline=pd.read_csv('reports/atm65_employee_sentiment/comparison.csv')
    for period,_ in PERIODS:
        expected=baseline[(baseline.universe=='all')&baseline.model.eq('rl_sentiment')&baseline.period.eq(period)].iloc[0]
        got=next(r for r in primary if r['model']=='rl_baseline' and r['period']==period)
        np.testing.assert_allclose(got['annualized_excess_sharpe'],expected.annualized_excess_sharpe,atol=1e-12)
        assert got['entries']==expected.entries
    pd.DataFrame(entry_rows).to_csv(REPORT/'entries.csv',index=False)
    pd.DataFrame(changes).to_csv(REPORT/'policy_changes.csv',index=False)
    fit_table=pd.concat(fits,ignore_index=True)
    details={m:dict(enabled_refits=int(fit_table[m+'_status'].eq('enabled').sum()),
        enabled_refits_using_vrp_value=int(fit_table[m+'_refit_vrp_value_splits'].fillna(0).gt(0).sum())) for m in MODELS}
    result=dict(source_files_verified=len(manifest['code_and_sources']),frozen_files_verified=len(frozen['files']),
        archive_hash_verified=True,monthly_baseline_controls_verified=len(fit_table),
        original_states_unchanged=True,strict_completed_outcome_purge_verified=True,
        every_completed_single_entry_rule_verified=True,completed_inventory_and_cash_ledgers_verified=True,
        full_historical_rule_has_unresolved_path=True,missing_mark_confirmed_at_vendor=True,
        no_nn_retrained=True,accounting=accounting,refits=details,
        exposures=exposures,paired_quarter_differences=uncertainty(),
        uncertainty_note='Descriptive paired calendar-quarter resampling of inspected portfolio paths; no refitting or correction for model selection.')
    write_json(REPORT/'audit.json',result)
    pairs=pd.read_csv(REPORT/'pairs.csv');pairs=pairs[(pairs.universe=='all')&pairs.period.eq('full')]
    base=pairs[pairs.model.eq('rl_baseline')].set_index('pair');delta=[]
    for m in MODELS:
        candidate=pairs[pairs.model.eq(m)].set_index('pair')
        difference=pd.DataFrame(dict(model=m,label=candidate.label,pnl_change=candidate.net_excess_pnl-base.net_excess_pnl,
            entry_change=candidate.entries-base.entries)).reset_index();delta.append(difference)
    pd.concat(delta,ignore_index=True).to_csv(REPORT/'pair_changes.csv',index=False)
    old=REPORT/'summary.txt'
    matched=REPORT/'matched_subset_summary.txt'
    if not matched.exists():old.rename(matched)
    notes=[
        'Single-leg RL trial: historical and absolute variance-premium definitions',
        'PRIMARY RESULTS: primary_comparison.csv and primary_annual.csv use all 345 original sleeves.',
        'Historical rule full period is incomplete: RCL/NCLH holds option 132591447 with no 2020-03-13 quote in the local cache or a fresh bounded WRDS query. No price was imputed and the sleeve was not silently removed.',
        'All other model-period paths are complete, including the historical rule recent-period book starting flat in 2022.',
        'The supplementary comparison.csv/annual.csv and all_* portfolio files are the 344-sleeve common-complete subset. Its baseline differs from the frozen 345-sleeve reference; do not mix them.',
        'Original frozen baseline is reproduced: annualized excess Sharpe 0.368 full / 0.428 recent.',
        'Same six new per-leg variance-premium states, matched purged training rows, monthly refits and eight-leaf relative-value partition across the three new RL policies. Pair-only control permits the original paired ladder; flexible menus additionally permit a single selected leg.',
        'Historical longs/shorts require both names below/above their own past 252-session premium mean. Absolute longs/shorts require trailing RV above/below IV for both names. Trailing RV is a forecast proxy, not a new calibrated RV forecast.',
        'Both flexible menus retain pair actions and flat. Own-leg completed-profitability gates apply to singles; original paired gates, beta/correlation eligibility and employee inputs are unchanged.',
        'ATM30 calls, daily delta hedging, own expiry settlement, $450 maximum entry vega per selected leg, $900 entry net-vega cap per sleeve, original other risk limits. Training commission $0.65, evaluated commission $0.03 and 0.1-vol-point option entry penalty. No NN refits.',
        'Standalone actions deliberately add outright volatility exposure. The reference checkpoint and market caches remain unchanged.',
        '',pd.DataFrame(primary).to_string(index=False),'',
        'Verification:',json.dumps(result,indent=2)]
    old.write_text('\n'.join(notes)+'\n')
    write_json(OUT/'task_status.json',dict(status='complete_with_explicit_data_gap',requested_pairs=345,
        incomplete_model_periods=accounting['unresolved'],complete_primary_model_periods=7,models=MODELS))
    print(pd.DataFrame(primary).to_string(index=False),flush=True)
    print(pd.DataFrame(exposures).to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ['exposures','paired_quarter_differences']},indent=2),flush=True)


if __name__=='__main__':main()
