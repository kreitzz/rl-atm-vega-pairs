"""Retrospective validation selection; evaluation is a separate, locked phase."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.submission import digest, require_supervisor
from scripts.submission import BUNDLE, CAPITAL

OUT = Path('artifacts/contracts/p25_validation_split')
REPORT = Path('reports/p25_validation_split')
VALIDATION_END = pd.Timestamp('2021-12-31')
TEST_START = pd.Timestamp('2022-01-01')
WINDOWS = ['6M_floor','9M_floor','12M_floor','18M_floor','24M_floor',
           '12M_high_floor','12M_hl6','36M_hl6','36M_hl12','5Y_reference']
FEATURES = ['control','rv_risk','smile_price','smile_shape','term_structure',
            'physical_tail','variance_premium','iv_factor','volume_oi','employee','transcripts']
UNGATED = ['ungated_baseline','quarter_floor','skip_floor','flexible',
           'tree16','window36','tree4','window12']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def sharpe(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite daily returns')
    return float(np.sqrt(252)*values.mean()/values.std(ddof=1)) if values.std(ddof=1)>1e-12 else 0.


def validation_score(nav):
    """Never include 2022 onward; remove only the first initialization return."""
    selected = nav.loc[(nav.index >= '2018-01-01') & (nav.index <= VALIDATION_END), 'excess_return']
    if len(selected)<500 or selected.index.has_duplicates:
        raise ValueError('Incomplete validation calendar')
    return sharpe(selected.iloc[1:])


def candidates():
    rows=[]
    for family, names, root in [
        ('windows', WINDOWS, 'p25_recency'),
        ('features', FEATURES, 'p25_feature_search'),
        ('actions', [f'{year}Y_{menu}' for year in [1,3,5] for menu in ['binary','floor','quarter','flex']], 'p25_action_windows'),
        ('ungated', UNGATED, 'p25_ungated_search')]:
        for name in names:
            folder=Path('artifacts/contracts')/root/'replays'/name
            allocation=folder/'scored_allocations.parquet'
            if family=='windows' and name=='5Y_reference':
                allocation=Path('artifacts/contracts/p25_action_windows/replays/5Y_floor/scored_allocations.parquet')
            elif family=='ungated' and name=='ungated_baseline':
                allocation=Path('artifacts/contracts/p25_memory_ablation/replays/no_memory_gate/scored_allocations.parquet')
            rows.append(dict(candidate=family+'/'+name, pool='ungated' if family=='ungated' else 'screened',
                             family=family, name=name,
                             folder=str(folder),allocation=str(allocation)))
    return rows


def select():
    if (REPORT/'selection.json').exists():
        print('Validation choices already locked; retaining selection', flush=True)
        return
    grid=candidates()
    write(REPORT/'protocol.json', dict(validation='2018-01-02 to2021-12-31',
        evaluation='2022-01-03 to2025-08-29;2025 partial',
        ranking='Highest validation annualized daily excess Sharpe; lexical candidate ID breaks ties',
        screened_validation_equity=345_000_000, ungated_validation_equity=10_000_000,
        candidate_grid=grid, previously_inspected_history=True, untouched_holdout=False,
        inherited_choices='30D25-delta puts, roster/formation, direction screens and cost model fixed from prior research; not reselected here',
        training='Monthly completed-outcome fits remain causal; later completed evaluation outcomes may train subsequent fits',
        evaluation_boundary='Continuing books; carried inventory included; no reset at2022; no future-PnL filtering',
        original_checkpoints_preserved=True, automatic_promotion=False,
        resources='One worker20%onecore,384MiB RSSstop,35%headroom,64MiB swapgrowth'))
    records=[]
    for row in grid:
        folder=Path(row['folder'])
        nav=pd.read_parquet(folder/'daily_nav.parquet',columns=['excess_return'])
        score=validation_score(nav)
        records.append(dict(**row,validation_sharpe=score,
            nav_sha256=digest(folder/'daily_nav.parquet'),
            allocations_sha256=digest(row['allocation'])))
        del nav
        gc.collect()
    ranked=pd.DataFrame(records).sort_values(['pool','validation_sharpe','candidate'],ascending=[True,False,True])
    ranked.to_csv(REPORT/'validation_candidates.csv',index=False)
    selected={pool:group.iloc[0].to_dict() for pool,group in ranked.groupby('pool')}
    write(REPORT/'selection.json',dict(validation_end=str(VALIDATION_END.date()),
        evaluation_start=str(TEST_START.date()),selected=selected,
        ranking_depends_only_on_validation=True,untouched_holdout=False,
        candidate_grid_sha256=digest(REPORT/'protocol.json')))
    print(ranked[['candidate','validation_sharpe']].to_string(index=False),flush=True)
    print('Locked choices: '+json.dumps({k:v['candidate'] for k,v in selected.items()}),flush=True)


def locked_selection():
    value=json.loads((REPORT/'selection.json').read_text())
    assert digest(REPORT/'protocol.json')==value['candidate_grid_sha256']
    for row in value['selected'].values():
        assert digest(row['allocation'])==row['allocations_sha256']
        assert digest(Path(row['folder'])/'daily_nav.parquet')==row['nav_sha256']
    return value


def replay():
    """Funded selected screened policy and an additional fixed75 control only."""
    from scripts.run_p25_ungated_search import LazyDiskMarket, EXECUTION_COLUMNS
    from scripts import uniform_smile_engine as engine
    from scripts.uniform_smile_risk import book_risk
    from scripts.option_flex.contract_engine import Settings
    from scripts.submission_market import verify_ledger
    from scripts.cheap_vega_runtime import mark_book, metrics, annual_metrics
    locked=locked_selection()
    ck=json.loads(Path('config/p25_submission.json').read_text())
    cfg=Settings(**json.loads((BUNDLE/CAPITAL/'replays/10m_learned/config.json').read_text()))
    for case in ['screened_selected','screened_fixed75']:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():continue
        rows=pd.read_parquet(locked['selected']['screened']['allocation'],columns=EXECUTION_COLUMNS)
        assert len(rows)==608
        if case=='screened_fixed75':rows['learned_level']=3
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        market=LazyDiskMarket(set(rows.secid_a.astype(int))|set(rows.secid_b.astype(int)))
        cal=market.rates.index[market.rates.index>='2018-01-01']
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            nav,trades,decisions,ledger=engine.simulate(market,rows,cal,cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=ck['corporate_events'],wing_capital=1_000_000)
        finally:engine.book_risk=saved
        mark_book(nav,market.rates,cfg.capital)
        audit=verify_ledger(nav,trades,decisions,rows)
        nav.to_parquet(folder/'daily_nav.parquet')
        trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        pd.DataFrame(annual_metrics(nav,case,cfg.capital)).to_csv(folder/'annual.csv',index=False)
        write(folder/'status.json',dict(status='complete',**metrics(nav,cfg.capital),**audit,**ledger))
        print(f'{case}: full-history Sharpe {metrics(nav,cfg.capital)["annualized_excess_sharpe"]:.6f}',flush=True)
        market.db.close();market.stockdb.close()
        del market,rows,nav,trades,decisions
        gc.collect()


def period_metrics(nav, decisions, start, stop):
    """Preserve carried inventory, daily returns and the pre-period cash ledger."""
    mask=(nav.index>=pd.Timestamp(start)) & (nav.index<=pd.Timestamp(stop))
    part=nav.loc[mask]
    if part.empty:raise ValueError('Empty report period')
    first=nav.index.get_loc(part.index[0]);before=nav.iloc[max(0,first-1)]
    elapsed=(part.index[-1]-nav.index[max(0,first-1)]).days/365.25
    returns=part.excess_return.iloc[1:] if first==0 else part.excess_return
    relative=part.nav/part.cash_benchmark
    account_growth=float(part.nav.iloc[-1]/before.nav)
    cash_relative_growth=float(relative.iloc[-1]/(before.nav/before.cash_benchmark))
    entered=decisions.status.eq('entered') & decisions.entry_date.between(part.index[0],part.index[-1])
    ratio=pd.concat([pd.Series([before.nav/before.cash_benchmark]),relative.reset_index(drop=True)],ignore_index=True)
    return dict(start=str(part.index[0].date()),end=str(part.index[-1].date()),
        proposals=int(decisions.entry_date.between(part.index[0],part.index[-1]).sum()),
        annualized_excess_sharpe=sharpe(returns),
        annualized_total_account_return_pct=100*(account_growth**(1/elapsed)-1),
        annualized_growth_relative_to_cash_pct=100*(cash_relative_growth**(1/elapsed)-1),
        profit_above_cash=float((part.nav.iloc[-1]-part.cash_benchmark.iloc[-1])-(before.nav-before.cash_benchmark)),
        entries=int(entered.sum()),active_pairs=int(decisions.loc[entered,'pair'].nunique()),
        average_reserve_pct=100*float((part.capital_reserve/part.nav).mean()),
        peak_reserve_pct=100*float((part.capital_reserve/part.nav).max()),
        max_cash_relative_drawdown_pct=100*float((ratio/ratio.cummax()-1).min()),
        post_entry_risk_breach_days=int(part.post_trade_risk_breach.sum()))


def evaluate():
    locked=locked_selection()
    records=[]
    for row in candidates():
        folder=Path(row['folder'])
        nav=pd.read_parquet(folder/'daily_nav.parquet',columns=['excess_return'])
        records.append(dict(candidate=row['candidate'],pool=row['pool'],
            validation_sharpe=validation_score(nav),
            retrospective_holdout_sharpe=sharpe(nav.loc[nav.index>=TEST_START,'excess_return']),
            selected=locked['selected'][row['pool']]['candidate']==row['candidate']))
        del nav
    pd.DataFrame(records).to_csv(REPORT/'candidate_periods.csv',index=False)
    arms={
        'screened_selected':OUT/'replays/screened_selected',
        'screened_fixed50':BUNDLE/CAPITAL/'replays/10m_fixed50',
        'screened_fixed75':OUT/'replays/screened_fixed75',
        'screened_fixed100':BUNDLE/CAPITAL/'replays/10m_fixed100',
        'ungated_selected':Path(locked['selected']['ungated']['folder']),
        'ungated_fixed25':Path('artifacts/contracts/p25_ungated_search/replays/fixed25'),
        'ungated_fixed50':Path('artifacts/contracts/p25_ungated_search/replays/fixed50'),
        'ungated_fixed75':Path('artifacts/contracts/p25_ungated_search/replays/fixed75'),
        'ungated_fixed100':Path('artifacts/contracts/p25_memory_ablation/replays/no_memory_gate_fixed100'),
        'legacy_screened':BUNDLE/CAPITAL/'replays/10m_learned',
        'legacy_ungated':Path('artifacts/contracts/p25_memory_ablation/replays/no_memory_gate'),
    }
    results=[];annual=[];decisions_out=[];book={}
    for case,folder in arms.items():
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date','signal_date'])
        book[case]=nav
        for name,start,stop in [('validation','2018-01-01','2021-12-31'),
                                ('retrospective_holdout','2022-01-01','2025-08-29'),
                                ('full_history','2018-01-01','2025-08-29')]:
            results.append(dict(case=case,period=name,**period_metrics(nav,decisions,start,stop)))
            subset=decisions.loc[decisions.entry_date.between(start,stop)]
            for status,count in subset.status.value_counts().items():
                decisions_out.append(dict(case=case,period=name,status=status,count=int(count)))
        from scripts.cheap_vega_runtime import annual_metrics
        annual.extend(annual_metrics(nav,case,10_000_000))
        for row in annual[-8:]:
            row['entries']=int((decisions.status.eq('entered') & decisions.entry_date.dt.year.eq(row['year'])).sum())
    frame=pd.DataFrame(results);frame.to_csv(REPORT/'periods.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(decisions_out).to_csv(REPORT/'admission.csv',index=False)
    intervals=[]
    for pool in ['screened','ungated']:
        a=book[pool+'_selected'].loc[TEST_START:,'excess_return'].to_numpy()
        b=book[pool+'_fixed100'].loc[TEST_START:,'excess_return'].to_numpy()
        assert len(a)==len(b)
        rng=np.random.default_rng(2026)
        changes=[]
        for _ in range(1000):
            starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/126)))
            idx=((starts[:,None]+np.arange(126)[None,:])%len(a)).ravel()[:len(a)]
            changes.append(sharpe(a[idx])-sharpe(b[idx]))
        intervals.append(dict(pool=pool,point=sharpe(a)-sharpe(b),
            lower_95=float(np.quantile(changes,.025)),upper_95=float(np.quantile(changes,.975)),
            block_sessions=126,replications=1000,paired=True,circular=True,selection_adjusted=False))
    pd.DataFrame(intervals).to_csv(REPORT/'holdout_intervals.csv',index=False)
    write(REPORT/'evaluation.json',dict(selection_sha256=digest(REPORT/'selection.json'),
        arms={k:str(v) for k,v in arms.items()},
        metrics='Continuing cash books; period dollar profit is change in NAV-minus-cash-benchmark; relative growth rebases NAV/cash benchmark at preceding close',
        inherited_universe_and_instruments=True,retrospective_holdout=True,untouched_holdout=False))
    print(frame.loc[frame.period.eq('retrospective_holdout'),['case','annualized_excess_sharpe','entries','profit_above_cash','average_reserve_pct']].to_string(index=False),flush=True)


def package():
    """Save the selection evidence and new funded ledgers without loading quotes."""
    import zipfile
    locked=locked_selection()
    evaluation=json.loads((REPORT/'evaluation.json').read_text())
    paths=[]
    curves=[]
    for case,folder in evaluation['arms'].items():
        nav=pd.read_parquet(Path(folder)/'daily_nav.parquet',columns=['nav','cash_benchmark','excess_return'])
        part=nav.loc[TEST_START:].copy()
        previous=nav.loc[nav.index<TEST_START].iloc[-1]
        part['cash_relative_growth_pct']=100*((part.nav/part.cash_benchmark)/(previous.nav/previous.cash_benchmark)-1)
        part['case']=case
        part.index.name='date'
        curves.append(part.reset_index())
    pd.concat(curves,ignore_index=True).to_csv(REPORT/'holdout_paths.csv',index=False)
    for row in candidates():
        paths.extend([Path(row['folder'])/'daily_nav.parquet',Path(row['allocation'])])
    paths.extend(p for p in (Path('artifacts/contracts/p25_feature_search/models/control')).glob('*') if p.is_file())
    paths.extend(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['STOP','progress.json'])
    paths=sorted(set(paths))
    write(REPORT/'bundle_manifest.json',dict(files={str(p):digest(p) for p in paths},
        scope='Validation scores for41 saved configurations,90 selected screened monthly models,2 new funded ledgers. Existing submission/memory/ungated bundles supply other models, quotes and controls.'))
    with zipfile.ZipFile('data/p25_validation_split_experiment.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in paths:archive.write(path,str(path))
    original=json.loads(Path('config/p25_submission.json').read_text())
    write(Path('config/p25_validation_2018_2021.json'),dict(status='retrospective_validation_selection',
        validation='2018-2021',evaluation='2022-2025-08-29',untouched_holdout=False,
        capital=10_000_000,screened=dict(candidate='features/control',inputs=original['features'][:27],
            lookback_months=18,max_leaves=8,actions=[.5,.75,1.]),
        ungated=dict(candidate='ungated/quarter_floor',inputs=original['features'],
            lookback_months=18,max_leaves=8,actions=[.25,.5,.75,1.]),
        original_checkpoint_preserved='config/p25_submission.json',
        selection_record='reports/p25_validation_split/selection.json',
        result_table='reports/p25_validation_split/periods.csv',
        inherited_choices='Instrument, roster, formation calendar, entry direction and execution/risk assumptions come from earlier research.'))
    (REPORT/'reproduction.txt').write_text(
        'Restore data/submission using scripts.submission --restore-cache. Then unzip -n the memory, ungated and validation-split experiment archives under data/.\n'
        'Run scripts.run_signal_discovery_guarded with module scripts.run_p25_validation_split and -- --phase PHASE. Phases: select,replay,evaluate,package,test.\n'
        'Use one worker, duty-cycle0.2, RSS limit384MiB,35%free headroom,64MiB maximum swap growth and60s low-headroom pause. No WRDS download needed.\n'
        'Selection already locked: select refuses replacement; replay resumes completed statuses. A fresh replay requires a separate checkout without the two restored new replay status files.\n'
        'The grid is a union of saved experiments, not a Cartesian search. Screened validation ranks original345m research books; funded evaluation uses10m. Ungated validation and evaluation both use10m.\n'
        'Monthly saved causal models are reused. This split did not refit models or change original checkpoints; only screened selected and fixed75 funded books are new.\n')
    (REPORT/'findings.md').write_text(
        '# Validation selection and retrospective holdout\n\n'
        'Select within the inherited30D25-delta put setup using2018–2021 only, then evaluate2022–29August2025. These later dates were inspected in earlier research, so this is not an untouched holdout. Instrument, roster and formation calendar were not reselected.\n\n'
        '| Proposal pool | Validation choice | Validation Sharpe |2022–2025 selected Sharpe | Fixed-full Sharpe |\n|---|---|---:|---:|---:|\n'
        '| Screened |27 inputs,18months,50/75/100% |0.575 |1.263 |1.055 |\n'
        '| Broader |33 inputs,18months,25/50/75/100% |1.309 |0.456 |0.567 |\n\n'
        'Screened validation uses the archived345m research account; its selected policy is replayed at10m for final evaluation. All final controls use10m. The two pools select separately, so their difference is not a pure gate ablation.\n\n'
        'The screened policy makes110 new entries and adds$248,673 above cash, versus75 entries and$200,853 at fixed full. The Sharpe gain is0.208 (paired126-session circular-block bootstrap,1000 replications,95% interval−0.103 to+0.424; no search adjustment). Broader sizing loses to fixed full, with900 entries and$278,895 above cash versus459 and$330,791.\n\n'
        'Books continue across2022; carried inventory contributes to holdout P&L. Monthly refits use completed outcomes, including earlier holdout outcomes once completed. Only configuration choice is frozen.2025 is partial.\n\n'
        '[Locked selection](selection.json) · [All validation candidates](validation_candidates.csv) · [Period results](periods.csv) · [Yearly results](annual.csv) · [Intervals](holdout_intervals.csv) · [Reproduction](reproduction.txt)\n')
    print(f'Packaged{len(paths)} files; preserved original checkpoints',flush=True)


def main():
    require_supervisor()
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['select','replay','evaluate','package','test'],required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.phase=='select':select()
    elif args.phase=='replay':replay()
    elif args.phase=='evaluate':evaluate()
    elif args.phase=='package':package()
    elif args.phase=='test':
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_validation_split.py'))
        if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
