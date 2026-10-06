"""Lower training hurdles; reuse the frozen higher-hurdle implementation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import time
import unittest

import numpy as np
import pandas as pd

import scripts.run_put75_cost_hurdles as engine
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json, read_actions
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.curve_trading.config import config
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_selected_pair_backtest import scores

OUT = Path('artifacts/contracts/put75_lower_hurdles')
REPORT = Path('reports/put75_lower_hurdles')
HIGHER = Path('reports/put75_cost_hurdles')
HURDLES = {'003':.03,'025':.25,'050':.50}


def summarize(defs,market,cfg,sources):
    cal = market.calendar[market.calendar>='2018-01-01']
    cash = cash_book(cal,market.rates,cfg.capital)
    books = {}; details = {}; pair_rows = []; allocation_rows = []
    expected = pd.read_csv(sources.track(engine.FROZEN_REPORT/'comparison.csv'),float_precision='round_trip').set_index('model')
    for name in ['atm_frozen','put25_frozen','P75_flex','P75_floor']:
        book = pd.read_csv(sources.track(engine.FROZEN_REPORT/f'{name}_full_portfolio.csv'),index_col=0,parse_dates=[0],float_precision='round_trip')
        assert book.index.equals(cal)
        np.testing.assert_allclose(scores(book,cfg.capital*len(defs))['annualized_excess_sharpe'],expected.loc[name,'annualized_excess_sharpe'],atol=1e-10,rtol=0)
        books[name] = book[['nav','cash_benchmark']]
        if name.startswith('P75'):
            details[name] = dict(training_hurdle=.65,entries=int(expected.loc[name,'entries']),active_pairs=int(expected.loc[name,'active_pairs']))
    higher = pd.read_csv(sources.track(HIGHER/'comparison.csv'),float_precision='round_trip').set_index('model')
    for name in ['P75_flex_100','P75_flex_150']:
        book = pd.read_csv(sources.track(HIGHER/f'{name}_full_portfolio.csv'),index_col=0,parse_dates=[0],float_precision='round_trip')
        assert book.index.equals(cal)
        np.testing.assert_allclose(scores(book,cfg.capital*len(defs))['annualized_excess_sharpe'],higher.loc[name,'annualized_excess_sharpe'],atol=1e-10,rtol=0)
        books[name] = book[['nav','cash_benchmark']]
        details[name] = {key:float(higher.loc[name,key]) for key in ['training_hurdle','entries','active_pairs','commission','option_cost','hedge_cost','risk_breach_pair_sessions']}
    for key,hurdle in HURDLES.items():
        name = f'P75_flex_{key}'; book = cash[['nav','cash_benchmark']]*len(defs)
        detail = dict(training_hurdle=hurdle,entries=0,active_pairs=0,commission=0.,option_cost=0.,hedge_cost=0.,risk_breach_pair_sessions=0)
        for d in defs:
            root = OUT/'pairs'/d['pair']/key; full = engine.resolved_full(root,sources)
            status = json.loads(sources.track(full/'status.json').read_text()); assert status['status']=='complete'
            nav = cash if status.get('cash_only') else pd.read_parquet(sources.track(full/'daily_nav.parquet'))
            assert nav.index.equals(cal)
            book += nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
            for field in ['entries','commission','option_cost','hedge_cost']:
                detail[field] += status.get(field,0)
            detail['active_pairs'] += int(status['entries']>0)
            detail['risk_breach_pair_sessions'] += status.get('risk_breach_sessions',0)
            pair_rows.append(dict(model=name,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
            alloc = pd.read_csv(root/'scored_allocations.csv',index_col=0,parse_dates=[0]).allocation
            base = read_actions(engine.SIGNAL/'pairs'/d['pair']/'P75/spread/actions.csv')
            for level,count in alloc.loc[base.reindex(alloc.index,fill_value=0).ne(0)].value_counts().items():
                allocation_rows.append(dict(model=name,allocation=float(level),signals=int(count)))
        books[name] = book; details[name] = detail
    combinations = {'ATM_plus_put25':['atm_frozen','put25_frozen']}
    for name in list(details):
        combinations[f'put25_plus_{name}'] = ['put25_frozen',name]
        combinations[f'ATM_put25_plus_{name}'] = ['atm_frozen','put25_frozen',name]
    for name,members in combinations.items():
        books[name] = sum(books[m][['nav','cash_benchmark']] for m in members)
    rows = []; annual = []
    for name,book in books.items():
        book = book.copy(); book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_-book.cash_benchmark.pct_change().fillna(0)
        sleeves = len(combinations[name]) if name in combinations else 1
        rows.append(dict(model=name,funded_sleeve_sets=sleeves,net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),
                         **details.get(name,{}),**scores(book,cfg.capital*len(defs)*sleeves)))
        book.to_csv(REPORT/f'{name}_full_portfolio.csv')
        for year,g in book.groupby(book.index.year):
            annual.append(dict(model=name,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None))
    result = pd.DataFrame(rows); result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(allocation_rows).groupby(['model','allocation']).signals.sum().reset_index().to_csv(REPORT/'allocations.csv',index=False)
    grid = []
    indexed = result.set_index('model')
    for name in details:
        if name=='P75_floor':
            continue
        grid.append(dict(training_hurdle=details[name]['training_hurdle'],entries=details[name]['entries'],
                         standalone_annualized_excess_sharpe=indexed.loc[name,'annualized_excess_sharpe'],
                         downside_plus_upside_sharpe=indexed.loc[f'put25_plus_{name}','annualized_excess_sharpe'],
                         ATM_downside_upside_sharpe=indexed.loc[f'ATM_put25_plus_{name}','annualized_excess_sharpe']))
    grid = pd.DataFrame(grid).sort_values('training_hurdle')
    grid.to_csv(REPORT/'six_hurdle_comparison.csv',index=False)
    print(grid.to_string(index=False),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase',choices=['all','test','summarize'],default='all'); args=parser.parse_args()
    if args.phase=='test':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_put75_lower_hurdles.py'))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    started = time.monotonic(); OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    # Rebind only the new study's output/configuration inside this one worker.
    # The frozen higher-hurdle source files, models and ledgers are untouched.
    engine.OUT = OUT; engine.REPORT = REPORT; engine.HURDLES = HURDLES
    sources = engine.Sources()
    sources.expected.update(json.loads((HIGHER/'output_hashes.json').read_text()))
    if (OUT/'market_sources.json').exists():
        for path,value in json.loads((OUT/'market_sources.json').read_text()).items():
            assert digest(path)==value,f'Resumed source changed: {path}'
            sources[path] = value
            if path in sources.expected:
                assert value==sources.expected[path]; sources.verified.add(path)
    code_paths = [Path(__file__),Path(engine.__file__),Path('scripts/put75_cost_hurdle.py'),
                  Path('tests/test_put75_cost_hurdle.py'),Path('tests/test_put75_lower_hurdles.py'),
                  Path('scripts/run_smile_sizing.py'),Path('scripts/smile_sizing_policy.py'),
                  Path('scripts/research_opposite_wing_signals.py'),Path('scripts/analyze_tail_universe_change.py'),
                  Path('scripts/curve_trading/unit_paths.py'),Path('scripts/curve_trading/policy.py'),
                  Path('scripts/option_flex/pooled_tail_engine.py'),Path('scripts/option_flex/contract_risk.py')]
    for path in code_paths+[ROOT/'zero_curve.parquet',ROOT/'distributions.parquet',Path('data/raw/opposite_wing_DEXCAUS.csv')]:
        sources.track(path)
    cal = load_calendar(ROOT); cal = cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    for year in cal.year.unique():
        sources.track(ROOT/'underlying'/f'year={year}.parquet')
    defs = json.loads(sources.track(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    market = Market(cal); cfg = replace(config(),option_commission=.03)
    protocol = ('P75 flexible sizing: fixed lower training commission hurdles 0.03/0.25/0.50; actual commission stays 0.03. '
                'Reference 0.65 and completed 1.00/1.50 policies and ledgers remain frozen. '
                'Reuse the exact higher-hurdle preparation/training/replay implementation, changing training cash flows only. '
                'All 345 roster pairs and every observed state scored, unchanged features/calendar/rule directions/risk/execution. '
                'Monthly five-year completed-outcome pooled direct-Sharpe sizing with 0/25/50/75/100 choices; not a sequential inventory-aware agent. '
                'Baseline labels must reproduce bit-for-bit; fee-adjusted unit paths exactly recompute paired funding, including negative-cash borrowing. '
                'Actual actions identical to a completed reference may reuse its audited ledger. '
                'ATM/downside remain frozen; equal-capital separately funded combinations, no shared inventory or optimized weights. '
                'Known held-quote issues remain equally present; inspected history and six hurdle comparisons are exploratory, not an untouched holdout. '
                'One worker, 50% of one core, one BLAS/Arrow thread, 3 GiB RSS stop.')
    write_json(OUT/'inputs.json',dict(hurdles=HURDLES,actual_commission=.03,features=engine.FEATURES,levels=engine.LEVELS,protocol=protocol))
    if args.phase=='all':
        engine.prepare(defs,market,cfg,sources)
        engine.train(defs,cal,sources)
        engine.replay(defs,market,cfg,sources)
    result = summarize(defs,market,cfg,sources)
    fits = pd.read_csv(REPORT/'fits.csv'); assert len(fits)==270
    assert (pd.to_datetime(fits.last_training_outcome)<pd.to_datetime(fits.fit)).all()
    coverage = pd.read_csv(REPORT/'scoring_coverage.csv')
    assert len(coverage)==1035 and coverage.groupby('hurdle').scored_rows.sum().eq(58518).all()
    for d in defs:
        for key in HURDLES:
            full = engine.resolved_full(OUT/'pairs'/d['pair']/key,sources)
            assert json.loads((full/'status.json').read_text())['status']=='complete'
    for path,value in sources.items():
        assert digest(path)==value,f'Source changed during run: {path}'
    write_json(OUT/'market_sources.json',sources)
    record = dict(status='complete',models_retrained=270,roster_pairs=len(defs),all_states_scored=True,
                  actual_commission=.03,frozen_source_files_verified=len(sources.verified),combined_shared_inventory=False,
                  elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  finalization_only=args.phase=='summarize')
    write_json(REPORT/'audit.json',record)
    (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+json.dumps(record,indent=2)+'\n\n'+protocol+'\n')
    engine.progress('complete',**record)
    hashes = {str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
    for path in code_paths:
        hashes[str(path)] = digest(path)
    write_json(REPORT/'output_hashes.json',hashes)


if __name__=='__main__':
    main()
