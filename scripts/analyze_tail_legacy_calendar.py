"""Same broad spread history, restricted only to the saved legacy opportunity dates."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd

from scripts.analyze_tail_universe_change import BASE, COMPLETE, memory_filter, working_inputs
from scripts.repair_tail_put25_pooled import AccountingReplay, save
from scripts.run_tail_put25_rule_sizing import PutMarket
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.curve_trading.config import config
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores, audit
from scripts.run_contract_walkforward import digest
from scripts.option_flex.contract_engine import UnresolvedPosition

OUT = Path('artifacts/contracts/tail_calendar_ablation')
REPORT = Path('reports/tail_calendar_ablation')
MODELS = ['legacy_calendar_ungated', 'legacy_calendar_memory']


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((BASE / 'inputs.json').read_text()); defs = inputs['definitions']
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    cfg = replace(config(), option_commission=.03); market = Market(cal)
    evalcal = cal[cal >= '2018-01-01']; cash = cash_book(evalcal, market.rates, cfg.capital)
    sources = {**inputs['sources'], **json.loads((BASE / 'market_sources.json').read_text())}
    loader = PutMarket(market, cfg, sources)
    for n, d in enumerate(defs, 1):
        folder = OUT / d['pair']; folder.mkdir(exist_ok=True)
        if all((folder / m / 'full/status.json').exists() for m in MODELS): continue
        old = Path('artifacts/contracts/tail_put25_strategy') / d['pair'] / 'rule/actions.csv'
        new = COMPLETE / d['pair'] / 'ungated_spread/actions.csv'
        state = BASE / d['pair'] / 'states.parquet'
        codes = pd.Series(dtype=int, index=pd.DatetimeIndex([]), name='code')
        if old.exists() and new.exists() and state.exists():
            sources[str(old)] = digest(old); sources[str(new)] = digest(new); sources[str(state)] = digest(state)
            codes = read_actions(new)
            codes.loc[~codes.index.isin(read_actions(old).index)] = 0
            panel = pd.read_parquet(state)
            filtered = codes.where(memory_filter(panel.loc[codes.index]), 0)
        else: filtered = codes.copy()
        engine = s = None
        for model, actions in zip(MODELS, [codes, filtered]):
            dest = folder / model / 'full'; dest.mkdir(parents=True, exist_ok=True)
            actions.to_csv(dest.parent / 'actions.csv')
            if (dest / 'status.json').exists(): continue
            if not actions.ne(0).any():
                write_json(dest / 'status.json', dict(status='complete', entries=0, cash_only=True))
                continue
            if engine is None:
                loader.retain(d['secids']); assets = [loader.asset(sid) for sid in d['secids']]
                q, s, sel, divs = working_inputs(assets, d['secids'])
                engine = AccountingReplay(q, s, sel, d['secids'], divs, cfg)
            try:
                nav, trades, decisions = engine.run(actions, evalcal); mark_book(nav, market.rates, cfg.capital)
                if trades.empty:
                    checks = audit(nav, trades, s, cfg, d['secids'])
                    np.testing.assert_allclose(nav.nav, cash.nav, atol=1e-7, rtol=0)
                    write_json(dest / 'status.json', dict(status='complete', entries=0, cash_only=True, **checks))
                else: save(dest, nav, trades, decisions, cfg, s, d)
            except UnresolvedPosition as exc:
                write_json(dest / 'status.json', dict(status='incomplete', error=str(exc)))
            print(f'Legacy-date control {n}/{len(defs)} {d["label"]} {model}', flush=True)
        del engine, s; gc.collect()
    loader.cache.clear(); gc.collect()
    rows = []
    for model in MODELS:
        book = cash[['nav','cash_benchmark']] * len(defs); entries = active = 0
        for d in defs:
            dest = OUT / d['pair'] / model / 'full'; status = json.loads((dest / 'status.json').read_text())
            assert status['status'] == 'complete', status
            nav = cash if status.get('cash_only') else pd.read_parquet(dest / 'daily_nav.parquet')
            book += nav[['nav','cash_benchmark']] - cash[['nav','cash_benchmark']]
            entries += status['entries']; active += int(status['entries'] > 0)
        book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
        book.to_csv(REPORT / f'{model}_full_portfolio.csv')
        rows.append(dict(model=model, entries=entries, active_pairs=active,
            net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]), **scores(book, cfg.capital * len(defs))))
    prior = pd.read_csv('reports/tail_universe_change/comparison.csv')
    result = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True)
    result.to_csv(REPORT / 'comparison.csv', index=False)
    for path, sha in sources.items(): assert digest(path) == sha, f'Source changed: {path}'
    write_json(REPORT / 'audit.json', dict(status='complete', no_training=True, sources_verified=len(sources),
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        protocol='Restrict the unchanged broad spread/rank signals to the saved legacy rule action-date index for each pair, with and without the completed-profit/count filter. Preserve broad rank history, broad profitability histories, beta screen, sizing, quotes, costs and repaired lifecycle accounting. This isolates the legacy pair/date coverage restriction, not an untouched out-of-sample selection test.'))
    (REPORT / 'findings.txt').write_text(result.to_string(index=False)+'\n\n'+(REPORT / 'audit.json').read_text())
    hashes = {str(p): digest(p) for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'}
    hashes[str(Path(__file__))] = digest(Path(__file__))
    write_json(REPORT / 'output_hashes.json', hashes)
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__': main()
