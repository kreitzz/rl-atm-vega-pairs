"""Pooled sizing on the exact saved .756 rule opportunity set; bounded worker."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd

from scripts.analyze_tail_universe_change import BASE, memory_filter, working_inputs
from scripts.repair_tail_put25_pooled import AccountingReplay, save
from scripts.tail_put25_pooled import FEATURES, completed_mask, partition, aggregate_chunk, optimize_book
from scripts.tail_put25_rule_sizing import translate, SIZE_CODES
from scripts.tail_put25_policy import CODES
from scripts.curve_trading.policy import predict
from scripts.curve_trading.config import config, RUNOFF
from scripts.run_tail_put25_rule_sizing import PutMarket
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.run_contract_walkforward import digest
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores, audit

CONTROL = Path('artifacts/contracts/tail_calendar_ablation')
OUT = Path('artifacts/contracts/tail_put25_legacy_rl_all_months')
REPORT = Path('reports/tail_put25_legacy_rl_all_months')
MODELS = ['existing_pooled_rl', 'memory_trained_pooled_rl']


def training_selection(meta, cutoff):
    # No evaluation labels or per-pair model thresholds. Positive-history
    # criteria are evaluated at each historical signal date, not at fit time.
    return completed_mask(meta.date, meta.end, cutoff) & meta.finite_label & memory_filter(meta)


def train(defs, cal):
    parts = [pd.read_parquet(BASE / d['pair'] / 'training_rows.parquet')
             for d in defs if (BASE / d['pair'] / 'training_rows.parquet').exists()]
    meta = pd.concat(parts, ignore_index=True)
    folder = OUT / 'models'; folder.mkdir(exist_ok=True)
    logs = []
    for month in pd.period_range('2018-01-01', cal[-1] - pd.Timedelta(days=RUNOFF), freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        info = folder / f'{fitdate}.json'; path = folder / f'{fitdate}.joblib'
        if info.exists():
            logs.append(json.loads(info.read_text())); continue
        data = meta.loc[training_selection(meta, cutoff)].reset_index(drop=True)
        row = dict(fit=fitdate, training_rows=len(data), training_pairs=int(data.pair.nunique()))
        if len(data) < 126:
            # Coverage is retained even during a global cold start. No pair
            # falls back to flat simply because its own history is sparse.
            row['status'] = 'global_full_size_fallback'
        else:
            tree, leaves = partition(data[FEATURES], data.target)
            states = np.searchsorted(leaves, tree.apply(data[FEATURES]))
            first, last = int(data.position.min() + 1), int(data.end_position.max())
            book = np.zeros((last - first + 1, len(leaves) * 5))
            for pair, rows in data.groupby('pair', sort=False).groups.items():
                rows = np.asarray(rows)
                disk = np.load(BASE / pair / 'labels.npy', mmap_mode='r')
                labels = disk[data.loc[rows, 'label_row'].to_numpy(int)]
                aggregate_chunk(book, states[rows], labels, data.loc[rows, 'position'].to_numpy(int), first)
                del disk, labels
            model, details = optimize_book(book, leaves, tree)
            assert pd.Timestamp(data.end.max()) < cutoff
            row.update(status='enabled', last_training_outcome=str(pd.Timestamp(data.end.max()).date()),
                leaves=len(leaves), leaf_size_aliases=model[2].tolist(),
                split_features=[FEATURES[int(i)] for i in tree.tree_.feature if i >= 0], **details)
            joblib.dump(model, path)
            del tree, book, model
        write_json(info, row); logs.append(row)
        print(f'Memory-conditioned fit {fitdate}: {row["status"]}; {row["training_rows"]:,} rows, {row["training_pairs"]} pairs', flush=True)
        del data
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)
    del meta, parts; gc.collect()


def score_all(panel, reference, cal, model_root):
    wanted = (panel.index >= '2018-01-01') & (panel.index <= cal[-1] - pd.Timedelta(days=RUNOFF))
    dates = panel.index[wanted]
    base = reference.reindex(dates, fill_value=0).astype(int)
    output = base.copy().rename('code')
    alloc = pd.Series(np.nan, index=dates, name='allocation')
    fallback = scored = 0
    for month in dates.to_period('M').unique():
        idx = dates[(dates.to_period('M') == month) & panel.loc[dates, 'observed'].to_numpy(bool)]
        if not len(idx): continue
        path = model_root / f'{month.start_time.date()}.joblib'
        if path.exists():
            aliases = predict(joblib.load(path), panel.loc[idx, FEATURES])
            output.loc[idx] = translate(aliases, base.loc[idx])
            alloc.loc[idx] = [SIZE_CODES.index(int(a)) / 4 for a in aliases]
            scored += len(idx)
        else:
            alloc.loc[idx] = 1.; fallback += len(idx)
    assert output[base.eq(0)].eq(0).all()
    assert np.isin(output[base.eq(CODES[1])], [0, *CODES[1:5]]).all()
    assert np.isin(output[base.eq(CODES[-1])], [0, *CODES[5:]]).all()
    assert abs(output.ne(0).sum()) <= base.ne(0).sum()
    return output, alloc, scored, fallback


def replay(defs, market, cfg, sources):
    loader = PutMarket(market, cfg, sources)
    cal = market.calendar[market.calendar >= '2018-01-01']; cash = cash_book(cal, market.rates, cfg.capital)
    coverage = []
    for n, d in enumerate(defs, 1):
        folder = OUT / d['pair']; folder.mkdir(exist_ok=True)
        statepath = BASE / d['pair'] / 'states.parquet'
        referencepath = CONTROL / d['pair'] / 'legacy_calendar_memory/actions.csv'
        panel = pd.read_parquet(statepath) if statepath.exists() else None
        reference = read_actions(referencepath) if referencepath.exists() else pd.Series(dtype=int, index=pd.DatetimeIndex([]))
        row = dict(pair=d['pair'], label=d['label'], rule_signals=int(reference.ne(0).sum()))
        engine = stock = None
        for model, root in zip(MODELS, [BASE / 'models', OUT / 'models']):
            dest = folder / model / 'full'; dest.mkdir(parents=True, exist_ok=True)
            if panel is None:
                actions = reference.copy(); allocation = pd.Series(dtype=float)
                scored = fallback = 0
            else:
                actions, allocation, scored, fallback = score_all(panel, reference, market.calendar, root)
            actions.to_csv(dest.parent / 'actions.csv'); allocation.to_csv(dest.parent / 'scored_allocations.csv')
            row[f'{model}_scored'] = scored; row[f'{model}_fallback'] = fallback
            row[f'{model}_signals'] = int(actions.ne(0).sum())
            row[f'{model}_skipped'] = int((reference.reindex(actions.index, fill_value=0).ne(0) & actions.eq(0)).sum())
            if (dest / 'status.json').exists():
                assert json.loads((dest / 'status.json').read_text())['status'] == 'complete'; continue
            if not actions.ne(0).any():
                write_json(dest / 'status.json', dict(status='complete', cash_only=True, entries=0))
                continue
            if engine is None:
                loader.retain(d['secids']); assets = [loader.asset(sid) for sid in d['secids']]
                quotes, stock, selections, divs = working_inputs(assets, d['secids'])
                engine = AccountingReplay(quotes, stock, selections, d['secids'], divs, cfg)
            nav, trades, decisions = engine.run(actions, cal); mark_book(nav, market.rates, cfg.capital)
            if trades.empty:
                checks = audit(nav, trades, stock, cfg, d['secids'])
                np.testing.assert_allclose(nav.nav, cash.nav, atol=1e-7, rtol=0)
                assert not decisions.status.eq('entered').any()
                decisions.to_csv(dest / 'decisions.csv', index=False)
                write_json(dest / 'status.json', dict(status='complete', cash_only=True, entries=0, **checks))
            else:
                save(dest, nav, trades, decisions, cfg, stock, d)
        coverage.append(row)
        del engine, stock; gc.collect()
        write_json(OUT / 'progress.json', dict(stage='replay', pair=n, roster=len(defs), label=d['label']))
        if n % 10 == 0: print(f'Legacy RL replay {n}/{len(defs)} {d["label"]}', flush=True)
    loader.cache.clear(); gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)


def report(defs, market, cfg):
    cal = market.calendar[market.calendar >= '2018-01-01']; cash = cash_book(cal, market.rates, cfg.capital)
    comparison = []; annual = []; pairs = []; allocations = []
    for model, root, savedmodel in [('rule_full_size', CONTROL, 'legacy_calendar_memory'),
                                  *[(m, OUT, m) for m in MODELS]]:
        book = cash[['nav', 'cash_benchmark']] * len(defs)
        entries = active = breaches = 0
        costs = dict(commission=0., option_cost=0., hedge_cost=0.)
        for d in defs:
            folder = root / d['pair'] / savedmodel / 'full'
            status = json.loads((folder / 'status.json').read_text()); assert status['status'] == 'complete'
            nav = cash if status.get('cash_only') else pd.read_parquet(folder / 'daily_nav.parquet')
            assert nav.index.equals(cal)
            book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
            entries += status['entries']; active += int(status['entries'] > 0); breaches += status.get('risk_breach_sessions', 0)
            for key in costs: costs[key] += status.get(key, 0.)
            pairs.append(dict(model=model, pair=d['pair'], label=d['label'], entries=status['entries'],
                net_excess_pnl=float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])))
            if model != 'rule_full_size':
                a = read_actions(folder.parent / 'actions.csv')
                r = read_actions(CONTROL / d['pair'] / 'legacy_calendar_memory/actions.csv') if (CONTROL / d['pair'] / 'legacy_calendar_memory/actions.csv').exists() else pd.Series(dtype=int)
                if len(a):
                    chosen = pd.read_csv(folder.parent / 'scored_allocations.csv', index_col=0, parse_dates=[0]).allocation
                    for level, count in chosen[r.reindex(chosen.index, fill_value=0).ne(0)].value_counts().items():
                        allocations.append(dict(model=model, pair=d['pair'], allocation=level, signals=int(count)))
        book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
        book.to_csv(REPORT / f'{model}_full_portfolio.csv')
        comparison.append(dict(model=model, period='full', entries=entries, active_pairs=active,
            net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
            risk_breach_pair_sessions=breaches, **costs, **scores(book, cfg.capital * len(defs))))
        for year, g in book.groupby(book.index.year):
            excess = g.excess_return
            annual.append(dict(model=model, year=year, annualized_excess_sharpe=float(np.sqrt(252) * excess.mean() / excess.std())
                if excess.std() > 1e-12 else None, return_percent=100 * ((1 + g.return_).prod() - 1),
                excess_return_percent=100 * ((1 + g.return_).prod() - (1 + cash.return_.loc[g.index]).prod())))
    result = pd.DataFrame(comparison)
    old = pd.read_csv('reports/tail_calendar_ablation/comparison.csv').query("model == 'legacy_calendar_memory'").iloc[0]
    base = result.iloc[0]
    np.testing.assert_allclose(base.annualized_excess_sharpe, old.annualized_excess_sharpe, atol=1e-12, rtol=0)
    np.testing.assert_allclose(base.net_excess_pnl, old.net_excess_pnl, atol=1e-7, rtol=0)
    assert base.entries == 135 and base.active_pairs == 41
    result.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    allocations = pd.DataFrame(allocations)
    allocations.to_csv(REPORT / 'allocations_by_pair.csv', index=False)
    allocations.groupby(['model', 'allocation']).signals.sum().reset_index().to_csv(REPORT / 'allocations.csv', index=False)
    concentration = []
    detail = pd.DataFrame(pairs)
    for model in MODELS + ['rule_full_size']:
        leaders = detail[detail.model.eq(model)].nlargest(3, 'net_excess_pnl')
        book = pd.read_csv(REPORT / f'{model}_full_portfolio.csv', index_col=0, parse_dates=[0])
        for remove in [1, 3]:
            stripped = book[['nav', 'cash_benchmark']].copy()
            for pair in leaders.pair.iloc[:remove]:
                root, savedmodel = (CONTROL, 'legacy_calendar_memory') if model == 'rule_full_size' else (OUT, model)
                nav = pd.read_parquet(root / pair / savedmodel / 'full/daily_nav.parquet')
                stripped -= nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
            stripped['return_'] = stripped.nav.pct_change().fillna(0)
            stripped['excess_return'] = stripped.return_ - stripped.cash_benchmark.pct_change().fillna(0)
            concentration.append(dict(model=model, winners_removed=remove, pairs_removed=leaders.label.iloc[:remove].tolist(),
                net_excess_pnl=float(stripped.nav.iloc[-1] - stripped.cash_benchmark.iloc[-1]),
                **scores(stripped, cfg.capital * len(defs))))
    pd.DataFrame(concentration).to_csv(REPORT / 'concentration.csv', index=False)
    print(result.to_string(index=False), flush=True)
    return result


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    original = json.loads((BASE / 'inputs.json').read_text()); defs = original['definitions']
    sources = {**original['sources'], **json.loads((BASE / 'market_sources.json').read_text())}
    sources[str(Path(__file__))] = digest(Path(__file__))
    sources['tests/test_tail_put25_legacy_rl.py'] = digest(Path('tests/test_tail_put25_legacy_rl.py'))
    protected = {}
    for study in ['tail_put25_strategy', 'tail_put25_pooled', 'tail_put25_pooled_complete',
                  'tail_calendar_ablation', 'tail_universe_change']:
        protected.update(json.loads(Path(f'reports/{study}/output_hashes.json').read_text()))
    sources.update(protected)
    for path, sha in sources.items(): assert digest(path) == sha, f'Frozen input changed: {path}'
    write_json(OUT / 'inputs.json', dict(features=FEATURES, sources=sources, seed=2026,
        protocol='Exact saved .756 legacy-calendar/memory-filtered rule proposals. Fixed direction; RL sizes 0/25/50/75/100%. Existing broad pooled policy is a diagnostic. One new monthly five-year pooled fit uses all roster pairs with completed broad spread-direction labels that passed the positive 91/365-day memory criteria at their historical signal date. Training is pooled across the broad roster, not limited to the inherited ATM panels; deployment retains exactly the .756 pair/date calendar. Optional missing states have neutral values and flags. No per-pair history or validation activation veto. Global cold-start fallback is full size. Costs, quotes, accounting, per-pair capital and hold-to-own-expiry are identical to the .756 control. No direction search, new call test, parameter sweep, or NN. Previously inspected history, not untouched holdout.'))
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    train(defs, cal)
    market = Market(cal); cfg = replace(config(), option_commission=.03)
    replay(defs, market, cfg, sources)
    result = report(defs, market, cfg)
    coverage = pd.read_csv(REPORT / 'coverage.csv'); fits = pd.read_csv(REPORT / 'fits.csv')
    for path, sha in sources.items(): assert digest(path) == sha, f'Frozen input changed: {path}'
    for model in MODELS:
        expected = pd.read_csv('reports/tail_put25_pooled_complete/coverage.csv').evaluated_observed_rows.sum()
        assert coverage[f'{model}_scored'].sum() + coverage[f'{model}_fallback'].sum() == expected
    for r in fits[fits.status.eq('enabled')].itertuples(): assert pd.Timestamp(r.last_training_outcome) < pd.Timestamp(r.fit)
    record = dict(status='complete', preserved_source_files=len(sources), baseline_reproduced=True,
        available_pairs=int(coverage[f'{MODELS[0]}_scored'].gt(0).sum()),
        scored_rows={m:int(coverage[f'{m}_scored'].sum()) for m in MODELS},
        fallback_rows={m:int(coverage[f'{m}_fallback'].sum()) for m in MODELS},
        enabled_monthly_fits=int(fits.status.eq('enabled').sum()),
        pair_model_activation_vetoes=0, elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations='Direct Sharpe policy search on standalone additive shadow books, not sequential inventory-aware RL. Training shadows retain their prior accounting, including unresolved rows excluded only after completion; final inventory uses equally repaired accounting. Broad pooled training memory-qualified experience differs from legacy-calendar execution context. Initial vega neutrality, daily delta hedging, own-expiry hold; American early assignment unmodeled. Independent $1m pair risk sleeves, no consolidated name limits; no new holdout. Original calendar combines historical screens and ATM data availability, effects not isolated.')
    write_json(REPORT / 'audit.json', record)
    (REPORT / 'findings.txt').write_text(result.to_string(index=False) + '\n\n' + json.dumps(record, indent=2))
    hashes = {str(p):digest(p) for root in [OUT, REPORT] for p in root.rglob('*')
              if p.is_file() and p.name != 'output_hashes.json'}
    hashes[str(Path(__file__))] = digest(Path(__file__))
    write_json(REPORT / 'output_hashes.json', hashes)


if __name__ == '__main__': main()
