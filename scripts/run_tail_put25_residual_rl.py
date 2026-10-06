"""Three predeclared allocation tests around the frozen .756 put-tail rule."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import scripts.run_tail_put25_legacy_rl as reference
from scripts.analyze_tail_universe_change import BASE, working_inputs
from scripts.tail_put25_pooled import FEATURES, aggregate_chunk
from scripts.tail_put25_rule_sizing import SIZE_CODES
from scripts.repair_tail_put25_pooled import AccountingReplay, save
from scripts.run_tail_put25_rule_sizing import PutMarket
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.curve_trading.config import config, RUNOFF
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.run_contract_walkforward import digest
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit

OUT = Path('artifacts/contracts/tail_put25_residual_rl')
REPORT = Path('reports/tail_put25_residual_rl')
PREVIOUS = Path('artifacts/contracts/tail_put25_legacy_rl_all_months')
MODELS = ['residual_floor_rl', 'robust_residual_rl', 'context_residual_rl']
LEVELS = np.array([2, 3, 4])  # 50%, 75%, 100%; never replace an eligible signal with flat.


def context_mask(dates, panel_dates, evaluation_dates):
    dates = pd.DatetimeIndex(dates)
    return np.where(dates < pd.Timestamp('2018-01-01'), dates.isin(panel_dates), dates.isin(evaluation_dates))


def moments(book, dates, robust):
    result = [(book.mean(axis=0), np.cov(book, rowvar=False, ddof=1))]
    if robust:
        years, counts = np.unique(pd.DatetimeIndex(dates).year, return_counts=True)
        folds = []
        for year in years[counts >= 63]:
            other = book[pd.DatetimeIndex(dates).year != year]
            if len(other) >= 252:
                folds.append((other.mean(axis=0), np.cov(other, rowvar=False, ddof=1)))
        if len(folds) >= 2: result = folds
    return result


def score_choices(choices, stats, robust=False):
    choices = np.asarray(choices, dtype=int)
    cols = choices + np.arange(choices.shape[1])[None, :] * 5
    scores = []
    baseline = np.arange(choices.shape[1]) * 5 + 4
    for mean, cov in stats:
        m = mean[cols].sum(axis=1)
        v = cov[cols[:, :, None], cols[:, None, :]].sum(axis=(1, 2))
        sharpe = np.sqrt(252) * np.divide(m, np.sqrt(np.maximum(v, 0)), out=np.zeros_like(m), where=v > 1e-18)
        if robust:
            bv = cov[np.ix_(baseline, baseline)].sum()
            bs = np.sqrt(252) * mean[baseline].sum() / np.sqrt(bv) if bv > 1e-18 else 0.
            sharpe -= bs
        scores.append(sharpe)
    scores = np.asarray(scores)
    return scores.mean(axis=0) - (scores.std(axis=0, ddof=1) if robust and len(stats) > 1 else 0.)


def fit_allocations(book, dates, tree, leaves, robust=False):
    stats = moments(book, dates, robust)
    rng = np.random.default_rng(2026); ns = len(leaves)
    prob = np.full((ns, len(LEVELS)), 1 / len(LEVELS))
    best = np.full(ns, 4); best_score = float(score_choices(best[None], stats, robust)[0])
    for _ in range(50):
        aliases = (rng.random((192, ns))[:, :, None] > prob.cumsum(axis=1)[None]).sum(axis=2).clip(max=2)
        choice = LEVELS[aliases]; choice[0] = best; choice[1:4] = LEVELS[:, None]
        score = score_choices(choice, stats, robust)
        winner = score.argmax()
        if score[winner] > best_score:
            best = choice[winner].copy(); best_score = float(score[winner])
        elite = choice[np.argsort(score)[-19:]]
        frequency = np.stack([(elite == level).mean(axis=0) for level in LEVELS], axis=1)
        prob = .97 * (.3 * prob + .7 * frequency) + .03 / len(LEVELS)
    full_stats = moments(book, dates, False)
    return (tree, leaves, np.array(SIZE_CODES)[best]), dict(
        training_objective=best_score, objective='mean_minus_one_sd_of_leave_year_out_sharpe_improvement' if robust else 'daily_sharpe',
        training_shadow_daily_sharpe=float(score_choices(best[None], full_stats)[0]),
        baseline_shadow_daily_sharpe=float(score_choices(np.full((1, ns), 4), full_stats)[0]),
        training_folds=len(stats))


def shared_book(data, cal, tree, leaves):
    states = np.searchsorted(leaves, tree.apply(data[FEATURES]))
    first, last = int(data.position.min()+1), int(data.end_position.max())
    book = np.zeros((last-first+1, len(leaves)*5))
    for pair, rows in data.groupby('pair', sort=False).groups.items():
        rows = np.asarray(rows)
        disk = np.load(BASE / pair / 'labels.npy', mmap_mode='r')
        labels = disk[data.loc[rows, 'label_row'].to_numpy(int)]
        aggregate_chunk(book, states[rows], labels, data.loc[rows, 'position'].to_numpy(int), first)
        del disk, labels
    return book, cal[first:last+1]


def metadata(defs):
    parts = []; rows = []
    for d in defs:
        path = BASE / d['pair'] / 'training_rows.parquet'
        if not path.exists(): continue
        meta = pd.read_parquet(path)
        old = Path('artifacts/contracts/tail_put25_strategy') / d['pair'] / 'states.parquet'
        actions = reference.CONTROL / d['pair'] / 'legacy_calendar_memory/actions.csv'
        pre = pd.read_parquet(old, columns=[]).index if old.exists() else pd.DatetimeIndex([])
        post = read_actions(actions).index if actions.exists() else pd.DatetimeIndex([])
        meta['strategy_context'] = context_mask(meta.date, pre, post)
        eligible = reference.memory_filter(meta) & meta.finite_label
        rows.append(dict(pair=d['pair'], label=d['label'], memory_qualified=int(eligible.sum()),
            context_qualified=int((eligible & meta.strategy_context).sum()),
            post2018_memory_qualified=int((eligible & meta.date.ge('2018-01-01')).sum()),
            post2018_context_qualified=int((eligible & meta.strategy_context & meta.date.ge('2018-01-01')).sum())))
        parts.append(meta)
    pd.DataFrame(rows).to_csv(REPORT / 'training_context.csv', index=False)
    return pd.concat(parts, ignore_index=True)


def train(defs, cal):
    meta = metadata(defs); logs = []
    for month in pd.period_range('2018-01-01', cal[-1]-pd.Timedelta(days=RUNOFF), freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        broad = meta.loc[reference.training_selection(meta, cutoff)].reset_index(drop=True)
        assert pd.Timestamp(broad.end.max()) < cutoff
        oldtree, leaves, _ = joblib.load(PREVIOUS / 'models' / f'{fitdate}.joblib')
        book, dates = shared_book(broad, cal, oldtree, leaves)
        for model in MODELS[:2]:
            folder = OUT / model / 'models'; folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{fitdate}.joblib'; info = folder / f'{fitdate}.json'
            if info.exists(): logs.append(json.loads(info.read_text())); continue
            policy, details = fit_allocations(book, dates, oldtree, leaves, model == 'robust_residual_rl')
            row = dict(model=model, fit=fitdate, status='enabled', training_rows=len(broad), training_pairs=int(broad.pair.nunique()),
                last_training_outcome=str(pd.Timestamp(broad.end.max()).date()), leaves=len(leaves),
                leaf_allocations=[SIZE_CODES.index(int(a))/4 for a in policy[2]], **details)
            joblib.dump(policy, path); write_json(info, row); logs.append(row)
        del book
        model = MODELS[2]; folder = OUT / model / 'models'; folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{fitdate}.joblib'; info = folder / f'{fitdate}.json'
        if info.exists(): logs.append(json.loads(info.read_text()))
        else:
            context = broad.loc[broad.strategy_context].reset_index(drop=True)
            assert len(context) >= 16, 'Too little pooled context: stop rather than hide a fallback'
            minleaf = max(8, min(63, len(context)//8))
            tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=minleaf, random_state=2026).fit(context[FEATURES], context.target)
            leaves = np.unique(tree.apply(context[FEATURES]))
            book, dates = shared_book(context, cal, tree, leaves)
            policy, details = fit_allocations(book, dates, tree, leaves)
            row = dict(model=model, fit=fitdate, status='enabled', training_rows=len(context), training_pairs=int(context.pair.nunique()),
                last_training_outcome=str(pd.Timestamp(context.end.max()).date()), leaves=len(leaves), min_leaf=minleaf,
                leaf_allocations=[SIZE_CODES.index(int(a))/4 for a in policy[2]], **details)
            joblib.dump(policy, path); write_json(info, row); logs.append(row)
            del context, tree, book
        print(f'Residual fits {fitdate}: broad {len(broad)} rows, context {logs[-1]["training_rows"]}; all three enabled', flush=True)
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)
    del meta; gc.collect()


def replay(defs, market, cfg, sources):
    loader = PutMarket(market, cfg, sources); cal = market.calendar[market.calendar >= '2018-01-01']
    cash = cash_book(cal, market.rates, cfg.capital); coverage = []
    for n, d in enumerate(defs, 1):
        path = BASE / d['pair'] / 'states.parquet'
        panel = pd.read_parquet(path) if path.exists() else None
        path = reference.CONTROL / d['pair'] / 'legacy_calendar_memory/actions.csv'
        base = read_actions(path) if path.exists() else pd.Series(dtype=int, index=pd.DatetimeIndex([]))
        row = dict(pair=d['pair'], label=d['label'], rule_signals=int(base.ne(0).sum()))
        engine = stock = None
        for model in MODELS:
            folder = OUT / d['pair'] / model; dest = folder / 'full'; dest.mkdir(parents=True, exist_ok=True)
            if panel is None:
                actions = base.copy(); allocation = pd.Series(dtype=float); scored = fallback = 0
            else:
                actions, allocation, scored, fallback = reference.score_all(panel, base, market.calendar, OUT / model / 'models')
                assert fallback == 0
                assert actions.ne(0).sum() == base.ne(0).sum()
                assert allocation.dropna().ge(.5).all()
            actions.to_csv(folder / 'actions.csv'); allocation.to_csv(folder / 'scored_allocations.csv')
            row[f'{model}_scored'] = scored; row[f'{model}_fallback'] = fallback
            if (dest / 'status.json').exists():
                assert json.loads((dest / 'status.json').read_text())['status'] == 'complete'; continue
            if not actions.ne(0).any():
                write_json(dest / 'status.json', dict(status='complete', cash_only=True, entries=0)); continue
            if engine is None:
                loader.retain(d['secids']); assets = [loader.asset(sid) for sid in d['secids']]
                quotes, stock, selections, divs = working_inputs(assets, d['secids'])
                engine = AccountingReplay(quotes, stock, selections, d['secids'], divs, cfg)
            nav, trades, decisions = engine.run(actions, cal); mark_book(nav, market.rates, cfg.capital)
            if trades.empty:
                checks = audit(nav, trades, stock, cfg, d['secids'])
                np.testing.assert_allclose(nav.nav, cash.nav, atol=1e-7, rtol=0)
                decisions.to_csv(dest / 'decisions.csv', index=False)
                write_json(dest / 'status.json', dict(status='complete', cash_only=True, entries=0, **checks))
            else: save(dest, nav, trades, decisions, cfg, stock, d)
        coverage.append(row); del engine, stock; gc.collect()
        write_json(OUT / 'progress.json', dict(stage='replay', pair=n, roster=len(defs), label=d['label']))
        if n % 10 == 0: print(f'Residual replay {n}/{len(defs)} {d["label"]}', flush=True)
    loader.cache.clear(); gc.collect(); pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    original = json.loads((PREVIOUS / 'inputs.json').read_text()); sources = dict(original['sources'])
    sources.update(json.loads(Path('reports/tail_put25_legacy_rl_all_months/output_hashes.json').read_text()))
    for p in [Path(__file__), Path('tests/test_tail_put25_residual_rl.py')]: sources[str(p)] = digest(p)
    for path, sha in sources.items(): assert digest(path) == sha, f'Prior input changed: {path}'
    defs = json.loads((BASE / 'inputs.json').read_text())['definitions']
    signature = dict(sources=sources, models=MODELS, seed=2026, sizes=[.5,.75,1.],
        protocol='Three predeclared bounded tests; no new signal, horizon, direction or cost search. All share exact .756 rule proposals. Floor RL and robust floor RL reuse the prior memory-trained monthly state partitions and pool. Robust objective is mean minus one sample standard deviation of leave-calendar-year-out Sharpe improvement over full-size, evaluated only on completed training outcomes. Context floor RL uses historical original put-state dates before 2018 and exact saved legacy evaluation dates thereafter, and adaptively fits up to 8 pooled leaves (min leaf max(8,min(63,n//8))). No per-pair activation, no cold-start fallback, all observed states scored. Every eligible signal remains at least half size; actual risk/quote controls still apply. Training remains additive standalone cohorts; no claim of sequential inventory-aware RL. Inspected history, not new holdout. Pre-2018 panel availability and candidate roster remain retrospective dataset conditioning.')
    write_json(OUT / 'inputs.json', signature)
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    train(defs, cal); market = Market(cal); cfg = replace(config(), option_commission=.03)
    replay(defs, market, cfg, sources)
    # Reuse the already audited report with scoped destinations; no source edits.
    reference.OUT, reference.REPORT, reference.MODELS = OUT, REPORT, MODELS
    result = reference.report(defs, market, cfg)
    prior = pd.read_csv('reports/tail_put25_legacy_rl_all_months/comparison.csv').query("model == 'memory_trained_pooled_rl'")
    result = pd.concat([result, prior], ignore_index=True); result.to_csv(REPORT / 'comparison.csv', index=False)
    fits = pd.read_csv(REPORT / 'fits.csv'); coverage = pd.read_csv(REPORT / 'coverage.csv')
    assert len(fits) == 270 and fits.status.eq('enabled').all()
    for r in fits.itertuples(): assert pd.Timestamp(r.last_training_outcome) < pd.Timestamp(r.fit)
    for model in MODELS:
        assert coverage[f'{model}_scored'].sum() == 136459
        assert coverage[f'{model}_fallback'].sum() == 0
    for path, sha in sources.items(): assert digest(path) == sha, f'Prior input changed: {path}'
    write_json(OUT / 'market_sources.json', sources)
    record = dict(status='complete', baseline_reproduced=True, source_files_verified=len(sources),
        scored_rows_per_model=136459, scored_pairs=332, models_per_variant=90, per_pair_activation_vetoes=0,
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=signature['protocol'])
    write_json(REPORT / 'audit.json', record)
    (REPORT / 'findings.txt').write_text(result.to_string(index=False)+'\n\n'+json.dumps(record, indent=2))
    hashes = {str(p):digest(p) for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'}
    hashes[str(Path(__file__))] = digest(Path(__file__))
    write_json(REPORT / 'output_hashes.json', hashes)


if __name__ == '__main__': main()
