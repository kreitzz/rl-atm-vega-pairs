"""Bounded full-roster shared tail sizing experiment, without pair model vetoes."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd

from scripts.tail_put25_pooled import FEATURES, state, rule_codes, completed_mask, partition, aggregate_chunk, optimize_book
from scripts.tail_put25_rule_sizing import sizing_labels, translate
from scripts.tail_put25_policy import COMBO
from scripts.curve_trading.config import config, RUNOFF
from scripts.curve_trading.policy import Replay, predict
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.contract_learning_labels import beta_screen
from scripts.retrain_atm_commission import fast_memory
from scripts.run_tail_put25_rule_sizing import PutMarket
from scripts.run_atm65_volume_oi import configure, EMPLOYEE, CHECKPOINT
from scripts.run_atm65_ablation import write_json
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_aligned_contract_learning import asset_paths, mark_book
from scripts.replay_frozen_atm_commission import cash_book, save_result
from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition

OUT = Path('artifacts/contracts/tail_put25_pooled')
REPORT = Path('reports/tail_put25_pooled')
MODELS = ['ungated_spread', 'pooled_rl_sized']


def prior_vix(cal):
    raw = pd.read_csv('data/raw/VIX_History.csv')
    raw['DATE'] = pd.to_datetime(raw.DATE)
    joined = pd.merge_asof(pd.DataFrame({'date': cal}), raw[['DATE', 'CLOSE']].sort_values('DATE'),
        left_on='date', right_on='DATE', direction='backward', allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7))
    return pd.Series(joined.CLOSE.to_numpy(), index=cal)


def prepare(d, market, loader, training, vix, sources):
    folder = OUT / d['pair']; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'coverage.json').exists():
        return
    cal = market.calendar
    loader.retain(d['secids'])
    assets = [loader.asset(sid) for sid in d['secids']]
    if any(a['sel'].empty or a['q'].empty for a in assets):
        write_json(folder / 'coverage.json', dict(pair=d['pair'], label=d['label'], observed_rows=0,
            evaluated_observed_rows=0, beta_eligible_rows=0, spread_signal_rows=0,
            training_candidates=0, reason='missing_cached_put_selection_or_history'))
        return
    paths = asset_paths(d)
    returns = []
    for p, a, sid in zip(paths, assets, d['secids']):
        if p.exists():
            sources[str(p)] = digest(p)
            r = pd.read_parquet(p, columns=['ret']).ret.reindex(cal)
        else:
            s = a['s'].set_index('date')
            r = (s.close * s.cfadj).pct_change(fill_method=None)
        returns.append(r.rename(sid))
    returns = pd.concat(returns, axis=1)
    screen = beta_screen(returns, cal, beta_tolerance=.15)
    units = [UnitPaths(a['q'], a['s'], a['sel'], cal, cal, a['divs'], training) for a in assets]
    labels, ends, vegas, statuses = pair_labels(*units, COMBO, training)
    reward = np.divide(labels[:, [0, -1]].sum(axis=2) * training.entry_vega_per_leg,
        vegas[:, [0, -1]], out=np.full((len(cal), 2), np.nan), where=vegas[:, [0, -1]] > 0)
    memory = fast_memory(cal, cal[ends], reward, vegas[:, [0, -1]] > 0)
    panel = state(assets[0]['sel'], assets[1]['sel'], returns, screen, cal, memory, vix, .65)
    rule = rule_codes(panel)
    panel['rule_code'] = rule
    panel.to_parquet(folder / 'states.parquet')
    full = np.zeros((len(cal), 9, labels.shape[-1]), dtype=np.float32)
    full[:, 1:] = labels
    candidates = rule.ne(0).to_numpy()
    alloc = sizing_labels(full[candidates], rule.loc[candidates])
    np.save(folder / 'labels.npy', alloc)
    meta = panel.loc[candidates, FEATURES].copy()
    meta['date'] = meta.index
    meta['end'] = cal[ends[candidates]]
    meta['position'] = np.flatnonzero(candidates)
    meta['end_position'] = ends[candidates]
    meta['label_row'] = np.arange(len(meta))
    meta['finite_label'] = np.isfinite(alloc).all(axis=(1, 2))
    meta['target'] = alloc[:, -1].sum(axis=1) / 2
    meta['pair'] = d['pair']
    meta.to_parquet(folder / 'training_rows.parquet', index=False)
    evaluate = (cal >= '2018-01-01') & (cal <= cal[-1] - pd.Timedelta(days=RUNOFF))
    write_json(folder / 'coverage.json', dict(pair=d['pair'], label=d['label'],
        observed_rows=int(panel.observed.sum()), evaluated_observed_rows=int((panel.observed & evaluate).sum()),
        beta_eligible_rows=int((panel.observed & panel.eligible & evaluate).sum()),
        spread_signal_rows=int((rule.ne(0) & evaluate).sum()), training_candidates=int(candidates.sum()),
        resolved_training_candidates=int(meta.finite_label.sum()), unresolved_shadow_rows=int((statuses == 'unresolved').sum()),
        inherited_atm_panel_exists=(EMPLOYEE / d['pair'] / 'states.parquet').exists()))
    del units, labels, full, alloc
    gc.collect()


def train(defs, cal):
    parts = [pd.read_parquet(OUT / d['pair'] / 'training_rows.parquet')
             for d in defs if (OUT / d['pair'] / 'training_rows.parquet').exists()]
    meta = pd.concat(parts, ignore_index=True)
    logs = []
    stop = cal[-1] - pd.Timedelta(days=RUNOFF)
    folder = OUT / 'models'; folder.mkdir(exist_ok=True)
    for month in pd.period_range('2018-01-01', stop, freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        path = folder / f'{fitdate}.joblib'; info = folder / f'{fitdate}.json'
        if info.exists():
            logs.append(json.loads(info.read_text())); continue
        use = completed_mask(meta.date, meta.end, cutoff) & meta.finite_label
        data = meta.loc[use].reset_index(drop=True)
        record = dict(fit=fitdate, training_rows=len(data), training_pairs=int(data.pair.nunique()))
        if len(data) < 504:
            record['status'] = 'global_insufficient_training'
        else:
            tree, leaves = partition(data[FEATURES], data.target)
            states = np.searchsorted(leaves, tree.apply(data[FEATURES]))
            first, last = int(data.position.min() + 1), int(data.end_position.max())
            book = np.zeros((last - first + 1, len(leaves) * 5))
            for pair, rows in data.groupby('pair', sort=False).groups.items():
                rows = np.asarray(rows)
                disk = np.load(OUT / pair / 'labels.npy', mmap_mode='r')
                labels = disk[data.loc[rows, 'label_row'].to_numpy(int)]
                aggregate_chunk(book, states[rows], labels, data.loc[rows, 'position'].to_numpy(int), first)
                del disk, labels
            model, details = optimize_book(book, leaves, tree)
            assert pd.Timestamp(data.end.max()) < cutoff
            record.update(status='enabled', last_training_outcome=str(pd.Timestamp(data.end.max()).date()),
                leaves=len(leaves), leaf_size_aliases=model[2].tolist(),
                split_features=[FEATURES[int(i)] for i in tree.tree_.feature if i >= 0], **details)
            joblib.dump(model, path)
            del data, tree, book, model
        write_json(info, record); logs.append(record)
        print(f'Pooled fit {fitdate}: {record["status"]}; {record["training_rows"]:,} observations, {record["training_pairs"]} pairs', flush=True)
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)


def score_states(panel, cal):
    wanted = (panel.index >= '2018-01-01') & (panel.index <= cal[-1] - pd.Timedelta(days=RUNOFF))
    dates = panel.index[wanted]
    rule = panel.loc[dates, 'rule_code'].astype(int).rename('code')
    output = pd.Series(0, index=dates, name='code', dtype=int)
    allocation = pd.Series(np.nan, index=dates, name='allocation')
    scored = 0
    for month in dates.to_period('M').unique():
        path = OUT / 'models' / f'{month.start_time.date()}.joblib'
        if not path.exists():
            continue
        idx = dates[(dates.to_period('M') == month) & panel.loc[dates, 'observed'].to_numpy(bool)]
        if not len(idx):
            continue
        model = joblib.load(path)
        aliases = predict(model, panel.loc[idx, FEATURES])
        output.loc[idx] = translate(aliases, rule.loc[idx])
        from scripts.tail_put25_rule_sizing import SIZE_CODES
        allocation.loc[idx] = [SIZE_CODES.index(int(c)) / 4 for c in aliases]
        scored += len(idx)
    return rule, output, allocation, scored


def replay(d, market, loader, actual):
    folder = OUT / d['pair']
    if all((folder / m / 'full/status.json').exists() for m in MODELS):
        return
    coverage = json.loads((folder / 'coverage.json').read_text())
    if not (folder / 'states.parquet').exists():
        for m in MODELS:
            write_json(folder / m / 'full/status.json', dict(status='complete', pair=d['pair'], label=d['label'],
                cash_only=True, reason=coverage['reason'], entries=0))
        return
    panel = pd.read_parquet(folder / 'states.parquet')
    rule, output, allocation, scored = score_states(panel, market.calendar)
    coverage.update(scored_rows=scored, rl_signals=int(output.ne(0).sum()),
        rl_skipped=int((rule.ne(0) & output.eq(0)).sum()),
        rl_downsized=int((output.ne(0) & output.ne(rule)).sum()))
    write_json(folder / 'coverage.json', coverage)
    allocation.to_csv(folder / 'scored_allocations.csv')
    loader.retain(d['secids'])
    assets = [loader.asset(sid) for sid in d['secids']]
    q, s, selection, divs = [pd.concat([a[col] for a in assets], ignore_index=True) for col in ['q', 's', 'sel', 'divs']]
    engine = Replay(q, s, selection, d['secids'], divs, actual)
    cal = market.calendar[market.calendar >= '2018-01-01']
    for m, codes in zip(MODELS, [rule, output]):
        dest = folder / m / 'full'; dest.mkdir(parents=True, exist_ok=True)
        codes.to_csv(dest.parent / 'actions.csv')
        if (dest / 'status.json').exists():
            continue
        if not codes.ne(0).any():
            write_json(dest / 'status.json', dict(status='complete', pair=d['pair'], label=d['label'],
                entries=0, cash_only=True, reason='signal_or_policy_flat'))
            continue
        try:
            nav, trades, decisions = engine.run(codes, cal)
            mark_book(nav, market.rates, actual.capital)
            result = save_result(dest, nav, trades, decisions, actual, s, d)
            assert trades.empty or trades.optionid.isin(selection.optionid).all()
            result.update(decision_counts=decisions.status.value_counts().to_dict(),
                maximum_gross_vega=float(nav.gross_vega.max()), risk_breach_sessions=int(nav.post_trade_risk_breach.sum()))
            write_json(dest / 'status.json', result)
        except UnresolvedPosition as exc:
            write_json(dest / 'status.json', dict(status='incomplete', pair=d['pair'], label=d['label'], error=str(exc)))
            if hasattr(exc, 'partial_nav'):
                exc.partial_nav.to_parquet(dest / 'partial_nav.parquet')
    del engine, assets, q, s, selection, divs
    gc.collect()


def report(defs, market, cfg):
    cal = market.calendar[market.calendar >= '2018-01-01']
    cash = cash_book(cal, market.rates, cfg.capital)
    comparison, annual, pairs, incomplete = [], [], [], []
    coverage = pd.DataFrame([json.loads((OUT / d['pair'] / 'coverage.json').read_text()) for d in defs])
    coverage.to_csv(REPORT / 'coverage.csv', index=False)
    for model in MODELS:
        book = cash[['nav', 'cash_benchmark']] * len(defs)
        entries = active = 0; costs = dict(commission=0., option_cost=0., hedge_cost=0.)
        for d in defs:
            folder = OUT / d['pair'] / model / 'full'
            status = json.loads((folder / 'status.json').read_text())
            if status['status'] != 'complete':
                incomplete.append(dict(model=model, pair=d['pair'], error=status.get('error')))
                continue
            nav = cash if status.get('cash_only') else pd.read_parquet(folder / 'daily_nav.parquet')
            assert nav.index.equals(cal)
            book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
            entries += status['entries']; active += int(status['entries'] > 0)
            for key in costs:
                costs[key] += status.get(key, 0.)
            pairs.append(dict(model=model, pair=d['pair'], label=d['label'], entries=status['entries'],
                net_excess_pnl=float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])))
        if any(r['model'] == model for r in incomplete):
            comparison.append(dict(model=model, period='full', status='incomplete'))
            continue
        book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
        book.to_csv(REPORT / f'{model}_full_portfolio.csv')
        comparison.append(dict(model=model, period='full', status='complete', pairs=len(defs),
            entries=entries, active_pairs=active, net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
            **costs, **scores(book, cfg.capital * len(defs))))
        recent = book.loc['2022-01-01':]
        excess = recent.excess_return
        comparison.append(dict(model=model, period='recent_continuing', status='complete', pairs=len(defs),
            annualized_excess_sharpe=float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 1e-12 else None,
            net_excess_pnl=float((recent.nav - recent.cash_benchmark).iloc[-1] - (recent.nav - recent.cash_benchmark).iloc[0])))
        for year, g in book.groupby(book.index.year):
            excess = g.excess_return
            annual.append(dict(model=model, year=year, return_percent=100 * ((1 + g.return_).prod() - 1),
                excess_return_percent=100 * ((1 + g.return_).prod() -
                    (1 + book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod()),
                annualized_excess_sharpe=float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 1e-12 else None))
    pd.DataFrame(comparison).to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    frozen = pd.read_csv('reports/tail_put25_strategy/comparison.csv').query("model == 'rule'")
    frozen.to_csv(REPORT / 'frozen_rule_reference.csv', index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return incomplete, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-limit', type=int)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(); configure()
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    actual = replace(config(), option_commission=.03)
    training = replace(actual, option_commission=.65)
    defs = json.loads((EMPLOYEE / 'inputs.json').read_text())['definitions']
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    files = [Path(__file__), Path('scripts/tail_put25_pooled.py'), Path('tests/test_tail_put25_pooled.py'),
             EMPLOYEE / 'inputs.json', CHECKPOINT, Path('scripts/curve_trading/unit_paths.py'),
             Path('scripts/curve_trading/policy.py'), Path('scripts/curve_trading/config.py'),
             Path('scripts/option_flex/contract_engine.py'), Path('scripts/option_flex/contract_risk.py'),
             Path('scripts/run_tail_put25_rule_sizing.py'), Path('scripts/tail_put25_rule_sizing.py'),
             Path('scripts/tail_put25_policy.py'), Path('scripts/retrain_atm_commission.py'),
             Path('scripts/contract_learning_labels.py'), Path('data/raw/VIX_History.csv'),
             ROOT / 'zero_curve.parquet', ROOT / 'distributions.parquet',
             *[ROOT / 'underlying' / f'year={y}.parquet' for y in cal.year.unique()]]
    signature = dict(actual=asdict(actual), training=asdict(training), definitions=defs, features=FEATURES,
        sources={str(p): digest(p) for p in files}, seed=2026,
        protocol='All 345 roster pairs reconstructed from cached put25 quotes, not inherited ATM panels or old screen-year membership. Monthly globally pooled five-year completed-outcome fits, eight leaves and 0/25/50/75/100% fixed-signal sizing. No pair history minimum, no validation activation veto, no positive memory veto. Memory is an input. Optional missing signals use neutral values plus flags. Core spread rank uses 252 sessions and 63 observations, direction from bottom/top quintile. Retain causal monthly beta 0.85-1.15/correlation >=0.5. The signal may be flat, but all observed pair states are scored. Exact same 0.65 training commission and 0.03 actual costs/risk settings. Ungated full-size spread is matched control; frozen 0.711 gated rule is a separate historical reference. Actual inventory remains separate capped $1m pair sleeves, not global cross-pair netting. No NN retraining or parameter sweep. Recent reporting slices a continuing 2018 book, not a reset 2022 book. Inspected history, not untouched holdout.')
    manifest = OUT / 'inputs.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, 'Inputs changed'
    else:
        write_json(manifest, signature)
    sourcefile = OUT / 'market_sources.json'
    sources = json.loads(sourcefile.read_text()) if sourcefile.exists() else {}
    market = Market(cal); loader = PutMarket(market, actual, sources)
    if not args.report_only:
        vix = prior_vix(cal); newly = 0
        for n, d in enumerate(defs, 1):
            if (OUT / d['pair'] / 'coverage.json').exists():
                continue
            prepare(d, market, loader, training, vix, sources)
            write_json(sourcefile, sources); newly += 1
            print(f'Prepare {n}/{len(defs)} {d["label"]}; peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MiB', flush=True)
            if args.prepare_limit and newly >= args.prepare_limit:
                return
        loader.cache.clear(); gc.collect()
        train(defs, cal)
        for n, d in enumerate(defs, 1):
            replay(d, market, loader, actual)
            write_json(sourcefile, sources)
            write_json(OUT / 'progress.json', dict(stage='replay', pair=n, roster=len(defs), label=d['label'],
                elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'Replay {n}/{len(defs)} {d["label"]}; peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MiB', flush=True)
    loader.cache.clear(); gc.collect()
    incomplete, coverage = report(defs, market, actual)
    for path, sha in {**signature['sources'], **sources}.items():
        assert digest(path) == sha, f'Source changed: {path}'
    reference = json.loads(CHECKPOINT.read_text())
    assert digest(reference['archive']) == reference['archive_sha256']
    for path, sha in reference['files'].items():
        assert digest(path) == sha, f'Frozen ATM changed: {path}'
    verified = {}
    for study in ['tail_put25_strategy', 'tail_put25_rule_sizing', 'tail_put25_atm_transfer']:
        hashes = json.loads(Path(f'reports/{study}/output_hashes.json').read_text())
        for path, sha in hashes.items():
            assert digest(path) == sha, f'Prior study changed: {path}'
        verified[study] = len(hashes)
    fits = pd.read_csv(REPORT / 'fits.csv')
    record = dict(status='complete' if not incomplete else 'accounting_incomplete', unresolved=incomplete,
        requested_pairs=len(defs), observed_pairs=int(coverage.evaluated_observed_rows.gt(0).sum()),
        scored_pairs=int(coverage.scored_rows.fillna(0).gt(0).sum()),
        observed_rows=int(coverage.evaluated_observed_rows.sum()), scored_rows=int(coverage.scored_rows.fillna(0).sum()),
        enabled_monthly_fits=int(fits.status.eq('enabled').sum()), pair_validation_vetoes=0, profitability_vetoes=0,
        frozen_atm_files_verified=len(reference['files']), previous_study_files_verified=verified,
        elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations='Shared direct-Sharpe allocation policy using additive training trajectories, actual capped inventory on evaluation. Initial vega neutrality; daily delta hedge; hold each leg to expiry; American early exercise/assignment not modeled. Separate pair risk sleeves; no global inventory optimization. Candidate roster historically selected, not all listed options. Previously inspected sample, no new holdout.')
    write_json(REPORT / 'audit.json', record)
    write_json(OUT / 'task_status.json', record)
    text = ['Full-roster pooled tail allocation', '', signature['protocol'], '',
            pd.read_csv(REPORT / 'comparison.csv').to_string(index=False), '', json.dumps(record, indent=2)]
    (REPORT / 'findings.txt').write_text('\n'.join(text) + '\n')
    hashes = {str(p): digest(p) for root in [OUT, REPORT] for p in sorted(root.rglob('*')) if p.is_file()}
    hashes.update({str(p): digest(p) for p in [Path(__file__), Path('scripts/tail_put25_pooled.py'), Path('tests/test_tail_put25_pooled.py')]})
    write_json(REPORT / 'output_hashes.json', hashes)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
