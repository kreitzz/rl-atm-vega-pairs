"""Predeclared call/put ablations; reuse frozen labels, never change reference runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.option_flex.policy import ACTIONS, COMBOS, FLAT, Replay
from scripts.option_flex.market import FlexMarket
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.run_option_flex import BASE, LABELBASE, OUTPUT as FLEX, definitions, settings
from scripts.run_aligned_contract_learning import NEW_FEATURES, EXTRA, screen_dates, mark_book
from scripts.run_contract_surface_features import FEATURES
from scripts.run_contract_walkforward import digest
from scripts.run_broad_contract_replay import ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_selected_pair_backtest import audit, scores

OUTPUT = Path('artifacts/contracts/option_flex_ablation')
REPORT = Path('reports/option_flex_ablation')
CC = [i for i, (c, a) in enumerate(ACTIONS) if c == 'CC']
ALL = list(range(len(ACTIONS)))
MATCHED = [i for i, (c, a) in enumerate(ACTIONS) if c in ['CC', 'PP']]
DEFAULT = dict(allowed=ALL, memory='own', rows='joint', generations=50, extra_memory=False)
MATCHED_DEFAULT = dict(DEFAULT, allowed=MATCHED, rows='matched')
VARIANTS = {
    'cc_original': dict(DEFAULT, allowed=CC, memory='original', rows='cc'),
    'cc_matched_rows': dict(MATCHED_DEFAULT, allowed=CC, memory='original'),
    'same_type': dict(MATCHED_DEFAULT),
    'same_type_cc_memory': dict(MATCHED_DEFAULT, memory='original'),
    'same_type_no_sizing': dict(MATCHED_DEFAULT, allowed=[i for i in MATCHED if abs(ACTIONS[i][1]) in [0., 1.]]),
    'same_type_search_4x': dict(MATCHED_DEFAULT, generations=200),
    'same_type_memory_features': dict(MATCHED_DEFAULT, extra_memory=True),
}
FROZEN = ['frozen_CC', 'frozen_PP']
_market = None


def fit(x, daily, positions, ends, spec):
    """Same tree/CEM as both references, with explicit action subset and budget."""
    target = (daily[:, ACTIONS.index(('CC', 1.))].sum(axis=1)
              - daily[:, ACTIONS.index(('CC', -1.))].sum(axis=1)) / 2
    tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=63, random_state=2026).fit(x, target)
    leaves = np.unique(tree.apply(x)); states = np.searchsorted(leaves, tree.apply(x))
    allowed = np.asarray(spec['allowed']); labels = daily[:, allowed]; na = len(allowed); ns = len(leaves)
    first = int(positions.min() + 1); last = int(ends.max()); book = np.zeros((last-first+1, ns*na))
    offsets = np.arange(labels.shape[2])
    for state in range(ns):
        rows = np.flatnonzero(states == state); days = positions[rows, None]+1+offsets[None, :]-first
        valid = (days >= 0) & (days < len(book))
        for action in range(na):
            np.add.at(book[:, state*na+action], days[valid], labels[rows, action, :][valid])
    mean = book.mean(axis=0); covariance = np.cov(book, rowvar=False, ddof=1)
    rng = np.random.default_rng(2026); probabilities = np.full((ns, na), 1/na)
    best = np.full(ns, int(np.flatnonzero(allowed == FLAT)[0]), dtype=int); best_score = 0.
    for _ in range(spec['generations']):
        u = rng.random((192, ns))
        choices = (u[:, :, None] > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2).clip(max=na-1)
        choices[0] = best; choices[1:na+1] = np.arange(na)[:, None]
        columns = choices + np.arange(ns)[None, :]*na; m = mean[columns].sum(axis=1)
        var = covariance[columns[:, :, None], columns[:, None, :]].sum(axis=(1, 2))
        sharpe = np.sqrt(252)*np.divide(m, np.sqrt(np.maximum(var, 0)), out=np.zeros_like(m), where=var > 1e-18)
        winner = sharpe.argmax()
        if sharpe[winner] > best_score: best, best_score = choices[winner].copy(), float(sharpe[winner])
        elite = choices[np.argsort(sharpe)[-19:]]
        frequency = np.stack([(elite == a).mean(axis=0) for a in range(na)], axis=1)
        probabilities = .97*(.3*probabilities+.7*frequency)+.03/na
    return (tree, leaves, allowed[best]), dict(training_shadow_daily_sharpe=best_score)


def predict(model, x):
    tree, leaves, policy = model
    return policy[np.searchsorted(leaves, tree.apply(x))]


def eligibility(panel, spec):
    """Every entry gate uses only its signal-date state, never payoff availability."""
    out = pd.DataFrame(False, index=panel.index, columns=ALL); out[FLAT] = True
    for k in spec['allowed']:
        combo, action = ACTIONS[k]
        if not action: continue
        prefix = '' if spec['memory'] == 'original' else combo + '_'
        side = 'long' if action > 0 else 'short'
        out[k] = (panel.eligible & panel[combo+'_available']
                  & panel[f'{prefix}profit_{side}_365d'].gt(0) & panel[f'{prefix}count_{side}_365d'].ge(63)
                  & panel[f'{prefix}profit_{side}_91d'].gt(0) & panel[f'{prefix}count_{side}_91d'].ge(21)).fillna(False)
    return out


def apply_gate(codes, eligible):
    matrix = eligible.reindex(codes.index).to_numpy(bool)
    return codes.where(matrix[np.arange(len(codes)), codes.to_numpy(int)], FLAT)


def train(d, panel, daily, matched_ends, cc_ends, cal, replay, rates, folder, cfg, name):
    spec = VARIANTS[name]; ends = cc_ends if spec['rows'] == 'cc' else matched_ends
    finite = np.isfinite(daily[:, CC] if spec['rows'] == 'cc' else daily[:, MATCHED]).all(axis=(1, 2))
    start = pd.Timestamp('2018-01-01'); stop = cal[-1]-pd.Timedelta(days=cfg.max_dte+5)
    wanted = (panel.index >= start) & (panel.index <= stop) & screen_dates(d, panel.index)
    actions = pd.Series(FLAT, index=panel.index[wanted], dtype=int, name='action_code'); logs = []
    features = NEW_FEATURES+[c if c != 'commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_', 'count_'))]+FEATURES+['vix_previous_close']
    if spec['extra_memory']:
        features += [f'{c}_{kind}_{side}_{days}d' for c in ['PP']
                     for kind in ['profit', 'count'] for side in ['short', 'long'] for days in [91, 365]]
    eligible = eligibility(panel, spec)
    for month in pd.period_range(start, stop, freq='M'):
        cutoff = month.start_time; test = (actions.index >= cutoff) & (actions.index < (month+1).start_time)
        if not test.any(): continue
        vs = cutoff-pd.DateOffset(years=1); vc = cal[(cal >= vs) & (cal < cutoff)]
        tr = (panel.index >= vs-pd.DateOffset(years=5)) & (ends < vs) & finite
        row = dict(fit=str(cutoff.date()), training_rows=int(tr.sum()))
        if tr.sum() < 504 or len(vc) < 200:
            logs.append(dict(row, status='insufficient_training')); continue
        va = (panel.index >= vc[0]) & (panel.index <= vc[-1]-pd.Timedelta(days=cfg.max_dte+5))
        if not va.any(): continue
        if not eligible.loc[va, [k for k in spec['allowed'] if k != FLAT]].to_numpy().any():
            logs.append(dict(row, status='flat_no_validation_opportunities')); continue
        data = daily[tr].copy(); data[~panel.loc[tr, 'eligible'].to_numpy(bool)] = 0
        model, details = fit(panel.loc[tr, features], data, cal.get_indexer(panel.index[tr]), cal.get_indexer(ends[tr]), spec)
        raw = pd.Series(predict(model, panel.loc[va, features]), index=panel.index[va])
        proposal = apply_gate(raw, eligible); score = None; error = None
        try:
            nav, t, _ = replay.run(proposal, vc); mark_book(nav, rates, cfg.capital)
            audit(nav, t, replay.s, cfg, d['secids']); score = scores(nav, cfg.capital)['annualized_excess_sharpe']
        except (UnresolvedPosition, ValueError) as exc: error = str(exc)
        permitted = score is not None and score > 0
        if permitted:
            recent = (panel.index >= cutoff-pd.DateOffset(years=5)) & (ends < cutoff) & finite
            assert recent.sum() >= 504 and ends[recent].max() < cutoff
            data = daily[recent].copy(); data[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
            model, refit = fit(panel.loc[recent, features], data, cal.get_indexer(panel.index[recent]), cal.get_indexer(ends[recent]), spec)
            idx = actions.index[test]
            actions.loc[idx] = apply_gate(pd.Series(predict(model, panel.loc[idx, features]), index=idx), eligible)
            details.update(last_recent_outcome=str(ends[recent].max().date()), recent_training_rows=int(recent.sum()),
                           recent_shadow_sharpe=refit['training_shadow_daily_sharpe'])
        assert ends[tr].max() < vs and vc[-1] < cutoff
        logs.append(dict(row, status='enabled' if permitted else 'flat_validation_gate', validation_sharpe=score,
                         validation_error=error, last_training_outcome=str(ends[tr].max().date()), validation_start=str(vc[0].date()),
                         proposed_nonzero=int(raw.ne(FLAT).sum()), gated_nonzero=int(proposal.ne(FLAT).sum()), **details))
    pd.testing.assert_series_equal(actions, apply_gate(actions, eligible))
    pd.DataFrame(logs).to_csv(folder/'fits.csv', index=False)
    return actions


def save_book(d, name, actions, replay, cal, rates, cfg, folder):
    actions.rename('action_code').to_csv(folder/'actions.csv')
    row = dict(pair=d['pair'], label=d['label'], variant=name, nonzero_signals=int(actions.ne(FLAT).sum()))
    try:
        nav, t, dec = replay.run(actions, cal[cal >= '2018-01-01']); mark_book(nav, rates, cfg.capital)
        checks = audit(nav, t, replay.s, cfg, d['secids'])
        if len(t):
            cp = replay.q[['optionid', 'cp_flag']].drop_duplicates()
            assert not cp.optionid.duplicated().any(); t = t.merge(cp, on='optionid', validate='many_to_one')
        nav.to_parquet(folder/'daily_nav.parquet'); t.to_csv(folder/'trades.csv', index=False); dec.to_csv(folder/'decisions.csv', index=False)
        row.update(status='complete', entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,
                   **scores(nav, cfg.capital), **checks)
        if name in ['cc_original', 'frozen_CC']:
            expected = pd.read_parquet(BASE/d['pair']/'rl_sized/daily_nav.parquet')
            np.testing.assert_allclose(nav.nav, expected.nav, rtol=0, atol=1e-7)
            row['reference_control'] = 'identical'
            if name == 'cc_original':
                old = pd.read_csv(BASE/d['pair']/'actions.csv', index_col=0, parse_dates=[0]).rl_sized
                decoded = pd.Series([ACTIONS[k][1] for k in actions], index=actions.index)
                np.testing.assert_array_equal(decoded, old.reindex(decoded.index))
    except (UnresolvedPosition, ValueError) as exc: row.update(status='incomplete', error=str(exc))
    row['worker_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (folder/'status.json').write_text(json.dumps(row, indent=2)+'\n')
    return row


def worker(task):
    global _market
    d, names = task; cfg = settings(); results = []
    root = OUTPUT/d['pair']; root.mkdir(parents=True, exist_ok=True)
    todo = []
    for name in names:
        path = root/name/'status.json'
        if path.exists(): results.append(json.loads(path.read_text()))
        else: todo.append(name)
    if not todo: return results
    prior = json.loads((FLEX/d['pair']/'status.json').read_text())
    if prior['calls_control'] == 'verified_cash_only':
        # These sleeves cannot pass the unchanged beta/correlation filter.
        for name in todo:
            folder = root/name; folder.mkdir(exist_ok=True)
            row = dict(pair=d['pair'], label=d['label'], variant=name, status='complete', entries=0,
                       nav_source=str(BASE/d['pair']/'rl_sized/daily_nav.parquet'), structural_cash=True)
            (folder/'status.json').write_text(json.dumps(row, indent=2)+'\n'); results.append(row)
        return results
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    if _market is None: _market = FlexMarket(cal)
    q, s, selections, div = _market.flex_pair(d['secids']); replay = Replay(q, s, selections, d['secids'], div, cfg)
    panel = pd.read_parquet(FLEX/d['pair']/'states.parquet')
    with np.load(FLEX/d['pair']/'labels.npz', allow_pickle=False) as z:
        daily = z['daily']; cc_ends = pd.DatetimeIndex(z['CC_ends'])
        ends = pd.DatetimeIndex(np.maximum(z['CC_ends'], z['PP_ends']))
    original = pd.read_csv(BASE/d['pair']/'actions.csv', index_col=0, parse_dates=[0]).rl_sized
    try: decisions = pd.read_csv(BASE/d['pair']/'rl_sized/decisions.csv')
    except pd.errors.EmptyDataError: decisions = pd.DataFrame(columns=['status', 'signal_date'])
    entries = pd.DatetimeIndex(decisions.loc[decisions.status.eq('entered'), 'signal_date'])
    for name in todo:
        folder = root/name; folder.mkdir(exist_ok=True); started = time.monotonic()
        if name in FROZEN:
            combo = name.removeprefix('frozen_')
            # Preserve original actual entry signal dates/directions/requested size.
            # Contract availability, integer quantities and risk acceptance can differ.
            a = original.where(original.index.isin(entries), 0.)
            actions = pd.Series([ACTIONS.index((combo, float(v))) if v else FLAT for v in a], index=a.index)
        else:
            actions = train(d, panel, daily, ends, cc_ends, cal, replay, _market.rates, folder, cfg, name)
        row = save_book(d, name, actions, replay, cal, _market.rates, cfg, folder); results.append(row)
        print(f'{d["label"]} {name}: {row["status"]}, {row.get("entries", 0)} entries, {time.monotonic()-started:.1f}s', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int)
    parser.add_argument('--variants', nargs='+', choices=FROZEN+list(VARIANTS), default=FROZEN+list(VARIANTS))
    args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    definitions_ = definitions(); old = json.loads((FLEX/'inputs.json').read_text())
    assert digest('data/call_put/manifest.json') == old['put_manifest']
    assert digest('data/contract_history_shared/manifest.json') == old['call_cache']
    for p, h in old['code'].items(): assert digest(p) == h, p
    for p, h in old['reference_files'].items(): assert digest(p) == h, p
    fingerprint = dict(variants=VARIANTS, frozen=FROZEN, settings=asdict(settings()), definitions=definitions_,
                       previous_inputs=digest(FLEX/'inputs.json'), reference=digest('config/current_contract_reference.json'),
                       code={str(p): digest(p) for p in [Path(__file__), Path('scripts/run_contract_walkforward.py')]},
                       sources={str(p): digest(p) for d in definitions_ for p in [FLEX/d['pair']/'states.parquet',
                           *([FLEX/d['pair']/'labels.npz'] if (FLEX/d['pair']/'labels.npz').exists() else []),
                           BASE/d['pair']/'rl_sized/decisions.csv']})
    path = OUTPUT/'inputs.json'
    if path.exists(): assert json.loads(path.read_text()) == json.loads(json.dumps(fingerprint)), 'Inputs changed'
    path.write_text(json.dumps(fingerprint, indent=2)+'\n'); rows = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(worker, [(d, args.variants) for d in definitions_[:args.limit]]), 1):
            rows.extend(result); pd.DataFrame(rows).to_csv(REPORT/'status.csv', index=False)
            print(f'{i}/{len(definitions_)} pairs finished', flush=True)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started, pairs=len(rows)//len(args.variants), workers=2), indent=2)+'\n')


if __name__ == '__main__': main()
