"""Bounded feature ablations on frozen proposals, with completed-outcome fits."""
import argparse
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

from scripts import run_p25_action_windows as base
from scripts.p25_action_windows import fit_menu, score_model
from scripts.p25_recency import completed_months
from scripts.tail_feature_exploration import return_shape, surface_states, past_z
from scripts.atm65_mined_features import iv_factor_scores
from scripts import atm65_volume_oi_features as volume
from scripts import atm65_transcript_features as transcripts
from scripts import run_atm65_employee_sentiment as employee
from scripts.curve_trading.config import DATA

OUT = Path('artifacts/contracts/p25_feature_search')
REPORT = Path('reports/p25_feature_search')
NATIVE = Path('artifacts/contracts/uniform_smile')
REFERENCE = Path('artifacts/contracts/p25_recency')
GROUPS = {
    'control': [],
    'rv_risk': ['long_rv_iv', 'short_rv_iv', 'iv_times_decorrelation', 'log_vol_mismatch'],
    'smile_price': ['wing_premium', 'wing_premium_z', 'wing_premium_change21'],
    'smile_shape': ['risk_reversal25', 'risk_reversal25_z', 'butterfly25', 'tail_steepness'],
    'term_structure': ['wing_term_slope'],
    'physical_tail': ['realized_skew63', 'realized_kurt63', 'semivariance_share63', 'jump_fraction63', 'tail_es63'],
    'variance_premium': ['ewma_vrp', 'ewma_vrp_z', 'variance_ratio'],
    'iv_factor': ['iv_factor_reversion'],
    'volume_oi': ['volume_oi'],
    'employee': ['employee_leadership'],
    'transcripts': ['call_negative_change', 'call_qa_uncertainty', 'call_analyst_negative'],
}
RISK_FIELDS = set(GROUPS['rv_risk'])
PROTOCOL = (
    'Eleven declared feature blocks including exact control. Frozen608 P25 proposals, '
    'same entry direction, beta/correlation/memory/formation screens, costs, funded '
    'capital, original5% stress and other caps. 18-month monthly completed-outcome '
    'fits,8-leaf tree,CEM50x192 seed2026,50/75/100 sizing and full-size fallback. '
    'New signals use signal-close options/prices or cached lagged vendor states; '
    'long-minus-short gaps, zero fill plus availability flag, never gate proposals. '
    'Physical tails use backward windows; EWMA variance uses20-session half-life; '
    'VRP z-score uses strictly prior252 sessions. IV factor is leave-one-name-out '
    'median changes with lagged rolling beta. Options smile uses cached selected '
    '14/30/45-day surfaces, a context surface that can differ from audited traded '
    'contract. No future payoff field is a feature. Vendor historical revisions '
    'are not independently archived/PIT verified. Compare four constant sizes '
    'and select the constant control on2018-2023. Report full/recent/annual and '
    'negative results; previously explored history, no fresh holdout. '
    'One worker,50%of one core,3GiB RSS stop. Original checkpoint unchanged.'
)


def fields(case):
    return base.FEATURES + [c for f in GROUPS[case] for c in [f, f + '_available']]


def prepare(sources, cal):
    if (OUT / 'prepared.json').exists():
        return
    meta = pd.read_parquet(sources.track(base.SOURCE / 'P25_rows.parquet'))
    assets = sorted(set(meta.secid_a.astype(int)) | set(meta.secid_b.astype(int)))
    features = {}
    marks = {}
    columns = ['date', 'optionid', 'expiry', 'cp_flag', 'bucket', 'tenor', 'bs_iv', 'bs_delta', 'bs_vega', 'dte', 'bid', 'ask']
    for n, sid in enumerate(assets, 1):
        stock = pd.read_parquet(sources.track(NATIVE / 'market/assets' / f'{sid}_stocks.parquet'))
        shape = return_shape(stock, cal)
        native = pd.read_parquet(sources.track(NATIVE / 'market/assets' / f'{sid}_selection.parquet'), columns=columns)
        p25 = native.loc[native.cp_flag.eq('P') & native.bucket.eq('25') & native.tenor.eq(30)].set_index('date').reindex(cal)
        assert not p25.index.has_duplicates
        iv = p25.bs_iv.where(p25.bs_iv.gt(0))
        marks[sid] = iv
        parts = []
        for cp in ['C', 'P']:
            for year in sorted(set(cal.year)):
                path = DATA / 'selection' / cp / str(year) / f'{sid}.parquet'
                if path.exists():
                    parts.append(pd.read_parquet(sources.track(path), columns=columns))
        surface = surface_states(pd.concat(parts, ignore_index=True), cal)['P', '25']
        leg = surface[GROUPS['smile_price'] + GROUPS['smile_shape'] + GROUPS['term_structure']].copy()
        for f in ['realized_skew63', 'realized_kurt63', 'jump_fraction63']:
            leg[f] = shape[f]
        leg['semivariance_share63'] = shape.down_semivariance_share63
        leg['tail_es63'] = shape.down_tail_es63
        s = stock.set_index('date').reindex(cal)
        close = (s.close * s.cfadj).where(s.close.gt(0) & s.cfadj.gt(0))
        r = np.log(close / close.shift(1))
        variance = 252 * r.pow(2).ewm(halflife=20, min_periods=63, adjust=False).mean()
        leg['ewma_vrp'] = iv.pow(2) - variance
        leg['ewma_vrp_z'] = past_z(leg.ewma_vrp)
        leg['variance_ratio'] = variance / iv.pow(2)
        features[sid] = leg.replace([np.inf, -np.inf], np.nan)
        if n % 5 == 0:
            print(f'Features: cached surfaces/prices {n}/{len(assets)} names', flush=True)
            base.progress('features', names=n, total_names=len(assets))
        del native, surface, shape, parts, stock, s
        gc.collect()
    factor = iv_factor_scores(pd.DataFrame(marks, index=cal))
    for sid in assets:
        features[sid]['iv_factor_reversion'] = factor[sid]
    del marks, factor
    for path in volume.RAW.glob('*.parquet'):
        sources.track(path)
    for path in list(employee.LINKS.glob('*.parquet')) + list(employee.REVIEWS.glob('*.parquet')):
        sources.track(path)
    for label, builder, mapping in [
        ('volume_oi', volume.leg_states, {'volume_oi': None}),
        ('employee', employee.leg_states, {'employee_leadership': None}),
        ('transcripts', transcripts.leg_states, {f: f for f in transcripts.FIELDS}),
    ]:
        print(f'Features: backward joins for {label}', flush=True)
        states = builder(cal, assets)
        for sid in assets:
            for output, source in mapping.items():
                features[sid][output] = (states[sid] if source is None else states[sid][source]).to_numpy(float)
        del states
        gc.collect()
    added = [f for g in GROUPS.values() for f in g if f not in RISK_FIELDS]
    for f in added:
        meta[f] = 0.
        meta[f + '_available'] = 0.
    for pair, idx in meta.groupby('pair', sort=False).groups.items():
        g = meta.loc[idx]
        a, b = int(g.secid_a.iloc[0]), int(g.secid_b.iloc[0])
        dates = pd.DatetimeIndex(g.date)
        gap = (features[a][added].reindex(dates).to_numpy() - features[b][added].reindex(dates).to_numpy()) * g.direction.to_numpy()[:, None]
        for j, f in enumerate(added):
            available = np.isfinite(gap[:, j])
            meta.loc[idx, f] = np.where(available, gap[:, j], 0.)
            meta.loc[idx, f + '_available'] = available.astype(float)
    risk = pd.read_parquet(sources.track(Path('artifacts/contracts/p25_risk_sizing/training_rows.parquet')))
    pd.testing.assert_frame_equal(meta[list(pd.read_parquet(base.SOURCE / 'P25_rows.parquet').columns)], pd.read_parquet(base.SOURCE / 'P25_rows.parquet'))
    assert (risk[['date', 'pair']].to_numpy() == meta[['date', 'pair']].to_numpy()).all()
    for f in RISK_FIELDS:
        for c in [f, f + '_available']:
            meta[c] = risk[c].to_numpy()
    assert np.isfinite(meta[sorted(set(c for case in GROUPS for c in fields(case)))]).all().all()
    meta.to_parquet(OUT / 'training_rows.parquet', index=False)
    proposals = pd.read_parquet(sources.track(REFERENCE / 'signal_rows.parquet'))
    new = sorted(set(c for case in GROUPS for c in fields(case)) - set(base.FEATURES))
    joined = proposals.merge(meta[['date', 'pair'] + new], on=['date', 'pair'], how='left', validate='one_to_one', indicator=True)
    assert len(joined) == 608 and joined._merge.eq('both').all()
    joined = joined.drop(columns='_merge')
    pd.testing.assert_frame_equal(joined[proposals.columns], proposals)
    joined.to_parquet(OUT / 'signal_rows.parquet', index=False)
    coverage = [dict(feature=f, training_observed=int(meta[f + '_available'].sum()), qualified_training_observed=int(meta.loc[meta.signal_eligible, f + '_available'].sum()), proposal_observed=int(joined[f + '_available'].sum())) for f in list(RISK_FIELDS) + added]
    pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)
    base.write_json(OUT / 'prepared.json', dict(rows=len(meta), proposals=len(joined), names=len(assets), groups=GROUPS))
    sources.save()
    base.progress('prepared', proposals=608)


def train(sources, cal):
    meta = pd.read_parquet(sources.track(OUT / 'training_rows.parquet'))
    logs = []
    for month in pd.period_range('2018-01-01', cal[-1] - pd.Timedelta(days=base.RUNOFF), freq='M'):
        cutoff = month.start_time
        stamp = str(cutoff.date())
        rows = meta.loc[meta.signal_eligible & meta.finite_label & completed_months(meta.date, meta.end, cutoff, 18)].reset_index(drop=True)
        for case in GROUPS:
            folder = OUT / 'models' / case
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{stamp}.json'
            if path.exists():
                logs.append(json.loads(path.read_text()))
                continue
            tree = None
            leaves = np.array([0])
            choices = np.array([4])
            details = {}
            if len(rows) >= 16:
                tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=max(8, min(63, len(rows) // 8)), random_state=2026).fit(rows[fields(case)], rows.target)
                leaves = np.unique(tree.apply(rows[fields(case)]))
                states = np.searchsorted(leaves, tree.apply(rows[fields(case)]))
                first = int(rows.position.min() + 1)
                last = int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book = np.zeros((last - first + 1, len(leaves) * 5))
                for pair, idx in rows.groupby('pair', sort=False).groups.items():
                    idx = np.asarray(idx)
                    disk = np.load(sources.track(NATIVE / 'pairs' / pair / 'P25_labels.npy'), mmap_mode='r')
                    labels = base.old.directional_paths(disk[rows.loc[idx, 'label_row'].to_numpy(int)], rows.loc[idx, 'direction'].to_numpy(int))
                    base.aggregate(book, states[idx], labels, rows.loc[idx, 'position'].to_numpy(int), first)
                    del disk, labels
                choices, details = fit_menu(book, cal[first:last + 1], [2, 3, 4])
                del book
            model = dict(tree=tree, leaves=leaves, choices=choices, fit=stamp, features=fields(case))
            if case == 'control':
                old = joblib.load(sources.track(REFERENCE / 'models/18M_floor' / f'{stamp}.joblib'))
                np.testing.assert_array_equal(choices, old['choices'])
                np.testing.assert_array_equal(leaves, old['leaves'])
                if tree is not None:
                    for f in ['feature', 'threshold', 'children_left', 'children_right', 'value']:
                        np.testing.assert_array_equal(getattr(tree.tree_, f), getattr(old['tree'].tree_, f))
            record = dict(model=case, fit=stamp, training_rows=len(rows), last_training_outcome=str(rows.end.max().date()) if len(rows) else None, choices=choices.tolist(), **details)
            joblib.dump(model, folder / f'{stamp}.joblib')
            base.write_json(path, record)
            logs.append(record)
            del tree, model
            gc.collect()
        print(f'Feature fits {stamp}: {len(logs)} models', flush=True)
        sources.save()
        base.progress('train', fit=stamp, models_complete=len(logs))
    assert len(logs) == len(GROUPS) * 90
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)


def replay(sources, cal):
    base.cached.OUT = base.TRANSFER
    market = base.Market()
    cfg = replace(base.config(), option_commission=.03, capital=345_000_000, max_gross_vega=20_000 * 345)
    for case in list(GROUPS) + [f'fixed{i}' for i in [1, 2, 3, 4]]:
        folder = OUT / 'replays' / case
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'status.json').exists():
            continue
        rows = pd.read_parquet(OUT / 'signal_rows.parquet')
        rows['learned_level'] = int(case[-1]) if case.startswith('fixed') else 4
        if case in GROUPS:
            for month, idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
                model = joblib.load(OUT / 'models' / case / f'{month.start_time.date()}.joblib')
                rows.loc[idx, 'learned_level'] = score_model(model, rows.loc[idx], fields(case))
        rows.to_parquet(folder / 'scored_allocations.parquet', index=False)
        market.variant = case
        market.started = time.monotonic()
        market.reported_year = None
        saved = base.engine.book_risk
        try:
            base.engine.book_risk = base.book_risk
            nav, trades, decisions, ledger = base.engine.simulate(market, rows, cal[cal >= '2018-01-01'], cfg, 1_000_000, policy=lambda row, context: int(row['learned_level']), corporate_events=base.UNIFORM_CORPORATE_EVENTS, wing_capital=1_000_000)
        finally:
            base.engine.book_risk = saved
        base.mark_book(nav, market.rates, cfg.capital)
        audit = base.verify(nav, trades, decisions, rows, cfg)
        if case == 'control' or case.startswith('fixed'):
            path = Path('artifacts/contracts/p25_sharpe_search/replays') / ('cap050_floor' if case == 'control' else 'cap050_' + case)
            pd.testing.assert_frame_equal(nav, pd.read_parquet(sources.track(path / 'daily_nav.parquet')), rtol=1e-12, atol=1e-6)
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv', index=False)
        decisions.to_csv(folder / 'decisions.csv', index=False)
        base.write_json(folder / 'status.json', dict(status='complete', model=case, kind='fixed' if case.startswith('fixed') else 'learned', net_excess_pnl=float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1]), **audit, **ledger))
        print(f'{case}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, entries {audit["entries"]}', flush=True)
        sources.save()
        base.progress('replay', model=case, **audit)
        gc.collect()


def summarize(sources):
    records = []
    annual = []
    for folder in sorted((OUT / 'replays').iterdir()):
        r = json.loads((folder / 'status.json').read_text())
        nav = pd.read_parquet(folder / 'daily_nav.parquet')
        excess = nav.nav - nav.cash_benchmark
        records.append(dict(model=folder.name, kind=r['kind'], sharpe=r['annualized_excess_sharpe'], entries=r['entries'], net_excess_pnl=r['net_excess_pnl'], early_sharpe=base.sharpe(nav.loc[:'2023-12-31', 'excess_return'].iloc[1:]), recent_sharpe=base.sharpe(nav.loc['2024-01-01':, 'excess_return']), max_excess_drawdown=float((excess - excess.cummax()).min())))
        for year, g in nav.groupby(nav.index.year):
            annual.append(dict(model=folder.name, year=int(year), sharpe=base.sharpe(g.excess_return.iloc[1:] if year == 2018 else g.excess_return), excess_profit=float((nav.pnl - nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum())))
    result = pd.DataFrame(records).sort_values('sharpe', ascending=False)
    assert len(result) == len(GROUPS) + 4
    control = result.loc[result.kind.eq('fixed')].sort_values('early_sharpe', ascending=False).iloc[0]
    result['fixed_control'] = control.model
    result['sharpe_difference_vs_control'] = result.sharpe - control.sharpe
    result.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    chosen = result.loc[result.kind.eq('learned')].sort_values('early_sharpe', ascending=False).iloc[0]
    base.write_json(REPORT / 'selection.json', dict(selected_using='2018-2023', model=chosen.model, full_sharpe=chosen.sharpe, recent_sharpe=chosen.recent_sharpe, fixed_control=control.model, descriptive_best_full=result.iloc[0].model, no_fresh_holdout=True))
    fits = pd.read_csv(REPORT / 'fits.csv')
    known = fits.last_training_outcome.notna()
    assert (pd.to_datetime(fits.loc[known, 'last_training_outcome']) < pd.to_datetime(fits.loc[known, 'fit'])).all()
    sources.check()
    sources.save()
    base.write_json(REPORT / 'audit.json', dict(status='complete', models=len(fits), replays=len(result), original90_models_and_five_cash_books_reproduced=True, all608_proposals_preserved=True, all_ledgers_verified=True, outcomes_strictly_before_fit=True, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (REPORT / 'findings.txt').write_text(result.to_string(index=False) + '\n\n' + PROTOCOL + '\n')
    print(result.to_string(index=False), flush=True)
    base.progress('complete')
    base.write_json(REPORT / 'output_hashes.json', {str(p): base.digest(p) for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'train', 'replay', 'summarize', 'all'], default='all')
    args = p.parse_args()
    base.OUT, base.REPORT = OUT, REPORT
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    sources = base.Sources()
    for path in [Path(__file__), Path('scripts/tail_feature_exploration.py'), Path('scripts/atm65_mined_features.py'), Path('scripts/atm65_volume_oi_features.py'), Path('scripts/atm65_transcript_features.py'), Path('scripts/run_atm65_employee_sentiment.py'), Path('scripts/run_p25_action_windows.py'), Path('scripts/p25_action_windows.py'), Path('scripts/p25_recency.py')]:
        sources.track(path)
    sources.hashes[str(Path(__file__))] = base.digest(Path(__file__))
    cal = pd.read_parquet(sources.track(NATIVE / 'market/rates.parquet')).index
    base.write_json(OUT / 'inputs.json', dict(protocol=PROTOCOL, groups=GROUPS, original_features=base.FEATURES))
    if args.phase in ['prepare', 'all']:
        prepare(sources, cal)
    if args.phase in ['train', 'all']:
        train(sources, cal)
    if args.phase in ['replay', 'all']:
        replay(sources, cal)
    if args.phase in ['summarize', 'all']:
        summarize(sources)
    sources.check()
    sources.save()


if __name__ == '__main__':
    main()
