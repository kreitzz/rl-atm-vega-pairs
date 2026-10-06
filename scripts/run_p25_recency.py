"""Bounded follow-up to the frozen P25 action/window comparison."""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_p25_action_windows as base
from scripts import p25_recency as policy
from scripts.p25_action_windows import score_model

OUT = Path('artifacts/contracts/p25_recency')
REPORT = Path('reports/p25_recency')
PREVIOUS = Path('artifacts/contracts/p25_action_windows')
CASES = [dict(name=f'{m}M_floor', months=m, menu=[2,3,4], half_life=None) for m in [6,9,12,18,24]]
CASES += [dict(name='12M_high_floor', months=12, menu=[3,4], half_life=None)]
CASES += [dict(name=f'{m}M_hl{h}', months=m, menu=[2,3,4], half_life=h) for m,h in [(12,6),(36,6),(36,12)]]
PROTOCOL = (
    'Nine declared policies (eight new plus exact12M floor control):6/9/12/18/24 '
    'month windows with50/75/100 sizing;12month75/100;12month6month half-life; '
    '36month6/12month half-life. Recency weighting applies to tree signal-date '
    'weights and daily training-book mean/covariance. Same8leaf tree,minleaf, '
    'CEM50x192 seed2026,full-size anchor/fallback and monthly refits. Completed '
    'outcomes strictly before fit,signal date within lookback. Preserve original '
    '608P25 proposals,27features,approximately30D25delta puts,entry vega '
    'matching,daily delta hedging,own expiry,costs(.65 training,.03 actual '
    'commission,mid+.1volpoint,1bp stock),financing and original risk budgets. '
    'No learned skips,new risk features,seed search,exits or wings. Sizes can '
    'change which overlapping entries pass risk limits. Exact12M baseline '
    'model/NAV/trades/decisions reproduction required;fixed100 also reproduced. '
    'Ranking specified before fitting uses2018-2023 cash-excess Sharpe;report '
    '2024-partial2025,annuals and full-history separately. Previously inspected '
    'history and retrospective roster/calendar conditioning;no fresh holdout '
    'or automatic checkpoint promotion. One worker,50%one core,3GiB RSS stop.'
)


def configure():
    # Reuse audited proposal join, original engine and ledger verification.
    # These assignments affect this process only; earlier source/output stays intact.
    base.OUT = OUT; base.REPORT = REPORT
    base.CASES = [(case['name'], '') for case in CASES]
    base.name = lambda case, menu: case
    base.policy.WINDOWS = []  # No learned-skip controls in this positive-size study.
    base.allocations = allocations


def train(sources, cal):
    meta = pd.read_parquet(sources.track(base.SOURCE/'P25_rows.parquet')); logs = []
    months = pd.period_range('2018-01-01', cal[-1]-pd.Timedelta(days=base.RUNOFF), freq='M')
    for month in months:
        cutoff = month.start_time; stamp = str(cutoff.date())
        for case in CASES:
            folder = OUT/'models'/case['name']; folder.mkdir(parents=True, exist_ok=True)
            path = folder/f'{stamp}.json'
            if path.exists(): logs.append(json.loads(path.read_text())); continue
            mask = meta.signal_eligible & meta.finite_label & policy.completed_months(meta.date,meta.end,cutoff,case['months'])
            rows = meta.loc[mask].reset_index(drop=True)
            assert len(rows)==0 or rows.end.max()<cutoff
            tree = None; leaves = np.array([0]); book = None
            if len(rows)>=16:
                tree = DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026)
                if case['half_life'] is None: tree.fit(rows[base.FEATURES],rows.target)
                else: tree.fit(rows[base.FEATURES],rows.target,
                    sample_weight=policy.recency_weights(rows.date,cutoff,case['half_life']))
                leaves = np.unique(tree.apply(rows[base.FEATURES])); states = np.searchsorted(leaves,tree.apply(rows[base.FEATURES]))
                first = int(rows.position.min()+1); last = int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book = np.zeros((last-first+1,len(leaves)*5))
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx = np.asarray(idx)
                    disk = np.load(sources.track(base.cached.OUT/'pairs'/pair/'P25_labels.npy'),mmap_mode='r')
                    labels = base.old.directional_paths(disk[rows.loc[idx,'label_row'].to_numpy(int)],rows.loc[idx,'direction'].to_numpy(int))
                    base.aggregate(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first)
                    del disk,labels
            choices,details = (np.array([4]),{}) if book is None else policy.fit_recency(book,cal[first:last+1],case['menu'],cutoff,case['half_life'])
            assert np.isin(choices,case['menu']).all()
            model = dict(tree=tree,leaves=leaves,choices=choices,fit=stamp)
            if case['name']=='12M_floor':
                previous = joblib.load(sources.track(PREVIOUS/'models/1Y_floor'/f'{stamp}.joblib'))
                np.testing.assert_array_equal(choices,previous['choices']); np.testing.assert_array_equal(leaves,previous['leaves'])
                if tree is not None:
                    for field in ['feature','threshold','children_left','children_right','value']:
                        np.testing.assert_array_equal(getattr(tree.tree_,field),getattr(previous['tree'].tree_,field))
            record = dict(model=case['name'],fit=stamp,window_months=case['months'],half_life_months=case['half_life'],
                menu=case['menu'],training_rows=len(rows),training_pairs=int(rows.pair.nunique()),
                first_training_signal=str(rows.date.min().date()) if len(rows) else None,
                last_training_outcome=str(rows.end.max().date()) if len(rows) else None,
                choices=choices.tolist(),leaves=len(leaves),status='enabled' if tree is not None else 'original_fullsize_fallback',**details)
            joblib.dump(model,folder/f'{stamp}.joblib');base.write_json(path,record);logs.append(record)
            del rows,tree,book;gc.collect()
        if month.month in [1,4,7,10]: print(f'Completed recency fits {stamp}: {len(logs)} models',flush=True)
        sources.save();base.progress('train',fit=stamp,models_complete=len(logs))
    assert len(logs)==len(CASES)*len(months)
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(case):
    rows = pd.read_parquet(OUT/'signal_rows.parquet'); rows['learned_level'] = 4
    for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
        model = joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
        assert pd.Timestamp(model['fit'])<=rows.loc[idx,'date'].min()
        rows.loc[idx,'learned_level'] = score_model(model,rows.loc[idx],base.FEATURES)
    assert rows.learned_level.between(2,4).all()
    return rows


def replay(sources,cal):
    base.replay(sources,cal)
    current = OUT/'replays/12M_floor'; previous = PREVIOUS/'replays/1Y_floor'
    pd.testing.assert_frame_equal(pd.read_parquet(current/'daily_nav.parquet'),
        pd.read_parquet(sources.track(previous/'daily_nav.parquet')),rtol=1e-12,atol=1e-6)
    for file in ['trades.csv','decisions.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(current/file),pd.read_csv(sources.track(previous/file)),rtol=1e-12,atol=1e-7)
    # Copy the immutable original5Y reference for reporting and paired comparisons.
    folder = OUT/'replays/5Y_reference'; folder.mkdir(parents=True,exist_ok=True)
    for file in ['daily_nav.parquet','trades.csv','decisions.csv']:
        shutil.copyfile(sources.track(PREVIOUS/'replays/5Y_floor'/file),folder/file)
    old = json.loads(sources.track(PREVIOUS/'replays/5Y_floor/status.json').read_text())
    old.update(model='5Y_reference',kind='reference');base.write_json(folder/'status.json',old)
    print('12M models and replay exactly reproduced;original5Y reference preserved',flush=True)


def summarize(sources):
    rows = []; annual = []; books = {}
    for folder in sorted((OUT/'replays').iterdir()):
        record = json.loads((folder/'status.json').read_text()); assert record['status']=='complete'
        nav = pd.read_parquet(folder/'daily_nav.parquet'); books[folder.name] = nav
        decisions = pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date'])
        assert len(decisions)==608 and not decisions.level.eq(0).any()
        rows.append(dict(model=folder.name,kind=record['kind'],annualized_excess_sharpe=record['annualized_excess_sharpe'],
            selection_2018_2023_sharpe=base.sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:]),
            recent_2024_2025_sharpe=base.sharpe(nav.loc['2024-01-01':,'excess_return']),
            entries=record['entries'],active_pairs=record['active_pairs'],net_excess_pnl=record['net_excess_pnl'],
            mean_entered_size=float(decisions.loc[decisions.status.eq('entered'),'level'].mean()/4),
            worst_daily_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).min())))
        for year,g in nav.groupby(nav.index.year):
            annual.append(dict(model=folder.name,year=int(year),annualized_excess_sharpe=base.sharpe(g.excess_return.iloc[1:] if year==2018 else g.excess_return),
                net_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum()),
                entries=int((decisions.status.eq('entered') & decisions.entry_date.dt.year.eq(year)).sum())))
    comparison = pd.DataFrame(rows).sort_values('annualized_excess_sharpe',ascending=False)
    comparison.to_csv(REPORT/'comparison.csv',index=False)
    annual = pd.DataFrame(annual);annual.to_csv(REPORT/'annual.csv',index=False)
    annual.pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_sharpe.csv')
    variants = comparison.loc[comparison.kind.eq('model')]
    selected = variants.sort_values('selection_2018_2023_sharpe',ascending=False).iloc[0]
    base.write_json(REPORT/'selection.json',dict(predeclared_selection_period='2018-2023',selected_model=selected.model,
        selected_early_sharpe=selected.selection_2018_2023_sharpe,selected_recent_sharpe=selected.recent_2024_2025_sharpe,
        selected_full_sharpe=selected.annualized_excess_sharpe,descriptive_full_best=variants.iloc[0].model,
        no_fresh_holdout=True,no_checkpoint_promotion=True))
    uplift = []
    for reference in ['12M_floor','5Y_reference']:
        b = books[reference].excess_return.iloc[1:].to_numpy()
        for case in variants.model:
            if case==reference:continue
            assert books[case].index.equals(books[reference].index)
            a = books[case].excess_return.iloc[1:].to_numpy();rng = np.random.default_rng(2026);n=len(a);block=126;draws=[]
            for _ in range(1000):
                starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
                idx=(starts[:,None]+np.arange(block)).ravel()[:n]
                draws.append(base.sharpe(a[idx])-base.sharpe(b[idx]))
            uplift.append(dict(model=case,reference=reference,sharpe_difference=base.sharpe(a)-base.sharpe(b),
                ci_lower=float(np.quantile(draws,.025)),ci_upper=float(np.quantile(draws,.975))))
    uplift = pd.DataFrame(uplift);uplift.to_csv(REPORT/'uplift.csv',index=False)
    selected_case = next(case for case in CASES if case['name']==selected.model)
    interval = uplift.loc[uplift.model.eq(selected.model)&uplift.reference.eq('12M_floor')]
    candidate = dict(status='research_candidate',model=selected.model,
        reference_checkpoint='config/p25_frozen_strategy_checkpoint.json',
        previous_candidate='config/p25_action_window_candidate.json',
        models=str(OUT/'models'/selected.model),replay=str(OUT/'replays'/selected.model),
        training_window_months=selected_case['months'],half_life_months=selected_case['half_life'],
        sizing_actions=[x/4 for x in selected_case['menu']],retrain_frequency='monthly',
        signals_features_costs_and_holding='Unchanged from frozen reference',
        annualized_cash_excess_sharpe=float(selected.annualized_excess_sharpe),
        net_excess_pnl=float(selected.net_excess_pnl),entries=int(selected.entries),active_pairs=int(selected.active_pairs),
        selected_using='Predeclared2018-2023 cash-excess Sharpe across nine declared policies',
        selection_period_sharpe=float(selected.selection_2018_2023_sharpe),
        recent2024_2025_sharpe=float(selected.recent_2024_2025_sharpe),
        no_fresh_holdout=True,original_and_one_year_checkpoints_preserved=True)
    if len(interval):candidate['paired126_session_bootstrap95pct_interval_vs_one_year']=[float(interval.iloc[0].ci_lower),float(interval.iloc[0].ci_upper)]
    candidate_path=Path('config/p25_recency_candidate.json');base.write_json(candidate_path,candidate)
    yearly = annual.loc[annual.model.isin([selected.model,'12M_floor','5Y_reference','fixed_100'])]
    text = (
        f"Selected {selected.model}: annualized cash-excess Sharpe {selected.annualized_excess_sharpe:.6f}, "
        f"{selected.entries} entries, {selected.active_pairs} active pairs and ${selected.net_excess_pnl:,.2f} above cash.\n"
        f"Ranked first using2018-2023 only: {selected.selection_2018_2023_sharpe:.6f}. "
        f"2024-partial2025 remains negative: {selected.recent_2024_2025_sharpe:.6f}.\n\n"
        "The18-month50/75/100 policy improves full-history Sharpe versus12-month and5-year. "
        "Other lookbacks,75% minimum size and recency weighting did not improve the12-month reference. "
        "Versus12-month,18-month earns about$21k more above cash; gains in2018/2019/2022/2023 and "
        "smaller2024 losses offset worse2020/2021/partial2025 dollar P&L.2024 loss falls from$31.5k "
        "to$16.4k but its Sharpe is still-.919;2021 Sharpe deteriorates from-.161 to-.569. "
        "Worst daily excess loss falls from$23.8k to$21.7k. No consistent improvement in every year.\n\n"
        "Identical proposals,features,costs and risk budgets. Sizes change both individual exposure and "
        "subsequent risk-limit admission:208 versus211 entries,44 versus42 active pairs. Thus this "
        "is the executable sizing policy comparison,not identical accepted-trade sets. Both learn "
        "only50/75/100 sizes;no learned skips.\n\n"
        "Paired126-session block-bootstrap95% uplift interval versus12-month spans zero. The interval "
        "versus5-year excludes zero,but neither interval adjusts for repeated research/model selection. "
        "This is a new best research candidate,not fresh holdout confirmation;original checkpoints "
        "and the previous12-month candidate remain intact. The allocator is the existing contextual "
        "tree/direct-Sharpe sizing optimizer,not a newly trained sequential inventory policy.\n\n"
        +comparison.to_string(index=False)+'\n\n'
        +yearly.pivot(index='year',columns='model',values='annualized_excess_sharpe').to_string()
        +'\n\n2025 is partial through August29;last signal June30;own-expiry runoff.\n\n'+PROTOCOL+'\n'
    )
    (REPORT/'interpretation.txt').write_text(text)
    fits=pd.read_csv(REPORT/'fits.csv');known=fits.last_training_outcome.notna()
    assert len(fits)==810
    assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    assert (pd.to_datetime(fits.loc[known,'first_training_signal']) >=
        pd.Series([pd.Timestamp(r.fit)-pd.DateOffset(months=int(r.window_months)) for r in fits.loc[known].itertuples()],index=fits.index[known])).all()
    resources=json.loads((OUT/'resource_history.json').read_text())
    peak=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r['peak_rss_bytes'] for r in resources])
    sources.check();sources.save()
    base.write_json(REPORT/'audit.json',dict(status='complete',declared_policies=9,new_variants=8,monthly_models=810,
        one_year90_models_and_NAV_trades_decisions_reproduced=True,fixed100_reproduced=True,
        original_checkpoint_unchanged=True,all608_decisions_preserved=True,no_learned_skips=True,
        completed_outcome_and_lookback_audits_passed=True,no_fresh_holdout=True,peak_rss_bytes=peak))
    (REPORT/'findings.txt').write_text(comparison.to_string(index=False)+'\n\n'+annual.to_string(index=False)+'\n\n'+PROTOCOL+'\n')
    print(comparison.to_string(index=False),flush=True);base.progress('complete')
    paths=[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']+[candidate_path]
    base.write_json(REPORT/'output_hashes.json',{str(p):base.digest(p) for p in paths})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','train','replay','summarize','all'],default='all');args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_recency.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    configure();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources=base.Sources()
    sources.hashes[str(Path(__file__))] = base.digest(Path(__file__))
    for path in [Path(__file__),Path('scripts/p25_recency.py'),Path('tests/test_p25_recency.py'),
        Path('scripts/run_p25_action_windows.py'),Path('scripts/p25_action_windows.py'),Path('scripts/uniform_smile_engine.py'),
        Path('scripts/run_tail_put25_residual_rl.py'),Path('config/p25_frozen_strategy_checkpoint.json'),Path('config/p25_action_window_candidate.json')]:sources.track(path)
    for path in (base.TRANSFER/'market').glob('*.parquet'):sources.track(path)
    cal=pd.read_parquet(sources.track(base.cached.OUT/'market/rates.parquet')).index
    base.write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,cases=CASES,features=base.FEATURES,seed=2026,
        evaluation_end='2025-08-29',signal_cutoff=str((cal[-1]-pd.Timedelta(days=base.RUNOFF)).date())))
    if args.phase in ['prepare','all']:base.prepare(sources,cal)
    if args.phase in ['train','all']:train(sources,cal)
    if args.phase in ['replay','all']:replay(sources,cal)
    history_path=OUT/'resource_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));base.write_json(history_path,history)
    if args.phase in ['summarize','all']:summarize(sources)
    sources.check();sources.save()


if __name__=='__main__':main()
