"""Past-only straddle experience and spot/IV dynamics for contextual sizing.

The proposal pool, independently simulated 50/75/100% reward paths and funded
execution engine are inherited unchanged. Only features and context fitting
change. Every numerical phase must use the single-worker research guard.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from scripts import straddle_overnight_study as previous

CONFIG = Path('config/straddle_context_followup.json')
CFG = json.loads(CONFIG.read_text())
OUT, REPORT = Path(CFG['output_root']), Path(CFG['report_root'])
SOURCE = Path(CFG['feature_source'])
PARENT = previous.parent
CORE = ['residual_memory', 'iv', 'vix', 'normalized_memory', 'variance_ratio']
MEMORY = ['pool_profit91_score', 'pool_profit365_score', 'profit_regime',
          'name_profit365_shrunk', 'name_win365_shrunk', 'name_loss365_shrunk',
          'name_history_count', 'profit_history_available']
DYNAMICS = ['momentum21', 'momentum126', 'drawdown63', 'downside_share21',
            'jump_size21', 'rv_acceleration', 'iv_change5', 'iv_change21',
            'iv_volatility21', 'vix_change21', 'dynamics_available',
            'relative_memory_rank', 'relative_iv']
FEATURES = dict(memory=CORE+MEMORY, dynamics=CORE+DYNAMICS,
                combined=CORE+MEMORY+DYNAMICS, market=previous.FEATURES['market'])


def window_moments(dates, labels, days):
    """Strict completion cutoff: outcomes ending on the signal date are excluded."""
    labels = labels.sort_values('end')
    ends = pd.DatetimeIndex(labels.end).to_numpy()
    dates = pd.DatetimeIndex(dates)
    right = np.searchsorted(ends, dates.to_numpy(), side='left')
    left = np.searchsorted(ends, (dates-pd.Timedelta(days=days)).to_numpy(), side='left')
    profit = labels.full_profit.to_numpy(float)/450.
    values = np.column_stack([np.ones(len(labels)), profit, profit**2,
                              (profit > 0).astype(float), np.minimum(profit, 0)**2])
    prefix = np.vstack([np.zeros(5), np.cumsum(values, axis=0)])
    result = prefix[right]-prefix[left]
    return result


def experience_features(dates, own, all_labels):
    """Shrink sparse per-name observations toward the contemporaneous pool."""
    pool91 = window_moments(dates, all_labels, 91)
    pool365 = window_moments(dates, all_labels, 365)
    local = window_moments(dates, own, 365)
    n = np.maximum(pool365[:, 0], 1)
    mean = pool365[:, 1]/n
    variance = np.maximum(pool365[:, 2]/n-mean**2, 0)
    scale = np.maximum(np.sqrt(variance), .01)
    recent = pool91[:, 1]/np.maximum(pool91[:, 0], 1)
    prior = CFG['prior_strength']
    shrunk_mean = (local[:, 1]+prior*mean)/(local[:, 0]+prior)
    shrunk_win = (local[:, 3]+prior*pool365[:, 3]/n)/(local[:, 0]+prior)
    shrunk_loss = (local[:, 4]+prior*pool365[:, 4]/n)/(local[:, 0]+prior)
    return pd.DataFrame(dict(pool_profit91_score=recent/scale,
        pool_profit365_score=mean/scale, profit_regime=(recent-mean)/scale,
        name_profit365_shrunk=shrunk_mean/scale,
        name_win365_shrunk=shrunk_win,
        name_loss365_shrunk=shrunk_loss/np.maximum(pool365[:, 2]/n, .0001),
        name_history_count=np.log1p(local[:, 0]),
        profit_history_available=(pool365[:, 0] >= 21).astype(float)))


def dynamics_features(returns, iv, vix):
    sq = returns**2
    mean21 = sq.rolling(21, min_periods=21).mean()
    rms21 = np.sqrt(mean21)
    logprice = returns.fillna(0).cumsum()
    frame = pd.DataFrame(dict(
        momentum21=returns.rolling(21, min_periods=21).sum(),
        momentum126=returns.rolling(126, min_periods=63).sum(),
        drawdown63=logprice-logprice.rolling(63, min_periods=42).max(),
        downside_share21=sq.where(returns < 0, 0).rolling(21, min_periods=21).mean()/mean21.where(mean21>1e-12),
        jump_size21=returns.abs().rolling(21, min_periods=21).max()/rms21.where(rms21>1e-8),
        rv_acceleration=np.sqrt(sq.rolling(5, min_periods=5).mean()/sq.rolling(63, min_periods=42).mean().where(lambda s:s>1e-12)),
        iv_change5=iv.diff(5)/iv.shift(5).where(iv.shift(5)>.01),
        iv_change21=iv.diff(21)/iv.shift(21).where(iv.shift(21)>.01),
        iv_volatility21=iv.pct_change(fill_method=None).rolling(21, min_periods=15).std(),
        vix_change21=vix.diff(21)/vix.shift(21).where(vix.shift(21)>1)))
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame['dynamics_available'] = frame.notna().all(axis=1).astype(float)
    return frame.fillna(0)


def prepare_features(limit):
    folder = OUT/'features'; folder.mkdir(parents=True, exist_ok=True)
    roster = pd.read_parquet(SOURCE/'proposals.parquet', columns=['date', 'secid'], use_threads=False)
    meta = pd.read_parquet(SOURCE/'training.parquet', columns=['secid','end','full_profit','finite_label'], use_threads=False)
    valid = meta.loc[meta.finite_label].copy()
    cal = PARENT.unit.base.calendar()
    vix = pd.read_csv('data/raw/VIX_History.csv', usecols=['DATE','CLOSE'], parse_dates=['DATE']).set_index('DATE').CLOSE.sort_index().reindex(cal, method='ffill')
    done = 0
    for sid, group in roster.groupby('secid', sort=True):
        path = folder/f'{sid}.parquet'
        if path.exists(): continue
        if done >= limit: break
        r = previous.returns_for(int(sid), cal)
        iv = pd.read_parquet(PARENT.unit.base.OUT/'panel'/f'{sid}.parquet', columns=['date','iv'], use_threads=False).set_index('date').iv.reindex(cal)
        f = dynamics_features(r, iv, vix).reindex(group.date).reset_index(drop=True)
        memory = experience_features(group.date, valid.loc[valid.secid.eq(sid)], valid)
        for c in MEMORY: f[c] = memory[c].to_numpy()
        f['date'] = group.date.to_numpy(); f['secid'] = int(sid)
        assert np.isfinite(f[MEMORY+DYNAMICS[:-2]]).all().all()
        f.to_parquet(path, index=False); done += 1
        del f; gc.collect(); pa.default_memory_pool().release_unused()
        if done % 25 == 0: print(json.dumps(dict(phase='features', new_names=done)), flush=True)
    print(json.dumps(dict(phase='features_done', names=len(list(folder.glob('*.parquet'))))), flush=True)


def assemble():
    names = list(dict.fromkeys(sum(FEATURES.values(), [])))
    existing = [c for c in names if c not in MEMORY+DYNAMICS]
    rows = pd.read_parquet(SOURCE/'proposals.parquet', columns=list(dict.fromkeys(PARENT.unit.bandit.EXECUTION+['secid']+existing)), use_threads=False)
    features = pd.concat([pd.read_parquet(p, use_threads=False) for p in sorted((OUT/'features').glob('*.parquet'))], ignore_index=True)
    rows = rows.merge(features, on=['date','secid'], how='left', validate='one_to_one')
    rows['relative_memory_rank'] = rows.groupby('date').normalized_memory.rank(pct=True)
    rows['relative_iv'] = rows.iv/rows.groupby('date').iv.transform('median').clip(lower=.01)
    assert len(rows)==35774 and np.isfinite(rows[names]).all().all()
    original = pd.read_parquet(SOURCE/'base_proposals.parquet', columns=PARENT.unit.bandit.EXECUTION, use_threads=False)
    pd.testing.assert_frame_equal(rows[PARENT.unit.bandit.EXECUTION], original)
    rows.to_parquet(OUT/'proposals.parquet', index=False)
    base = OUT/'base_proposals.parquet'
    if not base.exists(): base.symlink_to((SOURCE/'base_proposals.parquet').resolve())
    meta = pd.read_parquet(SOURCE/'training.parquet', use_threads=False)
    meta = meta.drop(columns=[c for c in names if c in meta])
    meta = meta.merge(rows[['date','secid']+names], on=['date','secid'], how='left', validate='many_to_one')
    assert int(meta.finite_label.sum())==4116 and np.isfinite(meta.loc[meta.finite_label,names]).all().all()
    meta.to_parquet(OUT/'training.parquet', index=False)
    PARENT.core.dump(OUT/'training_manifest.json', {str(p):previous.digest(p) for p in [OUT/'training.parquet',OUT/'proposals.parquet',Path(CFG['source_root'])/'training_manifest.json']})
    PARENT.core.dump(REPORT/'protocol.json', dict(configuration=CFG, features=FEATURES,
        proposals=len(rows), training_examples=int(meta.finite_label.sum()),
        outcomes='Same exact independent integer-straddle 50/75/100% payoff paths; no new label selection.',
        experience='Strictly completed candidate straddles, not future outcomes or selectively realized live trades. Sparse per-name moments shrink toward the contemporaneous completed-outcome pool.',
        dynamics='Signal-close adjusted stock returns, ATM IV and VIX; all rolling histories end at the signal close.',
        fitting='Monthly; 36/48 months of completed episodes; ridge alpha100 and equal total weight per signal date, or four-leaf reward tree. Exhaustive Sharpe sizing-map search.',
        limits='Training aggregates independent paths and does not model funded admission. Funded replay applies the original inventory constraints.',
        execution_unchanged=True))
    print(json.dumps(dict(phase='assembled', proposals=len(rows), features=len(names))), flush=True)


def fit_model(meta, paths, cutoff, spec, cal):
    cutoff = pd.Timestamp(cutoff)
    eligible = meta.finite_label & meta.date.ge(cutoff-pd.DateOffset(months=spec['months'])) & meta.date.lt(cutoff) & meta.end.lt(cutoff)
    idx = np.flatnonzero(eligible.to_numpy()); rows = meta.iloc[idx]; names = FEATURES[spec['features']]
    model = dict(cutoff=str(cutoff.date()), specification=spec, features=names,
        training_samples=len(rows), choices=[CFG['warmup_action']],
        last_completed=str(rows.end.max().date()) if len(rows) else None)
    if len(rows)<CFG['minimum_training_samples']: return model
    values = rows[names].to_numpy(float)
    target = paths[idx,2].sum(axis=1)
    if spec['target']=='risk': target = PARENT.core.risk_adjusted_target_original(paths[idx])
    if spec['kind']=='tree':
        leaves = PARENT.unit.bandit.reward_partition(values, target, CFG['minimum_leaf_samples'], max_leaves=4)
        states = PARENT.unit.bandit.tree_contexts(values, leaves); model['tree'] = leaves
    else:
        center, scale = np.median(values, axis=0), np.std(values, axis=0)
        scale = np.maximum(scale, .001)
        design = np.column_stack([np.ones(len(values)), np.clip((values-center)/scale,-5,5)])
        # Multiple simultaneous cross-sectional labels are not independent dates.
        counts = rows.groupby('date').date.transform('size').to_numpy(float)
        weights = 1/counts; weights *= len(weights)/weights.sum()
        penalty = np.eye(design.shape[1])*CFG['ridge_alpha']; penalty[0,0] = 0
        beta = np.linalg.solve(design.T@(weights[:,None]*design)+penalty, design.T@(weights*target))
        score = design@beta; cuts = np.quantile(score,[1/3,2/3])
        states = np.searchsorted(cuts,score,side='left')
        model.update(center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),cuts=cuts.tolist(),clip=5.)
    occupancy = np.bincount(states)
    first, stop = cal.searchsorted(cutoff-pd.DateOffset(months=spec['months'])), cal.searchsorted(cutoff)
    book = np.zeros((stop-first,3*len(occupancy)))
    days = rows.entry_position.to_numpy(int)[:,None]+np.arange(paths.shape[2])[None]-first
    valid = (days>=0)&(days<len(book))
    for state in range(len(occupancy)):
        selected = np.flatnonzero(states==state)
        for action in range(3):
            np.add.at(book[:,3*state+action],days[selected][valid[selected]],paths[idx[selected],action][valid[selected]])
    choices, score = PARENT.core.map_actions(book,occupancy)
    model.update(choices=choices.tolist(),occupancy=occupancy.tolist(),training_shadow_sharpe=score)
    assert pd.Timestamp(model['last_completed'])<cutoff
    return model


def predict(model, rows):
    if len(model['choices'])==1: return np.full(len(rows),model['choices'][0],int)
    values = rows[model['features']].to_numpy(float)
    if 'tree' in model: states = PARENT.unit.bandit.tree_contexts(values,model['tree'])
    else:
        design = np.column_stack([np.ones(len(rows)),np.clip((values-model['center'])/model['scale'],-model['clip'],model['clip'])])
        states = np.searchsorted(model['cuts'],design@model['beta'],side='left')
    return np.asarray(model['choices'])[states]


def configure():
    previous.CONFIG=CONFIG; previous.CFG=CFG; previous.OUT=OUT; previous.REPORT=REPORT
    previous.DATA=Path(CFG['source_root']); previous.FEATURES=FEATURES
    previous.specifications=lambda:CFG['specifications']; previous.fit_model=fit_model
    previous.configure()
    PARENT.core.predict=predict
    PARENT.core.risk_adjusted_target_original=lambda paths:np.clip(np.divide(
        paths[:,2].sum(axis=1),np.sqrt((paths[:,2]**2).sum(axis=1)),
        out=np.zeros(len(paths)),where=(paths[:,2]**2).sum(axis=1)>1e-18),-3,3)


def sources():
    files=[CONFIG,Path(__file__),Path(previous.__file__),Path(PARENT.__file__),
           Path(PARENT.core.__file__),SOURCE/'training_manifest.json',
           Path('scripts/run_capped_straddle_book.py'),Path('scripts/atm_rv_straddle_bandit.py'),
           Path(PARENT.unit.engine.__file__),Path('paper_code/execution.json')]
    record={str(p):previous.digest(p) for p in files}
    path=OUT/'sources.json'
    if path.exists(): assert json.loads(path.read_text())==record,'Source changed; use a new output root'
    else:PARENT.core.dump(path,record)


def select():
    path=REPORT/'selection.json'
    baseline=pd.read_parquet('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/daily_nav.parquet',columns=['excess_return'],use_threads=False).excess_return.loc['2018':'2021']
    records=[]
    for spec in CFG['specifications']:
        nav,decisions=PARENT.core.funded_nav(spec['name'],CFG['validation_end'])
        r=nav.excess_return
        folds=[previous.sharpe(r.loc[r.index.year!=year])-previous.sharpe(baseline.reindex(r.loc[r.index.year!=year].index)) for year in range(2018,2022)]
        records.append(dict(case=spec['name'],feature_group=spec['features'],
            robust_delta=float(min(folds)),leave_year_out_deltas=folds,
            **previous.period_metrics(nav,decisions,'2018-01-01',CFG['validation_end'])))
    point=sorted(records,key=lambda r:(-r['annualized_excess_sharpe'],r['case']))[0]
    key=lambda r:(-r['robust_delta'],-r['annualized_excess_sharpe'],r['case'])
    robust=sorted(records,key=key)[0]
    winners={f:sorted([r for r in records if r['feature_group']==f],key=key)[0]['case'] for f in CFG['feature_groups']}
    later=sorted(set(list(winners.values())+[point['case'],robust['case']]))
    record=dict(study_id='straddle_context_followup',selected=point['case'],robust_selected=robust['case'],
        feature_winners=winners,later_cases=later,candidates=records,
        preferred_policy=point['case'] if point['annualized_excess_sharpe']>previous.sharpe(baseline) else 'fixed50',
        evaluation_status=CFG['evaluation_status'])
    if path.exists():assert json.loads(path.read_text())==record
    else:PARENT.core.dump(path,record)
    pd.DataFrame(records).to_csv(REPORT/'validation_candidates.csv',index=False)
    print(json.dumps(dict(primary=point['case'],secondary=robust['case'],later_cases=later,preferred=record['preferred_policy'])),flush=True)


def report():
    cases=json.loads((REPORT/'selection.json').read_text())['later_cases']
    frames=[];annual=[]
    for case in cases:
        frames.append(pd.read_csv(REPORT/'cases'/case/'results.csv'))
        annual.append(pd.read_csv(REPORT/'cases'/case/'annual.csv'))
    controls=pd.read_csv('reports/straddle_overnight_study/entry_comparison.csv')
    frames.append(controls.loc[controls.case.isin(['fixed50','fixed75','fixed100'])])
    table=pd.concat(frames,ignore_index=True);table.to_csv(REPORT/'results.csv',index=False)
    pd.concat(annual,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    selected=json.loads((REPORT/'selection.json').read_text())
    lines=['STRADDLE CONTEXT FOLLOW-UP','All Sharpe ratios annualized above cash.',
        'Validation2018-2021; later2022-Aug2025. Previously examined historical dates.',
        'Primary: '+selected['selected']+'; robust secondary: '+selected['robust_selected']+'; validation-preferred: '+selected['preferred_policy'],
        'All 35,774 proposals and original costs/risk limits retained. Actions50/75/100%.','']
    for case,g in table.groupby('case',sort=True):
        scores={r.period:round(float(r.annualized_excess_sharpe),4) for r in g.itertuples()}
        lines.append(case+': '+json.dumps(scores))
    lines += ['', 'Sizing can change admission and the executed portfolio. Training uses independent completed reward paths; evaluation uses the original funded integer-option ledger.',
        'Feature associations and conditional bootstrap intervals do not establish a repeatable learning advantage. All validation attempts remain in validation_candidates.csv.']
    (REPORT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['test','features','assemble','fit','book','select','analyze','report'])
    p.add_argument('--case');p.add_argument('--later',action='store_true');p.add_argument('--months',type=int,default=8);p.add_argument('--limit',type=int,default=250)
    a=p.parse_args();configure()
    if a.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_context_followup','tests.test_research_atomic_io','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources()
    if a.phase=='features':prepare_features(a.limit)
    elif a.phase=='assemble':assemble()
    elif a.phase=='fit':previous.allocations(a.case,a.later)
    elif a.phase=='book':previous.book(a.case,a.months,a.later)
    elif a.phase=='select':select()
    elif a.phase=='analyze':previous.analyze(a.case)
    elif a.phase=='report':report()


if __name__=='__main__':main()
