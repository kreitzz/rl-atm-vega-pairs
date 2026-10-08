"""Bounded sizing audit; see the predeclared sizing/protocol.json."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import Ridge
from scripts import run_p25_representation as base
from scripts.diagnose_p25_pca import training_window

ROOT = base.ROOT
OUT = ROOT / 'artifacts/contracts/p25_strategy_audit/sizing'
REPORT = ROOT / 'reports/p25_strategy_audit/sizing'
CONTROLS = dict(tree_cem='screened_selected', fixed50='screened_fixed50',
                fixed75='screened_fixed75', fixed100='screened_fixed100')
NEW = ['rank_terciles', 'ridge_terciles']
CASES = list(CONTROLS) + NEW


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT / 'STOP').exists():
            raise InterruptedError('Sizing audit STOP requested')


class Sources:
    def __init__(self):
        self.hashes = json.loads((OUT/'sources.json').read_text()) if (OUT/'sources.json').exists() else {}
        self.stats = {}

    def track(self, path):
        path = Path(path)
        key = path.relative_to(ROOT).as_posix()
        if key not in self.stats:
            actual=base.digest(path)
            if key in self.hashes and key != Path(__file__).relative_to(ROOT).as_posix() and actual!=self.hashes[key]:
                raise ValueError(f'Source changed between audit stages: {key}')
            self.hashes[key] = actual
            self.stats[key] = (path.stat().st_size,path.stat().st_mtime_ns)
        return path

    def save(self):
        for key,stat in self.stats.items():
            path=ROOT/key
            if (path.stat().st_size,path.stat().st_mtime_ns)!=stat:
                raise ValueError(f'Source changed during audit: {key}')
        base.write(OUT / 'sources.json', self.hashes)


def prepare(sources):
    protocol = json.loads(sources.track(REPORT / 'protocol.json').read_text())
    sources.track(Path(__file__))
    for p in ['scripts/run_p25_representation.py', 'scripts/uniform_smile_engine.py',
              'scripts/uniform_smile_risk.py', 'scripts/p25_representation.py']:
        sources.track(ROOT / p)
    cfg = json.loads(sources.track(base.CONFIG).read_text())
    inputs = cfg['screened']['inputs']
    manifest = json.loads(sources.track(ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    sources.track(base.ARCHIVE)
    meta = pd.read_parquet(sources.track(base.SOURCE), columns=inputs + [
        'date', 'end', 'pair', 'signal_eligible', 'finite_label', 'target', 'position', 'label_row', 'direction'])
    cal = pd.read_parquet(sources.track(base.MARKET / 'rates.parquet')).index
    with zipfile.ZipFile(base.ARCHIVE) as z:
        signals = pd.read_parquet(io.BytesIO(base.archive_read(z, base.REFERENCE + 'scored_allocations.parquet', manifest)))
    missing = [c for c in inputs if c not in signals]
    if missing:
        signals = signals.merge(meta[['date', 'pair'] + missing], on=['date', 'pair'], how='left', validate='one_to_one')
    assert len(signals) == 608 and np.isfinite(signals[inputs]).all().all()
    sources.save()
    return meta, signals, inputs, cal, manifest


def tercile_levels(values, edges):
    return np.searchsorted(edges, np.asarray(values), side='right') + 2


def train_simple_rules(rows, inputs):
    """All scalers and cutoffs depend solely on the supplied completed window."""
    x=rows[inputs].to_numpy(float)
    mean,sd=x.mean(axis=0),x.std(axis=0)
    sd[sd<1e-12]=1
    ridge=Ridge(alpha=len(rows),solver='svd').fit((x-mean)/sd,rows.target.to_numpy())
    return dict(rank_edges=rows.rank_strength.quantile([1/3,2/3]).to_numpy(),
        ridge=ridge,mean=mean,sd=sd,prediction_edges=np.quantile(ridge.predict((x-mean)/sd),[1/3,2/3]))


def fit_alternatives(meta, signals, inputs, manifest, sources, budget):
    allocations = signals[['date', 'pair']].copy()
    allocations['tree_cem'] = signals.learned_level.to_numpy()
    for c, v in [('fixed50', 2), ('fixed75', 3), ('fixed100', 4)]:
        allocations[c] = v
    for c in NEW:
        allocations[c] = np.nan
    monthly, leaves, coefficients = [], [], []
    previous = None
    union = set()
    with zipfile.ZipFile(base.ARCHIVE) as z:
        for month in pd.period_range('2018-01', '2025-06', freq='M'):
            rows = training_window(meta, month.start_time).reset_index(drop=True)
            stamp = str(month.start_time.date())
            ids = signals.index[signals.date.dt.to_period('M').eq(month)]
            model = joblib.load(io.BytesIO(base.archive_read(z, base.MODEL_ROOT + stamp + '.joblib', manifest)))
            original_meta = json.loads(base.archive_read(z, base.MODEL_ROOT + stamp + '.json', manifest))
            tree = model['tree']
            trleaf = tree.apply(rows[inputs])
            action = dict(zip(model['leaves'].tolist(), model['choices'].tolist()))
            actions = np.array([action[l] for l in trleaf])
            if len(ids):
                actual = tree.apply(signals.loc[ids, inputs])
                np.testing.assert_array_equal([action[l] for l in actual], signals.loc[ids, 'learned_level'])
            keys = pd.MultiIndex.from_frame(rows[['date', 'pair']])
            union.update(keys.tolist())
            current = pd.Series(actions, index=keys)
            common = current.index.intersection(previous.index) if previous is not None else []
            changed = int((current.loc[common] != previous.loc[common]).sum()) if len(common) else 0
            record = dict(fit=stamp, training_rows=len(rows), distinct_signal_dates=rows.date.nunique(),
                distinct_pairs=rows.pair.nunique(), leaves=len(model['leaves']),
                min_leaf_samples=int(min(tree.tree_.n_node_samples[model['leaves']])),
                max_leaf_samples=int(max(tree.tree_.n_node_samples[model['leaves']])),
                common_with_previous=len(common), changed_actions_on_common=changed,
                changed_action_fraction=changed/len(common) if len(common) else np.nan,
                latest_completed_outcome=str(rows.end.max().date()), proposals=len(ids),
                root_feature=inputs[tree.tree_.feature[0]],
                training_shadow_sharpe=original_meta.get('training_shadow_daily_sharpe'),
                training_fullsize_shadow_sharpe=original_meta.get('baseline_shadow_daily_sharpe'))
            for leaf in model['leaves']:
                part = rows.loc[trleaf == leaf]
                assert len(part) == tree.tree_.n_node_samples[leaf]
                np.testing.assert_allclose(part.target.to_numpy(float).mean(), tree.tree_.value[leaf].item(),rtol=1e-10,atol=1e-8)
                leaves.append(dict(fit=stamp, leaf=int(leaf), size=action[leaf]/4, samples=len(part),
                    signal_dates=part.date.nunique(), pairs=part.pair.nunique(), mean_target=part.target.mean(),
                    target_sd=part.target.std(), positive_fraction=part.target.gt(0).mean()))
            previous = current
            model_path = OUT / 'models' / (stamp + '.joblib')
            if model_path.exists():
                fitted = joblib.load(model_path)
            else:
                fitted = dict(**train_simple_rules(rows,inputs), columns=inputs, cutoff=stamp,
                              protocol_sha256=base.digest(REPORT/'protocol.json'),
                              training_source_sha256=sources.hashes[base.SOURCE.relative_to(ROOT).as_posix()],
                              last_outcome=str(rows.end.max().date()), n_train=len(rows))
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(fitted, model_path)
            assert fitted['n_train'] == len(rows) and fitted['last_outcome'] == record['latest_completed_outcome']
            assert fitted['protocol_sha256'] == base.digest(REPORT/'protocol.json')
            assert fitted['training_source_sha256'] == sources.hashes[base.SOURCE.relative_to(ROOT).as_posix()]
            if len(ids):
                allocations.loc[ids, 'rank_terciles'] = tercile_levels(signals.loc[ids, 'rank_strength'], fitted['rank_edges'])
                pred = fitted['ridge'].predict((signals.loc[ids, inputs].to_numpy()-fitted['mean'])/fitted['sd'])
                allocations.loc[ids, 'ridge_terciles'] = tercile_levels(pred, fitted['prediction_edges'])
            for f, coef in zip(inputs, fitted['ridge'].coef_):
                coefficients.append(dict(fit=stamp, feature=f, standardized_coefficient=float(coef)))
            monthly.append(record)
            budget.rest()
    assert allocations[CASES].isin([2,3,4]).all().all()
    allocations[CASES] = allocations[CASES].astype(int)
    allocations.to_csv(REPORT / 'allocations.csv', index=False)
    pd.DataFrame(monthly).to_csv(REPORT / 'monthly_diagnostics.csv', index=False)
    pd.DataFrame(leaves).to_csv(REPORT / 'leaf_diagnostics.csv', index=False)
    pd.DataFrame(coefficients).to_csv(REPORT / 'ridge_coefficients.csv', index=False)
    base.write(REPORT / 'training_verification.json', dict(all_608_saved_sizes_reproduced=True,
        all_leaf_counts_and_means_reproduced=True, distinct_training_examples=len(union), monthly_models=90,
        protocol_sha256=base.digest(REPORT/'protocol.json')))
    print('Verified90 saved trees and608 allocations; fitted90 ridge models and rank cutoffs.', flush=True)
    sources.save()
    return allocations


def controls(manifest, signals, sources):
    submission_manifest = json.loads(sources.track(base.BUNDLE/'manifest.json').read_text())
    sub_hash = {r['stored']:r['sha256'] for r in submission_manifest['files']}
    with zipfile.ZipFile(base.ARCHIVE) as z:
        for case, oldcase in CONTROLS.items():
            folder = OUT / 'replays' / case
            folder.mkdir(parents=True, exist_ok=True)
            prefix = f'artifacts/contracts/p25_validation_split/replays/{oldcase}/'
            for name in ['daily_nav.parquet', 'trades.csv', 'decisions.csv']:
                if case in ['fixed50','fixed100']:
                    source=base.BUNDLE/'artifacts/contracts/p25_capital_sensitivity/replays'/('10m_'+case)/name
                    sources.track(source)
                    raw=source.read_bytes()
                    expected=sub_hash[source.relative_to(ROOT).as_posix()]
                    if hashlib.sha256(raw).hexdigest()!=expected:
                        # Git's Windows checkout may convert tracked CSV LF to CRLF.
                        assert name.endswith('.csv') and hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==expected
                else:
                    raw = base.archive_read(z, prefix+name, manifest)
                (folder/name).write_bytes(raw)
            if case in ['fixed50','fixed100']:
                rows=signals.copy()
                rows['learned_level']=2 if case=='fixed50' else 4
                rows.to_parquet(folder/'scored_allocations.parquet',index=False)
                dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
                assert len(dec)==608 and dec.level.eq(2 if case=='fixed50' else 4).all()
                pd.testing.assert_frame_equal(dec[['signal_date','pair']].rename(columns={'signal_date':'date'}).sort_values(['date','pair']).reset_index(drop=True),
                    signals[['date','pair']].sort_values(['date','pair']).reset_index(drop=True))
            else:
                (folder/'scored_allocations.parquet').write_bytes(base.archive_read(z,prefix+'scored_allocations.parquet',manifest))
            nav = pd.read_parquet(folder/'daily_nav.parquet')
            if case == 'tree_cem':
                prior = pd.read_parquet(base.OUT/'replays/raw_full/daily_nav.parquet')
                pd.testing.assert_frame_equal(nav, prior, check_exact=False, rtol=1e-12, atol=1e-6)
            base.write(folder/'status.json', dict(status='complete', source='hash_verified_archive_or_submission', archive_case=oldcase))


def replay(signals, allocations, cal, sources, budget):
    pd.testing.assert_frame_equal(allocations[['date','pair']].reset_index(drop=True),signals[['date','pair']].reset_index(drop=True))
    verification=json.loads((REPORT/'training_verification.json').read_text())
    assert verification['protocol_sha256']==base.digest(REPORT/'protocol.json')
    for p in base.MARKET.iterdir():
        if p.is_file(): sources.track(p)
    settings = base.BUNDLE / 'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json'
    cfg = base.Settings(**json.loads(sources.track(settings).read_text()))
    ck = json.loads(sources.track(ROOT/'config/p25_submission.json').read_text())
    for case in NEW:
        folder = OUT/'replays'/case
        signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),
            allocation_table_sha256=base.digest(REPORT/'allocations.csv'),engine_sha256=base.digest(ROOT/'scripts/uniform_smile_engine.py'))
        if (folder/'status.json').exists():
            status=json.loads((folder/'status.json').read_text())
            saved=pd.read_parquet(folder/'scored_allocations.parquet')
            pd.testing.assert_frame_equal(saved[['date','pair']].reset_index(drop=True),signals[['date','pair']].reset_index(drop=True))
            np.testing.assert_array_equal(saved.learned_level,allocations[case])
            signature['scored_allocations_sha256']=base.digest(folder/'scored_allocations.parquet')
            for key,value in signature.items():
                if key in status and status[key]!=value: raise ValueError(f'Replay binding mismatch: {case}/{key}')
            if not all(k in status for k in signature):
                nav=pd.read_parquet(folder/'daily_nav.parquet')
                trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
                decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
                base.verify_ledger(nav,trades,decisions,saved)
                base.write(folder/'status.json',dict(**status,**signature))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        rows = signals.copy()
        rows['learned_level'] = allocations[case].to_numpy(int)
        rows.to_parquet(folder/'scored_allocations.parquet', index=False)
        market = base.ThrottledMarket(base.MARKET, budget)
        original = base.engine.book_risk
        print(f'Funded replay {case}', flush=True)
        try:
            base.engine.book_risk = base.book_risk
            nav, trades, decisions, ledger = base.engine.simulate(market, rows, cal[cal>='2018-01-01'], cfg, 1_000_000,
                policy=lambda row,context:int(row['learned_level']), corporate_events=ck['corporate_events'], wing_capital=1_000_000)
        finally:
            base.engine.book_risk = original
        base.mark_book(nav, market.rates, cfg.capital)
        audit = base.verify_ledger(nav,trades,decisions,rows)
        nav.to_parquet(folder/'daily_nav.parquet')
        trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        signature['scored_allocations_sha256']=base.digest(folder/'scored_allocations.parquet')
        base.write(folder/'status.json',dict(status='complete',case=case,**audit,**ledger,**signature))
        print(f'{case}: {audit["entries"]} entries; ledger reconciled.',flush=True)
        sources.save()
        del market,nav,trades,decisions,rows
        gc.collect()
        budget.rest()


def report(budget):
    books,decisions,periods,annual,rejections=[],{},[],[],[]
    books={}
    for case in CASES:
        folder=OUT/'replays'/case
        assert (folder/'status.json').exists()
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        books[case],decisions[case]=nav,dec
        for name,start,end in [('validation','2018-01-01','2021-12-31'),('evaluation','2022-01-01','2025-08-29'),('full','2018-01-01','2025-08-29')]:
            periods.append(dict(case=case,period=name,**base.period_metrics(nav,dec,start,end)))
            for reason,n in dec.loc[dec.entry_date.between(start,end),'status'].value_counts().items():
                rejections.append(dict(case=case,period=name,status=reason,count=int(n)))
        for year in range(2018,2026):
            annual.append(dict(case=case,year=year,**base.period_metrics(nav,dec,f'{year}-01-01',f'{year}-12-31')))
    period=pd.DataFrame(periods)
    period.to_csv(REPORT/'periods.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(rejections).to_csv(REPORT/'admissions.csv',index=False)
    intervals=[]
    b=books['tree_cem'].loc['2022-01-01':,'excess_return'].to_numpy()
    for case in CASES[1:]:
        assert books[case].index.equals(books['tree_cem'].index)
        a=books[case].loc['2022-01-01':,'excess_return'].to_numpy()
        for block in [21,63,126]:
            rng=np.random.default_rng(2026)
            values=[]
            for _ in range(2000):
                starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/block)))
                ix=((starts[:,None]+np.arange(block)[None])%len(a)).ravel()[:len(a)]
                values.append(base.sharpe(a[ix])-base.sharpe(b[ix]))
            intervals.append(dict(case=case,reference='tree_cem',block_sessions=block,point=base.sharpe(a)-base.sharpe(b),
                lower95=float(np.quantile(values,.025)),upper95=float(np.quantile(values,.975)),replications=2000))
            budget.rest()
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_intervals.csv',index=False)
    monthly=pd.read_csv(REPORT/'monthly_diagnostics.csv')
    leaf=pd.read_csv(REPORT/'leaf_diagnostics.csv')
    alloc=pd.read_csv(REPORT/'allocations.csv')
    size=[]
    for case in CASES:
        for level in [2,3,4]:
            size.append(dict(case=case,size_fraction=level/4,proposals=int(alloc[case].eq(level).sum())))
    pd.DataFrame(size).to_csv(REPORT/'size_distribution.csv',index=False)
    vals=period.loc[period.period.eq('validation')].set_index('case')
    evals=period.loc[period.period.eq('evaluation')].set_index('case')
    selected=max(['fixed50','fixed75','fixed100'],key=lambda c:vals.loc[c,'annualized_excess_sharpe'])
    summary=dict(selected_constant_on_validation=selected,
        train_rows=dict(min=int(monthly.training_rows.min()),median=float(monthly.training_rows.median()),max=int(monthly.training_rows.max())),
        training_pairs=dict(min=int(monthly.distinct_pairs.min()),median=float(monthly.distinct_pairs.median()),max=int(monthly.distinct_pairs.max())),
        training_dates=dict(min=int(monthly.distinct_signal_dates.min()),median=float(monthly.distinct_signal_dates.median()),max=int(monthly.distinct_signal_dates.max())),
        leaves_per_tree=dict(min=int(monthly.leaves.min()),median=float(monthly.leaves.median()),max=int(monthly.leaves.max())),
        leaf_samples=dict(min=int(leaf.samples.min()),median=float(leaf.samples.median()),max=int(leaf.samples.max())),
        leaf_pairs=dict(min=int(leaf.pairs.min()),median=float(leaf.pairs.median()),max=int(leaf.pairs.max())),
        leaves_total=len(leaf),leaf_samples_below15=int(leaf.samples.lt(15).sum()),
        adjacent_model_change_fraction=float(monthly.changed_actions_on_common.sum()/monthly.common_with_previous.sum()),
        median_monthly_change_fraction=float(monthly.changed_action_fraction.median()),
        peak_rss_mib=budget.peak,protocol_sha256=base.digest(REPORT/'protocol.json'))
    base.write(REPORT/'summary.json',summary)
    lines=['P25 position-sizing audit','',
        'Scope: current 27-input monthly tree/CEM versus 3 constants and 2 predeclared simpler pipelines; identical 608 qualified proposals. Historical results are exploratory, not an untouched holdout.','',
        'Case | Validation Sharpe | Later Sharpe | Later profit above cash | Later entries']
    for c in CASES:
        r=evals.loc[c]
        lines.append(f'{c} | {vals.loc[c,"annualized_excess_sharpe"]:.3f} | {r.annualized_excess_sharpe:.3f} | ${r.profit_above_cash:,.0f} | {int(r.entries)}')
    lines+=['',f'Constant selected using 2018–2021 only: {selected}.','',
        f'Training examples per monthly fit: {summary["train_rows"]}; leaf sample counts: {summary["leaf_samples"]}.',
        f'Leaves per tree: {summary["leaves_per_tree"]}; distinct pairs per tree: {summary["training_pairs"]}; distinct pairs per leaf: {summary["leaf_pairs"]}.',
        f'{summary["leaf_samples_below15"]}/{len(leaf)} monthly leaves have fewer than 15 observations. Overlapping dates/pairs and overlapping monthly windows mean these are not independent observations.',
        f'Across adjacent fits, {summary["adjacent_model_change_fraction"]:.1%} of common training-example sizing assignments change. This is descriptive model-update sensitivity on shared historical examples, not an out-of-sample turnover or refit-bootstrap estimate.','',
        'Mechanism: tree minimizes squared error of realized full-size standalone lifetime P&L, not Sharpe. CEM separately chooses 50/75/100 percent per leaf to maximize daily aggregate historical shadow-path Sharpe. Thus splits need not all affect size, and training return prediction does not directly optimize final funded risk-adjusted performance.',
        'A single shallow tree is a coherent interpretable mapping from entry context to size; it need not represent eight persistent market regimes. Monthly leaf IDs are local partitions and cannot be interpreted as the same economic state across different fits. The design is reasonable as a benchmark but needs evidence that its complexity improves funded performance.',
        'Original labels carry $0.65/contract training commission; funded engine uses $0.03/contract plus 0.10 vol-point option penalty and 1 bp stock hedge costs. Shadow paths are additive standalone trades; funded engine has shared cash/hedges, integer contracts, financing and admission limits. The surrogate objective and final account objective are different.',
        'No zero action is available to the sizing policy. Every qualified opportunity requests at least 50%; funded risk/eligibility checks can subsequently reject it. Opportunity quality cannot be solved entirely by this action space.',
        'Rank rule is monotonically larger for stronger rank_strength using completed-training terciles. Ridge is a regularized continuous linear score on all 27 original inputs, then quantized to the same three sizes using training-score terciles. It is not a test of truly continuous action sizes; retained engine only accepts the original ladder.',
        'Rank/ridge arms replace the full tree/CEM pipeline, including objective and flexibility. Their failure would not isolate tree topology or rule out a continuous portfolio-aware sizing optimizer. No hyperparameter search was conducted.','',
        'Paired later Sharpe differences: candidate minus current tree; 95% intervals, 126-session circular blocks.']
    for r in intervals:
        if r['block_sessions']==126: lines.append(f'{r["case"]}: {r["point"]:+.3f} [{r["lower95"]:+.3f}, {r["upper95"]:+.3f}]')
    lines+=['','Intervals use 2,000 paired calendar-block resamples; 21/63/126 sensitivity is in sharpe_intervals.csv. No correction for strategy search, new-arm design after prior history inspection, or multiple comparisons.','',
        'Verification: control files SHA256-verified (Windows CSV line endings canonicalized to LF where required); current tree NAV equals prior exact replay; all 608 saved sizing assignments and every training leaf count/mean reproduced; new funded ledgers independently reconcile. Models/cutoffs and source hashes are saved.','']
    (REPORT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    x=np.arange(len(CASES))
    axes[0].bar(x-.18,[vals.loc[c,'annualized_excess_sharpe'] for c in CASES],.36,label='2018–2021')
    axes[0].bar(x+.18,[evals.loc[c,'annualized_excess_sharpe'] for c in CASES],.36,label='2022–Aug2025')
    axes[0].set_xticks(x,CASES,rotation=30,ha='right');axes[0].set_ylabel('Daily cash-excess Sharpe');axes[0].legend()
    axes[0].set_title('Same opportunities; different entry sizing')
    for c in CASES:
        nav=books[c];anchor=nav.loc[:'2021-12-31'].iloc[-1];part=nav.loc['2022-01-01':]
        axes[1].plot(part.index,100*((part.nav/part.cash_benchmark)/(anchor.nav/anchor.cash_benchmark)-1),label=c)
    axes[1].set_title('Later growth relative to cash');axes[1].set_ylabel('%');axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x',rotation=25)
    fig.savefig(REPORT/'comparison.png',dpi=160);plt.close(fig)
    base.write(REPORT/'completion.json',dict(status='complete',cases=6,new_replays=2,reused_controls=4,**summary))
    print(period[['case','period','annualized_excess_sharpe','profit_above_cash','entries']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','train','replay','report'],default='all')
    args=parser.parse_args()
    if base.sklearn.__version__ != '1.9.0':
        raise RuntimeError('Use pinned sklearn1.9.0 environment for exact saved tree verification')
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib')
    budget=Budget()
    sources=Sources()
    meta,signals,inputs,cal,manifest=prepare(sources)
    if args.phase in ['all','train']:
        allocations=fit_alternatives(meta,signals,inputs,manifest,sources,budget)
    else:
        allocations=pd.read_csv(REPORT/'allocations.csv',parse_dates=['date'])
    del meta
    gc.collect()
    if args.phase in ['all','replay']:
        controls(manifest,signals,sources)
        replay(signals,allocations,cal,sources,budget)
    if args.phase in ['all','report']: report(budget)
    sources.save()


if __name__=='__main__':main()
