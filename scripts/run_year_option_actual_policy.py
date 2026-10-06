"""Small episodic policy search trained in the actual shared portfolio."""
import argparse
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_year_option_sharpe_policy as shadow
from scripts import year_prefix_engine as engine

base,previous = shadow.base,shadow.previous
OUT = Path('artifacts/contracts/year_option_actual_policy/P25')
REPORT = Path('reports/year_option_actual_policy')
NAME = 'actual_book_2state_48M'
SHADOW_NAME = 'matched_shadow_2state_48M'


def configure():
    shadow.configure()
    base.OUT,base.REPORT = OUT,REPORT
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    base.dump(OUT/'design.json',dict(algorithm='Episodic contextual policy optimization in executable shared-cash simulator',
        windows_months=48,refit='annual2019–2024',states=2,min_leaf_cohorts=12,
        actions=[1,2,4],maximum_candidates_per_fit=9,training_commission=.65,
        actual_commission=.03,entry_features=shadow.entry.FEATURES,
        open_inventory='Marked through last historical close; no forced terminal liquidation or future maturity PnL',
        objective='Annualized daily cash-excess Sharpe including shared hedges and risk admissions',
        matched_shadow_control='Identical annual states,25/50/100 menu and48M prefixes; chooses policy on fractional summed independent cohort rewards',
        selection='Adaptive final check after shadow-trained failures; all explored history; no fresh holdout'))


def prefix_replay(market,rows,cal):
    cfg = base.source.configuration(365,training=True)
    old_risk,old_quote = engine.book_risk,market.quote
    latest = [cal[0]]
    def quote(date,oid):
        if date>cal[-1]:
            raise AssertionError('Training queried a future contract price')
        latest[0] = max(latest[0],date)
        return old_quote(date,oid)
    try:
        engine.book_risk = base.LONG.book_risk
        market.quote = quote
        nav,trades,decisions,audit = engine.simulate(market,rows,cal,cfg,1_000_000,
            policy=lambda row,context:int(row['learned_level']),corporate_events=base.LONG.base.UNIFORM_CORPORATE_EVENTS,
            wing_capital=1_000_000,require_runoff=False,settlement_calendar=market.rates.index)
        base.LONG.base.mark_book(nav,market.rates,cfg.capital)
        flow = (-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0) if len(trades) else pd.Series(0.,index=nav.index)
        expected = flow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        error = float((expected-nav.pnl).abs().max())
        assert error<1e-5
        return nav,dict(entries=int(decisions.status.eq('entered').sum()),
            final_active_cohorts=int(nav.active_cohorts.iloc[-1]),final_option_value=float(nav.option_value.iloc[-1]),
            latest_quote=str(latest[0].date()),independent_ledger_error=error)
    finally:
        engine.book_risk,market.quote = old_risk,old_quote


def fit():
    from scripts.report_long_tenor_search import sharpe
    packed = joblib.load(previous.OUT/'365D/episodes.joblib')
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet')
    market = base.market_for(365,'actual_policy_training')
    market.rate = lambda date:float(market.rates.loc[date])
    cal = market.rates.index[market.rates.index>='2018-01-01']
    for cutoff in pd.date_range('2019-01-01',rows.date.max(),freq='YS'):
        folder = OUT/'models'/str(cutoff.date())
        path = folder/'policy.joblib'
        if path.exists():
            continue
        folder.mkdir(parents=True,exist_ok=True)
        prefix = shadow.observed_prefixes(packed['episodes'],cutoff,48)
        history = cal[(cal>=cutoff-pd.DateOffset(months=48))&(cal<cutoff)]
        train_rows = rows.loc[rows.date.ge(history[0])&rows.date.lt(cutoff)].copy()
        if len(prefix)<16 or len(history)<252:
            joblib.dump(None,path)
            base.dump(folder/'status.json',dict(fit=str(cutoff.date()),trained_through=str(history[-1].date()),
                fitted=False,prefix_cohorts=len(prefix),history_sessions=len(history),candidates=[]))
            print(f'{cutoff.date()}: cold start, {len(history)} historical sessions; retain fixed50',flush=True)
            continue
        x = np.stack([ep['entry_features'] for ep,_ in prefix])
        y = np.array([ep['reward_training'][take].sum()/ep['entry_vega']*252/(len(take)-1) for ep,take in prefix])
        lo,hi = np.quantile(y,[.01,.99])
        tree = DecisionTreeRegressor(max_leaf_nodes=2,min_samples_leaf=12,random_state=2026)
        tree.fit(x,np.clip(y,lo,hi))
        leaves = np.unique(tree.apply(x))
        state = np.searchsorted(leaves,tree.apply(train_rows[shadow.entry.FEATURES].to_numpy(float)))
        grid = base.action_grid(leaves,[1,2,4]).astype(int)
        scores = []
        for index,choices in enumerate(grid):
            note_path = folder/f'candidate_{index}.json'
            if note_path.exists():
                result = json.loads(note_path.read_text())
            else:
                scored = train_rows.assign(learned_level=choices[state])
                nav,audit = prefix_replay(market,scored,history)
                result = dict(choices=choices.tolist(),training_sharpe=sharpe(nav.excess_return.iloc[1:]),**audit)
                base.dump(note_path,result)
            scores.append(result)
            print(f'{cutoff.date()} actual policy {choices.tolist()}: Sharpe {result["training_sharpe"]:.3f}, {result["entries"]} entries',flush=True)
        # For a numerical tie prefer the smaller requested allocation.
        best_score = max(r['training_sharpe'] for r in scores)
        candidates = [r for r in scores if r['training_sharpe']>=best_score-1e-10]
        selected = min(candidates,key=lambda r:sum(r['choices']))
        assert selected['training_sharpe']>=next(r['training_sharpe'] for r in scores if all(a==2 for a in r['choices']))-1e-10
        policy = dict(tree=tree,leaves=leaves,choices=np.array(selected['choices']))
        joblib.dump(policy,path)
        base.dump(folder/'status.json',dict(fit=str(cutoff.date()),trained_through=str(history[-1].date()),
            prefix_cohorts=len(prefix),selected=selected,candidates=scores))


def replay():
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet')
    old_engine = base.LONG.base.engine
    try:
        base.LONG.base.engine = engine
        base.save_replay(365,'fixed50_engine_control',rows.assign(learned_level=2))
        actual = pd.read_parquet(OUT/'365D/replays/fixed50_engine_control/daily_nav.parquet')
        expected = pd.read_parquet('artifacts/contracts/long_tenor_policy_search/P25/365D/replays/fixed_50_hold/daily_nav.parquet')
        pd.testing.assert_frame_equal(actual,expected,rtol=1e-12,atol=1e-6)
        base.save_replay(365,'fixed25_hold',rows.assign(learned_level=1))
        schedule = {pd.Timestamp(p.parent.name):joblib.load(p) for p in sorted((OUT/'models').glob('*/policy.joblib'))}
        scored = rows.copy().assign(learned_level=2)
        for index,row in rows.iterrows():
            dates = [d for d in schedule if d<=row.date]
            if dates and schedule[max(dates)] is not None and row.native_features_available:
                policy = schedule[max(dates)]
                leaf = policy['tree'].apply(row[shadow.entry.FEATURES].to_numpy(float)[None,:])[0]
                scored.loc[index,'learned_level'] = policy['choices'][np.searchsorted(policy['leaves'],leaf)]
        assert scored.learned_level.isin([1,2,4]).all()
        base.save_replay(365,NAME,scored)
    finally:
        base.LONG.base.engine = old_engine


def shadow_control():
    packed = joblib.load(previous.OUT/'365D/episodes.joblib')
    cal = packed['calendar']
    schedule = {}
    for path in sorted((OUT/'models').glob('*/policy.joblib')):
        cutoff = pd.Timestamp(path.parent.name)
        actual = joblib.load(path)
        shadow_policy = None
        if actual is not None:
            prefix = shadow.observed_prefixes(packed['episodes'],cutoff,48)
            history = cal[(cal>=cutoff-pd.DateOffset(months=48))&(cal<cutoff)]
            x = np.stack([ep['entry_features'] for ep,_ in prefix])
            states = np.searchsorted(actual['leaves'],actual['tree'].apply(x))
            book = np.zeros((len(history),len(actual['leaves'])))
            for (ep,take),column in zip(prefix,states):
                idx = history.get_indexer(ep['dates'][take])
                assert (idx>=0).all()
                np.add.at(book[:,column],idx,ep['reward_training'][take]/ep['original_level'])
            grid = base.action_grid(actual['leaves'],[1,2,4])
            choices,score = shadow.select_policy(book,grid)
            shadow_policy = dict(actual,choices=choices)
            base.dump(path.parent/'shadow_control.json',dict(fit=str(cutoff.date()),
                choices=choices.tolist(),shadow_training_sharpe=score,
                last_reward=str(history[-1].date()),same_native_state_tree=True))
        joblib.dump(shadow_policy,path.parent/'shadow_policy.joblib')
        schedule[cutoff] = shadow_policy
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet').assign(learned_level=2)
    for index,row in rows.iterrows():
        dates = [d for d in schedule if d<=row.date]
        if dates and schedule[max(dates)] is not None and row.native_features_available:
            policy = schedule[max(dates)]
            leaf = policy['tree'].apply(row[shadow.entry.FEATURES].to_numpy(float)[None,:])[0]
            rows.loc[index,'learned_level'] = policy['choices'][np.searchsorted(policy['leaves'],leaf)]
    base.save_replay(365,SHADOW_NAME,rows)


def holding_scan():
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet').assign(learned_level=2)
    market = base.market_for(365,'long_contract_short_holding_calendar')
    cal = market.rates.index[market.rates.index>='2018-01-01']
    positions = {d:i for i,d in enumerate(cal)}
    for sessions in [30,90,180]:
        def policy(date,held,sessions=sessions):
            age = positions[date]-positions[held[0]['entry_date']]
            return 0. if age>=sessions else 1.
        base.save_replay(365,f'hold_{sessions}_sessions',rows,policy)


def summarize():
    from scripts.report_long_tenor_search import sharpe,paired_bootstrap
    folders = [OUT/'365D/replays'/c for c in ['fixed50_engine_control','fixed25_hold',NAME]]
    if (OUT/'365D/replays'/SHADOW_NAME/'status.json').exists():
        folders.append(OUT/'365D/replays'/SHADOW_NAME)
    folders += [p.parent for p in sorted((OUT/'365D/replays').glob('hold_*_sessions/status.json'))]
    folders += [Path('artifacts/contracts/long_tenor_policy_search/P25/365D/replays/fixed_100_hold'),
                shadow.OUT/'365D/replays/fixed75_hold']
    records,annual,returns = [],[],{}
    for folder in folders:
        status = json.loads((folder/'status.json').read_text())
        r = pd.read_parquet(folder/'daily_nav.parquet').excess_return.iloc[1:]
        assert np.isclose(sharpe(r),status['annualized_excess_sharpe'])
        returns[status['case']] = r
        status.update(source_folder=str(folder),recent_2024_partial2025_sharpe=sharpe(r.loc['2024-01-01':]),
                      post_warmup_2020_partial2025_sharpe=sharpe(r.loc['2020-01-01':]))
        records.append(status)
        annual.append(pd.read_csv(folder/'annual.csv').assign(case=status['case']))
    frame = pd.DataFrame(records)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(annual).to_csv(REPORT/'annual.csv',index=False)
    fixed = frame.loc[frame.case.isin(['fixed50_engine_control','fixed25_hold','fixed75_hold','fixed_100_hold'])].sort_values('annualized_excess_sharpe').iloc[-1]['case']
    lo,hi = paired_bootstrap(returns[NAME],returns[fixed])
    base.dump(REPORT/'paired_comparison.json',dict(case=NAME,control=fixed,
        sharpe_gain=sharpe(returns[NAME])-sharpe(returns[fixed]),unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
    if SHADOW_NAME in returns:
        lo,hi = paired_bootstrap(returns[NAME],returns[SHADOW_NAME])
        base.dump(REPORT/'actual_vs_shadow.json',dict(case=NAME,control=SHADOW_NAME,
            sharpe_gain=sharpe(returns[NAME])-sharpe(returns[SHADOW_NAME]),unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
    notes = [json.loads(p.read_text()) for p in (OUT/'models').glob('*/status.json')]
    assert all(pd.Timestamp(n['trained_through'])<pd.Timestamp(n['fit']) for n in notes)
    assert all(pd.Timestamp(r['latest_quote'])<pd.Timestamp(n['fit']) for n in notes for r in n['candidates'])
    base.dump(REPORT/'audit.json',dict(actual_training_shared_cash=True,actual_training_integer_risk_admissions=True,
        no_future_quote_calls=True,open_training_inventory_marked=True,fixed_control_reproduced=True,
        costs_and_capital_limits_unchanged=True,one_worker=True,cpu_core_fraction=.5,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_fresh_holdout=True))
    paths = [Path(__file__),Path(engine.__file__),Path('tests/test_year_prefix_engine.py'),
             Path('scripts/run_signal_discovery_guarded.py'),previous.OUT/'365D/episodes.joblib']
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.joblib','.parquet']]
    base.dump(REPORT/'source_manifest.json',{str(p):base.digest(p) for p in paths})
    columns = ['case','annualized_excess_sharpe','entries','net_excess_pnl','recent_2024_partial2025_sharpe','post_warmup_2020_partial2025_sharpe']
    (REPORT/'findings.txt').write_text(
        'ACTUAL-PORTFOLIO ONE-YEAR POLICY SEARCH\n\n'
        'Adaptive follow-up after14 learned variants failed. Two native entry states, allocations25/50/100%, annual48-month past-only refits2019–2024. Enumerate at most9 policies per fit in the actual executable shared portfolio at .65 training commissions. Select daily cash-excess Sharpe. This is episodic contextual allocation policy optimization, not sequential/neural RL.\n'
        'Same529 frozen pair/date/direction opportunities, unchanged real .03 commissions,0.1-vol-point execution penalty,1bp stock hedges, and original global/pair/wing limits. Hold to original leg expiries in evaluation. Integer eligibility and portfolio admission occur during training as well as evaluation.\n'
        'Training admits long contracts that mature after the cutoff, retains their true scheduled expiry dates, and marks them at the last observed close. It neither forces an artificial cutoff liquidation nor accesses future contract marks/maturity PnL. Runtime guard rejects contract quote requests after the last training close. Prefix cash flows reconcile independently. Frozen engine default behavior and full-history fixed50 control are reproduced.\n'
        'No fresh holdout; all histories have been explored. Intervals are unadjusted for adaptive model selection.\n\n'
        'Matched shadow control uses exactly the same annual native state trees,25/50/100 action menu and48-month past prefixes, but optimizes summed independent fractional rewards. This isolates executable portfolio training from the earlier changes in action space and cadence.\n\n'
        'Additional30/90/180-observed-session holding scan keeps the same365D entry contracts and fixed50 entry size; close instructions occur at the session threshold and fill next session with observed executable quotes and the original penalties. These are fixed rules, not learned RL exits.\n\n'
        +frame[columns].to_string(index=False)+'\n\n'+(REPORT/'paired_comparison.json').read_text()+'\n')
    print(frame[columns].to_string(index=False),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','fit','replay','shadow_control','checks','summarize','sweep'],required=True)
    args = p.parse_args()
    if args.phase=='test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_year_prefix_engine'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    configure()
    if args.phase in ['fit','sweep']:
        fit()
    if args.phase in ['replay','sweep']:
        replay()
    if args.phase in ['shadow_control','checks']:
        shadow_control()
    if args.phase=='checks':
        holding_scan()
    if args.phase in ['summarize','sweep']:
        summarize()
    configure()


if __name__=='__main__':
    main()
