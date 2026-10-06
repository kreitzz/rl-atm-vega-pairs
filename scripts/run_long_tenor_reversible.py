"""Matched enriched-state irreversible/reversible long-put Q study."""
import argparse
from functools import partial
import gc
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from scripts import run_long_tenor_fitted_q as old
from scripts import reversible_management_engine as engine
from scripts.option_flex.black_scholes import option_values
from scripts.option_flex.contract_engine import valid_quote,valid_greeks

base,full = old.base,old.full
OUT = Path('artifacts/contracts/long_tenor_reversible/P25')
REPORT = Path('reports/long_tenor_reversible')
MODES = ['enriched_irreversible','enriched_reversible','risk_irreversible','risk_reversible']
EXTRA = ['iv_mean','net_gamma_per_entry_vega','gross_gamma_per_entry_vega',
         'net_vega_per_entry_vega','gross_vega_ratio','cohort_reserve_per_entry_vega',
         'global_reserve_utilization']
FEATURES = old.FEATURES+EXTRA
ACTIONS = np.array([0.,.5,1.])


def configure():
    full.setup('P25')
    base.OUT,base.REPORT = OUT,REPORT
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    base.protected()
    path = OUT/'protected_references.json'
    if not path.exists():
        sources = {}
        for tenor in base.TENORS:
            folders = [Path('artifacts/contracts/long_tenor_policy_search/P25')/f'{tenor}D/replays/fixed_50_hold',
                       old.OUT/f'{tenor}D/replays/fixed50_fitted_q_conservative_aligned_runoff']
            for folder in folders:
                sources.update({str(p):base.digest(p) for p in folder.iterdir() if p.is_file()})
        base.dump(path,sources)
    assert all(base.digest(p)==want for p,want in json.loads(path.read_text()).items())


def risk_features(date,held,cfg,global_utilization):
    denominator = np.mean([p['initial_vega'] for p in held])
    net_gamma = gross_gamma = net_vega = gross_vega = reserve = 0.
    for p in held:
        q = p['quote']
        quantity = p['initial_quantity']*100
        spot = p['spot']
        gamma = float(option_values(spot-q['known_dividend_pv'],q['strike'],
            (q['expiry']-date).days/365,q['bs_rate'],q['bs_iv'],q['cp_flag'])[2])
        dollar_gamma = quantity*gamma*spot**2
        net_gamma += dollar_gamma
        gross_gamma += abs(dollar_gamma)
        net_vega += quantity*q['vega']
        gross_vega += abs(quantity)*q['vega']
        reserve += abs(quantity)*((q['bid']+q['ask'])/2+cfg.stock_reserve_fraction*abs(q['delta'])*spot)
    result = np.array([np.mean([p['quote']['bs_iv'] for p in held]),
        net_gamma/denominator,gross_gamma/denominator,net_vega/denominator,
        gross_vega/(2*denominator),reserve/denominator,global_utilization])
    assert denominator>0 and np.isfinite(result).all()
    return result


def prepare(tenor):
    folder = OUT/f'{tenor}D'
    if (folder/'prepared.json').exists():
        return
    folder.mkdir(parents=True,exist_ok=True)
    source = full.OUT/'P25'/f'{tenor}D'
    packed = joblib.load(source/'episodes.joblib')
    rows = pd.read_parquet(source/'signal_rows.parquet')
    lookup = {f'{r.date.date()}|{r.pair}|{r.wing}':r for _,r in rows.iterrows()}
    control = Path('artifacts/contracts/long_tenor_policy_search/P25')/f'{tenor}D/replays/fixed_50_hold'
    nav = pd.read_parquet(control/'daily_nav.parquet')
    cfg = base.source.configuration(tenor)
    utilization = nav.capital_reserve/(cfg.max_reserve_fraction*nav.nav)
    market = base.market_for(tenor,'enriched_training_states')
    episodes = []
    for count,original in enumerate(packed['episodes'],1):
        ep = dict(original)
        row = lookup[ep['cohort']]
        sids,oids = [int(row.secid_a),int(row.secid_b)],[int(row.optionid_a),int(row.optionid_b)]
        first = ep['dates'][0]
        entry = [market.quote(first,oid) for oid in oids]
        quantities = [-ep['hedge'][0,k]/(100*q.delta) for k,q in enumerate(entry)]
        assert all(abs(n-round(n))<1e-6 for n in quantities)
        initial_adjustment = [market.stock(first,sid)[1] for sid in sids]
        last = [q.copy() for q in entry]
        states = np.zeros((len(ep['dates']),len(FEATURES)))
        restore_valid = np.zeros(len(ep['dates']),bool)
        for j,date in enumerate(ep['dates']):
            if not ep['state_known'][j]:
                continue
            held,raw = [],[]
            for k,(sid,oid) in enumerate(zip(sids,oids)):
                spot,adjustment = market.stock(date,sid)
                q = market.quote(date,oid)
                raw.append(q)
                if q is not None and valid_greeks(q) and np.isfinite(q.get('bs_iv',np.nan)):
                    last[k] = q.copy()
                else:
                    q = last[k].copy()
                    rate = float(market.rates.loc[date])
                    pv = market.dividend_pv(date,sid,q.expiry,rate)
                    price,delta,_,vega = option_values(spot-pv,q.strike,(q.expiry-date).days/365,rate,q.bs_iv,q.cp_flag)
                    q.update(bid=float(price),ask=float(price),delta=float(delta),vega=float(vega),bs_rate=rate,known_dividend_pv=pv)
                quantity = quantities[k]*adjustment/initial_adjustment[k]
                held.append(dict(initial_quantity=quantity,initial_vega=abs(quantities[k])*100*entry[k].vega,
                                 quote=q,spot=spot))
            states[j] = np.r_[ep['states'][j],risk_features(date,held,cfg,float(utilization.loc[date]))]
            if all(q is not None and valid_quote(q,True,cfg) and valid_greeks(q) for q in raw):
                vegas = [abs(p['initial_quantity'])*100*q.vega for p,q in zip(held,raw)]
                restore_valid[j] = abs(vegas[0]-vegas[1])/max(vegas)<=cfg.max_entry_vega_imbalance
        ep['states'] = states
        ep['restore_valid'] = restore_valid
        episodes.append(ep)
        if count%40==0:
            print(f'{tenor}D enriched {count}/{len(packed["episodes"])} native episodes',flush=True)
    joblib.dump(dict(episodes=episodes,calendar=packed['calendar']),folder/'episodes.joblib')
    base.dump(folder/'prepared.json',dict(tenor=tenor,episodes=len(episodes),features=FEATURES,
        source_episode_hash=base.digest(source/'episodes.joblib'),
        training_portfolio_context='Logged fixed50 account reserve utilization; local counterfactual rewards do not jointly resimulate other cohorts',
        gamma='Current European BS gamma on current declared-dividend-adjusted spot',
        restoration_maximum='Original half-size allocation only'))


def inputs(states,fractions,actions):
    state = np.asarray(states).copy()
    # Native exposure columns describe original counts. The current inventory
    # state scales them by retained fraction; signal and global context do not.
    state[:,7:12] *= fractions[:,None]
    return np.column_stack([state,fractions,actions])


def action_values(model,states,fractions,reversible):
    values = model.predict(inputs(np.repeat(states,3,axis=0),np.repeat(fractions,3),np.tile(ACTIONS,len(states)))).reshape(len(states),3)
    if not reversible:
        values[ACTIONS[None,:]>fractions[:,None]+1e-12] = -np.inf
    values[(fractions==0),1:] = -np.inf
    return values


def choose_actions(model,states,fractions,reversible):
    x = inputs(np.repeat(states,3,axis=0),np.repeat(fractions,3),np.tile(ACTIONS,len(states)))
    predictions = np.stack([tree.predict(x).reshape(len(states),3) for tree in model.estimators_])
    index = np.where(fractions==1.,2,1)
    advantage = predictions-predictions[:,np.arange(len(states)),index][:,:,None]
    scores = advantage.mean(axis=0)-advantage.std(axis=0,ddof=1)
    if not reversible:
        scores[ACTIONS[None,:]>fractions[:,None]+1e-12] = -np.inf
    # Equal values keep current inventory rather than increasing turnover.
    best = scores.argmax(axis=1)
    hold = scores[np.arange(len(states)),index]
    best = np.where(scores[np.arange(len(states)),best]<=hold+1e-12,index,best)
    return ACTIONS[best]


def block_reward(ep,start,end,current,requested):
    fill = start+1
    allowed = bool(ep['can_exit'][fill])
    if requested>.5 and requested>current:
        allowed = allowed and bool(ep['restore_valid'][start]) and bool(ep['restore_valid'][fill])
    if requested==.5 and ep['split_seen'][start]:
        allowed = False
    retained = requested if allowed else current
    reward = 0.
    for j in range(fill,end+1):
        previous = current if j==fill else retained
        change = retained*ep['hedge'][j]-previous*ep['hprev'][j]
        hedge_cost = (np.abs(change)*ep['spot'][j]).sum()*.0001
        # Restoring pays the same spread/commission as reducing; never a
        # negative transaction cost or free inventory increase.
        cost = abs(previous-retained)*ep['exit_cost_training'][j]/2
        reward += previous*ep['gross_training'][j]-hedge_cost-cost
    return reward/ep['entry_vega'],retained


def transitions(ep,reversible):
    n = len(ep['dates'])
    steps = np.flatnonzero(ep['state_known'] & (np.arange(n)>=5) & (np.arange(n)%5==0) & (np.arange(n)<n-1))
    result = []
    for k,start in enumerate(steps):
        end = int(steps[k+1]) if k+1<len(steps) else n-1
        for fraction in [1.,.5]:
            menu = ACTIONS if reversible else ACTIONS[ACTIONS<=fraction]
            for action in menu:
                reward,retained = block_reward(ep,int(start),end,fraction,float(action))
                result.append(dict(cohort=ep['cohort'],start=ep['dates'][start],end=ep['dates'][end],
                    state=ep['states'][start],fraction=fraction,action=float(action),reward=reward,
                    next_state=ep['states'][end],next_fraction=retained,
                    terminal=retained==0 or k+1==len(steps),discount=old.DISCOUNT**((end-start)/5)))
    return result


def risk_coefficient(reward,weights):
    mean = float(np.average(reward,weights=weights))
    second = float(np.average(reward**2,weights=weights))
    # Calibrate on past full-hold rewards so a constant unit exposure is
    # optimal for that reference distribution. Conditional poorer/riskier
    # opportunities can then favor half size. No negative risk coefficient.
    return max(0.,mean)/max(second,1e-12)


def fit_q(records,reversible,risk_adjusted=False):
    states,next_states = [np.stack([r[f] for r in records]) for f in ['state','next_state']]
    fractions,actions,reward,next_fractions,discount = [np.array([r[f] for r in records]) for f in ['fraction','action','reward','next_fraction','discount']]
    terminal = np.array([r['terminal'] for r in records])
    x = inputs(states,fractions,actions)
    cohort = pd.Series([r['cohort'] for r in records])
    weights = (1/cohort.map(cohort.value_counts())).to_numpy()
    weights /= weights.mean()
    hold = (fractions==1.) & (actions==1.)
    coefficient = risk_coefficient(reward[hold],weights[hold]) if risk_adjusted else 0.
    utility = reward-.5*coefficient*reward**2
    y = utility.copy()
    for _ in range(old.ITERATIONS):
        model = ExtraTreesRegressor(n_estimators=20,max_depth=5,min_samples_leaf=40,n_jobs=1,random_state=2026)
        model.fit(x,y,sample_weight=weights)
        continuation = action_values(model,next_states,next_fractions,reversible).max(axis=1)
        continuation[terminal] = 0
        y = utility+discount*continuation
        assert np.isfinite(y).all()
    return model,coefficient


def enough_history(note,reversible):
    """Count observed weekly steps, independent of counterfactual action rows."""
    unique_steps = note['transitions']//(6 if reversible else 5)
    return note['cohorts']>=16 and unique_steps>=128


def fit(tenor):
    packed = joblib.load(OUT/f'{tenor}D/episodes.joblib')
    for mode in MODES:
        reversible = mode.endswith('_reversible')
        records = [r for ep in packed['episodes'] for r in transitions(ep,reversible)]
        for cutoff in pd.date_range('2018-01-01',packed['calendar'][-1],freq='QS-JAN'):
            folder = OUT/f'{tenor}D/models'/mode
            path = folder/f'{cutoff.date()}.joblib'
            if path.exists():
                continue
            past = old.causal_transitions(records,cutoff)
            cohorts = len({r['cohort'] for r in past})
            unique_steps = len(past)//(6 if reversible else 5)
            eligible = enough_history(dict(cohorts=cohorts,transitions=len(past)),reversible)
            model,coefficient = fit_q(past,reversible,mode.startswith('risk_')) if eligible else (None,0.)
            folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,path)
            base.dump(path.with_suffix('.json'),dict(fit=str(cutoff.date()),tenor=tenor,mode=mode,
                transitions=len(past),unique_weekly_steps=unique_steps,cohorts=cohorts,fitted=model is not None,
                reward_objective='past-hold-calibrated mean-variance' if mode.startswith('risk_') else 'mean reward',risk_coefficient=coefficient,
                last_training_step=str(max(r['end'] for r in past).date()) if past else None))
            print(f'{tenor}D {mode} {cutoff.date()}: {cohorts} past cohorts, {len(past)} steps',flush=True)
            gc.collect()


def callback(schedule,rv,cal,cfg,reversible):
    positions = {d:i for i,d in enumerate(cal)}
    def policy(date,held):
        current = min(p['management_fraction'] for p in held) if held else 1.
        if len(held)!=2:
            return current
        age = positions[date]-positions[held[0]['entry_date']]
        model = schedule[max(k for k in schedule if k<=date)]
        if model is None or age<5 or age%5:
            return current
        values = full.slow_features(date,held,rv)
        if values is None:
            return current
        exposure = risk_features(date,held,cfg,held[0]['book_context']['reserve_utilization'])
        state = np.r_[[values[f] for f in old.FEATURES],exposure][None,:]
        return float(choose_actions(model,state,np.array([current]),reversible)[0])
    return policy


def replay(tenor):
    rows = pd.read_parquet(full.OUT/'P25'/f'{tenor}D/signal_rows.parquet').assign(learned_level=2)
    market = base.market_for(tenor,'reversible_calendar')
    cal = market.rates.index[market.rates.index>='2018-01-01']
    rv,cfg = base.load_rv(),base.source.configuration(tenor)
    # A real full-history hold control must reproduce every original NAV field
    # before this fork is used for either learned controller.
    previous_engine = base.LONG.base.engine
    original_simulate = engine.simulate
    try:
        base.LONG.base.engine = engine
        engine.simulate = partial(original_simulate,reversible_management=True)
        base.save_replay(tenor,'fixed50_hold_fork_control',rows,lambda date,held:min(p['management_fraction'] for p in held))
        original = pd.read_parquet(Path('artifacts/contracts/long_tenor_policy_search/P25')/f'{tenor}D/replays/fixed_50_hold/daily_nav.parquet')
        actual = pd.read_parquet(OUT/f'{tenor}D/replays/fixed50_hold_fork_control/daily_nav.parquet')
        pd.testing.assert_frame_equal(actual,original,rtol=1e-12,atol=1e-6)
        for mode in MODES:
            reversible = mode.endswith('_reversible')
            engine.simulate = partial(original_simulate,reversible_management=reversible)
            schedule = {}
            for path in sorted((OUT/f'{tenor}D/models'/mode).glob('*.joblib')):
                note = json.loads(path.with_suffix('.json').read_text())
                enough = enough_history(note,reversible)
                schedule[pd.Timestamp(path.stem)] = joblib.load(path) if enough else None
            case = mode+'_aligned_warmup'
            base.save_replay(tenor,case,rows,callback(schedule,rv,cal,cfg,reversible))
            folder = OUT/f'{tenor}D/replays'/case
            trades = pd.read_csv(folder/'trades.csv',parse_dates=['date'])
            restored = trades.loc[trades.reason.eq('learned_policy_restore')]
            if len(restored):
                np.testing.assert_allclose(restored.commission,restored.quantity.abs()*cfg.option_commission)
                assert (pd.to_datetime(restored.exit_signal_date)<restored.date).all()
                assert restored.groupby(['cohort','date']).secid.nunique().eq(2).all()
            status = json.loads((folder/'status.json').read_text())
            status['restore_fills'] = len(restored)
            status['restored_cohorts'] = int(restored.cohort.nunique())
            log = json.loads((folder/'management_log.json').read_text())
            status['restoration_rejections'] = pd.Series([r['status'] for r in log if r['status'].startswith('restore_')]).value_counts().to_dict()
            base.dump(folder/'status.json',status)
    finally:
        engine.simulate = original_simulate
        base.LONG.base.engine = previous_engine


def summarize():
    from scripts.report_long_tenor_search import paired_bootstrap,sharpe
    records,years,comparisons = [],[],[]
    for tenor in base.TENORS:
        folders = [OUT/f'{tenor}D/replays/fixed50_hold_fork_control',
                   old.OUT/f'{tenor}D/replays/fixed50_fitted_q_conservative_aligned_runoff']
        folders += [OUT/f'{tenor}D/replays'/(mode+'_aligned_warmup') for mode in MODES]
        r_by_case = {}
        for folder in folders:
            status = json.loads((folder/'status.json').read_text())
            nav = pd.read_parquet(folder/'daily_nav.parquet')
            r = nav.excess_return.iloc[1:]
            assert np.isclose(sharpe(r),status['annualized_excess_sharpe'])
            status.update(source_folder=str(folder),recent_2024_partial2025_sharpe=sharpe(r.loc['2024-01-01':]))
            status.setdefault('restore_fills',0)
            records.append(status)
            r_by_case[status['case']] = r
            years.append(pd.read_csv(folder/'annual.csv').assign(tenor=tenor,case=status['case']))
        for mode in MODES:
            for control in ['fixed50_hold_fork_control','fixed50_fitted_q_conservative_aligned_runoff']:
                case = mode+'_aligned_warmup'
                lo,hi = paired_bootstrap(r_by_case[case],r_by_case[control])
                comparisons.append(dict(tenor=tenor,case=case,control=control,
                    sharpe_gain=sharpe(r_by_case[case])-sharpe(r_by_case[control]),unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
    frame = pd.DataFrame(records)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(years,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(comparisons).to_csv(REPORT/'paired_comparison.csv',index=False)
    portfolio_and_chart(frame)
    fits = [json.loads(p.read_text()) for p in OUT.glob('*D/models/*/*.json')]
    assert all(r['last_training_step'] is None or pd.Timestamp(r['last_training_step'])<pd.Timestamp(r['fit']) for r in fits)
    deployment = []
    for tenor in base.TENORS:
        for mode in MODES:
            eligible = sorted(r['fit'] for r in fits if r['tenor']==tenor and r['mode']==mode
                              and r['fitted'] and enough_history(r,mode.endswith('_reversible')))
            deployment.append(dict(tenor=tenor,mode=mode,first_active_fit=eligible[0] if eligible else None,
                                   active_refits=len(eligible)))
    base.dump(REPORT/'deployment.json',deployment)
    source_paths = [Path(__file__),Path('scripts/reversible_management_engine.py'),Path('tests/test_long_tenor_reversible.py'),
                    Path('scripts/run_signal_discovery_guarded.py'),Path('scripts/run_long_tenor_fitted_q.py'),
                    Path('scripts/run_long_tenor_full_universe.py'),Path('scripts/run_long_tenor_policy_search.py')]
    source_paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.joblib','.parquet','.csv']]
    source_paths += [full.OUT/'P25'/f'{t}D/episodes.joblib' for t in base.TENORS]
    source_paths += [full.OUT/'P25'/f'{t}D/signal_rows.parquet' for t in base.TENORS]
    base.dump(REPORT/'source_manifest.json',{str(p):base.digest(p) for p in source_paths})
    base.dump(REPORT/'audit.json',dict(all_original_checkpoints_unchanged=True,full_history_fork_hold_controls_reproduced=True,
        restored_inventory_at_most_original_half_size=True,restores_risk_checked_atomically=True,
        restore_increment_vega_tolerance=.1,costs_and_risk_limits_unchanged=True,
        all_529_proposals_explicit=True,completed_weekly_steps_purged=True,
        shared_activation_minimum='16 cohorts and 128 unique observed weekly transitions, independent of action augmentation',
        training_global_context='Historical fixed50 reference book; inference uses live book. Counterfactual training does not jointly resimulate other cohorts.',
        training_shadow='Fractional isolated hedges; actual evaluation integer shared cash',
        one_worker=True,cpu_core_fraction=.5,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (REPORT/'findings.txt').write_text(
        'ENRICHED INVENTORY STATE AND REVERSIBLE LONG-TENOR RL\n\n'
        'Same 529 pair/date/direction proposals, native 180D/365D P25 contracts, fixed 50% entries. Entry vega-matched, daily delta hedges; no continuous vega rebalance. Natural runoff matches the prior control, including different leg expiry dates.\n'
        'Compare prior selective FQI, richer-state irreversible FQI, and richer-state reversible FQI. Same ExtraTrees depth5, 20 trees, 16 Bellman updates, quarterly 48-month completed-weekly-step fits and ensemble-disagreement action penalty. State adds current IV level, net/gross dollar gamma, vega drift, local capital reserve and global reserve utilization. Shadow quantities scale with retained inventory.\n'
        'Both menus activate only after 16 cohorts and 128 unique weekly transitions. Earlier models/replays counted augmented action rows, allowing the reversible 365D model to activate one quarter earlier; those replay folders are retained and excluded from final comparison. A cached model below the shared activation minimum is not deployed.\n'
        'After the six-month mean-reward policies chose no trims, add an explicitly adaptive reward ablation: utility r-k*r^2/2, with k=max(0,past mean full-hold reward)/past full-hold second moment, using only completed prior transitions and cohort-balanced weights. This is a local Sharpe-like mean-variance surrogate, not an exact portfolio Sharpe objective. Compare irreversible/reversible menus under both reward definitions; retain the original mean-reward results.\n'
        'Reversible actions restore half-trimmed pairs to their original counts only; fully closed cohorts cannot reopen. Both legs require observed executable quotes, prior-known OI, live incremental vega match within10% and unchanged global/pair/wing risk checks after option and hedge costs. No added leverage. Failed requests retain inventory. Nondecision days retain current inventory.\n'
        'Logged fixed50 global capital state enters training; actual book state enters inference. Local counterfactual weekly rewards and fractional hedges approximate portfolio interactions, while final evaluation is the reconciled shared-cash integer book. Training commission .65, real .03, 0.1-vol-point option penalties and1bp stock execution unchanged. European BS and undeclared-dividend/assignment limitations persist.\n'
        'All history has been explored; intervals are unadjusted for search, and there is no fresh holdout.\n\n'
        +frame[['tenor','case','annualized_excess_sharpe','entries','trims','exits','restore_fills','recent_2024_partial2025_sharpe']].to_string(index=False)+'\n\n'
        +'Best saved-return blend (exploratory; no joint portfolio replay):\n'+(REPORT/'best_portfolio.json').read_text()+'\n')
    print(frame[['tenor','case','annualized_excess_sharpe','entries','trims','exits','restore_fills','recent_2024_partial2025_sharpe']].to_string(index=False),flush=True)


def portfolio_and_chart(frame):
    from scripts.report_long_tenor_search import optimal_nonnegative_weights,sharpe
    panel = pd.DataFrame({'30D':pd.read_parquet('artifacts/contracts/p25_long_management/replays/30D_manage_36M/daily_nav.parquet').excess_return.iloc[1:]})
    components = {'30D':'previous managed 30D checkpoint'}
    for tenor in base.TENORS:
        rows = frame.loc[frame.tenor.eq(tenor)]
        chosen = rows.loc[rows.annualized_excess_sharpe.idxmax()]
        panel[f'{tenor}D'] = pd.read_parquet(Path(chosen.source_folder)/'daily_nav.parquet').excess_return.iloc[1:]
        components[f'{tenor}D'] = chosen.case
    weights = optimal_nonnegative_weights(panel)
    r = panel@weights
    base.dump(REPORT/'best_portfolio.json',dict(annualized_excess_sharpe=sharpe(r),
        weights=dict(zip(panel.columns,weights)),components=components,
        previous_optimized_blend_sharpe=1.0994499458252018,
        full_history_optimized=True,shared_portfolio_replayed=False,no_fresh_holdout=True))
    r.to_csv(REPORT/'best_portfolio_daily.csv',header=['excess_return'])
    pd.DataFrame([dict(year=int(y),annualized_excess_sharpe=sharpe(v)) for y,v in r.groupby(r.index.year)]).to_csv(REPORT/'best_portfolio_annual.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    order = ['fixed50_hold_fork_control','fixed50_fitted_q_conservative_aligned_runoff']+[m+'_aligned_warmup' for m in MODES]
    labels = ['Fixed','Prior\nselective','Enriched\nexit/trim','Enriched\nrestore','Risk utility\nexit/trim','Risk utility\nrestore'][:len(order)]
    for ax,tenor in zip(axes,base.TENORS):
        selected = frame.loc[frame.tenor.eq(tenor)].set_index('case').loc[order]
        values = selected.annualized_excess_sharpe
        ax.bar(np.arange(len(values)),values,color=['#94a3b8','#64748b','#2563eb','#93c5fd','#166534','#86efac'][:len(values)])
        ax.set_xticks(np.arange(len(values)),labels,fontsize=8)
        ax.axhline(0,color='black',linewidth=.6)
        ax.set_title(f'{tenor}-day 25-delta puts')
        for i,v in enumerate(values):
            ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9)
    axes[0].set_ylabel('Annualized daily cash-excess Sharpe')
    fig.suptitle('Richer risk state and restoration within the original entry allocation')
    fig.text(.5,.01,'2018–partial 2025. Same costs and limits; mean-variance reward is an adaptive follow-up. No fresh holdout.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(REPORT/'comparison.png',dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','prepare','fit','replay','summarize','sweep'],required=True)
    p.add_argument('--tenor',type=int,choices=base.TENORS)
    args = p.parse_args()
    if args.phase=='test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_long_tenor_reversible'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    configure()
    for tenor in ([args.tenor] if args.tenor else base.TENORS):
        if args.phase in ['prepare','sweep']:
            prepare(tenor)
        if args.phase in ['fit','sweep']:
            fit(tenor)
        if args.phase in ['replay','sweep']:
            replay(tenor)
    if args.phase in ['summarize','sweep']:
        summarize()
    configure()


if __name__=='__main__':
    main()
