"""A costed fitted-Q P&L/risk frontier with path-dependent drawdown budgets.

All policies allocate fractional units of two saved, funded strategy indices.
They share next-close timing, costs, warmup and drawdown controls. An additional
discount-zero control distinguishes continuation value from one-week prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import atm_p25_allocation_q as old
from scripts.report_put_straddle_bandits import BOOKS, PERIODS
from paper_code.metrics import sharpe

CONFIG = Path('config/put_straddle_q_frontier.json')
CFG = json.loads(CONFIG.read_text())
OUT, REPORT = Path(CFG['output_root']), Path(CFG['report_root'])
LEVELS = np.array([[a['put'], a['straddle']] for a in CFG['actions']])
TARGETS = np.column_stack([LEVELS, 1-LEVELS.sum(axis=1)])
A, D, R = len(LEVELS), len(CFG['drawdown_grid']), 4
GRID = np.array(CFG['drawdown_grid'])
NS = R*D*A


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str)+'\n')


def sources():
    files = [CONFIG, Path(__file__), Path(old.__file__), Path('scripts/report_put_straddle_bandits.py'),
             Path('scripts/report_atm_p25_combination.py'), Path('paper_code/execution.json'), Path('data/raw/VIX_History.csv')]
    for name in ['put_bandit', 'straddle_bandit']:
        files += [BOOKS[name]/'daily_nav.parquet', BOOKS[name]/'trades.csv']
    record = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = OUT/'sources.json'
    if path.exists():
        assert json.loads(path.read_text()) == record, 'Changed sources: retain this run and use a new output root'
    else:
        dump(path, record)


def legal_actions(limit):
    legal = np.ones((D, A), bool)
    if limit is not None:
        legal[GRID >= CFG['soft_budget_fraction']] = LEVELS[:, 1] == 0
        legal[GRID >= 1] = np.arange(A) == 0
    return legal


def state_id(regime, drawdown_index, previous_index):
    return (np.asarray(regime)*D+np.asarray(drawdown_index))*A+np.asarray(previous_index)


def nearest_action(weights):
    return ((np.asarray(weights)[..., None, :2]-LEVELS)**2).sum(axis=-1).argmin(axis=-1)


def drawdown_index(drawdown, limit):
    ratio = np.asarray(drawdown)/(limit or .10)
    index = np.abs(ratio[..., None]-GRID).argmin(axis=-1)
    if limit is not None:
        index = np.where(ratio >= 1, D-1, index)
        index = np.where((ratio >= CFG['soft_budget_fraction']) & (ratio < 1), np.maximum(index,4), index)
        index = np.where(ratio < CFG['soft_budget_fraction'],np.minimum(index,3),index)
    return index


def week_path(m, previous, action, initial_drawdown, limit, already_halted=False):
    """Vectorized frozen-index execution; a close signal trades NEXT close.

    First-session returns accrue to previous holdings. Drawdown may overshoot
    while a queued risk reduction waits for that next close. Cash financing uses
    the implied underlying wallet balances, not gross index weights as a loan.
    """
    w = np.atleast_2d(previous).astype(float).copy()
    n = len(w)
    actions = np.broadcast_to(np.asarray(action, int), (n,)).copy()
    dd = np.broadcast_to(np.asarray(initial_drawdown, float), (n,))
    nav, peak = np.ones(n), 1/(1-dd)
    stopped = np.broadcast_to(np.asarray(already_halted, bool), (n,)).copy()
    hard = (dd >= limit) if limit is not None else np.zeros(n, bool)
    soft = np.zeros(n, bool)
    floor = np.zeros(n, int)
    profit, variance, hits, fees = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)
    daily, nav_paths, dd_paths, weight_paths, hit_paths = [], [], [], [], []
    for t, gross in enumerate(m['gross']):
        implied_cash = w[:, 2] + (w[:, :2]*m['cash_fractions'][t]).sum(axis=1)
        embedded_borrow = (w[:, :2]*np.minimum(m['cash_fractions'][t], 0)).sum(axis=1)
        finance = .01*m['elapsed'][t]*(np.minimum(implied_cash, 0)-embedded_borrow)
        values = w*gross
        values[:, 2] += finance
        growth = values.sum(axis=1)
        if (growth <= 0).any():
            raise ValueError('Nonpositive index-portfolio equity')
        w = values/growth[:, None]
        nav *= growth
        # The hard/soft flags here came from the preceding close.
        execute = np.full(n, t == 0)
        target = TARGETS[actions].copy()
        if t:
            execute |= soft
            target[soft] = TARGETS[floor[soft]]
        execute |= hard | stopped
        target[hard | stopped] = TARGETS[0]
        charge = np.zeros(n)
        charge[execute] = (np.abs(target[execute, :2]-w[execute, :2])*m['costs'][t]).sum(axis=1)
        if (charge >= 1).any():
            raise ValueError('Invalid reallocation fee')
        growth *= 1-charge
        nav *= 1-charge
        peak = np.maximum(peak, nav)
        fees += charge
        w[execute] = target[execute]
        stopped |= hard
        dd = 1-nav/peak
        excess = growth-1-m['cash'][t]
        profit += 100*excess
        variance += (100*excess)**2
        new_hit = ((dd >= limit) & ~stopped & ~hard) if limit is not None else np.zeros(n, bool)
        hits |= new_hit
        hard = new_hit
        soft = ((dd >= CFG['soft_budget_fraction']*limit) & (w[:, 1] > 1e-10) & ~stopped) if limit is not None else np.zeros(n, bool)
        floor = np.where(w[:, 0] > .1, 1, 0)
        daily.append(excess.copy()); nav_paths.append(nav.copy()); dd_paths.append(dd.copy()); weight_paths.append(w.copy());hit_paths.append(new_hit.copy())
    return dict(profit=profit, variance=variance, hit=hits, stopped=stopped, pending=hard,
        drawdown=dd, weights=w, fees=fees, daily=np.stack(daily), nav=np.stack(nav_paths),
        drawdowns=np.stack(dd_paths), daily_weights=np.stack(weight_paths), hard_crossings=np.stack(hit_paths))


def prepare():
    import joblib
    dest = OUT/'prepared.joblib'
    if dest.exists():
        return
    columns = ['nav', 'return_', 'excess_return', 'cash_benchmark', 'cash', 'capital_reserve',
               'gross_vega', 'gross_hedge_notional', 'stress_loss']
    pair, straddle = [pd.read_parquet(BOOKS[k]/'daily_nav.parquet', columns=columns, use_threads=False)
                      for k in ['put_bandit', 'straddle_bandit']]
    cash = old.blend.aligned_cash(pair, straddle)
    known = old.market_features(pair, straddle)
    regimes = ((known.atm_sharpe63 > known.pair_sharpe63).astype(int)
               + 2*(known.vix >= 1.25).astype(int)).to_numpy()
    costs = []
    for name, book in [('put_bandit', pair), ('straddle_bandit', straddle)]:
        trades = pd.read_csv(BOOKS[name]/'trades.csv', usecols=['date','cohort','optionid','quantity'], parse_dates=['date'])
        counts = old.contract_counts(trades, pair.index)
        costs.append((.1*book.gross_vega+.0001*book.gross_hedge_notional+.03*counts)/book.nav)
        del trades
    cost = np.column_stack(costs)
    fractions = np.column_stack([pair.cash/pair.nav, straddle.cash/straddle.nav])
    fractions = np.vstack([fractions[0], fractions[:-1]])
    elapsed = pair.index.to_series().diff().dt.days.fillna(0).to_numpy()/365
    gross = np.column_stack([1+pair.return_, 1+straddle.return_, 1+cash])
    weeks, previous = [], None
    for _, positions in pd.Series(np.arange(len(pair)), index=pair.index).groupby(pair.index.to_period('W-FRI')):
        ii = positions.to_numpy(); prior = int(ii[0])-1
        before = np.tile(TARGETS[1], (A, 1)) if previous is None else previous['canonical_after']
        m = dict(cutoff=pair.index[ii[0]], end=pair.index[ii[-1]], feature_date=pair.index[prior] if prior >= 0 else None,
            regime=int(regimes[max(prior, 0)]), indices=ii, before=before,
            gross=gross[ii], cash=cash.iloc[ii].to_numpy(), costs=cost[ii], cash_fractions=fractions[ii], elapsed=elapsed[ii])
        m['canonical_after'] = week_path(m, TARGETS, np.arange(A), np.zeros(A), None)['weights']
        weeks.append(m); previous = m
    data = dict(pair=pair, straddle=straddle, cash=cash, weeks=weeks)
    from scripts.research_atomic_io import atomic_dump
    atomic_dump(data,dest,joblib.dump)
    dump(REPORT/'protocol.json', dict(configuration=CFG, market_regimes='VIX below/above25, crossed with straddle-minus-put trailing63-session Sharpe sign; prior-close only.',
        states=f'{R} market regimes x {D} drawdown buckets x {A} prior-exposure buckets; {A} actions.',
        counterfactuals='Each completed market week supplies outcomes for all discretized prior-exposure/drawdown/action states; these are dependent, not independent extra episodes.',
        policy='Tabular fitted Bellman iteration, conditional means shrunk toward the same-action/previous-exposure/drawdown mean across market regimes by8 episodes.',
        drawdown=CFG['drawdown'], source_selection=CFG['scope']))
    print(json.dumps(dict(weeks=len(weeks), dates=len(pair), actions=A, states=NS)), flush=True)


def load():
    import joblib
    return joblib.load(OUT/'prepared.joblib')


def load_transitions(cap):
    with np.load(OUT/f'transitions_{cap}.npz') as archive:
        return {k:archive[k] for k in archive.files}


def transitions(cap):
    dest = OUT/f'transitions_{cap}.npz'
    if dest.exists():
        return
    data = load(); limit = CFG['drawdown_budgets'][cap]
    di, pi, ai = np.indices((D,A,A)).reshape(3,-1)
    legal = legal_actions(limit)[di,ai]
    records = {k: [] for k in ['row','following','profit','variance','stop_penalty','done']}
    for i, m in enumerate(data['weeks']):
        z = week_path(m, m['before'][pi], ai, GRID[di]*(limit or .10), limit)
        next_regime = data['weeks'][min(i+1,len(data['weeks'])-1)]['regime']
        follow = state_id(next_regime, drawdown_index(z['drawdown'],limit), nearest_action(z['weights']))
        row = state_id(m['regime'],di,pi)*A+ai
        for key, value in dict(row=row, following=follow, profit=z['profit'], variance=z['variance'],
            stop_penalty=z['hit']*100*(limit or 0), done=z['stopped']).items():
            records[key].append(value[legal])
    pending=dest.with_name('.'+dest.stem+'.pending.npz')
    np.savez_compressed(pending, **{k:np.stack(v) for k,v in records.items()})
    pending.replace(dest)
    print(json.dumps(dict(phase='transitions',cap=cap,shape=np.stack(records['row']).shape)), flush=True)


def fit(data, trans, index, window, preference, gamma, cap):
    cutoff = data['weeks'][index]['cutoff']
    train = [i for i,m in enumerate(data['weeks'][:index]) if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=window)]
    if len(train) < CFG['minimum_training_weeks']:
        return None
    rows = trans['row'][train].ravel()
    following = trans['following'][train].ravel()
    done = trans['done'][train].ravel()
    reward = (trans['profit'][train]-preference*trans['variance'][train]-trans['stop_penalty'][train]).ravel()
    counts = np.bincount(rows, minlength=NS*A).reshape(R,D,A,A)
    pooled_count = counts.sum(axis=0)
    legal = np.broadcast_to(legal_actions(CFG['drawdown_budgets'][cap])[None,:,None,:], (R,D,A,A)).reshape(NS,A)
    q = np.zeros((NS,A))
    for _ in range(CFG['bellman_iterations'] if gamma else 1):
        continuation = np.where(legal,q,-1e12).max(axis=1)
        target = reward+gamma*continuation[following]*(~done)
        sums = np.bincount(rows, weights=target, minlength=NS*A).reshape(R,D,A,A)
        pooled = np.divide(sums.sum(axis=0),pooled_count,out=np.zeros_like(sums[0]),where=pooled_count>0)
        q = ((sums+CFG['prior_episodes']*pooled)/(counts+CFG['prior_episodes'])).reshape(NS,A)
    last = max(data['weeks'][i]['end'] for i in train)
    assert last < cutoff
    return dict(q=q.tolist(),cutoff=str(cutoff.date()),last_training_end=str(last.date()),training_weeks=len(train),
        window=window,preference=preference,gamma=gamma,cap=cap)


def label(window, preference, gamma):
    return f'w{window}_risk{preference:g}_g{gamma:g}'


def model_for(data, trans, index, window, preference, gamma, cap):
    m = data['weeks'][index]
    path = OUT/'models'/cap/label(window,preference,gamma)/(str(m['cutoff'].to_period('M'))+'.json')
    if path.exists():
        model = json.loads(path.read_text())
        if pd.Timestamp(model['cutoff']) > m['cutoff']:
            return None
    else:
        model = fit(data,trans,index,window,preference,gamma,cap)
        if model is not None:
            dump(path,model)
    if model is not None:
        assert pd.Timestamp(model['last_training_end']) < m['cutoff']
    return model


def replay(data, trans, cap, case, stop, preference, window=None, gamma=None, constant=None):
    path = OUT/'replays'/cap/case
    stem = 'validation' if stop == CFG['validation_end'] else 'full'
    if (path/(stem+'_complete.json')).exists():
        return pd.read_parquet(path/(stem+'.parquet'),use_threads=False), pd.read_csv(path/(stem+'_decisions.csv'))
    limit = CFG['drawdown_budgets'][cap]
    w = TARGETS[1].copy(); wealth = peak = 1.; halted = False
    daily, decisions = [], []
    for index,m in enumerate(data['weeks']):
        if m['cutoff'] > pd.Timestamp(stop):
            break
        dd = 1-wealth/peak
        model = None
        action = 1
        if halted:
            action = 0
        elif index >= CFG['minimum_training_weeks']:
            if constant is not None:
                action = constant
            else:
                model = model_for(data,trans,index,window,preference,gamma,cap)
                if model is not None:
                    row = int(state_id(m['regime'],drawdown_index(dd,limit),nearest_action(w)))
                    scores = np.array(model['q'])[row]
                    allowed = legal_actions(limit)[int(drawdown_index(dd,limit))]
                    # Preserve the put-only warmup action in exact ties.
                    order = np.array([1,0,5,2,3,4,6,7])
                    action = int(order[np.argmax(np.where(allowed,scores,-1e12)[order])])
        if limit is not None:
            if dd >= limit:
                action = 0
            elif dd >= CFG['soft_budget_fraction']*limit and LEVELS[action,1] > 0:
                action = 1 if LEVELS[action,0] else 0
        z = week_path(m,w[None],action,dd,limit,halted)
        global_nav = wealth*z['nav'][:,0]
        for j,position in enumerate(m['indices']):
            daily.append(dict(date=data['pair'].index[position],excess_return=z['daily'][j,0],cash_return=m['cash'][j],
                nav=CFG['capital']*global_nav[j],drawdown=z['drawdowns'][j,0],
                put_weight=z['daily_weights'][j,0,0],straddle_weight=z['daily_weights'][j,0,1],
                hard_penalty=100*(limit or 0)*z['hard_crossings'][j,0]))
        wealth=float(global_nav[-1]);peak=max(peak,float(global_nav.max()));w=z['weights'][0]
        halted=halted or bool(z['stopped'][0])
        decisions.append(dict(date=m['cutoff'],feature_date=m['feature_date'],action=CFG['actions'][action]['name'],
            fitted=model is not None,last_training_end=model['last_training_end'] if model else None,
            training_weeks=model['training_weeks'] if model else 0,drawdown_before=dd,drawdown_after=float(z['drawdown'][0]),
            halted=halted,stop_pending=bool(z['pending'][0]),resizing_cost_fraction=float(z['fees'][0])))
    book=pd.DataFrame(daily).set_index('date'); d=pd.DataFrame(decisions)
    highwater=np.maximum.accumulate(np.r_[CFG['capital'],book.nav.to_numpy()])[1:]
    np.testing.assert_allclose(book.drawdown,1-book.nav/highwater,atol=1e-12,rtol=0)
    path.mkdir(parents=True,exist_ok=True)
    book.to_parquet(path/(stem+'.parquet'))
    d.to_csv(path/(stem+'_decisions.csv'),index=False)
    dump(path/(stem+'_complete.json'),dict(rows=len(book),case=case,stop=stop))
    return book,d


def score(book,start,stop,preference):
    from scripts.report_atm_p25_combination import stats
    result=stats(book.excess_return,book.cash_return,start,stop)
    part=book.loc[start:stop];first=book.index.get_loc(part.index[0]);before=max(0,first-1)
    benchmark=CFG['capital']*(1+book.cash_return).cumprod()
    result.update(profit_above_cash=float((part.nav.iloc[-1]-benchmark.loc[part.index[-1]])-(book.nav.iloc[before]-benchmark.iloc[before])),
        max_account_drawdown_pct=-100*float(part.drawdown.max()),
        average_put_exposure=float(part.put_weight.mean()),average_straddle_exposure=float(part.straddle_weight.mean()))
    r=part.excess_return.iloc[1:] if first==0 else part.excess_return
    result['utility']=float((100*r-preference*(100*r)**2).sum()-part.hard_penalty.sum())
    return result


def validation(cap):
    data=load();trans=load_transitions(cap);records=[]
    for risk in CFG['risk_preferences']:
        for window in CFG['training_weeks']:
            case=label(window,risk,CFG['discount'])
            book,d=replay(data,trans,cap,case,CFG['validation_end'],risk,window,CFG['discount'])
            row=dict(cap=cap,preference=risk,kind='q',case=case,window=window,
                     **score(book,'2018-01-01',CFG['validation_end'],risk))
            records.append(row);print(json.dumps(row),flush=True)
    for action in range(A):
        case='constant_'+CFG['actions'][action]['name']
        book,d=replay(data,trans,cap,case,CFG['validation_end'],0.,constant=action)
        for risk in CFG['risk_preferences']:
            records.append(dict(cap=cap,preference=risk,kind='constant',case=case,action=action,
                **score(book,'2018-01-01',CFG['validation_end'],risk)))
    pd.DataFrame(records).to_csv(REPORT/f'validation_{cap}.csv',index=False)


def select():
    records=pd.concat([pd.read_csv(REPORT/f'validation_{cap}.csv') for cap in CFG['drawdown_budgets']],ignore_index=True)
    selected=[]
    for (cap,risk),group in records.groupby(['cap','preference']):
        entry=dict(cap=cap,preference=float(risk))
        for kind in ['q','constant']:
            best=group.loc[group.kind.eq(kind)].sort_values(['utility','case'],ascending=[False,True]).iloc[0]
            entry[kind]=best.dropna().to_dict()
        selected.append(entry)
    dump(REPORT/'selection.json',dict(selected=selected,rule=CFG['selection']))
    records.to_csv(REPORT/'validation_candidates.csv',index=False)


def evaluate(cap):
    data=load();trans=load_transitions(cap)
    selected=json.loads((REPORT/'selection.json').read_text())['selected']
    for entry in selected:
        if entry['cap']!=cap:
            continue
        risk=entry['preference'];q=entry['q'];window=int(q['window'])
        for gamma in [CFG['discount'],0.]:
            case=label(window,risk,gamma)
            book,d=replay(data,trans,cap,case,CFG['evaluation_end'],risk,window,gamma)
            if gamma:
                earlier=pd.read_parquet(OUT/'replays'/cap/case/'validation.parquet',use_threads=False)
                np.testing.assert_allclose(book.excess_return.loc[earlier.index],earlier.excess_return,atol=1e-14,rtol=0)
        constant=entry['constant']
        replay(data,trans,cap,constant['case'],CFG['evaluation_end'],risk,constant=int(constant['action']))
    # Preserve the whole fixed frontier, including controls not selected on validation.
    for action in range(A):
        replay(data,trans,cap,'constant_'+CFG['actions'][action]['name'],CFG['evaluation_end'],0.,constant=action)


def report():
    rows=[];annual=[];comparisons=[]
    selected=json.loads((REPORT/'selection.json').read_text())['selected']
    for e in selected:
        cap,risk=e['cap'],e['preference'];window=int(e['q']['window'])
        cases=[('q',label(window,risk,CFG['discount'])),('myopic',label(window,risk,0.)),('selected_fixed',e['constant']['case'])]
        decisions={}
        for kind,case in cases:
            root=OUT/'replays'/cap/case
            book=pd.read_parquet(root/'full.parquet',use_threads=False)
            d=pd.read_csv(root/'full_decisions.csv',parse_dates=['date','feature_date','last_training_end'])
            fitted=d.loc[d.fitted]
            decisions[kind]=d
            assert fitted.last_training_end.lt(fitted.date).all()
            assert d.loc[d.feature_date.notna(),'feature_date'].lt(d.loc[d.feature_date.notna(),'date']).all()
            for period,start,stop in PERIODS:
                part=d.loc[d.date.between(start,stop)]
                rows.append(dict(cap=cap,preference=risk,kind=kind,case=case,period=period,window=window,
                    **score(book,start,stop,risk),fitted_weeks=int(part.fitted.sum()),
                    risk_halted=bool(part.halted.any()),first_halt_date=str(d.loc[d.halted,'date'].min()),
                    drawdown_budget_pct=100*(CFG['drawdown_budgets'][cap] or 0),
                    resizing_cost_bps=10000*float(part.resizing_cost_fraction.sum())))
            for year in range(2018,2026):
                annual.append(dict(cap=cap,preference=risk,kind=kind,year=year,
                    **score(book,f'{year}-01-01',min(f'{year}-12-31',CFG['evaluation_end']),risk)))
        for period,start,stop in PERIODS:
            qd=decisions['q'].set_index('date').loc[start:stop]
            for control in ['myopic','selected_fixed']:
                other=decisions[control].set_index('date').loc[start:stop]
                assert qd.index.equals(other.index)
                comparisons.append(dict(cap=cap,preference=risk,period=period,control=control,
                    action_disagreement_fraction=float(qd.action.ne(other.action).mean())))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    fixed=[]
    for cap in CFG['drawdown_budgets']:
        for action in range(A):
            case='constant_'+CFG['actions'][action]['name'];book=pd.read_parquet(OUT/'replays'/cap/case/'full.parquet',use_threads=False)
            for period,start,stop in PERIODS:
                fixed.append(dict(cap=cap,case=case,period=period,**score(book,start,stop,0.)))
    pd.DataFrame(fixed).to_csv(REPORT/'fixed_frontier.csv',index=False)
    comparisons=pd.DataFrame(comparisons)
    for i,row in comparisons.iterrows():
        group=result.loc[result.cap.eq(row.cap)&result.preference.eq(row.preference)&result.period.eq(row.period)].set_index('kind')
        for metric in ['annualized_excess_sharpe','profit_above_cash','annualized_growth_relative_to_cash_pct','max_account_drawdown_pct']:
            comparisons.loc[i,'difference_'+metric]=group.loc['q',metric]-group.loc[row.control,metric]
    comparisons.to_csv(REPORT/'comparisons.csv',index=False)
    data=load();anchors=[]
    for name,raw in [('put_bandit',data['pair']),('straddle_bandit',data['straddle'])]:
        book=raw.copy();book['cash_return']=data['cash'];book['drawdown']=1-book.nav/np.maximum.accumulate(np.r_[CFG['capital'],book.nav.to_numpy()])[1:]
        book['put_weight']=float(name=='put_bandit');book['straddle_weight']=float(name=='straddle_bandit');book['hard_penalty']=0.
        for period,start,stop in PERIODS:anchors.append(dict(case=name,period=period,**score(book,start,stop,0.)))
    pd.DataFrame(anchors).to_csv(REPORT/'unmodified_source_anchors.csv',index=False)
    summary=result.loc[result.kind.eq('q'),['cap','preference','period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct','profit_above_cash','max_account_drawdown_pct','risk_halted']]
    (REPORT/'FINDINGS.txt').write_text(__doc__+'\n\n'+summary.to_string(index=False)+'\n\n'+comparisons.to_string(index=False)+'\n\n'+json.dumps(CFG,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=pd.read_csv(REPORT/'results.csv');fixed=pd.read_csv(REPORT/'fixed_frontier.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    colors=dict(dd02='#607da4',dd05='#3a9a81',dd10='#d99b37',unlimited='#b55c67')
    for ax,period,title in zip(axes,['validation','evaluation'],['2018–2021 validation','2022–Aug 2025 later period']):
        f=fixed.loc[fixed.period.eq(period)]
        ax.scatter(f.annualized_growth_relative_to_cash_pct,f.annualized_excess_sharpe,c='#aaaaaa',marker='x',alpha=.65,label='Fixed allocations + same guards')
        for cap,color in colors.items():
            q=result.loc[result.period.eq(period)&result.kind.eq('q')&result.cap.eq(cap)].sort_values('preference')
            label='No drawdown budget' if cap=='unlimited' else cap[2:]+'% drawdown budget'
            ax.plot(q.annualized_growth_relative_to_cash_pct,q.annualized_excess_sharpe,'o-',color=color,label=label)
            for row in q.itertuples():
                ax.annotate(f'λ={row.preference:g}',(row.annualized_growth_relative_to_cash_pct,row.annualized_excess_sharpe),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_title(title);ax.set_xlabel('Annualized growth relative to cash (%)');ax.set_ylabel('Annualized excess Sharpe')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.suptitle('Fitted-Q allocation: profit / Sharpe / drawdown preferences')
    fig.tight_layout();fig.savefig(REPORT/'frontier.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['test','prepare','transitions','validation','select','evaluate','report','charts'],required=True)
    p.add_argument('--cap',choices=list(CFG['drawdown_budgets']));a=p.parse_args()
    if a.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_put_straddle_q_frontier','tests.test_research_memory_guard','tests.test_research_atomic_io'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources()
    if a.phase in ['transitions','validation','evaluate']:
        if not a.cap:p.error('--cap required')
        globals()[a.phase](a.cap)
    else:globals()[a.phase]()


if __name__=='__main__':main()
