"""Small fitted-Q policies and causal completed-close portfolio decisions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

SIGNAL_FEATURES=['iv','gap','rank','vrp','memory','vix','demand']
INVENTORY_FEATURES=['stress','reserve','pair_stress','pair_cohorts','name_overlap',
                    'remaining_dte','release_7d','drawdown','recent_vol','iv_balance','gross_vega']
FEATURES=SIGNAL_FEATURES+INVENTORY_FEATURES
# Action2 preserves the original full-size proposal rule. Action3 commits
# full-size capital only to candidate pairs absent from the current book.
ACTION_NAMES={0:'skip',1:'half',2:'full',3:'unheld_pairs'}
SPECS={
    'q_capacity':dict(gamma=.99,penalty=0.,actions=[0,1,2],inventory=True,conservative=False),
    'q_risk':dict(gamma=.99,penalty=.5,actions=[0,1,2],inventory=True,conservative=False),
    'q_conservative':dict(gamma=.99,penalty=.5,actions=[0,1,2],inventory=True,conservative=True),
    'q_portfolio':dict(gamma=.99,penalty=.5,actions=[0,1,2,3],inventory=True,conservative=True),
    'myopic_risk':dict(gamma=0.,penalty=.5,actions=[0,1,2],inventory=True,conservative=False),
    'q_no_inventory':dict(gamma=.99,penalty=.5,actions=[0,1,2],inventory=False,conservative=False),
}


def action_inputs(states,actions):
    return np.column_stack([states,np.eye(4)[np.asarray(actions,int)]])


@dataclass
class QPolicy:
    model: object
    columns: list
    actions: list
    conservative: bool
    audit: dict

    def values(self,states):
        s=np.atleast_2d(states)[:,self.columns]
        x=action_inputs(np.repeat(s,len(self.actions),axis=0),np.tile(self.actions,len(s)))
        return self.model.predict(x).reshape(len(s),len(self.actions))

    def choose(self,state):
        s=np.asarray(state)[self.columns]
        x=action_inputs(np.repeat(s[None,:],len(self.actions),axis=0),self.actions)
        reference=self.actions.index(2)
        if self.conservative:
            trees=np.stack([tree.predict(x) for tree in self.model.estimators_])
            advantage=trees-trees[:,[reference]]
            values=advantage.mean(axis=0)-advantage.std(axis=0,ddof=1)
        else:values=self.model.predict(x)
        # Ties and negligible estimated improvements retain the baseline action.
        best=int(np.argmax(values))
        return self.actions[best] if values[best]>values[reference]+1e-8 else 2


def fit_policy(records,daily,cutoff,spec,rest=lambda:None):
    """Use only fully observed transition endpoints, never future runoff labels.

    A still-open option may contribute known partial transitions. Censoring a
    training window never changes a nonterminal transition into terminal cash.
    """
    cutoff=pd.Timestamp(cutoff)
    use=records.loc[records.end.lt(cutoff)&records.start.lt(cutoff)&records.action.isin(spec['actions'])].copy()
    if len(use)<100 or use.start.nunique()<60:raise ValueError('Insufficient distinct historical decision dates')
    columns=list(range(len(FEATURES))) if spec['inventory'] else list(range(len(SIGNAL_FEATURES)))
    states=np.stack(use.state.map(np.asarray))[:,columns]
    next_states=np.stack(use.next_state.map(np.asarray))[:,columns]
    x=action_inputs(states,use.action.to_numpy())
    past=daily.loc[daily.date.lt(cutoff),'bps'].to_numpy(float)
    scale=max(float(past.std(ddof=1)),.01)
    reward=use.reward.to_numpy()-spec['penalty']*use.squared_reward.to_numpy()/(2*scale)
    terminal=use.terminal.to_numpy(bool)
    discounts=np.power(spec['gamma'],use.sessions.to_numpy())
    y=reward.copy();trace=[]
    counts=use.groupby('start').start.transform('size').to_numpy()
    weights=1/counts;weights/=weights.mean()
    for iteration in range(24 if spec['gamma'] else 1):
        model=ExtraTreesRegressor(n_estimators=24,max_depth=3,min_samples_leaf=30,
                                  n_jobs=1,random_state=2026)
        model.fit(x,y,sample_weight=weights)
        nxt=action_inputs(np.repeat(next_states,len(spec['actions']),axis=0),np.tile(spec['actions'],len(use)))
        continuation=model.predict(nxt).reshape(len(use),len(spec['actions'])).max(axis=1)
        continuation[terminal]=0
        target=reward+discounts*continuation
        trace.append(float(np.mean(np.abs(target-y))))
        if not np.isfinite(target).all():raise ValueError('Nonfinite Bellman targets')
        y=target;rest()
    audit=dict(cutoff=str(cutoff.date()),last_transition_end=str(use.end.max().date()),
        transitions=len(use),distinct_market_dates=int(use.start.nunique()),
        daily_reward_scale_bps=scale,iterations=len(trace),bellman_target_change=trace,
        terminal_transitions=int(terminal.sum()),spec=spec,
        caveat='Repeated behavior trajectories share market history; they are not independent observations. Aggregated state approximates the full book.')
    return QPolicy(model,columns,spec['actions'],spec['conservative'],audit)


class Planner:
    """One decision at signal close; original engine executes next session."""
    def __init__(self,signals,rates,cfg,kind,model=None,active_from='2018-01-01',seed=2026):
        self.groups={pd.Timestamp(d):g for d,g in signals.groupby('date')}
        self.rates=rates;self.cfg=cfg;self.kind=kind;self.model=model
        self.active_from=pd.Timestamp(active_from);self.rng=np.random.default_rng(seed)
        self.actions={};self.levels={};self.events=[];self.daily=[]
        self.previous=None;self.benchmark=cfg.capital;self.peak=1.;self.sticky={}

    def observe(self,date,day,positions,pairs,global_risk):
        elapsed=0 if self.previous is None else (date-self.previous).days/365
        cash_return=0 if self.previous is None else float(self.rates.loc[self.previous])*elapsed
        bps=(day['return_']-cash_return)*10000
        if abs(bps)<1e-8:bps=0.
        self.benchmark*=1+cash_return
        relative=day['nav']/self.benchmark;self.peak=max(self.peak,relative)
        self.daily.append(dict(date=date,bps=bps));self.previous=date
        if date not in self.groups:return
        group=self.groups[date];equity=day['nav'];unit=450*equity/self.cfg.capital
        names={};vegas=[];maturities=[];signed_iv=0.
        for p in positions:
            v=abs(p['quantity'])*100*p.get('last_vega',0.)
            names[p['secid']]=names.get(p['secid'],0.)+v
            vegas.append(v);maturities.append(max(0,(p['expiry']-date).days))
            signed_iv+=np.sign(p['quantity'])*v*p.get('last_iv',0.)
        total=sum(vegas);pair_equity=1e6*equity/self.cfg.capital
        dte=np.average(maturities,weights=vegas) if total else 0.
        release=sum(v for v,d in zip(vegas,maturities) if d<=7)/total if total else 0.
        risk=[pairs.get(p,{}) for p in group.pair]
        overlap=max([max(names.get(int(r.secid_a),0),names.get(int(r.secid_b),0))/unit for r in group.itertuples()],default=0.)
        state=np.array([group.iv_mean.mean(),group.oriented_iv_gap.mean(),group.rank_strength.max(),
            group.oriented_vrp_gap.mean(),group.profit_91d.mean(),group.vix.iloc[0],len(group),
            global_risk.get('stress_loss',0)/(.05*equity),global_risk.get('capital_reserve',0)/(.8*equity),
            max([r.get('stress_loss',0)/(.05*pair_equity) for r in risk],default=0.),
            max([r.get('active_cohorts',0) for r in risk],default=0.),overlap,dte/55,release,
            max(0.,1-relative/self.peak),np.std([r['bps'] for r in self.daily[-20:]]),
            signed_iv/total if total else 0.,global_risk.get('gross_vega',0)/self.cfg.max_gross_vega])
        if not np.isfinite(state).all():raise ValueError('Nonfinite causal portfolio state')
        action=2
        if date>=self.active_from:
            if self.model is not None:action=self.model.choose(state)
            elif self.kind=='half':action=1
            elif self.kind=='diversified':action=3
            elif self.kind=='random':action=int(self.rng.choice(4,p=[.2,.2,.4,.2]))
            elif self.kind=='sticky':
                week=str(date.to_period('W'))
                if week not in self.sticky:self.sticky[week]=int(self.rng.choice(4))
                action=self.sticky[week]
            elif self.kind=='congestion':action=1 if state[9]>.5 else 2
            elif self.kind not in ['full','learned']:raise ValueError(self.kind)
        self.actions[date]=action
        self.events.append(dict(date=date,action=action,state=state.tolist()))
        for r in group.itertuples():
            level={0:0,1:2,2:4,3:4}[action]
            if action==3 and pairs.get(r.pair,{}).get('active_cohorts',0)>0:level=0
            self.levels[(date,r.pair)]=level

    def entry(self,row,execution_context):
        # Actual execution state is deliberately ignored. Membership for the
        # diversification action was frozen at the preceding signal close.
        return self.levels[(pd.Timestamp(row['date']),row['pair'])]


def transitions(planner):
    daily=pd.DataFrame(planner.daily).set_index('date')
    events=planner.events;records=[]
    for j,e in enumerate(events):
        terminal=j+1==len(events)
        end=daily.index[-1] if terminal else events[j+1]['date']
        segment=daily.loc[(daily.index>e['date'])&(daily.index<=end),'bps']
        if segment.empty:continue
        records.append(dict(start=e['date'],end=end,action=e['action'],state=e['state'],
            next_state=[0.]*len(FEATURES) if terminal else events[j+1]['state'],
            reward=float(segment.sum()),squared_reward=float((segment**2).sum()),
            sessions=len(segment),terminal=terminal))
    return pd.DataFrame(records),daily.reset_index()
