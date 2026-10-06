"""Causal held-contract spread states and small pooled management Q tables."""
import numpy as np

ACTIONS = np.array([0., .5, 1.])
VARIANTS = ['carry', 'paper', 'combined']
PAPER_BASE = 257 * 9


class SpreadHistory:
    """Paper-inspired excursion diagnostic; no future extrema or true mean."""
    def __init__(self):
        self.values, self.moves, self.events = [], [], []
        self.active = None

    def update(self, value):
        t=len(self.values)
        prior=np.asarray(self.values[-126:],float)
        finite=prior[np.isfinite(prior)]
        mean=float(finite.mean()) if len(finite)>=63 else np.nan
        sd=float(finite.std(ddof=1)) if len(finite)>=63 else np.nan
        z=(value-mean)/sd if np.isfinite(value+mean+sd) and sd>0 else np.nan
        differences=np.diff(np.asarray(self.values[-127:],float))
        differences=differences[np.isfinite(differences)]
        scale=float(differences.std(ddof=1)) if len(differences)>=20 else np.nan
        change=value-self.values[-1] if self.values else np.nan
        scaled=change/scale if np.isfinite(change+scale) and scale>0 else np.nan
        move=3 if scaled>1 else 2 if scaled>0 else 0 if scaled < -1 else 1
        self.values.append(value)
        self.moves.append(move if np.isfinite(scaled) else None)
        recent=self.moves[-4:]
        path=sum(m*4**i for i,m in enumerate(reversed(recent))) if len(recent)==4 and all(m is not None for m in recent) else 256
        age=0
        if self.active is not None:
            start, equilibrium, sign=self.active
            age=t-start
            crossed=np.isfinite(value) and sign*(value-equilibrium)<=0
            if crossed or age>=63:
                self.events.append((start,t,bool(crossed),age)); self.active=None
        if self.active is None and np.isfinite(z) and abs(z)>=2:
            self.active=(t,mean,np.sign(z))
        self.events=[e for e in self.events if e[0]>=t-252]
        completed=[e[3] for e in self.events if e[2]]
        count=len(self.events)+int(self.active is not None)
        failed=(sum(not e[2] for e in self.events)+int(self.active is not None))/count if count else np.nan
        duration=float(np.mean(completed)) if completed else np.nan
        reliable=np.isfinite(duration) and failed<.5 and count>=3
        reversion=(0 if duration<10 else 1) if reliable else 2
        zbin=0 if z < -1 else 2 if z>1 else 1
        return dict(paper_base=(path*3+zbin)*3+reversion,path_code=path,
                    reversion_days=duration,failed_or_censored_fraction=failed,
                    excursion_count=count,active_excursion_age=age,spread_z=z)


def dimensions(variant):
    return {'carry':12,'paper':PAPER_BASE*3,'combined':PAPER_BASE*12}[variant]*2


def encode(states, fractions, variant):
    s=np.asarray(states,int)
    paper,carry,age=s[:,0],s[:,1],s[:,2]
    base=carry*3+age if variant=='carry' else paper*3+age if variant=='paper' else (paper*4+carry)*3+age
    return base*2+(np.asarray(fractions)==.5).astype(int)


def backoff(state, variant):
    state=np.asarray(state,int)
    fraction=state%2; base=state//2; age=base%3; base=base//3
    if variant=='carry':return state
    carry=base%4 if variant=='combined' else np.zeros_like(base)
    paper=base//4 if variant=='combined' else base
    path=paper//9; remainder=paper%9
    coarse=(np.where(path==256,4,path%4)*9+remainder)
    if variant=='combined':coarse=coarse*4+carry
    return (coarse*3+age)*2+fraction


def pooled(states, targets, weights, variant, prior=12.):
    count=dimensions(variant)
    coarse=backoff(states,variant)
    coarse_count=int(backoff(np.arange(count),variant).max())+1
    mass=np.bincount(states,weights=weights,minlength=count)
    sums=np.bincount(states,weights=weights*targets,minlength=count)
    cmass=np.bincount(coarse,weights=weights,minlength=coarse_count)
    csums=np.bincount(coarse,weights=weights*targets,minlength=coarse_count)
    means=np.divide(csums,cmass,out=np.zeros_like(csums),where=cmass>0)
    group=backoff(np.arange(count),variant)
    return (sums+prior*means[group])/(mass+prior),cmass[group]


def fit_q(data, variant, iterations=24, discount=.98):
    states=encode(data['state'],data['fraction'],variant)
    following=encode(data['next_state'],data['next_fraction'],variant)
    action=data['action']; rewards=data['reward']; terminal=data['terminal']
    # Equal cohort mass avoids treating the many transitions in a long trade
    # as many independent examples. Fine states shrink toward coarse states.
    ids,counts=np.unique(data['cohort'],return_counts=True)
    weights=1/counts[np.searchsorted(ids,data['cohort'])]
    q=np.zeros((dimensions(variant),3)); support=np.zeros_like(q)
    target=rewards.copy()
    for _ in range(iterations):
        for k,a in enumerate(ACTIONS):
            take=action==a
            if take.any():q[:,k],support[:,k]=pooled(states[take],target[take],weights[take],variant)
        values=q[following].copy()
        values[support[following]<=0]=-np.inf
        values[(data['next_fraction']==.5),2]=-np.inf
        continuation=values.max(axis=1)
        continuation[terminal|~np.isfinite(continuation)]=0.
        target=rewards+data['discount']*continuation
        assert np.isfinite(target).all()
    return dict(q=q,support=support,variant=variant,iterations=iterations,
                gamma=discount,prior_cohort_mass=12.,reward='actual training PnL per initial vega')


def choose(model, states, fractions):
    index=encode(states,fractions,model['variant'])
    values=model['q'][index].copy()
    values[model['support'][index]<=0]=-np.inf
    values[ACTIONS[None,:]>np.asarray(fractions)[:,None]+1e-12]=-np.inf
    choices=2-np.argmax(values[:,::-1],axis=1)
    result=ACTIONS[choices]
    result[~np.isfinite(values).any(axis=1)]=np.asarray(fractions)[~np.isfinite(values).any(axis=1)]
    return result


def reward_step(ep,start,end,current,requested):
    """Next-close fill; failed trim/close keeps held PnL and continuation."""
    fill=start+1
    available=bool(ep['can_exit'][fill])
    if requested==.5 and not ep['can_trim'][fill]:available=False
    retained=requested if available else current
    reward=0.
    for j in range(fill,end+1):
        previous=current if j==fill else retained
        turnover=(np.abs(retained*ep['hedge'][j]-previous*ep['hprev'][j])*ep['spot'][j]).sum()*.0001
        close=(previous-retained)*ep['exit_cost_training'][j]
        reward+=previous*ep['gross_training'][j]-turnover-close
    return reward/ep['entry_vega'],retained


def transitions(ep, cohort_id):
    n=len(ep['dates']); age=np.arange(n)
    decisions=np.flatnonzero(ep['state_known']&(age>=5)&(age%5==0)&(age<n-1))
    result=[]
    for k,start in enumerate(decisions):
        end=int(decisions[k+1]) if k+1<len(decisions) else n-1
        for current in [1.,.5]:
            for a in ACTIONS[ACTIONS<=current]:
                reward,retained=reward_step(ep,int(start),end,current,float(a))
                result.append((cohort_id,ep['dates'][start].value,ep['dates'][end].value,
                    ep['states'][start],current,a,reward,ep['states'][end],retained,
                    retained==0 or k+1==len(decisions),.98**((end-start)/5)))
    return result


def pack(records):
    keys=['cohort','start','end','state','fraction','action','reward','next_state','next_fraction','terminal','discount']
    return {k:np.asarray([r[i] for r in records],dtype=(np.int64 if k in ['cohort','start','end','state','next_state'] else bool if k=='terminal' else float)) for i,k in enumerate(keys)}


def causal_prefix(data,cutoff,months=48):
    import pandas as pd
    cutoff=pd.Timestamp(cutoff)
    take=(data['start']>=(cutoff-pd.DateOffset(months=months)).value)&(data['end']<cutoff.value)
    assert np.all(data['start'][take]<data['end'][take])
    return {k:v[take] for k,v in data.items()}
