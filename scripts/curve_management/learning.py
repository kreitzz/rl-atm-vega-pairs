"""Training-only maturity shortlist, entry Sharpe search and optimal stopping."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scripts.curve_management.config import COMBOS,ACTIONS,ATM,SHORTLIST,MAX_EXIT_COHORTS_PER_COMBO

def gate(codes,index,panel,gates,available):
    out=np.asarray(codes).copy();rows=panel.index.get_indexer(index);assert (rows>=0).all()
    for code in np.unique(out):
        if not code:continue
        combo,size=ACTIONS[int(code)];side=int(size>0)
        okay=panel.eligible.to_numpy(bool)[rows]&available[combo,rows]&gates[combo,rows,side]
        out[(out==code)&~okay]=0
    return pd.Series(out,index=index,dtype=int,name='code')

def fit_entry(x,labels,positions,ends,codes):
    # Multi-output partitions retain each contract combination's payoff target.
    target=np.column_stack([(labels[:,1+j*8+7].sum(axis=1)-labels[:,1+j*8].sum(axis=1))/2
                            for j in range((len(codes)-1)//8)])
    tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=63,random_state=2026).fit(x,target)
    leaves=np.unique(tree.apply(x));state=np.searchsorted(leaves,tree.apply(x));ns=len(leaves);na=len(codes)
    first=int(positions.min()+1);last=int(ends.max());book=np.zeros((last-first+1,ns*na))
    offsets=np.arange(labels.shape[2])
    for s in range(ns):
        rows=np.flatnonzero(state==s);days=positions[rows,None]+1+offsets[None,:]-first
        valid=(days>=0)&(days<len(book))
        for action in range(na):
            values=labels[rows,action,:];nonzero=valid&(values!=0)
            np.add.at(book[:,s*na+action],days[nonzero],values[nonzero])
    mean=book.mean(axis=0);active=np.flatnonzero(np.any(book!=0,axis=0));cov=np.zeros((ns*na,ns*na))
    if len(active):cov[np.ix_(active,active)]=np.atleast_2d(np.cov(book[:,active],rowvar=False,ddof=1))
    rng=np.random.default_rng(2026);prob=np.full((ns,na),1/na);best=np.zeros(ns,int);best_score=0.
    population=max(192,na+1)
    for _ in range(50):
        u=rng.random((population,ns));choice=(u[:,:,None]>prob.cumsum(axis=1)[None,:,:]).sum(axis=2).clip(max=na-1)
        choice[0]=best;choice[1:na+1]=np.arange(na)[:,None]
        cols=choice+np.arange(ns)[None,:]*na;m=mean[cols].sum(axis=1)
        v=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
        score=np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(v,0)),out=np.zeros_like(m),where=v>1e-18)
        winner=score.argmax()
        if score[winner]>best_score:best,best_score=choice[winner].copy(),float(score[winner])
        elite=choice[np.argsort(score)[-max(19,population//10):]]
        frequency=np.stack([(elite==a).mean(axis=0) for a in range(na)],axis=1)
        prob=.97*(.3*prob+.7*frequency)+.03/na
    return (tree,leaves,np.array(codes)[best]),dict(training_shadow_daily_sharpe=best_score)

def predict_entry(model,x):
    tree,leaves,policy=model
    return policy[np.searchsorted(leaves,tree.apply(x))]

def shortlist(year,allowed,panel,labels,ends,gates,cal):
    cutoff=pd.Timestamp(year-1,1,1);start=cutoff-pd.DateOffset(years=5)
    positions=cal.get_indexer(panel.index);first=cal.searchsorted(start);last=cal.searchsorted(cutoff)-1
    records=[]
    for combo in allowed:
        daily=labels[combo][:,[0,-1]]
        use=(panel.index>=start)&(cal[ends[combo]]<cutoff)&np.isfinite(daily).all(axis=(1,2))&panel.eligible.to_numpy(bool)
        scores=[]
        for side in range(2):
            rows=np.flatnonzero(use&gates[combo,:,side]&np.any(daily[:,side]!=0,axis=1))
            book=np.zeros(max(1,last-first+1));days=positions[rows,None]+1+np.arange(daily.shape[-1])-first
            valid=(days>=0)&(days<len(book));np.add.at(book,days[valid],daily[rows,side,:][valid])
            sd=book.std(ddof=1);scores.append(float(np.sqrt(252)*book.mean()/sd) if len(rows)>=21 and sd>1e-12 else None)
        values=[s for s in scores if s is not None]
        records.append(dict(year=year,combo=combo,score=max(values) if values else None,
                            ranking_cutoff=str(cutoff.date()),ranking_start=str(start.date())))
    ranked=sorted([r for r in records if r['score'] is not None],key=lambda r:(-r['score'],r['combo']))
    chosen=[ATM]+[r['combo'] for r in ranked if r['combo']!=ATM][:SHORTLIST-1]
    for r in records:r['selected']=r['combo'] in chosen
    return chosen,records

def exit_stops(model,features,valid,positions,ends):
    """Decision at close t fills t+1. No decision is made on future rows."""
    n,w=valid.shape;fallback=ends-positions-1
    if model is None:return fallback.copy()
    tree,leaves,close=model
    wants=np.zeros((n,w),bool)
    if valid.any():wants[valid]=close[np.searchsorted(leaves,tree.apply(features[valid]))]
    steps=np.arange(w)
    return np.where(wants,steps[None,:]+1,w).min(axis=1).clip(max=fallback)

def apply_stops(daily,cost,stops,positions,ends,calendar,rates):
    """Mark-to-market exit costs and realized excess cash carry to original end."""
    result=np.array(daily,copy=True,dtype=np.float32);n,w=result.shape
    rows=np.arange(n);result[rows,stops]-=cost[rows,stops]
    profit=np.cumsum(result,axis=1)[rows,stops].astype(float)
    interest,growth,prior,now=carry_arrays(positions,calendar,rates,w)
    carry=profit[:,None]*prior/growth[rows,stops,None]*interest
    after=np.arange(w)[None,:]>stops[:,None]
    result[after]=np.where(now<=ends[:,None],carry,0.)[after]
    return result

def carry_arrays(positions,calendar,rates,w):
    now=np.minimum(positions[:,None]+1+np.arange(w)[None,:],len(calendar)-1)
    prev=np.maximum(now-1,0)
    elapsed=(calendar.values[now]-calendar.values[prev])/np.timedelta64(1,'D')/365
    interest=rates[prev]*elapsed
    growth=np.cumprod(1+interest,axis=1);prior=np.column_stack([np.ones(len(positions)),growth[:,:-1]])
    return interest,growth,prior,now

def fit_exit(data,rows,combos,cutoff):
    """Bounded training cohorts, an eight-leaf held-state partition, all 2^L rules.

    Full expiry outcomes must precede cutoff even if a candidate exits earlier.
    Cohort subsampling is deterministic and independent of rewards.
    """
    pieces=[];positions=[];ends=[];samples=[];last_outcome=None
    for combo in combos:
        valid_rows=np.flatnonzero(rows & data.panel.eligible.to_numpy(bool) & data.entered[combo]
                                 & np.isfinite(data.labels[combo]).all(axis=(1,2)))
        if len(valid_rows)>MAX_EXIT_COHORTS_PER_COMBO:
            valid_rows=valid_rows[np.linspace(0,len(valid_rows)-1,MAX_EXIT_COHORTS_PER_COMBO,dtype=int)]
        if not len(valid_rows):continue
        last=data.cal[data.ends[combo,valid_rows]].max()
        assert last<cutoff
        last_outcome=last if last_outcome is None else max(last_outcome,last)
        held,valid=data.held_features(combo)
        costs=data.stop_cost(combo);daily=data.labels[combo]
        for side,action in [(0,0),(1,7)]:
            x=data.direction_features(held[valid_rows],side)
            pieces.append((x,valid[valid_rows],daily[valid_rows,action],costs[valid_rows,action]))
            positions.extend(data.positions[valid_rows]);ends.extend(data.ends[combo,valid_rows]);samples.extend(valid_rows)
    if not pieces:return None,dict(exit_training_cohorts=0)
    x=np.concatenate([p[0] for p in pieces]);valid=np.concatenate([p[1] for p in pieces])
    daily=np.concatenate([p[2] for p in pieces]);cost=np.concatenate([p[3] for p in pieces])
    positions=np.asarray(positions);ends=np.asarray(ends)
    remaining=daily.sum(axis=1)[:,None]-np.cumsum(daily,axis=1)+np.roll(cost,-1,axis=1)
    if valid.sum()<252:return None,dict(exit_training_cohorts=len(daily),exit_training_rows=int(valid.sum()))
    tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=63,random_state=2026).fit(x[valid],remaining[valid])
    leaves=np.unique(tree.apply(x[valid]));assignment=np.zeros(valid.shape,int)
    assignment[valid]=np.searchsorted(leaves,tree.apply(x[valid]))
    first=int(positions.min()+1);last=int(ends.max());indices=positions[:,None]+1+np.arange(daily.shape[1])-first
    admissible=(indices>=0)&(indices<=last-first)
    best_score=-np.inf;best=None
    n,w=daily.shape;cum=np.cumsum(daily,axis=1)
    interest,growth,prior,now=carry_arrays(positions,data.cal,data.rates,w)
    for first_pattern in range(0,1<<len(leaves),32):
        patterns=np.arange(first_pattern,min(first_pattern+32,1<<len(leaves)))
        close=((patterns[:,None]>>np.arange(len(leaves))[None,:])&1).astype(bool)
        wants=valid[:,:,None]&close[:,assignment].transpose(1,2,0)
        stop=np.where(wants,np.arange(w)[None,:,None]+1,w).min(axis=1)
        stop=np.minimum(stop,(ends-positions-1)[:,None]);rows=np.arange(n)[:,None]
        penalty=cost[rows,stop];profit=cum[rows,stop]-penalty
        later=np.arange(w)[None,:,None]>stop[:,None,:]
        at=np.arange(w)[None,:,None]==stop[:,None,:]
        carry=profit[:,None,:]*prior[:,:,None]/growth[rows,stop][:,None,:]*interest[:,:,None]
        payoffs=np.where(later,np.where((now<=ends[:,None])[:,:,None],carry,0.),daily[:,:,None]-at*penalty[:,None,:])
        offsets=np.arange(len(patterns));key=indices[:,:,None]*len(patterns)+offsets
        book=np.bincount(key[admissible].ravel(),weights=payoffs[admissible].ravel(),minlength=(last-first+1)*len(patterns)).reshape(-1,len(patterns))
        sd=book.std(axis=0,ddof=1);score=np.divide(np.sqrt(252)*book.mean(axis=0),sd,out=np.full(len(patterns),-np.inf),where=sd>1e-12)
        winner=score.argmax()
        if score[winner]>best_score:best_score=float(score[winner]);best=close[winner].copy()
    if best is None:best=np.zeros(len(leaves),bool)
    return (tree,leaves,best),dict(exit_training_cohorts=len(daily),exit_training_rows=int(valid.sum()),
        exit_last_outcome=str(last_outcome.date()),exit_shadow_sharpe=float(best_score),exit_close_leaves=int(best.sum()))
