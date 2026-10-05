"""Bounded Sharpe policy search over a training-only contract shortlist."""
import copy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scripts.curve_trading.config import ACTIONS, COMBOS, LADDER
from scripts.option_flex.contract_engine import PreparedMarket, simulate

ATM=COMBOS.index((30,'C','atm','atm'))

def memory_gates(dates, ends, rewards, entered):
    """Strictly prior completed outcomes, including ties and varying maturities."""
    result=np.ones((len(dates),2),bool)
    times=pd.DatetimeIndex(dates).asi8;finish=pd.DatetimeIndex(ends).asi8
    for side in range(2):
        good=entered[:,side]&np.isfinite(rewards[:,side])
        order=np.argsort(finish[good],kind='stable');e=finish[good][order];r=rewards[good,side][order]
        cumulative=np.r_[0.,np.cumsum(r)]
        right=np.searchsorted(e,times,side='left')
        for days,minimum in [(91,21),(365,63)]:
            left=np.searchsorted(e,times-pd.Timedelta(days=days).value,side='left')
            count=right-left;total=cumulative[right]-cumulative[left]
            result[:,side] &= (count>=minimum)&(total>0)
    return result

def fit(x,labels,positions,ends,codes):
    """Same eight-leaf direct policy search as reference; variable action count."""
    target=(labels[:,codes.index(1+ATM*8+7)].sum(axis=1)-labels[:,codes.index(1+ATM*8)].sum(axis=1))/2
    tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=63,random_state=2026).fit(x,target)
    leaves=np.unique(tree.apply(x));states=np.searchsorted(leaves,tree.apply(x));ns=len(leaves);na=len(codes)
    first=int(positions.min()+1);last=int(ends.max());book=np.zeros((last-first+1,ns*na))
    offsets=np.arange(labels.shape[2])
    for state in range(ns):
        rows=np.flatnonzero(states==state);days=positions[rows,None]+1+offsets[None,:]-first
        valid=(days>=0)&(days<len(book))
        for action in range(na):np.add.at(book[:,state*na+action],days[valid],labels[rows,action,:][valid])
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
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

def predict(model,x):
    tree,leaves,policy=model
    return policy[np.searchsorted(leaves,tree.apply(x))]

def gate(codes,index,panel,gates,available):
    out=np.asarray(codes).copy();rows=panel.index.get_indexer(index);assert (rows>=0).all()
    for code in np.unique(out):
        if not code:continue
        combo,size=ACTIONS[int(code)];side=int(size>0)
        okay=panel.eligible.to_numpy()[rows]&available[combo,rows]&gates[combo,rows,side]
        out[(out==code)&~okay]=0
    return pd.Series(out,index=index,dtype=int)

class Replay:
    def __init__(self,q,s,selections,secids,divs,cfg):
        self.q,self.s,self.secids,self.divs,self.cfg=q,s,secids,divs,cfg
        self.prepared=PreparedMarket(q,s)
        self.lookup={(r.date,r.secid,int(r.tenor),r.cp_flag,str(r.bucket)):r.optionid for r in selections.itertuples()}

    def run(self,codes,calendar):
        assert np.isin(codes,np.arange(len(ACTIONS))).all()
        sizes=pd.Series([ACTIONS[int(c)][1] if c else 0. for c in codes],index=codes.index,dtype=float)
        selected={}
        for date,code in codes[codes.ne(0)].items():
            combo,_=ACTIONS[int(code)];tenor,cp,ba,bb=COMBOS[combo]
            for sid,bucket in zip(self.secids,[ba,bb]):
                oid=self.lookup.get((date,sid,tenor,cp,str(bucket)))
                if oid is not None:selected[(date,sid)]=oid
        prepared=copy.copy(self.prepared);prepared.selection_ids=selected
        selection=pd.DataFrame(columns=['date','secid','optionid']);prepared.sources=(self.q,self.s,selection)
        return simulate(self.q,self.s,sizes,calendar,self.secids,self.cfg,self.divs,selection,prepared)
