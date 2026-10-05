"""33 actions: four independent option-type combinations, eight sizes, and flat."""
import copy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scripts.option_flex.contract_engine import PreparedMarket,simulate

COMBOS=['CC','CP','PC','PP']
ACTIONS=[(c,a) for a in [-1.,-.75,-.5,-.25] for c in COMBOS]+[('CC',0.)]+[(c,a) for a in [.25,.5,.75,1.] for c in COMBOS]
FLAT=16


def fit(x,labels,positions,ends):
    target=(labels[:,ACTIONS.index(('CC',1.))].sum(axis=1)-labels[:,ACTIONS.index(('CC',-1.))].sum(axis=1))/2
    tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=63,random_state=2026).fit(x,target)
    leaves=np.unique(tree.apply(x));states=np.searchsorted(leaves,tree.apply(x));nstates=len(leaves);na=len(ACTIONS)
    first=int(positions.min()+1);last=int(ends.max());book=np.zeros((last-first+1,nstates*na))
    offsets=np.arange(labels.shape[2])
    for state in range(nstates):
        rows=np.flatnonzero(states==state);days=positions[rows,None]+1+offsets[None,:]-first
        valid=(days>=0)&(days<len(book))
        for action in range(na):np.add.at(book[:,state*na+action],days[valid],labels[rows,action,:][valid])
    mean=book.mean(axis=0);covariance=np.cov(book,rowvar=False,ddof=1)
    rng=np.random.default_rng(2026);probabilities=np.full((nstates,na),1/na);best=np.full(nstates,FLAT,dtype=int);best_score=0.
    for _ in range(50):
        u=rng.random((192,nstates));choices=(u[:,:,None]>probabilities.cumsum(axis=1)[None,:,:]).sum(axis=2).clip(max=na-1)
        choices[0]=best;choices[1:na+1]=np.arange(na)[:,None]
        columns=choices+np.arange(nstates)[None,:]*na;m=mean[columns].sum(axis=1)
        var=covariance[columns[:,:,None],columns[:,None,:]].sum(axis=(1,2))
        sharpe=np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(var,0)),out=np.zeros_like(m),where=var>1e-18)
        winner=sharpe.argmax()
        if sharpe[winner]>best_score:best,best_score=choices[winner].copy(),float(sharpe[winner])
        elite=choices[np.argsort(sharpe)[-19:]]
        frequency=np.stack([(elite==a).mean(axis=0) for a in range(na)],axis=1)
        probabilities=.97*(.3*probabilities+.7*frequency)+.03/na
    return (tree,leaves,best),dict(training_shadow_daily_sharpe=best_score)


def predict(model,x):
    tree,leaves,policy=model
    return policy[np.searchsorted(leaves,tree.apply(x))]


def gate(codes,panel):
    result=codes.copy();s=panel.reindex(codes.index)
    for k,(combo,action) in enumerate(ACTIONS):
        if not action or not codes.eq(k).any():continue
        side='long' if action>0 else 'short'
        keep=(s.eligible & s[f'{combo}_available'] & s[f'{combo}_profit_{side}_365d'].gt(0)
              & s[f'{combo}_count_{side}_365d'].ge(63) & s[f'{combo}_profit_{side}_91d'].gt(0)
              & s[f'{combo}_count_{side}_91d'].ge(21))
        result.loc[codes.eq(k)&~keep.fillna(False)]=FLAT
    return result


class Replay:
    def __init__(self,q,s,selections,secids,divs,cfg):
        self.q,self.s,self.selections,self.secids,self.divs,self.cfg=q,s,selections,secids,divs,cfg
        self.prepared=PreparedMarket(q,s)
        self.lookup={c:frame.set_index(['date','secid']).optionid.to_dict() for c,frame in selections.items()}

    def run(self,codes,calendar):
        if not np.isin(codes,np.arange(len(ACTIONS))).all():raise ValueError('Invalid action code')
        sizes=pd.Series([ACTIONS[int(c)][1] for c in codes],index=codes.index)
        selected={}
        for date,k in codes[codes.ne(FLAT)].items():
            combo,_=ACTIONS[int(k)]
            for sid in self.secids:
                oid=self.lookup[combo].get((date,sid))
                if oid is not None:selected[(date,sid)]=oid
        prepared=copy.copy(self.prepared);prepared.selection_ids=selected
        selection_frame=pd.DataFrame(columns=['date','secid','optionid'])
        prepared.sources=(self.q,self.s,selection_frame)
        return simulate(self.q,self.s,sizes,calendar,self.secids,self.cfg,self.divs,selection_frame,prepared)
