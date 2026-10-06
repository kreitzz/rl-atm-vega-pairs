"""Same eight-leaf Sharpe search, with an optional single-leg action menu."""
import copy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scripts.option_flex.single_leg_engine import WEIGHTS,simulate
from scripts.option_flex.contract_engine import PreparedMarket
from scripts.curve_trading.policy import memory_gates

ADDED=['vrp_level_a','vrp_level_b','vrp_z_a','vrp_z_b','vrp_z_a_available','vrp_z_b_available']
MAX_SINGLE_NET_VEGA=900.


def add_vrp(panel,normalizers=None):
    result=panel.copy()
    for suffix,side in [('amd','a'),('nvda','b')]:
        iv=panel['iv_'+suffix]/100;rv=panel['trailing_rv_'+suffix]/100
        premium=(iv**2-rv**2).where((iv>0)&(rv>=0))
        if normalizers is None:
            history=premium.shift(1).rolling(252,min_periods=126)
            mean=history.mean();std=history.std()
        else:
            norm=normalizers[0 if side=='a' else 1].reindex(panel.index)
            mean=norm['mean'];std=norm['std']
        z=(premium-mean)/std.where(std>1e-8)
        result['vrp_level_'+side]=premium.fillna(0.)
        result['vrp_z_'+side]=z.replace([np.inf,-np.inf],np.nan).fillna(0.)
        result['vrp_z_'+side+'_available']=z.notna().astype(float)
    pd.testing.assert_frame_equal(result[panel.columns],panel)
    return result


def single_conditions(panel,rule='historical'):
    if rule=='historical':
        a=panel.vrp_z_a;b=panel.vrp_z_b
        ready=panel.vrp_z_a_available.eq(1)&panel.vrp_z_b_available.eq(1)
    elif rule=='absolute':
        a=panel.vrp_level_a;b=panel.vrp_level_b
        ready=panel.iv_amd.gt(0)&panel.iv_nvda.gt(0)&panel.trailing_rv_amd.ge(0)&panel.trailing_rv_nvda.ge(0)
    else:raise ValueError(rule)
    return (ready&(a<0)&(b<0)).to_numpy(),(ready&(a>0)&(b>0)).to_numpy()


def single_labels(unit,cfg):
    """All eight signed sizes, exact cash funding, own hedge and own expiry."""
    ids=unit.ids(30,'C','atm');n=len(ids);w=unit.width
    sizes=np.array([a for a,b in WEIGHTS[9:17]])
    v=unit.entry_vega[ids]
    exists=(ids>=0)&unit.entered[ids]
    with np.errstate(divide='ignore',invalid='ignore'):
        quantity=np.floor(cfg.entry_vega_per_leg*abs(sizes)[None,:]/v[:,None])
    entered=exists[:,None]&(quantity>=1)
    quantity=np.where(entered,quantity,0.)
    vegas=quantity*v[:,None]
    live=entered.any(axis=1)
    ends=np.where(live,unit.ends[ids],np.minimum(unit.positions+1,len(unit.calendar)-1))
    valid=~live|(unit.resolved[ids]&((unit.stock_bad_prefix[ends+1]-unit.stock_bad_prefix[unit.positions])==0))
    cash=np.full((n,8),cfg.capital,float);benchmark=np.full(n,cfg.capital,float)
    previous=cash.copy();previous_benchmark=benchmark.copy();daily=np.zeros((n,8,w))
    sign=np.sign(sizes)[None,:]
    max_steps=int(np.max(np.where(live,ends-unit.positions,0),initial=0))
    for step in range(1,min(w,max_steps)+1):
        now=np.minimum(unit.positions+step,len(unit.calendar)-1)
        prior=np.minimum(unit.positions+step-1,len(unit.calendar)-1)
        dt=(unit.calendar[now]-unit.calendar[prior]).days.to_numpy()/365
        rate=unit.rates[prior]
        cash+=cash*(rate[:,None]+np.where(cash<0,cfg.borrowing_spread,0))*dt[:,None]
        benchmark*=1+rate*dt
        long=np.nan_to_num(unit.flow_long[ids,step-1]);short=np.nan_to_num(unit.flow_short[ids,step-1])
        cash+=quantity*np.where(sign>0,long[:,None],short[:,None])
        marked=quantity*sign*np.nan_to_num(unit.mark[ids,step-1])[:,None]
        nav=cash+marked
        daily[:,:,step-1]=np.where((live&(now<=ends))[:,None],
            (nav-previous-(benchmark-previous_benchmark)[:,None])/cfg.entry_vega_per_leg,0.)
        previous=nav;previous_benchmark=benchmark.copy()
    daily[~valid]=np.nan
    reward=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
        out=np.full((n,2),np.nan),where=vegas[:,[0,-1]]>0)
    gates=memory_gates(unit.dates,unit.calendar[ends],reward,vegas[:,[0,-1]]>0)
    return daily,ends,vegas,gates


def action_masks(panel,pair_gates,pair_available,solo,rule):
    n=len(panel);mask=np.zeros((n,25),bool);mask[:,0]=True
    for code,(a,b) in enumerate(WEIGHTS[1:9],1):
        mask[:,code]=pair_available&pair_gates[:,int(a>0)]
    cheap,rich=single_conditions(panel,rule)
    for first,gates in [(9,solo[0]),(17,solo[1])]:
        for offset in range(8):
            side=int(offset>=4)
            mask[:,first+offset]=gates[:,side]&(cheap if side else rich)
    mask[:,1:] &= panel.eligible.to_numpy(bool)[:,None]
    return mask


def fit(x,labels,positions,ends,codes):
    # Both candidates use precisely the same relative-value tree partition.
    target=(labels[:,8].sum(axis=1)-labels[:,1].sum(axis=1))/2
    tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=63,random_state=2026).fit(x,target)
    leaves=np.unique(tree.apply(x));states=np.searchsorted(leaves,tree.apply(x));ns=len(leaves);na=len(codes)
    first=int(positions.min()+1);last=int(ends.max());book=np.zeros((last-first+1,ns*na))
    offsets=np.arange(labels.shape[2])
    for state in range(ns):
        rows=np.flatnonzero(states==state);days=positions[rows,None]+1+offsets[None,:]-first
        valid=(days>=0)&(days<len(book))
        for j,action in enumerate(codes):np.add.at(book[:,state*na+j],days[valid],labels[rows,action,:][valid])
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    rng=np.random.default_rng(2026);prob=np.full((ns,na),1/na);best=np.zeros(ns,int);best_score=0.
    for _ in range(50):
        u=rng.random((192,ns));choice=(u[:,:,None]>prob.cumsum(axis=1)[None,:,:]).sum(axis=2).clip(max=na-1)
        choice[0]=best;choice[1:na+1]=np.arange(na)[:,None]
        cols=choice+np.arange(ns)[None,:]*na;m=mean[cols].sum(axis=1)
        variance=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
        score=np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(variance,0)),out=np.zeros_like(m),where=variance>1e-18)
        winner=score.argmax()
        if score[winner]>best_score:best,best_score=choice[winner].copy(),float(score[winner])
        elite=choice[np.argsort(score)[-19:]]
        frequency=np.stack([(elite==a).mean(axis=0) for a in range(na)],axis=1)
        prob=.97*(.3*prob+.7*frequency)+.03/na
    return (tree,leaves,np.array(codes)[best]),dict(training_shadow_daily_sharpe=best_score,
        vrp_value_splits=sum(str(x.columns[int(f)]).startswith('vrp_') and not str(x.columns[int(f)]).endswith('_available') for f in tree.tree_.feature if f>=0))


def predict(model,x):
    tree,leaves,policy=model
    return policy[np.searchsorted(leaves,tree.apply(x))]


def gate(codes,index,panel,masks):
    rows=panel.index.get_indexer(index);assert (rows>=0).all()
    out=np.asarray(codes,dtype=int).copy();out[~masks[rows,out]]=0
    return pd.Series(out,index=index,dtype=int)


class Replay:
    def __init__(self,q,s,selections,secids,divs,cfg):
        self.q,self.s,self.secids,self.divs,self.cfg=q,s,secids,divs,cfg
        self.prepared=PreparedMarket(q,s)
        self.lookup={(r.date,r.secid):r.optionid for r in selections.itertuples()
                     if r.tenor==30 and r.cp_flag=='C' and str(r.bucket)=='atm'}

    def run(self,codes,calendar):
        selected={(date,sid):self.lookup[date,sid] for date in codes[codes.ne(0)].index
                  for sid in self.secids if (date,sid) in self.lookup}
        prepared=copy.copy(self.prepared);prepared.selection_ids=selected
        selection=pd.DataFrame(columns=['date','secid','optionid']);prepared.sources=(self.q,self.s,selection)
        return simulate(self.q,self.s,codes,calendar,self.secids,self.cfg,self.divs,selection,prepared,MAX_SINGLE_NET_VEGA)
