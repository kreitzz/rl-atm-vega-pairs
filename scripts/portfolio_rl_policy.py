"""Completed-month state tree + cross-entropy direct Sharpe policy search."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

LADDER=np.array([0.,.25,.5,.75,1.])
FEATURES=['median_iv','mean_absolute_value','rv_iv_ratio','value_dispersion','earnings_share','relative_width','mean_dte','memory_6m','memory_12m']


def completed_mask(states,cutoff):
    return (states.end<cutoff)&(states.date<cutoff)&(states.date>=cutoff-pd.DateOffset(years=5))


def state_features(q):
    if q.empty:return dict.fromkeys(FEATURES[:-2],np.nan)
    return dict(median_iv=q.bs_iv.median(),mean_absolute_value=q.score.abs().mean(),
                rv_iv_ratio=(np.exp(q.log_rv21)/(q.bs_iv**2)).median(),value_dispersion=q.score.std(),
                earnings_share=q.event.mean(),relative_width=((q.ask-q.bid)/((q.ask+q.bid)/2)).median(),mean_dte=np.exp(q.log_dte).mean())


def fit_predict(train,test,paths):
    """Paths index fully completed, independent monthly portfolio episodes."""
    x=train[FEATURES].to_numpy(float);xt=test[FEATURES].to_numpy(float)
    keep=np.isfinite(x).any(axis=0);x=x[:,keep];xt=xt[:,keep]
    imp=SimpleImputer(strategy='median',add_indicator=True);x=imp.fit_transform(x);xt=imp.transform(xt)
    scaler=StandardScaler();x=scaler.fit_transform(x);xt=scaler.transform(xt)
    target=np.array([paths[d][1][:, -1].sum() for d in train.date])
    tree=DecisionTreeRegressor(max_leaf_nodes=3,min_samples_leaf=6,random_state=2026).fit(x,target)
    leaves=np.unique(tree.apply(x));states=np.searchsorted(leaves,tree.apply(x));ns=len(leaves);na=len(LADDER)
    calendar=pd.DatetimeIndex(sorted({day for d in train.date for day in paths[d][0]}))
    book=np.zeros((len(calendar),ns*na))
    for d,state in zip(train.date,states):
        dates,rewards=paths[d];assert dates.max()<test.signal_date.iloc[0]
        positions=calendar.get_indexer(dates)
        for action in range(na):np.add.at(book[:,state*na+action],positions,rewards[:,action])
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    rng=np.random.default_rng(2026);prob=np.full((ns,na),1/na);best=np.zeros(ns,dtype=int);best_score=0.
    for _ in range(40):
        u=rng.random((192,ns));choices=(u[:,:,None]>prob.cumsum(axis=1)[None,:,:]).sum(axis=2).clip(max=na-1)
        choices[0]=best;choices[1:na+1]=np.arange(na)[:,None]
        cols=choices+np.arange(ns)[None,:]*na
        mu=mean[cols].sum(axis=1);var=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
        score=np.sqrt(252)*np.divide(mu,np.sqrt(np.maximum(var,0)),out=np.zeros_like(mu),where=var>1e-18)
        winner=score.argmax()
        if score[winner]>best_score:best_score=float(score[winner]);best=choices[winner].copy()
        elite=choices[np.argsort(score)[-19:]]
        frequency=np.stack([(elite==a).mean(axis=0) for a in range(na)],axis=1)
        prob=.97*(.3*prob+.7*frequency)+.03/na
    action=int(best[np.searchsorted(leaves,tree.apply(xt))[0]])
    center=target.mean();scale=max(target.std(),1e-8)
    nn=MLPRegressor(hidden_layer_sizes=(16,),activation='tanh',solver='lbfgs',alpha=1.,max_iter=2000,random_state=2026)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);nn.fit(x,(target-center)/scale)
    prediction=float(nn.predict(xt)[0]*scale+center)
    return LADDER[action],float(prediction>0),dict(training_shadow_sharpe=best_score,leaves=ns,policy=best.tolist(),
        nn_prediction=prediction,nn_converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught))
