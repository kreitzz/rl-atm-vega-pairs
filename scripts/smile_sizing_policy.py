"""Sizing only: fixed pair direction, optional feature flags, pooled daily Sharpe."""
import numpy as np
import pandas as pd
from scripts.curve_trading.config import COMBOS
from scripts.tail_put25_pooled import completed_mask

CORE=['iv_mean','relative_iv_cheapness','rank_strength']
OPTIONAL=['factor_cheapness','risk_reversal_cheapness','right_slope_cheapness',
          'beta','correlation','vix','profit_91d','count_91d','profit_365d','count_365d']
FEATURES=CORE+OPTIONAL+[c+'_available' for c in OPTIONAL]
LADDERS={'floor':[2,3,4],'flex':[0,1,2,3,4]}


def features(panel):
    direction=np.where(panel.wing_iv_rank.le(.2),1.,np.where(panel.wing_iv_rank.gt(.8),-1.,0.))
    out=pd.DataFrame(index=panel.index)
    out['iv_mean']=panel.iv_mean
    out['relative_iv_cheapness']=-direction*panel.wing_iv_gap/panel.iv_mean
    out['rank_strength']=(panel.wing_iv_rank-.5).abs().mul(2).fillna(0.)
    for name,source in [('factor_cheapness','vol_factor_residual_gap'),
                        ('risk_reversal_cheapness','risk_reversal_gap'),
                        ('right_slope_cheapness','right_steepness_gap')]:
        out[name]=-direction*panel[source]
    out['factor_cheapness']/=panel.iv_mean
    for c in OPTIONAL[3:]:out[c]=panel[c]
    for days in [91,365]:out[f'profit_{days}d']/=100.
    out=out.replace([np.inf,-np.inf],np.nan)
    for c in OPTIONAL:
        out[c+'_available']=out[c].notna().astype(float)
        out[c]=out[c].fillna(0.)
    assert np.isfinite(out.loc[panel.observed,FEATURES]).all().all()
    return out


def translate(levels,rule,cp,bucket):
    levels,rule=np.asarray(levels),np.asarray(rule)
    assert levels.shape==rule.shape and np.isin(levels,np.arange(5)).all()
    combo=COMBOS.index((30,cp,bucket,bucket));short=1+combo*8;long=short+7
    assert np.isin(rule,[0,short,long]).all()
    out=np.zeros(len(rule),int)
    for level in range(1,5):
        out[(levels==level)&(rule==short)]=short+4-level
        out[(levels==level)&(rule==long)]=short+3+level
    return out


def directional_labels(daily,rank):
    """Eight actual integer-sized directional paths become five size choices."""
    daily=np.asarray(daily);rank=np.asarray(rank)
    result=np.zeros((len(rank),5,daily.shape[-1]),dtype=daily.dtype)
    for level in range(1,5):
        short,long=rank>.8,rank<=.2
        result[short,level]=daily[short,4-level]
        result[long,level]=daily[long,3+level]
    return result


def training_mask(meta,cutoff):
    return completed_mask(meta.date,meta.end,cutoff)&meta.finite_label


def choice_scores(choices,mean,cov):
    cols=np.asarray(choices)+np.arange(np.asarray(choices).shape[1])[None,:]*5
    m=mean[cols].sum(axis=1);v=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
    return np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(v,0)),out=np.zeros_like(m),where=v>1e-18)


def fit_sizes(book,tree,leaves,levels):
    levels=np.asarray(levels);ns=len(leaves);rng=np.random.default_rng(2026)
    mean,cov=book.mean(axis=0),np.cov(book,rowvar=False,ddof=1)
    prob=np.full((ns,len(levels)),1/len(levels));best=np.full(ns,4)
    baseline=best_score=float(choice_scores(best[None],mean,cov)[0])
    for _ in range(50):
        aliases=(rng.random((192,ns))[:,:,None]>prob.cumsum(axis=1)[None]).sum(axis=2).clip(max=len(levels)-1)
        choices=levels[aliases];choices[0]=best;choices[1:len(levels)+1]=levels[:,None]
        score=choice_scores(choices,mean,cov);winner=score.argmax()
        if score[winner]>best_score:best,best_score=choices[winner].copy(),float(score[winner])
        elite=choices[np.argsort(score)[-19:]]
        frequency=np.stack([(elite==level).mean(axis=0) for level in levels],axis=1)
        prob=.97*(.3*prob+.7*frequency)+.03/len(levels)
    return (tree,leaves,best),dict(training_shadow_daily_sharpe=best_score,
        baseline_shadow_daily_sharpe=baseline,allocations=(best/4).tolist())


def predict(model,x):
    tree,leaves,levels=model
    if tree is None:return np.full(len(x),int(levels[0]))
    return levels[np.searchsorted(leaves,tree.apply(x))]
