"""Joint daily-Sharpe search for fixed-direction smile proposals."""
import numpy as np
from scripts.smile_sizing_policy import choice_scores


def optimize(book, baseline, wings, population=192, iterations=50):
    baseline=np.asarray(baseline,int);wings=np.asarray(wings);ns=len(baseline)
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    rng=np.random.default_rng(2026);prob=np.full((ns,5),.2)
    best=baseline.copy();initial=best_score=float(choice_scores(best[None],mean,cov)[0])
    anchors=np.vstack([baseline,np.zeros(ns,int),*(np.full(ns,a) for a in range(1,5)),
                       np.where(wings=='P75',0,baseline),np.where(wings=='P25',baseline,0)])
    for _ in range(iterations):
        choices=(rng.random((population,ns))[:,:,None]>prob.cumsum(axis=1)[None]).sum(axis=2).clip(max=4)
        choices[:len(anchors)]=anchors;choices[len(anchors)]=best
        score=choice_scores(choices,mean,cov);winner=score.argmax()
        if score[winner]>best_score:best=choices[winner].copy();best_score=float(score[winner])
        elite=choices[np.argsort(score)[-max(1,population//10):]]
        frequency=np.stack([(elite==a).mean(axis=0) for a in range(5)],axis=1)
        prob=.97*(.3*prob+.7*frequency)+.03/5
    assert best_score>=initial-1e-12
    return best,dict(training_shadow_daily_sharpe=best_score,
                     frozen_sizes_shadow_daily_sharpe=initial,allocations=(best/4).tolist())


def predict(model,row,features,context):
    group=model['groups'].get((row['wing'],int(row['base_level'])))
    if group is None:return int(row['base_level'])
    tree,leaves,offset=group
    if tree is None:leaf=0
    else:
        x=np.array([[context.get(c,row[c]) for c in features]],float)
        leaf=int(np.searchsorted(leaves,tree.apply(x)[0]))
    return int(model['levels'][offset+leaf])
