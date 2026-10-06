"""Declared action menus around the unchanged P25 directional signal."""
import numpy as np
from scripts.run_tail_put25_residual_rl import moments, score_choices

WINDOWS = [1,3,5]
MENUS = {'binary':np.array([2,4]), 'floor':np.array([2,3,4]),
         'quarter':np.array([1,2,3,4]), 'flex':np.arange(5)}


def fit_menu(book, dates, menu):
    """Same direct-Sharpe CEM kernel, seed, budget and full-size anchor."""
    menu=np.asarray(menu,int)
    if len(menu)<2 or 4 not in menu or not np.isin(menu,np.arange(5)).all():
        raise ValueError('Declared0/25/50/75/100 menu must include full size')
    stats=moments(book,dates,False);ns=book.shape[1]//5
    rng=np.random.default_rng(2026)
    probability=np.full((ns,len(menu)),1/len(menu))
    best=np.full(ns,4);best_score=float(score_choices(best[None],stats)[0])
    for _ in range(50):
        aliases=(rng.random((192,ns))[:,:,None]>probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=len(menu)-1)
        choices=menu[aliases];choices[0]=best;choices[1:len(menu)+1]=menu[:,None]
        scores=score_choices(choices,stats)
        winner=scores.argmax()
        if scores[winner]>best_score:
            best=choices[winner].copy();best_score=float(scores[winner])
        elite=choices[np.argsort(scores)[-19:]]
        frequency=np.stack([(elite==level).mean(axis=0) for level in menu],axis=1)
        probability=.97*(.3*probability+.7*frequency)+.03/len(menu)
    return best,dict(training_shadow_daily_sharpe=best_score,
        baseline_shadow_daily_sharpe=float(score_choices(np.full((1,ns),4),stats)[0]),
        training_objective=best_score,objective='daily_sharpe')


def score_model(model, rows, features):
    if model['tree'] is None:return np.full(len(rows),4,int)
    leaves=model['tree'].apply(rows[features])
    return model['choices'][np.searchsorted(model['leaves'],leaves)]
