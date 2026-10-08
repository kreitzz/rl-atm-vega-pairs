"""Independent shallow action-value forests with a one-hot public interface.

A shared shallow tree can spend every split on state and ignore action. Each
legal action instead receives its own state-only regressor. Unobserved actions
remain unsupported; their values are never silently imputed or extrapolated.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def split_inputs(x,n_actions):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]<=n_actions or not np.isfinite(x).all():
        raise ValueError('Finite state plus one-hot action matrix required')
    codes=x[:,-n_actions:]
    if not np.isin(codes,[0.,1.]).all() or not np.equal(codes.sum(axis=1),1.).all():
        raise ValueError('Action inputs must be exactly one-hot')
    return x[:,:-n_actions],codes.argmax(axis=1)


class ActionTree:
    """One aligned tree from each supported action forest, routed by action."""
    def __init__(self,trees,n_actions,n_features):
        self.trees=trees;self.n_actions=n_actions;self.n_features_in_=n_features

    def predict(self,x):
        states,actions=split_inputs(x,self.n_actions)
        if np.asarray(x).shape[1]!=self.n_features_in_:
            raise ValueError('Critic input width changed')
        output=np.empty(len(states))
        for action in np.unique(actions):
            if int(action) not in self.trees:
                raise ValueError(f'Action{action} has no observed training support')
            mask=actions==action
            output[mask]=self.trees[int(action)].predict(states[mask])
        return output


class ActionForest:
    """Sklearn-like fitted critic over states followed by n_actions one-hot.

    Only observed action codes are fitted. This allows a four-column action
    interface to represent a policy whose legal action set contains three.
    ``estimators_`` route aligned trees, preserving per-tree advantages for a
    conservative extraction heuristic. Their dispersion is not an inferential
    confidence interval. Feature importances average supported state forests;
    routing action columns receive zeros rather than artificial split credit.
    """
    def __init__(self,n_actions,n_estimators=24,max_depth=3,min_samples_leaf=30,
                 random_state=2026,n_jobs=1):
        if int(n_actions)!=n_actions or n_actions<2:
            raise ValueError('At least two declared actions required')
        self.n_actions=int(n_actions);self.n_estimators=n_estimators
        self.max_depth=max_depth;self.min_samples_leaf=min_samples_leaf
        self.random_state=random_state;self.n_jobs=n_jobs

    def fit(self,x,y,sample_weight=None):
        states,actions=split_inputs(x,self.n_actions)
        y=np.asarray(y,float)
        if y.ndim!=1 or len(y)!=len(states) or not np.isfinite(y).all():
            raise ValueError('Finite scalar target for each transition required')
        weights=None if sample_weight is None else np.asarray(sample_weight,float)
        if weights is not None and (weights.shape!=y.shape or not np.isfinite(weights).all() or np.any(weights<=0)):
            raise ValueError('Positive finite transition weights required')
        self.n_features_in_=np.asarray(x).shape[1]
        self.present_actions_=[int(action) for action in np.unique(actions)]
        self.action_counts_={action:int((actions==action).sum()) for action in self.present_actions_}
        self.forests_={}
        for action in self.present_actions_:
            mask=actions==action
            forest=ExtraTreesRegressor(n_estimators=self.n_estimators,
                max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,n_jobs=self.n_jobs)
            forest.fit(states[mask],y[mask],sample_weight=None if weights is None else weights[mask])
            self.forests_[action]=forest
        self.estimators_=[ActionTree({action:forest.estimators_[j]
            for action,forest in self.forests_.items()},self.n_actions,self.n_features_in_)
            for j in range(self.n_estimators)]
        importance=np.mean([forest.feature_importances_ for forest in self.forests_.values()],axis=0)
        self.feature_importances_=np.r_[importance,np.zeros(self.n_actions)]
        return self

    def predict(self,x):
        if not hasattr(self,'forests_'):raise ValueError('ActionForest is not fitted')
        states,actions=split_inputs(x,self.n_actions)
        if np.asarray(x).shape[1]!=self.n_features_in_:
            raise ValueError('Critic input width changed')
        output=np.empty(len(states))
        for action in np.unique(actions):
            if int(action) not in self.forests_:
                raise ValueError(f'Action{action} has no observed training support')
            mask=actions==action
            output[mask]=self.forests_[int(action)].predict(states[mask])
        return output
