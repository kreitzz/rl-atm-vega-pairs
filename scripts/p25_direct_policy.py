"""Direct CEM policies with the original daily shadow-path Sharpe objective.

Linear and spline scores are dimensionless decision scores, not payoff forecasts.
Only supplied training rows determine their bases. Sizes stay on 50/75/100%.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scripts.p25_additive_sizing import _fit_main


def size_codes(scores):
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite policy scores')
    return np.searchsorted([-.5, .5], scores, side='right') + 2


@dataclass
class DirectPolicy:
    features: list
    kind: str
    mains: dict
    slices: dict
    coefficients: np.ndarray

    def design(self, frame):
        return np.column_stack([np.ones(len(frame))] +
                               [self.mains[f].transform(frame) for f in self.features])

    def predict(self, frame):
        return self.design(frame) @ self.coefficients

    def size_codes(self, frame):
        return size_codes(self.predict(frame))

    def contributions(self, frame):
        result = pd.DataFrame({'intercept': np.full(len(frame), self.coefficients[0])}, index=frame.index)
        for f in self.features:
            result[f] = self.mains[f].transform(frame) @ self.coefficients[self.slices[f]]
        return result


def make_policy(rows, features, kind):
    if kind not in ['linear', 'gam']:
        raise ValueError('Expected linear or gam policy')
    if not features or len(set(features)) != len(features):
        raise ValueError('Expected unique features')
    if set(features) & {'target', 'end', 'date', 'pair', 'label_row', 'position'}:
        raise ValueError('Outcome and identity columns cannot be predictors')
    mains, slices, pos = {}, {}, 1
    for f in features:
        term, basis, _ = _fit_main(rows, f, kind)
        mains[f] = term
        slices[f] = slice(pos, pos + basis.shape[1])
        pos += basis.shape[1]
    return DirectPolicy(list(features), kind, mains, slices, np.zeros(pos))


class ShadowScorer:
    """Sparse calendar paths preserve overlaps, true action costs and zero days."""
    def __init__(self, matrix, n_rows):
        self.matrix = matrix.tocsr()
        self.n_rows = n_rows
        if matrix.shape[1] != 3*n_rows or matrix.shape[0] < 2:
            raise ValueError('Expected calendar by trade/action matrix')

    def daily(self, choices):
        choices = np.asarray(choices)
        if choices.ndim == 1:
            choices = choices[None, :]
        if choices.shape[1] != self.n_rows or not np.isin(choices, [2, 3, 4]).all():
            raise ValueError('Expected candidate by trade action codes 2/3/4')
        selected = np.zeros((3*self.n_rows, len(choices)))
        selected[np.arange(self.n_rows)[None, :]*3 + choices-2,
                 np.arange(len(choices))[:, None]] = 1.
        return self.matrix @ selected

    def score(self, choices):
        daily = self.daily(choices)
        variance = daily.var(axis=0, ddof=1)
        return np.sqrt(252)*np.divide(daily.mean(axis=0), np.sqrt(variance),
            out=np.zeros(len(variance)), where=variance > 1e-18)


def shadow_scorer(rows, calendar, label_folder):
    """Load exactly the directional paths used by p25_representation.state_book."""
    first = int(rows.position.min()+1)
    last = int(calendar.get_indexer(rows.end).max())
    rr, cc, vv = [], [], []
    for pair, indices in rows.groupby('pair', sort=False).groups.items():
        idx = np.asarray(indices, int)
        disk = np.load(label_folder/pair/'P25_labels.npy', mmap_mode='r')
        raw = disk[rows.loc[idx, 'label_row'].to_numpy(int)]
        direction = rows.loc[idx, 'direction'].to_numpy(int)
        days = rows.loc[idx, 'position'].to_numpy(int)[:, None]+1+np.arange(raw.shape[-1])[None]-first
        valid = (days >= 0) & (days <= last-first)
        for action in [2, 3, 4]:
            values = raw[np.arange(len(idx)), np.where(direction > 0, 4+action, 5-action)].astype(float)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite completed training path')
            active = valid & (values != 0)
            rr.extend(days[active]); cc.extend(np.broadcast_to(idx[:, None]*3+action-2, values.shape)[active])
            vv.extend(values[active])
        del disk, raw
    return ShadowScorer(coo_matrix((vv, (rr, cc)), shape=(last-first+1, 3*len(rows))), len(rows))


def fit_direct(policy, rows, scorer, seed=2026, iterations=50, population=192, rest=lambda: None):
    """Bounded diagonal Gaussian CEM; exactly population*iterations evaluations.

    Same elite count (19/192), smoothing (.3/.7), constants and best anchor as
    original categorical CEM. Gaussian search geometry necessarily differs.
    No payoff regression, ridge penalty, validation tuning or OOS selection.
    """
    if population < 10 or iterations < 1:
        raise ValueError('Insufficient search budget')
    design = policy.design(rows)
    dimension = design.shape[1]
    initial_sd = np.full(dimension, 1/np.sqrt(len(policy.features)))
    initial_sd[0] = 1.
    mean, sd = np.zeros(dimension), initial_sd.copy()
    best = np.zeros(dimension); best[0] = 1.
    best_score = -np.inf
    rng = np.random.default_rng(seed)
    elite_count = max(1, int(.1*population))
    trace = []
    baseline = None
    for iteration in range(iterations):
        candidates = np.clip(rng.normal(mean, sd, size=(population, dimension)), -3., 3.)
        candidates[0] = best
        candidates[1:4] = 0.
        candidates[1:4, 0] = [-1., 0., 1.]
        actions = size_codes(design @ candidates.T).T
        scores = scorer.score(actions)
        if baseline is None:
            baseline = float(scores[3])
            best_score = baseline
        winner = int(scores.argmax())
        if scores[winner] > best_score:
            best, best_score = candidates[winner].copy(), float(scores[winner])
        elite = candidates[np.argsort(scores)[-elite_count:]]
        mean = .3*mean + .7*elite.mean(axis=0)
        sd = np.maximum(.3*sd + .7*elite.std(axis=0), .03*initial_sd)
        trace.append(dict(iteration=iteration+1, best_score=best_score,
                          batch_max=float(scores.max()), batch_median=float(np.median(scores)),
                          unique_assignments=len(np.unique(actions, axis=0))))
        rest()
    policy.coefficients = best
    check = float(scorer.score(policy.size_codes(rows))[0])
    np.testing.assert_allclose(check, best_score, rtol=1e-12, atol=1e-12)
    return policy, dict(training_shadow_daily_sharpe=best_score, baseline_shadow_daily_sharpe=baseline,
        evaluations=iterations*population, seed=seed, parameters=dimension,
        parameter_bound=3., final_ten_iteration_gain=best_score-trace[max(0, len(trace)-11)]['best_score'],
        trace=trace)
