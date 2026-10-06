"""Causal training windows and recency-weighted positive-size allocation."""
import numpy as np
import pandas as pd
from scripts.p25_action_windows import fit_menu
from scripts.run_tail_put25_residual_rl import score_choices


def completed_months(dates, ends, cutoff, months):
    cutoff = pd.Timestamp(cutoff)
    return (dates >= cutoff-pd.DateOffset(months=months)) & (dates < cutoff) & (ends < cutoff)


def recency_weights(dates, cutoff, half_life_months):
    dates = pd.DatetimeIndex(dates); cutoff = pd.Timestamp(cutoff)
    if not (dates < cutoff).all():
        raise ValueError('Weights require strictly past dates')
    age = (cutoff-dates).total_seconds().to_numpy()/86400
    return np.exp2(-age/(365.25/12*half_life_months))


def weighted_moments(book, weights):
    weights = np.asarray(weights, float)
    if len(weights) != len(book) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive finite weights required for every daily row')
    weights = weights/weights.sum()
    mean = weights @ book
    centered = book-mean
    cov = (centered*weights[:, None]).T @ centered/(1-weights@weights)
    return [(mean, cov)]


def fit_recency(book, dates, menu, cutoff, half_life_months=None):
    if half_life_months is None:
        return fit_menu(book, dates, menu)
    menu = np.asarray(menu, int)
    if len(menu) < 2 or 4 not in menu or not np.isin(menu, [2, 3, 4]).all():
        raise ValueError('Positive sizing menu must include full size')
    weights = recency_weights(dates, cutoff, half_life_months)
    stats = weighted_moments(book, weights); ns = book.shape[1]//5
    rng = np.random.default_rng(2026)
    probability = np.full((ns, len(menu)), 1/len(menu))
    best = np.full(ns, 4); best_score = float(score_choices(best[None], stats)[0])
    for _ in range(50):
        aliases = (rng.random((192, ns))[:, :, None] > probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=len(menu)-1)
        choices = menu[aliases]; choices[0] = best; choices[1:len(menu)+1] = menu[:, None]
        scores = score_choices(choices, stats); winner = scores.argmax()
        if scores[winner] > best_score:
            best = choices[winner].copy(); best_score = float(scores[winner])
        elite = choices[np.argsort(scores)[-19:]]
        frequency = np.stack([(elite == level).mean(axis=0) for level in menu], axis=1)
        probability = .97*(.3*probability+.7*frequency)+.03/len(menu)
    w = weights/weights.sum()
    return best, dict(training_shadow_daily_sharpe=best_score,
        baseline_shadow_daily_sharpe=float(score_choices(np.full((1, ns), 4), stats)[0]),
        training_objective=best_score, objective='recency_weighted_daily_sharpe',
        effective_daily_observations=float(1/(w@w)))
