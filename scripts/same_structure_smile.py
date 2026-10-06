"""The P25 directional sizing architecture, shared by every smile wing."""
import numpy as np

from scripts.tail_put25_pooled import FEATURES, OPTIONAL
from scripts.run_tail_put25_residual_rl import fit_allocations
from scripts.tail_put25_rule_sizing import SIZE_CODES

MENUS = {'floor': np.array([2, 3, 4]), 'flex': np.arange(5)}


def oriented_states(raw):
    """Identical 27 field definitions; only the selected contract changes."""
    x = raw.copy()
    direction = x.rule_direction.to_numpy(int)
    x['direction'] = direction
    x['oriented_iv_gap'] = direction * x.iv_gap_relative
    x['rank_strength'] = (x.iv_rank - .5).abs() * 2
    x['oriented_vrp_gap'] = direction * x.vrp_gap_relative
    x['oriented_vrp_gap_available'] = x.vrp_gap_relative_available
    x['oriented_dte_gap'] = direction * x.dte_gap
    x['oriented_dte_gap_available'] = x.dte_gap_available
    for days in [91, 365]:
        for prefix in ['profit', 'count']:
            col = f'{prefix}_{days}d'
            for suffix in ['', '_available']:
                x[col + suffix] = np.where(direction > 0,
                    x[f'{prefix}_long_{days}d{suffix}'], x[f'{prefix}_short_{days}d{suffix}'])
    x['signal_eligible'] = (x.observed & x.eligible & x.context & x.direction.ne(0) &
                            x.count_91d.ge(21) & x.count_365d.ge(63) &
                            x.profit_91d.gt(0) & x.profit_365d.gt(0))
    assert np.isfinite(x[FEATURES]).all().all()
    # Optional availability flags are state, never an additional veto.
    assert all(c + '_available' in x for c in OPTIONAL)
    x['target'] = np.where(direction > 0, x.target_long, x.target_short)
    return x


def directional_paths(labels, direction):
    """0/25/50/75/100% paths, with the proposal direction fixed."""
    result = np.zeros((len(labels), 5, labels.shape[-1]), labels.dtype)
    for size in range(1, 5):
        result[:, size] = labels[np.arange(len(labels)), np.where(direction > 0, 4 + size, 5 - size)]
    return result


def fit_sizes(book, dates, tree, leaves, variant):
    if variant == 'floor':
        # Exact optimizer and full-size anchor of the strongest saved P25 model.
        fitted, log = fit_allocations(book, dates, tree, leaves)
        choices = np.array([SIZE_CODES.index(int(code)) for code in fitted[2]])
        return choices, log
    if variant != 'flex':
        raise ValueError(variant)
    ns = len(leaves)
    mean, cov = book.mean(axis=0), np.cov(book, rowvar=False, ddof=1)
    def score(choices):
        cols = choices + np.arange(ns)[None, :] * 5
        m = mean[cols].sum(axis=1)
        v = cov[cols[:, :, None], cols[:, None, :]].sum(axis=(1, 2))
        return np.sqrt(252) * np.divide(m, np.sqrt(np.maximum(v, 0)), out=np.zeros_like(m), where=v > 1e-18)
    menu = MENUS[variant]
    rng = np.random.default_rng(2026)
    probability = np.full((ns, len(menu)), 1 / len(menu))
    best = np.full(ns, 4)
    best_score = float(score(best[None])[0])
    for _ in range(50):
        aliases = (rng.random((192, ns))[:, :, None] > probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=len(menu) - 1)
        choices = menu[aliases]
        choices[0] = best
        choices[1:len(menu) + 1] = menu[:, None]
        values = score(choices)
        winner = values.argmax()
        if values[winner] > best_score:
            best, best_score = choices[winner].copy(), float(values[winner])
        elite = choices[np.argsort(values)[-19:]]
        frequency = np.stack([(elite == level).mean(axis=0) for level in menu], axis=1)
        probability = .97 * (.3 * probability + .7 * frequency) + .03 / len(menu)
    return best, dict(training_shadow_daily_sharpe=best_score,
                      baseline_shadow_daily_sharpe=float(score(np.full((1, ns), 4))[0]),
                      training_objective=best_score, objective='daily_sharpe')


def predict(model, rows):
    if model['tree'] is None:
        return np.full(len(rows), 4, int)
    leaves = model['tree'].apply(rows[FEATURES])
    return model['choices'][np.searchsorted(model['leaves'], leaves)]
