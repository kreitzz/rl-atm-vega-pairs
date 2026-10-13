"""Monthly regression-tree states and categorical CEM entry-sizing search.

This is offline contextual policy optimization. Training sums completed shadow
trade paths; action-dependent funded admission happens later in ledger.py.
"""
import numpy as np
import pandas as pd

from .signals import FEATURES


def completed_window(rows, cutoff):
    """18 months by signal date; both signal and completion precede the fit."""
    cutoff = pd.Timestamp(cutoff)
    return (rows.signal_eligible & rows.finite_label
            & rows.date.ge(cutoff - pd.DateOffset(months=18))
            & rows.date.lt(cutoff) & rows.end.lt(cutoff))


def state_book(rows, states, leaf_count, calendar, label_folder):
    """Aggregate action-specific daily paths with one pair loaded at a time.

    Label arrays have nine columns: flat, four short-A sizes, four long-A sizes.
    Do not linearly scale full-size paths: the archived paths are action-specific.
    Zero-profit calendar sessions remain in the Sharpe denominator.
    """
    first = int(rows.position.min() + 1)
    last = int(calendar.get_indexer(rows.end).max())
    if last < first or (calendar.get_indexer(rows.end) < 0).any():
        raise ValueError("Training completion sessions must exist in the calendar")
    book = np.zeros((last - first + 1, leaf_count * 5))
    for pair, indices in rows.groupby("pair", sort=False).groups.items():
        indices = np.asarray(indices)
        disk = np.load(label_folder / pair / "P25_labels.npy", mmap_mode="r")
        raw = disk[rows.loc[indices, "label_row"].to_numpy(int)]
        direction = rows.loc[indices, "direction"].to_numpy(int)
        paths = np.zeros((len(raw), 5, raw.shape[-1]), raw.dtype)
        for action in range(1, 5):
            columns = np.where(direction > 0, 4 + action, 5 - action)
            paths[:, action] = raw[np.arange(len(raw)), columns]
        days = (rows.loc[indices, "position"].to_numpy(int)[:, None] + 1
                + np.arange(paths.shape[-1])[None] - first)
        valid = (days >= 0) & (days < len(book))
        for leaf in np.unique(states[indices]):
            local = np.flatnonzero(states[indices] == leaf)
            for action in range(5):
                np.add.at(book[:, int(leaf) * 5 + action], days[local][valid[local]],
                          paths[local, action][valid[local]])
        del disk, raw, paths
    return book


def search_sizes(book):
    """Exact retained CEM: 50 rounds, 192 maps, 19 elites, seed 2026."""
    mean, covariance = book.mean(axis=0), np.cov(book, rowvar=False, ddof=1)
    states = book.shape[1] // 5
    menu = np.array([2, 3, 4])  # quarter units: 50%, 75%, 100%

    def score(choices):
        columns = choices + np.arange(states)[None, :] * 5
        expected = mean[columns].sum(axis=1)
        variance = covariance[columns[:, :, None], columns[:, None, :]].sum(axis=(1, 2))
        return np.sqrt(252) * np.divide(
            expected, np.sqrt(np.maximum(variance, 0)),
            out=np.zeros_like(expected), where=variance > 1e-18)

    rng = np.random.default_rng(2026)
    probabilities = np.full((states, 3), 1 / 3)
    best = np.full(states, 4)
    best_score = baseline = float(score(best[None])[0])
    for _ in range(50):
        aliases = (rng.random((192, states))[:, :, None]
                   > probabilities.cumsum(axis=1)[None]).sum(axis=2).clip(max=2)
        choices = menu[aliases]
        choices[0] = best
        choices[1:4] = menu[:, None]  # include each constant-size control
        scores = score(choices)
        winner = scores.argmax()
        if scores[winner] > best_score:
            best, best_score = choices[winner].copy(), float(scores[winner])
        elite = choices[np.argsort(scores)[-19:]]
        frequency = np.stack([(elite == level).mean(axis=0) for level in menu], axis=1)
        probabilities = .97 * (.3 * probabilities + .7 * frequency) + .03 / 3
    return best, {"training_shadow_daily_sharpe": best_score,
                  "baseline_shadow_daily_sharpe": baseline}


def fit_month(meta, calendar, label_folder, cutoff):
    """Fit one causal month; insufficient history falls back to full size."""
    from sklearn.tree import DecisionTreeRegressor

    cutoff = pd.Timestamp(cutoff)
    rows = meta.loc[completed_window(meta, cutoff)].reset_index(drop=True)
    model = {"tree": None, "leaves": np.array([0]), "choices": np.array([4]),
             "features": FEATURES, "fit": str(cutoff.date())}
    log = {"training_rows": len(rows), "last_training_outcome":
           str(rows.end.max().date()) if len(rows) else None}
    if len(rows) >= 16:
        tree = DecisionTreeRegressor(
            max_leaf_nodes=8, min_samples_leaf=max(8, min(63, len(rows) // 8)),
            random_state=2026).fit(rows[FEATURES], rows.target)
        leaves = np.unique(tree.apply(rows[FEATURES]))
        states = np.searchsorted(leaves, tree.apply(rows[FEATURES]))
        book = state_book(rows, states, len(leaves), calendar, label_folder)
        choices, details = search_sizes(book)
        model.update(tree=tree, leaves=leaves, choices=choices)
        log.update(details)
    return model, log


def predict(model, rows):
    if model["tree"] is None:
        return np.full(len(rows), 4, dtype=int)
    leaves = model["tree"].apply(rows[FEATURES])
    return model["choices"][np.searchsorted(model["leaves"], leaves)]


def verify_model(actual, expected):
    """Compare the refactored fit with the archived tree and sizing map."""
    for field in ("choices", "leaves"):
        np.testing.assert_array_equal(actual[field], expected[field])
    if (actual["tree"] is None) != (expected["tree"] is None):
        raise AssertionError("Full-size fallback differs")
    if actual["tree"] is not None:
        for field in ("feature", "threshold", "children_left", "children_right", "value"):
            np.testing.assert_array_equal(getattr(actual["tree"].tree_, field),
                                          getattr(expected["tree"].tree_, field))
