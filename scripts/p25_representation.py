"""Training-only representations and the retained P25 allocation search."""
import numpy as np

from scripts.diagnose_p25_pca import CORE, standardized_pca

CASES = ['core3', 'raw_full', 'pca3', 'pca80', 'pca_all']


def fit_representation(rows, inputs, case):
    if case not in CASES:
        raise ValueError(case)
    flags = [f for f in inputs if f.endswith('_available')]
    economic = [f for f in inputs if f not in flags]
    if case in ['core3', 'raw_full']:
        # All arms retain identical flags; these are constant in every saved fit.
        columns = (CORE + flags) if case == 'core3' else inputs
        return dict(case=case, columns=columns, components=None,
                    economic_dimensions=len(CORE) if case == 'core3' else len(economic))
    pca = standardized_pca(rows[economic])
    k = min(3, len(pca['ratio'])) if case == 'pca3' else (
        int(np.searchsorted(np.cumsum(pca['ratio']), .8) + 1)
        if case == 'pca80' else len(pca['ratio']))
    # Keep every direction in pca_all, including sample-null directions.
    # There are >=16 rows and 15 economic inputs, so the SVD is full dimensional.
    scale = np.where(pca['active'], pca['sd'], 1.)
    return dict(case=case, columns=economic, flags=flags, mean=pca['mean'],
                scale=scale, active=pca['active'], components=pca['weights'][:k],
                economic_dimensions=k, variance_retained=float(pca['ratio'][:k].sum()))


def transform(rows, representation):
    x = rows[representation['columns']].to_numpy(dtype=float)
    if representation['components'] is not None:
        z = (x - representation['mean']) / representation['scale']
        z[:, ~representation['active']] = 0.
        x = np.column_stack([z @ representation['components'].T,
                             rows[representation['flags']].to_numpy(dtype=float)])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite representation')
    return x


def fit_menu(book):
    """Exact retained 50x192 CEM, seed2026, 50/75/100 and full-size anchor."""
    mean = book.mean(axis=0)
    cov = np.cov(book, rowvar=False, ddof=1)
    ns = book.shape[1] // 5
    menu = np.array([2, 3, 4])

    def score(choices):
        cols = choices + np.arange(ns)[None, :] * 5
        m = mean[cols].sum(axis=1)
        v = cov[cols[:, :, None], cols[:, None, :]].sum(axis=(1, 2))
        return np.sqrt(252) * np.divide(m, np.sqrt(np.maximum(v, 0)),
                                       out=np.zeros_like(m), where=v > 1e-18)

    rng = np.random.default_rng(2026)
    probability = np.full((ns, len(menu)), 1 / len(menu))
    best = np.full(ns, 4)
    best_score = baseline = float(score(best[None])[0])
    for _ in range(50):
        aliases = (rng.random((192, ns))[:, :, None] >
                   probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=2)
        choices = menu[aliases]
        choices[0] = best
        choices[1:4] = menu[:, None]
        scores = score(choices)
        winner = scores.argmax()
        if scores[winner] > best_score:
            best, best_score = choices[winner].copy(), float(scores[winner])
        elite = choices[np.argsort(scores)[-19:]]
        frequency = np.stack([(elite == level).mean(axis=0) for level in menu], axis=1)
        probability = .97 * (.3 * probability + .7 * frequency) + .03 / len(menu)
    return best, dict(training_shadow_daily_sharpe=best_score,
                      baseline_shadow_daily_sharpe=baseline)


def state_book(rows, states, leaf_count, calendar, label_folder, track=lambda p: p):
    """Same directional-path arithmetic and daily aggregation as saved fitting."""
    first = int(rows.position.min() + 1)
    last = int(calendar.get_indexer(rows.end).max())
    book = np.zeros((last - first + 1, leaf_count * 5))
    for pair, indices in rows.groupby('pair', sort=False).groups.items():
        idx = np.asarray(indices)
        disk = np.load(track(label_folder / pair / 'P25_labels.npy'), mmap_mode='r')
        raw = disk[rows.loc[idx, 'label_row'].to_numpy(int)]
        direction = rows.loc[idx, 'direction'].to_numpy(int)
        labels = np.zeros((len(raw), 5, raw.shape[-1]), raw.dtype)
        for action in range(1, 5):
            labels[:, action] = raw[np.arange(len(raw)), np.where(direction > 0, 4 + action, 5 - action)]
        days = rows.loc[idx, 'position'].to_numpy(int)[:, None] + 1 + np.arange(labels.shape[-1])[None] - first
        valid = (days >= 0) & (days < len(book))
        for leaf in np.unique(states[idx]):
            local = np.flatnonzero(states[idx] == leaf)
            for action in range(5):
                np.add.at(book[:, int(leaf) * 5 + action], days[local][valid[local]],
                          labels[local, action][valid[local]])
        del disk, raw, labels
    return book


def score_model(model, rows):
    if model['tree'] is None:
        return np.full(len(rows), 4, int)
    leaves = model['tree'].apply(transform(rows, model['representation']))
    return model['choices'][np.searchsorted(model['leaves'], leaves)]
