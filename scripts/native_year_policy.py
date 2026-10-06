"""Small contextual episodic policy search on native year-long rewards.

Actions select direction and entry size, then hold to expiry. This is direct
policy optimization in a model of completed rewards, not neural/sequential Q.
"""
from itertools import product

import numpy as np
import pandas as pd

FRACTIONS = np.array([0., -1., -.75, -.5, -.25, .25, .5, .75, 1.])
MENUS = {'rl_direction_fixed': np.array([0, 1, 8]),
         'rl_direction_sized': np.array([0, 1, 3, 4, 5, 6, 8])}
FORECAST_MENUS = {'forecast_rl_sized': np.array([0, 5, 6, 8]),
                  'forecast_rl_sized_no_flat': np.array([5, 6, 8])}
MEMORY_FEATURES = ['memory_long', 'memory_short', 'memory_count']


def training_rows(rows, cutoff, years=5):
    return rows.loc[rows.signal_eligible & rows.finite_label & rows.end.lt(cutoff) &
                    rows.date.lt(cutoff) & rows.date.ge(cutoff-pd.DateOffset(years=years))].copy()


def native_memory(rows, label_folder):
    out = rows.copy()
    out[MEMORY_FEATURES] = 0.
    for pair, idx in out.groupby('pair', sort=False).groups.items():
        labels = np.load(label_folder / f'{pair}.npy', mmap_mode='r')
        group = out.loc[idx]
        dates, ends = group.date.to_numpy(), group.end.to_numpy()
        starts = (pd.DatetimeIndex(group.date) - pd.DateOffset(years=3)).to_numpy()
        eligible = (group.signal_eligible & group.finite_label & group.label_status.eq('complete')).to_numpy()
        sums = labels[group.label_row.to_numpy(int)][:, [8, 1]].sum(axis=2)
        memory = np.zeros((len(group), 3))
        for j, date in enumerate(dates):
            take = eligible & (ends < date) & (dates >= starts[j])
            if take.any():
                memory[j, :2], memory[j, 2] = sums[take].mean(axis=0), take.sum()
        out.loc[idx, MEMORY_FEATURES] = memory
        del labels
    return out


def select_policy(book, menu, states):
    """Enumerate every declared action assignment for at most four states."""
    assert states <= 4 and book.shape[1] == states * 9
    choices = np.array(list(product(menu, repeat=states)), int)
    mean = book.mean(axis=0)
    cov = np.cov(book, rowvar=False, ddof=1)
    cols = choices + 9 * np.arange(states)[None, :]
    mu = mean[cols].sum(axis=1)
    var = cov[cols[:, :, None], cols[:, None, :]].sum(axis=(1, 2))
    score = np.sqrt(252) * np.divide(mu, np.sqrt(np.maximum(var, 0)), out=np.zeros_like(mu), where=var > 1e-18)
    tied = np.flatnonzero(score >= score.max() - 1e-10)
    selected = tied[np.argmin(np.abs(FRACTIONS[choices[tied]]).sum(axis=1))]
    return choices[selected], dict(training_shadow_sharpe=float(score[selected]),
        assignments_evaluated=len(choices), choices=choices[selected].tolist())


def decode(code):
    action = FRACTIONS[np.asarray(code, int)]
    return np.sign(action).astype(int), np.round(4 * np.abs(action)).astype(int)


def forecast_size_rewards(rewards, direction):
    """Size-only action codes follow the frozen native forecast orientation."""
    out = np.zeros_like(rewards)
    if direction == 1:
        out[[5, 6, 8]] = rewards[[5, 6, 8]]
    elif direction == -1:
        out[[5, 6, 8]] = rewards[[4, 3, 1]]
    elif direction != 0:
        raise ValueError('Unexpected native forecast direction')
    return out


def executable_rows(rows):
    """Preserve explicit flat actions in the legacy nonzero-direction schema."""
    out = rows.copy()
    if (out.learned_level.gt(0) & ~out.direction.isin([-1, 1])).any():
        raise ValueError('A nonflat native action requires a real direction')
    out['native_requested_direction'] = out.direction
    out['direction'] = out.direction.where(out.learned_level.gt(0), 1).astype(int)
    # A flat action has zero exposure regardless of its unused orientation.
    np.testing.assert_array_equal(out.direction*out.learned_level,
                                  out.native_requested_direction*out.learned_level)
    return out
