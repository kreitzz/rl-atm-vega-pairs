"""A small inventory-dependent policy for sequential portfolio search."""
import numpy as np

SCALES = np.array([.5, 1., 2., 4.])
FEATURES = ['bias', 'expert_size', 'stress_usage', 'reserve_usage',
            'stock_hedge_usage', 'name_inventory', 'excess_drawdown',
            'pair_stress_usage', 'cash_borrowing']


def state(row, context):
    """Normalized prior-close account state; expert size uses frozen signals."""
    return np.array([1., (float(row['learned_level']) / 4 - .75) / .25,
        min(3., context['stress_utilization']),
        min(3., context['reserve_utilization']),
        min(3., context['hedge_utilization']),
        min(3., context['name_inventory_units'] / 8),
        min(3., context['excess_drawdown_fraction'] / .02),
        min(3., context['pair_stress_utilization']),
        min(3., context['borrowing_fraction'])], dtype=float)


def scale(weights, row, context):
    weights = np.asarray(weights, float)
    if weights.shape != (len(FEATURES),) or not np.isfinite(weights).all():
        raise ValueError('Capital policy requires one finite weight per state')
    score = float(weights @ state(row, context))
    return float(SCALES[np.searchsorted([-.5, .5, 1.5], score, side='right')])


def constant_weights(multiplier):
    if multiplier not in SCALES:
        raise ValueError('Unsupported constant leverage benchmark')
    weights = np.zeros(len(FEATURES))
    weights[0] = np.log2(multiplier)
    return weights
