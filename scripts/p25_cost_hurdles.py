"""Change the training commission reward while preserving market paths."""
from dataclasses import replace
import numpy as np
from scripts.retrain_atm_commission import cheaper_units
from scripts.curve_trading.unit_paths import pair_labels

FEES = [.65, .30, .15, .03]


def fee_name(fee):
    return f'train_{int(round(fee*100)):03d}'


def reprice_labels(a, b, fee):
    cfg = replace(a.cfg, option_commission=fee)
    assert a.cfg == b.cfg
    units = [cheaper_units(u, cfg) for u in [a, b]]
    return pair_labels(*units, (30, 'P', 25, 25), cfg)


def directional_targets(labels, direction):
    return labels[np.arange(len(labels)), np.where(np.asarray(direction)>0, 8, 1)].sum(axis=1)
