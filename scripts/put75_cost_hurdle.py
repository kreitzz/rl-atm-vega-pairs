"""Change only entry commissions in cached standalone contract cash flows."""
from copy import copy
from dataclasses import replace

import numpy as np


def with_training_commission(unit, commission):
    commission = float(commission)
    if not np.isfinite(commission) or commission < 0:
        raise ValueError('Commission must be finite and nonnegative')
    result = copy(unit)
    result.cfg = replace(unit.cfg, option_commission=commission)
    result.flow_long = unit.flow_long.copy()
    result.flow_short = unit.flow_short.copy()
    # UnitPaths charges one initial contract at entry. Pair-level integer
    # quantities and positive/negative cash financing are recomputed afterwards.
    entered = unit.entered & unit.resolved
    delta = commission - unit.cfg.option_commission
    result.flow_long[entered, 0] -= delta
    result.flow_short[entered, 0] -= delta
    return result
