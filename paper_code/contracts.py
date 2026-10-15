"""Quote checks, whole-contract vega matching, and execution charges.

Vega is dollars per share per ONE volatility point; contracts represent 100
shares. Settings are loaded from execution.json for the paper replay.
"""
from dataclasses import dataclass
import numpy as np

class UnresolvedPosition(RuntimeError):
    """An existing position cannot be valued/hedged; do not drop its outcome."""

class MarketRow(dict):
    """Attribute-compatible immutable-source market row for bounded lookups."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def copy(self):
        return MarketRow(self)

@dataclass
class Settings:
    capital: float = 1000000
    entry_vega_per_leg: float = 450
    max_gross_vega: float = 20000
    max_entry_vega_imbalance: float = 0.1
    option_commission: float = 0.65
    stock_cost_bps: float = 1
    borrowing_spread: float = 0.01
    stock_borrow_rate: float = 0.003
    min_open_interest: int = 100
    max_relative_spread: float = 0.25
    max_moneyness_distance: float = 0.1
    min_dte: int = 21
    max_dte: int = 45
    target_dte: int = 30
    holding_days: int = 30
    hold_to_expiry: bool = False
    repair_crossed_held_quotes: bool = False
    support_integer_splits: bool = False
    option_spread_fraction: float = 1.0
    option_penalty_vol_points: float | None = None
    allow_missing_flat_underlying: bool = False
    risk_controls: bool = False
    max_hedge_leverage: float = 2.0
    max_reserve_fraction: float = 0.8
    stock_reserve_fraction: float = 0.5
    max_stress_loss_fraction: float = 0.05
    spot_stress: float = 0.2
    vol_stress: float = 0.1

    def __post_init__(self):
        risk_values = [self.max_hedge_leverage, self.max_reserve_fraction, self.stock_reserve_fraction, self.max_stress_loss_fraction, self.spot_stress, self.vol_stress]
        if not np.isfinite(risk_values).all() or min(risk_values) <= 0:
            raise ValueError('Risk limits must be positive and finite')
        if max(self.max_reserve_fraction, self.stock_reserve_fraction, self.max_stress_loss_fraction) >= 1 or self.spot_stress >= 1:
            raise ValueError('Capital fractions and spot stress must be below one')
        if not np.isfinite(self.option_spread_fraction) or not 0 <= self.option_spread_fraction <= 1:
            raise ValueError('Option spread fraction must be between 0 (mid) and 1 (quoted side)')
        if self.option_penalty_vol_points is not None and (not np.isfinite(self.option_penalty_vol_points) or self.option_penalty_vol_points < 0):
            raise ValueError('Mid execution penalty must be finite and nonnegative')
        if min(self.capital, self.entry_vega_per_leg, self.max_gross_vega) <= 0:
            raise ValueError('Capital and vega limits must be positive')
        if min(self.option_commission, self.stock_cost_bps, self.borrowing_spread, self.stock_borrow_rate) < 0:
            raise ValueError('Costs must be nonnegative')

def valid_greeks(q):
    return np.isfinite([q['delta'], q['vega']]).all() and (0 <= q['delta'] <= 1 if q.get('cp_flag', 'C') == 'C' else -1 <= q['delta'] <= 0) and (q['vega'] >= 0)

def valid_quote(q, entry=False, cfg=None):
    fields = ['bid', 'ask', 'strike', 'multiplier', 'cfadj']
    if not all((np.isfinite(float(q[x])) for x in fields)):
        return False
    if not (0 <= q['bid'] <= q['ask'] and q['ask'] > 0 and (q['strike'] > 0)):
        return False
    if q['multiplier'] != 100 or not q['standard'] or q['cp_flag'] not in ['C', 'P']:
        return False
    if entry:
        mid = (q['bid'] + q['ask']) / 2
        return valid_greeks(q) and q['bid'] > 0 and (q['vega'] > 0) and (q['open_interest'] >= cfg.min_open_interest) and ((q['ask'] - q['bid']) / mid <= cfg.max_relative_spread)
    return True

def integer_size(vegas, target, tolerance):
    """Floor each leg to its budget; reject excessive residual vega."""
    v = np.asarray(vegas, dtype=float)
    if not np.isfinite(v).all() or (v <= 0).any():
        return None
    n = np.floor(target / v).astype(int)
    actual = n * v
    if (n < 1).any() or abs(actual[0] - actual[1]) / actual.max() > tolerance:
        return None
    return n

def verified_integer_split(old, quote, ratio):
    """Only ordinary forward splits corroborated by both stock and option data."""
    return np.isfinite(ratio) and ratio >= 2 and np.isclose(ratio, round(ratio)) and quote['standard'] and (quote['multiplier'] == 100) and (quote['cp_flag'] in ['C', 'P']) and (quote['cp_flag'] == old.get('cp_flag', 'C')) and (quote['expiry'] == old['expiry']) and np.isclose(quote['cfadj'] / old['cfadj'], ratio) and np.isclose(quote['strike'], old['strike'] / ratio, rtol=0, atol=0.00500001)

def execution_terms(quote, quantity, cfg, fallback_vega=None):
    """Fill price, separately charged penalty, total slippage dollars.

    Mid penalties are cash costs, not fictitious negative option prices.
    A forced exit with missing Greeks uses the last observed contract vega.
    Zero penalty requires no vega. Quoted-spread mode is unchanged.
    """
    mid = float((quote['ask'] + quote['bid']) / 2)
    if cfg.option_penalty_vol_points is None:
        half = cfg.option_spread_fraction * (quote['ask'] - quote['bid']) / 2
        return (float(mid + np.sign(quantity) * half), 0.0, float(abs(quantity) * 100 * half))
    if cfg.option_penalty_vol_points == 0 or quantity == 0:
        return (mid, 0.0, 0.0)
    vega = quote['vega']
    if not np.isfinite(vega) or vega < 0:
        vega = fallback_vega
    if vega is None or not np.isfinite(vega) or vega < 0:
        raise UnresolvedPosition('No observed vega for mid execution penalty')
    penalty = float(abs(quantity) * 100 * vega * cfg.option_penalty_vol_points)
    return (mid, penalty, penalty)
