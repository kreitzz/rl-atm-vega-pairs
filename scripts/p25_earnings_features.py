"""Earnings state construction with explicit historical-calendar assumptions."""
import numpy as np
import pandas as pd

FIELDS = ['long_earnings_days', 'short_earnings_days',
          'long_earnings_before_expiry', 'short_earnings_before_expiry']
CASES = ['historical_calendar', 'prior_reports_only']


def leg_features(dates, dte, events, etf=False, mode='prior_reports_only'):
    dates = pd.DatetimeIndex(dates)
    if mode not in CASES:
        raise ValueError(mode)
    events = pd.DatetimeIndex(events).dropna().unique().sort_values()
    dte = np.asarray(dte, float)
    if len(dte) != len(dates):
        raise ValueError('DTE and signal dates must align')
    if etf:
        # ETF has no issuer earnings release; not a constituent-event model.
        countdown = np.full(len(dates), 90.)
        available = np.ones(len(dates), bool)
    elif mode == 'historical_calendar':
        pos = events.searchsorted(dates, side='left')
        available = pos < len(events)
        countdown = np.full(len(dates), np.nan)
        countdown[available] = (events[pos[available]] - dates[available]).days
    else:
        # Treat same-day reports as unavailable: announcement time is unknown.
        pos = events.searchsorted(dates, side='left') - 1
        available = pos >= 0
        countdown = np.full(len(dates), np.nan)
        cadence = pd.Series(events).diff().dt.days.rolling(8, min_periods=3).median().clip(70, 120).fillna(91.)
        if available.any():
            elapsed = (dates[available] - events[pos[available]]).days
            countdown[available] = np.maximum(cadence.iloc[pos[available]].to_numpy() - elapsed, 0.)
    valid_dte = np.isfinite(dte) & (dte > 0)
    inside_available = available & valid_dte
    inside = inside_available & (countdown <= dte) & (countdown >= 0)
    if etf:
        inside[:] = False
    return pd.DataFrame({
        'earnings_days': np.where(available, np.minimum(countdown, 90.), 0.),
        'earnings_days_available': available.astype(float),
        'earnings_before_expiry': inside.astype(float),
        'earnings_before_expiry_available': inside_available.astype(float),
    }, index=dates)


def orient_legs(a, b, direction):
    """Preserve separate long/short event states; never filter a proposal."""
    direction = np.asarray(direction)
    if len(a) != len(b) or len(a) != len(direction):
        raise ValueError('Leg states and direction must align')
    out = {}
    for side in ['long', 'short']:
        take_a = direction > 0 if side == 'long' else direction < 0
        for field in ['earnings_days', 'earnings_before_expiry']:
            for suffix in ['', '_available']:
                out[f'{side}_{field}{suffix}'] = np.where(take_a, a[field+suffix], b[field+suffix])
    return pd.DataFrame(out)
