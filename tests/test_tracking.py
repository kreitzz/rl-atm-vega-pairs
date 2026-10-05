import unittest

import numpy as np
import pandas as pd

from scripts.tracking_baselines import Band, Settings, delta_point, interp_target, metrics, simulate


def row(date, oid, dte, delta, iv, vega=10.0, half_spread=0.5, monthly=True):
    date = pd.Timestamp(date)
    return dict(ticker='AMD', date=date, optionid=oid, exdate=date + pd.Timedelta(days=dte), dte=dte,
                strike=100.0, iv=iv, delta=delta, vega=vega, half_spread=half_spread, mid=5.0,
                open_interest=1000.0, monthly=monthly, candidate=True)


def book(days, iv_shift=0.0, index_shift=0.0, drop=None):
    """Two expiries (20 and 40 days) x two strikes (0.4 and 0.6 delta), IV moving in lockstep."""
    rows, idx = [], []
    for k, d in enumerate(days):
        for oid, dte, delta in [(1, 20, .4), (2, 20, .6), (3, 40, .4), (4, 40, .6)]:
            if drop and (k, oid) == drop:
                continue
            rows.append(row(d, oid, dte - k, delta, 50 + k * iv_shift))
        idx.append(50 + k * index_shift)
    return pd.DataFrame(rows), pd.DataFrame({'AMD': idx}, index=pd.DatetimeIndex(days))


DAYS = pd.bdate_range('2024-01-02', periods=4)
S = Settings(target_vega=1000.0, commission=0.0)


class TrackingTests(unittest.TestCase):
    def test_delta_interpolation_hits_target(self):
        r = pd.DataFrame({'optionid': [1, 2, 3], 'delta': [.3, .45, .65]})
        w = delta_point(r, .5)
        self.assertAlmostEqual(sum(w.values()), 1)
        self.assertAlmostEqual(w[2] * .45 + w[3] * .65, .5)

    def test_interp_target_lands_on_30_days_and_50_delta(self):
        chain, _ = book(DAYS[:1])
        w = interp_target(chain, S)
        c = chain.set_index('optionid')
        self.assertAlmostEqual(sum(w.values()), 1)
        self.assertAlmostEqual(sum(w[o] * c.loc[o, 'dte'] for o in w), 30)
        self.assertAlmostEqual(sum(w[o] * c.loc[o, 'delta'] for o in w), .5)

    def test_entry_cost_is_half_spread_per_contract_over_target_vega(self):
        chain, index = book(DAYS[:2])
        d = simulate(chain, index, 'AMD', 'interp', Band(99, 1, 9), S)
        # full target vega 1000 at vega 10 = 100 contracts, each paying half-spread 0.5
        self.assertAlmostEqual(d.cost.iloc[0], 100 * 0.5 / 1000)

    def test_lockstep_iv_tracks_perfectly(self):
        chain, index = book(DAYS, iv_shift=1.0, index_shift=1.0)
        d = simulate(chain, index, 'AMD', 'interp', Band(99, 1, 9), S)
        self.assertTrue(np.allclose(d.tracking_error, 0))
        self.assertAlmostEqual(metrics(d)['te_daily_rmse'], 0)

    def test_tracking_error_measures_the_gap(self):
        chain, index = book(DAYS, iv_shift=1.0, index_shift=2.0)
        d = simulate(chain, index, 'AMD', 'interp', Band(99, 1, 9), S)
        self.assertTrue(np.allclose(d.tracking_error, -1.0))

    def test_band_zero_rebalances_band_wide_does_not(self):
        chain, index = book(DAYS)          # contracts age, so maturity drifts off 30
        daily = simulate(chain, index, 'AMD', 'interp', Band(), S)
        wide = simulate(chain, index, 'AMD', 'interp', Band(99, 1, 9), S)
        self.assertTrue(daily.traded.iloc[1:].all())
        self.assertFalse(wide.traded.iloc[1:].any())

    def test_missing_held_quote_is_closed_and_not_measured(self):
        chain, index = book(DAYS, drop=(1, 3))
        d = simulate(chain, index, 'AMD', 'interp', Band(99, 1, 9), S)
        self.assertTrue(d.gap.iloc[0])                 # t=0 -> t=1 cannot be measured
        self.assertEqual(d.forced_closes.iloc[1], 1)
        self.assertTrue(np.isnan(d.tracking_error.iloc[0]))


if __name__ == '__main__':
    unittest.main()
