import unittest
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scripts.tail_feature_exploration import (return_shape, surface_states, pair_features,
    scheduled_nonoverlap, ranked_statistics)


class TailFeatureExplorationTest(unittest.TestCase):
    def test_return_shapes_are_causal_and_match_physical_moments(self):
        dates = pd.bdate_range('2018-01-01', periods=340)
        r = np.random.default_rng(12).standard_t(7, 340)*.01
        stock = pd.DataFrame(dict(date=dates, close=100*np.exp(np.cumsum(r)), cfadj=1.))
        original = return_shape(stock, dates)
        self.assertAlmostEqual(original.realized_skew63.iloc[-1], skew(r[-63:], bias=False))
        self.assertAlmostEqual(original.realized_kurt252.iloc[-1], kurtosis(r[-252:], fisher=True, bias=False))
        np.testing.assert_allclose((original.down_semivariance_share63+original.up_semivariance_share63).dropna(), 1.)
        future = stock.copy(); future.loc[300:, 'close'] *= 100
        pd.testing.assert_frame_equal(original.iloc[:300], return_shape(future, dates).iloc[:300])

    def selection(self):
        dates = pd.bdate_range('2018-01-01', periods=3); rows = []
        for date in dates:
            for tenor in [14, 30, 45]:
                for cp in ['C', 'P']:
                    for bucket, iv, delta in [('atm',.30,.5), ('25',.33 if cp=='C' else .39,.25), ('10',.40 if cp=='C' else .47,.1)]:
                        rows.append(dict(date=date, tenor=tenor, cp_flag=cp, bucket=bucket, bs_iv=iv, bs_delta=delta*(1 if cp=='C' else -1),
                            expiry=pd.Timestamp('2018-02-16') if tenor==30 else date+pd.Timedelta(days=tenor),
                            optionid=len(rows), dte=tenor, bid=1., ask=1.05, bs_vega=.1))
        return dates, pd.DataFrame(rows)

    def test_smile_uses_observed_puts_and_same_expiry(self):
        dates, selection = self.selection(); surface = surface_states(selection, dates)
        self.assertAlmostEqual(surface['P','25'].wing_premium.iloc[0], .3)
        self.assertAlmostEqual(surface['C','25'].risk_reversal25.iloc[0], -.2)
        self.assertAlmostEqual(surface['C','25'].butterfly25.iloc[0], .2)
        changed = selection.copy()
        bad = changed.date.eq(dates[0])&changed.tenor.eq(30)&changed.cp_flag.eq('P')&changed.bucket.eq('25')
        changed.loc[bad, 'expiry'] += pd.Timedelta(days=7)
        mismatch = surface_states(changed, dates)
        self.assertTrue(np.isnan(mismatch['C','25'].risk_reversal25.iloc[0]))
        self.assertTrue(np.isnan(mismatch['P','25'].wing_premium.iloc[0]))

    def test_nonoverlap_selection_does_not_skip_missing_outcomes(self):
        dates = pd.to_datetime(['2020-01-01','2020-01-02','2020-02-01'])
        ends = pd.to_datetime(['2020-01-20','2020-01-21','2020-02-20'])
        np.testing.assert_array_equal(scheduled_nonoverlap(dates, ends, [True]*3), [True,False,True])

    def test_partial_rank_control_removes_mechanical_association(self):
        rng = np.random.default_rng(123); n = 400
        control = rng.normal(size=n)
        data = pd.DataFrame(dict(pair='p', label='A/B', date=pd.bdate_range('2018-01-01',periods=n),
            signal=control+.05*rng.normal(size=n), directional_edge=control+.05*rng.normal(size=n), control=control))
        stats, _ = ranked_statistics(data, 'signal', controls=['control'])
        self.assertGreater(stats['rho'], .9)
        self.assertLess(abs(stats['partial']), .2)


if __name__=='__main__': unittest.main()
