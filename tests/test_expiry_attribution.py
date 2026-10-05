import unittest
import numpy as np
import pandas as pd
from scripts.contract_engine import PreparedMarket, simulate
from scripts.diagnose_expiry_signal_gap import leg_path
from tests.test_expiry_holding import fixture


class ExpiryAttributionTests(unittest.TestCase):
    def test_reconstructs_daily_hedges_and_gross_with_dividend_and_fallback(self):
        q, s, a, dates, cfg = fixture()
        # Vary both spot paths, with a declared cash dividend and missing Greeks.
        for sid in (1, 2):
            mask = s.secid.eq(sid)
            s.loc[mask, 'close'] += np.sin(np.arange(mask.sum()) * .7) * sid
        q.loc[q.date.eq(dates[3]), ['delta', 'vega', 'bs_iv']] = np.nan
        divs = pd.DataFrame([dict(secid=1, ex_date=dates[4],
                                  declare_date=dates[0], amount=.1)])
        nav, trades, _ = simulate(q, s, a, dates, [1, 2], cfg, divs)
        market = PreparedMarket(q, s)
        asset = pd.DataFrame({'iv': 40., 'forward_rv': 40.}, index=dates)
        gross = pd.Series(0., index=dates)
        fallbacks = 0
        for sid, leg in trades.groupby('secid'):
            record, hedge, pnl = leg_path(leg, market, dates, divs, asset)
            np.testing.assert_allclose(hedge.reindex(dates, fill_value=0),
                                       nav[f'hedge_{sid}'], atol=1e-9)
            gross = gross.add(pnl, fill_value=0)
            fallbacks += record['fallbacks']
        expected = (nav.pnl - nav.interest + nav.option_execution_cost
                    + nav.hedge_cost + nav.stock_borrow_cost)
        np.testing.assert_allclose(gross, expected, atol=1e-8, rtol=0)
        self.assertEqual(fallbacks, 2)
        self.assertNotEqual(nav.dividend_cash.sum(), 0)


if __name__ == '__main__':
    unittest.main()
