import unittest
from collections import OrderedDict
import sqlite3
import numpy as np
import pandas as pd
from scripts.run_p25_ungated_search import portfolio_score,fit_menu,LazyDiskMarket
from scripts.run_p25_earnings import fit_sizes


class SizingMenuTests(unittest.TestCase):
    def book(self):
        rng=np.random.default_rng(7)
        a=rng.normal(.2,1,252);b=rng.normal(-.2,1,252)
        return np.column_stack([np.zeros(252),a*.25,a*.5,a*.75,a,
                                np.zeros(252),b*.25,b*.5,b*.75,b])

    def test_original_optimizer_is_exact(self):
        a,details=fit_menu(self.book(),[2,3,4])
        b,reference=fit_sizes(self.book())
        np.testing.assert_array_equal(a,b)
        self.assertEqual(details,reference)

    def test_book_sharpe_matches_explicit_aggregation(self):
        book=self.book();choice=np.array([2,3])
        direct=book[:,2]+book[:,8]
        expected=np.sqrt(252)*direct.mean()/direct.std(ddof=1)
        self.assertAlmostEqual(float(portfolio_score(book,choice)[0]),expected,places=12)

    def test_skip_can_exclude_negative_leaf(self):
        choices,_=fit_menu(self.book(),[0,2,3,4])
        self.assertEqual(choices[1],0)
        self.assertGreater(choices[0],0)

    def test_zero_exposure_has_zero_score(self):
        self.assertEqual(float(portfolio_score(self.book(),[0,0])[0]),0.)

    def test_invalid_action_menu_rejected(self):
        with self.assertRaises(ValueError):fit_menu(self.book(),[0,2,5])


class DiskStockTests(unittest.TestCase):
    def setUp(self):
        self.market=object.__new__(LazyDiskMarket)
        self.market.stockdb=sqlite3.connect(':memory:')
        self.market.stockdb.execute('CREATE TABLE stocks(date TEXT,secid INTEGER,close REAL,cfadj REAL)')
        self.market.stockdb.executemany('INSERT INTO stocks VALUES(?,?,?,?)',[
            ('2025-07-17',101322,100.,2.),('2025-07-18',102968,150.,1.),
            ('2025-07-18',101322,999.,999.),('2025-07-21',102968,200.,1.)])
        self.market.stock_days=OrderedDict()

    def tearDown(self):self.market.stockdb.close()

    def test_corporate_conversion_uses_chevron_and_prior_hess_factor(self):
        spot,adjustment=self.market.stock(pd.Timestamp('2025-07-18'),101322)
        self.assertAlmostEqual(spot,150*1.025)
        self.assertEqual(adjustment,2.)

    def test_future_stock_prices_cannot_change_current_lookup(self):
        date=pd.Timestamp('2025-07-18')
        before=self.market.stock(date,102968)
        self.market.stockdb.execute("UPDATE stocks SET close=-1 WHERE date='2025-07-21'")
        self.market.stock_days.clear()
        self.assertEqual(before,self.market.stock(date,102968))


if __name__=='__main__':unittest.main()
