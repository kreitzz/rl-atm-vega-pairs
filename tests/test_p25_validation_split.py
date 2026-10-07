import unittest
import numpy as np
import pandas as pd
from scripts.run_p25_validation_split import validation_score,period_metrics


class SplitTests(unittest.TestCase):
    def test_future_returns_cannot_change_validation_ranking(self):
        dates=pd.bdate_range('2018-01-02','2025-08-29')
        x=pd.DataFrame({'excess_return':np.random.default_rng(5).normal(.001,.01,len(dates))},index=dates)
        before=validation_score(x)
        x.loc['2022-01-01':,'excess_return']=1e9
        self.assertEqual(before,validation_score(x))

    def test_incomplete_validation_calendar_rejected(self):
        x=pd.DataFrame({'excess_return':[.1,.2]},index=pd.to_datetime(['2021-12-30','2021-12-31']))
        with self.assertRaises(ValueError):validation_score(x)

    def test_period_includes_first_holdout_return_and_rebases_cash_growth(self):
        index=pd.to_datetime(['2021-12-31','2022-01-03','2022-12-30'])
        nav=pd.DataFrame(dict(nav=[120.,121.,130.],cash_benchmark=[110.,111.,115.],
            excess_return=[100.,.01,-.02],capital_reserve=[0.,10.,20.],post_trade_risk_breach=[False,False,True]),index=index)
        decisions=pd.DataFrame(dict(status=['entered','entered'],entry_date=pd.to_datetime(['2021-12-31','2022-01-03']),pair=['a','b']))
        result=period_metrics(nav,decisions,'2022-01-01','2022-12-31')
        self.assertEqual(result['profit_above_cash'],5.)
        self.assertEqual(result['entries'],1)
        self.assertEqual(result['post_entry_risk_breach_days'],1)
        self.assertAlmostEqual(result['annualized_excess_sharpe'],np.sqrt(252)*np.mean([.01,-.02])/np.std([.01,-.02],ddof=1))
        years=(index[-1]-index[0]).days/365.25
        self.assertAlmostEqual(result['annualized_growth_relative_to_cash_pct'],100*(((130/115)/(120/110))**(1/years)-1))


if __name__=='__main__':unittest.main()
