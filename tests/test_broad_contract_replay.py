import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.contract_engine import simulate,UnresolvedPosition
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_broad_contract_replay import universe,load_actions
from tests.test_contract_engine import fixture


class BroadReplayTests(unittest.TestCase):
    def test_missing_flat_stock_blocks_entry_and_keeps_cash(self):
        q,s,a,d,c=fixture();c=replace(c,allow_missing_flat_underlying=True)
        s.loc[s.date.le(d[1])&s.secid.eq(1),['close','cfadj']]=np.nan
        nav,trades,dec=simulate(q,s,a,d,[1,2],c)
        self.assertTrue(trades.empty)
        self.assertEqual(dec.status.iloc[0],'underlying_unavailable_at_entry')
        np.testing.assert_allclose(nav.nav,c.capital)
        audit(nav,trades,s,c,[1,2])

    def test_missing_held_stock_still_fails(self):
        q,s,a,d,c=fixture();c=replace(c,allow_missing_flat_underlying=True)
        s.loc[s.date.eq(d[2])&s.secid.eq(1),'close']=np.nan
        with self.assertRaises(UnresolvedPosition):simulate(q,s,a,d,[1,2],c)

    def test_cash_roundoff_is_not_a_sharpe(self):
        nav=pd.DataFrame({'nav':[100.,100.],'cash_benchmark':[100.,100.], 'excess_return':[1e-16,-1e-16]})
        self.assertIsNone(scores(nav,100)['annualized_excess_sharpe'])

    def test_complete_roster_has_no_orientation_duplicates(self):
        rows=universe()
        self.assertEqual(len(rows),358)
        self.assertEqual(sum(r['group']=='broad' for r in rows),350)
        self.assertEqual(len({tuple(sorted(r['secids'])) for r in rows}),358)
        for r in rows:self.assertEqual(r['pair'],'__'.join(map(str,sorted(r['secids']))))

    def test_saved_actions_respect_original_screen_folds(self):
        rows=universe();cal=pd.bdate_range('2018-01-01','2025-08-29')
        for r in rows[:5]:
            actions=load_actions(r,cal)
            years=set(y for f in r['screen_years'] for y in [f,f+1])
            self.assertTrue(set(actions.index.year).issubset(years))
            self.assertTrue(actions.index.max()<=cal[-1]-pd.Timedelta(days=35))


if __name__=='__main__':unittest.main()
