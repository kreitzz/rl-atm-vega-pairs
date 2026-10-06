import unittest
import numpy as np
import pandas as pd
from scripts.p25_action_windows import fit_menu,MENUS
from scripts.run_tail_put25_residual_rl import fit_allocations
from scripts.tail_put25_rule_sizing import SIZE_CODES
from scripts.tail_put25_pooled import completed_mask


class ActionWindowTest(unittest.TestCase):
    def test_floor_kernel_exactly_matches_original_optimizer(self):
        book=np.random.default_rng(91).normal(.03,1,(400,15))
        book[:,[0,5,10]]=0
        dates=pd.bdate_range('2015-01-01',periods=len(book))
        old,details=fit_allocations(book,dates,None,np.arange(3))
        choices,log=fit_menu(book,dates,MENUS['floor'])
        np.testing.assert_array_equal(choices,[SIZE_CODES.index(int(x)) for x in old[2]])
        self.assertEqual(log['training_objective'],details['training_objective'])

    def test_optimizer_never_uses_undeclared_actions(self):
        book=np.random.default_rng(92).normal(-.02,1,(300,10));book[:,[0,5]]=0
        dates=pd.bdate_range('2015-01-01',periods=len(book))
        for menu in MENUS.values():
            choices,_=fit_menu(book,dates,menu)
            self.assertTrue(np.isin(choices,menu).all())

    def test_future_or_unfinished_outcomes_cannot_enter_short_window(self):
        dates=pd.to_datetime(['2020-01-01','2022-02-01','2022-11-01','2023-01-02'])
        ends=pd.to_datetime(['2020-02-01','2022-03-01','2023-01-05','2023-02-01'])
        actual=completed_mask(dates,ends,pd.Timestamp('2023-01-01'),years=1)
        np.testing.assert_array_equal(actual,[False,True,False,False])


if __name__=='__main__':unittest.main()
