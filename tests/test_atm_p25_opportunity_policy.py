import unittest
import numpy as np
import pandas as pd
from scripts import atm_p25_opportunity_policy as s


class OpportunityPolicyTests(unittest.TestCase):
    def test_future_proposals_do_not_change_earlier_features(self):
        dates=pd.bdate_range('2020-01-01',periods=80)
        a=pd.DataFrame(dict(date=dates,iv=.3,rv21=.2,residual_memory=-.05))
        p=pd.DataFrame(dict(date=dates[::3],rank_strength=2.,oriented_vrp_gap=.1))
        first=s.opportunity_features(dates,a,p)
        changed_a=a.copy();changed_p=p.copy()
        changed_a.loc[changed_a.date>=dates[40],['iv','rv21','residual_memory']]=50.
        changed_p.loc[changed_p.date>=dates[40],['rank_strength','oriented_vrp_gap']]=50.
        second=s.opportunity_features(dates,changed_a,changed_p)
        pd.testing.assert_frame_equal(first.iloc[:40],second.iloc[:40])
        np.testing.assert_allclose(first.atm_iv_rv_gap,1/3)


if __name__=='__main__':unittest.main()
