import unittest
import numpy as np
import pandas as pd
from scripts import straddle_transfer_fit as fit


class TransferFitTests(unittest.TestCase):
    def test_streamed_predictions_preserve_full_roster_and_order(self):
        frame=pd.DataFrame(dict(date=pd.to_datetime(['2020-01-01','2020-01-01','2020-02-01']),secid=[3,7,3]))
        keys=pd.MultiIndex.from_frame(frame);levels=np.array([2,2,2])
        fit.scatter_levels(keys,frame.iloc[[2,0]],np.array([3,4]),levels)
        np.testing.assert_array_equal(levels,[4,2,3])
        with self.assertRaises(ValueError):fit.scatter_levels(keys,frame.iloc[[0,0]],[3,4],levels)

    def test_dense_fit_is_unchanged_by_unfinished_future_payoffs(self):
        fit.study.configure();cal=pd.bdate_range('2018-01-01',periods=500)
        dates=cal[:400]
        meta=pd.DataFrame(dict(date=dates,end=dates+pd.offsets.BDay(35),finite_label=True,
            entry_position=np.arange(400),residual_memory=np.linspace(-.4,-.1,400),iv=.4,vix=20.))
        paths=np.random.default_rng(12).normal(size=(400,3,36))
        spec=fit.study.CFG['specifications'][0];cutoff=cal[350]
        model=fit.study.core.fit_model(meta,paths,cutoff,spec,cal)
        paths[meta.end.ge(cutoff)]=1e12
        self.assertEqual(model,fit.study.core.fit_model(meta,paths,cutoff,spec,cal))
        self.assertEqual(model['training_samples'],315)
        self.assertGreater(len(model['choices']),1)


if __name__=='__main__':unittest.main()
