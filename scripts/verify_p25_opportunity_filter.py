"""Final causality tests, saved-decision checks, and source integrity audit."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_opportunity_filter as run
from tests import test_p25_opportunity_dataset as dataset_tests


def main():
    budget=run.Budget()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'tests.test_p25_opportunity_models','tests.test_analyze_p25_opportunity_filter'])
    suite.addTest(unittest.FunctionTestCase(dataset_tests.test_middle_rank_direction_features_are_reconstructed))
    suite.addTest(unittest.FunctionTestCase(dataset_tests.test_future_completion_cannot_change_signal_eligibility))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise AssertionError('Focused safeguards failed')
    budget.rest()
    protocol=json.loads((run.REPORT/'protocol.json').read_text())
    for path,expected in protocol['protected_sources'].items():
        assert run.base.digest(run.ROOT/path)==expected,path
    for path,expected in protocol['scripts'].items():
        assert run.base.digest(run.ROOT/'scripts'/path)==expected,path
    assert run.base.digest(run.OUT/'dataset/candidates.parquet')==protocol['dataset_sha256']
    windows=0;checked_predictions=0
    for case in run.CASES:
        scores=pd.read_parquet(run.OUT/'predictions'/f'{case}.parquet')
        selected=scores.loc[scores.selected]
        assert not selected.duplicated(['date','pair']).any()
        for month in pd.period_range('2018-01','2025-06',freq='M'):
            stamp=str(month.start_time.date())
            meta=json.loads((run.OUT/'models'/case/f'{stamp}.json').read_text())
            assert pd.Timestamp(meta['latest_training_outcome'])<month.start_time
            for fold in meta['folds']:
                assert pd.Timestamp(fold['latest_training_outcome'])<pd.Timestamp(fold['start'])
            windows+=1
        # Check serialization and production scoring in three separated periods.
        for month in ['2018-03','2022-06','2024-11']:
            own=scores.loc[scores.date.dt.to_period('M').eq(pd.Period(month,freq='M'))]
            assert len(own)>0
            rule=joblib.load(run.OUT/'models'/case/f'{month}-01.joblib')
            actual=rule.select(own)
            np.testing.assert_allclose(actual.prediction,own.prediction,rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(actual.selected,own.selected)
            checked_predictions+=len(own)
            budget.rest()
        if case.startswith('rank'):
            dec=pd.read_csv(run.OUT/'replays'/case/'decisions.csv',parse_dates=['signal_date'])
            combined=scores[['date','pair','selected']].merge(dec,left_on=['date','pair'],right_on=['signal_date','pair'],validate='one_to_one')
            assert len(combined)==len(scores)==len(dec)==4744
            np.testing.assert_array_equal(combined.level,np.where(combined.selected,4,0))
            assert combined.loc[~combined.selected,'status'].eq('flat').all()
    assert windows==360
    run.base.write(run.REPORT/'verification.json',dict(focused_tests=result.testsRun,all_passed=True,
        monthly_windows_checked=windows,serialized_prediction_rows_reproduced=checked_predictions,
        funded_decisions_match_saved_selections=True,protected_sources_unchanged=True,peak_rss_mib=budget.peak))
    print(f'Verified {result.testsRun} tests, {windows} windows and {checked_predictions} serialized predictions.',flush=True)


if __name__=='__main__':main()
