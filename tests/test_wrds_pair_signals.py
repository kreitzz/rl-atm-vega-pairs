import numpy as np
import pandas as pd
import unittest

from scripts.test_wrds_pair_signals import availability, build_analyst_states, interval_ids, match_states, statistics


def test_availability_waits_for_latest_public_or_vendor_date():
    f=pd.DataFrame({'public':['2020-01-03','2020-01-06'],
                    'activation':['2020-01-06','2020-01-03']})
    assert availability(f,['public','activation']).tolist()==[pd.Timestamp('2020-01-07')]*2


def test_historical_links_exclude_ambiguity_and_outside_intervals():
    dates=pd.date_range('2020-01-01',periods=5)
    links=pd.DataFrame({'start':pd.to_datetime(['2020-01-01','2020-01-03']),
                        'end':pd.to_datetime(['2020-01-03','2020-01-04']),
                        'id':['A','B']})
    got=interval_ids(dates,links,None,'start','end','id')
    assert got.tolist()==['A','A',None,'B',None]


def test_asof_join_ignores_future_and_stale_features():
    dates=pd.to_datetime(['2020-01-01','2020-01-03','2020-02-01'])
    states=pd.DataFrame({'id':['A','A'], 'available':pd.to_datetime(['2020-01-02','2020-02-02']),
                         'signal':[1.,999.]})
    got=match_states(dates,['A']*3,states,'id',['signal'],7)
    assert np.isnan(got.signal.iloc[0])
    assert got.signal.iloc[1]==1.
    assert np.isnan(got.signal.iloc[2])


def test_static_pair_difference_is_not_within_pair_signal():
    dates=pd.date_range('2020-01-01',periods=30)
    f=pd.DataFrame({'pair':['A']*30+['B']*30,'date':list(dates)*2,
                    'signal':[0.]*30+[1.]*30,'target':list(range(30))+list(range(30,60)),
                    'spread':list(range(30))*2,'rv_adjusted_value':list(range(30))*2})
    result=statistics(f,'signal','target')
    assert result['spearman']>.8
    assert np.isnan(result['within_pair_equal_weight_rank_corr'])


def test_future_estimate_values_do_not_change_current_features():
    rows=[]
    for day,values in [('2020-01-01',[1.,2.,3.]),('2020-01-10',[4.,5.,6.])]:
        for analyst,value in enumerate(values):
            rows.append(dict(ticker='A',fpedats='2020-12-31',estimator=analyst,analys=analyst,
                             actdats=day,acttims='12:00',anndats=day,anntims='12:00',value=value))
    e=pd.DataFrame(rows);dates=pd.to_datetime(['2020-01-03','2020-01-17'])
    first,_=build_analyst_states(e,dates)
    e.loc[e.actdats.eq('2020-01-10'),'value']=[100.,200.,300.]
    changed,_=build_analyst_states(e,dates)
    pd.testing.assert_frame_equal(first.iloc[:1],changed.iloc[:1])
    assert first.revision_magnitude.iloc[-1]!=changed.revision_magnitude.iloc[-1] or first.forecast_dispersion.iloc[-1]!=changed.forecast_dispersion.iloc[-1]


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))
