"""Backward joins of dated transcript scores onto saved option-pair states."""
import numpy as np
import pandas as pd
from scripts.run_atm65_employee_sentiment import LINKS, REVIEWS
from scripts.test_wrds_pair_signals import interval_ids, match_states, timestamp_columns
from scripts.test_wrds_alternative_pairs import numeric_ids

FIELDS = ['call_negative_change', 'call_qa_uncertainty', 'call_analyst_negative']
ADDED = [col for field in FIELDS for col in ['transcript_'+field+'_gap', 'transcript_'+field+'_available']]


def add_transcripts(panel, first, second):
    result = panel.copy()
    for field in FIELDS:
        gap = first[field].reindex(panel.index)-second[field].reindex(panel.index)
        observed = np.isfinite(gap)
        result['transcript_'+field+'_gap'] = gap.where(observed,0.).astype(float)
        result['transcript_'+field+'_available'] = observed.astype(float)
    pd.testing.assert_frame_equal(result[panel.columns],panel)
    return result


def leg_states(dates, assets):
    o = timestamp_columns(pd.read_parquet(LINKS/'option_links.parquet'),['sdate','edate'])
    o = o[o.score<=1].copy();o.edate=o.edate.fillna(pd.Timestamp('2026-01-01'))
    c = timestamp_columns(pd.read_parquet(LINKS/'company_links.parquet'),['linkdt','linkenddt'])
    c.linkdt=c.linkdt.fillna(pd.Timestamp('1900-01-01'));c.linkenddt=c.linkenddt.fillna(pd.Timestamp('2026-01-01'))
    c.gvkey=c.gvkey.map(lambda x:str(x).strip().zfill(6))
    links=timestamp_columns(pd.read_parquet(REVIEWS/'ciq_links.parquet'),['symbolstartdate','symbolenddate'])
    links.gvkey=links.gvkey.map(lambda x:str(x).strip().zfill(6))
    links.symbolstartdate=links.symbolstartdate.fillna(pd.Timestamp('1900-01-01'))
    links.symbolenddate=links.symbolenddate.fillna(pd.Timestamp('2026-01-01'))
    states=pd.read_parquet(REVIEWS/'transcripts_states.parquet')
    states.companyid=pd.Series(numeric_ids(states.companyid)).astype(np.int64)
    result={}
    for sid in assets:
        perm=interval_ids(dates,o[o.secid.eq(sid)],None,'sdate','edate','permno')
        gvkey=np.full(len(dates),None,object);companyid=gvkey.copy()
        for value in pd.Series(perm).dropna().unique():
            mask=perm==value
            gvkey[mask]=interval_ids(dates[mask],c[c.permno.eq(value)],None,'linkdt','linkenddt','gvkey')
        for value in pd.Series(gvkey).dropna().unique():
            mask=gvkey==value
            companyid[mask]=interval_ids(dates[mask],links[links.gvkey.eq(value)],None,
                'symbolstartdate','symbolenddate','relatedcompanyid')
        result[sid]=match_states(dates,numeric_ids(companyid),states,'companyid',FIELDS,180)
    return result
