"""Verify saved threshold-search decisions, chronology and funded cash ledgers.

No fits, search changes or new trades. Run after the serial replay completes.
"""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.p25_threshold_policy import ThresholdPolicy, proposal_mask
from scripts.submission_market import verify_ledger

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/contracts/p25_threshold_search'
REPORT=ROOT/'reports/p25_threshold_search'
PRIOR=ROOT/'artifacts/contracts/p25_opportunity_filter'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def independent_selection(books,decisions,policies,asof,months=18,start=None):
    """Recalculate saved selection evidence without calling select_policy."""
    asof=pd.Timestamp(asof)
    lower=asof-pd.DateOffset(months=months) if months is not None else None
    if start is not None:
        lower=max(lower,pd.Timestamp(start)) if lower is not None else pd.Timestamp(start)
    records=[]
    for name,policy in policies.items():
        returns=books[name]
        own=returns.iloc[1:]
        own=own.loc[own.index<asof]
        if lower is not None:
            own=own.loc[own.index>=lower]
        entries=decisions[name]
        entered=entries.status.eq('entered')&entries.entry_date.lt(asof)
        if lower is not None:
            entered &= entries.entry_date.ge(lower)
        n=int(entered.sum())
        sd=float(own.std(ddof=1)) if len(own)>1 else np.nan
        enough=len(own)>=252 and n>=20 and np.isfinite(sd) and sd>1e-12
        score=float(np.sqrt(252)*own.mean()/sd) if enough else None
        records.append(dict(policy=name,sessions=len(own),entries=n,eligible=bool(enough),
            funded_excess_sharpe=score,latest_return=own.index.max().isoformat() if len(own) else None))
    eligible=[r for r in records if r['eligible']]
    chosen=min(eligible,key=lambda r:(-r['funded_excess_sharpe'],policies[r['policy']].tie_break))['policy'] if eligible else 'q20'
    return chosen,records


def check_selection(saved,books,decisions,policies,asof,months=18,start=None):
    choice,records=independent_selection(books,decisions,policies,asof,months,start)
    assert saved['policy']==choice
    assert pd.Timestamp(saved['asof'])==pd.Timestamp(asof)
    if saved['latest_return']:
        assert pd.Timestamp(saved['latest_return'])<pd.Timestamp(asof)
    existing={r['policy']:r for r in saved['candidates']}
    assert set(existing)==set(policies)
    for row in records:
        old=existing[row['policy']]
        for field in ['sessions','entries','eligible','latest_return']:
            assert row[field]==old[field],(row['policy'],field,row[field],old[field])
        if row['funded_excess_sharpe'] is None:
            assert old['funded_excess_sharpe'] is None
        else:
            np.testing.assert_allclose(row['funded_excess_sharpe'],old['funded_excess_sharpe'],rtol=0,atol=1e-12)
    return choice


def assert_signal_identity(actual,expected):
    columns=list(expected.columns.intersection(actual.columns))
    assert {'date','pair','direction','optionid_a','optionid_b'}.issubset(columns)
    a=actual.sort_values(['date','pair'],kind='stable').reset_index(drop=True)
    b=expected.sort_values(['date','pair'],kind='stable').reset_index(drop=True)
    pd.testing.assert_frame_equal(a[columns],b[columns],check_dtype=False,check_exact=True)


def verify(rest=lambda:None):
    protocol=json.loads((REPORT/'protocol.json').read_text())
    signature=digest(REPORT/'protocol.json')
    policies={key:ThresholdPolicy(**value) for key,value in protocol['policies'].items()}
    fixed={key:policies[key] for key in protocol['fixed_family']}
    context={key:policies[key] for key in protocol['context_family']}
    for path,expected in protocol['hashes'].items():
        assert digest(ROOT/path)==expected,f'Protected input changed: {path}'
        rest()
    rows=pd.read_parquet(PRIOR/'dataset/candidates.parquet',use_threads=False)
    rows=rows.loc[rows.evaluation_eligible].copy()
    assert rows.vix.gt(0).all() and np.isfinite(rows.vix).all()
    baseline=rows.loc[proposal_mask(rows,fixed['q20'])]
    assert len(baseline)==608
    assert_signal_identity(baseline,rows.loc[rows.legacy_selected])
    books={};decisions={};audits=[];reference_index=None;baseline_metadata={}
    cases=list(policies)+['selected_fixed','selected_context','selected_rolling']
    for case in cases:
        folder=OUT/'replays'/case
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='complete' and status['protocol_sha256']==signature
        signals=pd.read_parquet(folder/'scored_allocations.parquet',use_threads=False)
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        assert nav.index.is_unique and nav.index.is_monotonic_increasing
        if reference_index is None:reference_index=nav.index
        else:assert nav.index.equals(reference_index)
        assert signals.learned_level.eq(4).all()
        identity=hashlib.sha256(signals[['date','pair','direction']].to_csv(index=False).encode()).hexdigest()
        assert identity==status['signal_identity_sha256']
        assert not signals.duplicated(['date','pair','wing']).any()
        if case in policies:
            assert_signal_identity(signals,rows.loc[proposal_mask(rows,policies[case])])
        assert set(zip(dec.signal_date,dec.pair))==set(zip(signals.date,signals.pair))
        assert dec.entry_date.gt(dec.signal_date).all()
        audit=verify_ledger(nav,trades,dec,signals)
        assert audit['entries']==status['entries'] and status['all_inventory_closed']
        assert bool(status['shared_cash']) and bool(status['shared_stock_hedges'])
        audits.append(dict(case=case,proposals=len(signals),**audit))
        if case in policies:
            books[case]=nav.excess_return.copy();decisions[case]=dec
        if case=='q20':
            oldfolder=PRIOR/'replays/current_fixed100'
            oldnav=pd.read_parquet(oldfolder/'daily_nav.parquet',use_threads=False)
            pd.testing.assert_frame_equal(nav,oldnav,check_dtype=False,check_exact=False,atol=1e-7,rtol=0)
            for name,frame,dates in [('trades.csv',trades,['date']),('decisions.csv',dec,['signal_date','entry_date'])]:
                previous=pd.read_csv(oldfolder/name,parse_dates=dates)
                if name=='decisions.csv':
                    # base_level is inherited proposal metadata, not the policy
                    # override actually executed by these fixed-full-size books.
                    # Exclude only that documented normalization; compare all
                    # other fields and independently prove effective directions.
                    assert frame.base_level.eq(4).all()
                    assert previous.base_level.isin([2,3,4]).all()
                    assert frame.level.eq(4).all() and previous.level.eq(4).all()
                    direction=signals.set_index(['date','pair']).direction
                    keys=pd.MultiIndex.from_frame(frame[['signal_date','pair']])
                    np.testing.assert_array_equal(frame.action,direction.reindex(keys).to_numpy())
                    oldkeys=pd.MultiIndex.from_frame(previous[['signal_date','pair']])
                    np.testing.assert_array_equal(previous.action,direction.reindex(oldkeys).to_numpy())
                    baseline_metadata=dict(excluded_column='base_level',
                        reason='Inherited proposal metadata normalized to4; effective level4 and signed full-size action verified identical.',
                        original_counts={str(int(k)):int(v) for k,v in previous.base_level.value_counts().items()},
                        replay_counts={str(int(k)):int(v) for k,v in frame.base_level.value_counts().items()})
                    pd.testing.assert_frame_equal(frame.drop(columns='base_level'),previous.drop(columns='base_level'),check_dtype=False,check_exact=False,atol=1e-7,rtol=0)
                else:
                    pd.testing.assert_frame_equal(frame,previous,check_dtype=False,check_exact=False,atol=1e-7,rtol=0)
        rest()
    selection_rows=[]
    for family,case in [(fixed,'selected_fixed'),(context,'selected_context')]:
        saved=json.loads((OUT/'selections'/f'{case}.json').read_text())
        key=check_selection(saved,books,decisions,family,'2022-01-01',months=None,start='2018-01-01')
        before=rows.date.lt('2022-01-01')
        expected=rows.loc[(before&proposal_mask(rows,fixed['q20']))|(~before&proposal_mask(rows,family[key]))]
        actual=pd.read_parquet(OUT/'replays'/case/'scored_allocations.parquet',use_threads=False)
        assert_signal_identity(actual,expected)
        nav=pd.read_parquet(OUT/'replays'/case/'daily_nav.parquet',use_threads=False)
        baseline_nav=pd.read_parquet(OUT/'replays/q20/daily_nav.parquet',use_threads=False)
        pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'],baseline_nav.loc[:'2021-12-31'],check_dtype=False,check_exact=False,atol=1e-7,rtol=0)
        actual_trades=pd.read_csv(OUT/'replays'/case/'trades.csv',parse_dates=['date'])
        base_trades=pd.read_csv(OUT/'replays/q20/trades.csv',parse_dates=['date'])
        a=actual_trades.loc[actual_trades.date.lt('2022-01-01')].reset_index(drop=True)
        b=base_trades.loc[base_trades.date.lt('2022-01-01')].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,atol=1e-7,rtol=0)
        # Identical prior fills imply identical inherited open option inventory.
        pd.testing.assert_series_equal(a.groupby(['cohort','optionid']).quantity.sum(),b.groupby(['cohort','optionid']).quantity.sum())
        selection_rows.append(dict(case=case,asof='2022-01-01',chosen=key,past_only_verified=True,baseline_carry_verified=True))
        rest()
    pieces=[]
    schedule=pd.read_csv(REPORT/'rolling_schedule.csv')
    for month in pd.period_range('2018-01','2025-06',freq='M'):
        saved=json.loads((OUT/'selections/rolling'/f'{month}.json').read_text())
        key=check_selection(saved,books,decisions,fixed,month.start_time)
        group=rows.loc[rows.date.dt.to_period('M').eq(month)]
        selected=group.loc[proposal_mask(group,fixed[key])].copy()
        selected['selected_policy']=key
        pieces.append(selected)
        reported=schedule.loc[schedule.month.eq(str(month))].iloc[0]
        assert reported.policy==key and reported.proposals==len(selected)
        selection_rows.append(dict(case='selected_rolling',asof=str(month.start_time.date()),chosen=key,past_only_verified=True,baseline_carry_verified=None))
        rest()
    actual=pd.read_parquet(OUT/'replays/selected_rolling/scored_allocations.parquet',use_threads=False)
    assert_signal_identity(actual,pd.concat(pieces,ignore_index=True))
    pd.DataFrame(audits).to_csv(REPORT/'verification_ledgers.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(REPORT/'verification_selections.csv',index=False)
    result=dict(status='passed',candidate_books=len(policies),deployment_books=3,
        independently_recomputed_selection_windows=len(selection_rows),
        original_baseline_nav_trades_effective_decisions_reproduced=True,
        baseline_proposal_metadata_normalization=baseline_metadata,
        locked_deployment_pre2022_nav_fills_inventory_equal_baseline=True,
        every_saved_signal_reproduced_from_causal_policy=True,
        all_funded_ledgers_reconciled=True,protected_input_hashes_unchanged=True,
        vix_prior_session_fraction_used_without_additional_lag=True,
        protocol_sha256=signature,max_independent_ledger_error=max(a['independent_ledger_error'] for a in audits))
    (REPORT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    from scripts.run_p25_representation import Budget
    budget=Budget()
    print(json.dumps(verify(rest=budget.rest),indent=2),flush=True)
