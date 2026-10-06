"""Maturity remapping and calendar reporting without changing policy decisions."""
import numpy as np
import pandas as pd

WINDOWS = [('2018_2020', '2018-01-01', '2020-12-31'),
           ('2021_2023', '2021-01-01', '2023-12-31'),
           ('2024_2025_partial', '2024-01-01', '2025-08-29')]


def remap(anchor, choices, wing):
    """Preserve proposals, even when the selected contract is missing."""
    rows=anchor.copy()
    rows['wing']=wing
    for side in ['a','b']:
        rows['optionid_'+side]=np.nan
        for sid,idx in rows.groupby('secid_'+side).groups.items():
            selected=choices[(int(sid),wing)]
            assert not selected.date.duplicated().any()
            rows.loc[idx,'optionid_'+side]=selected.set_index('date').optionid.reindex(pd.DatetimeIndex(rows.loc[idx,'date'])).to_numpy()
    return rows


def common_entries(rows, last_date, runoff=110):
    return rows.loc[rows.date.le(pd.Timestamp(last_date)-pd.Timedelta(days=runoff))].reset_index(drop=True)


def period_stats(nav, decisions, start, end):
    """Continuous-book calendar slices, including already open inventory."""
    g=nav.loc[start:end]
    r=g.excess_return
    if len(g) and g.index[0]==nav.index[0]:
        r=r.iloc[1:]
    benchmark_change=nav.cash_benchmark.diff().fillna(0).reindex(g.index)
    entered=decisions.loc[decisions.status.eq('entered') & decisions.entry_date.between(start,end)]
    return dict(sessions=len(r),entries=len(entered),active_pairs=int(entered.pair.nunique()),
        active_sessions=int(g.active_cohorts.gt(0).sum()),
        annualized_excess_sharpe=float(np.sqrt(252)*r.mean()/r.std()) if r.std()>1e-12 else None,
        net_excess_pnl=float((g.pnl-benchmark_change).sum()),
        funded_return_percent=float(100*((1+g.return_).prod()-1)))
