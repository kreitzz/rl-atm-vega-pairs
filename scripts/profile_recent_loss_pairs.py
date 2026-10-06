"""Compare the two losing pairs with the cached roster, without changing trades."""
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path('artifacts/contracts/tail_put25_pooled')
ASSETS = Path('artifacts/contracts/uniform_smile/market/assets')
OUT = Path('reports/p25_pair_profiles')
TARGETS = {'125537__189845': '2024-05-10', '214904__215427': '2024-02-14'}
DATES = pd.to_datetime(list(TARGETS.values()))
FIELDS = ['beta', 'correlation', 'iv_mean', 'rv_mean', 'rv_ratio_a_b',
          'reverse_beta', 'beta_distance', 'iv_level_correlation',
          'iv_change_correlation']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}; cache = {}; profiles = []; snapshots = []; monthly = []
    def track(path):
        if str(path) not in sources:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    definitions = json.loads(track(ROOT/'inputs.json').read_text())['definitions']
    def iv(sid):
        if sid not in cache:
            path = track(ASSETS/f'{sid}_selection.parquet')
            table = pq.read_table(path, columns=['date','cp_flag','bucket','bs_iv'],
                                  filters=[('cp_flag','=','P'),('bucket','=','25')], use_threads=False).to_pandas()
            assert not table.date.duplicated().any()
            cache[sid] = table.set_index('date').bs_iv.sort_index().where(lambda x: x.gt(0))
        return cache[sid]
    for n,d in enumerate(definitions,1):
        pair = d['pair']; path = ROOT/pair/'states.parquet'
        if not path.exists():
            raise FileNotFoundError(path)
        x = pq.read_table(track(path),columns=['date','observed','eligible','beta','correlation',
                          'iv_mean','rv_mean','beta_available','correlation_available'],use_threads=False).to_pandas()
        assert isinstance(x.index,pd.DatetimeIndex)
        a,b = [iv(int(sid)).reindex(x.index) for sid in d['secids']]
        observed = a.notna() & b.notna()
        np.testing.assert_allclose(((a+b)/2).loc[observed], x.loc[observed,'iv_mean'],atol=1e-9,rtol=1e-9)
        x['iv_level_correlation'] = a.rolling(126,min_periods=84).corr(b)
        # Differences have to be between consecutive calendar observations;
        # no interpolation or compression over missing quote days.
        x['iv_change_correlation'] = a.diff().rolling(126,min_periods=84).corr(b.diff())
        known = x.beta_available.eq(1)&x.correlation_available.eq(1)
        x.loc[~known,['beta','correlation']] = np.nan
        x['rv_ratio_a_b'] = x.beta/x.correlation
        x['reverse_beta'] = x.correlation.pow(2)/x.beta
        x['beta_distance'] = (x.beta-1).abs()
        period = x.loc['2018-01-01':'2025-08-29'].copy()
        # Month-frozen screens count once per month; daily IV and RV do not.
        screens = period.groupby(period.index.to_period('M')).head(1)
        stats = {'pair':pair,'label':d['label'],'known_months':int(screens.beta.notna().sum()),
                 'eligible_month_fraction':float(screens.loc[screens.beta.notna(),'eligible'].mean()),
                 'observed_days':int(period.observed.sum())}
        for field in FIELDS:
            own = screens[field] if field in ['beta','correlation','rv_ratio_a_b','reverse_beta','beta_distance'] else period.loc[period.observed,field]
            available = own.dropna()
            stats[field+'_median'] = float(available.median()) if len(available) else np.nan
            stats[field+'_observations'] = len(available)
        profiles.append(stats)
        for date in DATES:
            if date not in x.index: continue
            row = x.loc[date]
            snapshots.append(dict(date=date,pair=pair,label=d['label'],observed=bool(row.observed),
                eligible=bool(row.eligible),iv_a=float(a.loc[date]),iv_b=float(b.loc[date]),
                iv_level_paired_sessions_126=int((a.notna()&b.notna()).rolling(126).sum().loc[date]),
                iv_change_paired_sessions_126=int((a.diff().notna()&b.diff().notna()).rolling(126).sum().loc[date]),
                **{field:float(row[field]) for field in FIELDS}))
        if pair in TARGETS:
            own = screens[['beta','correlation','eligible','rv_ratio_a_b','reverse_beta']].copy()
            own['pair'] = pair; own['label']=d['label']; monthly.append(own)
        if n%50==0: print(f'Compared {n}/{len(definitions)} pairs',flush=True)
    profiles = pd.DataFrame(profiles); snapshots=pd.DataFrame(snapshots)
    profiles.to_csv(OUT/'pair_history.csv',index=False)
    snapshots.to_csv(OUT/'date_universe.csv',index=False)
    pd.concat(monthly).to_csv(OUT/'target_monthly_screens.csv',index_label='date')
    comparisons=[]
    for pair,date in TARGETS.items():
        target = snapshots.loc[snapshots.pair.eq(pair)&snapshots.date.eq(date)].iloc[0]
        for scope in ['observed_roster','beta_eligible_roster']:
            group = snapshots.loc[snapshots.date.eq(date)&snapshots.observed]
            if scope=='beta_eligible_roster': group=group.loc[group.eligible]
            for field in FIELDS:
                values=group[field].dropna(); value=float(target[field])
                comparisons.append(dict(pair=pair,label=target.label,date=date,comparison=scope,
                    metric=field,value=value,group_n=len(values),group_median=float(values.median()) if len(values) else np.nan,
                    percentile=float(100*((values<value).mean()+.5*(values==value).mean())) if len(values) and np.isfinite(value) else np.nan))
    comparisons=pd.DataFrame(comparisons)
    comparisons.to_csv(OUT/'entry_comparisons.csv',index=False)
    history=[]
    for pair in TARGETS:
        target=profiles.loc[profiles.pair.eq(pair)].iloc[0]
        for field in [c for c in profiles.columns if c.endswith('_median')]+['eligible_month_fraction']:
            values=profiles[field].dropna();value=float(target[field])
            history.append(dict(pair=pair,label=target.label,metric=field,value=value,
                group_median=float(values.median()) if len(values) else np.nan,group_n=len(values),
                percentile=float(100*((values<value).mean()+.5*(values==value).mean())) if len(values) and np.isfinite(value) else np.nan))
    pd.DataFrame(history).to_csv(OUT/'history_comparisons.csv',index=False)
    track(Path(__file__))
    (OUT/'audit.json').write_text(json.dumps(dict(pairs=len(profiles),no_retraining=True,no_trade_changes=True,
        entry_snapshots_verified_against_cached_pair_states=True,
        beta_definition='126-session A-on-B return beta, lagged and frozen at month formation',
        iv_definition='selected approximately 30D 25-delta put BS IV; rolling126-session correlations require84 observations',
        history_start='2018-01-01',history_end='2025-08-29',
        percentile_definition='100 * (fraction below + half fraction equal)',
        rv_ratio_definition='beta / return correlation from the same formation sample; not20-day RV ratio',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_hashes=sources),indent=2)+'\n')
    print(comparisons.loc[comparisons.comparison.eq('beta_eligible_roster')].to_string(index=False),flush=True)
    print(pd.DataFrame(history).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
