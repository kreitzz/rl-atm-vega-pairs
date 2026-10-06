"""Summarize completed tail diagnostics; exploratory labels, no policy fitting."""
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    report = Path('reports/tail_feature_exploration')
    correlations = pd.read_csv(report / 'correlations.csv', dtype={'bucket': str})
    tails = correlations[correlations.bucket.ne('atm')]
    candidates = tails.pivot(index=['cp', 'bucket', 'feature'], columns=['sample', 'period'],
                             values=['rho', 'partial', 'pairs', 'observations', 'ci_low', 'ci_high'])
    candidates.columns = ['_'.join(c) for c in candidates.columns]
    candidates.reset_index().to_csv(report / 'feature_comparison.csv', index=False)

    frames = [pd.read_parquet(p) for p in
              Path('artifacts/contracts/tail_feature_exploration/pairs').glob('*/P10.parquet')]
    data = pd.concat(frames, ignore_index=True)
    data = data[data.resolved & data.wing_iv_gap_pct.notna()].copy()
    data['quintile'] = np.minimum(np.floor(data.wing_iv_gap_pct * 5).astype(int), 4) + 1
    # A descriptive, post-exploration illustration, not a trained strategy or holdout.
    # Lower historical spread quintile: long A/short B; upper quintile: inverse.
    data = data[data.quintile.isin([1, 5])].copy()
    data['net_per_vega'] = np.where(data.quintile.eq(1), data.long_net, data.short_net)
    summaries, years, pairs = [], [], []
    for period, mask in [('early_complete', data.end.lt('2022-01-01')),
                         ('later', data.date.ge('2022-01-01'))]:
        group = data[mask]
        for sample, h in [('all', group), ('nonoverlap', group[group.nonoverlap]),
                          ('tight_delta', group[group.delta_error_max.le(.03)])]:
            per_pair = h.groupby(['pair', 'label']).net_per_vega.agg(['count', 'mean', 'sum'])
            per_pair = per_pair.reset_index().rename(columns={'count': 'observations',
                'mean': 'mean_net_per_vega', 'sum': 'sum_net_per_vega_labels'})
            pairs.append(per_pair.assign(period=period, sample=sample))
            valid = per_pair[per_pair.observations.ge(8)]
            summaries.append(dict(period=period, sample=sample, observations=len(h),
                mean_net_per_vega=h.net_per_vega.mean(), win_rate=h.net_per_vega.gt(0).mean(),
                pairs_with_8_observations=len(valid), positive_pairs_with_8_observations=int(valid.mean_net_per_vega.gt(0).sum()),
                equal_pair_mean_net_per_vega=valid.mean_net_per_vega.mean()))
            per_year = h.groupby(h.date.dt.year).net_per_vega.agg(['count', 'mean']).reset_index()
            years.append(per_year.rename(columns={'date': 'year', 'count': 'observations',
                         'mean': 'mean_net_per_vega'}).assign(period=period, sample=sample))
    pd.DataFrame(summaries).to_csv(report / 'put10_spread_illustration.csv', index=False)
    pd.concat(years, ignore_index=True).to_csv(report / 'put10_spread_illustration_by_year.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(report / 'put10_spread_illustration_by_pair.csv', index=False)


if __name__ == '__main__':
    main()
