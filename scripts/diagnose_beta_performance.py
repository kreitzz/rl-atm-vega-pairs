"""Descriptive entry-beta associations on existing expiry trades; no refitting."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path('reports/expiry_signal_gap')
    trades = pd.read_csv(root/'cohorts.csv', parse_dates=['cohort'])
    trades = trades[trades.policy.isin(['trained_rl', 'trained_nn'])].copy()
    chunks = []
    for pair, group in trades.groupby('pair'):
        screen = pd.read_csv(Path('artifacts/contracts/expiry_trained')/pair/'beta_screen.csv',
                             index_col=0, parse_dates=[0])
        group = group.join(screen[['beta', 'correlation', 'eligible']], on='cohort', validate='many_to_one')
        assert group.eligible.all() and group.beta.notna().all()
        chunks.append(group)
    trades = pd.concat(chunks, ignore_index=True)
    trades['beta_distance'] = (trades.beta-1).abs()
    trades['gross_per_vega'] = trades.actual_gross/trades.mean_entry_vega
    trades['beta_bucket'] = pd.cut(trades.beta_distance, [-1e-9, .05, .10, .15, .25],
                                    labels=['0–0.05', '0.05–0.10', '0.10–0.15', '0.15–0.25'])
    rows = []
    for (policy, bucket), group in trades.groupby(['policy', 'beta_bucket'], observed=True):
        rows.append(dict(policy=policy, distance_from_one=bucket, trades=len(group),
            pairs=group.pair.nunique(), gross_pnl=group.actual_gross.sum(),
            gross_per_entry_vega=group.actual_gross.sum()/group.mean_entry_vega.sum(),
            median_gross_per_vega=group.gross_per_vega.median(),
            gross_win_rate=group.actual_gross.gt(0).mean()))
    buckets = pd.DataFrame(rows)
    pairs = pd.read_csv(root/'pairs.csv')
    profiles = trades.groupby(['policy', 'pair']).agg(
        mean_beta_distance=('beta_distance', 'mean'), mean_beta=('beta', 'mean'),
        mean_correlation=('correlation', 'mean'), entries=('cohort', 'size')).reset_index()
    profiles = profiles.merge(pairs.drop(columns='entries'), on=['policy', 'pair'], validate='one_to_one')
    sharpes = []
    for row in profiles.itertuples():
        model = 'rl_sized' if row.policy == 'trained_rl' else 'nn_fixed'
        nav = pd.read_parquet(Path('artifacts/contracts/expiry_trained')/row.pair/model/'daily_nav.parquet', columns=['excess_return'])
        r = nav.excess_return.iloc[1:]
        sharpes.append(np.sqrt(252)*r.mean()/r.std() if r.std()>1e-12 else np.nan)
    profiles['annualized_excess_sharpe'] = sharpes
    associations = []
    for policy, group in profiles.groupby('policy'):
        t = trades[trades.policy.eq(policy)]
        associations.append(dict(policy=policy, active_pairs=len(group), trades=len(t),
            pair_distance_vs_net_sharpe=group.mean_beta_distance.corr(group.annualized_excess_sharpe, method='spearman'),
            pair_correlation_vs_net_sharpe=group.mean_correlation.corr(group.annualized_excess_sharpe, method='spearman'),
            trade_distance_vs_gross_per_vega=t.beta_distance.corr(t.gross_per_vega, method='spearman')))
    associations = pd.DataFrame(associations)
    trades.to_csv(root/'beta_trade_profiles.csv', index=False)
    profiles.to_csv(root/'beta_pair_profiles.csv', index=False)
    buckets.to_csv(root/'beta_buckets.csv', index=False)
    associations.to_csv(root/'beta_associations.csv', index=False)
    print(buckets.to_string(index=False))
    print(associations.to_string(index=False))


if __name__ == '__main__':
    main()
