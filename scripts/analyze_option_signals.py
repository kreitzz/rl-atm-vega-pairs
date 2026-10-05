"""As-of signal versus subsequent net option/hedge payoff, before policy selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.option_flex_ablation import FLEX, REPORT, definitions, settings, NEW_FEATURES, EXTRA, FEATURES, screen_dates
from scripts.option_flex.policy import ACTIONS
from scripts.run_contract_walkforward import digest


def correlation(x, y, method='pearson'):
    mask = np.isfinite(x) & np.isfinite(y)
    a, b = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(a) < 20 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return np.nan
    if method == 'spearman':
        a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def nonoverlap(dates, ends):
    """Greedy chronological thinning, independently for each pair/type."""
    chosen = np.zeros(len(dates), bool); last = pd.Timestamp.min
    for i, (date, end) in enumerate(zip(dates, ends)):
        if date > last: chosen[i] = True; last = end
    return chosen


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--common-only', action='store_true'); args = parser.parse_args()
    prefix = 'common_' if args.common_only else ''
    REPORT.mkdir(parents=True, exist_ok=True); records = []; outcomes = []; coverage = []
    cfg = settings(); stop = pd.Timestamp('2025-08-29')-pd.Timedelta(days=cfg.max_dte+5)
    for d in definitions():
        folder = FLEX/d['pair']
        if not (folder/'labels.npz').exists(): continue
        panel = pd.read_parquet(folder/'states.parquet')
        features = NEW_FEATURES+[c if c != 'commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
        features += [c for c in panel if c.startswith(('profit_', 'count_'))]+FEATURES+['vix_previous_close']
        features = list(dict.fromkeys(c for c in features if not c.endswith('_available')))
        wanted = (panel.index >= '2018-01-01') & (panel.index <= stop) & screen_dates(d, panel.index) & panel.eligible
        with np.load(folder/'labels.npz', allow_pickle=False) as z:
            daily = z['daily']
            common = np.ones(len(panel), bool)
            if args.common_only:
                for c in ['CC', 'PP']:
                    ix = [ACTIONS.index((c, a)) for a in [-1., 1.]]
                    common &= np.isfinite(daily[:, ix]).all(axis=(1, 2)) & (z[c+'_vegas'][:, [0, -1]] > 0).all(axis=1)
            for combo in ['CC', 'PP']:
                cols = [ACTIONS.index((combo, a)) for a in [-1., 1.]]
                vega = z[combo+'_vegas'][:, [0, -1]]; ends = pd.DatetimeIndex(z[combo+'_ends'])
                complete = np.isfinite(daily[:, cols]).all(axis=(1, 2)) & (vega > 0).all(axis=1)
                use = np.asarray(wanted) & complete & common
                coverage.append(dict(pair=d['pair'], label=d['label'], combination=combo,
                                     screened_dates=int(wanted.sum()), entered_resolved_opportunities=int(use.sum())))
                if not use.any(): continue
                assert (ends[use] > panel.index[use]).all()
                net = np.divide(daily[:, cols].sum(axis=2)*cfg.entry_vega_per_leg, vega,
                                out=np.full_like(vega, np.nan), where=vega > 0)
                out = pd.DataFrame(dict(date=panel.index[use], end=ends[use], pair=d['pair'], label=d['label'], combination=combo,
                                        short_net=net[use, 0], long_net=net[use, 1],
                                        directional_edge=(net[use, 1]-net[use, 0])/2))
                out['nonoverlap'] = nonoverlap(out.date, out.end); outcomes.append(out)
                own = panel[features].copy()
                for days in [91, 365]:
                    for side in ['short', 'long']:
                        own[f'own_profit_{side}_{days}d'] = panel[f'{combo}_profit_{side}_{days}d']
                    own[f'memory_edge_{days}d'] = panel[f'{combo}_profit_long_{days}d']-panel[f'{combo}_profit_short_{days}d']
                for feature in own:
                    values = own[feature]
                    if feature+'_available' in panel: values = values.where(panel[feature+'_available'].astype(bool))
                    # Current signal is ranked only against its own trailing observed signals.
                    percentile = values.rolling(252, min_periods=63).rank(pct=True)
                    r = out.copy(); r['feature'] = feature; r['signal'] = values.loc[use].to_numpy()
                    r['percentile'] = percentile.loc[use].to_numpy(); records.append(r)
    data = pd.concat(records, ignore_index=True); out = pd.concat(outcomes, ignore_index=True)
    pd.DataFrame(coverage).to_csv(REPORT/(prefix+'signal_coverage.csv'), index=False)
    # Licensed row-level derived data stays in ignored artifacts; reports are aggregate.
    root = Path('artifacts/contracts/option_signal_diagnostics'+('_common' if args.common_only else '')); root.mkdir(parents=True, exist_ok=True)
    out.to_parquet(root/'outcomes.parquet', index=False)
    (root/'inputs.json').write_text(json.dumps(dict(source=digest(FLEX/'inputs.json'), code=digest(Path(__file__)),
        targets='Net standalone expiry P&L / actual mean initial leg vega. Fixed capital, before portfolio risk limits.',
        common_only=args.common_only, period=['2018-01-01', str(stop.date())]), indent=2)+'\n')
    summary = []; by_pair = []; yearly = []; bins = []
    for (combo, feature), g in data.groupby(['combination', 'feature'], sort=False):
        for target in ['directional_edge', 'long_net', 'short_net']:
            h = g.loc[np.isfinite(g.signal) & np.isfinite(g[target])].copy()
            dx = h.signal-h.groupby('pair').signal.transform('mean')
            dy = h[target]-h.groupby('pair')[target].transform('mean')
            pair_corr = []
            for pair, v in h.groupby('pair'):
                if len(v) < 40: continue
                rho = correlation(v.signal, v[target], 'spearman')
                if np.isfinite(rho): pair_corr.append(rho)
                by_pair.append(dict(combination=combo, feature=feature, target=target, pair=pair,
                                    label=v.label.iloc[0], observations=len(v), spearman=rho))
            thin = h[h.nonoverlap]
            summary.append(dict(combination=combo, feature=feature, target=target, observations=len(h), pairs=h.pair.nunique(),
                pearson=correlation(h.signal, h[target]), spearman=correlation(h.signal, h[target], 'spearman'),
                within_pair_pearson=correlation(dx, dy), equal_pair_mean_spearman=np.mean(pair_corr) if pair_corr else np.nan,
                median_pair_spearman=np.median(pair_corr) if pair_corr else np.nan,
                nonoverlap_observations=len(thin), nonoverlap_spearman=correlation(thin.signal, thin[target], 'spearman')))
            for year, v in h.groupby(h.date.dt.year):
                yearly.append(dict(combination=combo, feature=feature, target=target, year=year, observations=len(v),
                                   spearman=correlation(v.signal, v[target], 'spearman')))
            ranked = h.dropna(subset=['percentile']).copy()
            ranked['quintile'] = np.minimum(np.floor(ranked.percentile*5).astype(int), 4)+1
            for bucket, v in ranked.groupby('quintile'):
                bins.append(dict(combination=combo, feature=feature, target=target, quintile=int(bucket), observations=len(v),
                                 mean_net_vega_units=v[target].mean(), median_net_vega_units=v[target].median(),
                                 fraction_positive=v[target].gt(0).mean()))
    result = pd.DataFrame(summary); result.to_csv(REPORT/(prefix+'signal_correlations.csv'), index=False)
    pd.DataFrame(by_pair).to_csv(REPORT/(prefix+'signal_correlations_by_pair.csv'), index=False)
    pd.DataFrame(yearly).to_csv(REPORT/(prefix+'signal_correlations_by_year.csv'), index=False)
    pd.DataFrame(bins).to_csv(REPORT/(prefix+'signal_quintiles.csv'), index=False)
    selected = ['spread', 'rv_adjusted_value', 'spread_reversion', 'beta', 'correlation', 'vix_previous_close',
                'memory_edge_91d', 'memory_edge_365d', 'surface_downside_skew_difference',
                'surface_smile_curvature_difference', 'surface_term_slope_difference', 'surface_term_bend_difference']
    lines = ['# Signal versus subsequent net pair payoff', '',
        ('Common-date sample: both CC and PP must have observed, resolved entries on the same pair/date.' if args.common_only
         else 'Each contract type uses its own available opportunities; CC/PP sample composition differs. See the common-date study for a like-for-like comparison.'), '',
        'All screened opportunities with observed, resolved standalone entries are included, before the policy chooses trades. CC and PP are evaluated separately. Positive directional payoff means long vega in the first name and short vega in the second was better than the reverse. Outcomes include entry costs, daily stock hedging, dividends, borrowing and financing through expiry, normalized by actual initial leg vega.', '',
        'These are descriptive associations on already inspected 2018–2025 research dates. The standalone training trajectories do not apply portfolio capacity/risk limits. Missing/unresolved outcomes are disclosed in coverage. Overlapping ~30-day outcomes and common market exposures make observations dependent; no IID p-values or causal claims are made. The nonoverlapping check thins each pair/type chronologically but does not remove cross-pair dependence. Quintiles use trailing 252-observation signal ranks known at the signal date, not future outcome ranks.', '',
        '| Type | Signal | Pooled Spearman | Within-pair Pearson | Equal-pair mean Spearman | Nonoverlap Spearman |',
        '|---|---|---:|---:|---:|---:|']
    for combo in ['CC', 'PP']:
        for feature in selected:
            r = result[(result.combination == combo) & (result.feature == feature) & (result.target == 'directional_edge')].iloc[0]
            lines.append(f'| {combo} | {feature} | {r.spearman:.3f} | {r.within_pair_pearson:.3f} | {r.equal_pair_mean_spearman:.3f} | {r.nonoverlap_spearman:.3f} |')
    lines += ['', f'[Full correlations](option_flex_ablation/{prefix}signal_correlations.csv) · [By year](option_flex_ablation/{prefix}signal_correlations_by_year.csv) · [By pair](option_flex_ablation/{prefix}signal_correlations_by_pair.csv) · [Signal quintiles](option_flex_ablation/{prefix}signal_quintiles.csv) · [Coverage](option_flex_ablation/{prefix}signal_coverage.csv)', '',
              '[All available dates](option_signal_relationships.md) · [Common dates](option_signal_relationships_common.md)', '']
    REPORT.with_name('option_signal_relationships'+('_common' if args.common_only else '')+'.md').write_text('\n'.join(lines))
    print(f'Analyzed {len(out):,} pair/type opportunities, {out.pair.nunique()} pairs; {out.nonoverlap.sum():,} nonoverlapping pair/type cohorts', flush=True)


if __name__ == '__main__': main()
