"""Render the completed, predeclared entry-filter experiment without model selection.

Consumes saved CSV analysis outputs. Does not fit, tune, replay, or promote a rule.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/p25_opportunity_filter'
LABELS = {'current_fixed100': 'Current filter', 'rank_all_fixed100': 'All extreme-rank',
          'rank_ridge': 'Rank + ridge', 'rank_spline': 'Rank + spline',
          'broad_ridge': 'Broad ridge', 'broad_spline': 'Broad spline'}
FUNDED = ['current_fixed100', 'rank_all_fixed100', 'rank_ridge', 'rank_spline']
LEARNED = ['broad_ridge', 'broad_spline', 'rank_ridge', 'rank_spline']
SHADOW = ['current_fixed100', 'rank_all_fixed100'] + LEARNED


def pct(value):
    return f'{100*value:.1f}%' if np.isfinite(value) else 'undefined'


def interval_sentence(row):
    return f'{row.point:+.3f} [{row.lower95:+.3f}, {row.upper95:+.3f}]'


def verdict(intervals):
    """Report both predeclared challengers; never choose the larger Sharpe."""
    positive, lower, inconclusive = [], [], []
    for case in ['rank_ridge', 'rank_spline']:
        own = intervals.loc[intervals.case.eq(case) & intervals.period.eq('evaluation')]
        if len(own) != 2 or set(own.block_sessions) != {63, 126}:
            raise ValueError(f'Missing interval sensitivity for {case}')
        if own.lower95.gt(0).all():
            positive.append(LABELS[case])
        elif own.upper95.lt(0).all():
            lower.append(LABELS[case])
        else:
            inconclusive.append(LABELS[case])
    if positive:
        lead = ('The funded experiment provides exploratory evidence of a better filter for '
                + ', '.join(positive) + ' within the existing extreme-rank opportunity set.')
    else:
        lead = ('The funded experiment does not establish that either contextual filter improves '
                'on the current filter at fixed 100% reference sizing.')
    detail = []
    if lower:
        detail.append('Both block-length intervals are below zero for '+', '.join(lower)+'.')
    if inconclusive:
        detail.append('The interval evidence is inconclusive across the block-length checks for '+', '.join(inconclusive)+'.')
    return lead, ' '.join(detail)


def read_tables(report):
    paths = {name: report/f'{name}.csv' for name in [
        'shadow_periods', 'shadow_coverage', 'shadow_sharpe_intervals', 'shadow_pairs',
        'shadow_annual', 'prediction_bins', 'fits']}
    paths.update({f'funded_{name}': report/'funded'/f'{name}.csv' for name in [
        'periods', 'annual', 'sharpe_intervals', 'pair_admissions']})
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('Experiment outputs are not complete: '+', '.join(missing))
    tables={name: pd.read_csv(path) for name, path in paths.items()}
    for name in ['concentration','pair_periods']:
        path=report/'funded'/f'{name}.csv'
        if path.exists():
            tables[f'funded_{name}']=pd.read_csv(path)
    return tables


def selected_coverage(artifact_root):
    """Read-only coverage audit; coverage never changes the model selections."""
    with zipfile.ZipFile(ROOT/'data/p25_memory_ablation_experiment.zip') as archive:
        prepared=json.loads(archive.read('artifacts/contracts/p25_memory_ablation/prepared.json'))
    ids=set(prepared['quote_ids'])
    result=[]
    for case in SHADOW:
        frame=pd.read_parquet(artifact_root/'shadow'/f'{case}_selected.parquet', columns=[
            'pair','date','direction','optionid_a','optionid_b','path_available','finite_label','archived_path_status'], use_threads=False)
        present=frame.optionid_a.isin(ids)&frame.optionid_b.isin(ids)
        missing=~present
        unresolved=~frame.path_available
        result.append(dict(case=case, selected=len(frame), missing_contract_pair_dates=int(missing.sum()),
            missing_contract_pairs=int(frame.loc[missing,'pair'].nunique()),
            unresolved_paths=int(unresolved.sum()), unresolved_pairs=int(frame.loc[unresolved,'pair'].nunique()),
            completed_training_outcomes=int(frame.finite_label.sum()),
            stored_all_actions_zero=int(frame.archived_path_status.eq('all_actions_zero_no_entry').sum()),
            stored_full_size_zero_other_nonzero=int(frame.archived_path_status.eq('full_size_zero_other_actions_nonzero').sum())))
    return pd.DataFrame(result)


def activity_diagnostics(artifact_root, fits):
    winners=pd.read_parquet(artifact_root/'prediction_winners.parquet', columns=[
        'case','date','iv_rank','selected','prediction'],use_threads=False)
    winners['month']=pd.to_datetime(winners.date).dt.to_period('M').astype(str)
    winners['rank_bin']=pd.cut(winners.iv_rank, bins=[0,.2,.4,.6,.8,1.],
        labels=['0–20%','20–40%','40–60%','60–80%','80–100%'],include_lowest=True)
    acceptance=[];activity=[];inactive=[]
    for case in LEARNED:
        own=winners.loc[winners.case.eq(case)]
        models=fits.loc[fits.case.eq(case)]
        active=int(own.loc[own.selected].month.nunique())
        activity.append(dict(case=case,months_with_candidates=int(own.month.nunique()),
            months_with_selection=active,months_without_selection=90-active,
            insufficient_training=int(models.status.eq('insufficient_training_rows').sum()),
            insufficient_validation=int(models.status.eq('insufficient_chronological_validation').sum()),
            validation_skip_all=int(models.status.eq('validation_skip_all').sum()),
            fitted_months=int(models.status.eq('fitted').sum()),
            distinct_finite_hurdles=int(models.hurdle.dropna().nunique()),
            minimum_finite_hurdle=float(models.hurdle.min()),maximum_finite_hurdle=float(models.hurdle.max())))
        for label, group in own.groupby('rank_bin',observed=False):
            acceptance.append(dict(case=case,rank_bin=label,candidates=len(group),
                selected=int(group.selected.sum()),acceptance_rate=float(group.selected.mean()) if len(group) else np.nan,
                mean_prediction=float(group.prediction.mean()) if len(group) else np.nan))
        for r in models.loc[models.status.ne('fitted')].itertuples():
            stamp=str(pd.Timestamp(r.cutoff).date())
            meta=json.loads((artifact_root/'models'/case/f'{stamp}.json').read_text())
            fold_detail='; '.join(f'{f["start"][:10]}: train={f["train_rows"]}, validation={f["validation_rows"]}, usable={f["usable"]}' for f in meta.get('folds',[]))
            inactive.append(dict(case=case,cutoff=stamp,status=r.status,training_pair_dates=r.training_pair_dates,fold_detail=fold_detail))
    return pd.DataFrame(activity),pd.DataFrame(acceptance),pd.DataFrame(inactive)


def make_figure(t, report):
    os.environ['MPLCONFIGDIR'] = str(report/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    colors = ['#8096a8', '#176a91']
    for ax, cases, frame, periods, metric in [
        (axes[0], FUNDED, t['funded_periods'], ['validation','evaluation'], 'annualized_excess_sharpe'),
        (axes[1], SHADOW, t['shadow_periods'], ['earlier','later'], 'observed_shadow_sharpe')]:
        x = np.arange(len(cases))
        for i, period in enumerate(periods):
            rows = frame.loc[frame.period.eq(period)].set_index('case')
            values = [float(rows.loc[c, metric]) for c in cases]
            bars = ax.bar(x+(i-.5)*.36, values, width=.36, color=colors[i],
                          label=['2018–2021', '2022–Aug 2025'][i])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.axhline(0, color='#59636c', linewidth=.8)
        ax.set_xticks(x, [LABELS[c].replace(' + ', '\n+ ').replace('All extreme-rank','All extreme\nrank') for c in cases], rotation=25, ha='right')
        ax.margins(y=.22)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title('Funded portfolio: replaces profitability screen', loc='left', fontsize=12, fontweight='bold', y=1.11, pad=3)
    axes[0].set_ylabel('Daily cash-excess return Sharpe')
    axes[0].text(0, 1.03, 'Rank cutoff retained · fixed reference size · $0.03 commission', transform=axes[0].transAxes, fontsize=9, color='#56616b')
    axes[1].set_title('Archived standalone paths: discovery diagnostic', loc='left', fontsize=12, fontweight='bold', y=1.11, pad=3)
    axes[1].set_ylabel('Observed daily P&L mean / SD × √252')
    axes[1].text(0, 1.03, 'Broad rules remove rank cutoff · $0.65 commission', transform=axes[1].transAxes, fontsize=9, color='#56616b')
    fig.suptitle('Contextual opportunity filters: two distinct test scopes', x=.06, ha='left', fontsize=15, fontweight='bold')
    unresolved = t['shadow_coverage'].loc[t['shadow_coverage'].unresolved_paths.gt(0)]
    note = ('Right: unresolved selected paths: '+', '.join(f'{LABELS[r.case]} ({int(r.unresolved_paths)})' for r in unresolved.itertuples())+'. Other plotted selections have finite stored paths. '
            if len(unresolved) else 'Right: standalone paths ignore shared portfolio admission and capital constraints. ')
    fig.text(.06, .025, note+'\nCompare policies within each panel; the panels are not equivalent portfolio performance tests. Historical results are exploratory.', fontsize=9, color='#56616b')
    fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.82, wspace=.25)
    fig.savefig(report/'comparison.png', dpi=170, facecolor='white')
    plt.close(fig)


def generate(report=REPORT, rest=lambda: None):
    report = Path(report)
    t = read_tables(report)
    selected=selected_coverage(ROOT/'artifacts/contracts/p25_opportunity_filter')
    selected.to_csv(report/'selected_coverage.csv',index=False)
    protocol = json.loads((report/'protocol.json').read_text())
    fits = t['fits']
    if len(fits) != 360 or any(fits.groupby('case').size().reindex(LEARNED) != 90):
        raise ValueError('Expected all 360 monthly fitting/decision windows before producing final findings')
    fitted_models=int(fits.status.isin(['fitted','validation_skip_all']).sum())
    activity,acceptance,inactive=activity_diagnostics(ROOT/'artifacts/contracts/p25_opportunity_filter',fits)
    activity.to_csv(report/'model_activity.csv',index=False)
    acceptance.to_csv(report/'rank_acceptance.csv',index=False)
    inactive.to_csv(report/'inactive_models.csv',index=False)
    later = t['funded_periods'].loc[t['funded_periods'].period.eq('evaluation')].set_index('case')
    earlier = t['funded_periods'].loc[t['funded_periods'].period.eq('validation')].set_index('case')
    lead, evidence = verdict(t['funded_sharpe_intervals'])
    lines = ['P25 contextual opportunity-filter experiment', '', lead, evidence,
        'The broader question—replacing the fixed volatility-rank cutoff itself—remains unresolved at the funded portfolio level because the available quote bundle cannot replay the full broad candidate set.', '',
        'What was tested',
        'Four predeclared pipelines: ridge and smooth spline regression, each trained either on all eligible pair-dates in both directions or on extreme-rank candidates in the inherited direction. Each chooses skip or the fixed full reference size; no position-sizing optimization was performed.',
        'Broad scope: 11,977 evaluation pair-dates across 112 pairs, including 7,233 middle-rank pair-dates rejected by the old 20/80 rule. Rank scope: 4,744 extreme-rank candidates. The current filter retains 608 of those candidates.',
        'Monthly models use the preceding 18 months of completed outcomes. Two chronological three-month validation folds choose among three regularization strengths using prediction error, then choose a nonnegative score hurdle from a small predeclared set. The hurdle objective aggregates lifetime payoff by signal date; it is an unfunded surrogate, not funded portfolio Sharpe.',
        f'360 monthly fitting/decision windows were processed: 90 per pipeline, January 2018–June 2025. {fitted_models} windows produced fitted predictive models (including models whose validation rule skips all); the remaining windows skipped because training or chronological validation was insufficient. Each available monthly model and hurdle is frozen. Candidate features vary daily, so the implied entry boundary changes with conditions even while the monthly hurdle is frozen.',
        'The target is conditional on historically completed, executable training outcomes. Known failed/no-entry and unresolved outcomes are excluded from model fitting, not reclassified as zero-return training trades. Evaluation candidates include failed-entry cases and are selected without consulting their future outcomes; execution eligibility is then checked by the replay. This avoids using future outcome availability to screen the evaluation universe, while leaving a training-to-execution distribution difference.', '',
        'Training capacity (overlapping directions, pairs and dates are not independent observations)',
        'Pipeline | Training direction rows min / median / max | Distinct pair-dates min / median / max | Skip-all or insufficient months']
    capacity = []
    for case in LEARNED:
        own = fits.loc[fits.case.eq(case)]
        row = dict(case=case, models=len(own), minimum_rows=int(own.training_rows.min()), median_rows=float(own.training_rows.median()), maximum_rows=int(own.training_rows.max()),
            minimum_pair_dates=int(own.training_pair_dates.min()), median_pair_dates=float(own.training_pair_dates.median()), maximum_pair_dates=int(own.training_pair_dates.max()), nonfitted_months=int(own.status.ne('fitted').sum()))
        capacity.append(row)
        lines.append(f'{LABELS[case]} | {row["minimum_rows"]} / {row["median_rows"]:g} / {row["maximum_rows"]} | {row["minimum_pair_dates"]} / {row["median_pair_dates"]:g} / {row["maximum_pair_dates"]} | {row["nonfitted_months"]}')
    lines += ['', 'Model activity and changing hurdles',
        'Pipeline | Months with selections / 90 | Insufficient training | Insufficient validation | Validation skip-all | Distinct finite hurdles | Finite hurdle range']
    for r in activity.itertuples():
        lines.append(f'{LABELS[r.case]} | {r.months_with_selection} / 90 | {r.insufficient_training} | {r.insufficient_validation} | {r.validation_skip_all} | {r.distinct_finite_hurdles} | {r.minimum_finite_hurdle:.3f} to {r.maximum_finite_hurdle:.3f}')
    lines += ['Months without entries may reflect unavailable validation outcomes, a validation skip-all rule, scores below the fitted hurdle, or portfolio admission. They are not evidence that the model confidently identified an absence of opportunities. See inactive_models.csv for each unavailable model and chronological fold counts.']
    for r in inactive.head(6).itertuples():
        lines.append(f'Inactive example: {LABELS[r.case]} {r.cutoff}, {r.status}, {int(r.training_pair_dates)} outer training pair-dates; {r.fold_detail}.')
    lines += ['', 'Acceptance across fixed volatility-rank bins (all evaluation dates)',
        'Pipeline | Rank bin | Candidate pair-dates | Selected | Acceptance']
    for r in acceptance.itertuples():
        lines.append(f'{LABELS[r.case]} | {r.rank_bin} | {r.candidates} | {r.selected} | {pct(r.acceptance_rate)}')
    lines += ['Broad rules may choose either direction or skip within every rank bin. Rank rules score only the inherited extreme-rank direction. Acceptance within bins is descriptive; it does not establish that a rank bin is profitable.']
    lines += ['', 'Funded results: same $10 million account, fixed 100% reference sizing',
        'Pipeline | Earlier Sharpe | Later Sharpe | Later profit above cash | Later entries | Later entry pairs | Later mean reserve | Later max cash-relative drawdown']
    for case in FUNDED:
        a, b = earlier.loc[case], later.loc[case]
        lines.append(f'{LABELS[case]} | {a.annualized_excess_sharpe:.3f} | {b.annualized_excess_sharpe:.3f} | ${b.profit_above_cash:,.0f} | {int(b.entries)} | {int(b.active_entry_pairs)} | {b.average_reserve_pct:.2f}% | {b.max_cash_relative_drawdown_pct:.2f}%')
    lines += ['All funded arms retain the old extreme-rank cutoff. Learned arms replace the historical-profit screen. All extreme-rank removes that screen without learning a replacement. Decisions include skips and subsequent risk/eligibility rejection; selected proposals and funded entries are different counts.',
        'The reference is the current filter with constant 100% sizing (later Sharpe approximately 1.055). The previously reported 1.263 belongs to the separate tree/CEM sizing strategy and is not the reference for this opportunity-selection comparison.', '',
        'Paired later Sharpe differences: contextual filter minus current fixed-size filter',
        'Pipeline | Block sessions | Difference [95% interval]']
    intervals = t['funded_sharpe_intervals']
    for case in ['rank_ridge','rank_spline']:
        for r in intervals.loc[intervals.case.eq(case)&intervals.period.eq('evaluation')].sort_values('block_sessions').itertuples():
            lines.append(f'{LABELS[case]} | {int(r.block_sessions)} | {interval_sentence(r)}')
    lines += ['Intervals use 2,000 paired circular calendar-block resamples. They condition on fitted strategies, preserve simultaneous calendar shocks, and do not adjust for earlier strategy research or multiple comparisons. Neither the earlier nor later history is a fresh holdout. These intervals cannot certify a deployable edge.', '',
        'Annual funded stability: daily cash-excess Sharpe / profit above cash / entries']
    for year in range(2018,2026):
        items = []
        for case in FUNDED:
            r = t['funded_annual'].loc[t['funded_annual'].case.eq(case)&t['funded_annual'].year.eq(year)].iloc[0]
            items.append(f'{LABELS[case]} {r.annualized_excess_sharpe:.2f} / ${r.profit_above_cash:,.0f} / {int(r.entries)}')
        lines.append(f'{year}: '+'; '.join(items))
    lines += ['', 'Funded stability interpretation']
    for case in ['rank_ridge','rank_spline']:
        own=t['funded_annual'].loc[t['funded_annual'].case.eq(case)]
        year2020=own.loc[own.year.eq(2020)].iloc[0]
        recent=own.loc[own.year.ge(2022)]
        losses=recent.loc[recent.profit_above_cash.lt(0),'year'].astype(int).tolist()
        earlier_profit=float(earlier.loc[case,'profit_above_cash'])
        concentration_note=('The 2020 contribution exceeds the entire earlier-period profit, so the other earlier years lose money in aggregate.'
                            if year2020.profit_above_cash>earlier_profit and earlier_profit>0 else 'Assess the individual earlier years rather than relying only on their aggregate.')
        lines.append(f'{LABELS[case]}: 2020 profit above cash ${year2020.profit_above_cash:,.0f}, versus ${earlier_profit:,.0f} across all of 2018–2021. {concentration_note} Later losses occur in {len(losses)} of {len(recent)} annual observations: '+', '.join(map(str,losses))+'.')
    lines += ['The 2025 observation ends in August and is not a full calendar year. The annual pattern is consistent with unstable predictive usefulness, rather than a durable improvement demonstrated across periods; it does not establish the causal mechanism behind the deterioration.']
    lines += ['', 'Funded pair entry concentration (coverage, not P&L attribution)']
    for case in FUNDED:
        own = t['funded_pair_admissions'].loc[t['funded_pair_admissions'].case.eq(case)&t['funded_pair_admissions'].period.eq('evaluation')]
        entries = own.entries.sort_values(ascending=False)
        share = float(entries.iloc[:3].sum()/entries.sum()) if entries.sum() else np.nan
        lines.append(f'{LABELS[case]}: {int((entries>0).sum())} pairs entered; top three pairs account for {pct(share)} of entries.')
    attributed=[]
    if 'funded_concentration' in t and 'funded_pair_periods' in t:
        concentration=t['funded_concentration'].loc[t['funded_concentration'].period.eq('evaluation')]
        attributed=[case for case in FUNDED if case in set(concentration.case) and case in set(t['funded_pair_periods'].case)]
        if attributed:
            lines += ['', 'Reconciled funded pair P&L: later period',
                'Pipeline | Positive pairs | Negative pairs | Net pair P&L | Shared financing versus cash | Top-three share of positive pair profit']
            for case in attributed:
                r=concentration.loc[concentration.case.eq(case)].iloc[0]
                lines.append(f'{LABELS[case]} | {int(r.positive_pairs)} | {int(r.negative_pairs)} | ${r.net_pair_pnl:,.0f} | ${r.shared_financing_vs_cash:,.0f} | {pct(r.top3_share_positive_profit)}')
            lines += ['Pair net P&L includes attributed option/hedge outcomes and execution/borrow costs. Shared financing remains separate. These attributed cases reconcile daily to the portfolio change in profit above cash; the pair table and financing together explain the account, not independent pair portfolios.']
    unavailable=[LABELS[c] for c in FUNDED if c not in attributed]
    if unavailable:
        lines.append('Funded pair-level net P&L is unavailable for '+', '.join(unavailable)+'. Entry breadth alone cannot establish breadth of profitability for those cases.')
    lines += ['',
        'Broad opportunity discovery: archived standalone-path diagnostic',
        'These figures aggregate stored holding-period P&L paths at a fixed reference 450 vega. They have no funded shared-capital denominator and are not ROI. They omit portfolio admission, hedge netting and shared financing. Do not compare their Sharpe or dollar levels directly with the funded table.',
        'Pipeline | Earlier observed-path Sharpe | Later observed-path Sharpe | Selected pair-dates | Middle-rank selections | Finite stored paths | Unresolved selected paths']
    shadow_earlier = t['shadow_periods'].loc[t['shadow_periods'].period.eq('earlier')].set_index('case')
    shadow_later = t['shadow_periods'].loc[t['shadow_periods'].period.eq('later')].set_index('case')
    coverage = t['shadow_coverage'].set_index('case')
    for case in SHADOW:
        a,b,c = shadow_earlier.loc[case], shadow_later.loc[case], coverage.loc[case]
        lines.append(f'{LABELS[case]} | {a.observed_shadow_sharpe:.3f} | {b.observed_shadow_sharpe:.3f} | {int(c.selected)} | {int(c.selected_middle_rank)} | {int(c.finite_paths)} | {int(c.unresolved_paths)}')
    complete_learned=[LABELS[c] for c in LEARNED if int(coverage.loc[c,'unresolved_paths'])==0]
    if complete_learned:
        lines.append('All selected stored paths are finite for '+', '.join(complete_learned)+'. Their shadow findings are not caused by excluding unresolved selected paths. Finite stored paths still do not provide the missing funded contract quotes.')
    lines += ['Any case with unresolved selected paths is an observed-component diagnostic, not complete strategy performance. Missing paths are never imputed as zero; excluded outcomes can change the result in either direction. Finite stored no-entry paths are preserved as stored zeros, while those no-entry observations are not used as completed training outcomes.', '',
        'Coverage of the actual selected broad opportunities',
        'Pipeline | Selected | Missing at least one contract ID | Pairs with missing IDs | Unresolved selected paths / pairs | Completed training outcomes | Stored all-actions-zero paths']
    for r in selected.loc[selected.case.str.startswith('broad')].itertuples():
        lines.append(f'{LABELS[r.case]} | {r.selected} | {r.missing_contract_pair_dates} | {r.missing_contract_pairs} | {r.unresolved_paths} / {r.unresolved_pairs} | {r.completed_training_outcomes} | {r.stored_all_actions_zero}')
    lines += ['Contract ID presence is only an initial coverage check, not proof of day-by-day quote availability. These coverage counts were audited after selection and never used to filter the prediction universe.', '',
        'Later observed-path difference versus current filter (126-session blocks; incomplete paths remain a limitation)']
    for r in t['shadow_sharpe_intervals'].loc[t['shadow_sharpe_intervals'].block_sessions.eq(126)].itertuples():
        lines.append(f'{LABELS[r.case]}: {interval_sentence(r)}')
    lines += ['', 'Later shadow pair-profit concentration (standalone observed paths only)']
    concentration = []
    for case in SHADOW:
        own = t['shadow_pairs'].loc[t['shadow_pairs'].case.eq(case)&t['shadow_pairs'].period.eq('later')]
        values = own.observed_shadow_net_pnl
        positive = values.clip(lower=0).sort_values(ascending=False)
        share = float(positive.iloc[:3].sum()/positive.sum()) if positive.sum() else np.nan
        row = dict(case=case, positive_pairs=int(values.gt(0).sum()),negative_pairs=int(values.lt(0).sum()),top3_share_positive_profit=share)
        concentration.append(row)
        lines.append(f'{LABELS[case]}: {row["positive_pairs"]} positive and {row["negative_pairs"]} negative pairs; top three supply {pct(share)} of positive observed shadow profit.')
    lines += ['', 'Annual observed-path diagnostic Sharpe (not funded returns)']
    for year in range(2018,2026):
        own = t['shadow_annual'].loc[t['shadow_annual'].year.eq(year)].set_index('case')
        lines.append(f'{year}: '+'; '.join(f'{LABELS[c]} {own.loc[c,"observed_shadow_sharpe"]:.2f}' for c in LEARNED))
    lines += ['', 'Prediction quality on completed candidate outcomes',
        'Payoff units are archived net dollars per reference entry vega. Score bins are retrospective descriptive quantiles, not replay entry thresholds. Overlapping trades prevent treating these counts as independent samples.',
        'Pipeline | Group | Outcomes | Mean predicted | Mean realized | 5th percentile realized | Positive outcomes']
    for case in LEARNED:
        own = t['prediction_bins'].loc[t['prediction_bins'].case.eq(case)&t['prediction_bins'].period.eq('evaluation')]
        labels = ['selected','rejected'] + sorted(own.loc[own.group.str.startswith('bin_'),'group'].tolist())
        for group in labels:
            part = own.loc[own.group.eq(group)]
            if part.empty:
                continue
            r=part.iloc[0]
            lines.append(f'{LABELS[case]} | {group} | {int(r.observations)} | {r.mean_prediction:.3f} | {r.mean_target:.3f} | {r.target_p05:.3f} | {pct(r.positive_fraction)}')
    lines += ['', 'Interpretation of the observed prediction failure',
        'The selected-trade forecasts are poorly calibrated in the later period: positive predicted payoffs do not translate into comparable realized payoffs. This is measured on completed candidate outcomes under the archived cost convention, not on the funded subset.']
    for case in LEARNED:
        allbins=t['prediction_bins'].loc[t['prediction_bins'].case.eq(case)]
        later_selected=allbins.loc[allbins.period.eq('evaluation')&allbins.group.eq('selected')]
        earlier_selected=allbins.loc[allbins.period.eq('validation')&allbins.group.eq('selected')]
        if len(later_selected) and len(earlier_selected):
            a,b=earlier_selected.iloc[0],later_selected.iloc[0]
            lines.append(f'{LABELS[case]}: later selected mean forecast {b.mean_prediction:+.3f}, realized {b.mean_target:+.3f}; earlier selected realized mean {a.mean_target:+.3f}.')
    for case in ['rank_ridge','rank_spline']:
        a,b=shadow_earlier.loc[case],shadow_later.loc[case]
        if a.observed_shadow_sharpe>0 and b.observed_shadow_sharpe<0:
            lines.append(f'{LABELS[case]} reverses from positive earlier shadow Sharpe ({a.observed_shadow_sharpe:.3f}) to negative later shadow Sharpe ({b.observed_shadow_sharpe:.3f}); the earlier result did not persist.')
    for case in ['broad_ridge','broad_spline']:
        own=t['prediction_bins'].loc[t['prediction_bins'].case.eq(case)&t['prediction_bins'].period.eq('evaluation')]
        bottom,top=own.loc[own.group.eq('bin_1')],own.loc[own.group.eq('bin_5')]
        if len(bottom) and len(top):
            lines.append(f'{LABELS[case]}: later pooled top-score quintile mean outcome {top.iloc[0].mean_target:+.3f}, bottom quintile {bottom.iloc[0].mean_target:+.3f}. This descriptive ranking contrast is not the same as the frozen monthly hurdle policy.')
    lines += ['The broad pooled top-versus-bottom score contrast leaves open whether some ranking information exists, but the intermediate bins are not a reliable monotonic payoff ladder. Pooled bins mix models and market periods and use the whole evaluation score distribution. They do not validate changing the entry rule to a post-hoc quintile cutoff.',
        'The observed result is lack of stable payoff calibration and weak or reversing selected-opportunity performance. Overfitting, changing market relationships, noisy correlated labels, and the surrogate/funded objective mismatch are possible explanations, not causes established by this experiment. No model or threshold was modified in response to these diagnostics.']
    missing = int(protocol['broad_funded_blocker']['missing_at_least_one_contract'])
    lines += ['', 'Limits and what the experiment answers',
        f'1. A full broad funded replay is unavailable: {missing:,} of 11,977 candidate pair-dates lack at least one contract ID in the available quote bundle. The remaining 6,265 have ID presence only; that is not a daily quote completeness test. Quote coverage was not used to select the broad candidates.',
        '2. Training labels and shadow paths retain the archived $0.65 per-contract commission. Funded replays use the current $0.03 commission. Both include the specified option execution penalty and stock hedge costs, but labels were not recosted because complete contract paths are unavailable. The target is normalized by reference 450 vega, not verified actual entry gross vega. This is a controlled proxy experiment, not the fully aligned target originally proposed.',
        '3. The inherited universe and formation calendar contain retrospective coverage restrictions. Temporal training safeguards do not remove that universe limitation.',
        '4. These tests can support or weaken the case for learned opportunity filtering. They do not prove the current rule is optimal, that PCA is necessary, or that unexplained input variance is profitable signal. A failed ridge/spline test does not rule out every useful contextual rule.',
        '5. The next decisive broad test needs complete point-in-time quotes, consistently costed outcomes, and a frozen future evaluation. Preserve this experiment and its negative as well as positive findings; do not search more thresholds on the same later period and call the result confirmation.', '',
        'Reproduction: python -m scripts.run_p25_opportunity_filter; python -m scripts.report_p25_opportunity_filter',
        'Source protocol, fitted monthly models, scores, candidate coverage, funded ledgers and detailed CSV tables are retained. The report generator does not tune or select a model.', '']
    (report/'Findings.txt').write_text('\n'.join(lines), encoding='utf-8')
    summary=dict(conclusion=lead, interval_evidence=evidence, monthly_decision_windows=len(fits), fitted_predictive_models=fitted_models,
        broad_funded_available=False, broad_missing_contract_pair_dates=missing,
        training_capacity=capacity, shadow_concentration=concentration,
        selected_coverage=selected.to_dict('records'),
        model_activity=activity.to_dict('records'),
        funded_pair_attribution_cases=attributed,
        later_funded=t['funded_periods'].loc[t['funded_periods'].period.eq('evaluation')].to_dict('records'))
    (report/'report_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    rest()
    make_figure(t,report)
    rest()
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    args=parser.parse_args()
    from scripts.run_p25_representation import Budget
    budget=Budget()
    result=generate(args.report,rest=budget.rest)
    print(result['conclusion'],flush=True)
    print(result['interval_evidence'],flush=True)


if __name__=='__main__':
    main()
