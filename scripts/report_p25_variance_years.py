"""Summarize verified year attribution and render a standalone research figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json

OUT = Path('reports/p25_variance_years')
BASE = Path('artifacts/contracts/p25_feature_search/replays')


def main():
    hashes = json.loads((OUT / 'output_hashes.json').read_text())
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    audit = json.loads((OUT / 'audit.json').read_text())
    assert audit['status'] == 'complete' and audit['maximum_daily_ledger_error'] < 1e-6
    annual = pd.read_csv(OUT / 'annual_components.csv')
    pairs = pd.read_csv(OUT / 'annual_pairs.csv')
    candidate = annual[annual.model.eq('variance_premium')].copy()
    columns = ['year', 'entries', 'annualized_cash_excess_sharpe', 'gross_delta_hedged_pnl', 'execution_and_borrow', 'financing_vs_cash', 'net_excess_pnl', 'worst_3_days_pnl', 'other_days_pnl']
    candidate[columns].to_csv(OUT / 'annual_summary.csv', index=False)
    comparison = annual.pivot(index='year', columns='model', values='net_excess_pnl')
    comparison['new_minus_previous'] = comparison.variance_premium - comparison.control
    comparison['new_minus_fixed50'] = comparison.variance_premium - comparison.fixed2
    comparison.to_csv(OUT / 'annual_policy_comparison.csv')
    pair_comparison = pairs[pairs.year.eq(2024)].pivot(index='label', columns='model', values='calendar_net_before_shared_financing').fillna(0)
    pair_comparison['new_minus_previous'] = pair_comparison.variance_premium - pair_comparison.control
    pair_comparison['new_minus_fixed50'] = pair_comparison.variance_premium - pair_comparison.fixed2
    pair_comparison.to_csv(OUT / '2024_policy_pairs.csv')
    books = {name: pd.read_parquet(BASE / name / 'daily_nav.parquet') for name in ['variance_premium', 'control', 'fixed2']}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    names = {'variance_premium': 'New variance-premium sizing', 'control': 'Previous learned sizing', 'fixed2': 'Fixed 50% vega'}
    colors = {'variance_premium': '#176d89', 'control': '#aa7928', 'fixed2': '#858a92'}
    for ax, year in zip(axes.flat, [2019, 2021, 2022, 2024]):
        for name, nav in books.items():
            daily = nav.nav.diff().fillna(0) - nav.cash_benchmark.diff().fillna(0)
            series = daily.loc[daily.index.year == year].cumsum() / 1000
            ax.plot(series.index, series.values, label=names[name], color=colors[name], linewidth=1.8)
        ax.axhline(0, color='#303840', linewidth=.6)
        ax.set_title(str(year), loc='left', fontweight='bold')
        ax.set_ylabel('Calendar cash-excess P&L ($ thousands)')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Saved P25 books: annual P&L paths\nSame proposals and costs; execution and overlap depend on size', fontsize=13)
    fig.savefig(OUT / 'yearly_pnl.png', dpi=180)
    plt.close(fig)
    sources = {
        'NVDA_earnings_announcement': 'https://nvidianews.nvidia.com/news/nvidia-sets-conference-call-for-second-quarter-financial-results-6769669',
        'SNAP_guidance_warning': 'https://www.sec.gov/Archives/edgar/data/1564408/000119312522157565/d516472d8k.htm',
        'AAL_guidance_cut': 'https://www.sec.gov/Archives/edgar/data/4515/000119312524148138/d800477d8k.htm',
    }
    write_json(OUT / 'event_sources.json', sources)
    text = (
        'Annual diagnosis of the retained1.0063 P25 research candidate\n\n'
        'All3 actual saved books reconciled to their observed option marks, shared stock '
        'hedges, costs, dividends and cash financing; maximum daily difference<$0.000001. '
        'No retraining, new trades, quote deletion, pair exclusion or performance revision. '
        'Pair and cohort contributions below exclude shared financing; annual book P&L '
        'includes it. Calendar years include inventory carried from prior years.\n\n'
        '2019:23 entries; net cash-excess loss$7663.73. Gross hedged loss$3391.01, '
        'execution/borrow$3342.83, financing-$929.89. AMD/NVDA accounts for-$8878.70 '
        'before financing across2 overlapping positions, long AMD/short NVDA entered '
        'August13/14. Both straddled NVDA August15 earnings, publicly announced '
        'in advance. On August19 NVDA rose7.03% versus AMD0.96%; the two cohorts '
        'lost$4389 combined that day. Across one full cohort, long AMD IV50.95 versus '
        'forward RV45.20 lost$3175; short NVDA IV50.96 versus forward RV50.82 '
        'lost$984. Relative IV rank admitted long vol that was not cheap on realized '
        'outcome. Forward RV is retrospective context, not a complete gamma P&L model.\n\n'
        '2021:23 entries; loss$9565.26. OIH/HAL lost$14294.91 across7 trades; '
        'HAL/SLB another$3114.98. Largest OIH/HAL cohort lost$8580.74, mainly '
        'long HAL-$6740.35 rather than short OIH-$1840.38. Both expired July16: '
        'HAL20-strike put finished with stock20.06, worthless; OIH200-strike short '
        'put finished with ETF189.35, in the money. Forward RVs44.46/43.99 look '
        'similar, but strike-crossing, timing and discrete hedging matter. July16 '
        'alone cost that cohort$5220.86. This was high observed return correlation '
        '(.910) and beta(.879), not a low-correlation admission failure. '
        'Execution-day absolute deltas were.086/.122, despite prior-close selection '
        'of25-delta contracts. Subsequent price moves change strike exposure; '
        'vega is matched at entry, not maintained continuously.\n\n'
        '2022:67 entries; profit$176776.85. Leading pair contributions: '
        'SNAP/UBER+$56202.95, GPS/TJX+$43551.32, AMD/MU+$34927.70, '
        'AEO/TJX+$25582.00, OIH/SLB+$23247.40, offset by losing pairs. '
        'May24 alone added$46479.20 to the portfolio after SNAP guidance warning '
        'on May23. SNAP fell43.08%, UBER9.38%; long SNAP puts captured large '
        'convex gains. The remainder of2022 still added$130297.65; this is a '
        'descriptive contribution, not a new backtest omitting a day. Several '
        'other winners monetize short-leg IV above subsequent RV: GPS sold near '
        '90.78 IV versus50.59 realized in one TJX/GPS cohort, and MU sold near '
        '52.37 IV versus32.00 realized in one AMD/MU cohort. '\
        'Three of the four SNAP/UBER entries overlap the same May event, so '
        'cohort count does not represent four independent event successes.\n\n'
        '2024:13 entries; loss$10720.09. Gross delta-hedged loss$22707.71; '
        'charges$2473.86; financing versus cash+$14461.48. UAL/AAL-$22184.12 '
        'and UPST/COIN-$21031.08 dominate losses. The UAL/AAL cohort held '
        'long UAL/short AAL through the AAL guidance reduction reported May28. '
        'On May29 AAL fell13.54% while UAL rose2.35%; short AAL put delta went '
        'from-.0555 to-.9398 and the pair lost$21166.63 that day. Full trade '
        'short AAL lost$20915.23: IV38.40 versus forward RV61.71. '
        'UPST/COIN differs: long UPST lost$17565.88, short COIN$3465.19. '
        'UPST entry IV86.12 versus subsequent RV57.09 was expensive long carry. '
        'COIN rose16.85% on February26; the pair lost$8718.50 that day. '
        'No external cause of that COIN move is asserted.\n\n'
        'Why the new policy improves2024: it did not avoid the two main losing '
        'cohorts; they are identical to the previous18-month policy. Most improvement '
        'comes from avoiding MGM/WYNN-$3388 and gaining an additional$4365 '
        'in UAL/DAL, partly offset by$2615 less TRV/CB profit. Those are actual '
        'proposal-size/risk-admission differences, not evidence of predicting '
        'the AAL/COIN shocks. Fixed50 admitted3 overlapping UPST/COIN trades '
        'versus1 learned, and2 UAL/AAL trades versus1 learned. Their combined '
        'loss was$70.2k under fixed50 versus$43.2k learned. The shared inventory '
        'effect arises from size and inherited caps; the learner was not '
        'trained as a sequential inventory controller.\n\n'
        'Quote nuance: PENN/CZR made+$4761.87 overall, but near-expiry midpoint '
        'marks generated approximately+$11030 on September18 and-$10531 on '
        'September20. PENN bid/ask changed from.01/.04 to.01/.93 with a '
        'stock move of-.81%; inferred IV rose from80%to230%. These are wide '
        'observed quote midpoints, not proof of executable value or verified '
        'vendor error. Held marks pass the current observed-mark convention; '
        'entry quote screens do not guarantee later liquidity. This can distort '
        'daily Sharpe even when a cohort ultimately profits. A common causal '
        'mark/Greek quality sensitivity across all years and controls is warranted; '
        'selectively deleting bad-looking days would be invalid. No mark changes '
        'are included in these saved performance results.\n\n'
        'Research implications: the weaker years merit targeted work rather than '
        'automatic rejection of the checkpoint. Examine earnings asymmetry and '
        'execution delta/moneyness drift (absent from current33 feature state), '
        'time-varying relative gamma/strike risk, overlapping event/name inventory '
        'and historical jump asymmetry. Correlation alone is inadequate: '
        'profitable SNAP/UBER near.52 resembles losing UPST/COIN near.51, '
        'while losing OIH/HAL was near.91. Do not select name bans or event '
        'vetoes using these future outcomes. Any new input must be available '
        'before trading; historical earnings columns here are diagnostic, '
        'not newly established point-in-time features. The same asymmetric '
        'jump mechanism generates winners and losers; manage size with observable '
        'risk rather than remove only the losses.\n\n'
        '2025 is partial toAugust29. This is historical attribution, not fresh '
        'validation of the selected Sharpe. Resource peak0.72GB, one worker '
        'at50%of onecore with3GiB stop. Source hashes and exact financial '
        'reconciliation checks are in audit.json.\n\n'
        + candidate[columns].to_string(index=False) + '\n\n'
        + comparison.to_string() + '\n\n'
        + '\n'.join(name + ': ' + url for name, url in sources.items()) + '\n'
    )
    (OUT / 'findings.txt').write_text(text)
    write_json(OUT / 'summary_audit.json', dict(status='complete', verified_previous_output_hashes=len(hashes), unchanged_candidate_cash_book_hash=digest(BASE / 'variance_premium/daily_nav.parquet'), no_strategy_changes=True, year_attribution_reconciled=True, figure=str(OUT / 'yearly_pnl.png'), source_script_hash=digest(Path(__file__))))
    write_json(OUT / 'output_hashes.json', {str(p): digest(p) for p in OUT.glob('*') if p.is_file() and p.name != 'output_hashes.json'})
    print(candidate[columns].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
