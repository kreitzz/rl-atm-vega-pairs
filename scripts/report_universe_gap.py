"""Summarize the prespecified broader independent-policy experiment."""
from pathlib import Path
import pandas as pd


def main():
    root=Path('reports/universe_gap')
    scores=pd.read_csv(root/'scores.csv');portfolio=pd.read_csv(root/'portfolio.csv');dependence=pd.read_csv(root/'dependence.csv');bins=pd.read_csv(root/'bins.csv')
    table=scores.pivot(index='pair',columns='variant',values='sharpe')
    new=scores.loc[~scores.original_pair,'pair'].unique()
    n_corr=int((table.loc[new,'correlation_gap']>table.loc[new,'baseline']).sum());n_beta=int((table.loc[new,'beta_gap']>table.loc[new,'baseline']).sum())
    text='''# Wider pair universe: stock-versus-IV dependence gaps

Eight additional pairs were specified from business similarity and local data availability before their payoff results were examined: AMAT/LRCX, MSFT/ORCL, UPS/FDX, UNP/CSX, T/VZ, DUK/SO, AMGN/GILD, and C/WFC. They join the four existing pairs. This remains a handpicked universe of available historical survivors, not systematic market-wide selection or an untouched holdout.

Each pair has its own independently trained policy and sizing ladder. Compare the existing five state inputs with either one additional correlation-gap input or one additional beta-gap input. There is no joint state, cross-pair decision, or transfer of unused budgets.

## Equal-budget portfolio results

Each portfolio allocates a fixed equal share of its entry budget to each constituent pair. A variant uses the same feature specification for every constituent; there is no test-selected mixture of pair-specific winners.

| Universe | Baseline | + Correlation gap | + Beta gap |
| --- | ---: | ---: | ---: |
'''
    for group,label in [('new','Eight new pairs'),('original','Four original pairs'),('all','All twelve pairs')]:
        x=portfolio[portfolio.group.eq(group)].set_index('variant')
        text+=f'| {label} | {x.loc["baseline","sharpe"]:.3f} | {x.loc["correlation_gap","sharpe"]:.3f} | {x.loc["beta_gap","sharpe"]:.3f} |\n'
    text+='''
The correlation gap helps most new pairs individually, but the eight-pair portfolio remains close to flat and the twelve-pair improvement is small. Every paired-bootstrap interval below includes zero. The beta-gap variant is weaker at portfolio level. This is useful candidate discovery, not confirmation of a generalizable diversification benefit.

## Every pair

Annualized 30-day proxy-payoff Sharpe on the same 3,918 entry dates, 2010–August 2025. No pair is omitted for poor performance.

| Pair | Baseline | + Correlation gap | + Beta gap |
| --- | ---: | ---: | ---: |
'''
    order=['AMD_NVDA','XOM_CVX','JPM_BAC','HD_LOW','AMAT_LRCX','MSFT_ORCL','UPS_FDX','UNP_CSX','T_VZ','DUK_SO','AMGN_GILD','C_WFC']
    for pair in order:
        x=table.loc[pair];text+=f'| {pair.replace("_","/")} | {x.baseline:.3f} | {x.correlation_gap:.3f} | {x.beta_gap:.3f} |\n'
    text+=f'\nAmong the eight additional pairs, adding the correlation gap improves Sharpe for **{n_corr}/8**, and adding the beta gap improves it for **{n_beta}/8**. These counts compare full feature ablations, not the conditional subgroup Sharpes from the earlier diagnostic. A favorable subgroup relationship does not guarantee that a refitted tree will use the extra feature effectively.\n'
    text+='''
## Uncertainty in the portfolio difference

Paired 63-entry circular moving-block bootstrap, 2,000 replicates. Each draw resamples the same calendar blocks for all pairs and variants, preserving their contemporaneous dependence. Intervals describe sampling variation conditional on the fitted policies and this sample; they do not include retraining uncertainty or correct for trying multiple features and pairs.

| Universe | Feature added | Sharpe change | 95% paired block interval |
| --- | --- | ---: | --- |
'''
    for row in portfolio[portfolio.variant.ne('baseline')].itertuples():text+=f'| {row.group} | {row.variant.replace("_"," ")} | {row.delta_sharpe:+.3f} | [{row.delta_ci_low:+.3f}, {row.delta_ci_high:+.3f}] |\n'
    text+='''
## Dependence profiles

Median trailing estimates during the test period. Betas regress the first ticker on the second. These are stock-return and IV-change regression betas, not market-index beta or option-implied equity beta. Each gap is calculated date by date before its median is taken, so its median need not equal the difference between the component medians.

| Pair | Stock correlation | IV-change correlation | Correlation gap | Stock beta | IV beta | Beta gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for pair in order:
        x=dependence.set_index('pair').loc[pair]
        text+=f'| {pair.replace("_","/")} | {x.median_stock_correlation:.3f} | {x.median_iv_correlation:.3f} | {x.median_correlation_gap:.3f} | {x.median_stock_beta:.3f} | {x.median_iv_beta:.3f} | {x.median_beta_gap:.3f} |\n'
    text+='''
The conditional-group CSV separately checks whether the baseline policy does better when each gap exceeds the preceding training sample's median. These are conditional Sharpes of existing decisions, not the performance of a deployable filter. No group thresholds were optimized for test outcomes.

## Data and method

- 5,678 common complete model rows, December 31, 2002–August 29, 2025; all variants and pairs share the calendar. The four omitted dates relative to the original 3,922-date test are July 2, 2010; May 9, 2011; July 3, 2012; and April 11, 2022. Training dates also differ slightly, so baseline numbers can change from earlier reports through tree/configuration selection. Compare variants within this report.
- Dependence estimates use a trailing 252-session window, with at least 200 paired valid observations and a full window of elapsed history. IV changes are differences in vol points; stock changes use CRSP ex-distribution daily returns. Neither quotes nor returns are filled across gaps. Both covariance and variance use the same matched sample. The initial perfectly-complete-window approach was discarded after its data-coverage audit excluded entire years; a small number of original-pair results had been printed before that correction. No new-pair policy result was used to choose the missing-data rule.
- State is pair IV spread, two earnings countdowns, and two trailing 30-calendar-day RVs. Added features are `stock_return_correlation - IV_change_correlation` or `stock_return_beta - IV_change_beta`, using data through entry close. This retains the existing entry-close timing assumption and actual-earnings-dates-known-within-30-days assumption.
- Pair cost is `0.025 * (IV_A + IV_B) / 2`, multiplied by absolute allocation. Each pair chooses signed sizes in 25% increments up to its fixed budget; unused allocation stays idle. The budget applies per entry cohort, not total overlapping positions.
- Expanding training, preceding two-year validation, subsequent two-year test blocks, with future payoffs purged at boundaries. For each pair/variant/fold, select 4/8/16 tree leaves and 126/252 minimum leaf counts by validation Sharpe. Each candidate uses five seeds, 80 policy-search iterations and 256 candidates. No hyperparameters were retuned to the additional pairs' test results.
- The quarterly earnings audit found three LRCX 2007 fiscal periods assigned a later common date in Compustat. Overrides use the original releases: [July 24, 2007 SEC filing](https://www.sec.gov/Archives/edgar/data/707549/000095013407015737/f32108e8vk.htm), [October 10, 2007 company release](https://newsroom.lamresearch.com/2007-10-10-Lam-Corporation-Announces-Selected-Financial-Results-for-the-Quarter-Ended-September-23%2C-2007), and [January 24, 2008 SEC filing](https://www.sec.gov/Archives/edgar/data/707549/000089161808000051/f37301e8vk.htm). Corrections were applied before any AMAT/LRCX model was evaluated.

Sharpes retain `sqrt(365/30)` scaling of overlapping 30-day realized-minus-implied proxy payoffs, including flat dates. They are not daily marked option-and-hedge portfolio Sharpes. Results are exploratory; finding profitable candidates or a feature winner in this scan is not independent confirmation of an edge. The current four-pair default is unchanged.

[All pair scores](universe_gap/scores.csv) · [Annual pair results](universe_gap/annual.csv) · [Portfolio comparisons](universe_gap/portfolio.csv) · [Annual portfolio results](universe_gap/portfolio_annual.csv) · [Dependence profiles](universe_gap/dependence.csv) · [Conditional groups](universe_gap/bins.csv) · [Coverage](universe_gap/coverage.csv) · [Selected configurations](universe_gap/configs.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.explore_universe_gap --source-root /path/to/wrds
python -m scripts.report_universe_gap
python -m unittest discover -s tests -q
```

Detailed panels and predictions stay local under ignored `artifacts/universe_gap/`. Only configuration, code, tests, and aggregate reports are published. Thirty-seven tests pass, including matched-observation beta calculations, future-data exclusion, earnings overrides, and paired-bootstrap identity.
'''
    Path('reports/universe_gap.md').write_text(text)
    print('Wrote reports/universe_gap.md')


if __name__=='__main__':main()
