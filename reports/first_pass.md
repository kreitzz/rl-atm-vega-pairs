# First pass: maximize net payoff Sharpe

The spread-only policy stays flat on 68.5% of test dates. It achieves annualized net payoff Sharpe of 0.672, below the simple short-higher-IV baseline's 0.956. Annualization uses `sqrt(365 / 30) × mean(net 30-day payoff) / sample SD`, including zero payoffs for flat entry dates. This horizon-based square-root-of-time convention assumes independence across nonoverlapping holding periods; it is not a serial-correlation-adjusted estimate. Daily entry frequency does not justify multiplying overlapping 30-day payoffs by `sqrt(252)`. They are not daily marked portfolio Sharpes.

## Policy and cost

- Sole input: AMD 30-day IV minus NVDA 30-day IV, in percentage points.
- If the spread is below 2.967 points: flat.
- At or above 2.967 points: short AMD / long NVDA.
- Equal initial dollar vega per leg; assume daily delta hedging and a 30-calendar-day hold.
- Long-AMD gross reward: `(RV_AMD - IV_AMD) - (RV_NVDA - IV_NVDA)`; reverse trades negate this.
- Net reward: `action * long_AMD_reward - cost * abs(action)`, with actions -1, 0, +1.
- Default cost: **0.5 vol points per entire pair round trip**, charged once per nonflat entry. This is an illustrative combined allowance for entry, exit, and daily hedge costs, not a calibrated execution estimate. At $1,000 vega per leg, it means $500 total per pair trade. Flat costs zero; there is no additional inactivity penalty.

## Evaluation

| Strategy | Annualized net payoff Sharpe | Mean net payoff per entry date | Dates traded | Winning trades |
| --- | ---: | ---: | ---: | ---: |
| Sharpe-trained policy | 0.672 | 1.493 | 31.5% | 65.2% |
| Short higher IV / long lower IV | 0.956 | 3.252 | 100% | 61.7% |
| Always long NVDA / short AMD | -0.042 | -0.149 | 100% | 50.2% |
| Always long AMD / short NVDA | -0.239 | -0.851 | 100% | 46.7% |
| Flat | 0 by convention | 0.000 | 0% | — |

Payoffs are vol points per $1 of dollar vega per leg. Mean payoff includes flat dates. All strategies pay the same cost when trading. Daily trade outcomes overlap; these figures are not returns on capital or a marked portfolio simulation.

| Split | Entry dates | Entries | Latest outcome |
| --- | --- | ---: | --- |
| Train | 2002-01-02–2016-12-01 | 3,756 | 2016-12-31 |
| Validation | 2017-01-03–2020-12-01 | 969 | 2020-12-31 |
| Test | 2021-01-04–2025-08-29 | 1,170 | 2025-09-28 |

Trades whose outcomes cross split boundaries are excluded. No model fitting uses test outcomes. However, this test period was already viewed for the initial gross-reward experiment before switching to Sharpe and costs, so the revised results are exploratory rather than a fresh untouched test.

A circular moving-block bootstrap (63-entry blocks, 2,000 resamples, seed 2026) gives a 95% interval of **[-0.018, 1.253]** for annualized policy payoff Sharpe, and **[-0.030, 3.113]** for mean net payoff. These intervals reflect local dependence, not model-selection or regime uncertainty.

## Training

Direct episodic policy search uses the cross-entropy method to maximize **annualized net payoff Sharpe**, rather than Q-learning on average P&L. Each candidate maps IV-spread buckets to the three actions; replaying the training history gives its terminal Sharpe objective. Because trades do not interact through inventory or capital, this is a contextual policy-search problem rather than a multi-step control problem.

Quantile boundaries use training data only. For each of 5, 10, and 20 buckets, fit five seeds (0–4), 80 iterations, and 256 candidates per iteration. Update action probabilities toward the best 10% with smoothing and exploration; preserve the best training policy. Select the seed by training Sharpe and the bucket count by validation Sharpe. Validation selected five buckets. Do not refit after validation. All five selected-model seeds produced the same test actions.

Training and validation both subtract the cost before calculating Sharpe. Flat outcomes remain in the denominator's sample, and an all-flat policy has objective zero by convention. The 30-day outcome never enters the policy's state.

Retraining at alternative cost assumptions produced the same selected actions:

| Pair round-trip cost | Annualized net test payoff Sharpe | Dates traded |
| --- | ---: | ---: |
| 0.00 | 0.736 | 31.5% |
| 0.25 | 0.704 | 31.5% |
| 0.50 | 0.672 | 31.5% |
| 1.00 | 0.606 | 31.5% |

Thus Sharpe optimization already favors abstention here; the cost did not independently cause the no-trade region. Costs were not selected by test performance.

## Data and reproduction

OptionMetrics 30-day 50-delta call surface IV is the ATM proxy. SECIDs: AMD 101121 and NVDA 108321. CRSP PERMNOs: AMD 61241 and NVDA 86580. Forward realized volatility is `100 * sqrt(252 * mean(log(1 + retx)^2))`, using split-adjusted ex-distribution daily returns strictly after entry through entry + 30 calendar days. On nontrading end dates, use the preceding trading close. Entry is assumed at the observed IV close without execution delay. The source daily returns have no missing values or duplicate dates for either stock over January 2002–December 2025; IV limits entries to August 2025.

Install `requirements.txt`, then run:

```sh
python -m unittest discover -s tests -v
python scripts/run_first_pass.py --source-root /path/to/convert-rv-rl/data/raw/wrds --cost 0.5
```

To retrain at another cost, change `--cost` and use a different `--output` directory. Dependencies were available in the existing `convert-rv-rl/.venv/bin/python` environment. Initial local source: `/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds`.

Ignored outputs in `artifacts/sharpe_pass/` include the dataset, test actions and gross/net rewards, scores, annual means, policy, and JSON settings/results. Licensed market data and individual observations are not committed. Five tests check annualization, forward-window timing, split purging, cost-driven abstention on synthetic data, and the policy-search objective against exhaustive enumeration.
