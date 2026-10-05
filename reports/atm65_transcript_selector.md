# Monthly reference/transcript policy selection

Use the saved current-best and transcript policies. Each pair/month chooses the enabled policy with higher prior-year validation Sharpe at the unchanged $0.65 training hurdle. Ties retain the reference; missing or disabled transcript proposals cannot replace an enabled reference. No evaluation-period realized returns are used for monthly selection, and no models are refitted. Original recent-five-year refit actions, costs, risk limits and contract selection remain unchanged. Evaluation uses $0.03 commission.

| Model | Full Sharpe | Recent Sharpe | Full entries |
|---|---:|---:|---:|
| rl_baseline | 0.368 | 0.428 | 224 |
| rl_selector | 0.331 | 0.407 | 254 |
| nn_baseline | -0.084 | -0.103 | 217 |
| nn_selector | -0.172 | -0.258 | 260 |

Periods start flat in 2018 and 2022 and end in August 2025. Sharpe is above cash and annualized with √252. All comparisons use the common completed roster; failed paths are reported. This selection rule was tried after observing the mixed transcript result on previously examined history. The prior-year scores were already used for proposal validation; selecting between them adds selection uncertainty and is not an independent validation test.

[Comparison](atm65_transcript_selector/comparison.csv) · [Annual](atm65_transcript_selector/annual.csv) · [Selection logs](atm65_transcript_selector/fits.csv) · [Coverage](atm65_transcript_selector/coverage.json)
