# Sizing and earnings tests on the ungated strategy

Eight initially declared comparisons and two subsequent follow-ups retain all 4,744 proposals, original directions and contracts, daily delta hedges, expiry holding, training/execution costs and the $10m funded account limits. Profitability memory stays an input with no minimum-profit or count gate. Four-state and 12-month variants were added after observing that 16 states and a 36-month window hurt performance. These are further research on inspected dates, not independent confirmations.

| Variant | Annualized excess Sharpe | Trades | Learned skips | Profit above cash |
| --- | ---: | ---: | ---: | ---: |
| Ungated 50/75/100 baseline | 0.835 | 1334 | 0 | $786,690 |
| Ungated fixed full size | 0.713 | 873 | 0 | $702,500 |
| Constant25% sizing | 0.634 | 2740 | 0 | $403,754 |
| Constant50% sizing | 0.655 | 1685 | 0 | $560,689 |
| Constant75% sizing | 0.593 | 1156 | 0 | $577,241 |
| 25/50/75/100 sizing | 0.830 | 1938 | 0 | $697,466 |
| 0/50/75/100 sizing | 0.389 | 621 | 3081 | $252,845 |
| 0/25/50/75/100 sizing | 0.508 | 846 | 2802 | $340,648 |
| 16-leaf state model | 0.672 | 1340 | 0 | $631,040 |
| 36-month training window | 0.717 | 1352 | 0 | $668,719 |
| Prior-report earnings countdown | 0.768 | 1334 | 0 | $707,003 |
| Prior-report earnings + skip | 0.639 | 605 | 3077 | $426,257 |
| Historical earnings dates assumed known | 0.755 | 1337 | 0 | $706,634 |
| 4-leaf state model | 0.824 | 1283 | 0 | $783,552 |
| 12-month training window | 0.493 | 1290 | 0 | $476,426 |

The highest full-history score among the new variants is 0.830, from 25/50/75/100 sizing, versus 0.835 for the original ungated learner. The best of the four constant-size controls scores 0.713. These descriptive rankings use the same previously researched history. They are not a new holdout or automatically promoted checkpoints.

The learned menu changes only entry sizing and admission. It never reverses direction or exits held positions. A zero action explicitly skips a proposal. The 16-leaf test retains the original minimum-leaf formula. All cases refit monthly using completed outcomes; the longer-window case uses 36 months rather than 18. CEM remains 50 iterations, 192 candidates and seed 2026. Menus with a full-size anchor can reproduce that anchor rather than being forced to change size.

Earnings adds days until the next report and an indicator for a report before expiry, separately for long and short, plus availability flags. The prior-report countdown uses strictly earlier reports. Historical-calendar mode assumes final report dates were known at signal time; an archive of announced schedules is unavailable. Same-day announcement time is unresolved. These versions use 41 inputs and never apply an earnings ban.

All 90 baseline monthly fits and the complete 0.835-funded ledger reproduce. New ledgers reconcile with closed inventory. Seven focused tests pass. A single worker remains capped at 20% of one core and 384 MiB RSS. After a safe stop on system swap growth, stock prices were moved to a disk-backed lookup; subsequent funded replays peaked below 200 MiB RSS without changing the baseline ledger.

The original 18-month, eight-state, 50/75/100 sizing policy remains best in this batch. Earnings inputs do not improve it. The 25% floor permits 45% more trades with nearly the same Sharpe, but lower profit. Adaptive sizing beats every constant-size control descriptively; the unadjusted block-bootstrap interval for its Sharpe gain over full sizing includes zero, while its gain over constant half sizing excludes zero. Neither comparison corrects for the prior research search.

See [yearly returns and Sharpe](annual.csv), [requested sizes](requested_sizes.csv), [fit dates](fits.csv), [earnings coverage](earnings_coverage.csv), and [block-bootstrap differences](sharpe_change_intervals.csv). Confidence intervals do not correct for the strategy search. Training shadow-path Sharpe excludes funded-book admission interactions, and differs from funded Sharpe. 2025 is partial through 29 August.
