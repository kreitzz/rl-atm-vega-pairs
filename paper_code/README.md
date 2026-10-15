# Paper implementation

This is the compact implementation of the [current report](../paper/options_relative_value_draft.pdf): approximately 30-day, 25-delta put pairs, daily delta hedging, and monthly contextual entry sizing. It is independent of the exploratory `scripts/` import chain.

The default strategy is the **27-input screened model**, not the earlier 33-input checkpoint. The retained specification produces excess Sharpe of **0.573 in validation and 1.263 in later evaluation**, versus **0.478 and 1.055** at fixed full size.

[Download the source archive](../paper/paper_code_submission.zip). It contains this package and tests; use it inside the repository with the existing data bundles. [Recorded verification](verification.json) includes all four funded replays.

## Run

Use Python 3.11 and the existing committed data directories. No WRDS connection, credentials, or bulk cache extraction is required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r paper_code/requirements.txt

.venv/bin/python -m paper_code test
.venv/bin/python -m paper_code verify
.venv/bin/python -m paper_code metrics
.venv/bin/python -m paper_code fit --month 2018-01-01
.venv/bin/python -m paper_code replay --policy learned
```

`verify` recomputes the four main policies' validation/evaluation metrics and the paired bootstrap interval, checks the saved validation winner, and reproduces all 608 sizing choices from their monthly models. `metrics` writes period and annual results to `artifacts/paper_code/`.

`fit` reconstructs one monthly tree and CEM sizing map and compares them exactly with the saved model. Explicitly pass `--month all` for all 90 fits. `replay` reconstructs the shared account and checks daily NAV, option fills, and every proposal's admission against the saved reference. Its other policies are `fixed50`, `fixed75`, and `fixed100`; each must be requested separately.

Fit and replay automatically run one worker at 50% of one core, with one numerical thread, a 384 MiB aggregate RSS stop, and macOS headroom/swap checks. They stop rather than continue under unsafe memory pressure. Create `artifacts/paper_code/STOP` to stop the worker. These commands do not change original models, papers, or research outputs.

## Read the code

| Module | Purpose |
|---|---|
| `signals.py` | Completed profitability memory, beta/correlation, 27 state inputs, entry screen |
| `policy.py` | 18-month completed-only fitting window, tree states, daily-path aggregation, CEM |
| `contracts.py` | Quote checks, integer vega matching, execution prices and charges |
| `pricing.py` | Black–Scholes fitting and Greeks with declared-dividend adjustment |
| `risk.py` | Netted stock hedge, reserve, gross exposure, and nine-scenario stress |
| `ledger.py` | Next-session entries, overlapping positions, daily cash accounting and expiry |
| `metrics.py` | Cash benchmark, annualized Sharpe, period returns, paired block bootstrap |
| `data.py` | Bounded quote cache and readers for the committed data bundles |
| `__main__.py` | Reproduction commands and exact comparisons with saved evidence |
| `guard.py` | CPU/memory supervision for fitting and replay |

`strategy.json` records the selected specification. `execution.json` records the actual execution and risk assumptions. `provenance.json` identifies the original source snapshot. Numerical ledger formulas are retained; unused leverage and management branches are removed.

## Accounting and timing

Signals qualify at a close and execute at the next session's close. Profitability memory includes only candidate outcomes whose **later leg completion precedes the signal date**. A monthly fit also requires both signal and outcome dates to precede its cutoff. Candidate memory observations can overlap and need not have been admitted to the funded account.

The learner partitions completed full-size profits into up to eight tree states, then CEM assigns 50/75/100% exposure to maximize the combined daily shadow-path Sharpe. These paths are separately simulated by action; they are not simply scaled full-size profits. Funded inventory and admission effects enter the later replay, not the training objective.

The ledger tracks option cash flows and marks, stock price P&L on the existing hedge, dividends on the ex-date, cash interest, stock borrow, and transaction costs. It then adjusts the stock hedge. Vega matching is an entry constraint; option quantities remain fixed to each leg's expiry except documented corporate-event exits. Expiry uses cash intrinsic settlement; early exercise and partial fills are not simulated.

Validation is 2018–2021; later evaluation is January 2022–August 2025. Accounts continue across the boundary. The instrument, universe, and specification inherit prior research; later history is not an untouched holdout.

## Data and supporting experiments

The package reads `data/submission/` and `data/p25_validation_split_experiment.zip` directly. Input states and completed training outcomes remain separate. The selected-contract quote bundle supports the frozen proposals, not a reconstruction of every possible historical contract or pair.

The paper's secondary comparisons retain their original implementations: [simple sizing controls](../scripts/audit_p25_sizing.py), [direct linear/additive policy search](../scripts/p25_direct_policy.py), [admission attribution](../scripts/diagnose_p25_generalization.py), and [report figures](../paper/build_performance_chart.py). Those research pipelines are not run implicitly by this package.

Vendor data remain subject to the repository's [academic data-use notice](../data/DATA_LICENSE.txt).
