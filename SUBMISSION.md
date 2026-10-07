# Submission guide

Start with the [10-page paper](paper/options_relative_value.pdf) and [latest saved presentation](slides/230ZB_volatility_spreads%20%281%29.pptx). The editable paper is [HTML](paper/options_relative_value.html); its [figures](paper/figures), [chart tables](paper/figure_data) and [source hashes](paper/source_manifest.json) are included. The browser presentation is [slides/index.html](slides/index.html). Earlier decks and your teammate's material remain available.

## Retained experiment

The submission reference is [config/p25_submission.json](config/p25_submission.json). It uses approximately 30-day 25-delta puts, 252-session IV-spread ranks, beta 0.85–1.15, return correlation at least 0.5, and positive completed-trade profitability memory. Contracts are selected at the signal close and executed the next session. Integer positions match vega within 10% at entry, hedge delta daily and hold each leg to its own expiry except audited corporate exits. Vega neutrality is an entry constraint, not continuous rebalancing of option quantities.

The retained allocation model has 33 inputs, an eight-leaf regression tree, a 50/75/100% size ladder and cross-entropy search on aggregated portfolio Sharpe. It refits monthly on an 18-month window of outcomes completed before the fit date. This is contextual allocation with direct policy search; the separate fitted-Q management experiments use continuation values and exits.

| $10m funded policy | Annualized excess Sharpe | Funded CAGR | Cash-relative CAGR | Executed pair entries |
|---|---:|---:|---:|---:|
| Learned sizing | 1.003 | 2.963% | 0.342% | 212 |
| Fixed 100% vega budget | 0.769 | 2.908% | 0.289% | 133 |
| Fixed 50% vega budget | 0.761 | 2.882% | 0.264% | 263 |

All three receive the same 608 qualified proposals and use the same execution and risk rules. A fixed vega budget changes contract counts to match vega; it is not equal contract counts. Sizing changes inventory and subsequent admission, so the executed portfolios differ. Compare policies within this table; the original $345m research reference and other maturities use distinct runs.

Sharpe is `sqrt(252) * mean(daily funded return - daily cash return) / std(..., ddof=1)`. The first initialization row is excluded. Calendar-year returns are period returns, not annualized partial-year returns. 2025 ends on 29 August.

The initial January 2018 fit contains 74 completed examples. Its optimized shadow-path training Sharpe is 2.015 versus 1.573 at full size. Training uses $0.65 commission and aggregated label paths; funded evaluation uses $0.03 execution commission. These training and evaluation levels are not directly comparable.

## Data and code

- [Finalized input CSV](data/p25_variance_premium_inputs.csv): 36,327 rows across 125 cached pairs; contract IDs, entry flags and the 33 retained model inputs. Forward payoffs are excluded from this CSV.
- [Submission manifest](data/submission/manifest.json): SHA-256 hashes and original paths for saved monthly models, separate derived training labels, scored proposals, selected option quotes, stocks, dividends, rates and final execution ledgers.
- [Retained feature training](scripts/run_p25_feature_search.py), [sizing optimizer](scripts/p25_action_windows.py), [contract engine](scripts/uniform_smile_engine.py), [risk accounting](scripts/uniform_smile_risk.py) and [funded capital comparison](scripts/run_p25_capital_sensitivity.py).
- [Submission runner](scripts/submission.py): local verification, cache restoration, result recomputation, initial-fold refitting and frozen portfolio replay. It never downloads data or launches a broad search implicitly.
- [Original ATM inputs](data/current_rl_inputs.csv.gz) and [manifest](data/current_rl_data_manifest.json) remain as historical data. They are not the retained P25 training state.

Only selected execution histories and derived caches are packaged. The approximately 51 GB of local raw/intermediate data are not a submission dependency for the retained replay. Full universe reconstruction, alternative-data joins and older exploratory pipelines still require licensed WRDS histories and their original caches. The latest one-year cheap-vega quote database is compressed in the submission bundle; final ledgers and fitted management models are also included.

## Reproduction

Use Python 3.11 from the repository root. Basic integrity verification needs only the standard library:

```sh
python3 -m scripts.submission --verify
```

For numerical reproduction, install the recorded environment into a local virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.submission --restore-cache
```

Cache restore verifies hashes, streams the quote archive and restores missing files under ignored `artifacts/`. It refuses to overwrite differing research files. Keep approximately 1 GB free for restored caches and outputs, in addition to the virtual environment.

Run numerical work with the supervisor. These settings preserve the conservative limits used after the memory-pressure investigation: one worker, one numerical thread, 20% duty cycle, 384 MiB aggregate worker RSS, at least 35% macOS memory headroom, and a 64 MiB swap-growth stop. Temporary headroom dips pause the same process for up to 60 seconds; critical pressure or less than 25% headroom stops it. The CPU limit is a duty cycle, not an OS-enforced quota. On Linux the supervisor checks worker RSS; the macOS system-pressure and swap checks do not apply.

```sh
.venv/bin/python -m scripts.run_signal_discovery_guarded \
  --module scripts.submission --stop-file artifacts/submission/STOP --timeout 1800 --duty-cycle 0.2 \
  --rss-limit-mib 384 --min-free-percent 35 \
  --warning-rss-limit-mib 384 --max-swap-growth-mib 64 \
  --pause-low-headroom-seconds 60 -- --metrics
```

Replace `--metrics` with one of these commands:

- `--fit-check`: refit the initial 18-month training window, match the saved tree and allocations exactly, and verify all 608 scored proposals against their saved monthly models. This is a focused training check, not a rerun of every feature search.
- `--replay --policy learned`: rebuild the retained funded portfolio from selected execution histories and match daily NAV, trades and decisions against the saved reference. Use `fixed50`, `fixed100` or `all` for controls.
- `--tests`: run unit-test modules in separate processes, serially. Increase the supervisor timeout for the complete suite; `--test-pattern test_cheap_vega_runtime.py` selects a particular module.

Fresh outputs go to `artifacts/submission/`; protected research checkpoints are preserved. No WRDS credentials are needed for the retained checks. Legacy runners may have larger default searches, so use the submission commands for review.

For paper generation, also install `requirements-report.txt`. The existing rebuild script reads saved results, not newly fitted models. [paper/reproduction.txt](paper/reproduction.txt) records the guarded command. The saved PDF and visuals require no rebuild for submission.

[Submission validation](reports/submission_validation.json) records integrity checks, initial-fold fitting, result recomputation, fresh-checkout replay comparisons and test coverage. All three funded portfolios match their saved NAV, trades and decisions; the local replay peaked at 370.4 MiB. GitHub repeats these checks on every push.

## Results and research history

| Question | Saved evidence |
|---|---|
| Does adaptive sizing improve the retained signal? | [Capital comparison](reports/p25_capital_sensitivity/comparison.csv), [yearly results](reports/p25_capital_sensitivity/annual.csv), [audit](reports/p25_capital_sensitivity/audit.json) |
| Which states and training windows helped? | [Feature ablations](reports/p25_feature_search/comparison.csv), [recency search](reports/p25_recency/comparison.csv), [validation](reports/p25_variance_validation/research_summary.txt) |
| Does earnings awareness improve the retained policy? | [Earnings-state comparison](reports/p25_earnings/findings.md), [yearly results](reports/p25_earnings/annual.csv), [reproduction](reports/p25_earnings/reproduction.txt), [inputs/models/replays](data/p25_earnings_experiment.zip); 0.977/0.956 versus 1.003 retained |
| Does completed-profitability memory need to gate proposals? | [Gate ablations](reports/p25_memory_ablation/findings.md), [yearly results](reports/p25_memory_ablation/annual.csv), [reproduction](reports/p25_memory_ablation/reproduction.txt), [proposals/models/quotes/replays](data/p25_memory_ablation_experiment.zip); retrained sizing scores 0.626 with counts only and 0.835 ungated versus 1.003 retained, with more trades and profit |
| Where do profits and losses occur? | [Annual pair contributions](reports/p25_variance_years/annual_pairs.csv), [recent-loss investigation](reports/p25_recent_losses) |
| Does the signal transfer across the smile or to 90 days? | [Matched maturity comparison](reports/p25_maturity_matched/comparison.csv), [long-expiry parity](reports/long_expiry_parity/findings.txt) |
| Can management improve longer-dated positions? | [30-day management](reports/p25_short_management/comparison.csv), [180-day fitted Q](reports/long_tenor_fitted_q/comparison.csv), [365-day entry](reports/native_year_entry/comparison.csv), [365-day management](reports/year_reversion_management/comparison.csv) |
| Does sequential Q improve the short-dated checkpoint? | [Fitted-Q comparison](reports/p25_reversion_q/comparison.csv), [audit](reports/p25_reversion_q/audit.json); it did not beat the retained checkpoint |
| Can we trade cheap standalone vega instead of pairs? | [Six results](reports/year_cheap_vega/comparison.csv), [yearly results](reports/year_cheap_vega/annual.csv), [chart](reports/year_cheap_vega/results.png), [audit](reports/year_cheap_vega/audit.json); all six lost money above cash after costs |
| Do portfolio combinations improve the sleeves? | [Three-sleeve weighting](reports/native_year_entry/replacement_blend/walk_forward_24M/comparison.csv), [30D/180D combination](reports/native_year_entry/replacement_blend/walk_forward_24M_30D_180D/comparison.csv); saved sleeve returns, not a shared-inventory execution replay |
| What came before the retained strategy? | [Original findings](reports/research_findings.md), [ATM sentiment](reports/atm65_employee_sentiment.md), [Q revision](reports/q_batch.md), [broader pairs](reports/broad_pairs.md), [contract Q](reports/contract_q.md), [finite episodes](reports/contract_episodes.md), [coverage](reports/contract_data.md), [entry Q sizing](reports/q_sizing.md), [tracking](reports/tracking.md) |

All research scripts, configurations, tests and saved reports are retained. The latest one-year management result is conditional: carry-based Q improves fixed-half sizing from 0.254 to 0.280, but worsens the learned-entry policy from 0.313 to 0.214. Standalone cheap-vega Q reduces losses relative to expiry holding, but does not establish a profitable strategy.

## Interpretation and data use

This is a results-driven research submission, not a prospective performance claim. The universe formation calendar is retrospective and the evaluation history has been repeatedly inspected. Feature, action and window selection are not adjusted for the full research search. The $10m sizing advantage has an unadjusted bootstrap interval crossing zero, profits are concentrated in particular periods, and some vendor historical revisions are not independently archived. Risk limits are inherited modeling assumptions, not broker margin requirements; positions can breach limits after entry without forced liquidation.

Financing and cash interest are included, so a positive funded return does not necessarily imply profitable trading above cash. Initial vega matching does not neutralize all volatility factors or gamma risk. The paper discusses these limits and preserves unsuccessful experiments.

Academic use only. Source data remain subject to OptionMetrics, CRSP, Compustat, Revelio and other vendor/institutional agreements. This notice grants no additional data rights. Keep the repository private and use it with the applicable institutional permissions. [Data-use notice](data/DATA_LICENSE.txt).
