# Strategy economics: retained research extension

Read [the illustrated report](index.html), or [its Markdown source](report.md).
The original repository paper is unchanged. This extension covers the short
ATM straddle's return mechanism and one exploratory joint-account risk rule;
it does not claim new put-pair alpha or a reliable Sharpe improvement.

## Reproduce the publication

From the repository root, using the `mfe_wrds` conda environment:

```powershell
conda run -n mfe_wrds python -m scripts.strategy_economics
```

The renderer uses Python, NumPy, pandas and Matplotlib. It runs offline with
the small derived CSV snapshots in `data/`; it does not require WRDS, quote
caches, old diagnostic directories or the archived development scripts.
Dependency ranges are in [requirements.txt](../../scripts/strategy_economics/requirements.txt).
Python 3.10 or later is required. The cleaned package was tested with Python
3.12.13, NumPy 2.4.4, pandas 3.0.2 and Matplotlib 3.10.8 in `mfe_wrds`.

The command verifies input hashes, derives the IV/RV and path groups,
recomputes the later account Sharpes and mean returns, checks P&L component
identities, and regenerates all five figures, two display tables, report
formats and `verification.json`. Generated figures and the report are retained
so a reader can inspect the work without installing Python.

This reproduces the **publication from frozen research outputs**. It does not
independently reconstruct raw quotes, replay trades, refit the warning, or
certify the original backtest. The original strategy code and data-access
documentation remain in the baseline repository. No raw WRDS quote history is
distributed in this extension.

## Evidence map

| Retained input | Purpose |
|---|---|
| `data/profit_bridge.csv` | Full-history straddle profit and accounting bridge |
| `data/trade_outcomes.csv` | 2,599 completed cohorts, entry IV, subsequent RV, path persistence and attributed profit |
| `data/trade_examples.csv` | Daily paths for selected actual cohorts; the report displays contrasting hedge-gain and hedge-loss cases |
| `data/account_daily.csv` | Three joint-account return paths, NAV and cash benchmarks |
| `data/account_performance.csv` | Previously computed performance and cash-relative drawdown |
| `data/sharpe_intervals.csv` | Existing paired block-bootstrap uncertainty estimates |
| `data/warning.csv`, `data/risk_protocol.json` | Saved warning timing and the frozen response specification |
| `data/hedge_checks.json`, `data/ivrv_checks.json` | Source reconciliation and sensitivity summaries |

`data/manifest.json` identifies source paths and SHA256 hashes, source working
commit `8152b445521cc3dd24068918df373e02c9e4b8d5`, and remote baseline commit
`87642cd556f71d3ba408b726fbc97f14b193b540`. Source paths are provenance only;
the old development directories are intentionally not runtime dependencies.
The source working commit is preserved in a local recovery bundle, not made
a prerequisite of a clone of the cleaned publication.

## Scope of the cleanup

The publication excludes abandoned model variants, duplicate report exports,
bulk intermediate outputs, recovery runners, worker logs and unfinished
historical reconstruction. Those local additions were archived before removal.
The remote baseline's existing files and history are preserved. Negative
experiments are acknowledged in the report; omission of their development
files does not make the retained historical risk trial a fresh holdout.

## Interpretation limits

- Full-history straddle attribution (2018–August 2025) and the joint-account
  trial (2022–August 2025) are different populations.
- Option plus gross hedge P&L omits remaining stock costs, dividends and
  financing; the full-account bridge includes them separately.
- Subsequent RV and persistence are hindsight descriptions, not entry signals.
- The conditional Sharpe improvement's interval versus holding includes zero;
  maximum drawdown is slightly worse.
- No market/sector-adjusted alpha, causal risk-premium attribution, industry
  filter or reliable superiority of an ML layer is established.
