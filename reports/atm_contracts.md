# Daily ATM / 30-day contracts

Completed: **778,907 selections** across **316 underlyings**, January 2010–August 2025; **775,453 successful fits (99.56%)**. GSLC has no selections under the eligibility filters. The 3,454 failed fits remain flagged (3,428 unsupported distributions; 26 invalid inputs/price bounds). Local Parquet output is about 96 MiB; peak extraction RAM was 358 MiB.

Each underlying/session selects one eligible standard 100-share **call**: expiry closest to 30 calendar days at next-session entry, then strike closest to the previous stock close. Existing filters apply: 21–45 days, strike within 10% of spot, positive bid, width at most 25% of midpoint, open interest at least 100, and valid vendor Greeks. Ties use higher open interest, then lower option ID. No eligible contract means no selection. This is nearest **eligible** ATM, not necessarily the unfiltered nearest strike.

The local dataset is `data/atm_30d/<year>/batch=*.parquet`; [coverage](atm_contracts/coverage.csv) and [summary](atm_contracts/summary.json) describe the completed extraction. Each row retains contract ID, expiry, strike, bid/ask, vendor IV/Greeks and adds Black–Scholes midpoint IV, delta, gamma, vega, rate, known-dividend PV and fit status. Failed fits remain present. Licensed quotes remain local.

The model uses ACT/365 maturity, the latest available zero curve within seven days, linearly interpolated in maturity (flat outside quoted nodes), and spot less discounted ordinary USD cash dividends declared by the quote date. Unknown announcement dates and undeclared dividends are not used or forecast. Announced unsupported distributions prevent fitting. These are European-call approximations; American exercise and discrete dividends can create differences from market/vendor IV. [OIC model reference](https://prd-web.optionseducation.org/advancedconcepts/black-scholes-formula). IV is decimal; vega is dollars per share per one volatility point, multiplied by 100 for a standard contract. Invalid prices or solutions outside 0–1,000% IV are flagged, not clipped.

The simulator accepts `--greeks-source black_scholes` to use fitted delta for daily stock hedges and fitted vega for sizing. It fits marks on the **same held contract** throughout each position; the daily ATM series does not silently roll existing positions. Executions still use observed bid/ask and NAV uses observed midpoints. Missing fitted Greeks trigger the existing pair-liquidation rule. Black–Scholes mode rechecks eligibility with fitted Greeks, so its selected contracts can differ from the vendor-eligible extraction when a fit fails. The midpoint inversion reproduces the input price; it is not an independent fair-value forecast.

```sh
python -m scripts.select_atm_contracts
python -m scripts.report_atm_contracts
python -m scripts.run_contract_pilot --source-root /path/to/optionmetrics --greeks-source black_scholes --repair-crossed-held-quotes --output artifacts/contracts/pilot_black_scholes
```

Extraction streams 5,000 raw rows at a time with one worker. The output manifest records input hashes and conventions. The AMD/NVDA pilot still requires the earlier pilot cache and saved policy actions; the full-universe daily selection is preparation for broader contract backtests, not a completed backtest of every pair.

[Pilot validation](atm_contracts/pilot_comparison.csv): the unchanged saved AMD/NVDA actions produce 70 entries with either Greek source. January–August 2025 return with fitted Greeks is 8.97% including cash, with annualized daily excess Sharpe 1.16 (vendor: 8.02%, 0.91). Both use the previously documented four crossed-quote envelope repairs. These are engineering sensitivities on an already-inspected period, not new holdout results; failed-fit exits also differ between the runs. NAV reconciles and every original contract position closes.
