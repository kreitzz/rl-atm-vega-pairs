# Expanded historical option-pair discovery

The existing research roster is much narrower than the available source universe. The old formation rule retained at most four names per three-digit SIC industry and 180 stock pairs per screen, plus predefined ETFs. The uncapped search below finds **2,450 new industry-peer pairs**, including **269 pairs whose two names already have downloaded option contracts**. They have not yet been trained or backtested as these new combinations.

## Discovery rule

- Quarterly formations from January 2018 through July 2025 (31 dates), using only information preceding each date.
- Start from 1,922 prepared linked-security histories. Prior 252-session median call volume ≥100 and open interest ≥500, with at least 200 jointly observed return/activity dates.
- Pairwise stock-return beta 0.85–1.15 and correlation ≥0.5, using all preceding 126 sessions with no missing returns. Beta regresses the lower SECID on the higher; pair orientation is not optimized.
- Stock returns come directly from CRSP `retx`, respecting security-link intervals. They are not masked by missing standardized IV. Name and SIC metadata are dated before formation.
- No four-name cap, no 180-pair cap, and no ranking by realized trading profits or standardized IV levels.
- Record same-industry, broader-sector, cross-sector, ETF and exact-same-issuer matches separately. Same-issuer classification uses historical issuer text; it is a screening flag, not an exhaustive corporate-ownership audit.

| Relationship at latest qualifying formation | New pairs | Both names have downloaded contracts |
|---|---:|---:|
| Different two-digit SIC sectors | 25,106 | 6,225 |
| At least one ETF | 584 | 256 |
| Same historical issuer text | 1 | 0 |
| Same two-digit SIC sector, different three-digit industry | 881 | 169 |
| Same three-digit SIC industry | 1,569 | 100 |

Across all categories there are 28,141 new statistical candidates. The industry-peer subset would require 409 additional underlyings to cover every candidate. These are unions across historical formation dates, not simultaneous current opportunities. High market-wide correlation can produce many cross-sector matches; those should not be treated automatically as economically comparable volatility pairs.

## Illustrative untested industry peers

| Pair | Qualifying quarters | Median beta | Median correlation | Latest qualifying formation | Contracts downloaded for both? |
|---|---:|---:|---:|---|---|
| HES/COP | 24 | 0.961 | 0.828 | 2024-10-01 | Yes |
| KEY/RF | 23 | 0.940 | 0.895 | 2025-07-01 | No |
| APA/DVN | 22 | 0.967 | 0.815 | 2025-07-01 | Yes |
| AEM/RGLD | 21 | 0.960 | 0.809 | 2025-07-01 | No |
| SWKS/ADI | 20 | 0.937 | 0.780 | 2025-07-01 | No |
| D/DUK | 19 | 0.915 | 0.739 | 2025-07-01 | No |
| ON/NXPI | 17 | 1.003 | 0.804 | 2025-07-01 | No |
| MGM/HLT | 15 | 0.972 | 0.681 | 2025-07-01 | Yes |

Example statistics are medians over qualifying formation dates only. Labels are historical and are not assertions of present-day listings. These examples were not selected using trading P&L. Full-period frequency is descriptive; a backtest must use the dated candidate membership, not assume all discovered pairs were known from 2018.

## What this establishes

The hypothesis that candidate discovery has been too narrow is supported: there are many liquid, historically related combinations beyond the handpicked and capped roster. It does not establish that these pairs have profitable option-relative-value signals. The existing policy also has restrictive validation and profitability gates, so expanding the candidate roster and changing policy gates are separate experiments.

The first practical expansion is the 269 new industry-peer pairs using already-downloaded names. Next steps are to check actual selected-contract coverage and overlapping training history, build pair-specific features and completed expiry rewards, and evaluate the fixed VIX specification using historical membership. Existing contract caches do not guarantee every pair/date is tradable; the simulator must still check maturity, quotes, contract standardness and corporate actions. No new trading results or Sharpe improvements are claimed here.

## Coverage limits

This is a broader search of the local prepared universe, not a complete scan of every optionable US security. Upstream asset inclusion still depended on standardized-IV availability and historical earnings coverage, including full-history preparation requirements. Liquidity fields also come from those prepared panels, while beta now uses direct underlying returns. These inherited coverage restrictions should be removed in a later full-market data expansion. The original contract cache contains calls only with 0–60 days to expiry and covers 317 requested underlyings.

The screen completed in 25.1 seconds in one process, with peak RSS 0.89 GiB. Formation-time return tests and the full 147-test suite pass. The current trading reference was not changed.

[New industry-peer roster](expanded_pair_discovery/new_industry_peers.csv) · [Peers using downloaded names](expanded_pair_discovery/new_industry_peers_cached_names.csv) · [Dated membership for cached peers](expanded_pair_discovery/cached_peer_historical_membership.csv) · [All historical candidates](expanded_pair_discovery/candidate_screens.csv) · [Formation counts](expanded_pair_discovery/formation_counts.csv) · [Data manifest](expanded_pair_discovery/manifest.json)
