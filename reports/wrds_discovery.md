# WRDS discovery scan — October 1, 2026

The live WRDS login can read several datasets that support a different research direction. The most practical first experiment is an information-driven stock portfolio using SEC disclosure changes and analyst revisions. Revelio workforce and employee sentiment are the most novel alternatives, but their historical publication and revision practices need validation before backtesting. These are research hypotheses; this scan did not measure predictive power or trading returns.

## Verified shortlist

Each table below returned a successful, nonempty read under the current login. Priority reflects implementation effort and usefulness for a first experiment, rather than measured alpha.

| Priority | Dataset | Concrete signal candidates | Strategy to test |
|---|---|---|---|
| 1 | SEC filing sentiment and text similarity | Change in negative/uncertain language; lower similarity to the comparable previous filing | Trade stocks after disclosure, with sector and market risk controls; test whether information is incorporated gradually |
| 2 | I/B/E/S forecasts and actuals | Revision breadth and magnitude; forecast dispersion; earnings surprise after announcement | Analyst-revision and post-earnings stock portfolios; combine with filing information |
| 3 | Capital IQ earnings-call transcripts | Management tone changes; uncertainty; Q&A versus prepared remarks; guidance changes | Post-call stock signals, supplemented by analyst revisions |
| 4 | Revelio workforce, job postings, reviews and layoffs | Hiring acceleration; abnormal attrition; employee business-outlook changes; publicly announced layoffs | Monthly stock selection or event strategies, subject to a point-in-time data audit |
| 5 | Insider filings | Clustered open-market buying; unusual purchase size | Stock event signals after public filing, excluding grants, exercises and other non-market transactions |
| 6 | Academic bond and liquidity datasets | Issuer credit-spread changes versus stock moves; liquidity regime changes | Equity-credit information signals, or a separate credit strategy with realistic execution |

I/B/E/S and insider filings are conventional financial information, rather than alternative data. Workforce and reviews offer the strongest departure from our current price/volatility features. SEC and transcript text offer a middle ground with relatively transparent economic interpretation.

## Tables, timestamps and history

The earliest and latest dates below are table-level observations or clearly labeled metadata estimates. They do not establish continuous history, coverage for every company, or historical availability of the current version of a record.

### SEC filing changes

Readable tables: `wrdssec_secsa.wrds_nlpsa`, `wrdssec_secsa.bow_cossim`, and `wrdssec_secsa.bow_filingsummary`.

The sentiment table includes word counts, Loughran–McDonald negative/positive/uncertainty categories and readability measures. The similarity table includes `accession`, `accession2`, `lookback`, `cosine`, `jaccard` and edit distance. Filing metadata includes `secadate`, `secatime` and `secpdate`, which help establish public availability. Match comparable forms and reporting periods; do not assume every reduction in similarity is bad news or that a whole-document measure isolates risk-factor changes.

Observed sentiment `fdate` bounds: January 2, 1980–September 30, 2026. The unusually early bound should not be interpreted as regular electronic filing coverage beginning in 1980. WRDS describes its bag-of-words coverage as beginning in 1993. Audit form and issuer coverage before choosing the research period. [WRDS bag-of-words description](https://wrds-www.wharton.upenn.edu/pages/data-announcements/wrds-bag-words-available/).

The economic hypothesis has an academic precedent: changes in corporate disclosures can contain information investors incorporate slowly. That precedent does not establish an edge in our universe or test period. [Cohen, Malloy and Nguyen, *Lazy Prices*](https://www.nber.org/papers/w25084).

### Analyst revisions and earnings

Readable tables: `tr_ibes.det_epsus`, `tr_ibes.statsum_epsus`, and `tr_ibes.act_epsus`.

Detail forecasts include estimate announcement (`anndats`, `anntims`) and vendor activation (`actdats`, `acttims`) timestamps, analyst identifiers, `value` and forecast period. Summary data include estimate counts, up/down revisions, consensus and dispersion. The product supports both individual and consensus estimates. [WRDS I/B/E/S overview](https://wrds-www.wharton.upenn.edu/pages/grid-items/overview-ibes-demo/).

Observed activation-date bounds for detailed EPS estimates: January 28, 1980–August 20, 2026. Observed summary `statpers` bounds: January 15, 1976–August 20, 2026.

Use estimates only after their availability timestamp. Forecast rows also contain subsequent actual earnings fields: exclude them until the earnings release is public. Check adjustment conventions and historical security links; unadjusted estimate tables are catalogued and may avoid later split-adjustment issues when combined with contemporary split data.

### Earnings-call transcripts

Readable tables: `ciq_transcripts.wrds_transcript_detail` and `ciq_transcripts.ciqtranscriptcomponent`. Components contain ordered text and speaker/type identifiers. Detail metadata contains company, event, collection type and creation timestamps. [WRDS transcript description](https://wrds-www.wharton.upenn.edu/pages/grid-items/capital-iq-transcripts/).

Approximate PostgreSQL statistics show event dates around September 2004–September 2026 and transcript creation timestamps around June 2008–September 2026. These are histogram endpoints, not exact coverage bounds; expensive full scans were skipped. The vendor advertises earlier North American coverage, so neither source establishes a clean common start date for this account. [Vendor coverage documentation](https://wrds-www.wharton.upenn.edu/documents/1305/SP_CapIQ_TRANSCRIPTS.pdf).

Trade after transcript creation/publication, rather than assuming the full transcript existed at the call's start. Audit initial and revised collection versions and issuer links. Start with dictionary or small text-model features on a bounded universe, rather than processing the whole archive.

### Revelio labor and employee sentiment

Readable tables:

- `revelio_workforce_dynamics.workforce_dynamics_geo`: monthly headcount, inflow/outflow, external moves, salary and tenure by geography, role and seniority.
- `revelio_job_postings.postings_cosmos`: posting dates, removal dates, role, seniority, salary and expected hiring.
- `revelio_sentiment.sentiment_individual_reviews`: dated reviews, overall rating and business-outlook/leadership ratings.
- `revelio_layoffs.layoffs`: notice date, effective layoff date and affected employee count.
- `revelio_common.company_mapping`: issuer identifiers and company/parent mappings.

WRDS advertises workforce coverage from 2008. Exact observed bounds: review dates January 25, 2008–September 2, 2026; posting dates January 1, 2008–September 6, 2026; layoff notice dates March 1, 1989–September 9, 2026. Posting sources and coverage have changed over time: an early timestamp is not proof of a continuous 2008 history for the unified feed. [WRDS Revelio description](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/revelio-labs/).

The vendor describes monthly delivery of the previous month's data on the 15th. However, resume-based workforce histories can be reconstructed or revised. A fixed publication lag alone does **not** remove hindsight from backfilled employment histories. Confirm archived vintages or a documented point-in-time product before treating these as historical trading inputs. [Revelio data FAQ](https://www.data-dictionary.reveliolabs.com/faq.html).

Use dated individual reviews, with sufficient review counts and shrinkage; the catalogued company-level `sentiment_scores` table has no date and is unsuitable for historical states. Use layoff notice availability, rather than future effective dates. Posting removal dates are unavailable before removal. Validate historical company mappings and source composition.

Workforce and job-posting tables contain billions of estimated rows. Extract a small issuer universe with server-side aggregation; never download full tables for this experiment.

### Insider transactions and institutional holdings

`tr_insiders.table1` and `wrdssec_insiders.document_info` are readable. Thomson fields include transaction and filing dates, transaction codes, shares, prices and ownership roles. SEC metadata supplies acceptance/publication date fields and a 10b5-1 indicator. Anchor signals to disclosure and reconcile amendments; transaction date is not the date investors learned of a trade. Access here is to these legacy/SEC tables, not a verified subscription to every newer WRDS insider product.

Thomson filing-date histogram endpoints are approximately August 2007–September 2026; exact bounds were skipped to avoid a large scan.

`tr_13f.s34type3` is also readable, with observed filing-date bounds March 31, 1980–March 31, 2026. Holdings are reported with delay. Quarter-end dates and vendor filing fields should not be assumed to be actual public acceptance timestamps.

### Credit and liquidity

Readable `contrib_corporate_bond_returns.bonds` supplies **monthly** bond returns, an equity `permno` link, duration, yield spread, rating and a value signal. `contrib_bond_dickerson.dickerson_bonds_daily` supplies daily prices, credit spreads, duration, trade count and bid/ask-related fields. `wrdsapps_bondret.bondret` supplies monthly returns, yield, duration and stale-price indicators. A credit-to-equity signal could use bond information while trading stocks, but stale marks, timing and cross-sectional coverage require validation.

Readable `contrib_liquidity_taq.bbd` and `.ilc` are **monthly derived measures**, not daily order-flow data. They contain price-impact and institutional liquidity-cost estimates. Treat them as lagged liquidity controls, not evidence that we downloaded or validated raw intraday TAQ.

## Products not available through this login

Direct reads were denied for FINRA short interest/short volume, Markit CDS, Estimize consensus, ETF Global fund flows, Zacks consensus and FactSet Revere supply-chain relationships. Schema metadata also indicates no full RavenPack news entitlement or Markit securities-finance access. RavenPack's trial equities table is readable, but trial access is not a verified long-history feed.

Some public-facing aliases show table SELECT permission while the underlying product schema denies access. The shortlist above therefore uses actual read tests, not catalog visibility alone. These are the tested products, not a definitive statement about every historical variant offered by each vendor.

## Proposed first experiment

1. Freeze the existing option checkpoint. Build a separate stock experiment using SEC text changes plus I/B/E/S revision/surprise features, which have explicit availability fields and need relatively little text infrastructure.
2. Establish issuer coverage and chronological availability for a broad, historically defined liquid-stock universe. Test 5-, 20- and 60-trading-day forward residual returns; separate pre-earnings information from information first released at earnings.
3. Use training-only signal selection, rolling validation and a final untouched period. Purge overlapping outcome windows. Measure signal rank correlation, stability by year/sector and economic returns after costs before training a trading policy.
4. Compare a simple supervised portfolio with RL sizing and exits using identical features, available assets, costs, risk limits and retraining schedules. Control sector and market exposure and account for short borrowing. Select neither signals nor policies on final-test Sharpe.
5. Add transcripts if they contribute incrementally. Add workforce/reviews only after confirming publication and historical-vintage validity.

RL may improve portfolio decisions once a predictive signal exists. This experiment should first determine whether these inputs forecast anything tradable; it does not assume a Sharpe target or a guaranteed improvement over the saved option result.

## Scan evidence and resource use

The scan used one read-only WRDS connection, an eight-second per-statement timeout and a two-second lock timeout. It read catalog/column metadata and performed one-row existence probes. Date bounds used indexed or small-table plans; large scan plans were skipped. Peak local memory observed was 300,990,464 bytes, approximately 287 MiB. No full dataset or training job was launched.

- [Schema permissions](wrds_discovery/schemas.csv)
- [Exact read results](wrds_discovery/probes.json)
- [Column metadata](wrds_discovery/columns.json)
- [Date-bound results and skipped plans](wrds_discovery/date_bounds.json)
- [Catalog and scan timestamp](wrds_discovery/catalog.json)
- [Reusable scan script](../scripts/scan_wrds_catalog.py)

These files contain discovery metadata and access results, not a redistributed licensed dataset.
