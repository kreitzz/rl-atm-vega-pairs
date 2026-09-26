# Historical contract data for previously explored pairs

Downloaded and checksum-verified **153,309,227 actual call-option quote rows** for **317 of 317 requested underlyings**, covering 358 unique pairs. 358 pairs have at least one overlapping quote date. This is a data-coverage statement, not a claim that every pair/date is tradable.

- All 317 underlyings request January 2010–September 2025: up to 15.75 years of history. Actual coverage depends on listing, delisting and option availability. September 2025 supplies runoff for August entries.
- This extends a fixed retrospective research roster; it does not make the earlier pair-selection process point-in-time or establish an untouched test sample.
- 287 pairs span at least ten calendar years of overlapping quote history; 287 also have at least 2,520 common quote dates. A long first-to-last span alone does not establish continuous coverage. Short histories remain visible rather than being silently excluded.
- Calls only, all strikes, 0–60 calendar days to expiry. This covers the current simulator’s 21–45-day entry rule and subsequent held-contract marks. Longer maturities and puts are not part of this cache.
- Raw quote records retain crossed markets, missing Greeks, zero bids, nonstandard settlements and corporate-action fields. No future-outcome, liquidity or profitable-pair filter is applied to the download.
- Includes underlying OHLC, adjustment factors and returns; the interest-rate term structure; and distributions with declaration, ex- and payment dates. This does not provide an as-of earnings-calendar feed, stock-borrow quotes, intraday option depth or broker margin.
- 274 verified Parquet partitions, 5.97 GiB. Checks cover hashes, row counts, keys, request bounds and partition overlap.
- 7,537 crossed quote rows remain in the raw data. 55 requested name-years have underlying records but no calls inside the maturity window; these gaps are explicit in the coverage table.

## Files

Licensed raw files are local in `data/option_contracts/`: `quotes/year=YYYY/batch=ID.parquet`, `underlying/year=YYYY.parquet`, `zero_curve.parquet`, and `distributions.parquet`. Each file has a request/checksum manifest. `requests.json` identifies the active partitions; older superseded batches remain on disk but are excluded by the reader and audit to prevent duplicates. Raw data are git-ignored; collaborators need their own authorized WRDS access to reproduce the download.

[Pair roster](contract_data/pairs.csv) · [Underlying roster](contract_data/underlyings.csv) · [Name/year coverage](contract_data/coverage.csv) · [Pair overlap coverage](contract_data/pair_coverage.csv) · [Partition manifest](contract_data/partitions.csv)

```sh
python -m scripts.download_pair_contract_history --workers 1
python -m scripts.audit_contract_history
```

The downloader resumes verified completed partitions and retries failed requests. One worker and 5,000-row batches are the default; checksums stream in 1 MiB buffers. The audit also streams batches and checks duplicate keys on disk with a 4 MiB SQLite cache. `iter_contract_batches(root, secids, start, end)` streams a chosen pair and interval; `read_contracts` is capped at 250,000 rows by default. Names are keyed by OptionMetrics SECID; ticker labels can be historical. The roster merges the broad 2018+ screens with the earlier stock and ETF configurations. Candidates with insufficient prior model training are retained. Pre-2010 contract training history is not included.
