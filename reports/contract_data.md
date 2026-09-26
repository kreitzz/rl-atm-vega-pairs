# Historical contract data for previously explored pairs

Downloaded and checksum-verified **113,735,358 actual call-option quote rows** for **317 of 317 requested underlyings**, covering 358 unique pairs. 358 pairs have at least one overlapping quote date. This is a data-coverage statement, not a claim that every pair/date is tradable.

- Broad-universe names: January 2018–September 2025. The 41 names from the earlier 25 stock/ETF pairs additionally cover 2010–2017 where listed. September 2025 supplies runoff for August entries.
- Calls only, all strikes, 0–60 calendar days to expiry. This covers the current simulator’s 21–45-day entry rule and subsequent held-contract marks. Longer maturities and puts are not part of this cache.
- Raw quote records retain crossed markets, missing Greeks, zero bids, nonstandard settlements and corporate-action fields. No future-outcome, liquidity or profitable-pair filter is applied to the download.
- Includes underlying OHLC, adjustment factors and returns; the interest-rate term structure; and distributions with declaration, ex- and payment dates. This does not provide an as-of earnings-calendar feed, stock-borrow quotes, intraday option depth or broker margin.
- 170 verified Parquet partitions, 4.45 GiB. Checks cover hashes, row counts, keys, request bounds and partition overlap.
- 6,878 crossed quote rows remain in the raw data. 25 requested name-years have underlying records but no calls inside the maturity window; these gaps are explicit in the coverage table.

## Files

Licensed raw files are local in `data/option_contracts/`: `quotes/year=YYYY/batch=ID.parquet`, `underlying/year=YYYY.parquet`, `zero_curve.parquet`, and `distributions.parquet`. Each file has a request/checksum manifest. Raw data are git-ignored; collaborators need their own authorized WRDS access to reproduce the download.

[Pair roster](contract_data/pairs.csv) · [Underlying roster](contract_data/underlyings.csv) · [Name/year coverage](contract_data/coverage.csv) · [Pair overlap coverage](contract_data/pair_coverage.csv) · [Partition manifest](contract_data/partitions.csv)

```sh
python -m scripts.download_pair_contract_history --workers 1
python -m scripts.audit_contract_history
```

The downloader resumes verified completed partitions and retries failed requests. One worker and 5,000-row batches are the default; checksums stream in 1 MiB buffers. The audit also streams batches and checks duplicate keys on disk with a 4 MiB SQLite cache. `iter_contract_batches(root, secids, start, end)` streams a chosen pair and interval; `read_contracts` is capped at 250,000 rows by default. Names are keyed by OptionMetrics SECID; ticker labels can be historical. The roster merges the broad 2018+ screens with the earlier stock and ETF configurations. Candidates with insufficient prior model training are retained. Pre-2010 contract training history is not included.
