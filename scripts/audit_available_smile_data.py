"""Bounded raw-put coverage audit; no retraining or trading simulation.

Raw bucket diagnostics use vendor delta; the AMD/NVDA pilot separately refits
Black-Scholes deltas to reproduce the selector and isolate its restrictions.
"""
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.prepare_curve_contracts import anchors
from scripts.option_flex.black_scholes import fit_quotes
from scripts.run_broad_contract_replay import ROOT

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = Path('artifacts/contracts/smile_data_audit')
REPORT = Path('reports/smile_data_audit')
RAW = Path('data/curve_trading/puts_raw')
START, END = pd.Timestamp('2018-01-01'), pd.Timestamp('2025-06-30')
PILOT = [101121, 108321]
STAGES = ['raw_anchor_expiry', 'vendor_bucket', 'positive_standard_quote', 'oi100', 'oi100_width25', 'oi10_width25', 'oi1_width25', 'any21_45_oi100_width25', 'any21_45_oi10_width25', 'saved_bs_selection', 'current_finite_iv']


class Budget:
    def __init__(self):
        self.wall, self.cpu = time.monotonic(), time.process_time()
        self.peak = 0
        try:
            os.nice(15)
        except PermissionError:
            pass

    def check(self):
        self.peak = max(self.peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if self.peak > 3 * 1024**3:
            raise MemoryError('3 GiB RSS stop')
        if Path('artifacts/contracts/signal_discovery/STOP').exists():
            raise InterruptedError('STOP file present')
        if time.monotonic() - self.wall > 1200:
            raise TimeoutError('20-minute limit; partition checkpoints retained')
        delay = 2 * (time.process_time() - self.cpu) - (time.monotonic() - self.wall)
        while delay > 0:
            time.sleep(min(delay, .5))
            delay = 2 * (time.process_time() - self.cpu) - (time.monotonic() - self.wall)


def read(path, columns=None, filters=None):
    return pq.ParquetFile(path).read(columns=columns, use_threads=False).to_pandas() if filters is None else pq.read_table(path, columns=columns, filters=filters, use_threads=False).to_pandas()


def verify_saved():
    """Reconcile cached IV presence to the existing pair coverage report."""
    budget = Budget()
    rows = read(OUT / 'asset_days.parquet')
    assert not rows.duplicated(['date', 'secid', 'wing']).any()
    for lo, hi in [('vendor_bucket','raw_anchor_expiry'), ('positive_standard_quote','vendor_bucket'),
                   ('oi100','positive_standard_quote'), ('oi100_width25','oi100'),
                   ('oi100_width25','oi10_width25'), ('oi10_width25','oi1_width25'),
                   ('current_finite_iv','saved_bs_selection')]:
        assert not (rows[lo] & ~rows[hi]).any(), (lo, hi)
    defs = json.loads(Path('artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']
    dates = pd.DatetimeIndex(sorted(rows.date.unique()))
    record = []
    for wing in ['P25','P75']:
        matrices = {stage: rows.loc[rows.wing.eq(wing)].pivot(index='date',columns='secid',values=stage).reindex(dates).eq(True) for stage in STAGES}
        for d in defs:
            sa, sb = d['secids']
            counts = {}
            for stage, matrix in matrices.items():
                counts[stage] = int((matrix[sa].to_numpy() & matrix[sb].to_numpy()).sum()) if sa in matrix and sb in matrix else 0
            record.append(dict(pair=d['pair'], wing=wing, **counts))
            budget.check()
    pairs = pd.DataFrame(record)
    pairs.to_csv(REPORT / 'pair_day_coverage.csv', index=False)
    original = pd.read_csv('reports/uniform_smile/coverage.csv')
    check = pairs[['pair','wing','current_finite_iv']].merge(original.loc[original.wing.isin(['P25','P75']), ['pair','wing','observed_test_rows']], on=['pair','wing'], validate='one_to_one')
    check['difference'] = check.current_finite_iv - check.observed_test_rows
    check.to_csv(REPORT / 'coverage_reconciliation.csv', index=False)
    assert check.difference.eq(0).all(), check.loc[check.difference.ne(0)].head().to_dict('records')
    q = read(OUT / 'pilot_fitted_quotes.parquet', ['date','secid','expiry','anchor_expiry','bs_delta','bs_vega','bs_status','bid','ask','open_interest'])
    pilot_counts = []
    for bucket in [25,75]:
        target = q.bs_delta.abs().sub(bucket/100).abs().le(.075) & q.bs_status.eq('ok') & q.bs_vega.gt(0)
        same = q.expiry.eq(q.anchor_expiry)
        width = ((q.ask-q.bid)/((q.ask+q.bid)/2)).le(.25)
        for stage, mask in dict(oi100_same_expiry=target & same & width & q.open_interest.ge(100),
                               oi10_same_expiry=target & same & width & q.open_interest.ge(10),
                               oi1_same_expiry=target & same & width & q.open_interest.ge(1),
                               oi100_any21_45=target & width & q.open_interest.ge(100),
                               oi10_any21_45=target & width & q.open_interest.ge(10)).items():
            counts = q.loc[mask, ['date','secid']].drop_duplicates().groupby('date').size()
            pilot_counts.append(dict(wing=f'P{bucket}', stage=stage, pair_days=int(counts.eq(2).sum())))
        expected = int(pairs.loc[pairs.pair.eq('101121__108321') & pairs.wing.eq(f'P{bucket}'), 'current_finite_iv'].iloc[0])
        assert pilot_counts[-5]['pair_days'] == expected, (bucket, pilot_counts[-5], expected)
        budget.check()
    pd.DataFrame(pilot_counts).to_csv(REPORT / 'pilot_pair_coverage.csv', index=False)
    (REPORT / 'verification.json').write_text(json.dumps(dict(status='complete', reconciled_pair_wings=len(check), exact_iv_coverage=True, exact_pilot_current_selector_coverage=True, peak_rss_bytes=budget.peak), indent=2))
    print(pairs.groupby('wing')[STAGES].sum().to_string())
    print(pd.DataFrame(pilot_counts).to_string(index=False))
    print('Every pair/wing IV-coverage count reconciles exactly to the existing report.')


def main():
    budget = Budget()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((RAW / 'requests.json').read_text())
    inventory = []
    for job in jobs:
        path = RAW / job['path']
        meta = path.with_suffix('.json')
        saved = json.loads(meta.read_text()) if meta.exists() else {}
        inventory.append(dict(year=job['year'], path=str(path), data_exists=path.exists(), metadata_exists=meta.exists(), recorded_bytes_match=path.exists() and path.stat().st_size == saved.get('bytes'), rows=pq.ParquetFile(path).metadata.num_rows if path.exists() else 0, recorded_rows=saved.get('rows'), sha256=saved.get('sha256')))
    pd.DataFrame(inventory).to_csv(REPORT / 'download_inventory.csv', index=False)
    assert all(x['data_exists'] and x['metadata_exists'] and x['recorded_bytes_match'] and x['rows'] == x['recorded_rows'] for x in inventory)
    jobs = [j for j in jobs if j['year'] >= START.year]
    columns = ['secid', 'date', 'optionid', 'exdate', 'strike_price', 'best_bid', 'best_offer', 'open_interest', 'delta', 'ss_flag', 'contract_size']
    parts = []
    for number, job in enumerate(jobs, 1):
        path = RAW / job['path']
        dest = OUT / f'{job["year"]}_{path.stem}.parquet'
        pilot_path = dest.with_suffix('.pilot.parquet')
        stamp = dest.with_suffix('.audit.json')
        if dest.exists() and stamp.exists() and json.loads(stamp.read_text()).get('version') == 2:
            parts.append(read(dest))
            continue
        a = anchors(job)
        a = a.loc[a.tenor.eq(30) & a.date.between(START, END)].copy()
        assert not a.duplicated(['date', 'secid']).any()
        a = a[['date', 'secid', 'expiry']].rename(columns={'expiry': 'anchor_expiry'}).reset_index(drop=True)
        a['row'] = np.arange(len(a))
        flags = np.zeros((len(a), 2, len(STAGES)), bool)
        pilot = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=columns, use_threads=False):
            q = batch.to_pandas()
            q['date'], q['exdate'] = pd.to_datetime(q.date), pd.to_datetime(q.exdate)
            q = q.merge(a, on=['date', 'secid'], how='inner', validate='many_to_one')
            if len(q):
                idx = q.row.to_numpy(int)
                same = q.exdate.eq(q.anchor_expiry)
                dte = (q.exdate - q.date).dt.days
                quote = q.ss_flag.astype(str).eq('0') & q.contract_size.eq(100) & q.best_bid.gt(0) & q.best_offer.ge(q.best_bid)
                width = (q.best_offer - q.best_bid) / ((q.best_offer + q.best_bid) / 2)
                liquid = width.le(.25)
                for wi, delta in enumerate([.25, .75]):
                    bucket = q.delta.abs().sub(delta).abs().le(.075)
                    masks = [same, same & bucket, same & bucket & quote, same & bucket & quote & q.open_interest.ge(100), same & bucket & quote & q.open_interest.ge(100) & liquid, same & bucket & quote & q.open_interest.ge(10) & liquid, same & bucket & quote & q.open_interest.ge(1) & liquid, bucket & quote & q.open_interest.ge(100) & liquid & dte.between(21,45), bucket & quote & q.open_interest.ge(10) & liquid & dte.between(21,45)]
                    for stage, mask in enumerate(masks):
                        np.logical_or.at(flags[:, wi, stage], idx, mask.to_numpy(bool))
                keep = q.secid.isin(PILOT) & quote & q.open_interest.ge(1) & dte.between(21, 45)
                pilot.append(q.loc[keep].copy())
            budget.check()
        keys = pd.MultiIndex.from_frame(a[['date', 'secid']])
        for sid in job['secids']:
            native = Path('data/curve_trading/selection/P') / str(job['year']) / f'{sid}.parquet'
            current = Path('artifacts/contracts/uniform_smile/market/assets') / f'{sid}_selection.parquet'
            for stage, source in [(9, native), (10, current)]:
                if not source.exists():
                    continue
                cols = ['date', 'secid', 'tenor', 'bucket', 'bs_iv'] + (['cp_flag'] if stage == 10 else [])
                selection = read(source, cols)
                selection = selection.loc[selection.tenor.eq(30) & selection.date.between(START, END)]
                if stage == 10:
                    selection = selection.loc[selection.cp_flag.eq('P') & np.isfinite(selection.bs_iv) & selection.bs_iv.gt(0)]
                for wi, bucket in enumerate(['25', '75']):
                    subset = selection.loc[selection.bucket.eq(bucket), ['date', 'secid']]
                    flags[:, wi, stage] |= keys.isin(pd.MultiIndex.from_frame(subset))
                budget.check()
        frames = []
        for wi, wing in enumerate(['P25', 'P75']):
            frame = a[['date', 'secid']].copy()
            frame['wing'] = wing
            frame[STAGES] = flags[:, wi]
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(dest, index=False)
        if pilot and any(len(x) for x in pilot):
            pd.concat(pilot, ignore_index=True).to_parquet(pilot_path, index=False)
        stamp.write_text(json.dumps(dict(version=2, raw_source=str(path), raw_metadata=str(path.with_suffix('.json')))))
        parts.append(frame)
        totals = frame.groupby('wing')[STAGES].sum()
        print(f'Raw partition {number}/{len(jobs)} year={job["year"]}: {len(a)} anchor asset-days; P25={totals.loc["P25", "oi100_width25"]}, P75={totals.loc["P75", "oi100_width25"]}', flush=True)
        budget.check()
    all_rows = pd.concat(parts, ignore_index=True)
    all_rows.to_parquet(OUT / 'asset_days.parquet', index=False)
    summary = all_rows.groupby('wing')[STAGES].sum()
    summary['anchor_asset_days'] = all_rows.groupby('wing').size()
    summary.to_csv(REPORT / 'asset_day_coverage.csv')
    all_rows.groupby(['secid', 'wing'])[STAGES].sum().to_csv(REPORT / 'name_coverage.csv')
    print(summary.to_string(), flush=True)
    # Exact BS reconstruction on two liquid names, with sensitivity flags.
    candidate_paths = sorted(OUT.glob('*.pilot.parquet'))
    if candidate_paths:
        q = pd.concat([read(p) for p in candidate_paths], ignore_index=True).drop_duplicates(['date', 'optionid'])
        q = q.rename(columns={'exdate': 'expiry', 'best_bid': 'bid', 'best_offer': 'ask', 'contract_size': 'multiplier'})
        q['strike'], q['standard'] = q.strike_price / 1000, q.ss_flag.astype(str).eq('0')
        q['cp_flag'] = 'P'
        stocks = read('artifacts/contracts/uniform_smile/market/stocks.parquet', ['date','secid','close'])
        stocks = stocks.loc[stocks.secid.isin(PILOT)]
        zero, divs = read(ROOT / 'zero_curve.parquet'), read(ROOT / 'distributions.parquet')
        fitted = []
        for offset in range(0, len(q), 4096):
            part = fit_quotes(q.iloc[offset:offset+4096].drop(columns=['row','anchor_expiry']), stocks, zero, divs)
            part['anchor_expiry'] = q.iloc[offset:offset+4096].anchor_expiry.to_numpy()
            fitted.append(part)
            budget.check()
            if offset % 16384 == 0:
                print(f'AMD/NVDA BS audit {offset}/{len(q)} quotes', flush=True)
        q = pd.concat(fitted, ignore_index=True)
        q.to_parquet(OUT / 'pilot_fitted_quotes.parquet', index=False)
        records = []
        for sid in PILOT:
            base = all_rows.loc[all_rows.secid.eq(sid) & all_rows.wing.eq('P25'), ['date']].drop_duplicates()
            for bucket in [25,75]:
                z = q.loc[q.secid.eq(sid)].copy()
                vendor = z.delta.abs().sub(bucket/100).abs().le(.075)
                target = z.bs_delta.abs().sub(bucket/100).abs().le(.075) & z.bs_status.eq('ok') & z.bs_vega.gt(0)
                same = z.expiry.eq(z.anchor_expiry)
                width = ((z.ask-z.bid)/((z.ask+z.bid)/2)).le(.25)
                masks = dict(vendor_target_same=vendor & same,
                    vendor_target_bs_failure=vendor & same & z.bs_status.ne('ok'),
                    vendor_target_bs_delta_outside=vendor & same & z.bs_status.eq('ok') & ~target,
                    bs_target_same_oi1=target & same & width & z.open_interest.ge(1),
                    bs_target_same_oi10=target & same & width & z.open_interest.ge(10),
                    bs_target_same_oi100=target & same & width & z.open_interest.ge(100),
                    bs_target_any21_45_oi100=target & width & z.open_interest.ge(100))
                for name, mask in masks.items():
                    records.append(dict(secid=sid, wing=f'P{bucket}', stage=name, asset_days=int(z.loc[mask,'date'].nunique()), anchor_asset_days=len(base)))
        pd.DataFrame(records).to_csv(REPORT / 'pilot_bs_coverage.csv', index=False)
        print(pd.DataFrame(records).to_string(index=False), flush=True)
    audit = dict(status='complete', raw_partitions=256, scanned_evaluation_partitions=len(jobs), start=str(START.date()), end=str(END.date()), raw_delta_diagnostics='Vendor delta, not BS delta; exact BS pilot is separate.', no_retraining=True, no_backtest=True, downloaded_files_checked='Existence, recorded bytes and parquet-footer row counts; download hashes retained, not reread.', resource_limits='One worker, one Arrow/BLAS thread, self-throttled to 50% process CPU, 3 GiB RSS stop.', peak_rss_bytes=budget.peak, elapsed_seconds=time.monotonic()-budget.wall)
    (REPORT / 'audit.json').write_text(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
