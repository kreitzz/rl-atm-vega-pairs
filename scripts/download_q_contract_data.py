"""Download only one requested pair, reusing the resumable WRDS contract cache.

Use conda environment mfe_wrds. Authentication is libpq's existing local store;
no password argument, prompt, or password file is created by this script.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from scripts.download_pair_contract_history import download


def jobs(secids, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start.year < 1996:
        raise ValueError('Invalid OptionMetrics date interval')
    ids = sorted(set(int(s) for s in secids))
    if len(ids) != 2:
        raise ValueError('Exactly two different SECIDs are required')
    result = []
    for year in range(start.year, end.year+1):
        lo, hi = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
        common = dict(secids=ids, start=str(lo.date()), end=str(hi.date()), year=year)
        result += [dict(kind='quotes', table=f'optionm.opprcd{year}',
                        path=f'quotes/year={year}/pair.parquet', **common),
                   dict(kind='underlying', table=f'optionm.secprd{year}',
                        path=f'underlying/year={year}.parquet', **common)]
    result += [dict(kind='rates', table='optionm.zerocd', start=str(start.date()),
                    end=str(end.date()), path='zero_curve.parquet'),
               dict(kind='distributions', table='optionm.distrd', secids=ids,
                    start=str(start.date()), end=str(end.date()), path='distributions.parquet')]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--secids', type=int, nargs=2, default=[101121, 108321])
    ap.add_argument('--start', default='2010-01-01')
    ap.add_argument('--end', default='2025-09-30')
    ap.add_argument('--output', type=Path, default=Path('data/q_contracts/AMD_NVDA'))
    ap.add_argument('--username', default=os.getenv('WRDS_USERNAME'))
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    requested = jobs(args.secids, args.start, args.end)
    args.output.mkdir(parents=True, exist_ok=True)
    request_file = args.output/'requests.json'
    if request_file.exists() and json.loads(request_file.read_text()) != requested:
        raise ValueError('This cache has a different request. Use a separate output directory.')
    request_file.write_text(json.dumps(requested, indent=2)+'\n')
    if args.plan_only:
        print(f'{len(requested)} partitions for two underlyings, {args.start} through {args.end}')
        return
    if not args.username:
        ap.error('Set WRDS_USERNAME or supply --username; do not supply passwords on the command line')
    for i, job in enumerate(requested):
        result = download(job, args.output, args.username)
        print(f"{i+1}/{len(requested)} {result['status']}: {job['path']} ({result['rows']:,} rows)", flush=True)


if __name__ == '__main__':
    main()
