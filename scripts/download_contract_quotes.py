"""Download licensed, unfiltered daily call histories for a bounded contract pilot.

Requires psycopg2 (available with wrds). Uses libpq/.pgpass; never prompts for or
stores passwords. Raw quotes stay under ignored artifacts/. Query all strikes
and maturities so selection and subsequent marks cannot introduce survivorship.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/contract_pilot.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/contracts/quotes'))
    parser.add_argument('--username', default=os.environ.get('WRDS_USERNAME', getpass.getuser()))
    args = parser.parse_args()
    import psycopg2
    cfg = json.loads(args.config.read_text())
    start, end = pd.Timestamp(cfg['data_start']), pd.Timestamp(cfg['data_end'])
    ids = [int(x) for x in cfg['secids']]
    args.output.mkdir(parents=True, exist_ok=True)
    # Explicit libpq connection avoids the WRDS package's interactive fallback.
    with psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737,
                          dbname='wrds', user=args.username, sslmode='require',
                          connect_timeout=15) as db:
        for year in range(start.year, end.year + 1):
            query = f'''select secid, date, optionid, exdate, cp_flag, strike_price,
                best_bid, best_offer, volume, open_interest, impl_volatility,
                delta, gamma, vega, theta, ss_flag, cfadj, contract_size
                from optionm.opprcd{year}
                where secid = ANY(%s) and date between %s and %s and cp_flag = 'C'
                order by date, secid, optionid'''
            with db.cursor() as cur:
                cur.execute(query, (ids, start.date(), end.date()))
                frame = pd.DataFrame(cur.fetchall(), columns=[x.name for x in cur.description])
            if frame.empty:
                raise RuntimeError(f'No contract quotes returned for {year}')
            path = args.output / f'year={year}.parquet'
            frame.to_parquet(path, index=False)
            manifest = dict(source=f'OptionMetrics optionm.opprcd{year}', secids=ids,
                            start=str(start.date()), end=str(end.date()), rows=len(frame),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            license='Licensed WRDS/OptionMetrics; no redistribution',
                            query=query)
            path.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'{year}: {len(frame):,} contract quotes saved to {path}', flush=True)


if __name__ == '__main__':
    main()
