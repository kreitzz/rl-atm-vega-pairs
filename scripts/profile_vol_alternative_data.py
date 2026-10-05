"""Bounded, read-only coverage checks for new option-risk signals."""
from contextlib import closing
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import resource

import pandas as pd
import psycopg2

from scripts.scan_wrds_catalog import rows, write_json


def main():
    os.nice(15)
    out = Path('reports/vol_alternative_data'); out.mkdir(parents=True, exist_ok=True)
    cache = Path('artifacts/contracts/vol_alternative_data'); cache.mkdir(parents=True, exist_ok=True)
    links = pd.read_parquet('artifacts/contracts/wrds_pair_signals/company_links.parquet')
    ciks = sorted({str(int(str(c).strip())) for c in links.cik.dropna()})
    keys = sorted(set(ciks + [c.zfill(10) for c in ciks]))
    tables = [('late_filings', 'audit_audit_comp', 'feed20_nt',
               'company_fkey, file_date, file_accepted, form_fkey', 'file_date'),
              ('restatements', 'audit_audit_comp', 'feed39_financial_restatements',
               'company_fkey, event_date, file_date, file_accepted, date_of_8k_402', 'event_date')]
    profiles = []
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737,
                 dbname='wrds', user=os.getenv('WRDS_USERNAME', getpass.getuser()),
                 sslmode='require', connect_timeout=20,
                 application_name='bounded_vol_alternative_profile')) as db:
        db.set_session(readonly=True, autocommit=True)
        with db.cursor() as cursor:
            cursor.execute("SET statement_timeout='8s'")
            cursor.execute("SET lock_timeout='2s'")
        for name, schema, table, columns, date_column in tables:
            try:
                records = rows(db, f'SELECT {columns} FROM {schema}.{table} '
                               f'WHERE company_fkey=ANY(%s) AND {date_column}>=%s '
                               f'AND {date_column}<=%s LIMIT 20001', (keys, '2010-01-01', '2025-08-29'))
                if len(records) > 20000: raise MemoryError('Coverage row cap exceeded')
                frame = pd.DataFrame(records)
                frame.to_parquet(cache / (name + '.parquet'), index=False)
                dates = pd.to_datetime(frame[date_column]) if len(frame) else pd.Series(dtype='datetime64[ns]')
                ids = {str(int(str(c).strip())) for c in frame.company_fkey} if len(frame) else set()
                profiles.append(dict(source=name, table=schema+'.'+table, status='readable', rows=len(frame),
                                     matched_company_keys=len(ids), first_observed=str(dates.min()),
                                     last_observed=str(dates.max()), before_2022=int((dates<'2022-01-01').sum()),
                                     from_2022=int((dates>='2022-01-01').sum())))
            except (psycopg2.Error, MemoryError) as exc:
                profiles.append(dict(source=name, table=schema+'.'+table, status='failed', error=str(exc)))
        metadata = rows(db, '''SELECT table_schema AS schema, table_name AS table,
                               column_name AS column, data_type AS type
                          FROM information_schema.columns
                         WHERE (table_schema='optionm' AND table_name IN ('borrate2024','opvold'))
                            OR (table_schema='audit_corp_legal' AND table_name IN
                                ('f14_lit_legal_case','f14_lit_legal_party'))
                         ORDER BY table_schema,table_name,ordinal_position''')
    write_json(out/'columns.json', metadata)
    pd.DataFrame(profiles).to_csv(out/'coverage.csv', index=False)
    write_json(out/'manifest.json', dict(scanned_at=datetime.now(timezone.utc).isoformat(),
        coverage_scope='Historical cached company links from the 128-pair signal screen, not full 345-pair coverage.',
        queried_ciks=len(ciks), company_key_assumption='Company keys matched to padded/unpadded CIK; identifier audit remains required.',
        statement_timeout_seconds=8, row_cap_per_source=20000, readonly=True,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        task='Source access and event coverage only. No profitability test or policy retraining.'))
    print(pd.DataFrame(profiles).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
