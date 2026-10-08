"""Download only missing same-strike calls for the frozen selected P25 trades.

Run in mfe_wrds from the repo root:
    python scripts/download_p25_call_transfer.py --plan-only
    python scripts/download_p25_call_transfer.py

No training, full option chains, or stock/rate/dividend downloads. Credentials
are entered locally and never written to the cache. Successful query batches
are checkpointed so the command can be rerun after an interrupted download.
"""
import argparse
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/p25_validation_split_experiment.zip'
MEMBER = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
MARKET = ROOT / 'data/submission/artifacts/contracts/p25_signal_transfer/market'
DEFAULT_OUT = ROOT / 'data/p25_call_transfer'
KEYS = ['date', 'secid', 'strike', 'expiry']
RAW_COLUMNS = ['secid', 'date', 'optionid', 'exdate', 'cp_flag', 'strike_price',
               'best_bid', 'best_offer', 'open_interest', 'impl_volatility',
               'delta', 'vega', 'ss_flag', 'cfadj', 'contract_size']
MAX_REFERENCES = 414
MAX_QUOTE_KEYS = 25000
NORMAL_COLUMNS = ['date', 'secid', 'optionid', 'expiry', 'cp_flag', 'strike',
                  'bid', 'ask', 'multiplier', 'cfadj', 'standard', 'open_interest',
                  'vendor_iv', 'vendor_delta', 'vendor_vega']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def write_parquet(path, frame):
    temp = path.with_suffix('.partial')
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def verified_archive():
    expected = json.loads((ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    result = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in ['scored_allocations.parquet', 'trades.csv', 'daily_nav.parquet']:
            content = z.read(MEMBER + name)
            if hashlib.sha256(content).hexdigest() != expected[MEMBER + name]:
                raise RuntimeError(f'Frozen reference checksum mismatch: {name}')
            result[name] = (pd.read_parquet(io.BytesIO(content)) if name.endswith('.parquet')
                            else pd.read_csv(io.BytesIO(content), parse_dates=['date']))
    return result


def prepare(out):
    frozen = verified_archive()
    fills = frozen['trades.csv']
    entries = fills.loc[fills.reason.eq('entry')].copy()
    if len(entries) != MAX_REFERENCES or entries.cohort.nunique() != 207:
        raise RuntimeError('This downloader is pinned to the 207-pair selected checkpoint')
    if not fills.reason.isin(['entry', 'expiry_settlement']).all():
        raise RuntimeError('An exit convention changed; review download scope before proceeding')
    allocations = frozen['scored_allocations.parquet']
    allocations['cohort'] = [f'{r.date.date()}|{r.pair}|{r.wing}' for r in allocations.itertuples()]
    selected = allocations.loc[allocations.cohort.isin(entries.cohort)].copy()
    write_parquet(out / 'frozen_allocations.parquet', selected)
    refs = []
    for r in entries.itertuples():
        signal = pd.Timestamp(r.cohort.split('|')[0])
        end = fills.loc[fills.cohort.eq(r.cohort) & fills.optionid.eq(r.optionid)
                        & fills.reason.eq('expiry_settlement'), 'date']
        if len(end) != 1:
            raise RuntimeError('Each frozen leg must have one settlement')
        refs.append(dict(cohort=r.cohort, date=signal, entry_date=r.date,
                         end_date=end.iloc[0], secid=int(r.secid),
                         put_optionid=int(r.optionid), quantity=int(r.quantity)))
    refs = pd.DataFrame(refs)
    calendar = frozen['daily_nav.parquet'].index
    sources = json.loads((ROOT / 'data/submission/manifest.json').read_text())['files']
    expected = {r['stored']: r['sha256'] for r in sources}
    hashes = {}
    for year in sorted(set(refs.date.dt.year) | set(refs.end_date.dt.year)):
        path = MARKET / f'{year}_quotes.parquet'
        digest = sha256(path)
        relative = path.relative_to(ROOT).as_posix()
        if digest != expected[relative]:
            raise RuntimeError(f'Bundled market checksum mismatch: {relative}')
        hashes[relative] = digest
        q = pd.read_parquet(path)
        p = q.loc[q.cp_flag.eq('P'), ['date', 'optionid', 'expiry', 'strike']]
        mask = refs.date.dt.year.eq(year)
        join = refs.loc[mask, ['date', 'put_optionid']].merge(
            p, left_on=['date', 'put_optionid'], right_on=['date', 'optionid'],
            how='left', validate='many_to_one')
        if join[['expiry', 'strike']].isna().any().any():
            raise RuntimeError('A frozen put signal quote is missing')
        refs.loc[mask, 'expiry'] = join.expiry.to_numpy()
        refs.loc[mask, 'strike'] = join.strike.to_numpy()
    refs['expiry'] = pd.to_datetime(refs.expiry)
    if len(refs) > MAX_REFERENCES or (refs.expiry - refs.date).dt.days.max() > 56:
        raise RuntimeError('Reference count or maturity exceeds the frozen scope')
    # Revisit the adjacent year: a December signal can settle in January.
    all_contracts = refs[['secid', 'strike', 'expiry']].drop_duplicates()
    local_parts = []
    for year in sorted(hashes):
        q = pd.read_parquet(ROOT / year)
        q = q.loc[q.cp_flag.eq('C') & q.standard & q.multiplier.eq(100)]
        local_parts.append(q.merge(all_contracts, on=['secid', 'strike', 'expiry'], how='inner'))
    local = pd.concat(local_parts, ignore_index=True)
    choice = local.sort_values(KEYS + ['open_interest', 'optionid'],
                              ascending=[True] * 4 + [False, True]).drop_duplicates(KEYS)
    mapping = refs.merge(choice[KEYS + ['optionid']], on=KEYS, how='left', validate='many_to_one')
    wanted = quote_keys(mapping.dropna(subset=['optionid']), calendar)
    present = local[['date', 'optionid']].drop_duplicates()
    missing = anti_join(wanted, present, ['date', 'optionid'])
    total_keys = sum(((calendar >= r.date) & (calendar < r.end_date)).sum() for r in refs.itertuples())
    plan = dict(checkpoint='screened_selected; frozen 27-input P25 policy',
                pairs=207, option_legs=len(refs), unique_put_contracts=int(refs.put_optionid.nunique()),
                exact_counterpart_lookup_keys=len(refs[KEYS].drop_duplicates()),
                legs_with_local_counterpart=int(mapping.optionid.notna().sum()),
                missing_counterpart_lookup_keys=len(mapping.loc[mapping.optionid.isna(), KEYS].drop_duplicates()),
                cached_counterpart_quote_keys=len(wanted) - len(missing),
                known_counterpart_quote_keys_to_download=len(missing),
                maximum_leg_date_keys=int(total_keys),
                earliest_signal=str(refs.date.min().date()), last_settlement=str(refs.end_date.max().date()),
                date_scope='Signal close through last session before each original leg settles; no quotes after settlement',
                selection='Same underlying, strike, expiry; standard 100-share calls; highest signal-date OI then smallest optionid',
                reuse=['put quotes', 'stock prices', 'funding rates', 'declared dividends'],
                source_hashes=hashes,
                excluded='Rejected P25 proposals, training data, full chains, other strikes, other tenors, unrelated assets',
                credentials='Prompted locally; never stored by this script')
    if total_keys > MAX_QUOTE_KEYS:
        raise RuntimeError('Required quote keys exceed the download safety bound')
    write_parquet(out / 'put_references.parquet', refs)
    write_json(out / 'download_plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k not in ['source_hashes']}, indent=2), flush=True)
    return refs, mapping, local, calendar, plan


def anti_join(left, right, keys):
    merged = left.merge(right[keys].drop_duplicates().assign(_present=True), on=keys, how='left')
    return merged.loc[merged._present.isna()].drop(columns='_present')


def quote_keys(mapping, calendar):
    parts = []
    for r in mapping.itertuples():
        dates = calendar[(calendar >= r.date) & (calendar < r.end_date)]
        parts.append(pd.DataFrame(dict(date=dates, optionid=int(r.optionid))))
    result = (pd.concat(parts, ignore_index=True).drop_duplicates() if parts
              else pd.DataFrame(columns=['date', 'optionid']))
    if len(result) > MAX_QUOTE_KEYS:
        raise RuntimeError('Quote request is larger than the authorized checkpoint scope')
    return result


def normalize(raw):
    q = raw.copy().rename(columns=dict(exdate='expiry', strike_price='strike',
                                      best_bid='bid', best_offer='ask', contract_size='multiplier',
                                      impl_volatility='vendor_iv', delta='vendor_delta', vega='vendor_vega'))
    q['strike'] = q.strike.astype(float) / 1000
    q['vendor_vega'] = q.vendor_vega.astype(float) / 100
    q['standard'] = q.ss_flag.astype(str).eq('0')
    for name in ['date', 'expiry']:
        q[name] = pd.to_datetime(q[name])
    return q[NORMAL_COLUMNS]


class Downloader:
    def __init__(self, out):
        self.out = out
        self.db = None

    def query(self, label, query, params, maximum):
        """One connection, bounded results, hash-checked reusable query cache."""
        folder = self.out / 'query_cache'
        folder.mkdir(exist_ok=True)
        fingerprint = hashlib.sha256((query + repr(params)).encode()).hexdigest()
        path = folder / f'{label}_{fingerprint[:16]}.parquet'
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if saved['query_sha256'] != fingerprint or sha256(path) != saved['file_sha256']:
                raise RuntimeError(f'Query checkpoint checksum mismatch: {label}')
            return pd.read_parquet(path)
        if self.db is None:
            import wrds
            print('\nEnter your WRDS login locally; approve the Duo push on your phone.', flush=True)
            username = input('WRDS username: ').strip()
            password = getpass.getpass('WRDS password: ')
            self.db = wrds.Connection(wrds_username=username, wrds_password=password,
                                      wrds_connect_args={'sslmode': 'require', 'connect_timeout': 120})
            del password
            self.db.connection.exec_driver_sql('SET default_transaction_read_only = on')
            self.db.connection.exec_driver_sql("SET statement_timeout = '120s'")
            self.db.connection.commit()
        print(f'Fetching {label}: at most {maximum:,} rows', flush=True)
        q = self.db.raw_sql(query, params=params, date_cols=['date', 'exdate'], chunksize=None)
        if len(q) > maximum:
            raise RuntimeError('Unexpected duplicate/excess records; stopped before saving')
        write_parquet(path, q)
        write_json(meta, dict(query_sha256=fingerprint, file_sha256=sha256(path), rows=len(q)))
        return q

    def close(self):
        if self.db is not None:
            self.db.close()


def download(out, refs, mapping, local, calendar, plan):
    downloader = Downloader(out)
    try:
        fields = ', '.join('q.' + c for c in RAW_COLUMNS)
        missing = mapping.loc[mapping.optionid.isna(), KEYS].drop_duplicates()
        found = []
        for year, g in missing.groupby(missing.date.dt.year):
            params = dict(dates=g.date.dt.date.tolist(), secids=g.secid.astype(int).tolist(),
                          strikes=np.rint(g.strike * 1000).astype(int).tolist(), expiries=g.expiry.dt.date.tolist())
            query = f'''SELECT DISTINCT ON (q.date,q.secid,q.strike_price,q.exdate) {fields}
                FROM optionm.opprcd{int(year)} q
                JOIN unnest(%(dates)s::date[], %(secids)s::bigint[],
                            %(strikes)s::bigint[], %(expiries)s::date[])
                     wanted(date,secid,strike_price,exdate)
                  ON q.date=wanted.date AND q.secid=wanted.secid
                 AND q.strike_price=wanted.strike_price AND q.exdate=wanted.exdate
                WHERE q.cp_flag='C' AND q.ss_flag='0' AND q.contract_size=100
                ORDER BY q.date,q.secid,q.strike_price,q.exdate,q.open_interest DESC NULLS LAST,q.optionid
                LIMIT {len(g) + 1}'''
            raw = downloader.query(f'lookup_{int(year)}', query, params, len(g))
            if len(raw):
                found.append(normalize(raw))
        if found:
            local = pd.concat([local, *found], ignore_index=True).drop_duplicates(['date', 'optionid'])
            choice = pd.concat(found, ignore_index=True)[KEYS + ['optionid']]
            mapping = refs.merge(pd.concat([mapping.dropna(subset=['optionid'])[KEYS + ['optionid']], choice]).drop_duplicates(KEYS),
                                 on=KEYS, how='left', validate='many_to_one')
        wanted = quote_keys(mapping.dropna(subset=['optionid']), calendar)
        remaining = anti_join(wanted, local[['date', 'optionid']], ['date', 'optionid'])
        for year, group in remaining.groupby(remaining.date.dt.year):
            for i in range(0, len(group), 1500):
                batch = group.iloc[i:i + 1500]
                params = dict(dates=batch.date.dt.date.tolist(), ids=batch.optionid.astype(int).tolist())
                query = f'''SELECT {fields} FROM optionm.opprcd{int(year)} q
                    JOIN unnest(%(dates)s::date[], %(ids)s::bigint[]) wanted(date,optionid)
                      ON q.date=wanted.date AND q.optionid=wanted.optionid
                    WHERE q.cp_flag='C' AND q.ss_flag='0' AND q.contract_size=100
                    LIMIT {len(batch) + 1}'''
                raw = downloader.query(f'history_{int(year)}_{i // 1500:03d}', query, params, len(batch))
                if len(raw):
                    local = pd.concat([local, normalize(raw)], ignore_index=True)
        quotes = local.merge(wanted, on=['date', 'optionid'], how='inner', validate='many_to_one')
        if quotes.duplicated(['date', 'optionid']).any():
            raise RuntimeError('Duplicate daily counterpart quote keys')
        # Verify downloaded contract identity on every date, including adjustments.
        identities = mapping.dropna(subset=['optionid'])[['optionid', 'secid', 'strike', 'expiry']].drop_duplicates()
        checked = quotes.merge(identities, on=['optionid', 'secid', 'strike', 'expiry'], how='left', indicator=True)
        if not checked._merge.eq('both').all():
            raise RuntimeError('A counterpart changed strike, expiry or underlying; review adjustment')
        missing_daily = anti_join(wanted, quotes[['date', 'optionid']], ['date', 'optionid'])
        write_parquet(out / 'counterpart_mapping.parquet', mapping)
        write_parquet(out / 'call_quotes.parquet', quotes.sort_values(['date', 'optionid']))
        write_parquet(out / 'missing_call_quotes.parquet', missing_daily)
        write_parquet(out / 'unmatched_put_references.parquet', mapping.loc[mapping.optionid.isna()])
        status = dict(status='download_finished', pairs=207, option_legs=len(refs),
                      matched_legs=int(mapping.optionid.notna().sum()),
                      quote_rows=len(quotes), missing_daily_quotes=len(missing_daily),
                      unmatched_legs=int(mapping.optionid.isna().sum()),
                      source_plan_sha256=sha256(out / 'download_plan.json'),
                      files={p.name: sha256(p) for p in out.glob('*.parquet')},
                      note='Missing quotes/references are reported explicitly; no strategy has been fitted or evaluated.')
        write_json(out / 'download_status.json', status)
        print(f'\nDownload finished. Saved {len(quotes):,} daily call quotes to {out}', flush=True)
        print(f'Unmatched legs: {status["unmatched_legs"]}; missing daily quotes: {len(missing_daily)}', flush=True)
    finally:
        downloader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true', help='Check local coverage without logging in or querying WRDS')
    args = parser.parse_args()
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    refs, mapping, local, calendar, plan = prepare(out)
    if not args.plan_only:
        download(out, refs, mapping, local, calendar, plan)


if __name__ == '__main__':
    main()
