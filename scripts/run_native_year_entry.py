"""Independent monthly one-year entry research, with bounded streamed data."""
import argparse
import gc
import getpass
import json
import os
from pathlib import Path
import resource
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import joblib
from sklearn.tree import DecisionTreeRegressor

from scripts import run_p25_long_expiries as old
from scripts import native_year_features as features
from scripts import native_year_policy as policy
from scripts.run_tail_put25_pooled import prior_vix
from scripts.audit_long_expiry_parity import BoundMarket
from scripts import run_long_tenor_policy_search as replay_base

BASE = Path('artifacts/contracts/native_year_entry')
OUT = BASE / 'return_reconciled_v2'
REPORT = Path('reports/native_year_entry')
DATA = Path('data/native_year_options')
NATIVE_CORPORATE_EVENTS = [
    dict(secid=102963, announced='2020-04-13', effective='2020-04-15',
         reason='Chesapeake announced 1-for-200 reverse split',
         source='https://www.sec.gov/Archives/edgar/data/895126/000089512620000096/ex9912020-04x13prreres.htm'),
    dict(secid=114726, announced='2020-04-01', effective='2020-04-15',
         reason='VanEck OIH announced 1-for-20 reverse split',
         source='https://www.vaneck.com/globalassets/home/us/insights/media-coverage/press-releases/vaneck-announces-reverse-share-split-of-five-etfs.pdf'),
    dict(secid=127107, announced='2020-03-24', effective='2020-03-30',
         reason='State Street XOP announced 1-for-4 reverse split (release March23)',
         source='https://www.nasdaq.com/press-release/state-street-global-advisors-announces-reverse-splits-of-two-etfs-2020-03-23'),
    dict(secid=104081, announced='2020-09-30', effective='2020-10-02',
         reason='ETrade merger with Morgan Stanley, expected closing announced',
         source='https://www.morganstanley.com/press-releases/morgan-stanley-receives-regulatory-approval-to-acquire-e-trade'),
    dict(secid=101477, announced='2020-09-30', effective='2020-10-06',
         reason='TD Ameritrade merger with Schwab, expected closing announced',
         source='https://www.sec.gov/Archives/edgar/data/316709/000095010320019329/dp137329_8k.htm'),
    dict(secid=105316, announced='2019-04-12', effective='2019-04-18',
         reason='Goldcorp merger with Newmont, expected closing announced',
         source='https://www.sec.gov/Archives/edgar/data/1164727/000110465919021342/a19-7870_5ex99d1.htm'),
    dict(secid=151404, announced='2020-12-30', effective='2021-01-07',
         reason='WPX merger with Devon, expected closing announced',
         source='https://investors.devonenergy.com/investors/press-releases/press-release-details/2020/Devon-Energy-and-WPX-Energy-Shareholders-Approve-Merger-of-Equals/default.aspx'),
    dict(secid=125179, announced='2024-08-12', effective='2024-08-16',
         reason='SunPower announced Nasdaq delisting after bankruptcy',
         source='https://www.sec.gov/Archives/edgar/data/867773/000086777324000093/ex991_2024812pressrelease.htm'),
    dict(secid=205711, announced='2021-12-28', effective='2021-12-29',
         reason='ARKK announced capital-gains distribution adjusts option strikes',
         source='https://www.miaxglobal.com/sites/default/files/alert-files/ARKK_Capital_Gains_49856.pdf'),
    dict(secid=104939, announced='2023-02-03', effective='2023-02-10',
         reason='Ford supplemental dividend adjusts option strikes',
         source='https://infomemo.theocc.com/infomemos?number=51906'),
    dict(secid=102833, announced='2023-12-06', effective='2023-12-14',
         reason='Avis December5 after-close special-dividend announcement',
         source='https://ir.avisbudgetgroup.com/news-releases/news-release-details/avis-budget-group-announces-special-cash-dividend'),
    dict(secid=106790, announced='2023-06-12', effective='2023-07-03',
         reason='Labcorp Fortrea spinoff changes option deliverable',
         source='https://ir.labcorp.com/node/34911'),
]
CORPORATE_CHECKPOINT = OUT / f'corporate_labels_{len(NATIVE_CORPORATE_EVENTS)}.json'


def configure_native():
    replay_base.configure()
    for event in NATIVE_CORPORATE_EVENTS:
        if not any(e.get('reason') == event['reason'] for e in old.long.base.UNIFORM_CORPORATE_EVENTS):
            old.long.base.UNIFORM_CORPORATE_EVENTS = [*old.long.base.UNIFORM_CORPORATE_EVENTS, event]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def roster():
    rows = pd.read_csv('reports/uniform_smile/coverage.csv')
    rows = rows[['pair', 'label']].drop_duplicates('pair')
    ids = rows.pair.str.split('__', expand=True).astype(int)
    rows['secid_a'], rows['secid_b'] = ids[0], ids[1]
    # Canonical pair keys and old labels need not share the same orientation.
    # Preserve canonical regression orientation, display actual A/B names.
    source = pd.read_csv('data/current_rl_pairs.csv', usecols=['label', 'secid_a', 'secid_b'])
    lookup = {}
    for row in source.itertuples():
        a, b = row.label.split('/')
        lookup[int(row.secid_a)], lookup[int(row.secid_b)] = a, b
    rows['label'] = rows.secid_a.map(lookup)+'/'+rows.secid_b.map(lookup)
    assert rows.label.notna().all()
    return rows.sort_values('pair').reset_index(drop=True)


def calendar():
    # Observed historical trading calendar; entry dates have no signal dependence.
    zero = pd.read_parquet(old.ROOT / 'zero_curve.parquet', columns=['date'])
    return pd.DatetimeIndex(sorted(pd.to_datetime(zero.date).unique()))


def signal_dates():
    cal = calendar()
    days = cal[(cal >= '2013-01-01') & (cal <= '2024-06-30')]
    return days.to_series().groupby(days.to_period('M')).first().to_numpy()


def relevant_dividends(quotes, dividends):
    """Bounded prefilter for events that can satisfy fit_quotes' exact mask."""
    return dividends.loc[dividends.secid.isin(quotes.secid.unique()) &
                         dividends.ex_date.gt(quotes.date.min()) &
                         dividends.ex_date.le(quotes.expiry.max()) &
                         dividends.declare_date.le(quotes.date.max())]


def probe():
    rows = roster()
    print(f'Native universe: {len(rows)} existing pairs, {len(set(rows.secid_a)|set(rows.secid_b))} assets; {len(signal_dates())} independent monthly dates', flush=True)
    for path in [old.ROOT / 'underlying/year=2010.parquet', old.ROOT / 'underlying/year=2025.parquet',
                 old.long.base.TRANSFER / 'market/rates.parquet']:
        print(path, pq.ParquetFile(path).schema.names, flush=True)
    db = old.connection(os.getenv('WRDS_USERNAME', getpass.getuser()))
    with db.cursor() as cursor:
        cursor.execute('select current_date')
        print(f'WRDS read-only connection available: {cursor.fetchone()[0]}', flush=True)
    db.commit()


def inspect_identifiers():
    db = old.connection(os.getenv('WRDS_USERNAME', getpass.getuser()))
    with db.cursor() as cursor:
        cursor.execute("select column_name from information_schema.columns where table_schema='optionm' and table_name='opprcd2023'")
        cols = [row[0] for row in cursor.fetchall()]
        fields = ['date', 'secid', 'optionid', 'exdate', 'strike_price', 'best_bid', 'best_offer', 'ss_flag', 'cfadj']
        fields += [c for c in ['symbol', 'symbol_flag'] if c in cols]
        cursor.execute('select '+', '.join(fields)+" from optionm.opprcd2023 where secid=102833 and date in ('2023-06-30','2023-07-03') and cp_flag='P' and exdate='2024-01-19' and strike_price between 120000 and 190000 order by date,strike_price")
        print(pd.DataFrame(cursor.fetchall(), columns=fields).to_string(index=False), flush=True)
    db.commit()


def inspect_invalid_quotes():
    q = pd.read_parquet(OUT / 'market/2023_quotes.parquet')
    print(q.loc[q.optionid.eq(146431123) & q.date.between('2023-06-23', '2023-07-07')].to_string(index=False), flush=True)
    print(pd.read_parquet(OUT / 'selected.parquet').loc[lambda x: x.optionid.eq(146431123)].to_string(index=False), flush=True)
    states = pd.read_parquet(OUT / 'signal_rows.parquet')
    wanted = pd.concat([states.loc[states.signal_eligible, ['date', 'pair', f'secid_{s}', f'optionid_{s}', 'known_contract_expiry']]
                       .rename(columns={f'secid_{s}': 'secid', f'optionid_{s}': 'optionid'}) for s in ['a', 'b']], ignore_index=True)
    for year in range(2013, 2026):
        q = pd.read_parquet(OUT / 'market' / f'{year}_quotes.parquet')
        invalid = q.loc[~q.standard, ['date', 'secid', 'optionid', 'bs_status']].sort_values('date').drop_duplicates('optionid')
        affected = invalid.merge(wanted, on=['secid', 'optionid'], suffixes=('_quote', '_signal'))
        affected = affected.loc[affected.date_signal.lt(affected.date_quote) & affected.date_quote.le(affected.known_contract_expiry)]
        if len(affected):
            print('Nonstandard held deliverables:', affected.groupby(['secid', 'date_quote', 'bs_status']).agg(cohorts=('pair', 'size')).to_string(), flush=True)


def repair_identifiers():
    checkpoint = OUT / 'identifier_repair.json'
    if checkpoint.exists():
        target = OUT / 'market/2023_quotes.parquet'
        metadata = target.with_suffix('.json')
        note = json.loads(metadata.read_text())
        if note['sha256'] != old.file_hash(target):
            note.update(sha256=old.file_hash(target), original_quote_sha256=note['sha256'],
                        contract_identity_repair=str(checkpoint), rows=pq.ParquetFile(target).metadata.num_rows)
            temp = metadata.with_suffix('.updated.json')
            dump(temp, note)
            os.replace(temp, metadata)
        return
    boundary = pd.Timestamp('2023-07-03')
    sel = pd.read_parquet(OUT / 'selected.parquet')
    selected = sel.loc[sel.secid.eq(102833) & sel.date.lt(boundary)]
    raw = pd.read_parquet(OUT / 'market/2023_quotes.parquet')
    changed = raw.loc[raw.secid.eq(102833) & raw.date.eq(boundary)]
    prior = raw.loc[raw.secid.eq(102833) & raw.date.eq('2023-06-30')]
    wrong = changed.merge(prior[['optionid', 'strike']], on='optionid', suffixes=('', '_prior'))
    ids = wrong.loc[wrong.strike.ne(wrong.strike_prior), 'optionid'].astype(int).tolist()
    selected = selected.loc[selected.optionid.isin(ids)].sort_values('date').drop_duplicates('optionid')
    assert not sel.loc[sel.date.ge(boundary), 'optionid'].isin(selected.optionid).any(), 'Logical IDs also used by newer entry contracts'
    path = DATA / 'identity_recovery' / 'CAR_2023.parquet'
    if not path.exists():
        db = old.connection(os.getenv('WRDS_USERNAME', getpass.getuser()))
        with db.cursor() as cursor:
            fields = ', '.join('q.'+c for c in old.QUOTE_SCHEMA.names)
            cursor.execute('select '+fields+", q.symbol, wanted.logical_optionid from optionm.opprcd2023 q join unnest(%s::bigint[], %s::date[], %s::double precision[]) wanted(logical_optionid,expiry,strike) on q.exdate=wanted.expiry and q.strike_price=wanted.strike where q.secid=102833 and q.cp_flag='P' and q.date>='2023-07-03' and q.date<'2023-12-14' order by wanted.logical_optionid,q.date",
                (selected.optionid.astype(int).tolist(), selected.expiry.dt.date.tolist(), (selected.strike*1000).tolist()))
            actual = pd.DataFrame(cursor.fetchall(), columns=old.QUOTE_SCHEMA.names+['symbol', 'logical_optionid'])
        db.commit()
        path.parent.mkdir(parents=True, exist_ok=True)
        actual.to_parquet(path, index=False)
    actual = pd.read_parquet(path)
    assert not actual.duplicated(['date', 'logical_optionid']).any()
    repaired = old.normalize_quotes(actual)
    repaired['vendor_optionid'] = repaired.optionid
    repaired['optionid'] = actual.logical_optionid.to_numpy(int)
    stock = pd.read_parquet(OUT / 'market/stocks.parquet')
    zero = pd.read_parquet(old.ROOT / 'zero_curve.parquet')
    zero.date = pd.to_datetime(zero.date)
    divs = old.repaired_distributions()
    repaired = old.fit_quotes(repaired, stock, zero, relevant_dividends(repaired, divs))
    target = OUT / 'market/2023_quotes.parquet'
    backup = target.with_name('2023_quotes_before_identifier_repair.parquet')
    if not backup.exists():
        os.link(target, backup)
    # Only these old, no-longer-selectable logical IDs have changed meaning.
    keep = raw.loc[~(raw.date.ge(boundary) & raw.optionid.isin(selected.optionid))]
    merged = pd.concat([keep, repaired], ignore_index=True).sort_values(['optionid', 'date'])
    temp = target.with_suffix('.repaired.parquet')
    merged.to_parquet(temp, index=False)
    os.replace(temp, target)
    asset = OUT / 'assets/102833/2023.parquet'
    asset_backup = asset.with_name('2023_before_identifier_repair.parquet')
    if not asset_backup.exists():
        os.link(asset, asset_backup)
    temp = asset.with_suffix('.repaired.parquet')
    merged.loc[merged.secid.eq(102833)].to_parquet(temp, index=False)
    os.replace(temp, asset)
    pd.DataFrame(actual[['date', 'logical_optionid', 'optionid', 'symbol']]).to_csv(REPORT / 'contract_identity_recovery.csv', index=False)
    dump(checkpoint, dict(asset='CAR', first_observed_reassignment=str(boundary.date()),
        logical_contracts=len(selected), recovered_observed_quotes=len(repaired), source=str(path),
        identity='Same issuer, put, expiry and original strike, using contemporaneous chain rows',
        reused_ID_never_selectable_for_new_native_entries_after_reassignment=True,
        original_quote_cache_preserved=True, raw_source_sha256=old.file_hash(path)))
    repair_identifiers()  # Refresh metadata atomically after the new cache exists.


def inspect_market():
    qcar = pd.read_parquet(OUT / 'market/2023_quotes.parquet', filters=[('optionid', 'in', [142372412, 142758390, 142758393])])
    print('CAR contract changes:', qcar.loc[qcar.date.between('2023-06-27', '2023-07-07')].to_string(index=False), flush=True)
    adjusted = pd.read_parquet(OUT / 'market/2021_quotes.parquet', filters=[('optionid', '=', 139605652)])
    print('Held option adjustment diagnostic:', adjusted.loc[adjusted.date.between('2021-12-15', '2022-01-03')].to_string(index=False), flush=True)
    stock = pd.read_parquet(OUT / 'market/stocks.parquet')
    print(stock.loc[stock.secid.eq(106899) & stock.date.between('2019-05-24', '2019-06-05')].to_string(index=False), flush=True)
    q = pd.read_parquet(OUT / 'market/2019_quotes.parquet', filters=[('secid', '=', 106899)])
    print(q.loc[q.date.between('2019-05-29', '2019-06-03'), ['date', 'optionid', 'strike', 'cfadj', 'standard', 'bid', 'ask']].to_string(index=False), flush=True)
    d = old.repaired_distributions()
    print(d.loc[d.secid.eq(106899) & d.ex_date.between('2019-01-01', '2019-12-31')].to_string(index=False), flush=True)
    raw = pd.read_parquet(old.ROOT / 'underlying/year=2019.parquet')
    print(raw.loc[raw.secid.eq(106899) & pd.to_datetime(raw.date).between('2019-05-24', '2019-06-05')].to_string(index=False), flush=True)
    raw = pd.read_parquet(old.ROOT / 'underlying/year=2022.parquet')
    print(raw.loc[raw.secid.eq(111450) & pd.to_datetime(raw.date).between('2022-07-20', '2022-08-03')].to_string(index=False), flush=True)
    q = pd.read_parquet(OUT / 'market/2022_quotes.parquet', filters=[('secid', '=', 111450)])
    print(q.loc[q.date.between('2022-07-25', '2022-07-29'), ['date', 'optionid', 'strike', 'cfadj', 'standard', 'bid', 'ask']].to_string(index=False), flush=True)
    ratios = stock.cfadj/stock.groupby('secid').cfadj.shift(1)
    changed = stock.date.ge('2018-01-01') & ratios.notna() & ~np.isclose(ratios, 1.)
    unsupported = changed & ((ratios < 1) | ~np.isclose(ratios, np.round(ratios)))
    names = roster()
    lookup = {}
    for row in names.itertuples():
        a, b = row.label.split('/')
        lookup[row.secid_a], lookup[row.secid_b] = a, b
    events = stock.loc[unsupported, ['date', 'secid', 'close', 'cfadj']].assign(
        ratio=ratios.loc[unsupported], ticker=stock.loc[unsupported, 'secid'].map(lookup))
    proposals = pd.read_parquet(OUT / 'signal_rows.parquet')
    counts = []
    for event in events.itertuples():
        g = proposals.loc[proposals.signal_eligible & proposals.date.lt(event.date) &
            proposals.known_contract_expiry.ge(event.date) &
            (proposals.secid_a.eq(event.secid) | proposals.secid_b.eq(event.secid))]
        counts.append(len(g))
    events['eligible_cohorts_spanning_event'] = counts
    events.to_csv(REPORT / 'nonstandard_stock_events.csv', index=False)
    print(events.to_string(index=False), flush=True)
    delists, cal_end = [], stock.date.max()
    for sid, g in stock.groupby('secid'):
        last = g.date.max()
        selected = proposals.loc[proposals.signal_eligible & proposals.date.le(last) &
            proposals.known_contract_expiry.gt(last) &
            (proposals.secid_a.eq(sid) | proposals.secid_b.eq(sid))]
        if last < cal_end and len(selected):
            delists.append(dict(secid=int(sid), ticker=lookup.get(sid), last_stock_date=str(last.date()), eligible_cohorts=len(selected)))
    print('Native held-underlying coverage endings:', delists, flush=True)
    meta = pd.read_parquet(OUT / 'training_rows.parquet')
    unresolved = meta.loc[meta.signal_eligible & meta.label_status.eq('unresolved')]
    wanted = pd.concat([unresolved[['date', f'optionid_{side}']].rename(columns={f'optionid_{side}': 'optionid'}) for side in ['a', 'b']])
    for file in sorted((OUT / 'market').glob('*_quotes.parquet')):
        quotes = pd.read_parquet(file, filters=[('optionid', 'in', wanted.optionid.dropna().astype(int).tolist())],
            columns=['date', 'secid', 'optionid', 'strike', 'cfadj', 'standard'])
        for oid, g in quotes.groupby('optionid'):
            g = g.sort_values('date')
            changed = (g.strike.ne(g.strike.shift(1)) | g.cfadj.ne(g.cfadj.shift(1))) & g.strike.shift(1).notna()
            if changed.any():
                print('Unresolved label option changes:', g.loc[changed].to_string(index=False), flush=True)


def download_chains():
    rows = roster()
    ids = sorted(set(rows.secid_a) | set(rows.secid_b))
    dates = pd.DatetimeIndex(signal_dates())
    dump(OUT / 'design.json', dict(pairs=len(rows), assets=len(ids), frequency='First observed trading session of each month',
         first_signal=str(dates[0].date()), last_signal=str(dates[-1].date()), tenor=365,
         bounds=[300, 400], option_type='P', absolute_delta=.25, delta_tolerance=.075,
         pair_eligibility='Prior-close 126-session beta 0.85–1.15 and correlation >=0.5',
         no_short_policy_dates_directions_sizes_or_profit_gates=True, original_costs_and_risk_limits=True,
         training='2013–2017 with maturity purge; walk-forward evaluation2018–partial2025',
         limitations='Retrospective existing asset/pair roster; previously researched calendar, no fresh holdout'))
    rows.to_csv(REPORT / 'roster.csv', index=False)
    for year in dates.year.unique():
        request = dates[dates.year == year].date.tolist()
        fields = ', '.join(old.QUOTE_SCHEMA.names)
        query = f'select {fields} from optionm.opprcd{int(year)} where secid=ANY(%s) and date=ANY(%s::date[]) and cp_flag=\'P\' and exdate between date+300 and date+403'
        old.read_query(query, (ids, request), DATA / 'signal_chains' / f'{year}.parquet')
        dump(OUT / 'progress.json', dict(stage='download_chains', year=int(year), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def prepare():
    if (OUT / 'prepared.json').exists():
        return
    pairs = roster()
    ids = sorted(set(pairs.secid_a) | set(pairs.secid_b))
    cal = calendar()
    cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    market = OUT / 'market'
    market.mkdir(exist_ok=True)
    parts = []
    for year in range(2010, 2026):
        stock = pd.read_parquet(old.ROOT / 'underlying' / f'year={year}.parquet', columns=['date', 'secid', 'close', 'cfadj', 'return'])
        stock.date = pd.to_datetime(stock.date)
        parts.append(stock.loc[stock.secid.isin(ids) & stock.date.isin(cal)])
    stocks = pd.concat(parts, ignore_index=True)
    del parts
    stocks, repairs = features.reconcile_adjustments(stocks)
    pd.DataFrame(repairs).to_csv(REPORT / 'adjustment_repairs.csv', index=False)
    stocks.to_parquet(market / 'stocks.parquet', index=False)
    close = stocks.pivot(index='date', columns='secid', values='close').reindex(cal)
    adjustment = stocks.pivot(index='date', columns='secid', values='cfadj').reindex(cal)
    adjusted = close * adjustment
    zero = pd.read_parquet(old.ROOT / 'zero_curve.parquet')
    zero.date = pd.to_datetime(zero.date)
    rate = pd.Series({date: np.interp(30., g.sort_values('days').days, g.sort_values('days').rate) / 100
                      for date, g in zero.groupby('date')}).reindex(cal)
    reference_rates = pd.read_parquet(old.long.base.TRANSFER / 'market/rates.parquet').rate
    rate.loc[rate.index.intersection(reference_rates.index)] = reference_rates.reindex(rate.index.intersection(reference_rates.index))
    assert rate.notna().all()
    rate.index.name = 'date'
    rate.rename('rate').to_frame().to_parquet(market / 'rates.parquet')
    divs = old.repaired_distributions()
    # Replay dividends use the existing ordinary distributions convention.
    ordinary = (divs.cancel_flag.eq('0') & divs.distr_type.eq('1') & divs.currency.eq('USD') &
                divs.adj_factor.eq(1) & divs.approx_flag.eq('0') & divs.liquid_flag.eq('0') & divs.secid.isin(ids))
    divs.loc[ordinary].to_parquet(market / 'dividends.parquet', index=False)
    monthly = pd.DatetimeIndex(signal_dates())
    asset_rows, returns = [], {}
    for sid in ids:
        if sid not in adjusted:
            continue
        values, r = features.variance_inputs(adjusted[sid], cal)
        returns[sid] = r
        asset_rows.append(values.reindex(monthly).assign(date=monthly, secid=sid))
    panel = pd.concat(asset_rows, ignore_index=True)
    panel, forecasts = features.forecast_rows(panel)
    panel.to_parquet(OUT / 'variance_forecasts.parquet', index=False)
    dump(REPORT / 'forecast_fits.json', forecasts)
    known = panel.loc[panel.date.ge('2018-01-01') & panel.future_variance.gt(0)]
    diagnostics = []
    for name in ['rv60', 'rv252', 'forecast_variance']:
        use = known.loc[known[name].gt(0)]
        ratio = use.future_variance / use[name]
        diagnostics.append(dict(model=name, observations=len(use),
            qlike=float((ratio - np.log(ratio) - 1).mean()),
            log_mse=float(np.log(ratio).pow(2).mean()),
            mean_forecast=float(use[name].mean()), mean_realized=float(use.future_variance.mean())))
    pd.DataFrame(diagnostics).to_csv(REPORT / 'forecast_accuracy.csv', index=False)
    selected = []
    following = pd.Series(cal[1:].to_numpy(), index=cal[:-1])
    for year in ([] if (OUT / 'selected.parquet').exists() else monthly.year.unique()):
        best = pd.DataFrame()
        stock = stocks.loc[stocks.date.dt.year.eq(year), ['date', 'secid', 'close']]
        path = DATA / 'signal_chains' / f'{year}.parquet'
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, use_threads=False):
            q = old.normalize_quotes(batch.to_pandas())
            mid = (q.bid + q.ask) / 2
            q = q.loc[q.standard & q.multiplier.eq(100) & q.bid.gt(0) & q.ask.ge(q.bid) &
                      q.open_interest.ge(old.configuration(365).min_open_interest) &
                      ((q.ask-q.bid)/mid).le(old.configuration(365).max_relative_spread)]
            if q.empty:
                continue
            q = old.fit_quotes(q, stock, zero, relevant_dividends(q, divs))
            choice = old.select_long(q, following, 365, 'P')
            if len(choice):
                best = old.select_long(pd.concat([best, choice], ignore_index=True), following, 365, 'P')
        selected.append(best)
        print(f'Independent365D selection {year}: {len(best)} asset/date contracts', flush=True)
        gc.collect()
    choice = pd.read_parquet(OUT / 'selected.parquet') if (OUT / 'selected.parquet').exists() else pd.concat(selected, ignore_index=True)
    choice.to_parquet(OUT / 'selected.parquet', index=False)
    joined = features.name_premiums(panel.merge(choice[['date', 'secid', 'optionid', 'bs_iv', 'bs_vega', 'expiry', 'entry_dte']],
                                on=['date', 'secid'], how='left', validate='one_to_one'))
    joined.to_parquet(OUT / 'name_signals.parquet', index=False)
    rows = []
    vix = prior_vix(cal).reindex(monthly) / 100
    for pair in pairs.itertuples():
        a, b = [joined.loc[joined.secid.eq(sid)].set_index('date').reindex(monthly) for sid in [pair.secid_a, pair.secid_b]]
        if pair.secid_a not in returns or pair.secid_b not in returns:
            continue
        ra, rb = returns[pair.secid_a], returns[pair.secid_b]
        beta = (ra.rolling(126, min_periods=126).cov(rb)/rb.rolling(126, min_periods=126).var()).shift(1).reindex(monthly)
        corr = ra.rolling(126, min_periods=126).corr(rb).shift(1).reindex(monthly)
        row = pd.DataFrame(dict(date=monthly, pair=pair.pair, label=pair.label, secid_a=pair.secid_a, secid_b=pair.secid_b,
                               optionid_a=a.optionid, optionid_b=b.optionid, beta=beta, correlation=corr,
                               iv_mean=(a.bs_iv+b.bs_iv)/2, iv_gap_relative=(a.bs_iv-b.bs_iv)/((a.bs_iv+b.bs_iv)/2),
                               edge_forecast=a.carry_forecast-b.carry_forecast, edge_slow=a.carry_slow-b.carry_slow,
                               edge_z=a.carry_z-b.carry_z, z_available=(a.carry_z.notna()&b.carry_z.notna()).astype(float),
                               forecast_ratio_a=a.forecast_variance/a.bs_iv.pow(2), forecast_ratio_b=b.forecast_variance/b.bs_iv.pow(2),
                               momentum_gap=a.momentum-b.momentum, dte_gap=a.entry_dte-b.entry_dte,
                               commission_vol_points=.65/(100*a.bs_vega)+.65/(100*b.bs_vega), vix=vix,
                               wing='P25', base_level=4, direction=features.native_direction(a.carry_forecast-b.carry_forecast),
                               known_contract_expiry=pd.concat([a.expiry, b.expiry], axis=1).max(axis=1)))
        row['signal_eligible'] = (row[['optionid_a', 'optionid_b']].notna().all(axis=1) &
             row.beta.between(.85, 1.15) & row.correlation.ge(.5) & np.isfinite(row[['edge_forecast', 'edge_slow']]).all(axis=1))
        row['z_direction'] = features.native_direction(row.edge_z.where(row.z_available.eq(1), row.edge_forecast))
        row['slow_direction'] = features.native_direction(row.edge_slow)
        row = features.native_iv_history(row)
        row[features.ENTRY_FEATURES] = row[features.ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
        rows.append(row.reset_index(drop=True))
    frame = pd.concat(rows, ignore_index=True).sort_values(['date', 'pair']).reset_index(drop=True)
    assert len(frame) == len(pairs) * len(monthly)
    assert np.isfinite(frame[features.ENTRY_FEATURES]).all().all()
    frame['position'] = cal.get_indexer(frame.date)
    frame.to_parquet(OUT / 'signal_rows.parquet', index=False)
    frame.groupby(frame.date.dt.year).agg(proposals=('pair', 'size'), native_eligible=('signal_eligible', 'sum'),
                                        z_observed=('z_available', 'sum')).to_csv(REPORT / 'coverage_by_year.csv')
    dump(OUT / 'prepared.json', dict(proposals=len(frame), eligible=int(frame.signal_eligible.sum()),
         eval_proposals=int(frame.date.ge('2018-01-01').sum()), eval_eligible=int((frame.date.ge('2018-01-01')&frame.signal_eligible).sum()),
         selected_contracts=len(choice), independent_calendar=True, no_short_policy_inputs=True,
         forecast_features=features.FORECAST_FEATURES, entry_features=features.ENTRY_FEATURES,
         own_history='24 previous observed one-year monthly quote premiums, minimum12; missingness explicit; forecast edge fallback for fixed rule',
         prior_close_beta=True, no_future_price_or_path_candidate_filter=True,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print((OUT / 'prepared.json').read_text(), flush=True)


def histories():
    selection = pd.read_parquet(OUT / 'selected.parquet')
    starts = selection.groupby('optionid').date.min().sort_index()
    ids = starts.index.astype(int).tolist()
    zero, divs = pd.read_parquet(old.ROOT / 'zero_curve.parquet'), old.repaired_distributions()
    for year in range(2013, 2026):
        target = OUT / 'market' / f'{year}_quotes.parquet'
        note = target.with_suffix('.json')
        if note.exists():
            assert old.file_hash(target) == json.loads(note.read_text())['sha256']
            continue
        fields = ', '.join('q.' + name for name in old.QUOTE_SCHEMA.names)
        query = f'select {fields} from optionm.opprcd{year} q join unnest(%s::bigint[],%s::date[]) wanted(optionid,first_signal) on q.optionid=wanted.optionid and q.date>=wanted.first_signal where q.cp_flag=\'P\' and q.date between %s and %s'
        raw = DATA / 'entry_history' / f'{year}.parquet'
        completed_download = DATA / 'history' / f'{year}.parquet'
        if completed_download.exists() and completed_download.with_suffix('.json').exists():
            assert old.file_hash(completed_download) == json.loads(completed_download.with_suffix('.json').read_text())['sha256']
            raw = completed_download
        else:
            old.read_query(query, (ids, pd.to_datetime(starts).dt.date.tolist(), f'{year}-01-01', min(f'{year}-12-31', '2025-08-29')), raw)
        stock = pd.read_parquet(old.ROOT / 'underlying' / f'year={year}.parquet', columns=['date', 'secid', 'close'])
        stock.date = pd.to_datetime(stock.date)
        temp, writer, count = target.with_suffix('.partial'), None, 0
        try:
            for batch in pq.ParquetFile(raw).iter_batches(batch_size=4096, use_threads=False):
                raw_quotes = old.normalize_quotes(batch.to_pandas())
                q = old.fit_quotes(raw_quotes, stock, zero, relevant_dividends(raw_quotes, divs))
                q['delta'], q['vega'] = q.bs_delta, q.bs_vega
                table = pa.Table.from_pandas(q[old.HISTORY_COLUMNS], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression='zstd')
                writer.write_table(table)
                count += len(q)
            if writer is None:
                raise RuntimeError(f'No selected native contract history in {year}')
            writer.close()
            writer = None
            temp.replace(target)
            dump(note, dict(rows=count, sha256=old.file_hash(target), source_sha256=old.file_hash(raw)))
        finally:
            if writer is not None:
                writer.close()
        print(f'Native365D histories {year}: {count:,} fitted rows', flush=True)
        dump(OUT / 'progress.json', dict(stage='histories', year=year, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del stock
        gc.collect()


def labels():
    from scripts.curve_trading import unit_paths
    unit_paths.SLOTS = [(365, 'P', 25)]
    # Partition one year at a time so each asset read avoids repeatedly
    # decoding the whole universe's year files. Peak memory stays bounded.
    for year in range(2013, 2026):
        marker = OUT / 'assets' / f'{year}.json'
        if marker.exists():
            continue
        q = pd.read_parquet(OUT / 'market' / f'{year}_quotes.parquet')
        files = {}
        for sid, group in q.groupby('secid', sort=False):
            path = OUT / 'assets' / str(int(sid)) / f'{year}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            group.to_parquet(path, index=False)
            files[str(path)] = old.file_hash(path)
        dump(marker, dict(year=year, rows=len(q), files=files))
        del q
        gc.collect()
    market = BoundMarket(OUT)
    cal = market.rates.index
    dates = pd.DatetimeIndex(signal_dates())
    selection = pd.read_parquet(OUT / 'selected.parquet')
    selection['tenor'], selection['bucket'] = 365, '25'
    rows = pd.read_parquet(OUT / 'signal_rows.parquet')
    cfg = old.configuration(365, training=True)
    cache = OrderedDict()
    def load(sid):
        if sid not in cache:
            chosen = selection.loc[selection.secid.eq(sid)].copy()
            ids = chosen.optionid.unique().astype(int).tolist()
            paths = [OUT / 'assets' / str(sid) / f'{year}.parquet' for year in range(2013, 2026)]
            pieces = [pd.read_parquet(path, filters=[('optionid', 'in', ids)]) for path in paths if path.exists() and ids]
            q = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=old.HISTORY_COLUMNS)
            stock = pd.DataFrame(dict(date=cal, close=market.spots[sid].reindex(cal),
                                     cfadj=market.adjustments[sid].reindex(cal), rate=market.rates.to_numpy()))
            cache[sid] = unit_paths.UnitPaths(q, stock, chosen, cal, dates, market.divs.loc[market.divs.secid.eq(sid)], cfg)
            while len(cache) > 6:
                cache.popitem(last=False)
        cache.move_to_end(sid)
        return cache[sid]
    statuses = []
    folder = OUT / 'labels'
    folder.mkdir(exist_ok=True)
    for index, (pair, g) in enumerate(rows.groupby('pair', sort=True), 1):
        path = folder / f'{pair}.npy'
        meta_path = folder / f'{pair}.parquet'
        if meta_path.exists():
            continue
        # Compute labels for all monthly states. Pair eligibility is exclusively
        # a signal-close fact and never determined by these future outcomes.
        a, b = load(int(g.secid_a.iloc[0])), load(int(g.secid_b.iloc[0]))
        daily, ends, vegas, status = unit_paths.pair_labels(a, b, (365, 'P', 25, 25), cfg)
        assert dates.equals(pd.DatetimeIndex(g.date))
        reward = np.zeros((len(dates), 9, daily.shape[-1]), np.float32)
        reward[:, 1:] = daily
        np.save(path, reward)
        meta = g.copy()
        meta['end'] = cal[ends]
        meta['label_row'] = np.arange(len(meta))
        meta['label_status'] = status
        meta['finite_label'] = np.isfinite(reward).all(axis=(1, 2))
        # Future execution failures have zero reward; unresolved valuation paths
        # are explicit missing labels, never inference/candidate exclusions.
        meta['target'] = (reward[:, 8].sum(axis=1) - reward[:, 1].sum(axis=1)) / 2
        meta.to_parquet(meta_path, index=False)
        for name, count in pd.Series(status).value_counts().items():
            statuses.append(dict(pair=pair, status=name, count=int(count)))
        if index % 20 == 0:
            print(f'Native365D labels {index}/{rows.pair.nunique()} pairs', flush=True)
        dump(OUT / 'progress.json', dict(stage='labels', pair=index, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del daily, reward, meta
    cache.clear()
    meta = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob('*.parquet'))], ignore_index=True)
    meta.to_parquet(OUT / 'training_rows.parquet', index=False)
    meta.groupby(['signal_eligible', 'label_status']).size().rename('rows').to_csv(REPORT / 'label_coverage.csv')


def audit_labels():
    from scripts.option_flex.contract_engine import integer_size
    meta = pd.read_parquet(OUT / 'training_rows.parquet')
    known = meta.loc[meta.signal_eligible & meta.finite_label & meta.label_status.eq('complete')].sort_values(['date', 'pair'])
    samples = pd.concat([known.loc[known.date.lt('2018-01-01')].head(2), known.loc[known.date.ge('2018-01-01')].head(2)])
    configure_native()
    market = BoundMarket(OUT)
    market.rate = lambda date: float(market.rates.loc[date])
    cfg = old.configuration(365, training=True)
    cal = market.rates.index
    results = []
    for _, row in samples.iterrows():
        start = int(cal.get_loc(row.date))
        end = int(cal.get_loc(row.end))
        live = [market.quote(cal[start+1], int(row[f'optionid_{side}'])) for side in ['a', 'b']]
        counts = integer_size([q.vega*100 for q in live], cfg.entry_vega_per_leg, cfg.max_entry_vega_imbalance)
        assert counts is not None
        cohort = f'{row.date.date()}|{row.pair}|P25'
        labels = np.load(OUT / 'labels' / f'{row.pair}.npy', mmap_mode='r')
        for direction, code in [(1, 8), (-1, 1)]:
            one = pd.DataFrame([row]).assign(direction=direction, learned_level=4)
            # Older OptionMetrics expiries can be Saturday even though
            # intrinsic settlement is Friday. Supply the following session
            # for the engine's calendar-runoff check; compare through expiry.
            nav, trades, decisions, ledger = old.long.replay(one, market, cal[start:end+2], 365,
                                                           training=True, locked_counts={cohort: counts})
            assert decisions.status.eq('entered').all(), decisions.to_dict('records')
            actual = (nav.nav.diff()-nav.cash_benchmark.diff()).iloc[1:1+end-start].to_numpy()/cfg.entry_vega_per_leg
            expected = labels[int(row.label_row), code, :len(actual)]
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-5)
            results.append(dict(date=str(row.date.date()), pair=row.pair, direction=direction,
                sessions=len(actual), max_daily_normalized_error=float(np.max(np.abs(actual-expected))),
                independent_ledger_error=ledger['independent_ledger_error']))
    assert len(results) == 8
    dump(REPORT / 'label_ledger_audit.json', dict(sampled_counterfactuals=8, native_one_year_shadow_matches_actual_single_cohort=True,
        counts_locked_to_unit_label=True, training_costs=True, results=results))


def corporate_labels():
    """Retain outcomes of native cohorts crossing publicly announced splits."""
    from scripts.option_flex.contract_engine import integer_size
    checkpoint = CORPORATE_CHECKPOINT
    if checkpoint.exists():
        return
    configure_native()
    market = BoundMarket(OUT)
    market.rate = lambda date: float(market.rates.loc[date])
    cal = market.rates.index
    cfg = old.configuration(365, training=True)
    rows = pd.read_parquet(OUT / 'training_rows.parquet')
    take = np.zeros(len(rows), bool)
    for event in NATIVE_CORPORATE_EVENTS:
        take |= (rows.signal_eligible & rows.date.lt(event['effective']) &
                 rows.known_contract_expiry.ge(event['effective']) &
                 (rows.secid_a.eq(event['secid']) | rows.secid_b.eq(event['secid']))).to_numpy()
    backup = OUT / f'labels_pre_corporate_{len(NATIVE_CORPORATE_EVENTS)}'
    backup.mkdir(exist_ok=True)
    audits = []
    for pair, selected in rows.loc[take].groupby('pair'):
        path = OUT / 'labels' / f'{pair}.npy'
        meta_path = path.with_suffix('.parquet')
        for source in [path, meta_path]:
            target = backup / source.name
            if not target.exists():
                os.link(source, target)
        # Write replacements atomically; hardlinked originals remain immutable.
        reward = np.load(backup / path.name).copy()
        meta = pd.read_parquet(backup / meta_path.name)
        for row in selected.itertuples():
            start, original_end = cal.get_loc(row.date), cal.get_loc(row.end)
            terminal = max(original_end, cal.searchsorted(row.known_contract_expiry, side='right')-1)
            live = [market.quote(cal[start+1], int(getattr(row, f'optionid_{s}'))) for s in ['a', 'b']]
            reward[row.label_row] = 0.
            ends = []
            for code in range(1, 9):
                direction, level = policy.decode(code)
                counts = None if any(q is None for q in live) else integer_size(
                    [q.vega*100 for q in live], cfg.entry_vega_per_leg*abs(policy.FRACTIONS[code]), cfg.max_entry_vega_imbalance)
                if counts is None:
                    continue
                cohort = f'{row.date.date()}|{pair}|P25'
                proposal = pd.DataFrame([row._asdict()]).assign(direction=int(direction), learned_level=int(level))
                nav, trades, decisions, ledger = old.long.replay(proposal, market, cal[start:terminal+2], 365,
                    training=True, locked_counts={cohort: counts})
                if not decisions.status.eq('entered').any():
                    continue
                end = cal.get_loc(pd.to_datetime(trades.date).max())
                daily = (nav.nav.diff()-nav.cash_benchmark.diff()).iloc[1:1+end-start].to_numpy()/cfg.entry_vega_per_leg
                reward[row.label_row, code, :len(daily)] = daily
                ends.append(end)
                audits.append(dict(pair=pair, signal=str(row.date.date()), action=code,
                    last_exit=str(cal[end].date()), independent_ledger_error=ledger['independent_ledger_error'],
                    corporate_exits=int(trades.reason.eq('announced_corporate_action_exit').sum())))
            idx = meta.index[meta.label_row.eq(row.label_row)]
            meta.loc[idx, 'end'] = cal[max(ends)] if ends else cal[start+1]
            meta.loc[idx, 'label_status'] = 'complete' if ends else 'no_entry'
            meta.loc[idx, 'finite_label'] = True
            meta.loc[idx, 'target'] = (reward[row.label_row, 8].sum()-reward[row.label_row, 1].sum())/2
            print(f'Native365D corporate reward retained: {row.date.date()} {pair}', flush=True)
        temp = path.with_suffix('.reconciled.npy')
        np.save(temp, reward)
        os.replace(temp, path)
        meta.to_parquet(meta_path.with_suffix('.reconciled.parquet'), index=False)
        os.replace(meta_path.with_suffix('.reconciled.parquet'), meta_path)
    rows = pd.concat([pd.read_parquet(p) for p in sorted((OUT / 'labels').glob('*.parquet'))], ignore_index=True)
    rows.to_parquet(OUT / 'training_rows.parquet', index=False)
    rows.groupby(['signal_eligible', 'label_status']).size().rename('rows').to_csv(REPORT / 'label_coverage.csv')
    if (OUT / 'models').exists():
        os.rename(OUT / 'models', OUT / f'models_pre_corporate_{len(NATIVE_CORPORATE_EVENTS)}')
    if (OUT / 'replays').exists():
        os.rename(OUT / 'replays', OUT / f'replays_pre_corporate_{len(NATIVE_CORPORATE_EVENTS)}')
    dump(checkpoint, dict(cohorts_reconciled=int(take.sum()), actions=len(audits),
        public_events=NATIVE_CORPORATE_EVENTS, accounting_audits=audits,
        inference_universe_unchanged=True, original_labels_and_policies_preserved=True))


def fit():
    meta = features.native_iv_history(pd.read_parquet(OUT / 'training_rows.parquet'))
    meta['label'] = meta.pair.map(roster().set_index('pair').label)
    meta = policy.native_memory(meta, OUT / 'labels')
    meta.to_parquet(OUT / 'native_training_states.parquet', index=False)
    columns = features.ENTRY_FEATURES + policy.MEMORY_FEATURES
    cal = pd.read_parquet(OUT / 'market/rates.parquet').index
    for cutoff in pd.date_range('2018-01-01', '2024-04-01', freq='QS-JAN'):
        folder = OUT / 'models' / str(cutoff.date())
        if (folder / 'status.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        train = policy.training_rows(meta, cutoff)
        model, details = None, {}
        if len(train) >= 32:
            target = train.target.to_numpy(float)
            lo, hi = np.quantile(target, [.01, .99])
            tree = DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=32, random_state=2026)
            tree.fit(train[columns].to_numpy(float), np.clip(target, lo, hi))
            leaves = np.unique(tree.apply(train[columns].to_numpy(float)))
            states = np.searchsorted(leaves, tree.apply(train[columns].to_numpy(float)))
            train = train.assign(state=states)
            first, last = int(train.position.min()+1), int(cal.searchsorted(cutoff)-1)
            book = np.zeros((last-first+1, len(leaves)*9))
            for pair, group in train.groupby('pair', sort=False):
                labels = np.load(OUT / 'labels' / f'{pair}.npy', mmap_mode='r')
                for row in group.itertuples():
                    start = int(row.position+1-first)
                    length = min(labels.shape[-1], len(book)-start)
                    # Each complete outcome must already be observable.
                    assert row.end < cutoff and cal[int(row.position)+1] < cutoff
                    book[start:start+length, 9*row.state:9*(row.state+1)] += labels[int(row.label_row), :, :length].T
                del labels
            choices = {}
            for name, menu in policy.MENUS.items():
                choices[name], details[name] = policy.select_policy(book, menu, len(leaves))
            model = dict(tree=tree, leaves=leaves, choices=choices, features=columns)
        joblib.dump(model, folder / 'policy.joblib')
        dump(folder / 'status.json', dict(fit=str(cutoff.date()), training_rows=len(train),
             last_training_outcome=str(train.end.max().date()) if len(train) else None,
             fitted=model is not None, features=columns, **details))
        print(f'Native365D fit {cutoff.date()}: {len(train)} completed native episodes; {details}', flush=True)


def fit_forecast_sizes():
    """Follow-up: size the independent forecast signal without reversing it."""
    meta = pd.read_parquet(OUT / 'native_training_states.parquet')
    columns = features.ENTRY_FEATURES + policy.MEMORY_FEATURES
    cal = pd.read_parquet(OUT / 'market/rates.parquet').index
    for cutoff in pd.date_range('2018-01-01', '2024-04-01', freq='QS-JAN'):
        folder = OUT / 'forecast_sizing_models' / str(cutoff.date())
        if (folder / 'status.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        train = policy.training_rows(meta, cutoff)
        target = pd.Series(0., index=train.index)
        for pair, group in train.groupby('pair', sort=False):
            rewards = np.load(OUT / 'labels' / f'{pair}.npy', mmap_mode='r')
            for row in group.itertuples():
                aligned = policy.forecast_size_rewards(rewards[int(row.label_row)], int(row.direction))
                target.loc[row.Index] = aligned[8].sum()
            del rewards
        model, details = None, {}
        if len(train) >= 32:
            lo, hi = np.quantile(target, [.01, .99])
            tree = DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=32, random_state=2026)
            tree.fit(train[columns].to_numpy(float), np.clip(target, lo, hi))
            leaves = np.unique(tree.apply(train[columns].to_numpy(float)))
            train = train.assign(state=np.searchsorted(leaves, tree.apply(train[columns].to_numpy(float))))
            first, last = int(train.position.min()+1), int(cal.searchsorted(cutoff)-1)
            book = np.zeros((last-first+1, len(leaves)*9))
            for pair, group in train.groupby('pair', sort=False):
                rewards = np.load(OUT / 'labels' / f'{pair}.npy', mmap_mode='r')
                for row in group.itertuples():
                    assert row.end < cutoff and row.date < cutoff
                    start = int(row.position+1-first)
                    length = min(rewards.shape[-1], len(book)-start)
                    aligned = policy.forecast_size_rewards(rewards[int(row.label_row), :, :length], int(row.direction))
                    book[start:start+length, 9*row.state:9*(row.state+1)] += aligned.T
                del rewards
            choices = {}
            for name, menu in policy.FORECAST_MENUS.items():
                choices[name], details[name] = policy.select_policy(book, menu, len(leaves))
            model = dict(tree=tree, leaves=leaves, choices=choices, features=columns)
        joblib.dump(model, folder / 'policy.joblib')
        dump(folder / 'status.json', dict(fit=str(cutoff.date()), training_rows=len(train),
            last_training_outcome=str(train.end.max().date()) if len(train) else None,
            fitted=model is not None, features=columns, native_forecast_direction_frozen=True, **details))
        print(f'Native365D forecast sizing fit {cutoff.date()}: {len(train)} episodes; {details}', flush=True)


def save_replay(case, rows):
    folder = OUT / 'replays' / case
    if (folder / 'status.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    rows = policy.executable_rows(rows)
    configure_native()
    market = BoundMarket(OUT)
    def native_rate(date):
        if market.reported_year != date.year:
            market.reported_year = date.year
            dump(OUT / 'replay_progress.json', dict(case=case, date=str(date.date()),
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'Native365D replay {case}: {date.date()}', flush=True)
        return float(market.rates.loc[date])
    market.rate = native_rate
    cal = market.rates.index[market.rates.index >= '2018-01-01']
    nav, trades, decisions, audit = old.long.replay(rows, market, cal, 365)
    assert len(decisions) == len(rows)
    if len(trades):
        entry = trades.loc[trades.reason.eq('entry')]
        assert entry.groupby('cohort').secid.nunique().eq(2).all()
        assert entry.groupby('cohort').optionid.nunique().eq(2).all()
    status = dict(case=case, tenor=365, **replay_base.metrics(nav, 10_000_000),
                  entries=int(decisions.status.eq('entered').sum()), proposals=len(rows),
                  independent_native_entries=True, **audit)
    nav.to_parquet(folder / 'daily_nav.parquet')
    trades.to_csv(folder / 'trades.csv', index=False)
    decisions.to_csv(folder / 'decisions.csv', index=False)
    rows.to_parquet(folder / 'scored_allocations.parquet', index=False)
    dump(folder / 'status.json', status)
    pd.DataFrame(replay_base.annual_metrics(nav, case, 10_000_000)).to_csv(folder / 'annual.csv', index=False)
    print(f'Native365D {case}: Sharpe {status["annualized_excess_sharpe"]:.4f}, {status["entries"]} actual entries', flush=True)


def replay():
    rows = pd.read_parquet(OUT / 'native_training_states.parquet')
    rows = rows.drop(columns=['end', 'target', 'label_row', 'label_status', 'finite_label', 'position'])
    rows = rows.loc[rows.date.ge('2018-01-01')].copy().sort_values(['date', 'pair']).reset_index(drop=True)
    for case, direction, level in [('slow_variance_fixed50', 'slow_direction', 2),
                                    ('forecast_variance_fixed25', 'direction', 1),
                                    ('forecast_variance_fixed50', 'direction', 2),
                                    ('forecast_variance_fixed100', 'direction', 4),
                                    ('historical_premium_fixed50', 'z_direction', 2),
                                    ('native_iv_history_fixed50', 'iv_history_direction', 2)]:
        scored = rows.copy()
        scored['direction'] = rows[direction]
        scored['learned_level'] = np.where(rows.signal_eligible & scored.direction.ne(0), level, 0)
        save_replay(case, scored)
    schedule = {pd.Timestamp(p.parent.name): joblib.load(p) for p in sorted((OUT / 'models').glob('*/policy.joblib'))}
    for case in policy.MENUS:
        scored = rows.copy()
        for date, idx in rows.groupby('date', sort=True).groups.items():
            model = schedule[max(d for d in schedule if d <= date)]
            if model is None:
                scored.loc[idx, 'direction'] = rows.loc[idx, 'direction']
                scored.loc[idx, 'learned_level'] = np.where(rows.loc[idx, 'signal_eligible'] & rows.loc[idx, 'direction'].ne(0), 4, 0)
            else:
                states = np.searchsorted(model['leaves'], model['tree'].apply(rows.loc[idx, model['features']].to_numpy(float)))
                code = model['choices'][case][states]
                code = np.where(rows.loc[idx, 'signal_eligible'].to_numpy(), code, 0)
                direction, level = policy.decode(code)
                scored.loc[idx, 'direction'], scored.loc[idx, 'learned_level'] = direction, level
        save_replay(case, scored)
        if case == 'rl_direction_sized':
            for level in [2, 4]:
                matched = scored.copy()
                matched['learned_level'] = np.where(scored.learned_level.gt(0), level, 0)
                save_replay(f'fixed{level*25}_same_rl_direction_skips', matched)


def replay_forecast_sizes():
    rows = pd.read_parquet(OUT / 'native_training_states.parquet')
    rows = rows.drop(columns=['end', 'target', 'label_row', 'label_status', 'finite_label', 'position'])
    rows = rows.loc[rows.date.ge('2018-01-01')].copy().sort_values(['date', 'pair']).reset_index(drop=True)
    schedule = {pd.Timestamp(p.parent.name): joblib.load(p) for p in sorted((OUT / 'forecast_sizing_models').glob('*/policy.joblib'))}
    for case in policy.FORECAST_MENUS:
        scored = rows.copy()
        for date, idx in rows.groupby('date', sort=True).groups.items():
            model = schedule[max(d for d in schedule if d <= date)]
            if model is None:
                level = np.full(len(idx), 4)
            else:
                states = np.searchsorted(model['leaves'], model['tree'].apply(rows.loc[idx, model['features']].to_numpy(float)))
                _, level = policy.decode(model['choices'][case][states])
            scored.loc[idx, 'learned_level'] = np.where(rows.loc[idx, 'signal_eligible'] & rows.loc[idx, 'direction'].ne(0), level, 0)
        np.testing.assert_array_equal(scored.direction, rows.direction)
        save_replay(case, scored)
        if case == 'forecast_rl_sized':
            for level in [2, 4]:
                matched = scored.copy()
                matched['learned_level'] = np.where(scored.learned_level.gt(0), level, 0)
                save_replay(f'forecast_fixed{level*25}_same_sized_skips', matched)


def summarize():
    from scripts.report_long_tenor_search import sharpe, paired_bootstrap
    from scipy.stats import spearmanr
    records, annual, returns, quote_quality, decision_counts = [], [], {}, [], []
    roster().to_csv(REPORT / 'roster.csv', index=False)
    for folder in sorted((OUT / 'replays').iterdir()):
        if not (folder / 'status.json').exists():
            continue
        record = json.loads((folder / 'status.json').read_text())
        nav = pd.read_parquet(folder / 'daily_nav.parquet')
        r = nav.excess_return.iloc[1:]
        assert np.isclose(sharpe(r), record['annualized_excess_sharpe'])
        record['recent_2024_partial2025_sharpe'] = sharpe(r.loc['2024-01-01':])
        record['formation_2018_2023_sharpe'] = sharpe(r.loc[:'2023-12-31'])
        record['source_folder'] = str(folder)
        records.append(record)
        returns[record['case']] = r
        annual.append(pd.read_csv(folder / 'annual.csv'))
        decisions = pd.read_csv(folder / 'decisions.csv')
        decision_counts.extend(dict(case=record['case'], status=status, proposals=int(count))
                               for status, count in decisions.status.value_counts().items())
        active = nav.active_cohorts.gt(0)
        quote_quality.append(dict(case=record['case'], active_sessions=int(active.sum()),
            sessions_with_modeled_held_marks=int(nav.causal_held_mark_fallbacks.gt(0).sum()),
            modeled_held_mark_evaluations=int(nav.causal_held_mark_fallbacks.sum()),
            sessions_with_greek_fallbacks=int(nav.greek_fallbacks.gt(0).sum()),
            greek_fallback_evaluations=int(nav.greek_fallbacks.sum()),
            repaired_crossed_quote_evaluations=int(nav.repaired_held_quotes.sum())))
    frame = pd.DataFrame(records)
    assert len(frame) == 14
    frame.to_csv(REPORT / 'comparison.csv', index=False)
    pd.concat(annual, ignore_index=True).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(quote_quality).to_csv(REPORT / 'held_quote_quality.csv', index=False)
    pd.DataFrame(decision_counts).to_csv(REPORT / 'decisions_summary.csv', index=False)
    comparisons = []
    controls = {
        'rl_direction_sized': ['forecast_variance_fixed50', 'rl_direction_fixed', 'fixed50_same_rl_direction_skips', 'fixed100_same_rl_direction_skips'],
        'forecast_rl_sized': ['forecast_variance_fixed50', 'forecast_variance_fixed100', 'forecast_fixed50_same_sized_skips', 'forecast_fixed100_same_sized_skips'],
        'forecast_rl_sized_no_flat': ['forecast_variance_fixed50', 'forecast_variance_fixed100'],
    }
    for case, benchmarks in controls.items():
        for control in benchmarks:
            a, b = returns[case], returns[control]
            for block in [252, 504]:
                lo, hi = paired_bootstrap(a, b, block=block)
                comparisons.append(dict(case=case, control=control, block_sessions=block,
                    replicates=1000, sharpe_gain=sharpe(a)-sharpe(b),
                    unadjusted_gain_95_low=lo, unadjusted_gain_95_high=hi))
    pd.DataFrame(comparisons).to_csv(REPORT / 'paired_comparison.csv', index=False)
    meta = pd.read_parquet(OUT / 'native_training_states.parquet')
    meta[['date', 'pair', 'label', 'secid_a', 'secid_b', 'optionid_a', 'optionid_b', 'signal_eligible',
          'direction', 'slow_direction', 'z_direction', 'iv_history_direction'] + features.ENTRY_FEATURES + policy.MEMORY_FEATURES].to_csv(REPORT / 'entry_inputs.csv', index=False)
    observed = meta.loc[meta.signal_eligible & meta.finite_label & meta.label_status.eq('complete')]
    correlations = []
    for split, data in [('native_pre2018', observed.loc[observed.date.lt('2018-01-01')]),
                        ('native_2018_partial2025', observed.loc[observed.date.ge('2018-01-01')])]:
        for feature in features.ENTRY_FEATURES + policy.MEMORY_FEATURES:
            corr = float(spearmanr(data[feature], data.target).statistic) if data[feature].nunique() > 1 else None
            correlations.append(dict(period=split, feature=feature, episodes=len(data),
                 spearman_with_directional_hedged_pnl=corr, inference='Descriptive overlapping episodes, not independent significance test'))
    pd.DataFrame(correlations).to_csv(REPORT / 'signal_profitability.csv', index=False)
    names = pd.read_parquet(OUT / 'name_signals.parquet')
    names = names.loc[names.date.ge('2018-01-01') & names.optionid.notna() & names.future_variance.gt(0)]
    accuracy = []
    for name in ['rv60', 'rv252', 'forecast_variance']:
        take = names.loc[names[name].gt(0)]
        ratio = take.future_variance / take[name]
        accuracy.append(dict(model=name, observations=len(take), qlike=float((ratio-np.log(ratio)-1).mean()),
                             log_mse=float(np.log(ratio).pow(2).mean())))
    pd.DataFrame(accuracy).to_csv(REPORT / 'forecast_accuracy_quoted.csv', index=False)
    # The old stream is accessed only here for the independence audit, never
    # during selection, forecasts, direction generation, training, or replay.
    anchor = pd.read_parquet(old.OUT / 'anchor.parquet', columns=['date', 'pair'])
    new = meta.loc[meta.date.ge('2018-01-01') & meta.signal_eligible, ['date', 'pair']]
    overlap = new.merge(anchor, on=['date', 'pair'], how='inner')
    dump(REPORT / 'independence_audit.json', dict(old_short_policy_proposals=len(anchor),
        native_evaluation_opportunities=len(new), coincident_pair_dates=len(overlap),
        selection_inputs='Regular monthly calendar + contemporaneous native365D chains + underlying history',
        old_short_policy_used_for_selection_or_direction=False, existing_roster_is_retrospective=True))
    fits = [dict(json.loads(p.read_text()), policy_family=family) for family in ['models', 'forecast_sizing_models']
            for p in sorted((OUT / family).glob('*/status.json'))]
    assert all(n['last_training_outcome'] is None or pd.Timestamp(n['last_training_outcome']) < pd.Timestamp(n['fit']) for n in fits)
    fit_rows, split_rows = [], []
    for n in fits:
        menus = policy.MENUS if n['policy_family'] == 'models' else policy.FORECAST_MENUS
        for case in menus:
            fit_rows.append(dict(fit=n['fit'], case=case, training_rows=n['training_rows'],
                last_training_outcome=n['last_training_outcome'], fitted=n['fitted'], **n.get(case, {})))
        model = joblib.load(OUT / n['policy_family'] / n['fit'] / 'policy.joblib')
        if model is not None:
            for feature in model['tree'].tree_.feature:
                if feature >= 0:
                    split_rows.append(dict(fit=n['fit'], policy_family=n['policy_family'], feature=model['features'][feature]))
    pd.DataFrame(fit_rows).to_csv(REPORT / 'policy_fits.csv', index=False)
    pd.DataFrame(split_rows).to_csv(REPORT / 'policy_splits.csv', index=False)
    forecasts = json.loads((REPORT / 'forecast_fits.json').read_text())
    assert all(n['last_label_end'] is None or pd.Timestamp(n['last_label_end']) < pd.Timestamp(n['fit']) for n in forecasts)
    eligible = meta.loc[meta.date.ge('2018-01-01') & meta.signal_eligible]
    audits = dict(original_reference_sharpe=.39601203019410985, original_best_fixed25_sharpe=.47077543773433556,
        native_entry_independence=True, opportunities=int(len(eligible)), replay_case_count=len(frame),
        no_future_path_candidate_filter=True, maturity_purge=True, prior_completed_profitability_memory=True,
        original_execution_costs_and_risk_limits=True, all_ledgers_reconciled=True,
        one_worker=True, cpu_core_fraction=.5, rss_stop_bytes=3*1024**3,
        observed_peak_rss_bytes=json.loads((OUT / 'resource_usage.json').read_text())['observed_peak_rss_bytes'] if (OUT / 'resource_usage.json').exists() else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_fresh_holdout=True, learning_algorithm='Contextual episodic model-based direct policy optimization; not neural/sequential Q learning')
    repairs = pd.read_csv(REPORT / 'adjustment_repairs.csv')
    audits['share_factor_resets_reconciled'] = len(repairs)
    audits['share_adjustment_method'] = 'Contemporaneous return/price reconciliation; known effective dates from already published OCC memos'
    audits['derived_data_root'] = str(OUT)
    corporate_audit = json.loads(CORPORATE_CHECKPOINT.read_text())
    audits['native_corporate_reward_cohorts_reconciled'] = corporate_audit['cohorts_reconciled']
    identity_audit = json.loads((OUT / 'identifier_repair.json').read_text())
    audits['contract_identity_repair'] = identity_audit
    label_audit = json.loads((REPORT / 'label_ledger_audit.json').read_text())
    assert label_audit['native_one_year_shadow_matches_actual_single_cohort']
    audits['sampled_label_counterfactuals_match_executable_ledger'] = label_audit['sampled_counterfactuals']
    dump(REPORT / 'audit.json', audits)
    paths = [Path(__file__), Path(features.__file__), Path(policy.__file__),
             Path(old.__file__), Path(old.long.__file__), Path(replay_base.__file__),
             Path('scripts/audit_long_expiry_parity.py'), Path('scripts/uniform_smile_data.py'),
             Path('scripts/option_flex/contract_engine.py'), Path('scripts/option_flex/contract_risk.py'),
             Path('scripts/option_flex/black_scholes.py'),
             Path('data/current_rl_pairs.csv'),
             Path('scripts/run_signal_discovery_guarded.py'), Path('tests/test_native_year_features.py'), Path('tests/test_native_year_policy.py'),
             Path('scripts/curve_trading/unit_paths.py'), Path('scripts/uniform_smile_engine.py'), Path('scripts/uniform_smile_risk.py'),
             OUT / 'signal_rows.parquet', OUT / 'variance_forecasts.parquet', OUT / 'native_training_states.parquet', OUT / 'selected.parquet',
             REPORT / 'forecast_fits.json', REPORT / 'roster.csv', OUT / 'design.json', OUT / 'prepared.json']
    paths += [REPORT / 'label_ledger_audit.json', REPORT / 'adjustment_repairs.csv', CORPORATE_CHECKPOINT, OUT / 'identifier_repair.json', DATA / 'identity_recovery/CAR_2023.parquet', Path('artifacts/contracts/year_option_action_critics/P25/references.json')]
    paths += [p for p in (OUT / 'market').glob('*') if p.is_file() and p.suffix in ['.parquet', '.json']]
    paths += [p for p in (OUT / 'models').rglob('*') if p.is_file()]
    paths += [p for p in (OUT / 'forecast_sizing_models').rglob('*') if p.is_file()]
    paths += [p for p in (OUT / 'labels').glob('*') if p.is_file()]
    paths += [p for p in (OUT / 'assets').rglob('*') if p.is_file()]
    paths += [p for p in (OUT / 'replays').rglob('*') if p.is_file()]
    dump(REPORT / 'source_manifest.json', {str(p): old.file_hash(p) for p in paths})
    columns = ['case', 'annualized_excess_sharpe', 'entries', 'net_excess_pnl', 'formation_2018_2023_sharpe', 'recent_2024_partial2025_sharpe']
    ranked = frame.sort_values('annualized_excess_sharpe', ascending=False)
    values = frame.set_index('case').annualized_excess_sharpe
    outcome = (f'Best native case: {ranked.iloc[0].case}, annualized cash-excess Sharpe {ranked.iloc[0].annualized_excess_sharpe:.6f}. '
               f'Learned direction and sizing: {values["rl_direction_sized"]:.6f}; learned direction/skip with fixed100: {values["rl_direction_fixed"]:.6f}. '
               f'Matched sized-policy directions/skips at fixed50: {values["fixed50_same_rl_direction_skips"]:.6f}; '
               f'at fixed100: {values["fixed100_same_rl_direction_skips"]:.6f}. '
               f'Forecast direction frozen, learned sizing with skips: {values["forecast_rl_sized"]:.6f}; '
               f'without skips: {values["forecast_rl_sized_no_flat"]:.6f}.\n\n')
    sizing_interval = next(n for n in comparisons if n['case']=='forecast_rl_sized' and
                           n['control']=='forecast_fixed50_same_sized_skips' and n['block_sessions']==252)
    outcome += (f'Native forecast rule at fixed50: {values["forecast_variance_fixed50"]:.6f}, versus slow-RV rule {values["slow_variance_fixed50"]:.6f}. '
                f'Size-only gain over matched forecast direction/skips at fixed50: {sizing_interval["sharpe_gain"]:.6f}; '
                f'unadjusted252-session block-bootstrap95% interval [{sizing_interval["unadjusted_gain_95_low"]:.6f}, {sizing_interval["unadjusted_gain_95_high"]:.6f}]. '
                'This interval spans zero; the observed sizing gain is modest and statistically uncertain. '
                'The prior0.470775 one-year checkpoint remains numerically better but trades a different, short-policy-selected entry stream; it is preserved and is not a matched control.\n\n')
    (REPORT / 'findings.txt').write_text(
        'INDEPENDENT ONE-YEAR ENTRY SIGNAL AND EPISODIC SIZING\n\n' + outcome +
        '345 previously researched pairs,308 assets; first trading session of each month,2013–June2024. '
        'Select contemporaneous put contracts closest to365D, accepting300–400D and absolute fitted delta0.25 +/-0.075. '
        'Signal-close OI>=100, relative quoted width<=25%, standard100-share contracts. '
        'Prior-close126-session A-on-B beta0.85–1.15 and correlation>=0.5. '
        'A/B are canonical ascending security IDs; displayed names follow those IDs, independently of older label orientation. '
        'All pair/month proposals remain explicit; original30D dates,directions,RL sizes and profitability gates are not inputs. '
        'No subsequent contract path, entry fill, profitability or maturity payoff filter determines candidates.\n\n'
        'Forecast future252-session realized variance from EWMA20/60/126/252 and variance dispersion, with a pooled standardized ridge(alpha100) model. '
        'Quarterly refits use the past8 years of labels fully observed strictly before fit. '
        'The target excludes the signal-day return. Bias correction and prediction clipping use training residuals/quantiles only. '
        'Cold start uses EWMA252. Compare forecast versus EWMA60/252 using QLIKE and log error; report both since the forecast improves QLIKE but worsens log error on the broad asset/month sample.\n\n'
        'Raw expected edge per leg=(forecast variance-IV squared)/(2*IV). Pair edge is A minus B; positive means longA/shortB. '
        'The slow rule substitutes EWMA252. The historical rule normalizes each name premium against24 prior observed monthly one-year quotes,minimum12; missing history falls back to the forecast direction. '
        'Both quote observations and forecasts were available at signal close. IV is the selected25-delta put IV, not interpolated ATM IV.\n\n'
        'Additional native IV-history signal compares the current relative365D IV gap with24 previous observed monthly pair quotes,minimum6. '
        'LongA/shortB when the native gap is unusually low,reverse when unusually high; missing history falls back to the raw native IV gap direction. '
        'It uses neither short-dated IV nor the short-dated policy direction. Its z-score and availability are also states for the learner.\n\n'
        'Learning sees every eligible native state, including zero-confidence states. Three additional profitability-memory inputs use only fully completed native cohorts before the current signal date. '
        'Training rewards are actual-contract observed price paths with daily delta hedges, integer counts, cash interest, stock execution/borrow and dividends; unresolved paths are explicit missing training labels, never candidate exclusions. '
        'Portfolio training sums independent cohort rewards; executable evaluation uses shared netted stock hedges/cash, NAV-dependent sizing and all original risk limits. '
        'This shadow approximation is not exact portfolio RL training.\n\n'
        'Quarterly5-year fully completed-cohort training;32 episodes minimum;4-state tree with32-cohort minimum leaf. '
        'Enumerate every assignment of entry actions to learned states to maximize historical daily cash-excess reward Sharpe. '
        'Fixed-action learning chooses flat or either direction at100%; sized learning chooses flat or either direction at25/50/100%. '
        'Common cold-start fallback is native forecast direction at100%. Entry-only hold-to-expiry episodic policy optimization; not sequential/neural fitted-Q management. '
        'Matched fixed50/100 controls retain sized-policy directions and skips to separate sizing from trade selection, subject to actual different admissions.\n\n'
        'Follow-up after observing the initial direction learner underperform: freeze the independent native forecast direction and learn only0/25/50/100 sizing, '
        'plus a25/50/100 menu with no elective flat action; physical entry and risk admission still apply. Fit a separate four-state tree to completed PnL in that forecast direction, with the same quarterly5-year cutoff and costs. '
        'Historical size rewards are mapped to the actual forecast orientation before policy optimization; no reversal of the forecast is allowed. '
        'Fixed50/100 controls retain the size-only learner skips. All ten initial comparisons and all four follow-up replays are retained; this follow-up is exploratory.\n\n'
        'Costs and risk: $10m account; actual option commission$0.03/contract,training$0.65,0.1volpoint per leg entry penalty,1bp stock hedges, floating cash and0.5% stock borrow. '
        'Original global/pair/wing budgets;10% integer-vega matching tolerance at entry; daily delta hedging; natural individual leg expiries. '
        'No continuous vega rebalance; European intrinsic expiry settlement, no American assignment model; only declared dividends in Black–Scholes fits. '
        'Missing held quotes use the established causal Black–Scholes mark from the last observed IV and current spot; invalid fitted Greeks use the same causal-IV hedge fallback. '
        'Their frequencies are explicit in held_quote_quality.csv; the accounting check does not validate these model marks as executable quotes. '
        'Existing corporate-exit convention; newly encountered reverse splits, mergers and SunPower delisting have public dated notices. '
        f'{corporate_audit["cohorts_reconciled"]} affected training cohorts retain actual paired-close rewards at every size/direction, '
        'including their pre-exit losses and costs. Every evaluated candidate remains eligible under the same signal-close rules. '
        'Cash benchmark and independent ledger reconciliation retained.\n\n'
        f'Stock adjustment audit: {len(repairs)} documented corrections. Return-identical factor resets do not create held shares. '
        'McEwen premature July26/27 reverse-split factors follow the already published OCC50780 effective date July28. '
        'Genuine split ratios are retained. Reconciliation uses only contemporaneous observations/public notices, '
        'is tested for future-data invariance, and is applied before rebuilding forecasts, labels, policies and portfolio replays. '
        f'Derived data: {OUT}; first native construction and original short-dated controls are preserved.\n\n'
        'Contract identity: Avis option IDs reassigned across strikes on July3,2023. Held histories reconnect to the same issuer,put,expiry and original strike using observed WRDS chain quotes, '
        'with original and contemporaneous vendor IDs retained in contract_identity_recovery.csv. These old IDs cannot appear in newer native selections because their expiries have left the300–400D window. '
        'Repaired histories do not change entry contracts or discard candidates.\n\n'
        'Evaluation2018-01-02–2025-08-29,2025 partial. Annualized Sharpe is sqrt252 times daily cash-excess mean/sample std, initialization row excluded. '
        'Regular calendar makes entry opportunities independent of the old signal, but the asset/pair roster is retrospective and all evaluation years have already been researched. '
        'Results and paired circular block-bootstrap intervals (1000 draws,252/504-session blocks) are exploratory and unadjusted for research selection; no untouched holdout or live execution claim. '
        'Old0.396/.471 streams use different opportunities and are historical context, not matched controls.\n\n'
        + frame[columns].to_string(index=False) + '\n\n' + pd.DataFrame(comparisons).to_string(index=False) + '\n')
    plot_results(frame, returns)
    print(frame[columns].to_string(index=False), flush=True)


def plot_results(frame, returns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    labels = {'slow_variance_fixed50': 'Long-RV rule, fixed50', 'forecast_variance_fixed50': 'Forecast rule, fixed50',
              'historical_premium_fixed50': 'Historical premium rule, fixed50',
              'native_iv_history_fixed50': 'Native IV-history rule, fixed50',
              'rl_direction_fixed': 'Learned direction/skip, fixed100',
              'rl_direction_sized': 'Learned direction/skip/size',
              'fixed50_same_rl_direction_skips': 'Matched directions/skips, fixed50',
              'fixed100_same_rl_direction_skips': 'Matched directions/skips, fixed100',
              'forecast_rl_sized': 'Forecast direction, learned size/skip',
              'forecast_rl_sized_no_flat': 'Forecast direction, learned size only',
              'forecast_fixed50_same_sized_skips': 'Forecast direction/skips, fixed50',
              'forecast_fixed100_same_sized_skips': 'Forecast direction/skips, fixed100'}
    values = frame.set_index('case').loc[list(labels), 'annualized_excess_sharpe']
    axes[0].barh(list(labels.values()), values, color=['#64748b']*4 + ['#2563eb']*2 + ['#b45309']*2 + ['#7c3aed']*2 + ['#be185d']*2)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='black', linewidth=.5)
    axes[0].set_xlim(min(-.08, values.min()-.065), max(.1, values.max()+.07))
    axes[0].set_xlabel('Annualized daily cash-excess Sharpe')
    for i, value in enumerate(values):
        axes[0].text(value, i, f' {value:.3f}', va='center', ha='left' if value>=0 else 'right')
    annual = pd.read_csv(REPORT / 'annual.csv').pivot(index='year', columns='case', values='annualized_excess_sharpe')
    for case in ['forecast_variance_fixed50', 'historical_premium_fixed50', 'rl_direction_sized', 'fixed50_same_rl_direction_skips', 'forecast_rl_sized']:
        axes[1].plot(annual.index, annual[case], marker='o', label=labels[case])
    axes[1].axhline(0, color='black', linewidth=.5)
    axes[1].set_ylabel('Annualized daily cash-excess Sharpe')
    axes[1].set_xticks(annual.index, [f'{y}*' if y==2025 else str(y) for y in annual.index], rotation=45)
    axes[1].legend(fontsize=8)
    fig.suptitle('Independent native one-year entry signals and learned sizing')
    fig.text(.5, .015, 'Explored2018–2025 history; *2025 partial. Same execution/risk assumptions. No fresh holdout.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(REPORT / 'comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['probe', 'repair-identifiers', 'inspect-identifiers', 'inspect-invalid-quotes', 'inspect-market', 'download', 'prepare', 'histories', 'labels', 'corporate-labels', 'audit-labels', 'fit', 'fit-forecast-sizing', 'replay', 'replay-forecast-sizing', 'summarize', 'test'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    # Retain the first native construction. Quote prices/Greeks depend on raw
    # spot, not the corrected share factor, and are immutable reusable inputs.
    for source in [BASE / 'selected.parquet'] + list((BASE / 'market').glob('*_quotes.*')) + list((BASE / 'assets').rglob('*')):
        if not source.is_file():
            continue
        target = OUT / source.relative_to(BASE)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
    if (BASE / 'design.json').exists() and not (OUT / 'design.json').exists():
        dump(OUT / 'design.json', dict(json.loads((BASE / 'design.json').read_text()),
             share_factor_reconciliation='Only contemporaneous raw price and vendor daily return; factor resets do not create shares',
             base_native_construction_preserved=True))
    protected = Path('artifacts/contracts/year_option_action_critics/P25/references.json')
    references = json.loads(protected.read_text())
    assert all(old.file_hash(Path(path)) == expected for path, expected in references.items()), 'Original reference changed'
    usage_path = OUT / 'resource_usage.json'
    observed = [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]
    if usage_path.exists():
        observed.append(json.loads(usage_path.read_text())['observed_peak_rss_bytes'])
    for path in [OUT / 'progress.json', OUT / 'prepared.json']:
        if path.exists():
            observed.append(json.loads(path.read_text()).get('peak_rss_bytes', 0))
    dump(usage_path, dict(observed_peak_rss_bytes=max(observed), worker_count=1, cpu_core_fraction=.5, rss_stop_bytes=3*1024**3))
    if args.phase == 'probe':
        probe()
    elif args.phase == 'inspect-identifiers':
        inspect_identifiers()
    elif args.phase == 'inspect-invalid-quotes':
        inspect_invalid_quotes()
    elif args.phase == 'repair-identifiers':
        repair_identifiers()
    elif args.phase == 'inspect-market':
        inspect_market()
    elif args.phase == 'download':
        download_chains()
    elif args.phase == 'prepare':
        prepare()
    elif args.phase == 'histories':
        histories()
    elif args.phase == 'labels':
        labels()
    elif args.phase == 'corporate-labels':
        corporate_labels()
    elif args.phase == 'audit-labels':
        audit_labels()
    elif args.phase == 'fit':
        fit()
    elif args.phase == 'fit-forecast-sizing':
        fit_forecast_sizes()
    elif args.phase == 'replay':
        replay()
    elif args.phase == 'replay-forecast-sizing':
        replay_forecast_sizes()
    elif args.phase == 'summarize':
        summarize()
    else:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_native_year_features', 'tests.test_native_year_policy'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    dump(OUT / 'progress.json', dict(stage=args.phase, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    observed.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(usage_path, dict(observed_peak_rss_bytes=max(observed), worker_count=1, cpu_core_fraction=.5, rss_stop_bytes=3*1024**3))


if __name__ == '__main__':
    main()
