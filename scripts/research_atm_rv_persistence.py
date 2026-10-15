"""ATM call residual-persistence experiment, with streaming preparation.

Run phases under run_signal_discovery_guarded. The model predicts future
close-to-close RV minus today's IV; a separate actual-contract ledger determines
profit. Forward observations never determine today's features or eligibility.
"""
import argparse
from collections import OrderedDict
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import resource
import sqlite3
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CFG_PATH = Path('config/atm_rv_persistence.json')
CFG = json.loads(CFG_PATH.read_text())
SOURCE = Path('artifacts/contracts/uniform_smile/market/assets')
MARKET_SOURCE = Path('artifacts/contracts/uniform_smile/market')
OUT = Path('artifacts/contracts/atm_rv_persistence')
REPORT = Path('reports/atm_rv_persistence')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def progress(phase, **details):
    record = dict(phase=phase, time=time.time(),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **details)
    dump(OUT / 'progress.json', record)
    print(json.dumps(record, default=str), flush=True)


def assets():
    return sorted(int(p.name.split('_')[0]) for p in SOURCE.glob('*_selection.parquet'))


def calendar():
    return pd.read_parquet(MARKET_SOURCE / 'rates.parquet', use_threads=False).index


def completed_memory(dates, ends, residuals, days=365, min_count=126):
    """End strictly before decision; trailing completion-date window, no ffill."""
    dates, ends = pd.DatetimeIndex(dates), pd.DatetimeIndex(ends)
    residuals = np.asarray(residuals, float)
    valid = ends.notna() & np.isfinite(residuals)
    order = np.argsort(ends[valid].asi8, kind='stable')
    e = ends[valid].asi8[order]
    y = residuals[valid][order]
    sums = np.r_[0., np.cumsum(y)]
    hi = np.searchsorted(e, dates.asi8, side='left')
    lo = np.searchsorted(e, (dates - pd.Timedelta(days=days)).asi8, side='left')
    n = hi - lo
    mean = np.divide(sums[hi] - sums[lo], n, out=np.full(len(dates), np.nan), where=n >= min_count)
    return mean, n


def future_rv(square, entry_index, end_index):
    """Hedge opened at entry close; exclude the move into that close."""
    values = np.asarray(square)[entry_index + 1:end_index + 1]
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    return float(np.sqrt(252 * np.mean(values)))


def make_panel(selection, stock, cal, sid):
    """Vectorized preparation; cumulative sums skip no missing observations."""
    frame = selection[['date','expiry','optionid']].copy()
    positions = cal.get_indexer(frame.date)
    keep = (positions >= 0) & (positions+1 < len(cal))
    selection, frame, positions = selection.loc[keep].copy(), frame.loc[keep].copy(), positions[keep]
    frame['entry_date'] = cal[positions+1]
    frame['dte'] = (frame.expiry-frame.entry_date).dt.days
    spot = stock.close.reindex(frame.date).to_numpy()
    mid = (selection.bid+selection.ask)/2
    current = (frame.dte.between(CFG['min_dte'],CFG['max_dte']) & selection.bs_iv.gt(0)
        & np.isfinite(selection.bs_iv) & np.isfinite(spot) & (spot > 0)
        & selection.standard & selection.multiplier.eq(100) & selection.bid.gt(0)
        & selection.ask.ge(selection.bid) & selection.open_interest.ge(100)
        & ((selection.ask-selection.bid)/mid).le(.25)
        & (selection.strike/spot-1).abs().le(CFG['max_moneyness_distance']))
    frame, selection, positions = frame.loc[current].copy(), selection.loc[current], positions[current]
    price = (stock.close*stock.cfadj).where(stock.close.gt(0) & stock.cfadj.gt(0))
    square = np.log(price/price.shift(1)).pow(2)
    frame['rv21'] = np.sqrt(252*square.rolling(21,min_periods=21).mean()).reindex(frame.date).to_numpy()
    frame['secid'], frame['iv'] = sid, selection.bs_iv.to_numpy()
    end = cal.searchsorted(frame.expiry,side='right')-1
    start = positions+2
    complete = (frame.expiry <= cal[-1]).to_numpy() & (end >= start+1)
    finite = np.isfinite(square.to_numpy())
    sums = np.r_[0.,np.cumsum(square.fillna(0).to_numpy())]
    counts = np.r_[0,np.cumsum(finite)]
    length = end+1-start
    complete &= counts[end+1]-counts[start] == length
    rv = np.full(len(frame),np.nan)
    rv[complete] = np.sqrt(252*np.maximum(0.,sums[end[complete]+1]-sums[start[complete]])/length[complete])
    frame['end'] = cal[end].where(complete,pd.NaT)
    frame['target'], frame['realized_rv'] = rv-frame.iv.to_numpy(), rv
    frame['residual_memory'], frame['completed_count'] = completed_memory(
        frame.date,frame.end,frame.target,CFG['memory_days'],CFG['memory_min_completed'])
    return frame.reset_index(drop=True)


def probe():
    ids = assets()
    cal = calendar()
    examples = []
    for sid in ids[::max(1, len(ids)//6)][:6]:
        frame = pd.read_parquet(SOURCE / f'{sid}_selection.parquet', use_threads=False)
        atm = frame.loc[frame.cp_flag.eq('C') & frame.bucket.astype(str).eq('atm') & frame.tenor.eq(30)]
        examples.append(dict(secid=sid, selections=len(atm), first=atm.date.min(), last=atm.date.max(), columns=list(frame)))
    dump(REPORT / 'coverage_probe.json', dict(assets=len(ids), first=cal.min(), last=cal.max(), examples=examples))
    progress('probe', assets=len(ids), calendar_sessions=len(cal), examples=examples)


def prepare():
    (OUT / 'panel').mkdir(parents=True, exist_ok=True)
    cal = calendar()
    coverage = []
    for i, sid in enumerate(assets(), 1):
        path = OUT / 'panel' / f'{sid}.parquet'
        status = OUT / 'panel' / f'{sid}.json'
        if status.exists():
            coverage.append(json.loads(status.read_text()))
            continue
        selection = pd.read_parquet(SOURCE / f'{sid}_selection.parquet', use_threads=False)
        selection = selection.loc[selection.cp_flag.eq('C') & selection.bucket.astype(str).eq('atm') & selection.tenor.eq(30)].copy()
        selection = selection.sort_values('date').drop_duplicates('date')
        stock = pd.read_parquet(SOURCE / f'{sid}_stocks.parquet', columns=['date', 'close', 'cfadj'], use_threads=False).set_index('date').reindex(cal)
        frame = make_panel(selection, stock, cal, sid)
        if len(frame):
            frame.to_parquet(path, index=False)
        record = dict(secid=sid, rows=len(frame), eligible_memory=int(frame.residual_memory.notna().sum()) if len(frame) else 0)
        dump(status, record)
        coverage.append(record)
        if i % 20 == 0 or i == len(assets()):
            progress('prepare', assets_done=i, rows=sum(r['rows'] for r in coverage))
        del frame, selection, stock
        gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)


def linear_fit(x, y):
    """One-feature pooled OLS; no hyperparameter search or outcome screening."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        raise ValueError('Insufficient complete training observations')
    xbar, ybar = float(x.mean()), float(y.mean())
    variance = float(np.sum((x-xbar)**2))
    slope = float(np.sum((x-xbar)*(y-ybar))/variance) if variance > 1e-12 else 0.
    return ybar-slope*xbar, slope


def choose_quintile(frame, score):
    """Deterministic bottom 20%; ties resolved by security ID."""
    pieces = []
    for _, g in frame.groupby('date', sort=False):
        if len(g) < CFG['minimum_cross_section']:
            continue
        count = max(1, int(np.floor(len(g)*CFG['selection_fraction'])))
        pieces.append(g.sort_values([score, 'secid'], kind='stable').iloc[:count])
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def optimize_inputs():
    """Sufficient statistics reproduce pooled OLS without full-window tables."""
    periods = pd.period_range(CFG['evaluation_start'],CFG['last_signal'],freq='M')
    cuts = pd.DatetimeIndex([period.start_time for period in periods])
    folder = OUT / 'forecast_inputs'
    folder.mkdir(parents=True,exist_ok=True)
    records = []
    for i, path in enumerate(sorted((OUT / 'panel').glob('*.parquet')),1):
        marker = folder / (path.stem+'.json')
        if marker.exists():
            records.append(json.loads(marker.read_text()))
            continue
        frame = pd.read_parquet(path,use_threads=False)
        tr = frame.dropna(subset=['end','residual_memory','target'])
        start = np.searchsorted(cuts.asi8,pd.DatetimeIndex(tr.end).asi8,side='right')
        stop = np.searchsorted(cuts.asi8,pd.DatetimeIndex(tr.date+pd.DateOffset(years=CFG['training_years'])).asi8,side='right')
        valid = start < stop
        start, stop, tr = start[valid], stop[valid], tr.loc[valid]
        x, y = tr.residual_memory.to_numpy(), tr.target.to_numpy()
        totals = []
        for values in [np.ones(len(tr)),x,y,x*x,x*y]:
            diff = np.zeros(len(cuts)+1)
            np.add.at(diff,start,values); np.add.at(diff,stop,-values)
            totals.append(np.cumsum(diff)[:-1].tolist())
        last = []
        for j in range(len(cuts)):
            ends = tr.end.loc[(start <= j) & (stop > j)]
            last.append(str(ends.max()) if len(ends) else None)
        te = frame.loc[frame.date.ge(CFG['evaluation_start']) & frame.date.le(CFG['last_signal'])]
        te.to_parquet(folder / path.name,index=False,row_group_size=64)
        record = dict(secid=int(path.stem),totals=totals,last_completed=last)
        dump(marker,record)
        records.append(record)
        del frame, tr, te
        gc.collect()
        if i % 30 == 0:
            progress('optimize_inputs',names_done=i)
    total = np.sum([record['totals'] for record in records],axis=0)
    trained_names = np.sum([np.asarray(record['totals'][0]) > 0 for record in records],axis=0)
    fits = []
    for j, period in enumerate(periods):
        count,sx,sy,sxx,sxy = total[:,j]
        slope = (sxy-sx*sy/count)/(sxx-sx*sx/count)
        last_completed = max(record['last_completed'][j] for record in records if record['last_completed'][j] is not None)
        assert pd.Timestamp(last_completed) < period.start_time
        fits.append(dict(month=str(period),training_rows=int(round(count)),training_names=int(trained_names[j]),
                         last_completed=last_completed,intercept=(sy-slope*sx)/count,slope=slope))
    dump(OUT / 'training_fits.json',fits)
    progress('optimize_complete',names=len(records),monthly_fits=len(fits))


def predictions():
    import pyarrow.dataset as ds
    import pyarrow as pa
    dataset = ds.dataset(OUT / 'forecast_inputs', format='parquet', exclude_invalid_files=True)
    training_fits = {record['month']:record for record in json.loads((OUT / 'training_fits.json').read_text())}
    fits = []
    summaries = []
    for period in pd.period_range(CFG['evaluation_start'], CFG['last_signal'], freq='M'):
        folder = OUT / 'forecast_months' / str(period)
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / 'fit.json'
        if marker.exists():
            fits.append(json.loads(marker.read_text()))
            summaries.extend(json.loads((folder / 'quintiles.json').read_text()))
            continue
        cutoff, stop = period.start_time, period.end_time
        training = training_fits[str(period)]
        te = dataset.to_table(filter=(ds.field('date') >= cutoff) & (ds.field('date') <= min(stop, pd.Timestamp(CFG['last_signal']))), use_threads=False).to_pandas()
        te = te.dropna(subset=['residual_memory','rv21']).sort_values(['date','secid']).reset_index(drop=True)
        if training['training_rows'] < CFG['minimum_training_rows'] or te.empty:
            continue
        assert pd.Timestamp(training['last_completed']) < cutoff
        slope, intercept = training['slope'],training['intercept']
        te['residual_forecast'] = intercept+slope*te.residual_memory
        te['trailing_rv'] = te.rv21-te.iv
        te['random_quintile'] = np.random.default_rng(CFG['seed']+period.ordinal).random(len(te))
        te['quintile'] = te.groupby('date').residual_forecast.rank(method='first', pct=True).mul(5).apply(np.ceil).clip(1,5).astype(int)
        te.to_parquet(folder / 'predictions.parquet',index=False)
        for model in CFG['models']:
            chosen = choose_quintile(te, model).copy()
            chosen['pair'] = 'single_'+chosen.secid.astype(str)
            chosen['secid_a'] = chosen.secid
            chosen['secid_b'] = chosen.secid
            chosen['optionid_a'] = chosen.optionid
            chosen['optionid_b'] = np.nan
            chosen['wing'] = 'ATM_C'
            chosen['direction'] = -1
            chosen['base_level'] = 4
            chosen['priority'] = -chosen[model]
            chosen.to_parquet(folder / f'{model}_proposals.parquet',index=False)
            del chosen
        month_summaries = []
        for quintile, g in te.dropna(subset=['target']).groupby('quintile'):
            month_summaries.append(dict(quintile=int(quintile), observations=len(g),
                predicted_sum=g.residual_forecast.sum(), residual_sum=g.target.sum(),
                iv_sum=g.iv.sum(), rv_sum=g.realized_rv.sum()))
        record = dict(**training,predictions=len(te))
        dump(folder / 'quintiles.json',month_summaries)
        dump(marker,record)
        fits.append(record); summaries.extend(month_summaries)
        del te
        gc.collect()
        if len(fits) % 6 == 0:
            progress('predict', months_done=len(fits))
    for filename in ['predictions.parquet']+[f'{model}_proposals.parquet' for model in CFG['models']]:
        writer = None
        try:
            for record in fits:
                for batch in pq.ParquetFile(OUT / 'forecast_months' / record['month'] / filename).iter_batches(batch_size=4096,use_threads=False):
                    table = pa.Table.from_batches([batch])
                    if writer is None: writer = pq.ParquetWriter(OUT / filename,table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None: writer.close()
    pd.DataFrame(fits).to_csv(REPORT / 'fits.csv', index=False)
    diagnostics = pd.DataFrame(summaries).groupby('quintile').sum()
    for name, numerator in [('mean_predicted','predicted_sum'), ('mean_realized_residual','residual_sum'),
                            ('mean_iv','iv_sum'), ('mean_rv','rv_sum')]:
        diagnostics[name] = diagnostics[numerator]/diagnostics.observations
    diagnostics.to_csv(REPORT / 'forecast_quintiles.csv')
    progress('predict_complete', predictions=sum(f['predictions'] for f in fits))


QUOTE_COLUMNS = ['date','optionid','secid','strike','expiry','cp_flag','cfadj','multiplier',
                 'standard','bid','ask','delta','vega','bs_iv','bs_rate','known_dividend_pv','bs_status','open_interest']


def quote_database():
    conn = sqlite3.connect(OUT / 'quotes.sqlite')
    conn.execute('PRAGMA cache_size=-8192')
    conn.execute('PRAGMA journal_mode=WAL')
    definition = ','.join(f'{c} '+('TEXT' if c in ['date','expiry','cp_flag','bs_status'] else 'REAL') for c in QUOTE_COLUMNS)
    conn.execute(f'CREATE TABLE IF NOT EXISTS quotes ({definition}, PRIMARY KEY(date,optionid)) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS completed (secid INTEGER PRIMARY KEY, signature TEXT, rows INTEGER)')
    return conn


def prepare_quotes():
    wanted = {}
    for model in CFG['models']:
        proposals = pd.read_parquet(OUT / f'{model}_proposals.parquet', columns=['secid','optionid'], use_threads=False)
        for sid, g in proposals.groupby('secid'):
            wanted.setdefault(int(sid), set()).update(map(int, g.optionid))
        del proposals
    conn = quote_database()
    try:
        for i, (sid, ids) in enumerate(sorted(wanted.items()), 1):
            path = SOURCE / f'{sid}_quotes.parquet'
            stat = path.stat()
            signature = hashlib.sha256(json.dumps([stat.st_size,stat.st_mtime_ns,sorted(ids)]).encode()).hexdigest()
            done = conn.execute('SELECT signature,rows FROM completed WHERE secid=?',(sid,)).fetchone()
            if done:
                if done[0] != signature:
                    raise ValueError(f'Quote inputs changed for {sid}; preserve old cache and use a new study folder')
                continue
            count = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=QUOTE_COLUMNS, use_threads=False):
                frame = batch.to_pandas()
                frame = frame.loc[frame.optionid.isin(ids) & frame.date.ge(CFG['evaluation_start'])].copy()
                if frame.empty:
                    continue
                for col in ['date','expiry']:
                    frame[col] = frame[col].dt.strftime('%Y-%m-%d')
                frame['standard'] = frame.standard.astype(int)
                conn.executemany('INSERT OR IGNORE INTO quotes VALUES ('+','.join('?' for _ in QUOTE_COLUMNS)+')',
                                 frame.itertuples(index=False, name=None))
                count += len(frame)
            conn.execute('INSERT INTO completed VALUES (?,?,?)',(sid,signature,count))
            conn.commit()
            if i % 20 == 0 or i == len(wanted):
                progress('quotes', names_done=i, names_total=len(wanted))
            gc.collect()
        dump(REPORT / 'quote_coverage.json', dict(rows=conn.execute('SELECT count(*) FROM quotes').fetchone()[0],
                                                 names=conn.execute('SELECT count(*) FROM completed').fetchone()[0]))
    finally:
        conn.close()


class DiskMarket:
    """Two-session quote cache; never load a whole universe's option histories."""
    def __init__(self, secids, optionids):
        self.rates = pd.read_parquet(MARKET_SOURCE / 'rates.parquet', use_threads=False).rate
        self.rates=self.rates.loc[CFG['evaluation_start']:CFG['evaluation_end']]
        ids = pd.Index(sorted(set(map(int, secids)) | {102968,108969}))
        shape = (len(self.rates), len(ids))
        spots, adjustments = np.full(shape,np.nan), np.full(shape,np.nan)
        for batch in pq.ParquetFile(MARKET_SOURCE / 'stocks.parquet').iter_batches(
                batch_size=4096, columns=['date','secid','close','cfadj'], use_threads=False):
            frame = batch.to_pandas()
            dates, names = self.rates.index.get_indexer(frame.date), ids.get_indexer(frame.secid)
            take = (dates >= 0) & (names >= 0)
            spots[dates[take],names[take]] = frame.close.to_numpy()[take]
            adjustments[dates[take],names[take]] = frame.cfadj.to_numpy()[take]
        self.spots = pd.DataFrame(spots,index=self.rates.index,columns=ids,copy=False)
        self.adjustments = pd.DataFrame(adjustments,index=self.rates.index,columns=ids,copy=False)
        for row in CFG['observed_stock_repairs']:
            if row['secid'] in ids:
                date = pd.Timestamp(row['date'])
                self.spots.at[date,row['secid']] = row['close']
                self.adjustments.at[date,row['secid']] = row['cfadj']
        self.divs = pd.read_parquet(MARKET_SOURCE / 'dividends.parquet', use_threads=False)
        self.conn = sqlite3.connect(f'file:{OUT / "quotes.sqlite"}?mode=ro', uri=True)
        self.conn.execute('PRAGMA cache_size=-8192')
        self.allowed = set(map(int, optionids))
        self.days = OrderedDict()
        self.reported_year = None

    def stock(self, date, sid):
        if sid == 101322 and date == pd.Timestamp('2025-07-18'):
            return 1.025*self.spots.at[date,102968], self.adjustments.at[pd.Timestamp('2025-07-17'),sid]
        if sid == 111296 and date == pd.Timestamp('2024-11-22'):
            return .255*self.spots.at[date,108969], self.adjustments.at[pd.Timestamp('2024-11-21'),sid]
        return self.spots.at[date,sid], self.adjustments.at[date,sid]

    def quote(self, date, optionid):
        from scripts.option_flex.contract_engine import MarketRow
        if date not in self.days:
            quotes = {}
            for record in self.conn.execute('SELECT * FROM quotes WHERE date=?',(str(date.date()),)):
                if int(record[1]) not in self.allowed:
                    continue
                row = {k: (np.nan if value is None and k not in ['cp_flag','bs_status'] else value)
                       for k, value in zip(QUOTE_COLUMNS, record)}
                row['date'], row['expiry'] = date, pd.Timestamp(row['expiry'])
                quotes[int(row['optionid'])] = MarketRow(row)
            self.days[date] = quotes
            while len(self.days) > 2:
                self.days.popitem(last=False)
        self.days.move_to_end(date)
        return self.days[date].get(int(optionid))

    def rate(self, date):
        if self.reported_year != date.year:
            self.reported_year = date.year
            progress('replay', date=str(date.date()))
        return float(self.rates.at[date])

    def dividend_pv(self,date,sid,expiry,rate):
        dividends = self.divs.loc[self.divs.secid.eq(sid) & self.divs.declare_date.le(date)
                                  & self.divs.ex_date.gt(date) & self.divs.ex_date.le(expiry)]
        return float((dividends.amount*np.exp(-rate*(dividends.ex_date-date).dt.days/365)).sum())


def sparse_book_risk(positions,quotes,spots,date,cfg,scenario_cache=None):
    from scripts.atm_rv_persistence_engine import cash_book_risk
    from scripts.uniform_smile_risk import book_risk
    active = {sid: spots[sid] for sid in {p['secid'] for p in positions}}
    metrics = cash_book_risk(positions,quotes,active,date,cfg,scenario_cache,base_risk=book_risk)
    targets = dict.fromkeys(spots,0.)
    targets.update(metrics['target_hedge'])
    metrics['target_hedge'] = targets
    return metrics


def execution_settings():
    settings = json.loads(Path('paper_code/execution.json').read_text())
    settings.update(min_dte=CFG['min_dte'],max_dte=CFG['max_dte'],
                    max_moneyness_distance=CFG['max_moneyness_distance'])
    return settings


def replay(model,verify_month=None):
    import joblib
    from scripts import atm_rv_persistence_engine as engine
    from scripts.option_flex.contract_engine import Settings
    from scripts.cheap_vega_runtime import mark_book, metrics, annual_metrics
    from scripts.run_year_cheap_vega import EXTRA_CORPORATE_EVENTS
    cfg = Settings(**execution_settings())
    events = json.loads(Path('config/p25_submission.json').read_text())['corporate_events'] + EXTRA_CORPORATE_EVENTS + CFG['corporate_events'] + CFG['listing_risk_events']
    folder = OUT / 'replays' / model
    folder.mkdir(parents=True, exist_ok=True)
    # Keep forward labels and unused feature blocks outside the replay process.
    rows = pd.read_parquet(OUT / f'{model}_proposals.parquet',columns=[
        'date','pair','wing','secid_a','secid_b','optionid_a','optionid_b',
        'direction','base_level','priority'],use_threads=False)
    for column in ['pair','wing']:
        rows[column]=rows[column].astype('category')
    market = DiskMarket(rows.secid_a.unique(), rows.optionid_a)
    cal = market.rates.index[(market.rates.index >= CFG['evaluation_start']) &
                             (market.rates.index <= CFG['evaluation_end'])]
    engine.book_risk = sparse_book_risk
    schedule = {pd.Timestamp(date):g.sort_values(['priority','pair'],ascending=[False,True],kind='stable')
                for date,g in rows.groupby('date',sort=False)}
    try:
        state = None
        checkpoints = folder / 'months'
        checkpoints.mkdir(exist_ok=True)
        for period in pd.period_range(cal[0],cal[-1],freq='M'):
            month = checkpoints / str(period)
            marker = month / 'checkpoint.joblib'
            expected_state=expected_nav=None
            if marker.exists() and verify_month is not None and str(period) in verify_month:
                expected_state=joblib.load(marker)
                expected_nav=pd.read_parquet(month/'daily_nav.parquet',use_threads=False)
            elif marker.exists():
                state = joblib.load(marker)
                continue
            month.mkdir(exist_ok=True)
            nav, trades, decisions, audit = engine.simulate(market,rows,cal,cfg,
                CFG['per_name_reference_capital'], standalone=True,corporate_events=events,
                max_stale_quote_sessions=2, resume_state=state,through_date=min(period.end_time,cal[-1]),
                prepared_schedule=schedule)
            state = audit.pop('resume_state')
            if expected_state is not None:
                pd.testing.assert_frame_equal(nav,expected_nav,check_exact=False,rtol=0,atol=1e-8)
                core=['cash','hedge','positions','previous_spots','previous_adjustments','previous_date',
                      'previous_unavailable','previous_nav','previous_option_value','previous_rate',
                      'lag_pairs','lag_global','lag_cash','lag_names','benchmark','relative_peak','pending_management']
                assert joblib.hash({key:state[key] for key in core})==joblib.hash({key:expected_state[key] for key in core})
                dump(folder/'corporate_action_compatibility'/f'{period}.json',dict(month=str(period),
                    daily_nav_unchanged=True,terminal_economic_state_identical=True,
                    continuation_checkpoints_valid=True))
            nav.to_parquet(month / 'daily_nav.parquet')
            trades.to_pickle(month / 'trades.pkl')
            decisions.to_pickle(month / 'decisions.pkl')
            joblib.dump(state,marker)
            progress('replay_month',model=model,month=str(period),entries=int(decisions.status.eq('entered').sum()) if len(decisions) else 0,
                     peak_stress_pct=100*float((nav.stress_loss/nav.nav).max()),
                     rejections=decisions.status.value_counts().to_dict() if len(decisions) else {})
        months = sorted(path.parent for path in checkpoints.glob('????-??/checkpoint.joblib'))
        nav = pd.concat([pd.read_parquet(path / 'daily_nav.parquet',use_threads=False) for path in months])
        trades = pd.concat([pd.read_pickle(path / 'trades.pkl') for path in months],ignore_index=True)
        decisions = pd.concat([pd.read_pickle(path / 'decisions.pkl') for path in months],ignore_index=True)
        audit = dict(max_ledger_error=state['max_ledger_error'],shared_cash=True,shared_stock_hedges=True)
        mark_book(nav, market.rates, cfg.capital)
        assert nav.net_vega.le(1e-8).all()
        assert trades.loc[trades.reason.eq('entry'),'quantity'].lt(0).all()
        assert audit['max_ledger_error'] < 1e-5
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv',index=False)
        decisions.to_csv(folder / 'decisions.csv',index=False)
        result = dict(model=model,status='complete',proposals=len(rows),
                      entries=int(decisions.status.eq('entered').sum()),
                      **metrics(nav,cfg.capital),**audit)
        dump(folder / 'status.json', result)
        if (folder/'failure.json').exists():
            (folder/'failure.json').replace(folder/'resolved_failure.json')
        pd.DataFrame(annual_metrics(nav,model,cfg.capital)).to_csv(folder / 'annual.csv',index=False)
        progress('replay_complete', **result)
    except Exception as exc:
        dump(folder / 'failure.json', dict(model=model, error=str(exc)))
        raise
    finally:
        market.conn.close()


def report():
    from paper_code.metrics import period_metrics, paired_interval, sharpe
    results, annual, periods, components, rejections, audits = [], [], [], [], [], []
    returns={}
    for model in CFG['models']:
        folder = OUT / 'replays' / model
        if (folder / 'status.json').exists():
            results.append(json.loads((folder / 'status.json').read_text()))
            annual.append(pd.read_csv(folder / 'annual.csv'))
            nav = pd.read_parquet(folder / 'daily_nav.parquet',use_threads=False)
            results[-1]['annualized_return_to_volatility_ratio']=sharpe(nav.return_.iloc[1:])
            returns[model]=nav.excess_return.iloc[1:]
            decisions = pd.read_csv(folder / 'decisions.csv',usecols=['entry_date','status'],parse_dates=['entry_date'])
            trades = pd.read_csv(folder / 'trades.csv',parse_dates=['date'])
            # Reconcile from saved fills, independently of the simulator's cash ledger.
            flow = (-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
            expected = (flow+nav.option_value.diff().fillna(nav.option_value)
                        +nav.hedge_gross_pnl-nav.hedge_cost+nav.interest
                        -nav.stock_borrow_cost+nav.dividend_cash)
            error = float((expected-nav.pnl).abs().max())
            assert error < 1e-5
            assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
            charged=trades.reason.isin(['entry','announced_corporate_action_exit'])
            assert (trades.loc[charged,'commission']-trades.loc[charged,'quantity'].abs()*.03).abs().max() < 1e-8
            assert trades.loc[~charged,'commission'].eq(0).all()
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1]) < 1e-7
            audits.append(dict(model=model,independent_ledger_error=error,terminal_inventory_flat=True,
                               all_entry_quantities_short=bool(trades.loc[trades.reason.eq('entry'),'quantity'].lt(0).all())))
            for label,start,stop in [('full','2018-01-01','2025-08-29'),
                                     ('2018-2021','2018-01-01','2021-12-31'),
                                     ('2022-2025','2022-01-01','2025-08-29')]:
                periods.append(dict(model=model,period=label,**period_metrics(nav,decisions,start,stop)))
            components.append(dict(model=model,option_pnl_after_option_costs=float(nav.option_pnl.sum()),
                stock_hedge_pnl_before_costs=float(nav.hedge_gross_pnl.sum()),
                option_commission=float(trades.commission.sum()),
                option_execution_penalty=float(trades.execution_penalty.sum()),
                stock_execution_cost=float(nav.hedge_cost.sum()),interest=float(nav.interest.sum()),
                stock_borrow=float(nav.stock_borrow_cost.sum()),dividends=float(nav.dividend_cash.sum()),
                average_concurrent_cohorts=float(nav.active_cohorts.mean()),
                maximum_concurrent_cohorts=int(nav.active_cohorts.max()),
                global_risk_breach_days=int(nav.post_trade_risk_breach.sum())))
            for status,count in decisions.status.value_counts().items():
                rejections.append(dict(model=model,status=status,count=int(count)))
            del nav, decisions, trades, flow, expected
            gc.collect()
    pd.DataFrame(results).to_csv(REPORT / 'results.csv',index=False)
    pd.DataFrame(periods).to_csv(REPORT / 'periods.csv',index=False)
    pd.DataFrame(components).to_csv(REPORT / 'pnl_components.csv',index=False)
    pd.DataFrame(rejections).to_csv(REPORT / 'admissions.csv',index=False)
    dump(REPORT / 'ledger_audit.json',audits)
    if len(returns)==len(CFG['models']):
        uncertainty=[]
        for control in ['residual_forecast','trailing_rv','random_quintile']:
            for block in [63,126]:
                uncertainty.append(dict(comparison=f'residual_memory minus {control}',
                    **paired_interval(returns['residual_memory'],returns[control],block=block,draws=1000)))
        pd.DataFrame(uncertainty).to_csv(REPORT/'sharpe_comparisons.csv',index=False)
    if annual:
        pd.concat(annual).to_csv(REPORT / 'annual.csv',index=False)
    diagnostic_rows=[]
    for path in sorted((OUT/'forecast_months').glob('*/predictions.parquet')):
        cols=['date','secid','target','iv','realized_rv']+CFG['models']
        frame=pd.read_parquet(path,columns=cols,use_threads=False)
        for model in CFG['models']:
            bucket=np.ceil(frame.groupby('date')[model].rank(method='first',pct=True)*5).clip(1,5)
            for quintile,g in frame.loc[frame.target.notna()].groupby(bucket):
                diagnostic_rows.append(dict(model=model,quintile=int(quintile),count=len(g),
                    residual_sum=float(g.target.sum()),iv_sum=float(g.iv.sum()),rv_sum=float(g.realized_rv.sum())))
        del frame
    diagnostic=pd.DataFrame(diagnostic_rows).groupby(['model','quintile']).sum()
    for name in ['residual','iv','rv']:
        diagnostic[f'mean_{name}']=diagnostic[f'{name}_sum']/diagnostic['count']
    diagnostic.to_csv(REPORT/'all_ranking_quintiles.csv')
    dump(REPORT / 'protocol.json',dict(specification=CFG,
        settings=execution_settings(),
        source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            Path(__file__),Path('scripts/atm_rv_persistence_engine.py'),CFG_PATH,
            Path('paper_code/execution.json'),Path('config/p25_submission.json'),
            Path('scripts/run_year_cheap_vega.py'),Path('scripts/uniform_smile_risk.py'),
            Path('scripts/option_flex/contract_engine.py'),Path('scripts/option_flex/black_scholes.py')]},
        universe='All cached uniform-smile asset selections; retrospective universe, no pair beta or profitability gate.',
        model='One-feature pooled OLS on preceding-365-calendar-day completed RV-minus-IV residuals. Five-year training window; monthly refit.',
        entry='Bottom 20% by predicted residual at previous close; next-session integer short ATM call, $450 initial vega, overlapping cohorts.',
        accounting='Actual option and stock cash flows, daily delta hedges, declared dividends, interest and borrow, hold to expiry.',
        quotes='Observed cached selected-contract histories; last-known-IV causal held marks allowed for at most two sessions. Longer gaps halt, not retrospective trade deletion.',
        limits='Published $10m account/global risk limits; original $1m pair reference applies per name. Entry contract DTE21-45 and moneyness within10%.',
        scope='Fixed sizing first. These runs do not measure an RL improvement. Historical evaluation already examined in other research.',
        risk_note='Limits control new admissions. Existing positions can breach reserve/stress limits as markets move; hold-to-expiry does not force liquidation. Net short vega is not comparable to a vega-neutral pair book.',
        coverage_note='Missing held data halt the replay. Cached observed CRSP prices and explicitly documented public-announcement corporate/listing exits repair coverage consistently across policies. OCC cash-only merger deliverables are marked at fixed intrinsic claim value until accelerated expiration; stock hedges convert to cash. Settlement lag, early exercise and claim discounting are omitted. Missing future RV labels do not exclude current proposals, but cannot be used in the forecasting fits.'))
    progress('report', models_complete=len(results))


def write_charts_and_findings():
    """Small standalone figures; only one daily portfolio is loaded at a time."""
    results=pd.read_csv(REPORT/'results.csv').to_dict('records')
    assert len(results)==len(CFG['models'])
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=dict(residual_forecast='Rolling regression',residual_memory='Past-year residual',
               trailing_rv='Recent RV minus IV',random_quintile='Random quintile')
    fig,ax=plt.subplots(figsize=(8,4))
    for record in results:
        model=record['model']
        nav=pd.read_parquet(OUT/'replays'/model/'daily_nav.parquet',columns=['nav','cash_benchmark'],use_threads=False)
        ax.plot(nav.index,(nav.nav-nav.cash_benchmark)/1000,label=names[model],lw=1.3)
    ax.set(ylabel='Cumulative profit above cash ($ thousands)',xlabel='Date',
           title='Short approximately 30-day ATM calls: same costs, different rankings')
    ax.axhline(0,color='grey',lw=.7);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(REPORT/'cumulative_profit.png',dpi=140);plt.close(fig)
    annual=pd.read_csv(REPORT/'annual.csv')
    matrix=annual.pivot(index='case',columns='year',values='annualized_excess_sharpe').reindex(CFG['models'])
    fig,ax=plt.subplots(figsize=(8,3))
    bound=max(1.,float(np.abs(matrix.to_numpy()).max()))
    ax.imshow(matrix.to_numpy(),cmap='RdYlGn',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(len(matrix.columns)),matrix.columns)
    ax.set_yticks(range(len(matrix)),[names[m] for m in matrix.index])
    for i,row in enumerate(matrix.to_numpy()):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=9)
    ax.set_title('Annualized excess Sharpe by year (2025 through August)')
    fig.tight_layout();fig.savefig(REPORT/'annual_sharpe.png',dpi=140);plt.close(fig)
    lines=['ATM CALL RV-MINUS-IV PERSISTENCE EXPERIMENT','',
        'Tested 2018-01-02 through 2025-08-29, with last new signals in June 2025.',
        'Universe: 308 retrospectively cached equities and ETFs. No pair selection or vega neutrality.',
        'Predict realised annualized close-to-close RV through expiry minus signal-close ATM IV.',
        'Memory contains only candidate outcomes completed strictly before each signal, over the preceding year.',
        'The predictive model is a monthly pooled OLS regression on that memory, using five years of completed training outcomes.',
        'All four rankings short their daily bottom quintile using the same integer-contract sizing and risk limits.',
        'Actual next-session call fills, daily stock delta hedges, dividends, interest, borrow and expiry settlement.',
        'Costs: midpoint plus 0.10 vol-point penalty, $0.03 per option contract, and 1 bp stock turnover.',
        'Reference account $10m; target $450 initial vega per entry, scaled with NAV.',
        '', 'FULL-PERIOD RESULTS']
    for record in results:
        lines.append(f"{names[record['model']]}: excess Sharpe {record['annualized_excess_sharpe']:.3f}; entries {record['entries']}; profit above cash ${record['net_excess_pnl']:,.0f}; annualized growth relative to cash {record['relative_to_cash_cagr_pct']:.3f}%.")
    fits=pd.read_csv(REPORT/'fits.csv')
    lines += ['',f"The regression coefficient on residual memory is negative in {int(fits.slope.lt(0).sum())} of {len(fits)} monthly fits, reversing the direct historical ranking in those months."]
    lines += ['', 'Interpretation: realised RV-minus-IV is a forecasting target, not a substitute for option P&L.',
        'Headline Sharpe subtracts the cash return. The results CSV also reports the account return-to-volatility ratio without subtracting cash; that ratio includes interest on the account and should not be called excess Sharpe.',
        'The target is not gamma-weighted, and admission constraints can change which ranked proposals actually trade.',
        'The one-feature regression is a simple initial approximation; the predictive model used at work is unknown.',
        'Random selection is one fixed seed, not a full distribution of randomized portfolios.',
        'These dates were examined during prior research, and the universe is retrospective; this is exploratory replication, not an untouched holdout.',
        'Risk limits block additional entries rather than force liquidation of existing positions. Inspect breach/deployment statistics before judging scalability.',
        'This experiment has not yet evaluated RL sizing. No claim of an RL benefit follows from these results.',
        '', 'Files: results.csv (aggregate), periods.csv (two subperiods), annual.csv (year by year),',
        'pnl_components.csv (cash-flow components), admissions.csv, ledger_audit.json, protocol.json,',
        'sharpe_comparisons.csv (paired circular-block bootstrap, 1,000 draws, 63/126 sessions; conditional on these selected specifications),',
        'forecast_quintiles.csv, all_ranking_quintiles.csv, cumulative_profit.png and annual_sharpe.png.',
        'Detailed daily ledgers and monthly checkpoints are in artifacts/contracts/atm_rv_persistence/replays/.']
    lines += ['', 'REPRODUCTION',
        'Use the numerical Python environment with numpy, pandas, pyarrow and matplotlib.',
        'Run scripts.research_atm_rv_persistence through scripts.run_signal_discovery_guarded.',
        'Guard arguments: --duty-cycle 0.2 --rss-limit-mib 256 --warning-rss-limit-mib 256',
        '--min-free-percent 30 --warning-min-free-percent 30 --max-swap-growth-mib 256',
        '--pause-low-headroom-seconds 60 --stop-file artifacts/contracts/atm_rv_persistence/STOP',
        'After the guard arguments, pass -- --phase PHASE.',
        'Phase order: probe, prepare, optimize, predict, quotes, replay (each --model in config), test, report, charts.',
        'Data preparation and every monthly replay are resumable. Critical pressure, excess worker memory or swap growth halt the process.',
        'Corporate-action compatibility checks for older completed runs are recorded beside their ledgers.']
    (REPORT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    progress('charts_and_findings_complete',models=len(results))


def inspect_stock(sid, date):
    date = pd.Timestamp(date)
    paths = [MARKET_SOURCE / 'stocks.parquet',SOURCE / f'{sid}_stocks.parquet',
             Path(f'data/option_contracts/underlying/year={date.year}.parquet')]
    for path in paths:
        if not path.exists(): continue
        cols = [c for c in ['date','secid','close','cfadj'] if c in pq.ParquetFile(path).schema.names]
        frame = pd.read_parquet(path,columns=cols,filters=[('secid','==',sid),
            ('date','>=',date-pd.Timedelta(days=3)),('date','<=',date+pd.Timedelta(days=3))],use_threads=False)
        print(str(path),frame.to_string(index=False),flush=True)
    for path in Path('artifacts/broad_pairs/assets').glob(f's{sid}p*.parquet'):
        print('CRSP-derived asset columns:',pq.ParquetFile(path).schema.names,flush=True)
        frame = pd.read_parquet(path,use_threads=False)
        print(frame.loc[date-pd.Timedelta(days=3):date+pd.Timedelta(days=3)].to_string(),flush=True)
        permno = int(path.stem.split('p')[1])
        raw = Path(f'/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds/crsp_underlying_daily/year={date.year}.parquet')
        if raw.exists():
            print('Raw CRSP columns:',pq.ParquetFile(raw).schema.names,flush=True)
            fields=[c for c in ['date','permno','prc','ret','retx','cfacpr','cfacshr'] if c in pq.ParquetFile(raw).schema.names]
            observed = pd.read_parquet(raw,columns=fields,filters=[('permno','==',permno),
                ('date','>=',date-pd.Timedelta(days=3)),('date','<=',date+pd.Timedelta(days=3))],use_threads=False)
            print(observed.to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['probe','prepare','optimize','predict','quotes','replay','verify','report','charts','test','inspect'], default='probe')
    parser.add_argument('--model', choices=CFG['models'], default='residual_forecast')
    parser.add_argument('--secid',type=int,default=213108)
    parser.add_argument('--date',default='2020-10-13')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    def tests():
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_atm_rv_persistence'))
        if not result.wasSuccessful():
            raise SystemExit(1)
    dispatch = dict(probe=probe, prepare=prepare, optimize=optimize_inputs, predict=predictions, quotes=prepare_quotes, test=tests,
                    replay=lambda: replay(args.model), verify=lambda: replay(args.model,verify_month=['2023-10','2025-06','2025-07']),
                    report=report,charts=write_charts_and_findings, inspect=lambda: inspect_stock(args.secid,args.date))
    dispatch[args.phase]()


if __name__ == '__main__':
    main()
