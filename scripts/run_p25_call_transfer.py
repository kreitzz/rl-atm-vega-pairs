"""Frozen P25 -> same-strike C75 attribution; no fitting or WRDS connection.

Read the small download_p25_call_transfer cache. Separate identical executions
from unchanged entry constraints, and observed prices from European Greeks.
Run: python -m scripts.run_p25_call_transfer
"""
import argparse
import json
import os
from pathlib import Path
import time
from dataclasses import replace

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import pandas as pd

from scripts import uniform_smile_engine as engine
from scripts.cheap_vega_runtime import mark_book, metrics, sharpe
from scripts.diagnose_p25_pca import Budget
from scripts.download_p25_call_transfer import ROOT, MARKET, sha256, verified_archive, write_json
from scripts.option_flex.black_scholes import invert_options, option_values
from scripts.option_flex.contract_engine import Settings, MarketRow
from scripts.submission_market import FrozenMarket, verify_ledger
from scripts.uniform_smile_risk import book_risk

DATA = ROOT / 'data/p25_call_transfer'
OUT = DATA / 'analysis'
REPORT = ROOT / 'reports/p25_call_transfer'
PERIODS = [('2018_2021', '2018-01-01', '2021-12-31'),
           ('2022_2025', '2022-01-01', '2025-08-29'),
           ('full_history', '2018-01-01', '2025-08-29')]
REUSE_COMPLETED = False


class TransferMarket:
    def __init__(self, puts, calls, mapping, budget, mode='native'):
        self.puts = puts
        self.rates = puts.rates
        self.divs = puts.divs
        self.calls = {tuple([r['date'], int(r['optionid'])]): MarketRow(r)
                      for r in calls.to_dict('records')}
        self.mapping = mapping
        self.mode = mode
        self.budget = budget
        self.started = time.monotonic()
        self.ticks = 0

    def stock(self, date, sid):
        return self.puts.stock(date, sid)

    def rate(self, date):
        self.ticks += 1
        if self.ticks % 50 == 0:
            self.budget.rest(self.started)
            self.started = time.monotonic()
        return self.puts.rate(date)

    def dividend_pv(self, *args):
        return self.puts.dividend_pv(*args)

    def quote(self, date, oid):
        if self.mode == 'put':
            return self.puts.quote(date, oid)
        q = self.calls.get((date, int(oid)))
        if q is None:
            return None
        if self.mode == 'parity':
            p = self.puts.quote(date, self.mapping[int(oid)])
            if p is None:
                raise RuntimeError('Put quote missing from paired Greek control')
            q = q.copy()
            q.update(delta=p.delta + 1, vega=p.vega, bs_iv=p.bs_iv,
                     bs_rate=p.bs_rate, known_dividend_pv=p.known_dividend_pv,
                     bs_status=p.bs_status)
        return q


class ZeroCarryMarket(TransferMarket):
    """Theoretical identity check: shared BS IV, zero carry and no frictions."""
    def __init__(self, *args, synthetic_call=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.synthetic_call = synthetic_call
        self.rates = self.puts.rates * 0
        self.divs = self.puts.divs.iloc[:0]
        iv = args[1][['date', 'put_optionid', 'put_bs_iv']].sort_values(['put_optionid', 'date']).copy()
        iv['sigma'] = iv.groupby('put_optionid').put_bs_iv.ffill()
        self.sigma = iv.set_index(['date', 'put_optionid']).sigma.to_dict()

    def rate(self, date):
        super().rate(date)
        return 0.

    def dividend_pv(self, *args):
        return 0.

    def quote(self, date, oid):
        put_oid = self.mapping[int(oid)] if self.synthetic_call else int(oid)
        p = self.puts.quote(date, put_oid)
        if p is None:
            return None
        p = p.copy()
        flag = 'C' if self.synthetic_call else 'P'
        spot = self.stock(date, p.secid)[0]
        sigma = self.sigma.get((date, put_oid), p.bs_iv)
        if not np.isfinite(sigma):
            raise RuntimeError('Theoretical control lacks a past-observed IV')
        call, delta, _, vega = option_values(spot, p.strike, (p.expiry-date).days/365, 0., sigma, 'C')
        put = max(float(call)-spot+p.strike, 1e-10)
        call = max(put+spot-p.strike, 1e-10)
        put = call-spot+p.strike
        value = call if self.synthetic_call else put
        delta = delta if self.synthetic_call else delta-1
        p.update(optionid=int(oid), cp_flag=flag, bid=float(value), ask=float(value),
                 delta=float(delta), vega=float(vega), bs_iv=sigma, bs_status='synthetic_zero_carry',
                 bs_rate=0., known_dividend_pv=0.)
        return p


def fit_calls(puts, mapping):
    q = pd.read_parquet(DATA / 'call_quotes.parquet')
    pairs = mapping[['optionid', 'put_optionid']].drop_duplicates()
    parts = []
    for year in sorted(q.date.dt.year.unique()):
        p = pd.read_parquet(MARKET / f'{year}_quotes.parquet')
        p = p.loc[p.cp_flag.eq('P')]
        columns = ['date', 'optionid', 'bid', 'ask', 'delta', 'vega', 'bs_iv', 'bs_rate', 'known_dividend_pv', 'bs_status']
        p = p[columns].rename(columns={c: 'put_' + c for c in columns if c != 'date'})
        c = q.loc[q.date.dt.year.eq(year)].merge(pairs, on='optionid', validate='many_to_one')
        c = c.merge(p, on=['date', 'put_optionid'], how='left', validate='many_to_one')
        if c.put_bs_rate.isna().any() or c.put_known_dividend_pv.isna().any():
            raise RuntimeError('Missing carry inputs for exact paired call')
        c['spot'] = [puts.stock(r.date, int(r.secid))[0] for r in c.itertuples()]
        c['bs_rate'] = c.put_bs_rate
        c['known_dividend_pv'] = c.put_known_dividend_pv
        t = (c.expiry-c.date).dt.days / 365
        mid = (c.bid+c.ask)/2
        valid = c.bid.ge(0) & c.ask.ge(c.bid) & c.standard & c.multiplier.eq(100)
        values = invert_options(c.spot-c.known_dividend_pv, c.strike, t, c.bs_rate, mid.where(valid), 'C')
        for name, value in zip(['bs_iv', 'bs_price', 'delta', 'gamma', 'vega'], values):
            c[name] = value
        c['bs_status'] = np.where(c.bs_iv.notna(), 'ok', 'invalid_input_or_price_bounds')
        expected = c.spot-c.known_dividend_pv-c.strike*np.exp(-c.bs_rate*t)
        c['parity_residual'] = mid-(c.put_bid+c.put_ask)/2-expected
        c['parity_inside_bid_ask'] = (c.bid-c.put_ask).le(expected) & (c.ask-c.put_bid).ge(expected)
        c['iv_gap_vol_points'] = 100 * (c.bs_iv-c.put_bs_iv)
        c['delta_parity_error'] = c.delta-c.put_delta-1
        c['vega_ratio'] = c.vega/c.put_vega
        parts.append(c)
    fitted = pd.concat(parts, ignore_index=True)
    fitted.to_parquet(OUT / 'fitted_calls.parquet', index=False)
    write_json(REPORT / 'quote_diagnostics.json', dict(
        call_quotes=len(fitted), native_invalid_iv=int(fitted.bs_iv.isna().sum()),
        put_invalid_iv=int(fitted.put_bs_iv.isna().sum()),
        parity_inside_bid_ask_fraction=float(fitted.parity_inside_bid_ask.mean()),
        median_abs_parity_residual_dollars=float(fitted.parity_residual.abs().median()),
        median_abs_iv_gap_vol_points=float(fitted.iv_gap_vol_points.abs().median()),
        median_delta_parity_error=float(fitted.delta_parity_error.median()),
        median_vega_ratio=float(fitted.vega_ratio.median()),
        native_fit='Call midpoint inverted independently; carry inputs copied from the same-day same-strike/expiry put',
        limitations='European BS with declared-dividend escrow; actual options can be American and EOD quotes asynchronous'))
    return fitted


def cohort(row):
    return f'{row.date.date()}|{row.pair}|{row.wing}'


def counts_for(rows, references):
    quantities = references.set_index(['cohort', 'secid']).quantity.abs()
    return {cohort(r): tuple(int(quantities.loc[cohort(r), sid]) for sid in [r.secid_a, r.secid_b])
            for r in rows.itertuples()}


def locked_vega_checks(rows, market, counts, cfg):
    result = []
    modified = rows.copy()
    for index, r in rows.iterrows():
        date = market.rates.index[market.rates.index > r.date][0]
        quotes = [market.quote(date, int(oid)) for oid in [r.optionid_a, r.optionid_b]]
        vegas = [n*100*q.vega for n, q in zip(counts[cohort(r)], quotes)]
        imbalance = abs(vegas[0]-vegas[1])/max(vegas) if np.isfinite(vegas).all() and min(vegas) > 0 else np.nan
        blocked = np.isfinite(imbalance) and imbalance > cfg.max_entry_vega_imbalance
        if blocked:
            modified.loc[index, 'optionid_a'] = np.nan
        result.append(dict(date=r.date, pair=r.pair, entry_date=date,
                           imbalance=imbalance, blocked_by_locked_vega=bool(blocked)))
    pd.DataFrame(result).to_csv(REPORT / 'native_locked_vega_checks.csv', index=False)
    return modified


def run_case(case, rows, market, cfg, counts, events, force=False, zero=False):
    folder = OUT / 'replays' / case
    folder.mkdir(parents=True, exist_ok=True)
    if REUSE_COMPLETED and not zero and (folder / 'status.json').exists():
        saved = json.loads((folder / 'status.json').read_text())
        assert saved['admission_constraints_bypassed'] == force
        assert saved['zero_carry_theoretical_control'] == zero
        pd.testing.assert_frame_equal(pd.read_parquet(folder / 'allocations.parquet').reset_index(drop=True), rows.reset_index(drop=True))
        nav = pd.read_parquet(folder / 'daily_nav.parquet')
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
        decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        cf = (-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index, fill_value=0)
        reconstructed = cf+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        assert float((reconstructed-nav.pnl).abs().max()) < 1e-5
        assert trades.groupby(['cohort', 'optionid']).quantity.sum().eq(0).all()
        assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1]) < 1e-7
        print(f'{case}: reused completed, reconciled replay', flush=True)
        return nav, trades, decisions
    # Only admission gates are bypassed in the explicit attribution arm.
    # The production-policy arms retain the frozen entry/risk constraints.
    use_cfg = replace(cfg, min_open_interest=0, max_relative_spread=1e6,
                      max_entry_vega_imbalance=1.) if force else cfg
    if zero:
        use_cfg = replace(use_cfg, option_commission=0., stock_cost_bps=0., borrowing_spread=0.,
                          stock_borrow_rate=0., option_penalty_vol_points=0.)
    saved_risk, saved_breaches = engine.book_risk, engine.risk_breaches
    try:
        engine.book_risk = book_risk
        if force:
            engine.risk_breaches = lambda *args: []
        nav, trades, decisions, ledger = engine.simulate(
            market, rows, market.rates.index[market.rates.index >= '2018-01-01'],
            use_cfg, 1_000_000, policy=lambda row, ctx: int(row['learned_level']),
            corporate_events=events, wing_capital=1_000_000, locked_entry_counts=counts)
    finally:
        engine.book_risk, engine.risk_breaches = saved_risk, saved_breaches
    mark_book(nav, market.rates, cfg.capital)
    if zero or force:
        cashflow = (-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index, fill_value=0)
        expected = cashflow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        error = float((expected-nav.pnl).abs().max())
        assert error < 1e-5
        assert trades.groupby(['cohort', 'optionid']).quantity.sum().eq(0).all()
        assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1]) < 1e-7
        audit = dict(independent_ledger_error=error, all_inventory_closed=True)
    else:
        audit = verify_ledger(nav, trades, decisions, rows)
    nav.to_parquet(folder / 'daily_nav.parquet')
    trades.to_csv(folder / 'trades.csv', index=False)
    decisions.to_csv(folder / 'decisions.csv', index=False)
    rows.to_parquet(folder / 'allocations.parquet', index=False)
    write_json(folder / 'status.json', dict(case=case, admission_constraints_bypassed=force,
               zero_carry_theoretical_control=zero, entries=int(decisions.status.eq('entered').sum()),
               **metrics(nav, cfg.capital), **ledger, audit=audit))
    print(f'{case}: {decisions.status.value_counts().to_dict()}; full-history Sharpe {sharpe(nav.excess_return.iloc[1:]):.6f}', flush=True)
    return nav, trades, decisions


def period_metrics(nav, decisions, start, stop):
    part = nav.loc[start:stop]
    first = nav.index.get_loc(part.index[0])
    previous = nav.iloc[max(0, first-1)]
    r = part.excess_return if first else part.excess_return.iloc[1:]
    return dict(annualized_excess_sharpe=sharpe(r),
                profit_above_cash=float((part.nav.iloc[-1]-part.cash_benchmark.iloc[-1])-(previous.nav-previous.cash_benchmark)),
                entries=int((decisions.status.eq('entered') & decisions.entry_date.between(part.index[0], part.index[-1])).sum()))


def components(nav, trades, start, stop):
    part = nav.loc[start:stop]
    first = nav.index.get_loc(part.index[0])
    previous = nav.iloc[max(0, first-1)]
    t = trades.loc[trades.date.between(part.index[0], part.index[-1])]
    option_gross = float((-t.quantity*100*t.price).sum()+part.option_value.iloc[-1]-previous.option_value)
    c = dict(option_gross=option_gross, stock_hedge_gross=float(part.hedge_gross_pnl.sum()),
             dividends=float(part.dividend_cash.sum()),
             financing_above_cash=float(part.interest.sum()-(part.cash_benchmark.iloc[-1]-previous.cash_benchmark)),
             option_cost=float((t.commission+t.execution_penalty).sum()),
             stock_cost=float(part.hedge_cost.sum()), stock_borrow=float(part.stock_borrow_cost.sum()))
    c['gross_with_financing'] = c['option_gross']+c['stock_hedge_gross']+c['dividends']+c['financing_above_cash']
    c['net_excess_pnl'] = c['gross_with_financing']-c['option_cost']-c['stock_cost']-c['stock_borrow']
    return c


def main():
    global REUSE_COMPLETED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-completed', action='store_true',
                        help='Reuse this investigation\'s completed cases after source, allocation and ledger checks')
    REUSE_COMPLETED = parser.parse_args().reuse_completed
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    download = json.loads((DATA / 'download_status.json').read_text())
    hashes = {str(DATA / name): digest for name, digest in download['files'].items()}
    hashes[str(DATA / 'download_plan.json')] = download['source_plan_sha256']
    manifest = json.loads((ROOT / 'data/submission/manifest.json').read_text())['files']
    names = [str((MARKET / f'{kind}.parquet').relative_to(ROOT)).replace('\\', '/') for kind in ['stocks', 'rates', 'dividends']]
    names.append('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json')
    for r in manifest:
        if r['stored'] in names:
            hashes[str(ROOT / r['stored'])] = r['sha256']
    hashes.update({str(ROOT / k): v for k, v in json.loads((DATA / 'download_plan.json').read_text())['source_hashes'].items()})
    for source in ['config/p25_submission.json', 'scripts/run_p25_call_transfer.py',
                   'scripts/uniform_smile_engine.py', 'scripts/uniform_smile_risk.py',
                   'scripts/option_flex/black_scholes.py', 'scripts/submission_market.py']:
        hashes[str(ROOT / source)] = sha256(ROOT / source)
    for path, digest in hashes.items():
        if sha256(path) != digest:
            # Git may have canonicalized the tracked JSON to Windows CRLF.
            import hashlib
            if not path.endswith('.json') or hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
                raise RuntimeError(f'Source checksum mismatch: {path}')
    frozen = verified_archive()
    cfg = Settings(**json.loads((ROOT / names[-1]).read_text()))
    events = json.loads((ROOT / 'config/p25_submission.json').read_text())['corporate_events']
    rows = pd.read_parquet(DATA / 'frozen_allocations.parquet')
    refs = pd.read_parquet(DATA / 'put_references.parquet')
    mapping = pd.read_parquet(DATA / 'counterpart_mapping.parquet')
    if mapping.optionid.isna().any() or download['missing_daily_quotes']:
        raise RuntimeError('Complete matched data required for the full frozen transfer')
    counts = counts_for(rows, refs)
    puts = FrozenMarket(MARKET)
    calls = fit_calls(puts, mapping)
    call_to_put = dict(zip(mapping.optionid.astype(int), mapping.put_optionid.astype(int)))
    transferred = rows.copy()
    for side in ['a', 'b']:
        ids = mapping.set_index(['cohort', 'secid']).optionid
        transferred[f'optionid_{side}'] = [int(ids.loc[cohort(r), getattr(r, f'secid_{side}')]) for r in rows.itertuples()]
    results = {}
    def market(mode):
        return TransferMarket(puts, calls, call_to_put, budget, mode)

    results['put_reference'] = run_case('put_reference', rows, market('put'), cfg, counts, events)
    financial_columns = ['nav', 'cash', 'option_value', 'stock_value', 'interest', 'stock_borrow_cost', 'dividend_cash', 'hedge_cost']
    original = frozen['daily_nav.parquet']
    rerun = results['put_reference'][0]
    maximum_error = float((rerun[financial_columns]-original[financial_columns]).abs().max().max())
    if maximum_error > 1e-5 or len(results['put_reference'][1]) != len(frozen['trades.csv']):
        raise RuntimeError(f'Frozen P25 baseline did not reproduce: {maximum_error}')
    np.testing.assert_array_equal(results['put_reference'][1].quantity, frozen['trades.csv'].quantity)
    results['call_parity_greeks_frozen'] = run_case('call_parity_greeks_frozen', transferred, market('parity'), cfg, counts, events, force=True)
    if len(results['call_parity_greeks_frozen'][1]) != len(frozen['trades.csv']):
        raise RuntimeError('Parity Greek control did not retain every frozen execution')
    np.testing.assert_array_equal(results['call_parity_greeks_frozen'][1].quantity, frozen['trades.csv'].quantity)
    native_market = market('native')
    checked = locked_vega_checks(transferred, native_market, counts, cfg)
    results['call_native_constraints'] = run_case('call_native_constraints', checked, native_market, cfg, counts, events)
    results['call_parity_greeks_constraints'] = run_case('call_parity_greeks_constraints', transferred, market('parity'), cfg, counts, events)
    results['call_native_frozen_eligible'] = run_case('call_native_frozen_eligible', transferred, market('native'), cfg, counts, events, force=True)
    for case in ['call_native_constraints', 'call_parity_greeks_constraints', 'call_native_frozen_eligible']:
        accepted = results[case][2].loc[lambda x: x.status.eq('entered')]
        keys = set(f'{r.signal_date.date()}|{r.pair}|P25' for r in accepted.itertuples())
        subset = rows.loc[rows.cohort.isin(keys)]
        result_case = 'put_common_' + case
        results[result_case] = run_case(result_case, subset, market('put'), cfg, {k: v for k, v in counts.items() if k in keys}, events, force=True)
        # Trade identity and counts, not merely equal sample sizes.
        call_entries = results[case][1].loc[lambda x: x.reason.eq('entry')]
        put_entries = results[result_case][1].loc[lambda x: x.reason.eq('entry')]
        for name in ['date', 'cohort', 'secid', 'quantity']:
            np.testing.assert_array_equal(call_entries[name], put_entries[name])
    for flag in [False, True]:
        name = 'synthetic_zero_call' if flag else 'synthetic_zero_put'
        z = ZeroCarryMarket(puts, calls, call_to_put, budget, synthetic_call=flag)
        results[name] = run_case(name, transferred if flag else rows, z, cfg, counts, events, force=True, zero=True)
    identity_error = float((results['synthetic_zero_call'][0].nav-results['synthetic_zero_put'][0].nav).abs().max())
    if identity_error >= 1e-5:
        raise RuntimeError(f'Theoretical parity ledger failed: {identity_error}')
    periods, pnl = [], []
    for case, (nav, trades, decisions) in results.items():
        for period, start, stop in PERIODS:
            periods.append(dict(case=case, period=period, **period_metrics(nav, decisions, start, stop)))
            c = components(nav, trades, start, stop)
            expected = periods[-1]['profit_above_cash']
            if abs(c['net_excess_pnl']-expected) >= 1e-5:
                raise RuntimeError('Period P&L components do not reconcile')
            pnl.append(dict(case=case, period=period, **c))
    pd.DataFrame(periods).to_csv(REPORT / 'period_comparison.csv', index=False)
    pd.DataFrame(pnl).to_csv(REPORT / 'pnl_components.csv', index=False)
    frequency_diagnostics(results)
    residual_diagnostics(results, refs, mapping, calls, puts.rates)
    admission_diagnostics(results, mapping, calls, cfg)
    friction_sensitivity(results, puts.rates)
    write_findings()
    write_json(REPORT / 'protocol.json', dict(status='complete', frozen_pairs=207,
               source_hashes=hashes, baseline_max_financial_error=maximum_error,
               synthetic_zero_carry_parity_max_nav_error=identity_error,
               peak_rss_mib=budget.peak, numerical_threads=1, duty_cycle=.2,
               latest_1263_period='2022-01-03 through 2025-08-29',
               direction='Identical signed option quantities; call stock hedge replaces put stock hedge',
               no_training=True, no_new_download=True,
               accounting_sensitivities='Fixed positions/hedges; removed costs and midpoint residual cashflows accrue at the original funding rate. Hypothetical diagnostic assumptions, not executable outcomes.',
               frozen_attribution='Admission/risk gates bypassed only in labeled attribution arms; constraints arms retain production gates',
               funding='Existing floating cash funding, dividends, stock borrowing; strike cashflows do not automatically cancel across legs',
               limitations=['Retrospectively inspected sample; no fresh holdout', 'Cash settlement without early exercise/assignment',
                            'Independent European call IV fitting; actual bid/ask marks can violate that model',
                            'Frozen-native arm can exclude calls with invalid entry IV; paired put common-cohort controls supplied']))
    print(pd.DataFrame(periods).loc[lambda x: x.period.eq('2022_2025')].to_string(index=False), flush=True)
    print(f'All ledgers reconciled. Baseline error {maximum_error:.3g}; zero-carry parity error {identity_error:.3g}; peak RSS {budget.peak:.1f} MiB', flush=True)


def frequency_diagnostics(results):
    diagnostics = []
    for call_case, put_case in [('call_parity_greeks_frozen', 'put_reference'),
                               ('call_native_frozen_eligible', 'put_common_call_native_frozen_eligible'),
                               ('call_native_constraints', 'put_common_call_native_constraints'),
                               ('call_parity_greeks_constraints', 'put_common_call_parity_greeks_constraints')]:
        for period, start, stop in PERIODS:
            c = results[call_case][0]
            p = results[put_case][0]
            daily_c = (c.pnl-c.cash_benchmark.diff()).loc[start:stop]
            daily_p = (p.pnl-p.cash_benchmark.diff()).loc[start:stop]
            monthly_c = daily_c.resample('ME').sum()
            monthly_p = daily_p.resample('ME').sum()
            diagnostics.append(dict(call_case=call_case, put_case=put_case, period=period,
                                    daily_excess_pnl_correlation=float(daily_c.corr(daily_p)),
                                    monthly_excess_pnl_correlation=float(monthly_c.corr(monthly_p)),
                                    daily_pnl_std_ratio=float(daily_c.std()/daily_p.std())))
    pd.DataFrame(diagnostics).to_csv(REPORT / 'frequency_diagnostics.csv', index=False)


def admission_diagnostics(results, mapping, calls, cfg):
    q = mapping[['cohort', 'date', 'optionid']].merge(calls, on=['date', 'optionid'], validate='many_to_one')
    q['low_open_interest'] = q.open_interest.lt(cfg.min_open_interest)
    q['invalid_native_signal_iv'] = q.bs_iv.isna()
    q['spread_too_wide'] = ((q.ask-q.bid)/((q.ask+q.bid)/2)).gt(cfg.max_relative_spread)
    pairs = q.groupby('cohort')[['low_open_interest', 'invalid_native_signal_iv', 'spread_too_wide']].any()
    pairs.to_csv(REPORT / 'call_signal_eligibility.csv')
    native = results['call_native_constraints'][2]
    parity = results['call_parity_greeks_constraints'][2]
    checks = pd.read_csv(REPORT / 'native_locked_vega_checks.csv')
    write_json(REPORT / 'admission_diagnostics.json', dict(
        pairs=len(pairs), pair_flags_overlap=True, **{k: int(v) for k, v in pairs.sum().items()},
        native_locked_count_vega_failures=int(checks.blocked_by_locked_vega.sum()),
        native_status_counts={k: int(v) for k, v in native.status.value_counts().items()},
        parity_greeks_status_counts={k: int(v) for k, v in parity.status.value_counts().items()},
        native_entry_risk_limits=native.loc[native.status.eq('risk_limit_entry'), 'risk_limits'].value_counts().to_dict(),
        parity_entry_risk_limits=parity.loc[parity.status.eq('risk_limit_entry'), 'risk_limits'].value_counts().to_dict()))


def friction_sensitivity(results, rates):
    """Exact positive-cash, fixed-execution sensitivity to removal of fees.

    Reinvest each removed fee at the existing floating funding rate. Positions,
    hedges and admissions stay frozen; these are accounting sensitivities.
    """
    midpoint = pd.read_parquet(OUT / 'midpoint_residual_sensitivity.parquet')
    rows = []
    for case in ['put_reference', 'call_parity_greeks_frozen']:
        nav, trades, decisions = results[case]
        if nav.cash.lt(0).any():
            raise RuntimeError('Fee sensitivity needs piecewise borrowing treatment')
        fees = (trades.commission+trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index, fill_value=0.)
        fees += nav.hedge_cost+nav.stock_borrow_cost
        cash = 0.
        accrued = []
        for i, date in enumerate(nav.index):
            if i:
                before = nav.index[i-1]
                cash *= 1+float(rates.loc[before])*(date-before).days/365
            cash += fees.loc[date]
            accrued.append(cash)
        for remove_residual in ([False, True] if case.startswith('call') else [False]):
            modified = nav.copy()
            modified['nav'] = nav.nav+pd.Series(accrued, index=nav.index)
            if remove_residual:
                modified['nav'] -= midpoint.midpoint_residual_nav_component
            modified['pnl'] = modified.nav.diff().fillna(0.)
            modified['return_'] = modified.nav.pct_change().fillna(0.)
            mark_book(modified, rates, 10_000_000)
            for period, start, stop in PERIODS:
                rows.append(dict(case=case, period=period, costs_removed=True,
                                 midpoint_residual_removed=remove_residual,
                                 **period_metrics(modified, decisions, start, stop)))
    pd.DataFrame(rows).to_csv(REPORT / 'friction_sensitivity.csv', index=False)


def write_findings():
    periods = pd.read_csv(REPORT / 'period_comparison.csv')
    selected = periods.loc[periods.period.eq('2022_2025')].set_index('case')
    pnl = pd.read_csv(REPORT / 'pnl_components.csv').loc[lambda x: x.period.eq('2022_2025')].set_index('case')
    fees = pd.read_csv(REPORT / 'friction_sensitivity.csv').loc[lambda x: x.period.eq('2022_2025') & ~x.midpoint_residual_removed].set_index('case')
    rho = pd.read_csv(REPORT / 'midpoint_residual_sensitivity.csv').set_index('period').loc['2022_2025']
    frequency = pd.read_csv(REPORT / 'frequency_diagnostics.csv').loc[lambda x: x.period.eq('2022_2025') & x.call_case.eq('call_parity_greeks_frozen')].iloc[0]
    admission = json.loads((REPORT / 'admission_diagnostics.json').read_text())
    lines = ['Frozen P25 -> exact same-strike/expiry call investigation', '',
             'The P25 volatility trade transfers positively to call counterparts, but observed daily Sharpe is lower. Execution filters, hedge costs and pricing/Greek differences prevent treating the current call backtest as an interchangeable portfolio.', '',
             'Scope and accounting',
             '- Latest frozen 27-input sizing allocations; 207 executed pairs, 414 option legs, 354 unique counterpart contracts; 8,011 daily call quotes with no missing required dates.',
             '- No tree/CEM fitting, reward optimization, new WRDS requests, or production strategy changes.',
             '- Same signed option quantities, strikes, expiries and trade dates. Long puts map to long calls plus an adjusted stock hedge and financing, not short calls. The strike cashflows do not automatically cancel across names/quantities/expiries.',
             '- The 1.263 figure refers to Jan 2022-Aug 2025. Full-history P25 Sharpe is 0.960493. The history has been inspected retrospectively; it is not a fresh holdout.', '',
             'Results for Jan 2022-Aug 2025: annualized daily excess-return Sharpe; profit above the funded cash benchmark']
    for case in ['put_reference', 'call_native_frozen_eligible', 'call_parity_greeks_frozen', 'call_native_constraints', 'put_common_call_native_constraints']:
        r = selected.loc[case]
        lines.append(f'- {case}: Sharpe {r.annualized_excess_sharpe:.6f}; excess P&L ${r.profit_above_cash:,.2f}; {int(r.entries)} pairs entered in this period.')
    numeric = pnl.select_dtypes(include='number')
    gap = numeric.loc['call_parity_greeks_frozen'] - numeric.loc['put_reference']
    lines += ['',
              'Frozen portfolios versus admitted portfolios',
              '- The parity-Greek frozen arm retains all 207 original pairs; it uses actual observed call prices with put delta+1, put vega and put IV. It bypasses OI/spread/admission/risk gates for attribution only; it is not an approved deployable strategy.',
              '- The independently fitted call arm retains 206 of 207 full-history pairs; one 2018 signal quote has no valid European call IV. It retains all 110 2022-2025 pairs. Its matched put control excludes that one 2018 trade, so carried cash/funding can differ slightly from the full P25 reference.',
              f'- With unchanged production constraints, native calls enter 67/207 full-history pairs. {admission["low_open_interest"]} pairs have at least one signal call below the inherited OI minimum of 100; {admission["spread_too_wide"]} have a wide spread; {admission["native_locked_count_vega_failures"]} breach 10% vega neutrality at the frozen counts. Flags overlap.',
              '- The remaining admitted-call rejections are reserve or gross stock hedge limits. The reserve formula charges gross option premium plus 50% of stock hedge notional, so a call substitution does not preserve that capital rule.',
              '- Matched put controls preserve each admitted-call cohort, execution dates and signed counts. Comparing the full P25 portfolio with only admitted calls would confound allocation transfer with admission.', '',
              'Costs and marks explain the gap',
              f'- In the all-trade parity-Greek control, call-minus-put net excess P&L is ${gap.net_excess_pnl:,.2f}. Extra stock hedge costs are ${gap.stock_cost:,.2f}, extra stock borrow ${gap.stock_borrow:,.2f}, and option costs differ by ${gap.option_cost:,.2f}. Gross P&L including financing differs by ${gap.gross_with_financing:,.2f}. Components reconcile to period NAV.',
              f'- Removing costs as a fixed-execution accounting sensitivity (including reinvestment of saved fees) gives put profit ${fees.loc["put_reference", "profit_above_cash"]:,.2f}, call-control profit ${fees.loc["call_parity_greeks_frozen", "profit_above_cash"]:,.2f}. Their Sharpe ratios remain {fees.loc["put_reference", "annualized_excess_sharpe"]:.6f} and {fees.loc["call_parity_greeks_frozen", "annualized_excess_sharpe"]:.6f}, so matching aggregate profit does not guarantee matching daily risk.',
              f'- Daily excess-P&L correlation is {frequency.daily_excess_pnl_correlation:.6f}, monthly correlation {frequency.monthly_excess_pnl_correlation:.6f}; call daily P&L standard deviation is {frequency.daily_pnl_std_ratio:.3f} times the put reference.',
              f'- The daily call-minus-put P&L gap correlates {rho.observed_call_put_daily_gap_correlation_with_residual:.6f} with movements in the exposure-weighted midpoint parity residual. Removing that residual from marks and entry cashflows gives hypothetical call Sharpe {rho.annualized_excess_sharpe:.6f}. This is an accounting sensitivity, not an executable-price result or proof of erroneous quotes.',
              '- A separate theoretical BS control with shared causal IV, zero rates/dividends and zero costs verifies parity at the cash-account level. Its NAV discrepancy is recorded in protocol.json.', '',
              'Interpretation and remaining limits',
              '- Independent call midpoint IV fitting changes the hedge and can fail near model bounds. Across the 8,011 rows, 635 native call IVs and 77 put IVs fail fitting; held positions use the engine\'s causal last-IV fallback. These failures are not silently dropped.',
              '- Residuals can reflect asynchronous quotes, bid/ask midpoint effects, dividend/carry approximations and American exercise premiums. This investigation quantifies their accounting effect; it does not identify which cause dominates.',
              '- The engine uses cash intrinsic settlement and does not model early exercise or assignment. This matters for assessing executable American call/put equivalence.',
              '- European put-call parity motivates matching stock and financing as well as options: C-P=S-PV(dividends)-K*exp(-rT). The zero-carry theoretical control verifies the implementation; observed prices and funded costs explain why realized Sharpe need not match.',
              '- The useful next investigation is call quote/Greek quality and comparable capital/admission rules. These results alone do not establish out-of-sample robustness or justify changing production risk limits.', '',
              'Files and reproduction',
              '- period_comparison.csv: all cases and date periods, including common-cohort put controls.',
              '- pnl_components.csv: independently reconciled options, stock, dividends, financing and cost contributions.',
              '- frequency_diagnostics.csv, midpoint_residual_sensitivity.csv, friction_sensitivity.csv: risk and accounting sensitivities.',
              '- quote_diagnostics.json, admission_diagnostics.json, call_signal_eligibility.csv, native_locked_vega_checks.csv: pricing and admission diagnostics.',
              '- protocol.json: source hashes, conventions, baseline reproduction, synthetic identity test and resource use.',
              '- Raw quotes, fitted marks and full execution/NAV ledgers remain under ignored data/p25_call_transfer/analysis.',
              '- Run in mfe_wrds: python -m scripts.run_p25_call_transfer; --reuse-completed rechecks sources, saved allocations and cash ledgers before reusing this investigation\'s completed actual-data cases.',
              '- Theory source: https://www.optionseducation.org/advancedconcepts/put-call-parity', '']
    (REPORT / 'findings.txt').write_text('\n'.join(lines), encoding='utf-8')


def residual_diagnostics(results, refs, mapping, calls, rates):
    """Accounting sensitivity to the observed European midpoint parity gap.

    Keep actual frozen trades, hedges and fees. Remove the call midpoint gap
    from marks and entry cashflows, accruing that cash adjustment at the same
    floating funding rate. A valuation diagnostic, not a tradable price feed.
    """
    call, _, decisions = results['call_parity_greeks_frozen']
    put = results['put_reference'][0]
    if call.cash.lt(0).any():
        raise RuntimeError('Residual funding sensitivity needs piecewise borrowing treatment')
    r = mapping[['cohort', 'entry_date', 'end_date', 'optionid', 'quantity']]
    joined = calls[['date', 'optionid', 'parity_residual']].merge(r, on='optionid', validate='many_to_many')
    held = joined.loc[joined.date.ge(joined.entry_date) & joined.date.lt(joined.end_date)]
    exposure = (held.quantity*100*held.parity_residual).groupby(held.date).sum().reindex(call.index, fill_value=0.)
    entering = joined.loc[joined.date.eq(joined.entry_date)]
    deposits = (entering.quantity*100*entering.parity_residual).groupby(entering.date).sum().reindex(call.index, fill_value=0.)
    accrued = []
    cash = 0.
    for i, date in enumerate(call.index):
        if i:
            previous = call.index[i-1]
            cash *= 1+float(rates.loc[previous])*(date-previous).days/365
        cash += deposits.loc[date]
        accrued.append(cash)
    cash = pd.Series(accrued, index=call.index)
    residual_nav = exposure-cash
    adjusted = call.copy()
    adjusted['nav'] = call.nav-residual_nav
    adjusted['pnl'] = adjusted.nav.diff().fillna(0.)
    adjusted['return_'] = adjusted.nav.pct_change().fillna(0.)
    mark_book(adjusted, rates, 10_000_000)
    frame = pd.DataFrame(dict(call_nav=call.nav, put_nav=put.nav,
                              midpoint_residual_nav_component=residual_nav,
                              call_nav_without_midpoint_residual=adjusted.nav))
    frame.to_parquet(OUT / 'midpoint_residual_sensitivity.parquet')
    rows = []
    for period, start, stop in PERIODS:
        diff = (call.pnl-put.pnl).loc[start:stop]
        effect = residual_nav.diff().loc[start:stop]
        rows.append(dict(period=period, **period_metrics(adjusted, decisions, start, stop),
                         observed_call_put_daily_gap_correlation_with_residual=float(diff.corr(effect)),
                         daily_gap_std=float(diff.std()), residual_daily_std=float(effect.std()),
                         gap_after_residual_adjustment_std=float((diff-effect).std())))
    pd.DataFrame(rows).to_csv(REPORT / 'midpoint_residual_sensitivity.csv', index=False)


if __name__ == '__main__':
    main()
