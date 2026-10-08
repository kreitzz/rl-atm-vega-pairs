"""Maturity and Greek diagnostics on saved P25 fills; no policy/backtest changes.

Run: python -m scripts.diagnose_p25_maturity
Uses the selected 27-input policy and saved fixed-full control, reconstructing
existing shared-book attribution with scripts.shared_trade_ledger.
"""
import argparse
import gc
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from scripts.diagnose_p25_pca import Budget, ROOT, SOURCE, digest, training_window
from scripts.option_flex.black_scholes import option_values
from scripts.shared_trade_ledger import FIELDS, reconstruct_shared

MARKET = Path('data/submission/artifacts/contracts/p25_signal_transfer/market')
ARCHIVE = Path('data/p25_validation_split_experiment.zip')
SIGNALS = Path('data/submission/artifacts/contracts/p25_feature_search/signal_rows.parquet')
CONTROL = Path('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed100')
PERIODS = [('full', '2018-01-02', '2025-08-29'),
           ('validation', '2018-01-02', '2021-12-31'),
           ('evaluation', '2022-01-03', '2025-08-29')]


def maturity_band(dte):
    return pd.cut(dte, [0, 21, 28, 35, 46, np.inf], right=False,
                  labels=['<21', '21-27', '28-34', '35-45', '46+']).astype(str)


def intended_entry_dates(dates, calendar):
    """Next exchange session, including weekends/holidays; never same-day entry."""
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    positions = calendar.searchsorted(pd.DatetimeIndex(dates), side='right')
    if (positions >= len(calendar)).any():
        raise ValueError('Missing next exchange session')
    return pd.Series(calendar[positions], index=dates.index)


def entry_greeks(spot, strike, dte, rate, sigma, dividend_pv=0.):
    """European put, frozen dividend escrow; vega is dollars per vol point."""
    effective = spot - dividend_pv
    t = dte / 365
    if not np.isfinite([effective, strike, t, rate, sigma]).all() or min(effective, strike, t, sigma) <= 0:
        raise ValueError('Invalid entry pricing inputs')
    price, delta, gamma, vega = option_values(effective, strike, t, rate, sigma, 'P')
    d1 = (np.log(effective / strike) + (rate + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    density = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)
    theta = (-effective * density * sigma / (2 * np.sqrt(t)) +
             rate * strike * np.exp(-rate * t) * ndtr(-d2)) / 365
    # Constant escrow: an underlying $1 move is also an effective-spot $1 move.
    gamma_1pct = .5 * float(gamma) * (.01 * spot) ** 2
    return dict(model_price=float(price), model_delta=float(delta), gamma=float(gamma),
                model_vega=float(vega), theta_day=float(theta),
                gamma_1pct_per_vega=gamma_1pct / float(vega),
                theta_day_per_vega=float(theta) / float(vega))


def pair_metadata(legs):
    """Pair-level measures; keep earlier-leg close and later-leg close distinct."""
    records = []
    for cohort, own in legs.groupby('cohort', sort=False):
        if len(own) != 2 or set(own.role) != {'long', 'short'}:
            raise ValueError('Expected exactly one long and one short option leg')
        long = own.loc[own.role.eq('long')].iloc[0]
        short = own.loc[own.role.eq('short')].iloc[0]
        if long.entry_date != short.entry_date:
            raise ValueError('Leg entry dates differ')
        records.append(dict(cohort=cohort, pair=cohort.split('|')[1],
                            entry_date=long.entry_date, exit_date=own.exit_date.max(),
                            entry_dte_mean=own.entry_dte.mean(),
                            entry_dte_min=own.entry_dte.min(), entry_dte_max=own.entry_dte.max(),
                            expiry_gap_days=abs(long.entry_dte-short.entry_dte),
                            mean_holding_days=own.holding_days.mean(),
                            both_legs_open_days=own.holding_days.min(),
                            cohort_holding_days=own.holding_days.max(),
                            one_leg_tail_days=abs(long.holding_days-short.holding_days),
                            long_initial_vega=long.initial_vega, short_initial_vega=short.initial_vega,
                            mean_initial_vega=own.initial_vega.mean(),
                            vega_imbalance=abs(long.initial_vega-short.initial_vega)/own.initial_vega.max(),
                            gamma_1pct_gross_per_mean_vega=(own.initial_vega*own.gamma_1pct_per_vega).sum()/own.initial_vega.mean(),
                            theta_day_net_per_mean_vega=(own.initial_vega*own.theta_day_per_vega*own.role.map({'long': 1, 'short': -1})).sum()/own.initial_vega.mean()))
    result = pd.DataFrame(records).set_index('cohort')
    result['maturity_band'] = maturity_band(result.entry_dte_mean)
    result['expiry_match'] = np.where(result.expiry_gap_days.eq(0), 'same', 'different')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/p25_maturity_diagnostic'))
    parser.add_argument('--label-checks-only', action='store_true',
                        help='Inspect original labels, frozen states and saved admission; no fill reconstruction')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    manifests = json.loads((ROOT / 'data/submission/manifest.json').read_text())['files']
    expected = {row['stored']: row['sha256'] for row in manifests}
    archive_expected = json.loads((ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    hashes = {}
    newline_normalized = {}

    def verify(path):
        key = path.as_posix()
        if key not in hashes:
            hashes[key] = digest(ROOT / path)
            if hashes[key] != expected[key]:
                # Git on Windows converts tracked CSV line endings to CRLF.
                import hashlib
                raw = (ROOT / path).read_bytes()
                canonical = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
                if path.suffix != '.csv' or canonical != expected[key]:
                    raise ValueError(f'Submission hash mismatch: {key}')
                newline_normalized[key] = canonical
        return ROOT / path

    def parquet(path, **kwargs):
        return pd.read_parquet(verify(path), use_threads=False, **kwargs)

    def bundled(archive, name, is_csv=False):
        import hashlib
        raw = archive.read(name)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != archive_expected[name]:
            raise ValueError(f'Archive hash mismatch: {name}')
        hashes[str(ARCHIVE) + '::' + name] = sha
        return pd.read_csv(io.BytesIO(raw)) if is_csv else pd.read_parquet(io.BytesIO(raw), use_threads=False)

    if args.label_checks_only:
        with threadpool_limits(limits=1), zipfile.ZipFile(ROOT / ARCHIVE) as archive:
            label_checks(out, budget, parquet, verify, bundled, archive, archive_expected, hashes)
        protocol = dict(source_hashes=hashes, lf_canonical_hashes=newline_normalized,
                        script_sha256=digest(Path(__file__)), model_refitted=False,
                        backtest_replayed=False, threads=1, duty_cycle=.2,
                        peak_rss_mib=budget.peak,
                        population='Deduplicated union of 90 original training windows, plus complete saved later proposals; training and later results separate',
                        label_units='Full-size standalone cash-excess P&L / fixed 450 entry-vega budget',
                        risk_units='Same saved daily label units, entry through final settlement inclusive',
                        inference='Descriptive associations; overlapping trades, no independent-observation p-values')
        (out / 'label_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
        print(f'Label checks complete; peak RSS {budget.peak:.1f} MiB; {out}', flush=True)
        return

    all_legs, all_pairs, period_groups, reconciliations = [], [], [], []
    cfg = json.loads((ROOT / 'config/p25_submission.json').read_text())
    fees = dict(stock_cost_bps=cfg['costs']['stock_hedge_bps'], stock_borrow_rate=.003)
    # The latter is the unchanged replay Settings stock-borrow convention.
    with threadpool_limits(limits=1), zipfile.ZipFile(ROOT / ARCHIVE) as archive:
        proposals = parquet(SIGNALS, columns=['date', 'pair', 'optionid_a', 'optionid_b', 'dte_mean', 'oriented_dte_gap'])
        selected_root = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
        selected_decisions = bundled(archive, selected_root + 'decisions.csv', True)
        selected_decisions['signal_date'] = pd.to_datetime(selected_decisions.signal_date)
        selected_decisions['entry_date'] = pd.to_datetime(selected_decisions.entry_date)
        proposal_meta = proposals.rename(columns={'date': 'signal_date'}).merge(
            selected_decisions[['signal_date', 'entry_date', 'pair', 'action', 'status']],
            on=['signal_date', 'pair'], validate='one_to_one')
        if len(proposal_meta) != 608:
            raise ValueError('Incomplete proposal roster')
        stage = time.monotonic()
        rates = parquet(MARKET / 'rates.parquet')
        calendar = pd.DatetimeIndex(rates.index)
        if not intended_entry_dates(proposal_meta.signal_date, calendar).equals(proposal_meta.entry_date):
            raise ValueError('Exchange-calendar entry dates differ from saved decisions')
        train = parquet(SOURCE, columns=['date', 'end', 'signal_eligible', 'finite_label', 'dte_mean', 'oriented_dte_gap'])
        train['entry_date'] = intended_entry_dates(train.date, calendar)
        train['entry_dte_mean'] = train.dte_mean-(train.entry_date-train.date).dt.days
        train['entry_dte_min'] = train.entry_dte_mean-train.oriented_dte_gap.abs()/2
        train['entry_dte_max'] = train.entry_dte_mean+train.oriented_dte_gap.abs()/2
        seen = set()
        training_rows = []
        for name in sorted(n for n in archive.namelist() if '/models/control/' in n and n.endswith('.json')):
            import hashlib
            raw = archive.read(name)
            sha = hashlib.sha256(raw).hexdigest()
            if sha != archive_expected[name]:
                raise ValueError(f'Training metadata hash mismatch: {name}')
            hashes[str(ARCHIVE)+'::'+name] = sha
            meta = json.loads(raw)
            window = training_window(train, meta['fit'])
            if len(window) != meta['training_rows']:
                raise ValueError('Training population differs from archived tree')
            seen.update(window.index)
            training_rows.append(dict(fit=meta['fit'], training_rows=len(window),
                entry_dte_min=window.entry_dte_min.min(), entry_dte_max=window.entry_dte_max.max(),
                entry_dte_mean=window.entry_dte_mean.mean(),
                differing_expiry_fraction=window.oriented_dte_gap.ne(0).mean()))
        training_summary = pd.DataFrame(training_rows)
        training_summary.to_csv(out / 'training_maturity.csv', index=False)
        training_unique = train.loc[sorted(seen)]
        training_range = (float(training_unique.entry_dte_min.min()), float(training_unique.entry_dte_max.max()))
        del train, rates
        budget.rest(stage)
        lag = (proposal_meta.entry_date-proposal_meta.signal_date).dt.days
        proposal_meta['entry_dte_mean'] = proposal_meta.dte_mean-lag
        proposal_meta['entry_dte_min'] = proposal_meta.entry_dte_mean-proposal_meta.oriented_dte_gap.abs()/2
        proposal_meta['entry_dte_max'] = proposal_meta.entry_dte_mean+proposal_meta.oriented_dte_gap.abs()/2
        proposal_meta['expiry_gap_days'] = proposal_meta.oriented_dte_gap.abs()
        proposal_meta['maturity_band'] = maturity_band(proposal_meta.entry_dte_mean)
        proposal_meta['expiry_match'] = np.where(proposal_meta.expiry_gap_days.eq(0), 'same', 'different')
        proposal_meta['chosen_size'] = proposal_meta.action.abs()
        if not proposal_meta.chosen_size.isin([.5, .75, 1.]).all():
            raise ValueError('Unexpected selected sizing action')
        proposal_meta.to_csv(out / 'proposals.csv', index=False)
        proposal_summary = []
        for period, first, last in PERIODS:
            own = proposal_meta.loc[proposal_meta.signal_date.between(first, last)]
            for grouping in ['maturity_band', 'expiry_match']:
                for category, g in own.groupby(grouping, observed=True):
                    proposal_summary.append(dict(period=period, grouping=grouping, category=category,
                                                 proposals=len(g), entered=int(g.status.eq('entered').sum()),
                                                 entered_fraction=g.status.eq('entered').mean(),
                                                 average_chosen_size=g.chosen_size.mean(), mean_entry_dte=g.entry_dte_mean.mean()))
        pd.DataFrame(proposal_summary).to_csv(out / 'proposal_summary.csv', index=False)
        for arm in ['selected', 'fixed100']:
            started = time.monotonic()
            if arm == 'selected':
                nav = bundled(archive, selected_root + 'daily_nav.parquet')
                trades = bundled(archive, selected_root + 'trades.csv', True)
                decisions = selected_decisions.copy()
            else:
                nav = parquet(CONTROL / 'daily_nav.parquet')
                trades = pd.read_csv(verify(CONTROL / 'trades.csv'))
                decisions = pd.read_csv(verify(CONTROL / 'decisions.csv'))
                decisions['signal_date'] = pd.to_datetime(decisions.signal_date)
                decisions['entry_date'] = pd.to_datetime(decisions.entry_date)
            trades['date'] = pd.to_datetime(trades.date)
            if set(trades.reason)-{'entry', 'expiry_settlement'}:
                raise ValueError('Corporate exits need a separate maturity interpretation')
            ids = trades.optionid.unique().tolist()
            secids = trades.secid.unique().tolist()
            quote_parts = []
            for year in range(2018, 2026):
                stage = time.monotonic()
                quote_parts.append(parquet(MARKET / f'{year}_quotes.parquet', filters=[('optionid', 'in', ids)]))
                budget.rest(stage)
            started = time.monotonic()
            quotes = pd.concat(quote_parts, ignore_index=True)
            del quote_parts
            stocks = parquet(MARKET / 'stocks.parquet', columns=['date', 'secid', 'close', 'cfadj'], filters=[('secid', 'in', secids)])
            dividends = parquet(MARKET / 'dividends.parquet', columns=['secid', 'ex_date', 'amount'], filters=[('secid', 'in', secids)])
            print(f'Reconciling {arm}: {trades.cohort.nunique()} saved cohorts', flush=True)
            # Existing attribution, not a fills simulator; shared financing stays separate.
            legs, daily, financing, _, _, error, components = reconstruct_shared(nav, trades, quotes, stocks, dividends, fees)
            entries = trades.loc[trades.reason.eq('entry'), ['cohort', 'optionid', 'date']]
            initial = entries.merge(quotes, on=['date', 'optionid'], validate='many_to_one').merge(
                stocks, on=['date', 'secid'], validate='many_to_one')
            if len(initial) != len(entries):
                raise ValueError('Missing initial quote')
            greek_records = []
            for r in initial.itertuples():
                greek_records.append(dict(cohort=r.cohort, optionid=r.optionid,
                    **entry_greeks(r.close, r.strike, (r.expiry-r.date).days, r.bs_rate, r.bs_iv, r.known_dividend_pv)))
            legs = legs.merge(pd.DataFrame(greek_records), on=['cohort', 'optionid'], validate='one_to_one')
            legs['holding_days'] = (legs.exit_date-legs.entry_date).dt.days
            if not legs.holding_days.eq(legs.entry_dte.where(legs.entry_dte.ge(0))).all():
                # Weekend expirations settle on the final exchange session, not Sunday.
                if not (legs.entry_dte-legs.holding_days).between(0, 4).all():
                    raise ValueError('Unexpected holding/expiry discrepancy')
            legs['arm'] = arm
            legs['maturity_band'] = maturity_band(legs.entry_dte)
            pairs = pair_metadata(legs)
            pairs['arm'] = arm
            pairs['signal_date'] = pd.to_datetime(pairs.index.str.split('|').str[0])
            keys = decisions.assign(cohort=decisions.signal_date.dt.strftime('%Y-%m-%d')+'|'+decisions.pair+'|P25').set_index('cohort')
            pairs['chosen_size'] = keys.action.abs().reindex(pairs.index)
            pairs['lifecycle_net_before_shared_financing'] = daily.sum().reindex(pairs.index)
            pairs['lifecycle_costs'] = sum(components[f].sum() for f in FIELDS[3:]).reindex(pairs.index)
            pairs['lifecycle_net_per_entry_vega'] = pairs.lifecycle_net_before_shared_financing/pairs.mean_initial_vega
            for period, first, last in PERIODS:
                use = nav.index.to_series().between(first, last).to_numpy()
                period_pnl = daily.loc[use].sum().reindex(pairs.index)
                attributed = float(period_pnl.sum())
                funding = float(financing.loc[use].sum())
                benchmark = nav.cash_benchmark.diff().fillna(0)
                portfolio_excess = float((nav.pnl-benchmark).loc[use].sum())
                if abs(attributed+funding-portfolio_excess) > 1e-5:
                    raise ValueError('Period P&L does not reconcile')
                reconciliations.append(dict(arm=arm, period=period, net_before_shared_financing=attributed,
                                            shared_financing=funding, portfolio_profit_above_cash=portfolio_excess,
                                            daily_reconstruction_max_error=error))
                for grouping in ['maturity_band', 'expiry_match']:
                    for category, g in pairs.groupby(grouping, observed=True):
                        new = g.loc[g.entry_date.between(first, last)]
                        pnl = period_pnl.reindex(g.index)
                        period_groups.append(dict(arm=arm, period=period, grouping=grouping, category=category,
                            new_entries=len(new), contributing_cohorts=int(pnl.ne(0).sum()),
                            calendar_net_before_shared_financing=pnl.sum(),
                            new_entry_lifecycle_win_fraction=new.lifecycle_net_before_shared_financing.gt(0).mean(),
                            new_entry_median_lifecycle_net_per_vega=new.lifecycle_net_per_entry_vega.median(),
                            new_entry_worst_lifecycle_net=new.lifecycle_net_before_shared_financing.min(),
                            new_entry_median_holding_days=new.cohort_holding_days.median(),
                            new_entry_mean_chosen_size=new.chosen_size.mean(),
                            new_entry_median_gamma_1pct_gross_per_mean_vega=new.gamma_1pct_gross_per_mean_vega.median(),
                            new_entry_median_theta_day_net_per_mean_vega=new.theta_day_net_per_mean_vega.median()))
            all_legs.append(legs)
            all_pairs.append(pairs.reset_index())
            del nav, trades, quotes, stocks, dividends, components, daily, financing
            gc.collect()
            budget.rest(started)
        legs = pd.concat(all_legs, ignore_index=True)
        pairs = pd.concat(all_pairs, ignore_index=True)
        summary = pd.DataFrame(period_groups)
        legs.to_csv(out / 'executed_legs.csv', index=False)
        pairs.to_csv(out / 'executed_pairs.csv', index=False)
        summary.to_csv(out / 'period_summary.csv', index=False)
        pd.DataFrame(reconciliations).to_csv(out / 'reconciliation.csv', index=False)
        exposure = legs.groupby(['arm', 'maturity_band'], observed=True).agg(
            legs=('cohort', 'size'), median_dte=('entry_dte', 'median'),
            median_gamma_1pct_per_vega=('gamma_1pct_per_vega', 'median'),
            p10_gamma_1pct_per_vega=('gamma_1pct_per_vega', lambda s: s.quantile(.1)),
            p90_gamma_1pct_per_vega=('gamma_1pct_per_vega', lambda s: s.quantile(.9)),
            median_theta_day_per_vega=('theta_day_per_vega', 'median'),
            median_holding_days=('holding_days', 'median')).reset_index()
        exposure.to_csv(out / 'exposure_by_leg_maturity.csv', index=False)
        stage = time.monotonic()
        plot_results(out, proposal_meta, pairs, legs, summary)
        budget.rest(stage)
    selected = pairs.loc[pairs.arm.eq('selected')]
    selected_legs = legs.loc[legs.arm.eq('selected')]
    later = summary.loc[summary.arm.eq('selected') & summary.period.eq('evaluation') & summary.grouping.eq('maturity_band')]
    exposures = exposure.loc[exposure.arm.eq('selected')].set_index('maturity_band')
    gamma_ratio = exposures.loc['21-27', 'median_gamma_1pct_per_vega']/exposures.loc['35-45', 'median_gamma_1pct_per_vega']
    theta_ratio = exposures.loc['21-27', 'median_theta_day_per_vega']/exposures.loc['35-45', 'median_theta_day_per_vega']
    lines = ['# Maturity diagnostic for the selected P25 policy', '',
        'Descriptive inspection of saved trades for the 27-input policy associated with 1.263 excess Sharpe. No retraining, new order simulation, maturity restriction, or policy change.', '',
        '## Maturity and holding periods', '',
        f'- All 608 proposals: actual intended-entry DTE ranges {proposal_meta.entry_dte_min.min():.0f}–{proposal_meta.entry_dte_max.max():.0f} days. Signal-date DTE is longer by the signal-to-entry calendar lag; it is not the holding period.',
        f'- All 90 saved tree training windows were reconstructed. Their union contains {len(seen)} distinct pair/date examples, with intended-entry leg DTE {training_range[0]:.0f}–{training_range[1]:.0f} days. Training examples overlap between monthly windows.',
        f'- {int(proposal_meta.expiry_gap_days.ne(0).sum())} / 608 proposals have different leg expirations ({proposal_meta.expiry_gap_days.ne(0).mean():.1%}); largest expiry gap {proposal_meta.expiry_gap_days.max():.0f} days.',
        f'- Selected policy: {len(selected)} executed pairs; leg entry DTE {selected_legs.entry_dte.min():.0f}–{selected_legs.entry_dte.max():.0f}; actual leg holding {selected_legs.holding_days.min():.0f}–{selected_legs.holding_days.max():.0f} calendar days.',
        f'- {int(selected.expiry_gap_days.ne(0).sum())} / {len(selected)} executed pairs have different expirations. The longest single-leg tail is {selected.one_leg_tail_days.max():.0f} days; delta hedging continues but initial two-leg vega matching no longer applies once one leg settles.',
        '- Every saved fill in the two arms is an entry or expiry settlement. Calendar expiry can be up to a few days after the final exchange-session settlement.', '',
        '## Selected policy: 2022–August 2025', '',
        '| Mean entry DTE | New pairs | Calendar net before shared financing | New-entry lifecycle win fraction | Mean chosen size |',
        '|---|---:|---:|---:|---:|']
    lines += [f'| {r.category} | {r.new_entries} | ${r.calendar_net_before_shared_financing:,.0f} | {r.new_entry_lifecycle_win_fraction:.1%} | {r.new_entry_mean_chosen_size:.1%} |' for r in later.itertuples()]
    lines += ['', 'Calendar P&L includes carried positions from before a period boundary. Lifecycle statistics describe new entries and can include their subsequent runoff. Shared cash financing is separate, not allocated arbitrarily to maturity groups. Stock execution and borrow costs use the existing absolute-hedge-change / short-hedge allocation. Different groups have different names, entry dates, sizes, and overlapping positions; this is not a controlled maturity experiment or a per-band Sharpe comparison.', '',
        '## Entry gamma and theta', '',
        'European Black–Scholes Greeks are computed from the saved entry quote, actual expiry, spot, rate, IV and declared-dividend escrow, following the repo pricing convention. Theta holds dividend escrow fixed and is dollars per calendar day. It is an instantaneous model sensitivity, not realized decay or isolated theta P&L.', '',
        'Gamma is expressed as approximate option P&L for a 1% move in its own underlying, divided by entry vega in dollars per volatility point. Leg gamma is positive; pair gross gamma adds magnitudes rather than assuming separate stock shocks cancel. Pair theta signs reflect long versus short. Matching vega does not equate these exposures.', '',
        f'Among selected executed legs, the median gamma-per-vega measure is {gamma_ratio:.2f} times as high in the 21–27D band as in the 35–45D band; median absolute theta per vega is {theta_ratio:.2f} times as high. These groups also have different IVs and names, so these ratios are descriptive rather than isolated causal maturity effects.', '',
        '## Interpretation', '',
        'The precise two-week-versus-one-month example is not present in the saved proposal pool: minimum entry DTE is 21. The training-window range above checks the learner\'s actual examples separately. Nevertheless, duration varies and different leg expirations can leave a meaningful single-leg tail. Maturity pooling remains an assumption to investigate, not an established model failure.', '',
        'Compare exposure_by_leg_maturity.csv and period_summary.csv before proposing a restriction. Outcomes are confounded by IV, pair identity, market period, sizes and admission. An eventual narrow-DTE or same-expiry strategy must be refitted and replayed as a new controlled experiment; filtering these outcomes would not reproduce that strategy.', '',
        '## Reproduction and audit', '',
        '`python -m scripts.diagnose_p25_maturity` uses bundled sources, one numerical thread, 20% duty cycle, a 384 MiB RSS cap and macOS-only 35% memory-headroom floor. It reuses the existing shared-trade attribution, with daily option/stock/cost/P&L reconciliation and separate financing.', '',
        '- `proposals.csv` / `proposal_summary.csv`: all qualified proposals, actual-entry maturity and sizing/admission.',
        '- `training_maturity.csv`: all 90 original tree training populations and their maturity ranges.',
        '- `executed_legs.csv` / `executed_pairs.csv`: holdings, actual entry exposures and lifecycle attribution.',
        '- `period_summary.csv`: selected and fixed-full descriptions by maturity and expiry match.',
        '- `reconciliation.csv`: attributed P&L plus financing reconciles to the funded portfolio.',
        '- `exposure_by_leg_maturity.csv`: leg exposure distributions.',
        '- `maturity_and_exposure.png` / `outcomes_by_maturity.png`: diagnostic figures.',
        '- `protocol.json`: verified source hashes, accounting conventions and resource usage. Tracked CSVs with Windows CRLF are also verified against the manifest after LF canonicalization; data values are unchanged.', '']
    (out / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    protocol = dict(source_hashes=hashes, lf_canonical_hashes=newline_normalized, archive_sha256=digest(ROOT / ARCHIVE),
        script_sha256=digest(Path(__file__)), policy='screened_selected; features/control',
        control='saved $10m fixed100; different executed portfolios',
        backtest_replayed=False, model_refitted=False,
        costs=fees, attribution='scripts.shared_trade_ledger.reconstruct_shared; financing separate',
        theta='European put theta; dividend escrow held fixed; per calendar day',
        gamma='0.5 * gamma * (0.01*spot)^2 / vega_per_vol_point; not a shock simulation',
        maturity_bands='[0,21), [21,28), [28,35), [35,46), [46,infinity)',
        periods=PERIODS, peak_rss_mib=budget.peak, threads=1, duty_cycle=.2)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    print(f'Diagnostic complete; peak RSS {budget.peak:.1f} MiB; {out}', flush=True)


def frozen_leaf_ids(model, rows):
    """Traverse saved split arrays with sklearn's float32 input convention."""
    if model['tree'] is None:
        return np.zeros(len(rows), dtype=int)
    tree = model['tree'].tree_
    x = rows[model['features']].to_numpy(dtype=np.float32)
    nodes = np.zeros(len(rows), dtype=int)
    active = np.ones(len(rows), dtype=bool)
    while active.any():
        indices = np.flatnonzero(active)
        parents = nodes[indices]
        terminal = tree.children_left[parents] == -1
        active[indices[terminal]] = False
        indices, parents = indices[~terminal], parents[~terminal]
        left = x[indices, tree.feature[parents]] <= tree.threshold[parents]
        nodes[indices] = np.where(left, tree.children_left[parents], tree.children_right[parents])
    return nodes


def saved_path_metrics(path, sessions):
    """Exclude padding, retain actual zero-return days, and validate final exit."""
    if sessions < 2 or sessions > len(path) or np.any(path[sessions:] != 0):
        raise ValueError('Saved daily path does not match entry-to-exit calendar')
    daily = np.asarray(path[:sessions], dtype=float)
    if not np.isfinite(daily).all():
        raise ValueError('Nonfinite original label path')
    cumulative = np.r_[0., daily.cumsum()]
    return dict(holding_sessions=sessions, daily_sd=daily.std(ddof=1),
                realized_quadratic_exposure=np.square(daily).sum(),
                worst_day=daily.min(), path_max_drawdown=(np.maximum.accumulate(cumulative)-cumulative).max())


def label_checks(out, budget, parquet, verify, bundled, archive, archive_expected, hashes):
    """Descriptive failure-mechanism checks; never fit or optimize a model."""
    import hashlib
    import joblib
    import warnings
    features = json.loads((ROOT / 'config/p25_validation_2018_2021.json').read_text())['screened']['inputs']
    stage = time.monotonic()
    columns = list(dict.fromkeys(features + ['date', 'end', 'position', 'pair', 'direction',
                          'label_row', 'target', 'signal_eligible', 'finite_label']))
    rows = parquet(SOURCE, columns=columns)
    calendar = pd.DatetimeIndex(parquet(MARKET / 'rates.parquet').index)
    if not np.array_equal(calendar.get_indexer(rows.date), rows.position):
        raise ValueError('Label signal positions differ from saved calendar')
    rows['entry_date'] = intended_entry_dates(rows.date, calendar)
    rows['entry_dte_mean'] = rows.dte_mean-(rows.entry_date-rows.date).dt.days
    rows['holding_days'] = (rows.end-rows.entry_date).dt.days
    rows['abs_target'] = rows.target.abs()
    rows['quarter'] = rows.date.dt.to_period('Q').astype(str)
    metadata, seen = [], set()
    for name in sorted(n for n in archive.namelist() if '/models/control/' in n and n.endswith('.json')):
        raw = archive.read(name)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != archive_expected[name]:
            raise ValueError('Frozen training metadata hash mismatch')
        hashes[str(ARCHIVE)+'::'+name] = sha
        meta = json.loads(raw)
        window = training_window(rows, meta['fit'])
        if len(window) != meta['training_rows']:
            raise ValueError('Training population does not match original model')
        seen.update(window.index)
        metadata.append((name, meta))
    signals = parquet(SIGNALS, columns=['date', 'pair']+features)
    proposal_indices = rows.set_index(['date', 'pair']).index.get_indexer(signals.set_index(['date', 'pair']).index)
    if (proposal_indices < 0).any() or not np.allclose(rows.iloc[proposal_indices][features], signals[features]):
        raise ValueError('Saved proposal features do not match original training source')
    proposal_indices = rows.index[proposal_indices]
    complete_indices = proposal_indices[rows.loc[proposal_indices].finite_label]
    rows = rows.loc[sorted(seen | set(complete_indices))].copy()
    for pair, group in rows.groupby('pair'):
        path = Path('data/submission/artifacts/contracts/uniform_smile/pairs') / pair / 'P25_labels.npy'
        disk = np.load(verify(path), mmap_mode='r')
        for r in group.itertuples():
            original = disk[r.label_row, 8 if r.direction > 0 else 1]
            if not np.isclose(original.sum(), r.target, rtol=1e-6, atol=1e-5):
                raise ValueError('Original full-size daily path does not sum to tree label')
            metrics = saved_path_metrics(original, calendar.get_indexer([r.end])[0]-r.position)
            for key, value in metrics.items():
                rows.loc[r.Index, key] = value
        del disk
    budget.rest(stage)
    rows['maturity_band'] = maturity_band(rows.entry_dte_mean)
    later_rows = rows.loc[complete_indices].copy()
    later_rows['fit'] = later_rows.date.dt.to_period('M').dt.to_timestamp().dt.strftime('%Y-%m-%d')
    known_fits = {meta['fit'] for _, meta in metadata}
    if not set(later_rows.fit).issubset(known_fits):
        raise ValueError('Proposal has no saved contemporaneous monthly model')
    rows = rows.loc[sorted(seen)].copy()
    rows.to_csv(out / 'label_examples.csv', index=True, index_label='source_row')
    band_records, correlations = [], []
    outcomes = ['target', 'abs_target', 'daily_sd', 'realized_quadratic_exposure', 'path_max_drawdown']
    for period, own in [('all', rows), ('outcome_before_2022', rows.loc[rows.end.lt('2022-01-01')]),
                        ('outcome_2022_onward', rows.loc[rows.end.ge('2022-01-01')])]:
        for band, g in own.groupby('maturity_band', observed=True):
            band_records.append(dict(period=period, band=band, examples=len(g),
                mean_dte=g.entry_dte_mean.mean(), mean_holding_days=g.holding_days.mean(),
                mean_target=g.target.mean(), median_target=g.target.median(), target_sd=g.target.std(),
                win_fraction=g.target.gt(0).mean(), median_abs_target=g.abs_target.median(),
                median_daily_sd=g.daily_sd.median(), median_quadratic_exposure=g.realized_quadratic_exposure.median(),
                median_max_drawdown=g.path_max_drawdown.median()))
        for duration in ['entry_dte_mean', 'holding_days']:
            for y in outcomes:
                for method in ['pearson', 'spearman']:
                    correlations.append(dict(period=period, adjustment='none', duration=duration,
                        outcome=y, method=method, examples=len(own), correlation=own[duration].corr(own[y], method=method)))
                # Descriptive centering only, no fitted predictor; exclude tiny/constant cells.
                grouped = own.groupby(['pair', 'quarter'])
                include = (grouped[duration].transform('size').ge(4) & grouped[duration].transform('nunique').ge(2))
                dx = own[duration]-grouped[duration].transform('mean')
                dy = own[y]-grouped[y].transform('mean')
                correlations.append(dict(period=period, adjustment='within_pair_and_signal_quarter',
                    duration=duration, outcome=y, method='pearson', examples=int(include.sum()),
                    correlation=dx[include].corr(dy[include])))
    pd.DataFrame(band_records).to_csv(out / 'label_maturity_summary.csv', index=False)
    corr = pd.DataFrame(correlations)
    corr.to_csv(out / 'label_duration_correlations.csv', index=False)
    leaf_records, memberships = [], []
    for name, meta in metadata:
        stage = time.monotonic()
        model_name = name.replace('.json', '.joblib')
        raw = archive.read(model_name)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != archive_expected[model_name]:
            raise ValueError('Frozen model hash mismatch')
        hashes[str(ARCHIVE)+'::'+model_name] = sha
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            model = joblib.load(io.BytesIO(raw))
        if model['features'] != features:
            raise ValueError('Unexpected frozen feature roster')
        window = training_window(rows, meta['fit'])
        window['leaf'] = frozen_leaf_ids(model, window)
        window['fit'] = meta['fit']
        memberships.append(window[['fit', 'leaf', 'pair', 'quarter', 'date', 'entry_dte_mean',
                                   'holding_days', 'target', 'daily_sd', 'path_max_drawdown']])
        proposals = later_rows.loc[later_rows.fit.eq(meta['fit'])]
        if len(proposals):
            later_rows.loc[proposals.index, 'leaf'] = frozen_leaf_ids(model, proposals)
            means = window.groupby('leaf').target.mean()
            risks = window.groupby('leaf').daily_sd.median()
            state = later_rows.loc[proposals.index, 'leaf'].astype(int)
            later_rows.loc[proposals.index, 'training_state_mean_target'] = state.map(means)
            later_rows.loc[proposals.index, 'training_state_median_daily_sd'] = state.map(risks)
            later_rows.loc[proposals.index, 'size_code'] = state.map(dict(zip(model['leaves'], model['choices'])))
        if model['tree'] is not None:
            tree = model['tree'].tree_
            for leaf, g in window.groupby('leaf'):
                if len(g) != tree.n_node_samples[leaf] or not np.isclose(g.target.mean(), tree.value[leaf].item(), atol=1e-5):
                    raise ValueError('Frozen leaf sample count/target mean does not reproduce')
        for leaf, g in window.groupby('leaf'):
            leaf_records.append(dict(fit=meta['fit'], leaf=leaf, examples=len(g),
                dte_min=g.entry_dte_mean.min(), dte_max=g.entry_dte_mean.max(),
                holding_min=g.holding_days.min(), holding_max=g.holding_days.max(),
                short_examples=int(g.entry_dte_mean.lt(28).sum()), long_examples=int(g.entry_dte_mean.ge(35).sum()),
                mean_target=g.target.mean(), target_sd=g.target.std(),
                daily_sd_p10=g.daily_sd.quantile(.1), daily_sd_p90=g.daily_sd.quantile(.9),
                duration_target_correlation=g.holding_days.corr(g.target) if g.holding_days.nunique()>1 and g.target.nunique()>1 else np.nan))
        budget.rest(stage)
    leaves = pd.DataFrame(leaf_records)
    leaves.to_csv(out / 'label_frozen_leaf_ranges.csv', index=False)
    state_maturity_checks(out, pd.concat(memberships), later_rows, bundled, archive)
    # Saved option fills give actual lifespans; settlements occur before next orders.
    root = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
    decisions = bundled(archive, root+'decisions.csv', True)
    trades = bundled(archive, root+'trades.csv', True)
    for col in ['signal_date', 'entry_date']:
        decisions[col] = pd.to_datetime(decisions[col])
    trades['date'] = pd.to_datetime(trades.date)
    entries = trades.loc[trades.reason.eq('entry')].groupby('cohort').date.min()
    exits = trades.loc[trades.reason.eq('expiry_settlement')].groupby(['cohort', 'optionid']).date.max()
    lifetimes = pd.DataFrame(dict(entry_date=entries, exit_date=exits.groupby(level=0).max(),
                                  first_exit_date=exits.groupby(level=0).min()))
    lifetimes['pair'] = lifetimes.index.str.split('|').str[1]
    lifetimes['mean_holding_days'] = (exits.groupby(level=0).mean()-lifetimes.entry_date).dt.total_seconds()/86400
    inventory = []
    for r in decisions.itertuples():
        # Only earlier days; proposal roster has no duplicate pair/date orders.
        own = lifetimes.loc[lifetimes.pair.eq(r.pair) & lifetimes.entry_date.lt(r.entry_date) & lifetimes.exit_date.gt(r.entry_date)]
        inventory.append(dict(signal_date=r.signal_date, entry_date=r.entry_date, pair=r.pair, status=r.status,
            active_same_pair=len(own), long_holding_same_pair=int(own.mean_holding_days.ge(35).sum()),
            tail_same_pair=int(own.first_exit_date.le(r.entry_date).sum()),
            max_remaining_calendar_days=(own.exit_date-r.entry_date).dt.days.max(),
            max_elapsed_calendar_days=(r.entry_date-own.entry_date).dt.days.max(),
            risk_limits=r.risk_limits))
    inventory = pd.DataFrame(inventory)
    inventory.to_csv(out / 'label_admission_inventory.csv', index=False)
    inv_summary = inventory.groupby('status').agg(decisions=('pair', 'size'),
        with_active_same_pair=('active_same_pair', lambda s: int(s.gt(0).sum())),
        with_long_holding_same_pair=('long_holding_same_pair', lambda s: int(s.gt(0).sum())),
        with_single_leg_tail=('tail_same_pair', lambda s: int(s.gt(0).sum())),
        median_remaining_days=('max_remaining_calendar_days', 'median'))
    inv_summary.to_csv(out / 'label_admission_summary.csv')
    mixed = leaves.loc[leaves.short_examples.gt(0) & leaves.long_examples.gt(0)]
    latest = leaves.loc[leaves.fit.eq(leaves.fit.max())]
    show = corr.loc[corr.period.eq('all') & corr.duration.eq('holding_days') & corr.method.eq('pearson')]
    lines = ['# Original P25 labels and frozen-state diagnostic', '',
        f'{len(rows)} distinct examples actually used by the 90 saved trees. Original full-size paths reproduce every target; frozen leaf counts and target means reproduce every tree. No training, optimizer search or new strategy replay.', '',
        '## Does a longer holding period mean a larger label or greater risk?', '',
        '| Saved-path quantity | Holding-days correlation | Within pair and quarter |',
        '|---|---:|---:|']
    for y in outcomes:
        raw = show.loc[show.outcome.eq(y) & show.adjustment.eq('none')].iloc[0]
        centered = show.loc[show.outcome.eq(y) & show.adjustment.ne('none')].iloc[0]
        lines.append(f'| {y} | {raw.correlation:.3f} | {centered.correlation:.3f} (n={centered.examples}) |')
    lines += ['', 'The label is cumulative standalone P&L above cash divided by the fixed 450 vega budget, not a return standardized to 30 days. Daily standard deviation excludes zero padding after settlement. Quadratic exposure sums squared daily P&L: it can rise with time or unusually large moves, and is not an estimate of independent-day variance. These are associations, confounded by names, IV and market dates; overlapping trades prevent treating rows as independent trials.', '',
        '## Do frozen states pool materially different durations?', '',
        f'{len(mixed)} / {len(leaves)} monthly leaf populations contain both mean-entry maturities below 28 days and at least 35 days. Windows reuse examples, so this count is not independent evidence.',
        f'Latest tree ({leaves.fit.max()}): {len(latest)} leaves; {int((latest.short_examples.gt(0)&latest.long_examples.gt(0)).sum())} contain both bands. See label_frozen_leaf_ranges.csv for duration spans and within-state risk dispersion.', '',
        '## Do open positions coincide with rejected new entries?', '',
        '| Decision status | Count | Same pair open | Long-held pair open | Single-leg tail open |',
        '|---|---:|---:|---:|---:|',
        *[f'| {status} | {int(r.decisions)} | {int(r.with_active_same_pair)} | {int(r.with_long_holding_same_pair)} | {int(r.with_single_leg_tail)} |'
          for status, r in inv_summary.iterrows()], '',
        'Inventory is reconstructed from actual prior entries still open after that day\'s expiry settlements. "Long holding" means average actual leg holding at least 35 calendar days. Counts concern the same pair. A rejected trade with open inventory is evidence that the mechanism can be present; it does not prove maturity caused rejection or that a shorter holding policy would improve Sharpe. Risk caps can reject a trade even with no existing same-pair position.', '',
        '## Reproduce', '',
        '`python -m scripts.diagnose_p25_maturity --label-checks-only`', '',
        'Outputs: label_examples.csv, label_maturity_summary.csv, label_duration_correlations.csv, label_frozen_leaf_ranges.csv, label_admission_inventory.csv, label_admission_summary.csv and label_protocol.json. Core training dates, directional labels and frozen models are verified against source manifests. Centered correlations restrict pair/quarter cells to at least four examples and at least two durations; see CSVs for sample counts and earlier/later subperiods.', '']
    (out / 'label_findings.md').write_text('\n'.join(lines), encoding='utf-8')
    print(show.to_string(index=False), flush=True)
    print(inv_summary.to_string(), flush=True)


def state_maturity_checks(out, training, later, bundled, archive):
    """Conditional descriptive checks on frozen states, before/after fitting."""
    # Verify the archived sizing policy without predicting using newer models.
    scored = bundled(archive, 'artifacts/contracts/p25_validation_split/replays/screened_selected/scored_allocations.parquet')
    expected = scored.set_index(['date', 'pair']).learned_level
    actual = later.set_index(['date', 'pair']).size_code
    if not actual.eq(expected.reindex(actual.index)).all():
        raise ValueError('Contemporaneous frozen state does not reproduce saved sizing')
    later['leaf'] = later.leaf.astype(int)
    later['state_profit_residual'] = later.target-later.training_state_mean_target
    later['relative_daily_risk'] = later.daily_sd/later.training_state_median_daily_sd
    later.to_csv(out / 'state_later_proposals.csv', index=True, index_label='source_row')
    records = []
    for population, frame in [('training_windows', training), ('later_proposals', later)]:
        for period, own in [('all', frame), ('before_2022', frame.loc[frame.date.lt('2022-01-01')]),
                            ('2022_onward', frame.loc[frame.date.ge('2022-01-01')])]:
            for grouping in [['fit', 'leaf'], ['fit', 'leaf', 'pair']]:
                grouped = own.groupby(grouping)
                include = grouped.holding_days.transform('size').ge(4) & grouped.holding_days.transform('nunique').ge(2)
                centered_duration = own.holding_days-grouped.holding_days.transform('mean')
                for outcome in ['target', 'daily_sd', 'path_max_drawdown']:
                    centered_outcome = own[outcome]-grouped[outcome].transform('mean')
                    for method in ['pearson', 'spearman']:
                        records.append(dict(population=population, period=period, adjustment='+'.join(grouping),
                            outcome=outcome, method=method, rows=int(include.sum()),
                            cells=len(own.loc[include].groupby(grouping)),
                            correlation=centered_duration[include].corr(centered_outcome[include], method=method)))
    corr = pd.DataFrame(records)
    corr.to_csv(out / 'state_conditional_correlations.csv', index=False)
    band_summary = later.groupby(['maturity_band']).agg(examples=('target', 'size'),
        mean_target=('target', 'mean'), median_target=('target', 'median'),
        median_daily_sd=('daily_sd', 'median'), median_max_drawdown=('path_max_drawdown', 'median'),
        mean_state_profit_residual=('state_profit_residual', 'mean'),
        median_relative_daily_risk=('relative_daily_risk', 'median'),
        mean_chosen_size=('size_code', lambda s: s.mean()/4))
    band_summary.to_csv(out / 'state_later_maturity_summary.csv')
    cells = []
    for (fit, leaf), g in training.groupby(['fit', 'leaf']):
        shorter, longer = g.loc[g.entry_dte_mean.lt(28)], g.loc[g.entry_dte_mean.ge(35)]
        if len(shorter) >= 3 and len(longer) >= 3:
            cells.append(dict(fit=fit, leaf=leaf, short_count=len(shorter), long_count=len(longer),
                long_minus_short_mean_target=longer.target.mean()-shorter.target.mean(),
                long_to_short_median_daily_risk=longer.daily_sd.median()/shorter.daily_sd.median(),
                long_minus_short_median_drawdown=longer.path_max_drawdown.median()-shorter.path_max_drawdown.median()))
    cells = pd.DataFrame(cells)
    cells.to_csv(out / 'state_training_band_contrasts.csv', index=False)
    show = corr.loc[corr.period.eq('all') & corr.adjustment.eq('fit+leaf') & corr.method.eq('pearson')]
    lines = ['# Maturity pooling inside frozen P25 states', '',
        f'Checked all 608 saved proposals. {len(later)} have complete full-size standalone paths; 50 have no valid standalone entry and are excluded from outcome statistics. Each complete proposal is assigned using its original monthly tree, and every sizing code reproduces the archived policy. No tree, optimizer or funded portfolio was rerun.', '',
        '## Holding-duration associations after grouping by the same frozen state', '',
        '| Population | Outcome | Correlation | Rows in usable cells | Distinct monthly states |',
        '|---|---|---:|---:|---:|']
    lines += [f'| {r.population} | {r.outcome} | {r.correlation:.3f} | {r.rows} | {r.cells} |' for r in show.itertuples()]
    lines += ['', 'Each variable is centered within its own fit-month/leaf population. Cells need at least four rows and two distinct holding durations. CSVs also show within-state/pair centering and pre/post-2022 results. Training windows repeatedly use overlapping examples. Later proposals are after their tree fit, but the strategy configuration was retrospectively selected and this analysis was requested after observing performance; this is not an untouched holdout.', '',
        '## Complete later proposals by mean maturity at entry', '',
        '| Entry DTE | Proposals | Mean full-size profit label | Median daily P&L SD | Median daily risk / original training-state median | Mean chosen size |',
        '|---|---:|---:|---:|---:|---:|']
    lines += [f'| {band} | {int(r.examples)} | {r.mean_target:.2f} | {r.median_daily_sd:.2f} | {r.median_relative_daily_risk:.2f} | {r.mean_chosen_size:.1%} |' for band, r in band_summary.iterrows()]
    lines += ['', 'Profit and daily risk use the original fixed-450-vega label units, not percentage investment returns. Relative daily risk compares each subsequent path against its own original training state; the median across proposals is not a portfolio risk estimate. Proposal paths include entries rejected by funded risk caps, so these are signal/representation diagnostics, not funded returns or subgroup Sharpes.', '',
        '## Short versus long maturity inside training states', '',
        f'{len(cells)} monthly states have at least three short (<28D) and three long (>=35D) examples. Their median longer/shorter daily-risk ratio is {cells.long_to_short_median_daily_risk.median():.2f}; the 10th–90th percentile range is {cells.long_to_short_median_daily_risk.quantile(.1):.2f}–{cells.long_to_short_median_daily_risk.quantile(.9):.2f}. This variation checks whether maturity is uniformly interchangeable; it does not identify an optimal sizing change.', '',
        '## How to interpret this check', '',
        'Small conditional associations support maturity pooling as a working approximation; strong or persistent associations would motivate a targeted controlled experiment. A zero overall association does not establish uniformly homogeneous states. Risk/return differences can arise from market date, name, IV, overlapping trades and other inputs. Per-leaf samples are small, and many monthly states contain too few later proposals for a reliable maturity contrast.', '',
        'Reproduce with `python -m scripts.diagnose_p25_maturity --label-checks-only`. Outputs remain beside the earlier maturity diagnostics: state_conditional_correlations.csv, state_later_proposals.csv, state_later_maturity_summary.csv and state_training_band_contrasts.csv. Verified source hashes and resource use are in label_protocol.json.', '']
    (out / 'state_findings.md').write_text('\n'.join(lines), encoding='utf-8')
    print(show.to_string(index=False), flush=True)
    print(band_summary.to_string(), flush=True)


def plot_results(out, proposals, pairs, legs, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    chosen = pairs.loc[pairs.arm.eq('selected')]
    own = legs.loc[legs.arm.eq('selected')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].hist(proposals.entry_dte_mean, bins=np.arange(20, 48, 2), alpha=.5, label='608 proposals')
    axes[0].hist(chosen.entry_dte_mean, bins=np.arange(20, 48, 2), alpha=.7, label='207 executed pairs')
    axes[0].set(xlabel='Mean DTE at entry', ylabel='Pair count'); axes[0].legend()
    axes[1].scatter(own.entry_dte, own.gamma_1pct_per_vega, s=10, alpha=.4)
    axes[1].set(xlabel='Leg DTE at entry', ylabel='Gamma P&L for 1% spot move / entry vega')
    axes[2].scatter(own.entry_dte, own.theta_day_per_vega, s=10, alpha=.4)
    axes[2].set(xlabel='Leg DTE at entry', ylabel='Long-option daily theta / entry vega')
    fig.suptitle('Saved selected P25 entries: maturity and initial model exposures')
    fig.savefig(out / 'maturity_and_exposure.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bands = ['21-27', '28-34', '35-45']
    for ax, (period, first, last) in zip(axes, PERIODS[1:]):
        for j, arm in enumerate(['selected', 'fixed100']):
            g = summary.loc[summary.arm.eq(arm) & summary.period.eq(period) & summary.grouping.eq('maturity_band')].set_index('category')
            vals = g.calendar_net_before_shared_financing.reindex(bands, fill_value=0)/1000
            ax.bar(np.arange(3)+j*.35, vals, width=.35, label=arm)
        ax.set_xticks(np.arange(3)+.175, bands)
        ax.set(title=period, xlabel='Pair mean entry DTE band', ylabel='Calendar P&L before shared financing ($000)')
        ax.axhline(0, color='grey', lw=1); ax.legend()
    fig.suptitle('Attribution of existing portfolios; groups are not counterfactual strategies')
    fig.savefig(out / 'outcomes_by_maturity.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
