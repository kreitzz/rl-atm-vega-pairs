"""Add one employee-review feature to the frozen ATM65 procedure.

Cached licensed data only; single supervised worker, resumable monthly fits.
Baseline control must reproduce every saved RL action before evaluation.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.run_atm65_ablation import features, read_actions, write_json, NN, CURVE, FROZEN, PERIODS
from scripts.curve_trading.config import config, COMBOS
from scripts.curve_trading.policy import ATM, Replay, fit, predict, gate, memory_gates
from scripts.run_atm_cost_hurdles import make_units, reward_and_memory
from scripts.run_atm_nn_cost_hurdles import fit_fixed_nn, predict_fixed_nn
from scripts.replay_frozen_atm_commission import load_pair, cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_curve_trading import RUNOFF, screen_dates
from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import scores, audit
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.test_wrds_pair_signals import interval_ids, match_states, timestamp_columns
from scripts.test_wrds_alternative_pairs import numeric_ids
from scripts.retrain_atm_commission import check_fits

OUT = Path('artifacts/contracts/atm65_employee_sentiment')
REPORT = Path('reports/atm65_employee_sentiment')
LINKS = Path('artifacts/contracts/wrds_pair_signals')
REVIEWS = Path('artifacts/contracts/wrds_alternative_pairs')
ADDED = ['employee_leadership_gap', 'employee_leadership_available']
MODELS = ['rl_control', 'rl_sentiment', 'nn_sentiment']
CODES = [0, *range(1 + ATM * 8, 1 + ATM * 8 + 8)]


def add_sentiment(panel, first, second):
    """Require both names; neutral fill always accompanies an observed flag."""
    result = panel.copy()
    gap = first.reindex(panel.index) - second.reindex(panel.index)
    observed = np.isfinite(gap)
    result[ADDED[0]] = gap.where(observed, 0.).astype(float)
    result[ADDED[1]] = observed.astype(float)
    pd.testing.assert_frame_equal(result[panel.columns], panel)
    return result


def leg_states(dates, assets):
    o = timestamp_columns(pd.read_parquet(LINKS/'option_links.parquet'), ['sdate', 'edate'])
    o = o[o.score <= 1].copy(); o.edate = o.edate.fillna(pd.Timestamp('2026-01-01'))
    c = timestamp_columns(pd.read_parquet(LINKS/'company_links.parquet'), ['linkdt', 'linkenddt'])
    c.linkdt = c.linkdt.fillna(pd.Timestamp('1900-01-01'))
    c.linkenddt = c.linkenddt.fillna(pd.Timestamp('2026-01-01'))
    c.cik = c.cik.map(lambda x: str(x).strip().zfill(10))
    mapping = pd.read_parquet(REVIEWS/'revelio_mapping.parquet')
    mapping.cik = mapping.cik.map(lambda x: str(x).strip().zfill(10))
    mapping.rcid = mapping.rcid.map(int)
    states = pd.read_parquet(REVIEWS/'reviews_states.parquet')
    states.rcid = pd.Series(numeric_ids(states.rcid)).astype(np.int64)
    result = {}
    for sid in assets:
        perm = interval_ids(dates, o[o.secid.eq(sid)], None, 'sdate', 'edate', 'permno')
        cik = np.full(len(dates), None, object); rcid = cik.copy()
        for value in pd.Series(perm).dropna().unique():
            mask = perm == value
            cik[mask] = interval_ids(dates[mask], c[c.permno.eq(value)], None,
                                     'linkdt', 'linkenddt', 'cik')
        for value in pd.Series(cik).dropna().unique():
            hits = mapping[mapping.cik.eq(value)].rcid.unique()
            if len(hits) == 1: rcid[cik == value] = hits[0]
        result[sid] = match_states(dates, numeric_ids(rcid), states, 'rcid',
                                  ['review_leadership'], 60).review_leadership
    return result


def prepare(defs, cal):
    legs = leg_states(cal, sorted({sid for d in defs for sid in d['secids']}))
    rows = []
    for d in defs:
        source = NN/d['pair']/'fee_065'; folder = OUT/d['pair']; folder.mkdir(exist_ok=True)
        old = json.loads((CURVE/d['pair']/'atm_control/status.json').read_text())
        row = dict(pair=d['pair'], label=d['label'], state_rows=0, observed_rows=0,
                   observed_test_rows=0, observed_permitted_test_rows=0, retrain=False)
        if not old.get('structural_cash') and (source/'states.parquet').exists():
            panel = pd.read_parquet(source/'states.parquet')
            a, b = d['secids']; state = add_sentiment(panel, legs[a], legs[b])
            assert len(features(panel)) == 47
            state.to_parquet(folder/'states.parquet')
            test = ((panel.index >= '2018-01-01') & (panel.index <= cal[-1]-pd.Timedelta(days=RUNOFF))
                    & screen_dates(d, panel.index))
            with np.load(source/'gates.npz') as z:
                permitted = panel.eligible.to_numpy(bool) & z['available'] & z['gates'].any(axis=1)
            observed = state[ADDED[1]].eq(1).to_numpy()
            row.update(state_rows=len(panel), observed_rows=int(observed.sum()),
                       observed_test_rows=int((observed & test).sum()),
                       observed_permitted_test_rows=int((observed & test & permitted).sum()),
                       retrain=bool((observed & test & permitted).any()))
        rows.append(row)
    coverage = pd.DataFrame(rows); coverage.to_csv(REPORT/'coverage.csv', index=False)
    write_json(OUT/'prepared.json', dict(pairs=len(defs), retrained_pairs=int(coverage.retrain.sum()),
        original_features=47, added_features=ADDED, joint_reviews_required=True,
        available_lag='month-end plus 15 calendar days plus one business day',
        max_age_days=60, minimum_reviews_per_company_over_three_months=20))
    print(coverage[coverage.retrain][['label', 'observed_permitted_test_rows']].to_string(index=False), flush=True)
    print(f'{coverage.retrain.sum()}/{len(defs)} sleeves have reviews on permitted evaluation dates.', flush=True)


def train_pair(d, panel, daily, finish, gates, available, cal, replay, rates, folder):
    stop = cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted = ((panel.index >= '2018-01-01') & (panel.index <= stop) & screen_dates(d, panel.index))
    idx = panel.index[wanted]
    saved = read_actions(CURVE/d['pair']/'atm_control/actions.csv')
    saved_nn = read_actions(NN/d['pair']/'fee_065/actions.csv')
    assert idx.equals(saved.index) and idx.equals(saved_nn.index)
    actions = pd.DataFrame(0, index=idx, columns=MODELS, dtype=int)
    logs = []; checkpoint = folder/'month_checkpoint.json'
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text()); assert old['dates'] == [str(t) for t in idx]
        actions.iloc[:, :] = np.asarray(old['actions'], dtype=int); logs = old['fits']
    completed = {r['fit'] for r in logs}
    positions = cal.get_indexer(panel.index); completion = cal[finish]
    finite = np.isfinite(daily).all(axis=(1, 2)); original = features(panel)
    nn_fits = pd.read_csv(NN/d['pair']/'fee_065/fits.csv').set_index('fit')
    for month in pd.period_range('2018-01-01', stop, freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        test = (idx >= cutoff) & (idx < (month+1).start_time)
        if not test.any() or fitdate in completed: continue
        validation_start = cutoff-pd.DateOffset(years=1)
        use = ((panel.index >= validation_start-pd.DateOffset(years=5))
               & (completion < validation_start) & finite)
        recent = (panel.index >= cutoff-pd.DateOffset(years=5)) & (completion < cutoff) & finite
        vcal = cal[(cal >= validation_start) & (cal < cutoff)]
        val = (panel.index >= vcal[0]) & (panel.index <= vcal[-1]-pd.Timedelta(days=RUNOFF))
        opportunity = (panel.eligible.to_numpy(bool) & available[ATM] & gates[ATM].any(axis=1) & val).any()
        row = dict(fit=fitdate, training_rows=int(use.sum()), validation_start=str(vcal[0].date()),
                   validation_end=str(vcal[-1].date()), recent_rows=int(recent.sum()))
        dates = idx[test]
        for name in MODELS:
            columns = original if name == 'rl_control' else original+ADDED
            neural = name == 'nn_sentiment'
            fitter = fit_fixed_nn if neural else fit
            predictor = predict_fixed_nn if neural else predict
            row[name+'_status'] = 'insufficient_training' if use.sum() < 504 or len(vcal) < 200 else 'flat_no_validation_opportunities'
            if use.sum() < 504 or len(vcal) < 200 or not opportunity: continue
            labels = daily[use].copy(); labels[~panel.loc[use, 'eligible'].to_numpy(bool)] = 0
            model, details = fitter(panel.loc[use, columns], labels, positions[use], finish[use], CODES)
            row['last_training_outcome'] = str(completion[use].max().date())
            proposal = gate(predictor(model, panel.loc[val, columns]), panel.index[val], panel, gates, available)
            score = None; error = None
            try:
                nav, trades, _ = replay.run(proposal, vcal)
                mark_book(nav, rates, replay.cfg.capital)
                audit(nav, trades, replay.s, replay.cfg, d['secids'])
                score = scores(nav, replay.cfg.capital)['annualized_excess_sharpe']
            except (UnresolvedPosition, ValueError) as exc: error = str(exc)
            enabled = score is not None and score > 0 and recent.sum() >= 504
            row[name+'_status'] = 'enabled' if enabled else 'flat_validation_gate'
            row[name+'_validation_sharpe'] = score; row[name+'_validation_error'] = error
            row.update({name+'_proposal_'+k: v for k, v in details.items()})
            if enabled:
                own = daily[recent].copy(); own[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
                model, details = fitter(panel.loc[recent, columns], own, positions[recent], finish[recent], CODES)
                actions.loc[dates, name] = gate(predictor(model, panel.loc[dates, columns]), dates, panel, gates, available)
                row['last_recent_outcome'] = str(completion[recent].max().date())
                row.update({name+'_refit_'+k: v for k, v in details.items()})
        # Independently reconstruct enabled original NN refits; saved validation gates stay untouched.
        if nn_fits.loc[fitdate, 'status'] == 'enabled':
            own = daily[recent].copy(); own[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
            model, _ = fit_fixed_nn(panel.loc[recent, original], own, positions[recent], finish[recent], CODES)
            recovered = gate(predict_fixed_nn(model, panel.loc[dates, original]), dates, panel, gates, available)
            np.testing.assert_array_equal(recovered.to_numpy(), saved_nn.loc[dates].to_numpy())
        np.testing.assert_array_equal(actions.loc[dates, 'rl_control'].to_numpy(), saved.loc[dates].to_numpy(),
                                      err_msg=f'{d["label"]} {fitdate}: original RL was not reproduced')
        row['rl_actions_reproduced'] = True; row['nn_enabled_refits_reproduced'] = True
        logs.append(row)
        write_json(checkpoint, dict(dates=[str(t) for t in idx], actions=actions.to_numpy().tolist(), fits=logs))
        if month.month in [1, 7]: print(f'  {d["label"]} {month}: RL {row["rl_sentiment_status"]}; NN {row["nn_sentiment_status"]}', flush=True)
    np.testing.assert_array_equal(actions.rl_control.to_numpy(), saved.to_numpy())
    fits = pd.DataFrame(logs); check_fits(fits); fits.to_csv(folder/'fits.csv', index=False)
    for name in MODELS:
        dest = folder/name; dest.mkdir(exist_ok=True)
        a = actions[name].rename('code')
        pd.testing.assert_series_equal(a, gate(a, a.index, panel, gates, available), check_names=False)
        a.to_csv(dest/'actions.csv')
    return actions


def base_folder(pair, model, period):
    return FROZEN/pair/(period+'_003') if model.startswith('rl') else NN/pair/'fee_065'/period


def summary(defs, coverage, market, cfg):
    candidates = ['rl_baseline', 'rl_sentiment', 'nn_baseline', 'nn_sentiment']
    retrained = set(coverage.loc[coverage.retrain, 'pair'])
    all_rows = []; annual = []; pairs = []; unresolved = []
    def source(d, model, period):
        if model.endswith('sentiment') and d['pair'] in retrained: return OUT/d['pair']/model/period
        return base_folder(d['pair'], model, period)
    statuses = {(d['pair'], m, p): json.loads((source(d, m, p)/'status.json').read_text())
                for d in defs for m in candidates for p, _ in PERIODS}
    complete = [d for d in defs if all(statuses[d['pair'], m, p]['status'] == 'complete'
                                      for m in candidates for p, _ in PERIODS)]
    for d in defs:
        for m in candidates:
            for p, _ in PERIODS:
                s = statuses[d['pair'], m, p]
                if s['status'] != 'complete': unresolved.append(dict(pair=d['pair'], model=m, period=p, error=s.get('error')))
    write_json(REPORT/'accounting_coverage.json', dict(requested_pairs=len(defs), common_complete_pairs=len(complete), unresolved=unresolved))
    for universe in ['all', 'review_covered']:
        roster = complete if universe == 'all' else [d for d in complete if d['pair'] in retrained]
        if not roster: continue
        for m in candidates:
            for p, start in PERIODS:
                cal = market.calendar[market.calendar >= start]; cash = cash_book(cal, market.rates, cfg.capital)
                book = cash[['nav', 'cash_benchmark']]*len(roster); entries = active = 0
                for d in roster:
                    status = statuses[d['pair'], m, p]; folder = source(d, m, p)
                    nav = cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source', folder/'daily_nav.parquet'), columns=['nav', 'cash_benchmark'])
                    assert nav.index.equals(cal); book += nav[['nav', 'cash_benchmark']]-cash[['nav', 'cash_benchmark']]
                    entries += status['entries']; active += int(status['entries'] > 0)
                    pairs.append(dict(universe=universe, model=m, period=p, pair=d['pair'], label=d['label'], entries=status['entries'],
                        net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                book['return_'] = book.nav.pct_change().fillna(0)
                book['excess_return'] = book.return_-book.cash_benchmark.pct_change().fillna(0)
                book.to_csv(REPORT/f'{universe}_{m}_{p}_portfolio.csv')
                all_rows.append(dict(universe=universe, model=m, period=p, pairs=len(roster), entries=entries, active_pairs=active,
                    net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]), **scores(book, cfg.capital*len(roster))))
                for year, g in book.groupby(book.index.year):
                    sd = g.excess_return.std()
                    annual.append(dict(universe=universe, model=m, period=p, year=year,
                        return_percent=100*((1+g.return_).prod()-1),
                        annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd > 1e-12 else None))
    comparison = pd.DataFrame(all_rows); comparison.to_csv(REPORT/'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv', index=False); pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv', index=False)
    # The broad baseline must also match the saved performance, not merely actions.
    if len(complete) == len(defs):
        saved = pd.read_csv('reports/atm_nn_cost_hurdles/comparison.csv')
        for m, old in [('rl_baseline', 'rl_sized'), ('nn_baseline', 'nn_fixed')]:
            for p, _ in PERIODS:
                got = comparison.query('universe == "all" and model == @m and period == @p').iloc[0]
                prior = saved[(saved.model.eq(old)) & (saved.period.eq(p)) & saved.training_hurdle.eq(.65)].iloc[0]
                np.testing.assert_allclose(got.annualized_excess_sharpe, prior.annualized_excess_sharpe, atol=1e-12)
                assert got.entries == prior.entries
    lines = ['# Employee leadership sentiment on the ATM65 checkpoint', '',
        'One added signal: the first company’s trailing three-month employee leadership rating minus the second’s, plus an availability flag. Missing values are filled with zero and flagged. At least 20 leadership reviews per company; monthly publication proxy is month-end plus 15 calendar days plus one business day, with 60-day maximum carry. No direction or threshold was chosen from the test returns.', '',
        f'{len(defs)} original matched sleeves retained; {len(retrained)} have joint review observations on permitted test dates and receive new fits. Remaining sleeves reuse their own original RL or NN policy. Covered sleeves are refitted for the whole evaluation period, including missing-review dates, so this tests both the score and its availability flag. Coverage is determined by inputs and existing eligibility, never realized returns.', '',
        'All 47 original inputs remain unchanged. Monthly prior-year validation and recent-five-year refits, strict completed-outcome purging, original eight-leaf Sharpe policy search / fixed-size NN, 25/50/75/100% RL ladder, beta/correlation and profitability gates, actual ATM30 call contracts, daily delta hedging and hold to expiry are unchanged. Validation/training commission is $0.65; evaluation commission is $0.03, with the original spread, hedge, financing and risk limits. Original RL monthly actions and enabled NN refits are reconstructed and checked before candidate evaluation.', '',
        '| Universe | Model | Full Sharpe | Recent Sharpe | Full entries |', '|---|---|---:|---:|---:|']
    for universe in ['all', 'review_covered']:
        for m in candidates:
            a = comparison[(comparison.universe.eq(universe)) & comparison.model.eq(m) & comparison.period.eq('full')]
            b = comparison[(comparison.universe.eq(universe)) & comparison.model.eq(m) & comparison.period.eq('recent')]
            if len(a): lines.append(f'| {universe} | {m} | {a.iloc[0].annualized_excess_sharpe:.3f} | {b.iloc[0].annualized_excess_sharpe:.3f} | {int(a.iloc[0].entries)} |')
    lines += ['', 'Full: 2018–August 2025. Recent: 2022–August 2025, starting flat. Sharpe uses daily returns above cash and √252. Any unresolved contract paths are explicitly excluded from every row on a common roster.', '',
        'This is an exploratory ablation on previously inspected history, following a sentiment correlation screen on the same period. Current Revelio company mappings and historical review edits have not been validated against archived vendor vintages. The date lag prevents use of later review months but cannot remove those vintage risks. No correlation-study payoff labels were used for strategy training: labels were rebuilt from the saved contract selections at the checkpoint cost settings. The checkpoint is preserved.', '',
        '[Comparison](atm65_employee_sentiment/comparison.csv) · [Annual](atm65_employee_sentiment/annual.csv) · [Pairs](atm65_employee_sentiment/pairs.csv) · [Review coverage](atm65_employee_sentiment/coverage.csv) · [Accounting coverage](atm65_employee_sentiment/accounting_coverage.json)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['universe', 'model', 'period', 'pairs', 'entries', 'annualized_excess_sharpe']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--limit', type=int); parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((NN/'inputs.json').read_text()); roster = pd.read_csv('reports/atm_nn_cost_hurdles/matched_roster.csv')
    included = set(roster.loc[roster.included, 'pair']); defs = [d for d in inputs['definitions'] if d['pair'] in included]
    actual = replace(config(), option_commission=.03); training = replace(actual, option_commission=.65)
    assert asdict(actual) == inputs['execution_settings']
    sources = [Path(__file__), NN/'inputs.json', Path('config/atm30_fee065_checkpoint.json'),
        Path('reports/atm_nn_cost_hurdles/matched_roster.csv'), REVIEWS/'reviews_states.parquet', REVIEWS/'review_months.parquet',
        REVIEWS/'revelio_mapping.parquet', LINKS/'option_links.parquet', LINKS/'company_links.parquet',
        Path('scripts/curve_trading/policy.py'), Path('scripts/curve_trading/unit_paths.py'),
        Path('scripts/run_contract_walkforward.py'), Path('scripts/option_flex/contract_engine.py')]
    for d in defs:
        sources.extend(p for p in [NN/d['pair']/'fee_065/states.parquet', NN/d['pair']/'fee_065/gates.npz',
            NN/d['pair']/'fee_065/actions.csv', CURVE/d['pair']/'atm_control/actions.csv'] if p.exists())
    signature = dict(definitions=defs, actual=asdict(actual), training=asdict(training), added_features=ADDED,
                     code_and_sources={str(p): digest(p) for p in sources})
    if (OUT/'inputs.json').exists(): assert json.loads((OUT/'inputs.json').read_text()) == signature, 'Inputs changed'
    else: write_json(OUT/'inputs.json', signature)
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    if not (OUT/'prepared.json').exists(): prepare(defs, cal)
    if args.prepare: return
    coverage = pd.read_csv(REPORT/'coverage.csv'); selected = set(coverage.loc[coverage.retrain, 'pair'])
    market = Market(cal); started = time.monotonic(); newly = 0
    if not args.report_only:
        for n, d in enumerate([x for x in defs if x['pair'] in selected], 1):
            folder = OUT/d['pair']
            if all((folder/m/p/'status.json').exists() for m in ['rl_sentiment', 'nn_sentiment'] for p, _ in PERIODS): continue
            pair_started = time.monotonic(); print(f'{n}/{len(selected)} {d["label"]}: leadership signal', flush=True)
            panel = pd.read_parquet(folder/'states.parquet'); source = NN/d['pair']/'fee_065'
            q, s, selections, divs = load_pair(market, d['secids'])
            with np.load(source/'gates.npz') as z:
                gates = np.zeros((len(COMBOS), len(panel), 2), bool); gates[ATM] = z['gates']
                available = np.zeros((len(COMBOS), len(panel)), bool); available[ATM] = z['available']; saved_ends = z['ends'].copy()
            units = make_units(panel, q, s, selections, divs, cal, training)
            labels, finish, vegas, statuses, rewards, _ = reward_and_memory(units, training)
            np.testing.assert_array_equal(finish, saved_ends)
            np.testing.assert_array_equal(gates[ATM], memory_gates(panel.index, cal[finish], rewards, vegas[:, [0, -1]] > 0))
            daily = np.zeros((len(panel), 9, labels.shape[-1]), dtype=np.float32); daily[:, 1:] = labels
            del units, labels; gc.collect()
            replay = Replay(q, s, selections, d['secids'], divs, training)
            actions = train_pair(d, panel, daily, finish, gates, available, cal, replay, market.rates, folder)
            replay.cfg = actual
            for m in ['rl_sentiment', 'nn_sentiment']:
                for p, start in PERIODS:
                    dest = folder/m/p; dest.mkdir(parents=True, exist_ok=True)
                    if (dest/'status.json').exists(): continue
                    a = actions.loc[actions.index >= start, m]
                    if not a.ne(0).any(): result = dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True)
                    else:
                        try:
                            nav, trades, decisions = replay.run(a, cal[cal >= start]); mark_book(nav, market.rates, actual.capital)
                            result = save_result(dest, nav, trades, decisions, actual, s, d)
                        except UnresolvedPosition as exc: result = dict(status='incomplete', pair=d['pair'], label=d['label'], error=str(exc))
                    result.update(training_hurdle=.65, execution_commission=.03); write_json(dest/'status.json', result)
            newly += 1
            write_json(OUT/'progress.json', dict(completed_retrained_pairs=n, retrained_pairs=len(selected), last_pair=d['label'],
                last_pair_seconds=time.monotonic()-pair_started, elapsed_seconds=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: completed in {time.monotonic()-pair_started:.1f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB', flush=True)
            del panel, q, s, selections, divs, replay, daily, actions; gc.collect()
            if args.limit and newly >= args.limit: return
    summary(defs, coverage, market, actual)
    write_json(OUT/'task_status.json', dict(status='complete', pairs=len(defs), retrained_pairs=len(selected),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__': main()
