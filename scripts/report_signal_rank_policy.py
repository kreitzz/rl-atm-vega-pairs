"""Audit every declared variant and compare on one common complete roster."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.run_signal_rank_policy import (OUTPUT, REPORT, BASE, FLEX, VARIANTS, FROZEN, definitions,
    settings, eligibility, apply_gate, ACTIONS, FLAT)
from scripts.run_contract_walkforward import digest

LABELS = {
    'reference_calls': 'Original calls-only reference',
    'cc_matched_rows': 'Prior calls-only, matched training rows',
    'full_no_memory_gate': 'Full state, profitability gate removed',
    'rank_with_memory_gate': 'IV spread/reversion ranks, original profitability gate',
    'rank_no_memory_gate': 'IV spread/reversion ranks, no profitability gate',
    'rank_term_no_memory_gate': 'Spread/reversion + term-slope ranks, no profitability gate',
}


def sharpe(e):
    return np.sqrt(252)*e.mean()/e.std() if e.std() > 1e-12 else np.nan


def main():
    inputs = json.loads((OUTPUT/'inputs.json').read_text()); cfg = settings(); defs = definitions()
    for p, h in inputs['code'].items(): assert digest(p) == h, p
    for p, h in inputs['sources'].items(): assert digest(p) == h, p
    assert digest(FLEX/'inputs.json') == inputs['previous_inputs']
    assert digest('config/current_contract_reference.json') == inputs['reference']
    previous = json.loads((FLEX/'inputs.json').read_text())
    for group in ['code', 'reference_files']:
        for p, h in previous[group].items(): assert digest(p) == h, p
    assert digest('data/call_put/manifest.json') == previous['put_manifest']
    assert digest('data/contract_history_shared/manifest.json') == previous['call_cache']
    records = []; audits = []; mix = []; failed = set()
    for d in defs:
        panel = None
        for name in FROZEN+list(VARIANTS):
            folder = OUTPUT/d['pair']/name; row = json.loads((folder/'status.json').read_text()); records.append(row)
            if row['status'] != 'complete': failed.add(d['pair']); continue
            nav = pd.read_parquet(row.get('nav_source', folder/'daily_nav.parquet'))
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1]) < 1e-7
            row['net_excess_pnl'] = nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
            if row.get('structural_cash'):
                np.testing.assert_allclose(nav.nav, nav.cash_benchmark, rtol=0, atol=1e-7)
            else:
                a = pd.read_csv(folder/'actions.csv', index_col=0, parse_dates=[0]).action_code
                if name in VARIANTS:
                    assert np.isin(a, VARIANTS[name]['allowed']).all()
                    if panel is None: panel = pd.read_parquet(FLEX/d['pair']/'states.parquet')
                    pd.testing.assert_series_equal(a, apply_gate(a, eligibility(panel, VARIANTS[name])))
                    try: f = pd.read_csv(folder/'fits.csv')
                    except pd.errors.EmptyDataError: f = pd.DataFrame()
                    for end, start in [('last_training_outcome', 'validation_start'), ('last_recent_outcome', 'fit')]:
                        if end in f:
                            use = f[end].notna()
                            assert (pd.to_datetime(f.loc[use, end]) < pd.to_datetime(f.loc[use, start])).all()
                else:
                    combo = name.removeprefix('frozen_')
                    assert all(ACTIONS[k][0] == combo or k == FLAT for k in a)
                    try: old = pd.read_csv(BASE/d['pair']/'rl_sized/decisions.csv')
                    except pd.errors.EmptyDataError: old = pd.DataFrame(columns=['status', 'signal_date'])
                    entered_dates = pd.DatetimeIndex(old.loc[old.status.eq('entered'), 'signal_date'])
                    assert a[a.ne(FLAT)].index.isin(entered_dates).all()
                if name in ['cc_original', 'frozen_CC']: assert row['reference_control'] == 'identical'
                try: dec = pd.read_csv(folder/'decisions.csv')
                except pd.errors.EmptyDataError: dec = pd.DataFrame(columns=['status', 'signal_date'])
                entered = dec[dec.status.eq('entered')]
                if len(entered):
                    assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a', 'vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
                    t = pd.read_csv(folder/'trades.csv'); assert t.groupby(['cohort', 'optionid']).quantity.sum().eq(0).all()
                    assert set(t.reason).issubset({'entry', 'split_adjustment', 'expiry_settlement'})
                    settle = t[t.reason.eq('expiry_settlement')]
                    assert settle.commission.eq(0).all() and settle.execution_penalty.eq(0).all()
                    for e in entered.itertuples():
                        combo = ACTIONS[int(a.loc[pd.Timestamp(e.signal_date)])][0]
                        assert combo in ['CC', 'PP']
                        entry = t[t.reason.eq('entry') & t.cohort.eq(e.signal_date)].set_index('secid')
                        assert ''.join(entry.loc[sid, 'cp_flag'] for sid in d['secids']) == combo
                        mix.append(dict(pair=d['pair'], variant=name, combination=combo, signal_date=e.signal_date))
            audits.append(dict(pair=d['pair'], variant=name, checks='passed', control=row.get('reference_control', 'cash' if row.get('structural_cash') else 'n/a')))
    status = pd.DataFrame(records); status.to_csv(REPORT/'pair_results.csv', index=False)
    pd.DataFrame(audits).to_csv(REPORT/'audit.csv', index=False)
    pd.DataFrame(mix, columns=['pair', 'variant', 'combination', 'signal_date']).to_csv(REPORT/'entry_types.csv', index=False)
    status[status.status.ne('complete')].to_csv(REPORT/'incomplete.csv', index=False)
    matched = [d for d in defs if d['pair'] not in failed]
    summary = []; annual = []; components = []; pair_rows = []
    for name in LABELS:
        book = None; entries = 0; active = 0
        for d in matched:
            if name == 'reference_calls':
                folder = BASE/d['pair']/'rl_sized'; path = folder/'daily_nav.parquet'
                row = next(r for r in json.loads((BASE/d['pair']/'status.json').read_text()) if r['model'] == 'rl_sized')
            elif name == 'cc_matched_rows':
                folder = Path('artifacts/contracts/option_flex_ablation')/d['pair']/name
                row = json.loads((folder/'status.json').read_text())
                assert row['status'] == 'complete'
                path = row.get('nav_source', folder/'daily_nav.parquet')
            else:
                folder = OUTPUT/d['pair']/name; row = next(r for r in records if r['pair'] == d['pair'] and r['variant'] == name)
                path = row.get('nav_source', folder/'daily_nav.parquet')
            nav = pd.read_parquet(path); b = nav[['nav', 'cash_benchmark']]
            if book is None: book = b.copy()
            else:
                assert book.index.equals(b.index); book += b
            entries += row['entries']; active += int(row['entries'] > 0)
            excess_pnl = nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]
            pair_rows.append(dict(variant=name, pair=d['pair'], label=d['label'], entries=row['entries'], net_excess_pnl=excess_pnl))
            c = dict(variant=name, pair=d['pair'], gross_option_pnl=row.get('option_pnl', 0.)+nav.option_execution_cost.sum(),
                gross_hedge_pnl=row.get('hedge_pnl', 0.)+nav.hedge_cost.sum(), option_cost=-nav.option_execution_cost.sum(),
                hedge_cost=-nav.hedge_cost.sum(), stock_borrow=-nav.stock_borrow_cost.sum(), dividends=nav.dividend_cash.sum(),
                financing_above_cash=nav.interest.sum()-(nav.cash_benchmark.iloc[-1]-nav.cash_benchmark.iloc[0]))
            np.testing.assert_allclose(sum(v for k, v in c.items() if k not in ['variant', 'pair']), excess_pnl, rtol=0, atol=1e-5)
            components.append(c)
        r = book.nav.pct_change().fillna(0); e = r-book.cash_benchmark.pct_change().fillna(0)
        pnl = book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
        book.assign(excess_return=e, excess_pnl=pnl).to_csv(REPORT/f'{name}_portfolio.csv')
        summary.append(dict(variant=name, label=LABELS[name], pairs=len(matched), entries=entries, active_pairs=active,
                            annualized_excess_sharpe=sharpe(e.iloc[1:]), net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]))
        for year, v in e.groupby(e.index.year):
            annual.append(dict(variant=name, year=year, annualized_excess_sharpe=sharpe(v), net_excess_pnl=pnl.loc[v.index].sum()))
    result = pd.DataFrame(summary); result.to_csv(REPORT/'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT/'matched_pair_results.csv', index=False)
    pd.DataFrame(components).drop(columns='pair').groupby('variant').sum().to_csv(REPORT/'pnl_components.csv')
    lines = ['# Signal-rank policy experiments', '',
        f'{len(matched)}/{len(defs)} pairs; January 2018–August 2025. Calls/calls only, unchanged costs, sizing ladder, beta/correlation filters, monthly walk-forward schedule, validation gate and risk limits. All four new variants share the prior calls-only matched-label training rows and completion cutoffs.', '',
        '| Variant | Annualized excess Sharpe | P&L above cash | Entries | Active pairs |',
        '|---|---:|---:|---:|---:|']
    for r in result.itertuples():
        lines.append(f'| {r.label} | {r.annualized_excess_sharpe:.3f} | ${r.net_excess_pnl:,.0f} | {r.entries} | {r.active_pairs} |')
    lines += ['',
        'Pairs with an incomplete book in any new variant are excluded from every aggregate comparison, including both baselines. Their errors remain in incomplete.csv; no failed book is treated as flat or assigned a payoff. This is a common-complete diagnostic comparison, not a claim of a complete full-universe backtest.', '',
        'Predeclared follow-up to the signal/profitability study: (1) remove the hard profitability gate while retaining the full state, including memory inputs; (2) replace the state with trailing IV-spread and spread-reversion ranks while retaining the original gate; (3) combine the rank state with removal of that gate; (4) add the term-slope difference rank to (3). No hard-coded reversal of memory or direction based on full-sample outcomes. The policy learns direction, flat decisions and the existing sizing ladder from past completed net contract outcomes.', '',
        'Ranks compare the current signal to at most 252 observed signal rows including itself, with 63 observations required. Missing ranks use 0.5 with an availability flag. Term slope is masked when unavailable. These transformations are prefix-invariant and use no future observations. The original tree/CEM Sharpe objective, seed and search budget are unchanged. Removing the profitability gate does not remove beta/correlation, contract availability, positive validation Sharpe, or portfolio risk checks.', '',
        'All four new variants were specified before viewing their backtest results. However, feature selection was motivated by correlations on this same 2018–2025 history: these are adaptive research comparisons, not an untouched holdout or proof of improvement. Overlapping outcomes and shared names create dependence. Annualized excess Sharpe uses daily returns above the cash benchmark and includes all cash sleeves. Validation still checks the pre-refit proposal rather than the exact deployed weights. Initial vega matching and daily delta hedging do not prevent later vega drift; each leg is held to its own expiry. American early assignment remains unmodeled.', '',
        '[Annual results](signal_rank_policy/annual.csv) · [Pair results](signal_rank_policy/matched_pair_results.csv) · [P&L components](signal_rank_policy/pnl_components.csv) · [Audit](signal_rank_policy/audit.csv) · [Incomplete books](signal_rank_policy/incomplete.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)); print(result.to_string(index=False), flush=True)


if __name__ == '__main__': main()
