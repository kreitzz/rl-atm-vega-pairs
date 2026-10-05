"""Report broad replay coverage and a clearly labeled common-complete diagnostic."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_broad_contract_replay import MODELS
from scripts.run_selected_pair_backtest import scores

ROOT = Path('artifacts/contracts/broad_replay')
REPORT = Path('reports/broad_contract_replay')
LABELS = {'rl_reference': 'RL sizing, no memory', 'rl_sized': 'RL sizing + memory',
          'nn_fixed': 'NN fixed size + memory'}


def main():
    manifest = json.loads((REPORT/'manifest.json').read_text())
    status = pd.read_csv(REPORT/'status.csv')
    assert len(status) == manifest['pairs']*len(MODELS)
    assert not status.duplicated(['pair', 'model']).any()
    failed = status[status.status.ne('complete')].copy()
    failed['failure_type'] = failed.error.str.replace(r'^\d{4}-\d{2}-\d{2}: ', '', regex=True).str.replace(r' \d+$', '', regex=True)
    failed.to_csv(REPORT/'unresolved.csv', index=False)
    definitions = json.loads((REPORT/'universe.json').read_text())
    excluded = set(failed.pair)
    membership = pd.DataFrame([dict(pair=p['pair'], label=p['label'], group=p['group'],
                                    common_complete=p['pair'] not in excluded)
                               for p in definitions])
    membership.to_csv(REPORT/'diagnostic_membership.csv', index=False)
    summary, annual, breadth = [], [], []
    for group in ['broad', 'legacy_extra']:
        selected = membership[membership.group.eq(group)&membership.common_complete]
        for model in MODELS:
            members = status[status.pair.isin(selected.pair)&status.model.eq(model)]
            if members.empty:
                continue
            book = None
            costs = dict(option_execution_cost=0., option_model_penalty=0., hedge_cost=0., stock_borrow_cost=0.)
            for row in members.itertuples():
                nav = pd.read_parquet(ROOT/row.pair/model/'daily_nav.parquet')
                if book is None:
                    book = nav[['nav', 'cash_benchmark']].copy()
                else:
                    assert book.index.equals(nav.index)
                    book += nav[['nav', 'cash_benchmark']]
                for field in costs:
                    costs[field] += nav[field].sum()
            capital = len(members)*manifest['settings']['capital']
            book['return_'] = book.nav.pct_change().fillna(0)
            book['excess_return'] = book.return_-book.cash_benchmark.pct_change().fillna(0)
            years = (book.index[-1]-book.index[0]).days/365.25
            row = dict(group=group, model=model, pairs=len(members),
                       **scores(book, capital), cagr=(book.nav.iloc[-1]/capital)**(1/years)-1,
                       cash_return=book.cash_benchmark.iloc[-1]/capital-1,
                       **{k:v/capital for k,v in costs.items()})
            summary.append(row)
            book.to_csv(ROOT/f'diagnostic_{group}_{model}.csv')
            for year, dates in book.groupby(book.index.year):
                prior = capital if dates.index[0]==book.index[0] else book.loc[:dates.index[0]].nav.iloc[-2]
                excess = dates.excess_return
                annual.append(dict(group=group, model=model, year=year, total_return=dates.nav.iloc[-1]/prior-1,
                                   annualized_excess_sharpe=np.sqrt(252)*excess.mean()/excess.std() if excess.std()>1e-12 else None))
            traded = members[members.entries.gt(0)]
            breadth.append(dict(group=group, model=model, common_complete_pairs=len(members),
                                traded_pairs=len(traded), cash_only_pairs=int(members.entries.eq(0).sum()),
                                profitable_over_cash=int(traded.excess_return.gt(0).sum()),
                                unprofitable_over_cash=int(traded.excess_return.lt(0).sum())))
    table = pd.DataFrame(summary)
    table.to_csv(REPORT/'diagnostic_portfolio.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'diagnostic_annual.csv', index=False)
    pd.DataFrame(breadth).to_csv(REPORT/'diagnostic_breadth.csv', index=False)
    lines = ['# All-pair actual-contract replay', '',
             f"Attempted all **{manifest['pairs']} pairs × {len(MODELS)} policies** from {manifest['start']} through {manifest['end']}: 350 broad-screen pairs and eight additional legacy pairs. **{len(excluded)} pairs have at least one unresolved policy run.** Full-universe portfolio statistics remain unavailable; failed runs are not replaced with cash.", '',
             '## Common-complete diagnostic only', '',
             'The following basket excludes every pair with a failure in any of the three models, using the same pair roster across models. This is an ex-post data-complete subset and can have selection bias; it is not the requested full-universe result. Inactive pairs and pairs without a fitted historical policy remain in cash.', '',
             '| Basket | Policy | Pairs | Total return | CAGR | Annualized excess Sharpe | Max drawdown |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples():
        label = LABELS[r.model] if r.group=='broad' else r.model+' (legacy features)'
        lines.append(f'| {r.group} | {label} | {r.pairs} | {r.total_return:.2%} | {r.cagr:.2%} | {r.annualized_excess_sharpe:.3f} | {r.max_drawdown:.2%} |')
    lines += ['', 'Returns include interest on uninvested cash. Sharpe uses daily returns in excess of the cash rate, annualized with √252. The 2025 annual table is year-to-date.', '',
              '## Unresolved runs', '', '| Failure | Pair-policy runs |', '|---|---:|']
    for reason, count in failed.failure_type.value_counts().items():
        lines.append(f'| {reason} | {count} |')
    lines += ['', 'The simulator stops on these cases rather than inventing held-position prices or liquidating retroactively. These counts identify data/lifecycle blockers, not trading losses. Three missing WPX/RCL quotes on March 13, 2020 were checked against the raw downloaded partitions and are absent there as well ([check](broad_contract_replay/raw_missing_quote_check.csv)).', '',
              '## Method and limits', '',
              '- Replays saved out-of-sample proxy-trained actions with their original historical screen membership. No new monthly actual-contract retraining or beta eligibility filter is applied in this experiment.',
              '- Closest eligible ATM call to 30 days, selected from prior-session information. Initial dollar-vega matching, daily stock delta hedge, integer contracts and overlapping cohorts.',
              '- Execution at midpoint plus **0.10 volatility point per option fill**, plus option commissions, stock execution, financing and stock borrow. The existing liquidity filters and NAV-scaled exposure, reserve and stress limits remain active.',
              '- Fixed equal $1 million accounts per unique pair. No capital recycling from inactive pairs; no cross-pair hedge netting. The original proxy portfolio used changing fold rosters, so its allocation and payoff Sharpe are not directly comparable.',
              '- Broad policies: RL sizing without memory, RL sizing with profitability memory, and fixed-size NN with memory. Legacy extras use their original relative-value-feature policies; their feature treatment differs, so they are reported separately.',
              '- History begins in 2018 because these are the saved broad experiment’s out-of-sample signals, despite earlier contract data being available.',
              f"- Peak replay memory: {manifest['peak_rss_bytes']/1024**3:.2f} GiB. Each completed sleeve passed option/stock cash-ledger and ending-position reconciliation.", '',
              '[Every pair/policy result](broad_contract_replay/status.csv) · [Unresolved cases](broad_contract_replay/unresolved.csv) · [Diagnostic membership](broad_contract_replay/diagnostic_membership.csv) · [Annual diagnostic results](broad_contract_replay/diagnostic_annual.csv) · [Breadth](broad_contract_replay/diagnostic_breadth.csv)', '',
              'Reproduce using `python -m scripts.cache_broad_contracts`, `python -m scripts.run_broad_contract_replay`, then `python -m scripts.report_broad_contract_replay`. Shared data and detailed ledgers stay in ignored local directories.']
    Path('reports/broad_contract_replay.md').write_text('\n'.join(lines)+'\n')
    print(table.to_string(index=False))
    print(pd.DataFrame(breadth).to_string(index=False))


if __name__=='__main__':
    main()
