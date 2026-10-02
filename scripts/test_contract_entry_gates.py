"""Predefined as-of entry gates on frozen contract-trained policies.

No retraining or reward changes. Every changed policy is replayed through the
full book, rather than subtracting rejected trades from a saved NAV path.
"""
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, PreparedMarket, simulate, UnresolvedPosition
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_selected_pair_backtest import audit
from scripts.select_atm_contracts import load_calendar
from scripts.run_contract_walkforward import digest

SOURCE = Path('artifacts/contracts/aligned_learning')
OUTPUT = Path('artifacts/contracts/entry_gates')
REPORT = Path('reports/contract_entry_gates')
VARIANTS = ['memory_year', 'memory_both', 'commission_budget']
MODELS = ['rl_sized', 'nn_fixed']


def gate_actions(actions, states, variant):
    s = states.reindex(actions.index)
    side = actions.gt(0)
    if variant in ['memory_year', 'memory_both']:
        profit = s.profit_long_365d.where(side, s.profit_short_365d)
        count = s.count_long_365d.where(side, s.count_short_365d)
        keep = profit.gt(0) & count.ge(63)
        if variant == 'memory_both':
            keep &= s.profit_long_91d.where(side, s.profit_short_91d).gt(0)
            keep &= s.count_long_91d.where(side, s.count_short_91d).ge(21)
    elif variant == 'commission_budget':
        # Two legs, entry and exit: 4 * 0.10 vol-point option penalties.
        # Require estimated round-trip commissions no larger than that budget.
        keep = s.commission_vol_roundtrip.le(.4) & s.commission_vol_roundtrip.ge(0)
    else:
        raise ValueError(variant)
    return actions.where(keep.fillna(False), 0.)


def metrics(book):
    ret = book.nav.pct_change().fillna(0)
    cash = book.cash_benchmark.pct_change().fillna(0)
    rows = []
    for period, mask in [('development_2018_2021', book.index.year < 2022),
                         ('later_2022_2025', book.index.year >= 2022),
                         ('full', np.ones(len(book), dtype=bool))]:
        r, c = ret[mask], cash[mask]
        excess = r-c
        total, benchmark = (1+r).prod()-1, (1+c).prod()-1
        rows.append(dict(period=period, total_return=total, cash_return=benchmark,
                         excess_return=total-benchmark,
                         annualized_excess_sharpe=float(np.sqrt(252)*excess.mean()/excess.std())
                         if excess.std()>1e-12 else None))
    return rows


def report(definitions, records, manifest):
    status = pd.DataFrame(records)
    baseline = pd.read_csv('reports/aligned_contract_learning/status.csv')
    failed = set(status.loc[status.status.ne('complete'), 'pair']) | set(baseline.loc[baseline.status.ne('complete'), 'pair'])
    status.to_csv(REPORT/'status.csv', index=False)
    status[status.status.ne('complete')].to_csv(REPORT/'unresolved.csv', index=False)
    results, yearly = [], []
    for group in ['broad', 'legacy_extra']:
        members = [d['pair'] for d in definitions if d['group']==group and d['pair'] not in failed]
        for model in MODELS:
            for variant in ['baseline']+VARIANTS:
                book = None
                for pair in members:
                    path = SOURCE/pair/model if variant=='baseline' else OUTPUT/pair/f'{model}_{variant}'
                    nav = pd.read_parquet(path/'daily_nav.parquet', columns=['nav','cash_benchmark'])
                    if book is None: book=nav.copy()
                    else:
                        assert book.index.equals(nav.index)
                        book += nav
                if book is None: continue
                book.to_csv(REPORT/f'{group}_{model}_{variant}_portfolio.csv')
                for row in metrics(book):
                    results.append(dict(group=group, model=model, variant=variant, pairs=len(members), **row))
                excess = book.nav.pct_change()-book.cash_benchmark.pct_change()
                for year, values in excess.groupby(excess.index.year):
                    yearly.append(dict(group=group, model=model, variant=variant, year=year,
                        annualized_excess_sharpe=np.sqrt(252)*values.mean()/values.std() if values.std()>1e-12 else None))
    result = pd.DataFrame(results)
    result.to_csv(REPORT/'comparison.csv', index=False)
    pd.DataFrame(yearly).to_csv(REPORT/'annual.csv', index=False)
    selected = []
    for model in MODELS:
        dev = result[result.group.eq('broad') & result.model.eq(model) & result.period.eq('development_2018_2021')]
        valid = dev.dropna(subset=['annualized_excess_sharpe'])
        winner = valid.sort_values('annualized_excess_sharpe', ascending=False, kind='stable').iloc[0]
        selected.append(dict(model=model, variant=winner.variant,
                             development_sharpe=winner.annualized_excess_sharpe,
                             beats_cash_in_development=bool(winner.excess_return>0)))
    (REPORT/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    lines = ['# Contract entry gate experiment', '',
        'Frozen monthly contract-trained proposals, unchanged sizing ladder, costs and risk limits. Changed entry paths are fully resimulated. All 358 pairs attempted; comparisons retain the same complete pairs across every policy, including cash-only sleeves. Unresolved pairs are excluded explicitly, which can introduce selection bias.', '',
        '- `memory_year`: chosen direction has positive mean completed net shadow profit over 365 calendar days, with at least 63 completed entered cohorts.',
        '- `memory_both`: also positive over 91 days, with at least 21 cohorts.',
        '- `commission_budget`: estimated round-trip commissions at signal close are at most 0.40 vol points (the existing four option-fill penalties). Actual commissions are still charged.',
        '- The baseline is eligible for selection. Choose separately for RL and NN by broad-basket 2018–2021 annualized daily excess Sharpe; assess 2022 onward separately. These periods have been inspected previously: this is exploratory, not an untouched test.',
        '- Gates use only stored as-of signal states; profitability memory excludes outcomes ending on or after the signal date. Shadow cohorts overlap, so observation counts are not independent samples. Gates only suppress new entries; existing positions follow their original lifecycle.', '',
        '| Model | Gate | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in result[result.group.eq('broad')].itertuples():
        lines.append(f'| {r.model} | {r.variant} | {r.period} | {r.pairs} | {r.total_return:.3%} | {r.excess_return:.3%} | {r.annualized_excess_sharpe:.3f} |')
    lines += ['', 'Development-selected policies: '+', '.join(f"{s['model']}: {s['variant']}" for s in selected)+'.', '',
              f"Unresolved pairs in the comparison union: {len(failed)}. Peak RSS: {manifest['peak_rss_bytes']/1024**3:.2f} GiB.", '',
              '[Full results](contract_entry_gates/comparison.csv) · [Annual Sharpe](contract_entry_gates/annual.csv) · [Pair results](contract_entry_gates/status.csv) · [Unresolved](contract_entry_gates/unresolved.csv)', '']
    Path('reports/contract_entry_gates.md').write_text('\n'.join(lines))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    definitions = json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    previous = json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())
    cfg = Settings(**previous['settings'])
    fingerprint = dict(source_manifest=digest('reports/aligned_contract_learning/manifest.json'),
        source_inputs={str(p):digest(p) for p in SOURCE.glob('worker_*/inputs.json')},
        source_actions={d['pair']:digest(SOURCE/d['pair']/'actions.csv') for d in definitions},
        source_states={d['pair']:digest(SOURCE/d['pair']/'states.parquet') for d in definitions},
        code={p:digest(p) for p in [__file__, 'scripts/contract_engine.py', 'scripts/contract_risk.py']},
        variants=VARIANTS, development_end='2021-12-31', later_start='2022-01-01',
        selection='Maximum development daily excess Sharpe; baseline included; no threshold fitting')
    path=REPORT/'inputs.json'
    if path.exists() and json.loads(path.read_text())!=fingerprint: raise ValueError('Inputs changed')
    path.write_text(json.dumps(fingerprint,indent=2)+'\n')
    calendar=load_calendar(ROOT)
    calendar=calendar[(calendar>=previous['start']) & (calendar<=previous['end'])]
    market=Market(calendar); records=[]; started=time.monotonic()
    for i,d in enumerate(definitions,1):
        folder=OUTPUT/d['pair']; folder.mkdir(exist_ok=True)
        if (folder/'status.json').exists():
            records.extend(json.loads((folder/'status.json').read_text())); continue
        states=pd.read_parquet(SOURCE/d['pair']/'states.parquet')
        actions=pd.read_csv(SOURCE/d['pair']/'actions.csv',index_col=0,parse_dates=[0])
        q=stocks=selection=divs=prepared=None; pair_rows=[]
        for model in MODELS:
            for variant in VARIANTS:
                filtered=gate_actions(actions[model],states,variant)
                dest=folder/f'{model}_{variant}';dest.mkdir(exist_ok=True)
                filtered.to_csv(dest/'actions.csv')
                row=dict(pair=d['pair'],label=d['label'],group=d['group'],model=model,variant=variant,
                         original_signals=int(actions[model].ne(0).sum()),nonzero_signals=int(filtered.ne(0).sum()))
                try:
                    if not filtered.ne(0).any():
                        nav=pd.DataFrame(index=calendar)
                        nav['return_']=market.rates.shift(1).fillna(0)*calendar.to_series().diff().dt.days.fillna(0)/365
                        nav['nav']=cfg.capital*(1+nav.return_).cumprod()
                        mark_book(nav,market.rates,cfg.capital); entries=0; ledger=0.
                    else:
                        if prepared is None:
                            q,stocks,selection,divs=market.pair(d['secids'])
                            prepared=PreparedMarket(q,stocks,selection)
                        nav,trades,decisions=simulate(q,stocks,filtered,calendar,d['secids'],cfg,divs,selection,prepared)
                        mark_book(nav,market.rates,cfg.capital)
                        ledger=audit(nav,trades,stocks,cfg,d['secids'])['ledger_error']
                        trades.to_csv(dest/'trades.csv',index=False)
                        decisions.to_csv(dest/'decisions.csv',index=False)
                        entries=int(decisions.status.eq('entered').sum()) if len(decisions) else 0
                    nav.to_parquet(dest/'daily_nav.parquet')
                    row.update(status='complete',entries=entries,ledger_error=ledger)
                except (UnresolvedPosition,ValueError) as exc:
                    row.update(status='incomplete',error=str(exc))
                pair_rows.append(row)
        (folder/'status.json').write_text(json.dumps(pair_rows,indent=2)+'\n')
        records.extend(pair_rows)
        print(f"{i}/{len(definitions)} {d['label']}: "+', '.join(f"{r['model']}/{r['variant']} {r.get('entries',r.get('error'))}" for r in pair_rows),flush=True)
    manifest=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  inputs_sha256=digest(path))
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(definitions,records,manifest)


if __name__=='__main__': main()
