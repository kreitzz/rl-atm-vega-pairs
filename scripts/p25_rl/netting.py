"""Mechanical option-entry netting headroom in verified frozen baseline orders.

Run after the serial research reports:
    python -m scripts.p25_rl.netting

Opposing entries in the same contract/session can cancel across different
pairs only when fill price and per-contract commission/penalty agree. This
diagnostic changes no orders, NAV, risk limits or strategy. Its dollar savings
are a frozen-order feasibility calculation, not a funded replay or Sharpe gain.
"""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_name] = '1'

import argparse
from collections import defaultdict, deque
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'artifacts/contracts/p25_inventory_q/replays/baseline_verification'
CAPACITY_REPORT = ROOT / 'reports/p25_inventory_q'
REPORT = ROOT / 'reports/p25_rl_research'


class Budget:
    """Serial worker, cumulative 20% CPU duty and 384 MiB RSS ceiling."""
    def __init__(self):
        self.process = psutil.Process()
        self.started = time.monotonic()
        cpu = self.process.cpu_times()
        self.cpu_start = cpu.user + cpu.system
        self.peak = 0.
        try:
            self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
        except (psutil.Error, AttributeError):
            pass

    def rest(self):
        while True:
            self.peak = max(self.peak, self.process.memory_info().rss / 1024**2)
            if self.peak > 384:
                raise MemoryError('Netting worker exceeded 384 MiB RSS')
            if (REPORT / 'NETTING_STOP').exists():
                raise InterruptedError('Netting feasibility stop requested')
            cpu = self.process.cpu_times()
            delay = (cpu.user + cpu.system - self.cpu_start) / .2 - (time.monotonic() - self.started)
            if delay <= 0:
                return
            time.sleep(min(delay, .5))


def _integer(value):
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError('Finite integer quantity/contract ID required')
    return int(number)


def _maximum_cancelled(positive, negative):
    """Maximum integer cancellation with self-pair matching prohibited."""
    source, sink = ('source', ''), ('sink', '')
    capacity = defaultdict(dict)
    adjacency = defaultdict(list)

    def edge(a, b, quantity):
        adjacency[a].append(b); adjacency[b].append(a)
        capacity[a][b] = quantity; capacity[b][a] = 0

    limit = min(sum(positive.values()), sum(negative.values()))
    for pair, quantity in sorted(positive.items()):
        edge(source, ('positive', pair), quantity)
    for pair, quantity in sorted(negative.items()):
        edge(('negative', pair), sink, quantity)
    for positive_pair in sorted(positive):
        for negative_pair in sorted(negative):
            if positive_pair != negative_pair:
                edge(('positive', positive_pair), ('negative', negative_pair), limit)
    total = 0
    while True:
        parents = {source: None}; pending = deque([source])
        while pending and sink not in parents:
            node = pending.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in parents and capacity[node][neighbor] > 0:
                    parents[neighbor] = node; pending.append(neighbor)
        if sink not in parents:
            return total
        amount = limit; node = sink
        while parents[node] is not None:
            previous = parents[node]
            amount = min(amount, capacity[previous][node]); node = previous
        node = sink
        while parents[node] is not None:
            previous = parents[node]
            capacity[previous][node] -= amount
            capacity[node][previous] += amount
            node = previous
        total += amount


def potential_netting(trades, rest=lambda: None):
    """Return opposing entry groups and potential savings at unchanged fills.

    Input is an iterable of ledger dictionaries. Paid cost includes all saved
    option commissions and explicit execution penalties; expiry/split rows
    normally contribute zero. Other kinds of fills are never cancelled here.
    """
    grouped = defaultdict(list)
    paid_commission = paid_penalty = 0.
    entry_order_contracts = 0
    for index, trade in enumerate(trades):
        commission = float(trade['commission'])
        penalty = float(trade['execution_penalty'])
        if not all(math.isfinite(value) and value >= 0 for value in [commission, penalty]):
            raise ValueError('Invalid saved option commission/penalty')
        paid_commission += commission; paid_penalty += penalty
        if trade['reason'] == 'entry':
            quantity = _integer(trade['quantity'])
            if quantity == 0:
                raise ValueError('Zero entry quantity')
            price = float(trade['price'])
            if not math.isfinite(price) or price < 0:
                raise ValueError('Invalid saved option entry price')
            date = str(trade['date'])[:10]
            optionid = _integer(trade['optionid'])
            grouped[(date, optionid)].append(dict(
                pair=str(trade['pair']), quantity=quantity, price=price,
                commission_per_contract=commission / abs(quantity),
                penalty_per_contract=penalty / abs(quantity)))
            entry_order_contracts += abs(quantity)
        if index % 100 == 99:
            rest()
    rows = []
    for (date, optionid), orders in sorted(grouped.items()):
        positive = defaultdict(int); negative = defaultdict(int)
        for order in orders:
            destination = positive if order['quantity'] > 0 else negative
            destination[order['pair']] += abs(order['quantity'])
        if not positive or not negative:
            continue
        first = orders[0]
        equal_terms = all(
            math.isclose(order[key], first[key], rel_tol=1e-10, abs_tol=1e-12)
            for order in orders for key in
            ['price', 'commission_per_contract', 'penalty_per_contract'])
        matched = _maximum_cancelled(positive, negative) if equal_terms else 0
        cancelled = 2 * matched
        commission = cancelled * first['commission_per_contract']
        penalty = cancelled * first['penalty_per_contract']
        rows.append(dict(date=date, optionid=optionid,
            positive_contracts=sum(positive.values()), negative_contracts=sum(negative.values()),
            positive_pairs=';'.join(sorted(positive)), negative_pairs=';'.join(sorted(negative)),
            identical_per_contract_terms=equal_terms,
            matched_contracts_per_side=matched, cancelled_order_contracts=cancelled,
            potential_commission_savings=commission, potential_penalty_savings=penalty,
            potential_total_savings=commission + penalty))
        rest()
    paid = paid_commission + paid_penalty
    summary = dict(entry_contract_groups=len(grouped), opposing_entry_groups=len(rows),
        equal_term_opposing_groups=sum(row['identical_per_contract_terms'] for row in rows),
        cancellable_groups=sum(row['matched_contracts_per_side'] > 0 for row in rows),
        entry_order_contracts=entry_order_contracts,
        matched_contracts_per_side=sum(row['matched_contracts_per_side'] for row in rows),
        cancelled_order_contracts=sum(row['cancelled_order_contracts'] for row in rows),
        paid_option_commission=paid_commission, paid_option_execution_penalty=paid_penalty,
        paid_explicit_option_cost=paid,
        potential_commission_savings=sum(row['potential_commission_savings'] for row in rows),
        potential_penalty_savings=sum(row['potential_penalty_savings'] for row in rows),
        potential_total_savings=sum(row['potential_total_savings'] for row in rows))
    summary['share_of_paid_explicit_option_cost'] = summary['potential_total_savings'] / paid if paid else None
    summary['share_of_entry_order_contracts'] = summary['cancelled_order_contracts'] / entry_order_contracts if entry_order_contracts else None
    return rows, summary


def _digest(path, budget):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block); budget.rest()
    return result.hexdigest()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    budget = Budget()
    status_path = BASELINE / 'status.json'
    verification_path = CAPACITY_REPORT / 'baseline_verification.json'
    completion_path = CAPACITY_REPORT / 'completion.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    if (status['status'] != 'passed' or completion['status'] != 'passed'
            or verification['status'] != 'passed'
            or not all(verification.get(name) is True for name in
                       ['all_nav_columns_match', 'all_trades_match', 'all_decisions_match'])
            or status['kind'] != 'full'):
        raise ValueError('Run the verified baseline and entry-family report first')
    sources = [BASELINE / 'trades.csv', BASELINE / 'daily_nav.parquet',
               status_path, verification_path, completion_path, Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): _digest(path, budget) for path in sources}
    for name, expected in status['source_code'].items():
        if _digest(ROOT / name, budget) != expected:
            raise ValueError(f'Baseline replay source changed: {name}')
        hashes[name] = expected
    with (BASELINE / 'trades.csv').open(newline='', encoding='utf-8') as stream:
        rows, summary = potential_netting(csv.DictReader(stream), budget.rest)
    # Only two saved account columns are needed; no strategy is replayed.
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    table = pq.read_table(BASELINE / 'daily_nav.parquet',
                          columns=['nav', 'cash_benchmark'], use_threads=False)
    nav = table.column('nav'); cash = table.column('cash_benchmark')
    profit = float(nav[-1].as_py() - cash[-1].as_py()
                   - nav[0].as_py() + cash[0].as_py())
    summary['baseline_full_period_profit_above_cash'] = profit
    summary['share_of_baseline_profit_above_cash'] = summary['potential_total_savings'] / profit if profit > 0 else None
    budget.rest()
    REPORT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT / 'netting_feasibility.csv'
    columns = ['date', 'optionid', 'positive_contracts', 'negative_contracts',
        'positive_pairs', 'negative_pairs', 'identical_per_contract_terms',
        'matched_contracts_per_side', 'cancelled_order_contracts',
        'potential_commission_savings', 'potential_penalty_savings', 'potential_total_savings']
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    share_cost = summary['share_of_paid_explicit_option_cost']
    share_profit = summary['share_of_baseline_profit_above_cash']
    percent = lambda value: 'unavailable' if value is None else f'{100 * value:.2f}%'
    lines = ['FROZEN-ORDER OPTION NETTING FEASIBILITY', '',
        f'Opposing same-session contract groups: {summary["opposing_entry_groups"]}.',
        f'Cancellable across different pairs with equal fill/cost terms: {summary["cancellable_groups"]}.',
        f'Matched contracts per side: {summary["matched_contracts_per_side"]}; canceled order contracts: {summary["cancelled_order_contracts"]}.',
        f'Potential saved commission/penalty: ${summary["potential_total_savings"]:,.2f}.',
        f'Share of paid explicit option cost: {percent(share_cost)}.',
        f'Saved baseline full-period profit above cash: ${profit:,.2f}; potential savings share: {percent(share_profit)}.', '',
        ('No mechanical entry-netting savings are observed; this direction has no demonstrated cost headroom in these orders.'
         if summary['potential_total_savings'] == 0 else
         'Mechanical savings exist at these frozen orders; this quantifies cost headroom only.'),
        'No order, position, risk limit, NAV, or strategy was changed. This is not a funded portfolio test, Q-learning result, or Sharpe improvement.',
        'Savings use original accepted entry quantities and matching observed fill prices/per-contract commission and penalty.',
        'Option premiums cancel mechanically at the common price. Existing shared stock hedges are already netted.',
        'Cost/admission feedback, virtual pair accounting, and any alternative portfolio policy would require a separate replay.',
        'All historical dates are inherited from the verified 2018-August 2025 full baseline; no new holdout is introduced.', '']
    text_path = REPORT / 'netting_feasibility.txt'
    text_path.write_text('\n'.join(lines), encoding='utf-8')
    for name, expected in hashes.items():
        if _digest(ROOT / name, budget) != expected:
            raise ValueError(f'Netting input changed during analysis: {name}')
    result = dict(status='passed', kind='mechanical frozen-order feasibility',
        baseline=str(BASELINE.relative_to(ROOT)), period='2018-August2025 full baseline',
        summary=summary, source_hashes=hashes,
        output_hashes={path.name: _digest(path, budget) for path in [csv_path, text_path]},
        resources=dict(threads=1, workers=1, duty=.2, rss_limit_mib=384,
                       peak_rss_mib=budget.peak),
        limitations=['No altered fills or quantities', 'No portfolio/reinforcement-learning replay',
                     'No funded return or Sharpe calculation', 'No future holdout',
                     'Only explicit saved option commissions and penalties counted'])
    (REPORT / 'netting_feasibility.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('\n'.join(lines[:11]), flush=True)


if __name__ == '__main__':
    main()
