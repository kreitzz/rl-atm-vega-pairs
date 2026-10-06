"""Read-only audit of old smile proposals under the standardized signal rules.

No fitting, replay, or forward outcomes are used to classify eligibility.
Throttle to approximately half of one core and stop above 3 GiB RSS.
"""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    try:
        os.nice(15)
    except PermissionError:
        pass  # Sandbox can forbid priority changes; CPU throttling remains.
    wall, cpu = time.monotonic(), time.process_time()
    out = Path('reports/structure_erosion')
    out.mkdir(parents=True, exist_ok=True)
    old = pq.read_table('artifacts/contracts/joint_smile/training_rows.parquet',
                        columns=['date', 'pair', 'wing', 'direction'], use_threads=False).to_pandas()
    parts = []
    for (pair, wing), group in old.groupby(['pair', 'wing']):
        path = Path('artifacts/contracts/uniform_smile/pairs') / pair / f'{wing}_observed.parquet'
        raw = pq.read_table(path, use_threads=False).to_pandas()
        raw = raw.loc[raw.date.isin(group.date)].copy()
        raw['pair'], raw['wing'] = pair, wing
        calendar_path = Path('artifacts/contracts/tail_put25_strategy') / pair / 'rule/actions.csv'
        dates = pd.to_datetime(pd.read_csv(calendar_path).iloc[:, 0]) if calendar_path.exists() else []
        raw['formation_calendar'] = raw.date.isin(dates)
        parts.append(raw)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 3 * 1024**3:
            raise MemoryError('3 GiB RSS stop')
        # Wall time must be at least twice the process CPU consumed.
        delay = 2 * (time.process_time() - cpu) - (time.monotonic() - wall)
        if delay > 0:
            time.sleep(min(delay, 1))
    raw = pd.concat(parts, ignore_index=True)
    audit = old.rename(columns={'direction': 'old_direction'}).merge(raw, on=['date', 'pair', 'wing'], how='left', validate='one_to_one')
    audit['current_observed'] = audit.observed.fillna(False).astype(bool)
    audit['current_beta_eligible'] = audit.eligible.fillna(False).astype(bool)
    audit['current_context'] = audit.context.fillna(False).astype(bool)
    audit['formation_calendar'] = audit.formation_calendar.fillna(False).astype(bool)
    audit['current_rank_active'] = audit.rule_direction.fillna(0).ne(0)
    audit['same_direction'] = audit.rule_direction.eq(audit.old_direction)
    for days in [91, 365]:
        for prefix in ['profit', 'count']:
            audit[f'current_{prefix}_{days}d'] = np.where(audit.rule_direction.gt(0), audit[f'{prefix}_long_{days}d'], audit[f'{prefix}_short_{days}d'])
    audit['current_memory_pass'] = (audit.current_count_91d.ge(21) & audit.current_count_365d.ge(63) & audit.current_profit_91d.gt(0) & audit.current_profit_365d.gt(0))
    steps = ['current_observed', 'current_beta_eligible', 'current_context', 'formation_calendar', 'current_rank_active', 'current_memory_pass']
    keep = np.ones(len(audit), bool)
    audit['first_exclusion'] = 'retained'
    for step in steps:
        fail = keep & ~audit[step].to_numpy(bool)
        audit.loc[fail, 'first_exclusion'] = step
        keep &= audit[step].to_numpy(bool)
    audit['retained'] = keep
    decisions = pd.read_csv('artifacts/contracts/joint_smile/shared_frozen_native_limits/decisions.csv', parse_dates=['signal_date'])
    executed = decisions.loc[decisions.status.eq('entered'), ['signal_date', 'pair', 'wing']].rename(columns={'signal_date': 'date'})
    executed['old_entered'] = True
    audit = audit.merge(executed, on=['date', 'pair', 'wing'], how='left', validate='one_to_one')
    audit['old_entered'] = audit.old_entered.eq(True)
    summaries = []
    for population, data in [('all_old_proposals', audit), ('old_actual_entries', audit.loc[audit.old_entered])]:
        for wing, group in data.groupby('wing'):
            row = dict(population=population, wing=wing, old_count=len(group), retained=int(group.retained.sum()), retained_same_direction=int((group.retained & group.same_direction).sum()), retained_opposite_direction=int((group.retained & ~group.same_direction).sum()))
            row.update({f'first_excluded_{step}': int(group.first_exclusion.eq(step).sum()) for step in steps})
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'proposal_waterfall.csv', index=False)
    audit.to_csv(out / 'proposal_audit.csv', index=False)
    # Existing standalone ATM ledger, not the shared smile execution ledger;
    # the two books have three different entry decisions. Financing stays separate.
    cohorts = pd.read_csv('reports/current_trade_quality/cohorts.csv')
    cohorts = cohorts.loc[cohorts.model.eq('rl_baseline')].copy()
    cohorts['date'] = pd.to_datetime(cohorts.cohort)
    atm = audit.loc[audit.wing.eq('ATM'), ['date', 'pair', 'first_exclusion']]
    cohorts = cohorts.merge(atm, on=['date', 'pair'], how='left', validate='one_to_one')
    assert len(cohorts) == 224
    cohorts['first_exclusion'] = cohorts.first_exclusion.fillna('unmatched_old_proposal')
    pnl = cohorts.groupby('first_exclusion').agg(old_executed_cohorts=('pair', 'size'), old_net_before_shared_financing=('net_before_shared_financing', 'sum'))
    pnl.to_csv(out / 'old_atm_cohort_pnl.csv')
    fits = pd.read_csv('reports/same_structure_smile/fits.csv')
    fits = fits.loc[fits.variant.eq('floor')]
    training = fits.groupby('wing').agg(months=('training_rows', 'size'), minimum_training_rows=('training_rows', 'min'), median_training_rows=('training_rows', 'median'), maximum_training_rows=('training_rows', 'max'))
    training.to_csv(out / 'training_coverage.csv')
    (out / 'audit.json').write_text(json.dumps(dict(status='complete', no_retraining=True, no_backtest=True, classification_uses_forward_outcomes=False, waterfall_order=steps, pnl_scope='Old standalone ATM executed-cohort ledger, before shared financing. Eligibility counts use old shared smile entries; three execution decisions differ between books.', note='Exclusions overlap; first-exclusion counts depend on displayed order. A direction flip is a replacement, not a rejection.', peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), indent=2))
    print(summary.to_string(index=False))
    print(training.to_string())
    print(pnl.to_string())


if __name__ == '__main__':
    main()
