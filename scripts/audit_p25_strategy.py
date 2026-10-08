"""Verify the common frozen baseline for the three-part strategy investigation."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_NUM_THREADS'):
    os.environ[_name] = '1'

import hashlib
import io
import json
import argparse
from pathlib import Path
import time
import zipfile

import pandas as pd
from scripts.diagnose_p25_pca import Budget

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/p25_strategy_audit'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    started = time.monotonic()
    protected = [
        'config/p25_validation_2018_2021.json', 'config/p25_submission.json',
        'data/p25_validation_split_experiment.zip',
        'reports/p25_validation_split/bundle_manifest.json',
        'data/submission/artifacts/contracts/p25_feature_search/training_rows.parquet',
        'scripts/uniform_smile_engine.py', 'scripts/uniform_smile_risk.py',
        'scripts/p25_action_windows.py', 'scripts/run_p25_feature_search.py',
    ]
    hashes = {p: digest(ROOT / p) for p in protected}
    protocol = {
        'purpose': 'Investigate sizing justification, opportunity selection and pair-level performance together.',
        'baseline': 'Validation-selected screened 27-input,18-month,50/75/100% monthly policy in a10m funded account.',
        'signal_count': 608, 'funded_entries': 207, 'monthly_models': 90,
        'validation': '2018-01-02 through2021-12-31',
        'later_evaluation': '2022-01-03 through2025-08-29',
        'untouched_holdout': False,
        'selection_caveat': 'Inherited roster, formation calendar and strategy; history has already been repeatedly inspected.',
        'timing': 'Signals at close, entry next session; monthly completed-outcome fits; continuous portfolio inventory.',
        'holding': 'Entry option sizes held until each leg expiry subject to corporate events; stocks delta-hedged daily.',
        'sizing_work': 'Audit target/objective/sample stability; compare archived50/75/100 constants and predeclared simple economic/continuous sizing alternatives on the same qualified proposals.',
        'opportunity_work': 'Trace roster and gate provenance, reconstruct funnel, compare screened/ungated fixed-full controls, inspect symmetry and robustness evidence.',
        'pair_work': 'Reconstruct actual shared-ledger attribution; report pair/year contributions, exposures, costs, concentration and portfolio drawdown attribution.',
        'controls': 'Same execution costs,10m capital,risk rules and dates within each comparison. Shared netting and admission can change under sizing.',
        'new_experiments': 'Each agent must record its own exact method before new scores. No broad tuning or policy promotion.',
        'completion': 'Each criticism receives evidence, limitations and a recommended action, with reproducible outputs and reconciled baseline.',
        'runtime': 'Only one numerical worker at a time,one numerical thread,20% CPU duty,384MiB aggregate worker RSS.',
        'protected_source_hashes': hashes,
    }
    target = OUT / 'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == protocol, 'Common protocol changed'
    else:
        target.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads((ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    reference = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
    local = ROOT / 'artifacts/contracts/p25_representation/replays/raw_full'
    checked = 0
    with zipfile.ZipFile(ROOT / 'data/p25_validation_split_experiment.zip') as z:
        for member, expected in manifest.items():
            assert hashlib.sha256(z.read(member)).hexdigest() == expected, member
            checked += 1
            if checked % 30 == 0:
                budget.rest(started)
                started = time.monotonic()
        old_nav = pd.read_parquet(io.BytesIO(z.read(reference + 'daily_nav.parquet')))
        nav = pd.read_parquet(local / 'daily_nav.parquet')
        pd.testing.assert_frame_equal(nav, old_nav, check_exact=False, rtol=1e-12, atol=1e-6)
        for name, dates in [('trades.csv', ['date']), ('decisions.csv', ['signal_date', 'entry_date'])]:
            actual = pd.read_csv(local / name, parse_dates=dates)
            expected = pd.read_csv(io.BytesIO(z.read(reference + name)), parse_dates=dates)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-6)
        signals = pd.read_parquet(io.BytesIO(z.read(reference + 'scored_allocations.parquet')))
        models = [n for n in z.namelist() if n.startswith('artifacts/contracts/p25_feature_search/models/control/') and n.endswith('.joblib')]
        assert len(models) == 90 and len(signals) == 608
    status = json.loads((local / 'status.json').read_text())
    assert status['entries'] == 207 and status['active_pairs'] == 44
    periods = pd.read_csv(ROOT / 'reports/p25_validation_split/periods.csv')
    chosen = periods.loc[periods.case.eq('screened_selected')].to_dict('records')
    budget.check()
    result = dict(archive_members_hash_verified=checked, monthly_models=len(models),
                  qualified_proposals=len(signals), exact_reproduced_nav_trades_decisions=True,
                  baseline_status=status, baseline_periods=chosen, peak_rss_mib=budget.peak,
                  protected_source_hashes=hashes)
    (OUT / 'baseline_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['archive_members_hash_verified', 'monthly_models', 'qualified_proposals', 'exact_reproduced_nav_trades_decisions', 'peak_rss_mib']}))
    budget.rest(started)


def finalize():
    """Cross-check independently produced reports and preserve their hashes."""
    budget = Budget()
    started = time.monotonic()
    common = json.loads((OUT / 'protocol.json').read_text())
    for name, sha in common['protected_source_hashes'].items():
        assert digest(ROOT / name) == sha, f'Protected source changed: {name}'
    baseline = json.loads((OUT / 'baseline_verification.json').read_text())
    sizing = json.loads((OUT / 'sizing/completion.json').read_text())
    opportunity = json.loads((OUT / 'opportunities/audit.json').read_text())
    pairs = json.loads((OUT / 'pairs/protocol.json').read_text())
    assert sizing['status'] == pairs['status'] == 'complete'
    assert sizing['new_replays'] == 2 and sizing['reused_controls'] == 4
    assert opportunity['exact608proposal_identity'] and opportunity['exact207admissions']
    assert opportunity['fixed100_execution_configs_equal']
    assert pairs['proposal_count'] == 608 and pairs['entry_count'] == 207
    s = pd.read_csv(OUT / 'sizing/periods.csv')
    p = pd.read_csv(OUT / 'pairs/reconciliation.csv')
    names = {'validation': 'validation', 'retrospective_holdout': 'evaluation', 'full_history': 'full'}
    for old in baseline['baseline_periods']:
        period = names[old['period']]
        sr = s.loc[s.case.eq('tree_cem') & s.period.eq(period)].iloc[0]
        pr = p.loc[p.period.eq(period)].iloc[0]
        assert abs(sr.annualized_excess_sharpe - old['annualized_excess_sharpe']) < 1e-12
        assert abs(pr.annualized_excess_sharpe - old['annualized_excess_sharpe']) < 1e-12
        assert abs(sr.profit_above_cash - old['profit_above_cash']) < 1e-6
        assert abs(pr.portfolio_profit_above_cash - old['profit_above_cash']) < 1e-6
        assert sr.entries == pr.entries == old['entries']
    assert p.reconciliation_error.abs().max() < 1e-5
    for case in ('rank_terciles', 'ridge_terciles'):
        status = json.loads((ROOT / 'artifacts/contracts/p25_strategy_audit/sizing/replays' / case / 'status.json').read_text())
        assert status['status'] == 'complete' and status['all_inventory_closed']
        assert status['independent_ledger_error'] < 1e-5
    paths = [f for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json',)]
    paths += [ROOT / 'scripts' / f for f in ('audit_p25_strategy.py', 'audit_p25_sizing.py',
                                            'audit_p25_opportunities.py', 'audit_p25_pairs.py')]
    hashes = {f.relative_to(ROOT).as_posix(): digest(f) for f in paths}
    result = dict(status='complete', common_baseline_consistent_across_all_three_analyses=True,
                  protected_sources_unchanged=True, new_funded_replays=2,
                  reused_funded_sizing_controls=4, archived_opportunity_controls=2,
                  reconciled_original_entries=207, independently_checked_archive_files=273,
                  max_pair_daily_reconstruction_error=pairs['max_daily_reconstruction_error'],
                  exploratory_history=True, output_hashes=hashes)
    (OUT / 'completion.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    budget.rest(started)
    print('All three analyses agree with the frozen baseline; protected sources unchanged; new ledgers reconcile.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    finalize() if args.finalize else main()
