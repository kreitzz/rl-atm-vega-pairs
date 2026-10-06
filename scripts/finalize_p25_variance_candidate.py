"""Verify artifacts and save research checkpoints, inputs, and a concise report."""
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts import run_p25_action_windows as base
from scripts.run_p25_feature_search import fields

REPORT = Path('reports/p25_variance_validation')
SOURCE = Path('artifacts/contracts/p25_feature_search')
VALIDATION = Path('artifacts/contracts/p25_variance_validation')


def main():
    manifests = [Path('reports') / folder / 'output_hashes.json' for folder in ['p25_sharpe_search', 'p25_feature_search', 'p25_variance_validation']]
    checked = 0
    for manifest in manifests:
        for path, expected in json.loads(manifest.read_text()).items():
            assert base.digest(Path(path)) == expected, path
            checked += 1
    print(f'Verified {checked} saved output hashes', flush=True)
    suite = unittest.TestSuite()
    for pattern in ['test_tail_feature_exploration.py', 'test_volume_oi_features.py', 'test_atm65_employee_sentiment.py', 'test_p25_action_windows.py', 'test_p25_recency.py', 'test_p25_risk_sizing.py']:
        suite.addTests(unittest.defaultTestLoader.discover('tests', pattern=pattern))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    meta = pd.read_parquet(SOURCE / 'training_rows.parquet')
    inputs = meta[['date', 'pair', 'wing', 'secid_a', 'secid_b', 'optionid_a', 'optionid_b', 'direction', 'observed', 'eligible', 'context', 'signal_eligible'] + fields('variance_premium')].copy()
    assert np.isfinite(inputs[fields('variance_premium')]).all().all()
    assert not set(inputs).intersection({'target', 'target_long', 'target_short', 'finite_label', 'end', 'reward'})
    destination = Path('data/p25_variance_premium_inputs.csv')
    destination.parent.mkdir(exist_ok=True)
    inputs.to_csv(destination, index=False)
    candidate = pd.read_parquet(SOURCE / 'replays/variance_premium/daily_nav.parquet')
    for seed in [2026, 1, 7, 42, 2027]:
        pd.testing.assert_frame_equal(candidate, pd.read_parquet(VALIDATION / 'replays' / f'sharpe_seed{seed}' / 'daily_nav.parquet'))
    previous = pd.read_parquet(SOURCE / 'replays/control/daily_nav.parquet')
    annual = pd.read_csv(Path('reports/p25_feature_search/annual.csv'))
    annual = annual[annual.model.isin(['variance_premium', 'control', 'fixed2', 'fixed4'])].copy()
    annual.to_csv(REPORT / 'candidate_annual.csv', index=False)
    annual.pivot(index='year', columns='model', values='sharpe').to_csv(REPORT / 'candidate_annual_sharpe.csv')
    decisions = pd.read_csv(SOURCE / 'replays/variance_premium/decisions.csv', parse_dates=['signal_date', 'entry_date'])
    prior = pd.read_csv(SOURCE / 'replays/control/decisions.csv', parse_dates=['signal_date', 'entry_date'])
    pd.testing.assert_frame_equal(decisions[['signal_date', 'pair', 'wing']], prior[['signal_date', 'pair', 'wing']])
    merged = decisions[['signal_date', 'pair', 'wing', 'level', 'status']].merge(prior[['signal_date', 'pair', 'wing', 'level', 'status']], on=['signal_date', 'pair', 'wing'], suffixes=('_new', '_old'), validate='one_to_one')
    changed = merged[merged.level_new.ne(merged.level_old) | merged.status_new.ne(merged.status_old)]
    changed.to_csv(REPORT / 'candidate_changed_decisions.csv', index=False)
    proposed = pd.read_parquet(SOURCE / 'replays/variance_premium/scored_allocations.parquet')
    splits = []
    for month in pd.period_range('2018-01-01', '2025-06-01', freq='M'):
        model = joblib.load(SOURCE / 'models/variance_premium' / f'{month.start_time.date()}.joblib')
        if model['tree'] is not None:
            used = model['tree'].tree_.feature
            for f, count in zip(*np.unique(used[used >= 0], return_counts=True)):
                splits.append(dict(fit=str(month.start_time.date()), feature=model['features'][f], splits=int(count)))
    pd.DataFrame(splits).to_csv(REPORT / 'candidate_split_usage.csv', index=False)
    details = json.loads((SOURCE / 'replays/variance_premium/status.json').read_text())
    prior_config = Path('config/p25_recency_candidate.json')
    prior_hash = base.digest(prior_config)
    bootstrap = pd.read_csv(REPORT / 'bootstrap.csv')
    row = bootstrap[(bootstrap.comparison == 'vs_best_fixed') & (bootstrap.block_sessions == 126)].iloc[0]
    checkpoint = dict(
        status='research_candidate', model='P25_EWMA_variance_premium_18M_floor',
        previous_candidate=str(prior_config), previous_candidate_sha256=prior_hash,
        model_path=str(SOURCE / 'models/variance_premium'), replay=str(SOURCE / 'replays/variance_premium'),
        annualized_cash_excess_sharpe=details['annualized_excess_sharpe'], net_excess_pnl=details['net_excess_pnl'], entries=details['entries'], active_pairs=details['active_pairs'],
        evaluation_start='2018-01-02', evaluation_end='2025-08-29', last_signal='2025-06-30',
        training_window_months=18, refit_frequency='monthly', sizing_actions=[.5, .75, 1.],
        algorithm='Contextual regression-tree states with direct portfolio-Sharpe CEM allocation; not sequential inventory-trained RL',
        tree_max_leaves=8, optimizer_iterations=50, optimizer_candidates=192, retained_seed=2026,
        features=fields('variance_premium'), added_signals=['EWMA variance premium long-minus-short', 'Historical variance-premium z-score long-minus-short', 'EWMA realized-to-implied variance ratio long-minus-short'],
        ewma_half_life_sessions=20, ewma_minimum_observations=63, z_score_prior_window=252, z_score_minimum_observations=126,
        formation_calendar='Original retrospective pair calendar retained; not newly verified prospective universe selection',
        entry_rule='Original252-session IV-spread20/80 ranks; beta0.85-1.15; correlation>=0.5; positive91/365-day completed directional memory and count minima21/63',
        holding='Approximately30D25-delta puts; integer vega matched at entry within10%; daily delta hedge; each leg held to its expiry except audited corporate exits',
        costs=dict(training_commission=.65, actual_option_commission=.03, option_penalty_vol_points_per_leg=.1, stock_hedge_bps=1),
        risk=dict(stress_fraction=.05, reserve_fraction=.8, hedge_gross_multiple=2, full_vega_per_leg=450, pair_reference_capital=1_000_000, funded_capital=345_000_000, gross_vega_per_pair_reference=20_000, origin='Inherited implementation assumptions, not user-provided numerical risk limits'),
        selected_feature_block_using='2018-2023 Sharpe among11 declared blocks; all history previously inspected',
        recent_2024_2025_sharpe=base.sharpe(candidate.loc['2024-01-01':, 'excess_return']),
        fixed_50_sharpe=.769675349785327, fixed_100_sharpe=.756931810129242,
        prior_learned_sharpe=.9617449126497056, same_states_profit_objective_sharpe=.979114, same_states_diagonal_sharpe=1.007098,
        seed_check='All five CEM seeds2026/1/7/42/2027 produce identical daily NAV and213 entries; tree partition fixed',
        paired126_session_bootstrap95pct_vs_best_fixed=[float(row.ci_low), float(row.ci_high)],
        selection_adjusted=False, no_fresh_holdout=True, original_checkpoints_preserved=True,
        input_csv=str(destination), input_csv_sha256=base.digest(destination), findings=str(REPORT / 'research_summary.txt'),
    )
    base.write_json(Path('config/p25_variance_premium_candidate.json'), checkpoint)
    diagonal = json.loads((VALIDATION / 'replays/diagonal_sharpe/status.json').read_text())
    base.write_json(Path('config/p25_variance_diagonal_candidate.json'), dict(status='diagnostic_variant', base_candidate='config/p25_variance_premium_candidate.json', replay=str(VALIDATION / 'replays/diagonal_sharpe'), models=str(VALIDATION / 'models/diagonal_sharpe'), algorithm='Same contextual tree and direct-Sharpe CEM, covariance diagonalized during fitting', annualized_cash_excess_sharpe=diagonal['annualized_excess_sharpe'], entries=diagonal['entries'], net_excess_pnl=diagonal['net_excess_pnl'], no_fresh_holdout=True))
    assert base.digest(prior_config) == prior_hash
    study_comparison = pd.read_csv(Path('reports/p25_feature_search/comparison.csv'))
    validation_comparison = pd.read_csv(REPORT / 'comparison.csv')
    best_fixed = study_comparison.loc[study_comparison.model.eq('fixed2')].iloc[0]
    excess = candidate.nav - candidate.cash_benchmark
    peak = excess.cummax()
    summary = (
        'Best retained research candidate: annualized daily cash-excess Sharpe1.006304706; '
        '213 pair entries,42 traded pairs,$320285.67 above cash,maximum excess-profit '
        'drawdown$46012.45. Best observed objective variant1.007098 is effectively tied.\n'
        'Added20-session half-life EWMA expected variance premium, its prior252-session '
        'history z-score, and expected-realized/implied variance ratio, each as long-minus-short '
        'plus availability flag. Existing27 features,608 proposals, direction, contracts, '
        'profit-memory/pair/formation rules,18-month monthly fitting,50/75/100 actions, '
        'costs and original risk caps unchanged. No learned skipping.\n'
        'Fixed50/100 Sharpe0.769675/0.756932; previous learned0.961745; same-state '
        'profit-only sizing0.979114; standalone-state Sharpe0.725012; diagonal covariance1.007098. '
        'Five allocator seeds yield identical candidate NAV; this does not test alternate tree '
        'partitions or prove structural robustness. Matched proposals are identical; executable '
        'accepted trades differ because size affects entry risk admission and subsequent inventory.\n'
        'Ten of11 feature configurations beat the best constant size; transcripts fail. '
        'Only variance-premium additions improve the original learned baseline. Most extra '
        'features are unhelpful, and IV-factor coverage is sparse. Wider entry boundaries and '
        'removing caps reduced Sharpe in the preceding study. See all reports, not only winners.\n'
        'Profit-only sizing is close to direct-Sharpe sizing. This supports adaptive sizing '
        'over constant sizing on this sample, not universal RL superiority or a new NN result. '
        'Current method is contextual tree/CEM optimization, not sequential inventory-trained RL.\n'
        '2024 remains negative;2024-partial2025 combined Sharpe is-.021153. '
        '2022 supplies about55%of total net excess dollars. Bootstrap uncertainty is broad; '
        'the incremental feature improvement versus prior RL includes zero. Multiple research '
        'selection and retrospective universe/calendar conditioning are uncorrected. '
        'A1.0 sample estimate is achieved, not a guaranteed population Sharpe or fresh validation.\n'
        'Data ends2025-08-29;2025 is partial with final signal2025-06-30 and expiry runoff. '
        'Sharpe uses sqrt252 and daily net cash-excess returns,ddof1. '
        'Cash-excess risk is reported; funded NAV drawdown includes interest on large idle cash.\n'
        f'Inputs-only CSV:{destination};{len(inputs)} rows;33 model fields plus identifiers and eligibility. '
        'No target, reward, outcome completion or future availability fields. Counts/profit memory '
        'are completed-history input states, not future rewards. Cached option chains and training '
        'reward paths remain in existing local research files. Vendor inputs remain subject to '
        'existing data-access terms.\n\n'
        + study_comparison.to_string(index=False) + '\n\n'
        + validation_comparison.to_string(index=False) + '\n\n'
        + annual.pivot(index='year', columns='model', values='sharpe').to_string() + '\n'
    )
    (REPORT / 'research_summary.txt').write_text(summary)
    base.write_json(REPORT / 'final_audit.json', dict(status='complete', output_hashes_verified=checked, tests_run=result.testsRun, tests_passed=True, candidate_sharpe=base.sharpe(candidate.excess_return.iloc[1:]), all_seeds_same_cash_book=True, original_checkpoint_hash_preserved=prior_hash, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, input_rows=len(inputs), model_fields=33, changed_proposal_levels=int(merged.level_new.ne(merged.level_old).sum()), changed_decision_statuses=int(merged.status_new.ne(merged.status_old).sum()), worst_daily_cash_excess_pnl=float((candidate.pnl - candidate.cash_benchmark.diff().fillna(0)).min()), max_excess_drawdown=float((excess - peak).min()), methodology='Annualized daily cash-excess; previously explored history; no selection correction'))
    paths = [p for root in [VALIDATION, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json']
    paths += [destination, Path('config/p25_variance_premium_candidate.json'), Path('config/p25_variance_diagonal_candidate.json'), Path(__file__)]
    base.write_json(REPORT / 'output_hashes.json', {str(p): base.digest(p) for p in paths})
    print(summary, flush=True)


if __name__ == '__main__':
    main()
