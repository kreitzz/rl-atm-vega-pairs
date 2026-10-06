"""Check universal scoring and report learned allocation/return concentration."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.run_tail_put25_pooled import OUT, REPORT, MODELS
from scripts.curve_trading.config import ACTIONS
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json


def main():
    global OUT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed', action='store_true')
    args = parser.parse_args()
    if args.completed:
        OUT = Path('artifacts/contracts/tail_put25_pooled_complete')
        REPORT = Path('reports/tail_put25_pooled_complete')
    audit = json.loads((REPORT / 'audit.json').read_text())
    assert audit['status'] == 'complete'
    coverage = pd.read_csv(REPORT / 'coverage.csv')
    scored = coverage.scored_rows.fillna(0)
    np.testing.assert_array_equal(scored, coverage.evaluated_observed_rows)
    fits = pd.read_csv(REPORT / 'fits.csv')
    assert fits.status.eq('enabled').all()
    assert (pd.to_datetime(fits.last_training_outcome) < pd.to_datetime(fits.fit)).all()
    pairs = pd.read_csv(REPORT / 'pairs.csv')
    allocations, concentration, statuses = [], [], []
    for p in OUT.glob('*/pooled_rl_sized/actions.csv'):
        rl = pd.read_csv(p, index_col=0).code.astype(int)
        rule = pd.read_csv(p.parents[1] / 'ungated_spread/actions.csv', index_col=0).code.astype(int)
        pd.testing.assert_index_equal(rl.index, rule.index)
        a = np.array([ACTIONS[int(c)][1] if c else 0 for c in rl])
        b = np.array([ACTIONS[int(c)][1] if c else 0 for c in rule])
        assert ((a == 0) | (np.sign(a) == np.sign(b))).all()
        assert (abs(a) <= abs(b)).all()
        for size in [0., .25, .5, .75, 1.]:
            allocations.append(dict(pair=p.parents[1].name, absolute_size=size,
                count=int(((b != 0) & (abs(a) == size)).sum())))
    pd.DataFrame(allocations).to_csv(REPORT / 'allocation_counts.csv', index=False)
    for model in MODELS:
        book = pd.read_csv(REPORT / f'{model}_full_portfolio.csv', index_col=0, parse_dates=[0])
        cash_return = book.cash_benchmark.pct_change().fillna(0)
        winners = pairs[pairs.model.eq(model) & pairs.net_excess_pnl.gt(0)].nlargest(3, 'net_excess_pnl')
        without = book.nav.copy()
        for k, r in enumerate(winners.itertuples(), 1):
            nav = pd.read_parquet(OUT / r.pair / model / 'full/daily_nav.parquet')
            without -= nav.nav - nav.cash_benchmark
            excess = (without.pct_change().fillna(0) - cash_return).iloc[1:]
            concentration.append(dict(model=model, winners_removed=k,
                winner_names='; '.join(winners.iloc[:k].label),
                annualized_excess_sharpe=float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 1e-12 else None))
        for p in OUT.glob(f'*/{model}/full/status.json'):
            s = json.loads(p.read_text()); assert s['status'] == 'complete'
            statuses.append(s)
    pd.DataFrame(concentration).to_csv(REPORT / 'concentration.csv', index=False)
    checks = dict(status='passed', quote_covered_pairs=int(coverage.evaluated_observed_rows.gt(0).sum()),
        every_observed_state_scored=True, scored_rows=int(scored.sum()), monthly_pooled_models=len(fits),
        all_training_outcomes_before_fit=True, pair_model_activation_vetoes=0, memory_profitability_vetoes=0,
        completed_books=len(statuses), risk_breach_sessions=sum(s.get('risk_breach_sessions', 0) for s in statuses),
        quote_covered_pairs_outside_inherited_atm_panels=int((coverage.evaluated_observed_rows.gt(0) &
            coverage.inherited_atm_panel_exists.eq(False)).sum()))
    write_json(REPORT / 'universe_checks.json', checks)
    hashes = json.loads((REPORT / 'output_hashes.json').read_text())
    for p, sha in hashes.items():
        assert digest(p) == sha, f'Experiment output changed: {p}'
    for p in [Path(__file__), REPORT / 'allocation_counts.csv', REPORT / 'concentration.csv', REPORT / 'universe_checks.json']:
        hashes[str(p)] = digest(p)
    write_json(REPORT / 'output_hashes.json', hashes)
    print(json.dumps(checks, indent=2))
    print(pd.DataFrame(concentration).to_string(index=False))


if __name__ == '__main__':
    main()
