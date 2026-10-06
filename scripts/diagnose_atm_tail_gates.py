"""Audit saved activation and entry filters; no training or replay."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_atm65_volume_oi import EMPLOYEE, employee_roster, baseline_actions, baseline_results
from scripts.run_atm65_ablation import NN, CURVE, read_actions
from scripts.run_contract_walkforward import digest

OUT = Path('reports/atm_tail_gate_audit')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    defs = json.loads((EMPLOYEE / 'inputs.json').read_text())['definitions']
    covered = employee_roster(); fits = []; funnel = []; decisions = []; sources = {}
    def record(path):
        sources[str(path)] = digest(path)
        return path
    for d in defs:
        state_path = EMPLOYEE / d['pair'] / 'states.parquet'
        if not state_path.exists():
            continue
        panel = pd.read_parquet(record(state_path))
        apath = baseline_actions(d['pair'], 'rl', d['pair'] in covered)
        acts = read_actions(record(apath))
        if d['pair'] in covered:
            path = EMPLOYEE / d['pair'] / 'fits.csv'
            frame = pd.read_csv(record(path))
            fit = frame[['fit', 'training_rows', 'rl_sentiment_status']].rename(columns={'rl_sentiment_status': 'status'})
        else:
            path = CURVE / d['pair'] / 'atm_control/fits.csv'
            try: fit = pd.read_csv(record(path))
            except pd.errors.EmptyDataError: fit = pd.DataFrame(columns=['fit', 'status', 'training_rows'])
        fits.append(fit[['fit', 'status', 'training_rows']].assign(strategy='atm_rl', pair=d['pair'], label=d['label']))
        def stages(strategy, state, dates, gate_path):
            with np.load(record(gate_path)) as z:
                np.testing.assert_array_equal(z['dates'], state.index.values)
                gates = z['gates']; available = z['available']
            row = state.index.get_indexer(dates); assert (row >= 0).all()
            known = available[row]
            if 'feature_available' in state:
                known &= state.feature_available.to_numpy(bool)[row]
            eligible = known & state.eligible.to_numpy(bool)[row]
            memory = eligible & gates[row].any(axis=1)
            funnel.append(dict(strategy=strategy, pair=d['pair'], label=d['label'], evaluation_dates=len(dates),
                observed_pair_states=int(known.sum()), beta_eligible_states=int(eligible.sum()),
                memory_permitted_states=int(memory.sum()), memory_blocked_states=int((eligible & ~memory).sum())))
        stages('atm_rl', panel, acts.index, NN / d['pair'] / 'fee_065/gates.npz')
        for strategy, root, model in [
            ('tail_free_rl', Path('artifacts/contracts/tail_put25_strategy'), 'rl_sized'),
            ('tail_sizing_rl', Path('artifacts/contracts/tail_put25_rule_sizing'), 'rl_rule_sized')]:
            folder = root / d['pair']; state = pd.read_parquet(record(folder / 'states.parquet'))
            actionpath = folder / model / 'actions.csv'
            if not actionpath.exists(): continue
            own = read_actions(record(actionpath))
            stages(strategy, state, own.index, folder / 'gates.npz')
            fit = pd.read_csv(record(folder / ('fits.csv' if strategy == 'tail_free_rl' else model + '/fits.csv')))
            fits.append(fit[['fit', 'status', 'training_rows']].assign(strategy=strategy, pair=d['pair'], label=d['label']))
        for strategy, folder in [
            ('atm_rl', baseline_results(d['pair'], 'rl', 'full', d['pair'] in covered)),
            ('tail_rule', Path('artifacts/contracts/tail_put25_strategy') / d['pair'] / 'rule/full'),
            ('tail_free_rl', Path('artifacts/contracts/tail_put25_strategy') / d['pair'] / 'rl_sized/full'),
            ('tail_sizing_rl', Path('artifacts/contracts/tail_put25_rule_sizing') / d['pair'] / 'rl_rule_sized/full')]:
            path = folder / 'decisions.csv'
            if path.exists():
                frame = pd.read_csv(record(path)); frame = frame[frame.action.ne(0)]
                decisions.append(frame[['status']].assign(strategy=strategy, pair=d['pair']))
    joined = pd.concat(fits, ignore_index=True)
    activation = joined.groupby(['strategy', 'status']).agg(months=('fit', 'size'), pairs=('pair', 'nunique')).reset_index()
    activation.to_csv(OUT / 'model_activation.csv', index=False)
    bypair = pd.DataFrame(funnel); bypair.to_csv(OUT / 'entry_filters_by_pair.csv', index=False)
    summary = bypair.groupby('strategy').sum(numeric_only=True).reset_index()
    summary.to_csv(OUT / 'entry_filters.csv', index=False)
    inventory = pd.concat(decisions, ignore_index=True).groupby(['strategy', 'status']).size().rename('proposals').reset_index()
    inventory.to_csv(OUT / 'inventory_decisions.csv', index=False)
    extra_path = Path('reports/tail_put25_rule_sizing/fits.csv')
    extra = pd.read_csv(record(extra_path))
    rejected_positive = extra[(extra.status == 'fallback_validation_no_improvement') & extra.validation_sharpe.gt(0)
                             & extra.baseline_validation_sharpe.ge(extra.validation_sharpe-1e-12)
                             & extra.recent_rows.ge(504) & extra.validation_error.isna()]
    rejected_positive.to_csv(OUT / 'positive_models_blocked_by_added_comparison.csv', index=False)
    record(Path(__file__))
    for path, sha in sources.items():
        assert digest(path) == sha, f'Source changed during audit: {path}'
    (OUT / 'audit.json').write_text(json.dumps(dict(status='complete', no_refits=True, no_replays=True,
        sources_verified=len(sources), positive_candidate_months_blocked_only_by_added_rule_comparison=len(rejected_positive),
        affected_pairs=int(rejected_positive.pair.nunique())), indent=2) + '\n')
    (OUT / 'source_hashes.json').write_text(json.dumps(sources, indent=2) + '\n')
    lines = ['Saved ATM versus tail gating audit', '',
        'Counts use the saved evaluation calendars per pair, with the original roster and causal inputs. Pair-days are not independent trades.',
        'ATM and original free-direction tails share monthly positive prior-year validation, 504-row minimum and completed-outcome directional memory gates.',
        'The latest sizing overlay additionally requires outperforming the fixed rule during prior-year validation. A failed sizing activation retains the full-size rule; it prevents RL intervention rather than blocking that rule trade.',
        'Directional memory gate requires positive aggregate modeled completed-cohort P&L and at least 21 observations within91 calendar days and63 within365 days at the $0.65 training hurdle. This can block a quote-valid trade; it is not an exchange or risk requirement.', '',
        activation.to_string(index=False), '', summary.to_string(index=False), '', inventory.to_string(index=False), '',
        f'Positive validation candidate months rejected only by the newly added outperform-rule comparison: {len(rejected_positive)} across {rejected_positive.pair.nunique()} pairs.',
        'These counts demonstrate restrictions and unequal RL activation. They do not estimate the P&L from removing those restrictions. No new training or policy replay was performed.', '']
    (OUT / 'findings.txt').write_text('\n'.join(lines))
    print(activation.to_string(index=False)); print(summary.to_string(index=False))
    print(inventory.to_string(index=False)); print('Added comparison rejects positive candidates:', len(rejected_positive))


if __name__ == '__main__': main()
