"""Verify the saved baseline and summarize the sentiment feature ablation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_atm65_employee_sentiment import OUT, REPORT, ADDED
from scripts.run_atm65_ablation import CURVE, NN, read_actions, write_json
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits


def main():
    manifest = json.loads((OUT/'inputs.json').read_text())
    for path, expected in manifest['code_and_sources'].items():
        assert digest(path) == expected, f'Source changed: {path}'
    coverage = pd.read_csv(REPORT/'coverage.csv')
    rows = []; fits = []
    for d in manifest['definitions']:
        if not coverage.set_index('pair').loc[d['pair'], 'retrain']: continue
        folder = OUT/d['pair']; panel = pd.read_parquet(folder/'states.parquet')
        original = pd.read_parquet(NN/d['pair']/'fee_065/states.parquet')
        pd.testing.assert_frame_equal(panel[original.columns], original)
        table = pd.read_csv(folder/'fits.csv'); check_fits(table)
        assert table.rl_actions_reproduced.all() and table.nn_enabled_refits_reproduced.all()
        fits.append(table.assign(pair=d['pair']))
        saved = read_actions(CURVE/d['pair']/'atm_control/actions.csv')
        np.testing.assert_array_equal(read_actions(folder/'rl_control/actions.csv'), saved)
        for model in ['rl_sentiment', 'nn_sentiment']:
            baseline = saved if model == 'rl_sentiment' else read_actions(NN/d['pair']/'fee_065/actions.csv')
            candidate = read_actions(folder/model/'actions.csv')
            assert baseline.index.equals(candidate.index)
            status = json.loads((folder/model/'full/status.json').read_text())
            row = dict(pair=d['pair'], label=d['label'], model=model,
                baseline_nonzero_recommendations=int(baseline.ne(0).sum()),
                candidate_nonzero_recommendations=int(candidate.ne(0).sum()),
                changed_recommendations=int(candidate.ne(baseline).sum()), entries=status.get('entries'),
                entries_with_reviews=0, entries_without_reviews=0)
            path = folder/model/'full/decisions.csv'
            if path.exists():
                decisions = pd.read_csv(path, parse_dates=['signal_date', 'entry_date'])
                entered = decisions[decisions.status.eq('entered')].copy()
                assert (entered.entry_date > entered.signal_date).all()
                state = panel.reindex(pd.DatetimeIndex(entered.signal_date))
                valid = state[ADDED[1]].eq(1).to_numpy()
                row.update(entries_with_reviews=int(valid.sum()), entries_without_reviews=int((~valid).sum()))
                for side, keep in [('long', entered.action.gt(0).to_numpy()), ('short', entered.action.lt(0).to_numpy())]:
                    gap = state.loc[valid & keep, ADDED[0]]
                    row['mean_review_gap_when_'+side] = float(gap.mean()) if len(gap) else None
            rows.append(row)
    changes = pd.DataFrame(rows); changes.to_csv(REPORT/'policy_changes.csv', index=False)
    all_fits = pd.concat(fits, ignore_index=True); all_fits.to_csv(REPORT/'fits.csv', index=False)
    accounting = json.loads((REPORT/'accounting_coverage.json').read_text())
    report = dict(source_files_verified=len(manifest['code_and_sources']),
        baseline_action_files_unchanged=sum(path.endswith('/actions.csv') for path in manifest['code_and_sources']),
        refitted_pairs=int(coverage.retrain.sum()), monthly_controls_verified=len(all_fits),
        original_state_columns_unchanged=True, training_outcomes_strictly_purged=True,
        current_review_vintage_not_archived=True, accounting=accounting,
        entries_by_model=changes.groupby('model')[['entries', 'entries_with_reviews', 'entries_without_reviews']].sum().to_dict('index'))
    write_json(REPORT/'audit.json', report)
    pairs = pd.read_csv(REPORT/'pairs.csv'); pairs = pairs[(pairs.universe == 'all') & (pairs.period == 'full')]
    deltas = []
    for kind in ['rl', 'nn']:
        b = pairs[pairs.model.eq(kind+'_baseline')].set_index('pair')
        n = pairs[pairs.model.eq(kind+'_sentiment')].set_index('pair')
        own = pd.DataFrame({'label': n.label, 'pnl_change': n.net_excess_pnl-b.net_excess_pnl,
                            'entry_change': n.entries-b.entries}).reset_index()
        own['model'] = kind; deltas.append(own)
    delta = pd.concat(deltas, ignore_index=True); delta.to_csv(REPORT/'pair_changes.csv', index=False)
    path = REPORT.with_suffix('.md')
    lines = ['', '## Verification and trade coverage', '',
        f'{report["baseline_action_files_unchanged"]} baseline action files and all snapshotted inputs retain their SHA-256 hashes. '
        f'{len(all_fits)} monthly control rows reproduced the saved RL actions; enabled original NN refits also reproduced their saved decisions. '
        'Original state values and completed-outcome cutoffs passed the audit.', '',
        '| Candidate on refitted sleeves | Entries with reviews | Entries without reviews |', '|---|---:|---:|']
    for model, counts in report['entries_by_model'].items():
        lines.append(f'| {model} | {int(counts["entries_with_reviews"])} | {int(counts["entries_without_reviews"])} |')
    lines += ['', 'These counts refer to executed entries in the full-period book of the 28 refitted sleeves; untouched sleeves retain their original policies. '
              'Feature addition also exposes missing-data availability and can change tree partitions and validation decisions, so a performance change cannot automatically be attributed to leadership ratings alone.', '',
              '[Policy changes](atm65_employee_sentiment/policy_changes.csv) · [Pair P&L changes](atm65_employee_sentiment/pair_changes.csv) · [Audit](atm65_employee_sentiment/audit.json)', '']
    path.write_text(path.read_text()+'\n'.join(lines))
    print(json.dumps(report, indent=2), flush=True)
    print(delta[delta.model.eq('rl')].sort_values('pnl_change').iloc[np.r_[0:3, -3:0]].to_string(index=False), flush=True)


if __name__ == '__main__': main()
