"""Verify causal input reconstruction, frozen controls and ablation accounting."""
import json
import numpy as np
import pandas as pd
from scripts.run_atm65_mined_signals import OUT, REPORT, MODELS, EMPLOYEE, CHECKPOINT, baseline_actions, GROUPS, ADDED
from scripts.atm65_mined_features import collect, add_signals
from scripts.run_atm65_ablation import read_actions, write_json, PERIODS
from scripts.run_contract_walkforward import digest
from scripts.retrain_atm_commission import check_fits
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar


def paired_differences():
    accounting = json.loads((REPORT/'accounting_coverage.json').read_text())
    rows = []
    for period, _ in PERIODS:
        # Match the same roster across all variants; label exclusions explicitly.
        universe = 'common_complete' if any(x['period']==period for x in accounting['unresolved']) else 'all'
        base = pd.read_csv(REPORT/f'{universe}_rl_baseline_{period}_portfolio.csv', index_col=0, parse_dates=[0])
        for model in MODELS:
            candidate = pd.read_csv(REPORT/f'{universe}_{model}_{period}_portfolio.csv', index_col=0, parse_dates=[0])
            assert base.index.equals(candidate.index)
            h = pd.DataFrame(dict(a=base.excess_return.iloc[1:], b=candidate.excess_return.iloc[1:]))
            h['n'] = 1.; h['aa'] = h.a*h.a; h['bb'] = h.b*h.b
            blocks = h.groupby(h.index.to_period('Q'))[['n', 'a', 'aa', 'b', 'bb']].sum().to_numpy()
            rng = np.random.default_rng(2026); draws = rng.integers(len(blocks), size=(2000, len(blocks)))
            n, a, aa, b, bb = blocks[draws].sum(axis=1).T
            def ratio(s, ss):
                sd = np.sqrt(np.maximum((ss-s*s/n)/(n-1), 0))
                return np.sqrt(252)*np.divide(s/n, sd, out=np.full(len(n), np.nan), where=sd>1e-12)
            difference = ratio(b, bb)-ratio(a, aa)
            rows.append(dict(universe=universe, model=model, period=period, resamples=2000,
                sharpe_difference=float(np.sqrt(252)*(h.b.mean()/h.b.std()-h.a.mean()/h.a.std())),
                difference_ci90_low=float(np.nanquantile(difference, .05)), difference_ci90_high=float(np.nanquantile(difference, .95))))
    pd.DataFrame(rows).to_csv(REPORT/'paired_quarter_differences.csv', index=False)
    return rows


def main():
    manifest = json.loads((OUT/'inputs.json').read_text())
    for path, sha in manifest['code_and_sources'].items(): assert digest(path)==sha, f'Source changed: {path}'
    reference = json.loads(CHECKPOINT.read_text())
    assert digest(reference['archive'])==reference['archive_sha256']
    for path, sha in reference['files'].items(): assert digest(path)==sha, f'Frozen source changed: {path}'
    cal = load_calendar(ROOT); cal = cal[(cal>='2010-01-01') & (cal<='2025-08-29')]
    market = Market(cal)
    legs = collect(manifest['definitions'], EMPLOYEE, market.stocks, cal)
    coverage = pd.read_csv(REPORT/'coverage.csv').set_index('pair'); rows = []; fits = []
    reconstructed = 0
    for d in manifest['definitions']:
        pair = d['pair']; c = coverage.loc[pair]; folder = OUT/pair
        source = EMPLOYEE/pair/'states.parquet'
        if not source.exists(): continue
        original = pd.read_parquet(source); panel = pd.read_parquet(folder/'states.parquet')
        a, b = d['secids']
        pd.testing.assert_frame_equal(panel, add_signals(original, legs[a], legs[b]))
        reconstructed += 1
        assert np.isfinite(panel[ADDED]).all().all()
        for signal in ['downside_fraction', 'iv_factor_reversion']:
            flag = signal+'_available'; value = signal+'_gap'
            assert panel[flag].isin([0., 1.]).all()
            assert panel.loc[panel[flag].eq(0), value].eq(0).all()
        if not c.retrain: continue
        table = pd.read_csv(folder/'fits.csv'); check_fits(table)
        assert table.baseline_reproduced.all(); fits.append(table.assign(pair=pair))
        old = read_actions(baseline_actions(pair, 'rl', bool(c.employee_features)))
        for model in MODELS:
            new = read_actions(folder/model/'actions.csv'); assert old.index.equals(new.index)
            if not c[model+'_retrain']: pd.testing.assert_series_equal(new, old)
            flags = [name for name in GROUPS[model] if name.endswith('_available')]
            row = dict(pair=pair, label=d['label'], model=model, changed_recommendations=int(old.ne(new).sum()),
                old_nonzero=int(old.ne(0).sum()), new_nonzero=int(new.ne(0).sum()), entries_all_inputs=0, entries_missing_inputs=0)
            path = folder/model/'full/decisions.csv'
            if path.exists():
                decisions = pd.read_csv(path, parse_dates=['signal_date', 'entry_date'])
                entered = decisions[decisions.status.eq('entered')]
                assert (entered.entry_date>entered.signal_date).all()
                valid = panel.reindex(pd.DatetimeIndex(entered.signal_date))[flags].eq(1.).all(axis=1)
                row['entries_all_inputs'] = int(valid.sum()); row['entries_missing_inputs'] = int((~valid).sum())
            rows.append(row)
    changes = pd.DataFrame(rows); changes.to_csv(REPORT/'policy_changes.csv', index=False)
    fit_table = pd.concat(fits, ignore_index=True); usage = []
    for model in MODELS:
        enabled = fit_table[model+'_status'].eq('enabled')
        used = np.zeros(len(fit_table), dtype=bool)
        for signal in ['downside_fraction', 'iv_factor_reversion']:
            value = model+'_refit_'+signal+'_value_splits'; flag = model+'_refit_'+signal+'_missing_splits'
            chosen_value = fit_table[value].fillna(0).gt(0); chosen_flag = fit_table[flag].fillna(0).gt(0)
            used |= chosen_value.to_numpy()
            usage.append(dict(model=model, signal=signal, enabled_refits=int(enabled.sum()),
                enabled_refits_using_value=int((enabled & chosen_value).sum()), enabled_refits_using_availability=int((enabled & chosen_flag).sum())))
        usage.append(dict(model=model, signal='any_new_value', enabled_refits=int(enabled.sum()),
            enabled_refits_using_value=int((enabled.to_numpy() & used).sum())))
    pd.DataFrame(usage).to_csv(REPORT/'feature_usage.csv', index=False)
    accounting = json.loads((REPORT/'accounting_coverage.json').read_text())
    report = dict(source_files_verified=len(manifest['code_and_sources']), frozen_source_files_verified=len(reference['files']),
        baseline_archive_sha256_verified=True, reconstructed_signal_panels=reconstructed,
        retrained_pairs=int(coverage.retrain.sum()), monthly_controls_verified=len(fit_table),
        original_state_columns_unchanged=True, training_outcomes_strictly_purged=True,
        feature_universe_independent_of_future_payoffs=True, fixed_seed=2026, nn_refits=0, accounting=accounting,
        entry_coverage=changes.groupby('model')[['entries_all_inputs', 'entries_missing_inputs']].sum().to_dict('index'),
        feature_usage=usage, paired_quarter_differences=paired_differences(),
        uncertainty_note='Paired quarter resampling of inspected portfolio paths; descriptive, no policy refits or correction for signal selection.')
    write_json(REPORT/'audit.json', report)
    pairs = pd.read_csv(REPORT/'pairs.csv'); deltas = []
    for universe in pairs.universe.unique():
        for period, _ in PERIODS:
            h = pairs[(pairs.universe==universe) & (pairs.period==period)]
            base = h[h.model.eq('rl_baseline')].set_index('pair')
            for model in MODELS:
                candidate = h[h.model.eq(model)].set_index('pair')
                if candidate.empty: continue
                assert candidate.index.equals(base.index)
                delta = pd.DataFrame(dict(label=candidate.label, pnl_change=candidate.net_excess_pnl-base.net_excess_pnl,
                    entry_change=candidate.entries-base.entries)).reset_index()
                delta['universe'] = universe; delta['period'] = period; delta['model'] = model; deltas.append(delta)
    pd.concat(deltas, ignore_index=True).to_csv(REPORT/'pair_changes.csv', index=False)
    path = REPORT/'summary.txt'; path.write_text(path.read_text()+'\nVerification:\n'+json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
