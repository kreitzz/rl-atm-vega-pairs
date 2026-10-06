"""All long-tenor experiments, selection diagnostics, and exploratory blends."""
from itertools import combinations
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

from scripts.run_long_tenor_policy_search import digest, dump, protected, LONG
from scripts import run_long_tenor_policy_search as base

REPORT = Path('reports/long_tenor_search_summary')
FAMILIES = ['long_tenor_policy_search','long_tenor_full_universe','long_tenor_flexible_actions']


def sharpe(r):
    return LONG.base.sharpe(r)


def optimal_nonnegative_weights(returns):
    """Small active-set tangency search; explicitly retrospective when full sample."""
    mean, cov = returns.mean().to_numpy(), returns.cov().to_numpy()
    count = len(mean)
    best_score, best_weights = -np.inf, None
    for size in range(1, count+1):
        for members in combinations(range(count), size):
            idx = list(members)
            try:
                raw = np.linalg.solve(cov[np.ix_(idx, idx)], mean[idx])
            except np.linalg.LinAlgError:
                continue
            if (raw < -1e-10).any() or raw.sum() <= 0:
                continue
            weights = np.zeros(count)
            weights[idx] = np.maximum(raw, 0)/np.maximum(raw, 0).sum()
            score = sharpe(returns @ weights)
            if score > best_score:
                best_score, best_weights = score, weights
    assert best_weights is not None and np.isclose(best_weights.sum(), 1)
    assert best_score >= max(sharpe(returns[c]) for c in returns)-1e-10
    return best_weights


def paired_bootstrap(a, b, replicates=1000, block=126):
    rng = np.random.default_rng(2026)
    n = len(a)
    gaps = []
    for _ in range(replicates):
        starts = rng.integers(n, size=int(np.ceil(n/block)))
        idx = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        gaps.append(sharpe(a.iloc[idx])-sharpe(b.iloc[idx]))
    return np.quantile(gaps, [.025, .975]).tolist()


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    protected()
    # Refresh method descriptions and reproducibility hashes after the final
    # reporting/test changes. This does not rerun or mutate any saved replay.
    base.configure()
    for family in FAMILIES:
        for instrument in ['P25', 'C75']:
            base.INSTRUMENT = instrument
            base.OUT = Path('artifacts/contracts', family, instrument)
            base.REPORT = Path('reports', family, instrument)
            base.summarize()
    rows, books, annual = [], {}, []
    for family in FAMILIES:
        for instrument in ['P25','C75']:
            for path in sorted(Path('artifacts/contracts', family, instrument).glob('*D/replays/*/status.json')):
                record = json.loads(path.read_text())
                record.update(family=family, instrument=instrument,
                    key=f'{family}/{instrument}/{record["tenor"]}D/{record["case"]}', source_folder=str(path.parent))
                nav = pd.read_parquet(path.parent/'daily_nav.parquet')
                r = nav.excess_return.iloc[1:]
                assert np.isclose(sharpe(r), record['annualized_excess_sharpe'], atol=1e-12)
                record['formation_2018_2023_sharpe'] = sharpe(r.loc[:'2023-12-31'])
                record['recent_2024_partial2025_sharpe'] = sharpe(r.loc['2024-01-01':])
                record['post_warmup_2021_partial2025_sharpe'] = sharpe(r.loc['2021-01-01':])
                books[record['key']] = r
                rows.append(record)
                for year, use in r.groupby(r.index.year):
                    annual.append(dict(key=record['key'], year=int(year), annualized_excess_sharpe=sharpe(use)))
    table = pd.DataFrame(rows)
    table.to_csv(REPORT/'all_variants.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv', index=False)
    selections = []
    for instrument in ['P25','C75']:
        for tenor in [180, 365]:
            available = table.loc[table.instrument.eq(instrument)&table.tenor.eq(tenor)]
            learned = available.loc[available.case.str.startswith(('native_', 'flexible_'))]
            fixed = available.loc[available.family.eq(FAMILIES[0])&available.case.isin(['fixed_50_hold','fixed_100_hold'])]
            best_fixed = fixed.loc[fixed.annualized_excess_sharpe.idxmax()]
            for selection, criterion in [('full_history_exploratory','annualized_excess_sharpe'),
                                          ('2018_2023_formation','formation_2018_2023_sharpe')]:
                chosen = learned.loc[learned[criterion].idxmax()]
                baseline = best_fixed if selection == 'full_history_exploratory' else fixed.loc[fixed[criterion].idxmax()]
                a, b = books[chosen.key], books[baseline.key]
                lo, hi = paired_bootstrap(a, b)
                selections.append(dict(instrument=instrument, tenor=tenor, selection=selection,
                    learned_key=chosen.key, learned_sharpe=sharpe(a), learned_entries=int(chosen.entries),
                    fixed_key=baseline.key, fixed_sharpe=sharpe(b), fixed_entries=int(baseline.entries),
                    sharpe_gain=sharpe(a)-sharpe(b),
                    unadjusted_bootstrap_gain_95_low=lo, unadjusted_bootstrap_gain_95_high=hi,
                    learned_recent_sharpe=chosen.recent_2024_partial2025_sharpe,
                    fixed_recent_sharpe=baseline.recent_2024_partial2025_sharpe))
    selected = pd.DataFrame(selections)
    selected.to_csv(REPORT/'selected_comparison.csv', index=False)

    core_learned = pd.read_parquet('artifacts/contracts/p25_long_management/replays/30D_manage_36M/daily_nav.parquet').excess_return.iloc[1:]
    core_fixeds = {level:pd.read_parquet(f'artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed{level}/daily_nav.parquet').excess_return.iloc[1:] for level in [50,100]}
    core_fixed = max(core_fixeds.values(), key=sharpe)
    core_sizing = pd.read_parquet('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/daily_nav.parquet').excess_return.iloc[1:]
    core_rows = []
    for name, r in [('fixed_50',core_fixeds[50]), ('fixed_100',core_fixeds[100]),
                    ('learned_sizing_hold',core_sizing), ('learned_sizing_managed',core_learned)]:
        lo, hi = paired_bootstrap(r, core_fixed)
        core_rows.append(dict(case=name, annualized_excess_sharpe=sharpe(r),
            gain_over_best_fixed=sharpe(r)-sharpe(core_fixed),
            unadjusted_bootstrap_gain_95_low=lo, unadjusted_bootstrap_gain_95_high=hi))
    pd.DataFrame(core_rows).to_csv(REPORT/'core_comparison.csv', index=False)
    blends = []
    for selection in ['full_history_exploratory','2018_2023_formation']:
        learn, fix = {'30D':core_learned}, {'30D':core_fixed}
        chosen_rows = selected.loc[selected.selection.eq(selection)&selected.instrument.eq('P25')]
        for row in chosen_rows.itertuples():
            learn[f'{row.tenor}D'] = books[row.learned_key]
            fix[f'{row.tenor}D'] = books[row.fixed_key]
        learn, fix = pd.DataFrame(learn), pd.DataFrame(fix)
        pd.testing.assert_index_equal(learn.index, fix.index)
        sample = learn if selection == 'full_history_exploratory' else learn.loc[:'2023-12-31']
        fixed_sample = fix if selection == 'full_history_exploratory' else fix.loc[:'2023-12-31']
        for allocation, weights in [('80_10_10', np.array([.8,.1,.1])),
                                    ('optimized', optimal_nonnegative_weights(sample))]:
            a, b = learn@weights, fix@weights
            lo, hi = paired_bootstrap(a, b)
            blends.append(dict(selection=selection, allocation=allocation,
                weight_30D=weights[0], weight_180D=weights[1], weight_365D=weights[2],
                learned_sharpe=sharpe(a), fixed_same_weights_sharpe=sharpe(b),
                sharpe_gain=sharpe(a)-sharpe(b),
                learned_recent_sharpe=sharpe(a.loc['2024-01-01':]),
                fixed_recent_sharpe=sharpe(b.loc['2024-01-01':]),
                unadjusted_bootstrap_gain_95_low=lo, unadjusted_bootstrap_gain_95_high=hi))
            pd.DataFrame(dict(learned=a,fixed_same_weights=b)).to_csv(REPORT/f'blend_{selection}_{allocation}_daily.csv')
        fixed_weights = optimal_nonnegative_weights(fixed_sample)
        blends.append(dict(selection=selection, allocation='fixed_independently_optimized',
            weight_30D=fixed_weights[0], weight_180D=fixed_weights[1], weight_365D=fixed_weights[2],
            fixed_same_weights_sharpe=sharpe(fix@fixed_weights)))
    blend_table = pd.DataFrame(blends)
    blend_table.to_csv(REPORT/'put_portfolio_blends.csv', index=False)
    # Strongest available long sleeves can be fixed. Keep this distinct from
    # an assertion that long-expiry RL beats fixed sizing.
    panel = pd.DataFrame({'30D':core_learned})
    components = {'30D':'managed original 30D checkpoint'}
    for tenor in [180,365]:
        candidates = table.loc[table.instrument.eq('P25')&table.tenor.eq(tenor)]
        chosen = candidates.loc[candidates.annualized_excess_sharpe.idxmax()]
        panel[f'{tenor}D'] = books[chosen.key]
        components[f'{tenor}D'] = chosen.key
    weights = optimal_nonnegative_weights(panel)
    best_return = panel@weights
    fixed_panel = pd.DataFrame({'30D':core_fixed})
    for tenor in [180,365]:
        available = table.loc[table.instrument.eq('P25')&table.tenor.eq(tenor)&table.family.eq(FAMILIES[0])&table.case.isin(['fixed_50_hold','fixed_100_hold'])]
        chosen = available.loc[available.annualized_excess_sharpe.idxmax()]
        fixed_panel[f'{tenor}D'] = books[chosen.key]
    fixed_weights = optimal_nonnegative_weights(fixed_panel)
    fixed_return = fixed_panel@fixed_weights
    lo, hi = paired_bootstrap(best_return, fixed_return)
    dump(REPORT/'best_available_put_blend.json', dict(
        annualized_excess_sharpe=sharpe(best_return),
        weights=dict(zip(panel.columns, weights)), components=components,
        fixed_same_weights_sharpe=sharpe(fixed_panel@weights),
        independently_optimized_fixed_weights=dict(zip(fixed_panel.columns,fixed_weights)),
        independently_optimized_fixed_sharpe=sharpe(fixed_return),
        gain_over_independently_optimized_fixed=sharpe(best_return)-sharpe(fixed_return),
        unadjusted_bootstrap_gain_95_low=lo, unadjusted_bootstrap_gain_95_high=hi,
        full_history_optimized=True, shared_ledger_replayed=False))
    pd.DataFrame(dict(optimized_mixed=best_return,optimized_fixed=fixed_return,
                     fixed_same_weights=fixed_panel@weights)).to_csv(REPORT/'best_available_put_blend_daily.csv')
    best_years = []
    for year, r in best_return.groupby(best_return.index.year):
        fixed_year = fixed_return.loc[r.index]
        best_years.append(dict(year=int(year), sessions=len(r),
            optimized_mixed_sharpe=sharpe(r), optimized_fixed_sharpe=sharpe(fixed_year),
            partial_year=year==2025))
    pd.DataFrame(best_years).to_csv(REPORT/'best_available_put_blend_annual.csv', index=False)
    panel.corr().to_csv(REPORT/'best_available_put_correlations.csv')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_rows = selected.loc[selected.selection.eq('full_history_exploratory')]
    labels = ['30D puts']+[f'{int(r.tenor)}D {"puts" if r.instrument=="P25" else "calls"}' for r in plot_rows.itertuples()]
    learned_values = [sharpe(core_learned)]+plot_rows.learned_sharpe.tolist()
    fixed_values = [sharpe(core_fixed)]+plot_rows.fixed_sharpe.tolist()
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(labels))
    ax.bar(x-.18, fixed_values, width=.36, label='Best fixed size',color='#94a3b8')
    ax.bar(x+.18, learned_values, width=.36, label='Best learned sizing/management',color='#2563eb')
    ax.set_xticks(x, labels)
    ax.set_ylabel('Annualized daily excess Sharpe')
    ax.axhline(0,color='black',linewidth=.6)
    ax.set_title('Learned actions help the 30-day core; long-tenor gains are limited')
    ax.legend()
    fig.text(.5,.01,'2018–partial 2025. Best variants selected using all observed history; no fresh holdout.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(REPORT/'sleeve_comparison.png',dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    years = pd.DataFrame(best_years)
    x = np.arange(len(years))
    ax.bar(x-.18,years.optimized_fixed_sharpe,width=.36,label='Independently optimized fixed portfolio',color='#94a3b8')
    ax.bar(x+.18,years.optimized_mixed_sharpe,width=.36,label='Learned 30D core + fixed long sleeves',color='#2563eb')
    ax.set_xticks(x,[str(y)+('*' if y==2025 else '') for y in years.year])
    ax.axhline(0,color='black',linewidth=.6)
    ax.set_ylabel('Annualized daily excess Sharpe')
    ax.set_title(f'Full-history optimized blend: {sharpe(best_return):.3f} versus {sharpe(fixed_return):.3f}')
    ax.legend(fontsize=9)
    fig.text(.5,.01,'Saved-return blend after component costs; no joint portfolio replay. *2025 ends August 29.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(REPORT/'optimized_blend_by_year.png',dpi=160)
    plt.close(fig)

    source_hashes = {}
    for family in FAMILIES:
        for path in Path('artifacts/contracts', family).glob('*/protected_sources.json'):
            for name, want in json.loads(path.read_text()).items():
                assert digest(name) == want, name
        for path in Path('artifacts/contracts', family).glob('*/**/*'):
            if path.is_file() and path.suffix in ['.json','.joblib','.parquet']:
                source_hashes[str(path)] = digest(path)
    for path in [Path(__file__),Path('tests/test_long_tenor_policy_search.py'),Path('scripts/run_signal_discovery_guarded.py'),
                 Path('scripts/run_long_tenor_policy_search.py'),Path('scripts/run_long_tenor_full_universe.py'),
                 Path('scripts/run_long_tenor_flexible_actions.py')]:
        source_hashes[str(path)] = digest(path)
    dump(REPORT/'source_manifest.json', source_hashes)
    dump(REPORT/'audit.json', dict(all_original_checkpoints_unchanged=True,
        actual_return_sharpes_reproduced=True, variants=len(table),
        source_hashes=len(source_hashes), allocation_optimization='Retrospective full-history and previously inspected formation periods reported separately',
        bootstrap='126-session circular paired blocks, 1000 repetitions, seed2026; not adjusted for model or allocation selection',
        no_fresh_holdout=True, shared_portfolio_blends_replayed=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (REPORT/'findings.txt').write_text(
        'LONG-EXPIRY RL POLICY SEARCH\n\n'
        'Compare same frozen pair/date/direction proposals and actual matched-type contracts, shared cash, daily delta hedge, entry vega neutrality and unchanged execution/risk conventions. Every native entry proposal remains an explicit decision. P25 and same-strike approximately C75 trained separately.\n'
        'Three declared search families: (1) 24/48-month quarterly native entry/exit fits using original executed cohorts and four-leaf policies; (2) all remaining native executable opportunities independently labeled, 252-session EWMA variance forecast, two-leaf regularization and 2x training elective-option-exit penalty; (3) same broad labels/signals with four leaves and learned entry actions 0/25/50/75/100%, plus identical learned-skip fixed-size controls. Management always irreversible hold/half/close, weekly checks and next-session actual observed fills.\n'
        'Training uses only fully completed original scheduled expiries, minimum16 episodes and original cold-start sizes/hold. Compact episodic direct policy optimization learns action maps from realized rewards; not neural Q-learning. Shadow training uses fractional unnetted hedge/borrow costs; all final single-tenor tests execute integer quantities and reconcile independent cash flows.\n'
        'Full-history strongest variants/allocations are hyperoptimized exploratory results. A separate 2018-2023-selected comparison shows 2024-partial2025 behavior, but those dates were previously inspected and are not a fresh holdout. Bootstrap intervals do not adjust for this search. Saved-return portfolio blends include component costs, but not joint hedge netting or shared admission/integer allocations. American exercise/assignment and long-term expected-dividend/BS hedge-model limitations remain, especially for deep calls.\n\n'
        +'30D CORE: SIZING AND MANAGEMENT INCREMENTS\n'+pd.DataFrame(core_rows).to_string(index=False)+'\n\n'
        +selected.to_string(index=False)+'\n\nPUT PORTFOLIO BLENDS\n'+blend_table.to_string(index=False)+'\n\n'
        +'STRONGEST AVAILABLE PUT BLEND\n'+(REPORT/'best_available_put_blend.json').read_text()+'\n\n'
        +'YEARLY STRONGEST-BLEND COMPARISON\n'+pd.DataFrame(best_years).to_string(index=False)+'\n\n'
        +'Interpretation: the observed incremental learning benefit is concentrated in the original 30D core. Best learned put controllers do not beat the strongest unconditional fixed sizing at 180D/365D. Long call improvements are small in absolute Sharpe. The strongest mixed portfolio uses learned 30D sizing/management plus fixed-size longer-tenor puts. Its interval against independently optimized fixed allocation includes zero; it is not conclusive statistical evidence after model/allocation selection.\n'
        +'Charts: sleeve_comparison.png and optimized_blend_by_year.png. All sharpes annualize daily cash-excess returns with sqrt(252); 2025 is partial.\n')
    print(selected.to_string(index=False))
    print(blend_table.to_string(index=False))
    print((REPORT/'best_available_put_blend.json').read_text())


if __name__ == '__main__':
    main()
