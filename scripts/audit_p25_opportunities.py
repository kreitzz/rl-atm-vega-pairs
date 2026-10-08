"""Audit current P25 opportunity rules using saved data; no fitting/replay."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[key] = '1'
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
from scripts.diagnose_p25_pca import Budget, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/p25_strategy_audit/opportunities'
ART = ROOT / 'artifacts/contracts/p25_strategy_audit/opportunities'
SOURCE = ROOT / 'data/submission/artifacts/contracts/p25_feature_search/training_rows.parquet'
ARCHIVE = ROOT / 'data/p25_validation_split_experiment.zip'
REPLAY = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(ART / 'matplotlib')
    guard = Budget()
    started = time.monotonic()
    protocol = dict(baseline='screened_selected;27 inputs;18 months;10m;608 proposals',
        purpose='Trace opportunity rules and reproduce saved screens; no optimization, refit or portfolio replay',
        comparisons='Existing fixed100 screened versus fixed100 ungated at10m; retain2018-2021 and2022-2025 split',
        diagnostics='Gate funnel and pair breadth; reverse-beta relabelling sensitivity without P&L filtering',
        untouched_holdout=False, resources='One numerical worker,20% duty,384MiB RSS',
        limitation='Bundled training rows already satisfy observed/beta/context. Full raw pair-day funnel unavailable locally.')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    sources = {}
    def local(path):
        p = ROOT / path
        sources[path] = digest(p)
        return p
    manifest = json.loads(local('reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    with zipfile.ZipFile(ARCHIVE) as z:
        def member(name):
            raw = z.read(name)
            sha = hashlib.sha256(raw).hexdigest()
            assert sha == manifest[name], name
            sources['data/p25_validation_split_experiment.zip::'+name] = sha
            return io.BytesIO(raw)
        sig = pd.read_parquet(member(REPLAY+'scored_allocations.parquet'))
        dec = pd.read_csv(member(REPLAY+'decisions.csv'), parse_dates=['signal_date','entry_date'])
    periods = pd.read_csv(local('reports/p25_validation_split/periods.csv'))
    cols = ['date','pair','observed','eligible','context','direction','count_91d','count_365d',
            'profit_91d','profit_365d','signal_eligible','beta','correlation','oriented_iv_gap',
            'rank_strength','optionid_a','optionid_b']
    rows = pd.read_parquet(local(str(SOURCE.relative_to(ROOT)).replace('\\','/')), columns=cols)
    assert not rows.duplicated(['date','pair']).any()
    calendar = pd.read_csv(local('reports/p25_memory_ablation/formation_calendar.csv'), parse_dates=['date'])
    keys = pd.MultiIndex.from_frame(calendar[['date','pair']])
    permitted = pd.MultiIndex.from_frame(rows[['date','pair']]).isin(keys)
    date_ok = rows.date.between('2018-01-01','2025-06-30')
    base = rows.observed & rows.eligible & rows.context
    counts = rows.count_91d.ge(21) & rows.count_365d.ge(63)
    profits = rows.profit_91d.gt(0) & rows.profit_365d.gt(0)
    assert (rows.signal_eligible == (base & rows.direction.ne(0) & counts & profits)).all()
    stages = [
        ('Stored observed / beta / context candidate rows', date_ok & base),
        ('Also on inherited formation calendar', date_ok & base & permitted),
        ('Also extreme IV spread rank (ungated opportunities)', date_ok & base & permitted & rows.direction.ne(0)),
        ('Also minimum memory counts21/63', date_ok & base & permitted & rows.direction.ne(0) & counts),
        ('Also positive91/365day memory (current proposals)', date_ok & rows.signal_eligible & permitted)]
    funnel = pd.DataFrame([dict(stage=s, pair_dates=int(m.sum()), pairs=int(rows.loc[m,'pair'].nunique())) for s,m in stages])
    final = rows.loc[stages[-1][1]].copy()
    pd.testing.assert_frame_equal(final[['date','pair','direction','optionid_a','optionid_b']].sort_values(['date','pair']).reset_index(drop=True),
        sig[['date','pair','direction','optionid_a','optionid_b']].sort_values(['date','pair']).reset_index(drop=True),check_dtype=False)
    assert funnel.pair_dates.iloc[-3:].tolist() == [4744,3484,608]
    assert len(dec)==608 and int(dec.status.eq('entered').sum())==207
    funnel.to_csv(OUT/'funnel.csv',index=False)
    dec.groupby('status').agg(proposals=('pair','size'),pairs=('pair','nunique')).reset_index().to_csv(OUT/'admission.csv',index=False)
    final['reverse_beta'] = final.correlation.pow(2)/final.beta
    final['formation_vol_ratio_a_b'] = final.beta/final.correlation
    final['reverse_beta_pass'] = final.reverse_beta.between(.85,1.15)
    final = final.merge(dec[['signal_date','pair','status','action']], left_on=['date','pair'],right_on=['signal_date','pair'],validate='one_to_one')
    final[['date','pair','beta','correlation','reverse_beta','formation_vol_ratio_a_b','reverse_beta_pass','oriented_iv_gap','rank_strength','status','action']].to_csv(OUT/'proposal_structure.csv',index=False)
    diagnostic=[]
    for name, group in [('qualified',final),('entered',final.loc[final.status.eq('entered')])]:
        diagnostic.append(dict(population=name,n=len(group),pairs=int(group.pair.nunique()),reverse_pass=int(group.reverse_beta_pass.sum()),
            reverse_fail=int((~group.reverse_beta_pass).sum()),median_reverse_beta=float(group.reverse_beta.median()),
            long_higher_iv=int(group.oriented_iv_gap.gt(0).sum()),long_lower_iv=int(group.oriented_iv_gap.lt(0).sum())))
    pd.DataFrame(diagnostic).to_csv(OUT/'structural_diagnostics.csv',index=False)
    final.groupby('pair').agg(proposals=('date','size'),reverse_beta_failures=('reverse_beta_pass',lambda x:int((~x).sum())),
        entries=('status',lambda x:int(x.eq('entered').sum()))).reset_index().to_csv(OUT/'pair_coverage.csv',index=False)
    controls = periods.loc[periods.case.isin(['screened_fixed100','ungated_fixed100'])].copy()
    # Verify fixed controls directly from their saved daily books, not just prose.
    nav_s = pd.read_parquet(local('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed100/daily_nav.parquet'))
    cfg_s=json.loads(local('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed100/config.json').read_text())
    cfg_u=json.loads(local('data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json').read_text())
    assert cfg_s==cfg_u and cfg_s['capital']==10_000_000
    memory_manifest = json.loads(local('reports/p25_memory_ablation/bundle_manifest.json').read_text())['files']
    with zipfile.ZipFile(ROOT/'data/p25_memory_ablation_experiment.zip') as z:
        name='artifacts/contracts/p25_memory_ablation/replays/no_memory_gate_fixed100/daily_nav.parquet'
        raw=z.read(name)
        sha=hashlib.sha256(raw).hexdigest()
        assert sha==memory_manifest[name]
        sources['data/p25_memory_ablation_experiment.zip::'+name]=sha
        nav_u=pd.read_parquet(io.BytesIO(raw))
        name='artifacts/contracts/p25_memory_ablation/baseline_replay_check.json'
        raw=z.read(name)
        assert hashlib.sha256(raw).hexdigest()==memory_manifest[name]
        sources['data/p25_memory_ablation_experiment.zip::'+name]=memory_manifest[name]
        assert json.loads(raw)['exact_nav_trades_decisions']
    assert nav_s.index.equals(nav_u.index)
    np.testing.assert_allclose(nav_s.cash_benchmark,nav_u.cash_benchmark,rtol=0,atol=1e-8)
    for case,nav in [('screened_fixed100',nav_s),('ungated_fixed100',nav_u)]:
        for period,start,stop in [('validation','2018-01-01','2021-12-31'),('retrospective_holdout','2022-01-01','2025-08-29'),('full_history','2018-01-01','2025-08-29')]:
            r=nav.loc[start:stop,'excess_return']
            if start=='2018-01-01':r=r.iloc[1:]
            observed=float(np.sqrt(252)*r.mean()/r.std(ddof=1))
            expected=controls.loc[controls.case.eq(case)&controls.period.eq(period),'annualized_excess_sharpe'].iloc[0]
            assert abs(observed-expected)<1e-12
    controls.to_csv(OUT/'fixed_size_gate_control.csv',index=False)
    roster=pd.read_csv(local('reports/atm_nn_cost_hurdles/matched_roster.csv'))
    prior=set(json.loads(local('reports/expiry_trained_beta_085_115/matched_pairs.json').read_text()))
    vix=set(json.loads(local('reports/expiry_vix_momentum/matched_pairs.json').read_text()))
    assert set(roster.pair)==vix
    roster_rows=[dict(stage='Prior expiry/beta common-complete roster',pairs=len(prior),removed_from_previous=''),
        dict(stage='VIX/momentum common-complete roster',pairs=len(vix),removed_from_previous=';'.join(sorted(prior-vix))),
        dict(stage='NN cost hurdle common-complete roster inherited by P25',pairs=int(roster.included.sum()),removed_from_previous=';'.join(roster.loc[~roster.included,'label']))]
    pd.DataFrame(roster_rows).to_csv(OUT/'roster_history.csv',index=False)
    guard.rest(started)
    started = time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    short=funnel.iloc[-3:]
    labels=['Extreme spread rank','+ History counts','+ Positive history']
    axes[0].barh(labels,short.pair_dates,color=['#7c9ac1','#4f78a8','#235789'])
    for i,r in enumerate(short.itertuples()): axes[0].text(r.pair_dates+55,i,f'{r.pair_dates:,} / {r.pairs} pairs',va='center')
    axes[0].invert_yaxis();axes[0].set_xlim(0,6700);axes[0].set_xlabel('Qualified proposals');axes[0].set_title('Memory screens remove 87% of candidates')
    for i,(period,label) in enumerate([('validation','2018–2021'),('retrospective_holdout','2022–Aug 2025')]):
        values=controls.loc[controls.period.eq(period)].set_index('case').annualized_excess_sharpe
        axes[1].bar(np.array([0,1])+i*.35,[values['screened_fixed100'],values['ungated_fixed100']],.35,label=label)
    axes[1].set_xticks([.175,1.175],['Memory screened','No memory screen'])
    axes[1].set_title('Identical 100% sizing; $10m continuous books');axes[1].set_ylabel('Annualized daily excess Sharpe');axes[1].legend()
    fig.savefig(OUT/'opportunity_audit.png',dpi=150);plt.close(fig)
    for name in ['scripts/audit_p25_opportunities.py','config/p25_validation_2018_2021.json','config/broad_pair_research.json',
        'scripts/build_broad_pairs.py','scripts/run_broad_contract_replay.py','scripts/run_atm65_employee_sentiment.py',
        'scripts/contract_learning_labels.py','scripts/same_structure_smile.py','scripts/uniform_smile_data.py',
        'scripts/uniform_smile_policy.py','scripts/retrain_atm_commission.py','scripts/smile_contract_selection.py',
        'scripts/uniform_smile_engine.py','reports/atm_nn_cost_hurdles.md','reports/atm_nn_cost_hurdles/matched_roster.csv',
        'reports/p25_pair_profiles/findings.txt','reports/p25_maturity_diagnostic/findings.md',
        'scripts/report_expiry_vix_momentum.py','scripts/download_matched_puts.py','scripts/train_expiry_contracts.py',
        'reports/expiry_vix_momentum.md','reports/expiry_trained_beta_085_115.md',
        'scripts/run_p25_memory_ablation.py','scripts/run_p25_capital_sensitivity.py',
        'scripts/run_same_structure_calendar.py','scripts/analyze_tail_legacy_calendar.py','scripts/run_tail_put25_strategy.py']:
        local(name)
    guard.check()
    audit=dict(source_hashes=sources,exact608proposal_identity=True,exact207admissions=True,gate_rule_matches_all_stored_rows=True,
        count_funnel_reproduced=True,fixed100_sharpes_verified_against_daily_books=True,
        fixed100_execution_configs_equal=True,expanded_market_baseline_exact_archive_check=True,
        raw_stored_rows=len(rows),all_rows_already_base=bool(base.all()),
        diagnostics=diagnostic,peak_rss_mib=guard.peak, no_strategy_changes=True)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(funnel.to_string(index=False))
    print(json.dumps(diagnostic,indent=2))
    print('Peak RSS MiB:',guard.peak)
    guard.rest(started)


if __name__=='__main__':main()
