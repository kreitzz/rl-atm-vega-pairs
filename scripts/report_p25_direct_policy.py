"""Report the frozen matched-objective experiment without selecting a new policy."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
import json
import time
import numpy as np
import pandas as pd
from scripts import run_p25_direct_policy as run
from scripts.report_p25_additive_sizing import summarize_stability

NAMES={'tree_cem':'Tree + CEM','fixed100':'Fixed 100%',
       'regression_linear':'Regression linear','regression_additive':'Regression additive',
       'direct_linear':'Direct linear','direct_additive':'Direct additive'}
CASES=list(NAMES)


def main():
    os.environ['MPLCONFIGDIR']=str(run.OUT/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    budget=run.Budget()
    verified=json.loads((run.REPORT/'verification.json').read_text())
    assert verified['status']=='passed'
    paths=['funded/periods.csv','funded/annual.csv','funded/sharpe_intervals.csv',
           'matched_intervals.csv','fits.csv','seed_diagnostics.csv','perturbation.csv','refit_stability.csv']
    tables={name:pd.read_csv(run.REPORT/name) for name in paths}
    periods=tables['funded/periods.csv'];fits=tables['fits.csv'];seeds=tables['seed_diagnostics.csv']
    intervals=tables['funded/sharpe_intervals.csv'];matched=tables['matched_intervals.csv']
    def metric(case,period,column):
        selected=periods.loc[periods.case.eq(case)&periods.period.eq(period),column]
        assert len(selected)==1
        return float(selected.iloc[0])
    for case in CASES:assert metric(case,'full','proposals')==608
    for table in [intervals,matched]:
        for row in table.itertuples():
            expected=metric(row.case,row.period,'annualized_excess_sharpe')-metric(row.reference,row.period,'annualized_excess_sharpe')
            np.testing.assert_allclose(row.point,expected,atol=1e-10,rtol=0)
    stability=[]
    for name,denominator,label in [('perturbation','n','local_perturbation'),('refit_stability','n_common','monthly_refit')]:
        new=tables[name+'.csv']
        prior=pd.read_csv(run.ROOT/'reports/p25_additive_sizing'/f'{name}.csv')
        prior=prior.loc[prior.case.isin(['tree_cem','linear_score','additive_main'])].copy()
        prior['case']=prior.case.replace({'linear_score':'regression_linear','additive_main':'regression_additive'})
        combined=pd.concat([prior,new],ignore_index=True)
        sums=summarize_stability(combined,denominator,['case']);sums['diagnostic']=label
        assert sums.observations.nunique()==1
        stability.append(sums)
    stability=pd.concat(stability,ignore_index=True)
    stability.to_csv(run.REPORT/'stability_summary.csv',index=False)
    training=[]
    for case,own in fits.groupby('case',sort=False):
        gain=own.training_shadow_daily_sharpe-own.baseline_shadow_daily_sharpe
        training.append(dict(case=case,months=len(own),median_training_sharpe=own.training_shadow_daily_sharpe.median(),
            median_gain_over_fullsize=gain.median(),min_parameters=own.parameters.min(),max_parameters=own.parameters.max(),
            months_with_last_ten_gain_gt_001=int(own.final_ten_iteration_gain.gt(.01).sum()) if case in run.KINDS else None))
    pd.DataFrame(training).to_csv(run.REPORT/'training_summary.csv',index=False)
    paired_training=fits.pivot(index='fit',columns='case',values='training_shadow_daily_sharpe')
    training_comparisons=[]
    for case in run.KINDS:
        delta=paired_training[case]-paired_training.tree_cem
        training_comparisons.append(dict(case=case,reference='tree_cem',months=len(delta),
            median_training_sharpe_difference=delta.median(),mean_training_sharpe_difference=delta.mean(),
            months_ahead=int(delta.gt(1e-10).sum()),months_behind=int(delta.lt(-1e-10).sum())))
    pd.DataFrame(training_comparisons).to_csv(run.REPORT/'training_comparisons.csv',index=False)
    selection=[]
    allocations=pd.read_parquet(run.OUT/'allocations.parquet',use_threads=False)
    for case,own in allocations.groupby('case'):
        for code,count in own.learned_level.value_counts().sort_index().items():
            selection.append(dict(case=case,size_fraction=code/4,proposals=int(count)))
    pd.DataFrame(selection).to_csv(run.REPORT/'size_distribution.csv',index=False)
    def comparison(table,a,b):
        own=table.loc[table.case.eq(a)&table.reference.eq(b)&table.period.eq('evaluation')].sort_values('block_sessions')
        assert len(own)==2 and set(own.block_sessions)=={63,126}
        lines=[f'{NAMES[a]} minus {NAMES[b]}:']
        for r in own.itertuples():lines.append(f'  {r.block_sessions}-session blocks: change {r.point:+.3f}; conditional 95% interval [{r.lower95:+.3f}, {r.upper95:+.3f}].')
        if (own.lower95>0).all():lines.append('  Both block lengths favor the candidate, conditional on these fitted policies.')
        elif (own.upper95<0).all():lines.append('  Both block lengths favor the reference, conditional on these fitted policies.')
        else:lines.append('  A consistently signed difference is not established across both block lengths.')
        return lines
    outcome=('Neither direct policy exceeds the tree/CEM Sharpe in either period.'
             if all(metric(c,p,'annualized_excess_sharpe')<metric('tree_cem',p,'annualized_excess_sharpe')
                    for c in run.KINDS for p in ['validation','evaluation'])
             else 'Performance rankings differ across policies or periods; consult the funded table and paired intervals below.')
    lines=['MATCHED-OBJECTIVE DIRECT POSITION SIZING','',outcome,
        'Matching the downstream training objective did not establish a superior replacement in this experiment. Equal objective and candidate budgets do not establish equal optimization difficulty or prove the tree globally optimal.','',
        'Question: do linear or additive decision policies improve on the existing tree when trained against the same daily shadow-path Sharpe objective?',
        'All policies receive the same 608 original proposals and retain the original 20% opportunity gate. Execution uses the same 10m funded account, contracts, costs, risk limits, daily hedges and continuing inventory.',
        'The two new policies use all 15 economic variables with fixed identities. Monthly coefficients and training-only transforms change. Linear uses 16 parameters; additive uses up to 61, including an intercept. Additive terms are degree-2 splines with three training quantile knots and linear extrapolation; no interactions.',
        'The score is dimensionless and directly determines 50%/75%/100% requested size using boundaries -0.5/+0.5. It is not predicted dollars, a probability or an opportunity filter. Existing option quantities are not rebalanced to every new score.',
        '', 'WHAT IS MATCHED',
        'Training rows, completed-outcome chronology, daily action-specific label paths, unpenalized shadow Sharpe objective, executable sizes, seed 2026 and 50 x 192 = 9600 evaluated candidates per monthly fit are matched. Original categorical tree CEM choices are exactly reproduced in all 90 months.',
        'Direct policies use bounded diagonal Gaussian coefficient CEM; the original tree uses categorical leaf-action CEM. Same budget does not mean identical search geometry, parameter capacity, wall time or convergence. Coefficient bounds and simple bases constrain complexity, but no extra regularization penalty changes the common objective.',
        'All three constant sizes and the incumbent best policy are injected each iteration; the full-size policy is the initial incumbent. Policy hyperparameters are predeclared and not selected on later results.',
        'The original tree partitions are still fitted by profit-prediction squared error, with CEM optimizing the subsequent leaf sizes. New policies optimize their score coefficients directly. This comparison tests policy classes under matched downstream objective, not pure causal isolation of tree topology.',
        'Original legacy training paths retain their original costs; the funded backtest retains its current costs. The existing training/execution cost mismatch and standalone-path versus funded-admission mismatch remain shared limitations.',
        'Archived regression controls are descriptive comparisons to the previous approach. Moving from them to direct optimization also removes ridge/CV and tercile calibration, so that contrast is not a pure loss-function ablation.',
        '', 'FUNDED PERFORMANCE',
        'Earlier: 2018-2021. Later: January 2022-August 29 2025. Both histories were already inspected; neither is an untouched holdout.',
        'Model | Earlier Sharpe | Later Sharpe | Later profit above cash | Later entries | Later cash-relative drawdown']
    for case in CASES:
        lines.append(f'{NAMES[case]} | {metric(case,"validation","annualized_excess_sharpe"):.3f} | {metric(case,"evaluation","annualized_excess_sharpe"):.3f} | ${metric(case,"evaluation","profit_above_cash"):,.2f} | {int(metric(case,"evaluation","entries"))} | {metric(case,"evaluation","max_cash_relative_drawdown_pct"):.3f}%')
    lines+=['','LATER PAIRED COMPARISONS']
    for case in run.KINDS:lines+=comparison(intervals,case,'tree_cem')
    for a,b in [('direct_additive','direct_linear'),('direct_linear','regression_linear'),('direct_additive','regression_additive')]:lines+=comparison(matched,a,b)
    lines+=['Intervals resample common calendar blocks, not individual trades. They condition on fitted policies, do not retrain models in each resample, and do not adjust for the accumulated research search. No policy is automatically promoted.','',
        'TRAINING AND OPTIMIZATION DIAGNOSTICS',
        'Parameter counts describe the sizing search only. Tree split-feature selection and thresholds are not counted; spline coefficients include dependent basis directions. These are not comparable effective degrees of freedom for the whole fitted pipelines.',
        f'Each monthly training window has {int(fits.training_rows.min())}-{int(fits.training_rows.max())} completed outcomes; overlapping trades and reused windows are not independent observations.']
    for row in training:
        lines.append(f'{NAMES[row["case"]]}: median training shadow Sharpe {row["median_training_sharpe"]:.3f}; median training gain over full size {row["median_gain_over_fullsize"]:.3f}; {int(row["min_parameters"])}-{int(row["max_parameters"])} parameters.')
        if row['months_with_last_ten_gain_gt_001'] is not None:lines.append(f'  {row["months_with_last_ten_gain_gt_001"]}/90 fits improved best training Sharpe by over 0.01 in the last ten iterations. Small final improvement is not proof of global convergence.')
    for case,own in seeds.groupby('case'):
        delta=own.training_shadow_daily_sharpe-own.primary_shadow_sharpe
        lines.append(f'{NAMES[case]} training-only seeds 2027/2028 across six predeclared months: {len(own)} alternative fits; median assignment disagreement {own.assignment_disagreement.median():.1%}; shadow Sharpe change range [{delta.min():+.3f}, {delta.max():+.3f}].')
    for row in training_comparisons:
        lines.append(f'{NAMES[row["case"]]} versus tree on the same training windows: median Sharpe difference {row["median_training_sharpe_difference"]:+.3f}; ahead in {row["months_ahead"]}/90 months and behind in {row["months_behind"]}/90.')
    additive_training=next(r for r in training_comparisons if r['case']=='direct_additive')
    if additive_training['months_ahead']>45 and all(metric('direct_additive',p,'annualized_excess_sharpe')<metric('tree_cem',p,'annualized_excess_sharpe') for p in ['validation','evaluation']):
        lines.append('Interpretation: additive policies improve the training objective in most months, but the gain does not carry through to either funded evaluation period. This is consistent with overfitting and/or the gap between standalone shadow paths and funded portfolio outcomes; the experiment does not identify which mechanism dominates. The observed failure cannot simply be described as an inability to beat the tree on its training objective.')
    lines+=['Alternative seeds are not deployed or used for selection; these diagnostics do not establish out-of-sample seed robustness.','',
        'SIZING STABILITY',
        'Local: change one variable by +/-5% of its training standard deviation, keeping the others fixed. These are synthetic sensitivity checks, not feasible joint shocks.',
        'Refit: apply adjacent monthly fits to common historical training rows. This measures recommendation changes, not actual position turnover or prospective generalization.',
        'Model | Local changed decisions | Refit changed decisions']
    for case in CASES:
        own=stability.loc[stability.case.eq(case)]
        if case=='fixed100':lines.append('Fixed 100% | 0% by construction | 0% by construction');continue
        local=own.loc[own.diagnostic.eq('local_perturbation'),'changed_fraction'].iloc[0]
        refit=own.loc[own.diagnostic.eq('monthly_refit'),'changed_fraction'].iloc[0]
        lines.append(f'{NAMES[case]} | {local:.2%} | {refit:.2%}')
    lines+=['','VERIFICATION',
        'Four targeted unit tests passed. All 180 direct models and their 90 original tree counterparts were checked. Saved predictions, contributions, size choices, transforms and stability metrics reproduce; all direct training objectives were independently reaggregated using the original state-book implementation.',
        'Four complete direct CEM searches (both policy types in the first and last months) were rerun exactly. All six funded ledgers reconcile, archived controls are byte-identical and protected source hashes remain unchanged.',
        '', 'REPRODUCE',
        'Run sequentially in .venv-pca312: python -m scripts.run_p25_direct_policy; python -m scripts.verify_p25_direct_policy; python -m scripts.report_p25_direct_policy.',
        'One numerical worker/thread, 20% CPU duty, 384 MiB RSS limit. Saved models and ledgers: artifacts/contracts/p25_direct_policy.']
    (run.REPORT/'Findings.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    x=np.arange(len(CASES));labels=[NAMES[c].replace(' ','\n',1) for c in CASES]
    early=[metric(c,'validation','annualized_excess_sharpe') for c in CASES]
    late=[metric(c,'evaluation','annualized_excess_sharpe') for c in CASES]
    ax=axes[0,0];ax.bar(x-.18,early,.36,label='2018-2021',color='#9aaab5');ax.bar(x+.18,late,.36,label='2022-Aug 2025',color='#258093')
    ax.set_xticks(x,labels,fontsize=9);ax.set_ylabel('Annualized excess Sharpe');ax.legend();ax.set_title('Funded performance')
    ax=axes[0,1];profit=[metric(c,'evaluation','profit_above_cash')/1000 for c in CASES]
    bars=ax.bar(x,profit,color=['#566573']*4+['#337ca3','#16826c']);ax.set_xticks(x,labels,fontsize=9);ax.set_ylabel('Later profit above cash, $ thousands')
    for bar,c in zip(bars,CASES):ax.annotate(f'{int(metric(c,"evaluation","entries"))} trades',(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',xytext=(0,4),textcoords='offset points',fontsize=8)
    ax.set_title('Profit and executed entries')
    ax=axes[1,0]
    subset=['tree_cem','direct_linear','direct_additive'];sx=np.arange(3)
    for offset,diagnostic,label,color in [(-.18,'local_perturbation','Small one-variable change','#3579aa'),(.18,'monthly_refit','Monthly refit','#cd9141')]:
        values=[100*stability.loc[stability.case.eq(c)&stability.diagnostic.eq(diagnostic),'changed_fraction'].iloc[0] for c in subset]
        ax.bar(sx+offset,values,.36,label=label,color=color)
    ax.set_xticks(sx,[NAMES[c] for c in subset]);ax.set_ylabel('Evaluated sizing decisions changed, %');ax.legend(fontsize=9);ax.set_title('Recommendation stability')
    ax=axes[1,1];comparisons=[('direct_linear','tree_cem',intervals),('direct_additive','tree_cem',intervals),('direct_additive','direct_linear',matched)]
    for i,(a,b,table) in enumerate(comparisons):
        own=table.loc[table.case.eq(a)&table.reference.eq(b)&table.period.eq('evaluation')]
        for off,block,color in [(-.12,63,'#3579aa'),(.12,126,'#b47627')]:
            row=own.loc[own.block_sessions.eq(block)].iloc[0]
            ax.plot([row.lower95,row.upper95],[i+off]*2,color=color);ax.plot(row.point,i+off,'o',color=color,label=f'{block}-session blocks' if i==0 else None)
    ax.axvline(0,color='#888',ls='--');ax.set_yticks(range(3),[f'{NAMES[a]}\nminus {NAMES[b]}' for a,b,_ in comparisons],fontsize=9)
    ax.invert_yaxis();ax.legend(fontsize=9);ax.set_xlabel('Later Sharpe change and conditional 95% interval');ax.set_title('Matched-objective comparisons')
    fig.suptitle('Direct optimization of linear and additive sizing policies\nSame shadow Sharpe objective and search budget; exploratory history',fontsize=15)
    fig.savefig(run.REPORT/'comparison.png',dpi=160);plt.close(fig);budget.rest()
    outputs=['Findings.txt','comparison.png','stability_summary.csv','training_summary.csv','training_comparisons.csv','size_distribution.csv']
    run.base.write(run.REPORT/'report_manifest.json',dict(status='complete',peak_rss_mib=budget.peak,
        source_hashes={name:run.base.digest(run.REPORT/name) for name in paths},
        script_hash=run.base.digest(__file__),outputs={name:run.base.digest(run.REPORT/name) for name in outputs}))
    run.base.write(run.OUT/'progress.json',dict(stage='complete',models_complete=180,verified=True,
        report_directory=str(run.REPORT)))
    print(f'Report complete: {run.REPORT}',flush=True)


if __name__=='__main__':main()
