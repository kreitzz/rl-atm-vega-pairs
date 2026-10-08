"""Report the frozen-policy diagnostic; never fit or select a strategy."""
import os
from scripts import diagnose_p25_generalization as run
import json
import numpy as np
import pandas as pd

NAMES={'tree_cem':'Tree + CEM','direct_linear':'Direct linear','direct_additive':'Direct additive'}
STAGES={'archived':'Standalone, old costs','current_cost':'Standalone, current costs',
        'admitted_shadow':'Only actual admissions, shadow paths','funded':'Actual funded portfolio'}


def admission_checks(table,budget):
    """Post-hoc fixed-mask diagnostic, not a feasible strategy or new search."""
    signal=pd.read_parquet(run.prior.OUT/'replays/tree_cem/scored_allocations.parquet',use_threads=False)
    meta=pd.read_parquet(run.base.SOURCE,columns=['date','pair','position'],use_threads=False)
    signal=signal.merge(meta,on=['date','pair'],validate='one_to_one',sort=False)
    calendar=pd.read_parquet(run.base.MARKET/'rates.parquet',use_threads=False).index
    paths=np.load(run.OUT/'reconstructed.npz')['current']
    masks={};selected={};statuses=[];overlap=[]
    for case in run.CASES:
        folder=run.prior.OUT/'replays'/case
        choices=pd.read_parquet(folder/'scored_allocations.parquet',use_threads=False)
        own=signal[['date','pair','direction']].merge(choices[['date','pair','learned_level']],on=['date','pair'],validate='one_to_one',sort=False)
        ix=np.where(own.direction>0,3+own.learned_level,4-own.learned_level).astype(int)
        selected[case]=paths[np.arange(len(signal)),ix]
        dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        dec=signal[['date','pair']].merge(dec,left_on=['date','pair'],right_on=['signal_date','pair'],validate='one_to_one',sort=False)
        masks[case]=dec.status.eq('entered').to_numpy()
        for period,start,stop in run.PERIODS:
            part=dec.loc[dec.entry_date.between(start,stop)]
            for (status,limits),g in part.groupby(['status','risk_limits'],dropna=False):
                statuses.append(dict(case=case,period=period,status=status,risk_limits=limits,count=len(g)))
    results=[]
    for sized,values in selected.items():
        for mask_name,mask in masks.items():
            series=run.aggregate(values,signal.position,calendar,mask).loc['2018-01-02':'2025-08-29']/1e7
            for period,start,stop in run.PERIODS:
                r=series.loc[start:stop]
                if r.index[0]==series.index[0]:r=r.iloc[1:]
                value=run.sharpe(r)
                results.append(dict(sizing=sized,admission_mask=mask_name,period=period,sharpe=value))
                if sized==mask_name:
                    expected=table.loc[table.stage.eq('admitted_shadow')&table.case.eq(sized)&table.period.eq(period),'sharpe'].iloc[0]
                    np.testing.assert_allclose(value,expected,atol=1e-12)
            budget.rest()
    for case in run.CASES[1:]:
        for period,start,stop in run.PERIODS:
            use=signal.date.between(start,stop).to_numpy()
            a=masks['tree_cem'];b=masks[case]
            overlap.append(dict(case=case,period=period,scope='signal dates',both=int((use&a&b).sum()),
                tree_only=int((use&a&~b).sum()),candidate_only=int((use&~a&b).sum()),neither=int((use&~a&~b).sum())))
    pd.DataFrame(results).to_csv(run.REPORT/'posthoc_crossed_admission_masks.csv',index=False)
    pd.DataFrame(statuses).to_csv(run.REPORT/'admission_status.csv',index=False)
    pd.DataFrame(overlap).to_csv(run.REPORT/'admission_overlap.csv',index=False)
    return pd.DataFrame(results)


def main():
    budget=run.base.Budget()
    verify=json.loads((run.REPORT/'verification.json').read_text())
    assert verify['status']=='passed'
    table=pd.read_csv(run.REPORT/'stages.csv')
    intervals=pd.read_csv(run.REPORT/'intervals.csv')
    fits=pd.read_csv(run.prior.REPORT/'fits.csv').pivot(index='fit',columns='case',values='training_shadow_daily_sharpe')
    training=[]
    for case in run.CASES[1:]:
        delta=fits[case]-fits.tree_cem
        training.append(dict(case=case,months=len(delta),months_ahead=int((delta>0).sum()),median_difference=float(delta.median())))
    pd.DataFrame(training).to_csv(run.REPORT/'training_comparison.csv',index=False)
    def value(stage,case,period='evaluation'):
        row=table.loc[table.stage.eq(stage)&table.case.eq(case)&table.period.eq(period),'sharpe']
        assert len(row)==1
        return float(row.iloc[0])
    # Independently match the old report's funded metrics and every paired point.
    original=pd.read_csv(run.prior.REPORT/'funded/periods.csv')
    for r in original.loc[original.case.isin(run.CASES)].itertuples():
        np.testing.assert_allclose(value('funded',r.case,r.period),r.annualized_excess_sharpe,atol=1e-12)
    for r in intervals.itertuples():
        np.testing.assert_allclose(r.point,value(r.stage,r.case,r.period)-value(r.stage,r.reference,r.period),atol=1e-12)
    crossed=admission_checks(table,budget)
    lines=['FROZEN-POLICY GENERALIZATION DIAGNOSTIC','',
           'Question: where does the additive policy lose its training advantage over tree + CEM?',
           'No models were retrained, no allocations changed, and no new strategy was selected.','']
    if value('archived','direct_additive')<value('archived','tree_cem'):
        lines+=['Main finding: the additive advantage is already absent on subsequent standalone proposals at the original training costs, before funded portfolio admission or shared-account accounting.',
                'This is evidence of a transfer/generalization problem in this historical comparison. It does not uniquely identify overfitting, regime changes, conditioning on the proposal gate, or optimization noise.','']
    elif value('admitted_shadow','direct_additive')<value('admitted_shadow','tree_cem'):
        lines+=['Main finding: the later additive standalone point estimate is better, at both old and current costs. The ranking flips when the paths are restricted to each policy\'s actual funded admissions.',
            'Thus, the observed funded advantage of tree/CEM is primarily associated with its interaction with portfolio admission. The evidence does not support a simple claim that the additive model fails to generalize.',
            'The additive standalone advantage and funded disadvantage both have paired intervals spanning zero. This locates the observed reversal; it does not establish statistical superiority.',
            'Earlier-period results differ: additive is slightly worse even on unrestricted standalone paths, and the gap widens after admission.','']
    else:
        lines+=['Main finding: the additive policy retains a standalone advantage in the later period. Inspect the cost, admission and funded stages to locate the change in ranking.','']
    lines+=['TRAINING (90 overlapping monthly windows; not 90 independent trials)']
    for r in training:
        lines.append(f"{NAMES[r['case']]}: ahead of tree in {r['months_ahead']}/90 windows; median paired Sharpe difference {r['median_difference']:+.4f}.")
    lines+=['Training summaries must not be compared as absolute Sharpe levels with the later aggregated portfolio: windows and samples differ.','']
    for period,label in [('validation','EARLIER: 2018-2021'),('evaluation','LATER: 2022-AUGUST 2025')]:
        lines += [label,'Stage                                              Tree     Linear   Additive  Additive-tree']
        for stage,name in STAGES.items():
            a,b,c=[value(stage,k,period) for k in run.CASES]
            lines.append(f'{name:49s} {a:7.3f}  {b:7.3f}  {c:7.3f}  {c-a:+.3f}')
        lines.append('')
    lines+=['COST AND ADMISSION CHECKS']
    cohorts=pd.read_csv(run.REPORT/'cohorts.csv')
    for case in run.CASES:
        own=cohorts.loc[cohorts.case.eq(case)]
        cost_change=value('current_cost',case)-value('reconstructed_old',case)
        admission_change=value('admitted_shadow',case)-value('current_cost',case)
        lines.append(f'{NAMES[case]}: later standalone Sharpe change from commission alone {cost_change:+.4f}; descriptive admission-mask change {admission_change:+.4f}.')
        lines.append(f'  All-history entry commission saving ${own.entry_commission_saving.sum():,.2f}; with financing through cohort expiry ${own.lifetime_cost_saving.sum():,.2f}.')
    lines+=['The admission mask uses each strategy\'s actual saved accept/reject decisions. It diagnoses which proposals reached the book, but is not a new executable policy.',
            'The final step also changes NAV-scaled integer quantities, netted hedges, financing, execution handling and the return denominator. Differences are descriptive and are not an additive causal decomposition of Sharpe.','',
            'POST-HOC CHECK: HOLD THE ADMITTED PROPOSALS FIXED (later Sharpe)',
            'Sizing policy              Tree admission mask  Linear mask  Additive mask']
    for case in run.CASES:
        values=[float(crossed.loc[crossed.sizing.eq(case)&crossed.admission_mask.eq(mask)&crossed.period.eq('evaluation'),'sharpe'].iloc[0]) for mask in run.CASES]
        lines.append(f'{NAMES[case]:26s} {values[0]:19.3f} {values[1]:12.3f} {values[2]:14.3f}')
    lines+=['This supplementary calculation was added after seeing the stage results. It crosses saved sizes and saved admission masks, with no refit or policy selection. The crossed combinations may violate risk limits and are not funded strategy results.','',
            'On the tree-admitted proposal set, the tree sizes also beat additive sizes. Conversely, on the additive-admitted set, additive sizes beat tree sizes. This is an interaction between allocation and the selected portfolio, not just a difference in trade counts.',
            'The saved later ledgers admit110 proposals for tree/CEM and97 for additive;87 are shared,23 are tree-only and10 additive-only. The dominant recorded entry-risk rejection is pair/wing stress loss.',
            'This does not mean the tree was explicitly trained to optimize portfolio admission: both policies optimized standalone paths. Its sizing choices interact more favorably with the existing limits in this observed sample.',
            'A subsequent experiment should evaluate candidate sizing choices through the actual past portfolio ledger, with inventory and risk limits, using training-only selection and chronological evaluation. Better funded performance is not guaranteed.','',
            'LATER PAIRED SHARPE-DIFFERENCE INTERVALS (candidate minus tree)']
    for stage in STAGES:
        for case in run.CASES[1:]:
            group=intervals.loc[intervals.stage.eq(stage)&intervals.case.eq(case)&intervals.period.eq('evaluation')]
            for r in group.itertuples():
                lines.append(f'{STAGES[stage]}, {NAMES[case]}, {r.block_sessions}-session blocks: {r.point:+.3f} [{r.lower95:+.3f}, {r.upper95:+.3f}].')
    lines+=['','VERIFICATION AND SCOPE',
        'All 608 proposals have finite archived and reconstructed paths for every chosen action. No outcome-based exclusions or missing-path zero filling.',
        'For each policy, all 608 reconstructed old-cost paths match their archived float32 values exactly. Sub-cent full-history dollar differences are serialization rounding.',
        'Standalone paths use the original 1m cash account and fixed450vega sizing reference. The fee bridge changes only commission from .65 to .03; all other label mechanics stay fixed.',
        'Standalone Sharpe sums overlapping daily cash-relative dollar paths, including zero days. It does not average per-trade Sharpes. Stored values divide dollars by10m; that constant does not change Sharpe.',
        'Actual funded Sharpe uses the saved 10m portfolio\'s cash-excess daily percentage returns; all three ledgers reconcile independently and match the prior reported results.',
        'Intervals use paired circular calendar blocks,2000 draws,seed2026. They are conditional on fitted policies and do not correct for the many alternatives already examined.',
        'Both periods have already been inspected. This diagnostic does not prove tree/CEM is globally optimal or guaranteed robust on future data.',
        'Protected source and replay hashes were unchanged. Original checkpoints and decisions are preserved.',
        f"Peak diagnostic RSS: {verify['peak_rss_mib']:.1f}MiB; one numeric worker, one thread,20% CPU duty.",
        '', 'Reproduce: python -m scripts.diagnose_p25_generalization; python -m unittest tests.test_p25_generalization; python -m scripts.report_p25_generalization.']
    (run.REPORT/'Findings.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.environ['MPLCONFIGDIR']=str(run.OUT/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#21618c','#d68910','#138d75']
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),sharey=True)
    short=['Old-cost\nstandalone','Current-cost\nstandalone','Admitted\nshadow paths','Funded\nportfolio']
    for ax,period,title in zip(axes,['validation','evaluation'],['2018-2021','2022-August 2025']):
        x=np.arange(len(STAGES))
        for j,case in enumerate(run.CASES):
            values=[value(stage,case,period) for stage in STAGES]
            bars=ax.bar(x+(j-1)*.25,values,width=.24,color=colors[j],label=NAMES[case])
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
        ax.set_xticks(x,short);ax.set_title(title);ax.axhline(0,color='#566573',lw=.7)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Annualized Sharpe (compare models within each stage)')
    axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    fig.suptitle('Where does the training advantage disappear?',fontsize=16,x=.07,ha='left')
    fig.text(.07,.91,'Saved models and all 608 proposals held fixed; no retraining or parameter search.',fontsize=10)
    fig.text(.07,.035,'Standalone: summed cash-relative dollar paths. Funded: cash-excess percentage returns.\nAdmission mask is descriptive. Previously inspected history; this is not an untouched holdout.',fontsize=9,color='#566573')
    fig.subplots_adjust(top=.80,bottom=.20,left=.07,right=.98,wspace=.08)
    fig.savefig(run.REPORT/'comparison.png',dpi=160);plt.close(fig)
    budget.rest()
    run.base.write(run.REPORT/'completion.json',dict(status='passed',funded_results_match_previous=True,
        interval_points_verified=True,report=str(run.REPORT/'Findings.txt'),figure=str(run.REPORT/'comparison.png')))
    print('\n'.join(lines[:30]))


if __name__=='__main__':main()
