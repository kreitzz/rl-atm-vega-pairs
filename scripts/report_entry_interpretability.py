"""Compact interpretability figures from the exact frozen-policy audit."""
from pathlib import Path

import pandas as pd

ROOT=Path('reports/retained_entry_explanations')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pair=pd.read_csv(ROOT/'put_pair_path_importance.csv')
    straddle=pd.read_csv(ROOT/'straddle_feature_importance.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    table=pair.pivot(index='feature',columns='period',values='path_visits').fillna(0)
    table=table.div(table.sum(axis=0),axis=1)*100
    selected=table.sum(axis=1).nlargest(8).index
    table.loc[selected,['validation','evaluation']].iloc[::-1].plot.barh(ax=axes[0],color=['#9dc3d5','#24698c'])
    axes[0].set_title('Put pairs: inputs encountered on actual tree paths')
    axes[0].set_xlabel('Share of visited decision nodes (%)');axes[0].set_ylabel('')
    table=straddle.pivot(index='feature',columns='period',values='center_ablation_size_flip_rate').fillna(0)*100
    selected=table.sum(axis=1).nlargest(8).index
    table.loc[selected,['validation','evaluation']].iloc[::-1].plot.barh(ax=axes[1],color=['#9dc3d5','#24698c'])
    axes[1].set_title('Straddles: sensitivity of the learned size')
    axes[1].set_xlabel('Sizes changed after centering one input (%)');axes[1].set_ylabel('')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('What information do the retained sizing policies actually use?')
    fig.text(.02,.015,'Different measures: tree-path usage versus one-feature score sensitivity. Correlated inputs can substitute for one another; neither measure is a causal effect.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(ROOT/'entry_policy_explanations.png',dpi=140);plt.close(fig)
    sizes=pd.read_csv(ROOT/'size_and_admission.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,strategy in zip(axes,['put_pair','straddle']):
        f=sizes.loc[sizes.strategy.eq(strategy)]
        allrows=f.groupby(['period','size']).proposals.sum()
        entered=f.loc[f.status.eq('entered')].groupby(['period','size']).proposals.sum()
        table=pd.DataFrame({'All proposals':allrows,'Executed entries':entered}).fillna(0)
        table.index=[f'{p} / {s:.0%}' for p,s in table.index]
        table.plot.bar(ax=ax,color=['#9dc3d5','#24698c'],rot=35)
        ax.set_title(strategy.replace('_',' ').title());ax.set_ylabel('Count');ax.set_xlabel('Period / fraction of full size')
        ax.tick_params(axis='x',labelsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A proposed size and permission to enter are separate decisions')
    fig.tight_layout();fig.savefig(ROOT/'size_and_admission.png',dpi=140);plt.close(fig)
    examples=pd.read_csv(ROOT/'put_pair_rules.csv')
    examples.loc[examples.status.eq('entered')].sort_values('date').groupby(['period','size'],as_index=False).first().to_csv(ROOT/'put_pair_examples.csv',index=False)
    fields=['date','secid','period','status','size','score','state','state_size_map','cuts','last_completed','training_samples','fitted_context','intercept']
    examples=pd.read_csv(ROOT/'straddle_score_explanations.csv',usecols=fields)
    examples.loc[examples.status.eq('entered')].sort_values('date').groupby(['period','size'],as_index=False).first().to_csv(ROOT/'straddle_examples.csv',index=False)
    (ROOT/'INTERPRETABILITY_NOTES.txt').write_text(
        'INTERPRETABILITY FOR THE RETAINED IMPLEMENTATION\n\n'
        'There are three distinct questions: why was a proposal eligible, why did the learner assign this size, and why did the funded simulator admit or reject it? The new audit explains size exactly and retains the saved admission status; it does not replace the eligibility specification.\n\n'
        'Put pairs: follow the exact observed root-to-leaf conditions, then read the selected sizing action for that leaf. Completed profitability memory, IV, return correlation and execution cost appear frequently. These are local decision rules; there is no universal rule that increasing one input must increase size.\n\n'
        'Straddles: standardized inputs and ridge coefficients add exactly to a context score. Score thresholds assign a context, then the learned context-to-size map assigns50/75/100%. The score is a completed-trade dollar-profit partition target, not itself a Q-value. Size maps need not be monotonic because the subsequent objective is aggregate daily Sharpe.\n\n'
        'Setting IV to its monthly training center changes36.6% of fitted later-period proposal sizes; VIX change changes15.3%, normalized RV-minus-IV memory12.9%, and VIX level11.9%. These are conditional sensitivity measures across all proposals, including rejected ones, not estimated causal effects.\n\n'
        '608 put-pair sizes and35774 straddle sizes reproduce the saved policy levels exactly. Models are checked for completed outcomes before the monthly fit cutoff. Some early months use a default action and have no fitted feature attribution.\n\n'
        'The two importance charts intentionally use different definitions. Tree node usage and a ridge-score centering ablation should not be compared numerically as the same importance scale.\n\n'
        'Examples are the earliest executed entry of each size in each reporting period, selected without looking at their profit. Full rule paths, scores, size maps and admission statuses are in the accompanying CSVs.\n\n'
        'This adds decision transparency. It does not by itself establish an out-of-sample sizing advantage, causal explanations, execution feasibility or conference readiness. Stability tests and new forward evaluation remain useful.\n')
    print('Saved entry-policy charts and explanation examples',flush=True)


if __name__=='__main__':main()
