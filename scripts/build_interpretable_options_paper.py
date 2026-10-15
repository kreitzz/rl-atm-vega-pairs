"""Build a separate paper copy from audited entry and Q-allocation results.

All output stays in paper/interpretability_extension_2026_10_15. The submitted
paper is never edited. Run figures, manuscript, render1, render2, assemble,
and docx as separate guarded phases. Render with --page N in a fresh worker
for every page to bound the renderer's font and image allocator caches.
"""
import argparse
import gc
import hashlib
import html
import json
import zipfile
from pathlib import Path

DEST=Path('paper/interpretability_extension_2026_10_15')
FIG=DEST/'figures'
REPORT=Path('reports/allocation_interpretability_099')
RESULT=Path('reports/signal_allocation_discount')
BLUE,GRAY,ORANGE='#086ba5','#73808c','#c17b32'
FIGURES={
 'learning_layers':'Three decisions are kept separate: the signal proposes a contract trade, an entry bandit sizes that proposal, and the Q allocator divides one capital budget between the resulting strategy indices.',
 'cashflows':'Daily cash-flow accounting, rather than a terminal IV-versus-RV proxy, defines profit. Delta hedging reduces spot exposure but leaves nonlinear and event risks.',
 'bandit_results':'Annualized daily excess Sharpe for the same proposal pool and funded limits within each strategy. Entry policies change sizes, which can also change admission.',
 'bandit_explanations':'Left: fraction of actual put-tree decision nodes using each input. Right: fraction of straddle proposal sizes changed by setting one input to its monthly training center. These are different measures; neither estimates causal economic effects.',
 'q_validation':'All seven fixed put/straddle capital splits and the two discussed Q settings on 2018–2021. Labels give put/straddle weights. The Q settings are researched operating points, not the highest-validation-Sharpe selections.',
 'q_evaluation':'The same comparison on 2022–August 2025. Both strategies use their retained entry bandits. Lines connect discrete observed settings and do not represent a fitted efficient frontier.',
 'q_context':'For the penalty-0.20 Q policy, later-period action sensitivity is greatest to VIX, the straddle IV-versus-RV gap, and put-spread strength. Sensitivity is measured at actual fitted decision states by centering one input.',
 'q_schedule':'The penalty-0.20 Q policy mostly switches between the two strategies. Allocation labels are capital weights in funded strategy indices, not the fraction of cash invested in option premium.',
 'q_timing':'Same-average controls and discount-zero learners use the same underlying return series and resizing costs. Same-average exposure is a retrospective diagnostic, not a policy selected before trading.',
}


def read(path):
    import pandas as pd
    return pd.read_csv(path)


def figures(only_cashflows=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    def finish(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=165,bbox_inches='tight',facecolor='white')
        plt.close(fig);gc.collect()
    def style(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig,ax=plt.subplots(figsize=(10.5,2.25));ax.set(xlim=(0,10.5),ylim=(0,2.25));ax.axis('off')
    for x,title,text in [(0.1,'1  Economic signals','Relative downside IV\nPersistent RV minus IV'),
                         (3.75,'2  Entry bandits','Size each proposed trade\n50% / 75% / 100%'),
                         (7.4,'3  Portfolio Q-learning','Divide one capital budget\nPut pairs ↔ short straddles')]:
        ax.add_patch(FancyBboxPatch((x,.45),3.,1.45,boxstyle='round,pad=.08',fc='#edf4f8',ec='#9bb6c8'))
        ax.text(x+.15,1.63,title,weight='bold',color=BLUE,fontsize=11)
        ax.text(x+.15,1.05,text,va='center',linespacing=1.7,fontsize=10)
        if x<7:ax.annotate('',xy=(x+3.48,1.13),xytext=(x+3.12,1.13),arrowprops={'arrowstyle':'->','color':BLUE,'lw':2})
    finish(fig,'learning_layers')
    fig,ax=plt.subplots(figsize=(10.5,1.6));ax.axis('off')
    texts=['Option marks\nand settlement','Stock hedge\nP&L','Dividends\nand funding','Execution\nand borrow','Daily change\nin account NAV']
    for i,text in enumerate(texts):
        x=i*2.12
        ax.text(x+.85,.8,text,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.6',fc='#edf4f8' if i<4 else '#d8eaf5',ec='#9bb6c8'))
        if i<4:ax.text(x+1.94,.8,'=' if i==3 else ('−' if i==2 else '+'),ha='center',va='center',fontsize=14,color=BLUE)
    ax.set(xlim=(-.1,10.5),ylim=(0,1.6));finish(fig,'cashflows')
    if only_cashflows:
        print('Saved workflow and signed cash-flow figures',flush=True)
        return
    data=read(REPORT/'bandit_risk_return.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.35),sharey=True)
    for ax,strategy,title in zip(axes,['put','straddle'],['30-day 25-delta put pairs','30-day short ATM straddles']):
        for j,(period,label,color) in enumerate([('validation','2018–2021','#afcbdc'),('evaluation','2022–2025',BLUE)]):
            f=data.loc[(data.strategy==strategy)&(data.period==period)].set_index('policy')
            ax.bar(np.arange(4)+(j-.5)*.33,[f.loc[s,'annualized_excess_sharpe'] for s in ['fixed50','fixed75','fixed100','bandit']],width=.33,color=color,label=label)
        ax.set_xticks(range(4),['Fixed 50%','Fixed 75%','Fixed 100%','Bandit']);ax.tick_params(axis='x',labelsize=8)
        ax.set_title(title,fontsize=11);style(ax)
    axes[0].set_ylabel('Annualized excess Sharpe');axes[0].legend(frameon=False,fontsize=9);fig.tight_layout();finish(fig,'bandit_results')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5))
    p=read(REPORT/'put_tree_usage.csv');p=p.loc[p.period=='evaluation'].nlargest(5,'path_share').sort_values('path_share')
    names={'profit_365d':'Completed profit, 365 days','iv_mean':'Mean implied volatility','commission_vol_points':'Commission / contract vega','correlation':'Stock return correlation','profit_91d':'Completed profit, 91 days'}
    axes[0].barh([names.get(x,x) for x in p.feature],100*p.path_share,color=BLUE);axes[0].set_xlabel('Share of visited nodes (%)');axes[0].set_title('Put pairs: exact tree-path usage',fontsize=11)
    s=read('reports/retained_entry_explanations/straddle_feature_importance.csv');s=s.loc[s.period=='evaluation'].nlargest(5,'center_ablation_size_flip_rate').sort_values('center_ablation_size_flip_rate')
    names={'iv':'Implied volatility','vix_change21':'Recent VIX change','normalized_memory':'Normalized RV−IV memory','vix':'VIX level','benchmark_sharpe63':'Recent benchmark Sharpe'}
    axes[1].barh([names.get(x,x) for x in s.feature],100*s.center_ablation_size_flip_rate,color=BLUE);axes[1].set_xlabel('Proposed sizes changed (%)');axes[1].set_title('Straddles: one-input sensitivity',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',labelsize=8)
    fig.tight_layout(w_pad=3);finish(fig,'bandit_explanations')
    fixed=read(RESULT/'fixed_results.csv');qs=read(RESULT/'q_results.csv')
    for period,name in [('validation','q_validation'),('evaluation','q_evaluation')]:
        fig,ax=plt.subplots(figsize=(10.5,4.3));fig.subplots_adjust(left=.09,right=.97,bottom=.16,top=.92)
        f=fixed.loc[fixed.period==period].sort_values('straddle_target')
        q=qs.loc[(qs.period==period)&(qs.gamma==.99)&qs.risk.isin([.1,.2])].sort_values('profit_above_cash')
        ax.plot(f.profit_above_cash/1e6,f.annualized_excess_sharpe,'o-',color=GRAY,label='Seven fixed splits',lw=2)
        for r in f.itertuples():ax.annotate(f'{1-r.straddle_target:.0%}/{r.straddle_target:.0%}',(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=8,color=GRAY)
        ax.plot(q.profit_above_cash/1e6,q.annualized_excess_sharpe,'o-',color=BLUE,label='Dynamic Q',lw=2.5)
        for r in q.itertuples():
            offset=((-5,30) if r.risk==.2 else (15,-6)) if period=='validation' else (8,9)
            ax.annotate(f'Q {r.risk:g}: {r.annualized_excess_sharpe:.3f}',(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=offset,textcoords='offset points',fontsize=9,color=BLUE,weight='bold')
        ax.set(xlabel='Profit above cash ($ millions)',ylabel='Annualized excess Sharpe',xlim=(0,2.6),ylim=(.4,1.96));ax.legend(frameon=False,loc='upper right',ncol=2,fontsize=9);style(ax);finish(fig,name)
    s=read(REPORT/'q_sensitivity_summary.csv');s=s.loc[(s.case=='q_lambda0.2_g0.99')&(s.period=='evaluation')].nlargest(6,'action_changed').sort_values('action_changed')
    names={'vix':'VIX','atm_iv_rv_gap':'Straddle IV minus RV','pair_rank_strength':'Put spread strength','relative_sharpe63':'Relative recent Sharpe','log_vol_ratio':'Relative strategy volatility','atm_completed_premium':'Completed straddle premium','correlation63':'Strategy return correlation'}
    fig,ax=plt.subplots(figsize=(10.5,2.8));ax.barh([names.get(x,x) for x in s.feature],100*s.action_changed,color=BLUE)
    ax.set_xlabel('Later allocation decisions changed by centering one input (%)');ax.spines[['top','right']].set_visible(False);fig.tight_layout();finish(fig,'q_context')
    d=read(REPORT/'q_decisions.csv');d=d.loc[d.case=='q_lambda0.2_g0.99'];d['date']=__import__('pandas').to_datetime(d.date)
    fig,ax=plt.subplots(figsize=(10.5,2.8));ax.step(d.date,d.straddle_weight,where='post',color=BLUE,lw=1.6);ax.set(ylim=(-.05,1.1),ylabel='Straddle capital weight');ax.set_yticks([0,.5,1],['0%','50%','100%']);style(ax);fig.tight_layout();finish(fig,'q_schedule')
    c=read(REPORT/'q_controls.csv');c=c.loc[c.case=='q_lambda0.2_g0.99']
    fig,ax=plt.subplots(figsize=(10.5,3))
    for j,(period,color,label) in enumerate([('validation','#afcbdc','2018–2021'),('evaluation',BLUE,'2022–2025')]):
        f=c.loc[c.period==period].set_index('control')
        ax.bar(np.arange(3)+(j-.5)*.33,[f.loc[k,'annualized_excess_sharpe'] for k in ['same_mean_target_hindsight','saved_gamma_zero','actual_q']],width=.33,color=color,label=label)
    ax.set_xticks(range(3),['Fixed at Q’s average weight','One-period learner (γ = 0)','Sequential learner (γ = 0.99)']);ax.set_ylabel('Annualized excess Sharpe');ax.legend(frameon=False);style(ax);fig.tight_layout();finish(fig,'q_timing')
    print('Saved nine publication figures',flush=True)


def manuscript():
    import pandas as pd
    p=lambda t:dict(type='paragraph',text=t)
    h=lambda t:dict(type='heading',text=t)
    c=lambda name,height=190:dict(type='chart',name=name,height=height)
    def table(headers,rows,widths):return dict(type='table',headers=headers,rows=rows,widths=widths)
    pages=[]
    def page(title,blocks,subtitle=None):pages.append(dict(title=title,blocks=blocks,**({'subtitle':subtitle} if subtitle else {})))
    page('Learning exposure in options relative value',[
      h('Abstract'),
      p('We study two short-dated volatility signals and ask whether learning improves their use of capital. The first trades approximately 30-day, 25-delta out-of-the-money put pairs; the second sells approximately 30-day at-the-money straddles when completed RV-minus-IV history indicates expensive implied volatility. Actual-contract simulations include daily delta hedging, dividends, financing and transaction charges. Contextual entry-sizing policies raise historical excess Sharpe relative to fixed sizing within both signals. We then freeze those policies and use fitted Q-learning to allocate one capital budget between them.'),
      p('At one researched operating point, Q achieves annualized excess Sharpe 1.215 in 2018–2021 and 1.726 in 2022–August 2025, with $1.306 million later profit above cash. The best later fixed-split Sharpe is 1.612, earning $435,181. Exact decision explanations and exposure-matched timing controls locate the gain in conditional allocation; a discount-zero learner captures much, but not all, of it. These are selected historical results, not independent confirmation of a repeatable learning advantage.'),
      c('learning_layers',118),
      h('The research question'),
      p('A good trading signal, an appropriate entry size and a useful portfolio allocation are different achievements. We examine each separately. Entry bandits choose the size of a proposed option trade. The Q learner makes a later portfolio decision: how much capital should each already specified strategy receive? This separation makes it possible to explain what learning changes and which mechanism is associated with better returns.')
    ],'Adam Kreitzman and Soo Xin Yang · Research copy · 15 October 2026')
    page('1. From equity spreads to volatility exposure',[
      h('The motivating paper'),
      p('Ning and Lee (2024) construct an equity spread and learn how to trade it. They select a hedge coefficient to minimize an empirical measure of mean-reversion time, then use Q-learning with recent spread changes and current position as state. Their actions buy, hold or sell a unit. Training uses simulated Ornstein–Uhlenbeck paths; the market experiment forms ten US pairs in 2022 and trades them in 2023.'),
      p('The useful idea is to separate identifying an opportunity from deciding how to trade it. Its evidence is narrower than the general promise of reinforcement learning: one trading year does not establish behavior across regimes, and simulation requires a specified process and an equilibrium mean. Daily Sharpe figures in that study also cannot be compared directly with annualized options Sharpe. Our extension uses a longer historical record and explicit option cash flows, while inheriting its own limitations in universe construction, execution and repeated specification search.'),
      h('What transfers to options'),
      p('If two related stocks have unusually different implied volatilities, buying relatively cheap options and selling rich ones can express a relative-value view. A put-pair signal need not predict that the IV spread will converge before expiry. Its realized profit depends on option payoffs, daily stock hedges and funding. A spread extreme can also be justified by a company-specific event; similarity of stock returns does not guarantee similarity of option risk.'),
      p('The straddle signal expresses a different view. A straddle contains a call and put at the same strike and expiry. Selling it and hedging delta sells volatility exposure. If implied volatility repeatedly exceeds subsequent realized volatility, that persistence may identify favorable short-volatility conditions. The signal retains substantial jump and crash exposure, and its payoff should not be interpreted as market-neutral relative value.'),
      h('Two forms of learning'),
      p('The entry policies are contextual bandits optimized directly against historical portfolio rewards: they choose a size after an opportunity has been proposed. Direct policy search is an RL method even though it has no Bellman updates. The allocator is a sequential fitted Q learner. Its current weight changes the next inventory state and the cost of later resizing. These are related but distinct decision problems; we do not treat a profitable bandit as proof of sequential portfolio-management skill.')
    ])
    page('2. The actual-contract backtesting framework',[
      p('The retained put study inherits a 345-pair research roster. Usable prepared candidates cover 125 pairs and 161 equity or ETF underlyings; 46 pairs produce 608 screened proposals during 2018–2025. The roster and quote availability reflect earlier research, rather than a point-in-time universe designed for this comparison. All sizing policies within a signal receive the same proposals and contract choices.'),
      p('On a signal date, the simulator selects quoted contracts near the target maturity and delta. Orders execute in the following session. Quantities are whole contracts; put-pair entries require an initial vega mismatch of no more than 10%. Positions are normally held through expiry. Every day, the ledger marks options and stock, records settlement and corporate cash flows, and trades stock to restore delta neutrality. Stock hedges in the same name are netted.'),
      c('cashflows',87),
      p('Delta hedging does not make the payoff riskless. A local approximation decomposes option-and-hedge P&L into gamma gains from stock variation, theta decay, changes in implied volatility and higher-order terms. The backtest records actual marked cash flows; it does not substitute a gamma-minus-theta approximation for the ledger. Initial vega neutrality also does not remain exact as prices and expiries evolve.'),
      h('Costs and financing'),
      p('Paid option trades use midpoint plus a 0.10-vol-point adverse charge and $0.03 commission per contract. Stock turnover costs one basis point. Cash earns the prior benchmark rate; negative cash pays that rate plus 1% annually. Short stock incurs 0.3% annual borrow. Long hedge purchases reduce interest-bearing cash, while short sales increase it subject to borrow costs. Explicit dividends credit long shares and debit short shares, offsetting the ex-dividend price move in hedge accounting.'),
      p('Each source strategy begins with $10 million and shares inventory and risk limits across its own trades. An entry can be rejected even when its signal qualifies: quotes, integer sizing or existing pair/wing stress commitments may prevent it. Smaller entries can therefore change later admission. An all-cash account earning the same benchmark provides the reference for trading value.')
    ])
    page('3. Signals and contextual entry sizing',[
      h('Put pairs: relative IV with completed-profit memory'),
      p('The selected instrument is a roughly 30-day put with delta near −0.25. The signal ranks the pair’s IV difference against its preceding 252 sessions, requiring at least 63 observations. Above the 80th percentile it sells the relatively high-IV name and buys the other; below the 20th percentile it reverses. Monthly formation requires 126-session return beta between 0.85 and 1.15 and correlation of at least 0.5.'),
      p('A proposed direction also needs positive average profit per initial vega among standalone simulations completed in the preceding 91 and 365 calendar days, with at least 21 and 63 observations. Unfinished trades cannot contribute future profits. These eligibility screens are separate from learned sizing. A monthly regression tree uses 27 signal-date inputs and at most eight leaves. It fits completed full-size profit; cross-entropy search assigns each leaf 50%, 75% or 100% exposure to maximize aggregate daily training Sharpe over the preceding 18 months.'),
      h('Straddles: persistent realized-versus-implied differences'),
      p('The second signal uses the trailing completed past-year mean of subsequent 30-day RV minus entry ATM call IV as a persistence forecast. It ranks names directly on that memory and sells the bottom decile through delta-hedged ATM straddles. The original call-IV signal is retained when changing the traded instrument. A regularized linear score fitted on 60 months of completed examples groups proposals into three contexts; exhaustive search over 27 context-to-size maps selects 50%, 75% or 100% sizes using daily training Sharpe.'),
      p('Its inputs combine IV, RV-minus-IV memory, VIX, recent IV and VIX changes, and a causal fixed-half benchmark’s trailing performance and capital use. This benchmark describes conditions known before trading; it is not the learner’s future performance. Early months use 50% until enough completed examples exist. All 35,774 proposals remain available, including proposals without usable training outcomes.'),
      h('What the training objective does and does not include'),
      p('Both entry procedures optimize aggregate independent completed-trade profit paths. The funded replay then handles joint cash, integer quantities and admission. Reference exposure is $450 vega per IV point on each put-pair leg, versus $450 combined straddle vega. Put training retains $0.65 option commission, compared with $0.03 in funded evaluation. Thus the comparison measures a complete sizing-and-admission policy, not pure prediction on an identical executed trade set.')
    ])
    bandit=read(REPORT/'bandit_risk_return.csv')
    rows=[]
    profits=read('reports/straddle_policy_refinement/comparison.csv')
    puts=read('reports/p25_validation_split/periods.csv')
    for strategy in ('put','straddle'):
      for suffix in ('fixed50','fixed75','fixed100','bandit'):
        v=bandit.loc[(bandit.strategy==strategy)&(bandit.policy==suffix)&(bandit.period=='validation')].iloc[0]
        e=bandit.loc[(bandit.strategy==strategy)&(bandit.policy==suffix)&(bandit.period=='evaluation')].iloc[0]
        if strategy=='put':profit=puts.loc[(puts['case']==('screened_selected' if suffix=='bandit' else 'screened_'+suffix))&(puts.period=='retrospective_holdout'),'profit_above_cash'].iloc[0]
        else:profit=profits.loc[(profits['case']==('performance60' if suffix=='bandit' else suffix))&(profits.period=='evaluation'),'profit_above_cash'].iloc[0]
        rows.append([('Put pairs' if strategy=='put' else 'Straddles')+' / '+('bandit' if suffix=='bandit' else suffix.replace('fixed','fixed ')),f'{v.annualized_excess_sharpe:.3f}',f'{e.annualized_excess_sharpe:.3f}',str(e.entries),f'${profit/1e6:.3f}m'])
    page('4. Do the entry bandits improve the signals?',[
      p('Within each signal, the retained bandit has higher historical excess Sharpe than all three fixed entry-size benchmarks in both reporting periods. The economic scale differs: the straddle strategy earns substantially more money and carries substantially more return volatility. A similar Sharpe does not imply the same capital use, risk exposure or dollar opportunity.'),
      table(['Strategy / entry size','2018–21 Sh.','2022–25 Sh.','Later entries','Later profit'],rows,[174,79,79,80,100]),
      c('bandit_results',177),
      p('For put pairs, the bandit raises later Sharpe from 1.055 at full size to 1.263 and profit above cash by approximately $47,800. For straddles, the corresponding figures are 0.990 to 1.041 and approximately $173,000 additional profit. Existing paired-block intervals for the sizing gains include zero. These are descriptive gains for retained historical procedures, rather than reliable estimates of a repeatable learning advantage.')
    ])
    page('5. What do the bandits learn?',[
      c('bandit_explanations',180),
      p('The put audit follows every actual root-to-leaf path and reads the learned size assigned to that leaf. Completed 365-day profit accounts for 18.7% of later visited nodes; mean IV, commission relative to vega, and return correlation account for 14.9%, 12.7% and 12.3%. The model uses recent profitability, volatility level and trade economics jointly. It does not implement a single universal rule such as “more expensive volatility always deserves more exposure.”'),
      p('The straddle audit adds standardized ridge-score contributions exactly, then applies score thresholds and the saved size map. Centering IV changes 36.6% of fitted later proposal sizes; VIX change changes 15.3%, normalized RV-minus-IV memory 12.9%, and VIX level 11.9%. Larger scores need not mean larger positions because the subsequent action map maximizes aggregate Sharpe rather than enforcing monotonicity.'),
      table(['Later admission versus fixed full','Both enter','Bandit only','Fixed only'],[
        ['Put-pair proposals','74','36','1'],['Straddle proposals','897','307','43']],[260,84,84,84]),
      p('Admission is part of the mechanism. The put bandit enters 110 later trades versus 75 at full size; 36 are unique to the bandit. The straddle bandit enters 1,204 versus 940, including 307 unique entries. Smaller commitments can release capacity for later opportunities. These differences mean the funded improvement cannot be attributed solely to recognizing which identical trades deserve a larger size.'),
      p('There is some outcome separation: among quote-available completed straddle labels, later full-size profit averages $1,631 for proposals assigned 100%, versus $16 for those assigned 50%. That selected label subset is not the full live universe. Three within-date size shuffles produce later Sharpes of 0.834, 0.928 and 1.044; one exceeds the retained 1.041. The evidence for name-specific predictive sizing is therefore weaker than the funded policy comparison alone suggests.')
    ])
    page('6. Sequential allocation across the two signals',[
      p('We now freeze the best retained put-pair and straddle entry policies. The outer learner chooses one capital split each week: 100/0, 90/10, 75/25, 50/50, 25/75, 10/90 or 0/100 put/straddle. Weights sum to one. There is no outer leverage, cash action or added portfolio stop. All seven constant policies use the same action menu and start at their own target split.'),
      p('This experiment allocates fractional units of the two funded strategy indices. It preserves their option, hedge and funding returns and charges an additional source-based resizing cost. It is not a joint replay of shared integer option contracts and admission across strategies. The combination is not vega neutral: allocating to short straddles introduces net short-vega exposure.'),
      h('State, reward and transition'),
      p('The state contains put-spread strength, the oriented put variance-spread feature, straddle IV-versus-RV richness, completed straddle premium, relative source volatility, source return correlation, VIX, relative recent Sharpe, and the previous straddle weight. Market inputs use the prior close. The chosen allocation executes at the next close: the first session remains at old weights, after which the new allocation earns returns. Holdings drift between decisions and affect the cost of the next resize.'),
      p('For a completed week, reward is the sum of daily excess returns divided by training-window reference volatility, minus a variance penalty λ times the sum of squared standardized daily returns. λ controls risk preference. The independent discount γ = 0.99 controls the weight placed on continuation value; it is not 1 − λ.'),
      h('Fitting the action-value function'),
      p('We represent Q(s,a) as a quadratic in the straddle weight a, with coefficients linear in standardized state variables. Monthly fitted Q iteration uses the preceding 104 completed weeks, requiring at least 52. For each historical week it evaluates the seven previous allocations crossed with seven new actions. Each week has total weight one; these 49 counterfactuals are not 49 independent market observations. Ridge regularization and 200 Bellman updates are held fixed. Q uses 50/50 before sufficient history is available; fixed controls do not inherit that fallback.'),
      p('The matched discount-zero learner uses the same reward, state, training data and regularization but ignores continuation. Comparing it with γ = 0.99 separates a one-period contextual allocation policy from the additional effect of sequential valuation.')
    ])
    fixed=read(RESULT/'fixed_results.csv');q=read(RESULT/'q_results.csv')
    rows=[]
    for weight in [0,.1,.25,.5,.75,.9,1]:
      f=fixed.loc[fixed.straddle_target==weight].set_index('period');v,e=f.loc['validation'],f.loc['evaluation']
      rows.append([f'Fixed {1-weight:.0%} / {weight:.0%}',f'{v.annualized_excess_sharpe:.3f}',f'{e.annualized_excess_sharpe:.3f}',f'${e.profit_above_cash/1e6:.3f}m'])
    for risk in [.1,.2]:
      f=q.loc[(q.gamma==.99)&(q.risk==risk)].set_index('period');v,e=f.loc['validation'],f.loc['evaluation']
      rows.append([f'Dynamic Q, λ = {risk:g}',f'{v.annualized_excess_sharpe:.3f}',f'{e.annualized_excess_sharpe:.3f}',f'${e.profit_above_cash/1e6:.3f}m'])
    page('7. The profit–Sharpe tradeoff',[
      c('q_evaluation',213),
      table(['Put / straddle capital split','2018–21 Sharpe','2022–25 Sharpe','Later profit'],rows,[220,97,97,98]),
      p('The penalty-0.20 Q policy has later Sharpe 1.726, exceeding the highest of the seven fixed splits, 1.612 at 90/10. Its $1.306 million profit is also greater than fixed 50/50’s $1.238 million. The penalty-0.10 policy earns $1.388 million at Sharpe 1.491. Straddle-heavy fixed portfolios earn still more dollars at lower Sharpe: Q does not dominate every policy on every objective.'),
      p('The two Q settings are highlighted researched operating points. The highest-validation-Sharpe Q setting is λ = 0.025, whose later Sharpe is only 0.282. Appendix A shows the complete risk-preference sweep. This distinction prevents an attractive later point from being mistaken for the success of a validation-only selection rule.')
    ])
    page('8. Why does dynamic allocation do better?',[
      c('q_context',148),
      p('All fitted allocation decisions are reproduced from saved models. For each decision, the chosen-versus-runner-up Q gap is decomposed exactly into an intercept and state-feature contributions. Setting one feature to its monthly training center changes 13.6% of later penalty-0.20 decisions for VIX, 12.6% for straddle IV-versus-RV richness, and 11.0% for put-spread strength. Correlated inputs can substitute for one another; a sensitivity measure does not show that a feature causes profit.'),
      c('q_schedule',136),
      p('The policy behaves mainly as a regime switch. Of 191 later decisions, 141 choose the put strategy alone and 46 the straddle strategy alone; only four use interior splits. Its mean straddle target is 24.9%. In later weeks below the sample median VIX, mean straddle weight is 44.9%; above it, 4.7%. This is a description of the saved decisions, not a newly selected VIX threshold. The earlier-period pattern differs, so the explanation is conditional on the monthly fitted model rather than a stable one-variable trading rule.'),
      p('Frozen-policy replays support a conditional role for these inputs. Centering put-spread strength lowers later Sharpe to 1.400; centering straddle richness lowers it to 1.593. Yet centering previous weight slightly raises Sharpe to 1.733. These are interventions in the saved policy, not retrained feature removals, and they do not show that every input is necessary. The economic reading is selective participation in a larger, riskier short-volatility opportunity. The following controls test timing against average exposure.')
    ])
    controls=read(REPORT/'q_controls.csv');ct=controls.loc[controls.case=='q_lambda0.2_g0.99']
    rows=[]
    for key,label in [('same_mean_target_hindsight','Fixed at Q’s average target'),('saved_gamma_zero','One-period learner, γ = 0'),('actual_q','Sequential learner, γ = 0.99')]:
      f=ct.loc[ct.control==key].set_index('period');v,e=f.loc['validation'],f.loc['evaluation']
      rows.append([label,f'{v.annualized_excess_sharpe:.3f}',f'{e.annualized_excess_sharpe:.3f}',f'${e.profit_above_cash/1e6:.3f}m'])
    page('9. Timing, exposure and continuation controls',[
      c('q_timing',165),
      table(['Penalty-0.20 policy or control','2018–21 Sharpe','2022–25 Sharpe','Later profit'],rows,[220,97,97,98]),
      p('Holding the later average target of 24.9% straddles constant gives Sharpe 1.387 and $841,251 profit. Q reaches 1.726 and $1.306 million under the same source returns and resizing model. The earlier exposure-matched control also trails Q: 0.828 versus 1.215. Thus the gain is not explained by the mean capital split alone. These average-weight controls use the realized schedule retrospectively and are diagnostic rather than deployable benchmarks.'),
      p('We also shift the learned action schedule through 49 nonzero circular offsets within each reporting period, keeping the action frequencies and most serial structure. Costs and wealth are recomputed; the genuine early fallback and pre-period portfolio history are retained. The later median shifted Sharpe is 0.744, and none of the 49 offsets reaches Q’s 1.726. Three of 49 earlier shifts exceed Q. This supports a timing interpretation conditional on the researched schedule, not a selection-adjusted significance claim.'),
      p('The discount-zero learner reaches later Sharpe 1.645 and $1.184 million, compared with 1.726 and $1.306 million for Q. At identical actual states, continuation changes 9 of 191 later action choices for this risk preference, and 28 for penalty 0.10. Much of the improvement over static mixtures is available through conditional one-period allocation; the sequential extension adds a smaller historical increment. It would overstate the evidence to attribute the entire fixed-versus-Q gap to long-horizon planning.')
    ])
    page('10. Conclusions, limitations and next tests',[
      h('What the experiments show'),
      p('The retained entry bandits improve the historical risk-return tradeoff of two distinct signals. Put sizing depends strongly on completed profitability, IV level and trade economics. Straddle sizing depends on IV, market-volatility conditions and completed realized-versus-implied history. Both also change funded admission, so their gains combine exposure choices with different executed portfolios.'),
      p('Once those policies are frozen, dynamic allocation improves on the seven tested fixed capital mixes at selected historical operating points. Exposure-matched and shifted-schedule controls indicate that allocation timing matters. Exact Q-gap explanations identify the information used by the policy, while the matched discount-zero comparison shows that sequential valuation adds less than the entire dynamic-versus-static improvement. The practical contribution is a layered framework whose decisions can be audited, not a claim that every RL component is independently necessary.'),
      h('Limits on interpretation'),
      p('The 2018–2021 and 2022–2025 labels define reporting periods, not a new untouched experiment. The put specification was selected using earlier-period scores, but its contract segment and universe emerged from previous research. The straddle procedure explicitly searched both historical periods, and the highlighted Q risk preferences were investigated after examining later outcomes. Chronological model fits prevent unfinished outcomes from entering those fits; they cannot reverse specification-level selection.'),
      p('The midpoint-plus-penalty execution model still needs live or observed-spread validation. Pair trades carry legging risk, and American options introduce assignment and pin risks not fully represented by expiry approximations. Liquidity, borrow changes, jumps and forced liquidation can matter. The outer allocator trades fractional strategy indices rather than jointly resizing a shared integer-contract book. These modeling limits are economically important even when a historical Sharpe is high.'),
      h('A credible next experiment'),
      p('Freeze the two signal specifications, cost model, action menus, training rules and risk preference before collecting new evaluation data. Reproduce allocation in a shared contract-level ledger, including netted stock, actual constraints and execution measurements. Then test whether the same state-dependent decisions add value after realistic fills and whether the benefit survives concentration and uncertainty checks. Interpretability should remain part of that evaluation: the question is whether readable allocation behavior repeats, not whether another historical search can produce a larger headline.')
    ])
    allq=q.loc[(q.gamma==.99)&(q.period!='full')]
    sweep=[]
    for risk in [0,.025,.05,.1,.2]:
      f=allq.loc[allq.risk==risk].set_index('period');v,e=f.loc['validation'],f.loc['evaluation']
      sweep.append([f'{risk:g}',f'{v.annualized_excess_sharpe:.3f}',f'${v.profit_above_cash/1e6:.3f}m',f'{e.annualized_excess_sharpe:.3f}',f'${e.profit_above_cash/1e6:.3f}m'])
    page('Appendix A. Validation and the complete Q sweep',[
      c('q_validation',203),
      table(['Variance penalty λ','Earlier Sharpe','Earlier profit','Later Sharpe','Later profit'],sweep,[120,98,98,98,98]),
      p('Both highlighted risk preferences beat every fixed split on earlier Sharpe, but neither maximizes validation Sharpe within the complete Q sweep. The λ = 0.025 validation winner fails later. We therefore report the strong λ = 0.10 and 0.20 operating points as researched settings and preserve the weaker alternatives rather than present the later winner as validation-selected.'),
      p('Discount 0.5 and 0.99 were compared with the same 200 Bellman updates and unchanged reward, features and costs. The 0.5 policies produced the same Sharpes at 20 and 200 updates. For λ = 0.20, raising γ improves later Sharpe from 1.650 to 1.726 and profit from $1.179 million to $1.306 million. The effective value horizon changes; the risk penalty is held fixed.')
    ])
    annual=read(REPORT/'q_annual.csv');annual=annual.loc[annual.case=='q_lambda0.2_g0.99']
    ar=[[str(int(r.year)),f'{r.average_straddle_weight:.1%}',f'{r.annualized_excess_sharpe:.3f}',f'${r.profit_above_cash:,.0f}'] for r in annual.itertuples()]
    page('Appendix B. Annual results and failed reward designs',[
      table(['Year','Mean straddle target','Q excess Sharpe','Change in profit above cash'],ar,[80,130,120,182]),
      p('The mean target is averaged across weekly decisions. 2025 is partial, ending 29 August. Annual excess Sharpe can be negative while the dollar gap over the original all-cash account increases: interest earned on gains accumulated in earlier years contributes to that dollar gap. The daily excess-return definition and the dollar wealth-gap definition answer different accounting questions.'),
      h('Reward-function checks retained as failed trials'),
      p('Three follow-ups replaced the original variance-penalty reward with explicit profit–Sharpe blends: a local or exact change in a fixed-reference training Sharpe; return normalized by the candidate allocation’s forecast risk; and a Sharpe change around a training-selected reference mix. Their references and risk estimates used completed training data. None improved the later profit–Sharpe tradeoff of the retained risk-adjusted reward.'),
      table(['Reward family','Best later Sharpe among its tested settings'],[
        ['Fixed-reference Sharpe / profit blend','0.714'],
        ['Allocation-specific forecast-risk / profit blend','0.714'],
        ['Training-reference Sharpe / profit blend','0.714'],
        ['Original risk-adjusted Q, λ = 0.20','1.726']],[330,182]),
      p('The reward-family figures are retrospective maxima shown only to summarize failed experiments; they are not validation-selected performance. Explicitly naming a reward “Sharpe” does not make fitted one-step rewards maximize the eventual portfolio Sharpe. The successful historical result depends on the full combination of signal returns, state representation, reward, regularization and action timing.')
    ])
    page('Appendix C. Reproducibility and performance definitions',[
      h('Accounting definitions'),
      p('Let V(t) be account NAV, B(t) the all-cash benchmark, r(t) = V(t)/V(t−1) − 1, and c(t) = B(t)/B(t−1) − 1. Daily excess return is e(t) = r(t) − c(t). Annualized excess Sharpe is √252 times the sample mean of e divided by its sample standard deviation. Annualized total account return compounds 1+r; annualized growth relative to cash compounds (1+r)/(1+c). Annualization uses elapsed calendar time for compounded growth and 252 sessions for Sharpe.'),
      p('Cumulative dollar profit above cash is V(t) − B(t). A reporting-period profit is the change in this gap from the prior reporting boundary. It includes interest on previously accumulated gains. These dollar differences are not identical to compounding daily excess returns. Accounts continue through the 2022 boundary with their own cash and holdings.'),
      h('Exact explanations and controls'),
      p('For Q, standardized state z includes an intercept and nine inputs, clipped at five training standard deviations. The score is Q(s,a) = β₀·z + a β₁·z + a² β₂·z. For chosen weight a and runner-up b, feature j contributes zⱼ[(a−b)β₁ⱼ + (a²−b²)β₂ⱼ] to the score gap. These contributions sum exactly to the observed gap. The immediate component uses the matched discount-zero coefficients at the same state; the remaining gap is continuation. It explains model preference, not realized causal profit.'),
      p('All 608 put and 35,774 straddle saved sizes reproduce exactly. Q return paths and NAV reproduce the retained ledgers, including costs. Every explained model is checked for completed training outcomes before its fit cutoff and availability before its decision. Rotations preserve action frequencies and shift only fitted decisions within each period. There are 49 deterministic nonzero offsets per period and policy; their rank is descriptive, not an adjusted p-value.'),
      h('Sources and references'),
      p('The original submitted PDF, LaTeX and older Word companion are copied alongside this research version; their hashes are checked unchanged. Data tables and audit protocols are in reports/allocation_interpretability_099. Full Q results are in reports/signal_allocation_discount; all reward trials remain in reports/signal_reward_comparison. The accompanying manifest records source hashes, figure files and layout checks. Vendor market data retain their original institutional license restrictions.'),
      p('[1] Ning, B., and Lee, K. (2024). Advanced Statistical Arbitrage with Reinforcement Learning. arXiv:2403.12180v1. https://arxiv.org/abs/2403.12180'),
      p('[2] de Boer, P.-T., Kroese, D. P., Mannor, S., and Rubinstein, R. Y. (2005). A Tutorial on the Cross-Entropy Method. Annals of Operations Research, 134, 19–67.'),
      p('[3] scikit-learn documentation. DecisionTreeRegressor. The retained source models and exact tree thresholds are archived in the repository.')
    ])
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'manuscript.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False)+'\n')
    text='\n\n'.join(page['title']+'\n'+'\n\n'.join(b.get('text','') for b in page['blocks'] if b['type']!='chart') for page in pages)
    (DEST/'report_text.txt').write_text(text+'\n')
    print(f'Saved {len(pages)} pages: {len(pages)-3} main pages and three appendices',flush=True)


def render(part,only_page=None):
    from scripts import build_research_publication as pub
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pages=json.loads((DEST/'manuscript.json').read_text())
    if only_page is None:
        raise ValueError('Use --page N; each rendered page requires a fresh guarded worker')
    pub.FIG=FIG;pub.CHARTS={name:dict(caption=caption) for name,caption in FIGURES.items()}
    pub.PAPER_FIGURES={name:i+1 for i,name in enumerate(FIGURES)}
    class ResearchPage(pub.PaperPage):
        def text(self,x,y,text,size=10.35,color='#202832',family='serif',weight='normal'):
            # Dollar amounts are prose, not Matplotlib math delimiters.
            return self.fig.text(x/612,y/792,text,fontsize=size,color=color,
                                 family=family,weight=weight,va='top',parse_math=False)
    first,last=(0,7) if part==1 else (7,len(pages));checks=[]
    if only_page is not None:first,last=only_page-1,only_page
    for index in range(first,last):
        page=pages[index];canvas=ResearchPage(index+1,page['title'],page.get('subtitle'))
        for obj in list(canvas.fig.texts):
            if obj.get_text()=='RESEARCH DRAFT · 4 OCTOBER 2026':obj.set_text('RESEARCH COPY · 15 OCTOBER 2026')
        for block in page['blocks']:
            if block['type']=='paragraph':canvas.paragraph(block['text'])
            elif block['type']=='heading':canvas.heading(block['text'])
            elif block['type']=='chart':canvas.image(block['name'],block['height'])
            elif block['type']=='table':canvas.table(block)
        check=dict(page=index+1,bottom_content_points=canvas.y,overflow=canvas.y<49)
        checks.append(check)
        with PdfPages(DEST/f'page_{index+1:02d}.pdf') as pdf:pdf.savefig(canvas.fig)
        canvas.fig.savefig(DEST/f'page_{index+1:02d}.png',dpi=90)
        plt.close(canvas.fig);del canvas;gc.collect()
        (DEST/f'layout_page_{index+1:02d}.json').write_text(json.dumps(check,indent=2)+'\n')
        print(json.dumps(check),flush=True)
    (DEST/f'layout_part{part}.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)
    if any(r['overflow'] for r in checks):raise ValueError('Paper page overflow; adjust this copy only')


def assemble():
    from pypdf import PdfReader,PdfWriter
    pages=json.loads((DEST/'manuscript.json').read_text())
    checks=[json.loads((DEST/f'layout_page_{i:02d}.json').read_text()) for i in range(1,len(pages)+1)]
    assert len(checks)==len(pages) and not any(r['overflow'] for r in checks)
    writer=PdfWriter()
    for i in range(1,len(pages)+1):writer.append(PdfReader(DEST/f'page_{i:02d}.pdf'))
    writer.add_metadata({'/Title':'Learning Exposure in Options Relative Value','/Author':'Adam Kreitzman and Soo Xin Yang','/Subject':'Separate interpretability research copy; 15 October 2026'})
    with (DEST/'options_relative_value_interpretability.pdf').open('wb') as out:writer.write(out)
    writer.close()
    original=json.loads((DEST/'originals_manifest.json').read_text())
    for path,signature in original['originals'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==signature
    inputs=[REPORT/'q_protocol.json',REPORT/'bandit_protocol.json',DEST/'manuscript.json',Path(__file__),RESULT/'results.csv']
    manifest=dict(pages=len(pages),main_pages=len(pages)-3,appendices=3,layout_checks=checks,originals_unchanged=True,
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  figures=FIGURES,no_models_refit=True)
    (DEST/'publication_audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Portable HTML is an additional editable/viewable representation.
    body=[]
    for page in pages:
        blocks=['<h1>'+html.escape(page['title'])+'</h1>']
        for b in page['blocks']:
            if b['type'] in ('paragraph','heading'):blocks.append(('<h2>' if b['type']=='heading' else '<p>')+html.escape(b['text'])+('</h2>' if b['type']=='heading' else '</p>'))
            elif b['type']=='chart':blocks.append(f'<figure><img src="figures/{b["name"]}.svg"><figcaption>{html.escape(FIGURES[b["name"]])}</figcaption></figure>')
            elif b['type']=='table':blocks.append('<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in [b['headers']]+b['rows'])+'</table>')
        body.append('<section>'+''.join(blocks)+'</section>')
    css='body{font:11pt/1.4 Georgia;color:#202832;background:#edf1f5}section{background:white;max-width:760px;margin:24px auto;padding:48px;break-after:page}h1,h2{font-family:Arial;color:#086ba5}h1{font-size:21px}h2{font-size:16px}img{width:100%}figure{margin:14px 0}figcaption{font:9pt Arial;color:#536572}table{border-collapse:collapse;width:100%;font:10pt Arial}td{padding:7px;border-bottom:1px solid #ddd}@media print{body{background:white}section{margin:0;padding:0}}'
    (DEST/'options_relative_value_interpretability.html').write_text('<!doctype html><meta charset="utf-8"><title>Learning Exposure in Options Relative Value</title><style>'+css+'</style>'+''.join(body))
    def escape(text):
        substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
        return ''.join(substitutions.get(c,c) for c in text)
    tex=[r'\documentclass[11pt,letterpaper]{article}',r'\usepackage[margin=1in]{geometry}',r'\usepackage{fontspec,graphicx,booktabs,longtable,hyperref}',r'\graphicspath{{figures/}}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{0.5em}',r'\begin{document}']
    for i,page in enumerate(pages):
        if i:tex.append(r'\clearpage')
        tex.append(r'\section*{'+escape(page['title'])+'}')
        if page.get('subtitle'):tex.append(escape(page['subtitle'])+r'\par')
        for b in page['blocks']:
            if b['type']=='paragraph':tex.append(escape(b['text'])+'\n')
            elif b['type']=='heading':tex.append(r'\subsection*{'+escape(b['text'])+'}')
            elif b['type']=='chart':tex.append(r'\begin{center}\includegraphics[width=\textwidth]{'+b['name']+r'.pdf}\end{center}')
            elif b['type']=='table':
                tex.append(r'\begin{longtable}{'+('l'*len(b['headers']))+'}'+r'\toprule')
                tex.append(' & '.join(map(escape,b['headers']))+r'\\\midrule')
                for row in b['rows']:tex.append(' & '.join(escape(str(v)) for v in row)+r'\\')
                tex.append(r'\bottomrule\end{longtable}')
    tex.append(r'\end{document}')
    (DEST/'options_relative_value_interpretability.tex').write_text('\n'.join(tex)+'\n')
    print('Assembled research PDF; original report hashes unchanged',flush=True)


def docx():
    from scripts import export_research_docx as word
    pages=json.loads((DEST/'manuscript.json').read_text())
    path=DEST/'options_relative_value_interpretability.docx'
    def verify_copy(path,pages,charts,numbers):
        # Keep the original exporter's content checks, but derive counts from
        # this manuscript rather than its old ten-page/two-table constants.
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            document=word.ET.fromstring(archive.read('word/document.xml'))
            paragraphs=[''.join(p.itertext()) for p in document.findall('.//w:p',word.NS)]
            for page in pages:
                assert page['title'] in paragraphs
                for block in page['blocks']:
                    if block['type'] in ('paragraph','heading','equation'):assert block['text'] in paragraphs
                    elif block['type']=='chart':
                        name=block['name']
                        assert f'Figure {numbers[name]}. '+charts[name]['caption'] in paragraphs
                        assert archive.read(f'word/media/{name}.png')==(FIG/f'{name}.png').read_bytes()
                    elif block['type']=='table':
                        for row in [block['headers']]+block['rows']:assert all(str(v) in paragraphs for v in row)
            blocks=[b for p in pages for b in p['blocks']]
            assert len(document.findall('.//w:tbl',word.NS))==sum(b['type']=='table' for b in blocks)
            assert len(document.findall('.//w:drawing',word.NS))==sum(b['type']=='chart' for b in blocks)
            assert len(document.findall('.//w:pageBreakBefore',word.NS))==len(pages)-1
    word.verify=verify_copy
    word.export(pages,{name:dict(caption=caption) for name,caption in FIGURES.items()},
           {name:i+1 for i,name in enumerate(FIGURES)},path)
    # The shared exporter carries the old report date in its Word header.
    with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
    parts['word/header1.xml']=parts['word/header1.xml'].replace(b'6 OCTOBER 2026',b'15 OCTOBER 2026 RESEARCH COPY')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    print('Saved editable Word version',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['figures','manuscript','render1','render2','assemble','docx']);parser.add_argument('--page',type=int);parser.add_argument('--only-cashflows',action='store_true');args=parser.parse_args()
    if args.phase.startswith('render'):render(int(args.phase[-1]),args.page)
    elif args.phase=='figures':figures(args.only_cashflows)
    else:globals()[args.phase]()
