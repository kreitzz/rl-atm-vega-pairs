"""Rewrite the illustrated ten-page report from saved results; never fit models or edit slides."""
from pathlib import Path
import hashlib
import html
import json
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

from scripts import build_research_publication as pub

ROOT, PAPER, FIG = pub.ROOT, pub.PAPER, pub.FIG
BLUE, GRAY, GREEN, ORANGE, RED = pub.BLUE, pub.GRAY, pub.GREEN, pub.ORANGE, pub.RED
p, h, eq, chart, table = pub.p, pub.h, pub.eq, pub.chart, pub.table
DATA = PAPER / 'figure_data'
TITLE = 'From Equity Pairs to Options Relative Value'
DATE = '6 October 2026'


def record(name, df):
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / f'{name}.csv', index=False)


def save(name, fig, caption, sources, data=None):
    for path in sources:
        pub.source(path)
    pub.savechart(name, fig, caption, sources)
    if data is not None:
        record(name, data)
        pub.CHARTS[name]['plotted_data'] = f'figure_data/{name}.csv'


def boxes(name, titles, bodies, caption, source_paths, colors=None):
    fig, ax = plt.subplots(figsize=(8.1, 1.65))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.5); ax.axis('off')
    n = len(titles)
    width = 10 / n
    for i, (title, body) in enumerate(zip(titles, bodies)):
        x = i * width + .07
        color = (colors or [BLUE] * n)[i]
        ax.add_patch(pub.FancyBboxPatch((x, .2), width-.3, 1.05,
                     boxstyle='round,pad=.06', facecolor='#f1f5f9', edgecolor='#c8d6e3'))
        ax.text(x+(width-.3)/2, 1.03, title, ha='center', va='center',
                fontsize=10, weight='bold', color=color)
        ax.text(x+(width-.3)/2, .65, body, ha='center', va='center', fontsize=9)
        if i < n-1:
            ax.annotate('', xy=(x+width-.08, .7), xytext=(x+width-.25, .7),
                        arrowprops={'arrowstyle': '->', 'color': GRAY})
    fig.tight_layout(pad=.3)
    save(name, fig, caption, source_paths)


def extra_charts():
    boxes('research_map', ['Find a signal', 'Learn exposure', 'Test extensions'],
          ['Actual contracts\nSmile + features', 'Frozen proposals\n50 / 75 / 100%', 'Exits + maturities\nPortfolio allocation'],
          'Research design: separate discovery of tradable relative value from allocation and subsequent extensions. The comparisons below retain their individual data and accounting conventions.',
          ['slides/230ZB_volatility_spreads (1).pptx'])

    path = 'reports/p25_maturity_matched/comparison.csv'
    raw = pub.frame(path)
    rows = raw[(raw.variant.eq('floor')) & raw.wing.isin(['P25', 'ATM', 'P75'])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.55))
    labels = {'P25': '25-delta puts', 'ATM': 'ATM', 'P75': '75-delta puts'}
    order = ['P25', 'ATM', 'P75']
    for ax, tenor in zip(axes, [30, 90]):
        d = rows[rows.tenor.eq(tenor)].set_index('wing').loc[order]
        vals = d.annualized_excess_sharpe
        ax.bar(np.arange(3), vals, color=[BLUE, GRAY, ORANGE], width=.6)
        for i, r in enumerate(d.itertuples()):
            ax.annotate(f'{r.annualized_excess_sharpe:.3f}\n{r.entries} entries',
                        (i, r.annualized_excess_sharpe), xytext=(0, 5 if r.annualized_excess_sharpe >= 0 else -5),
                        textcoords='offset points', ha='center',
                        va='bottom' if r.annualized_excess_sharpe >= 0 else 'top', fontsize=8)
        ax.set_xticks(range(3), [labels[k] for k in order], fontsize=8)
        ax.set_ylim(-.54, .46); ax.axhline(0, color=GRAY, lw=.8)
        ax.set_title(f'{tenor}-day coverage-matched experiment', fontsize=10)
        pub.style(ax, 'Annualized excess Sharpe')
    fig.tight_layout()
    save('smile_transfer', fig,
         'Restricted 30D/90D formation comparison with learned positive sizing floors and matched proposal counts within each wing (P25 213, ATM 222, P75 55). P25 remains strongest at 30D; all three weaken at 90D. This restricted sample is distinct from the retained 608-proposal checkpoint.', [path], rows)

    path = 'slides/chart_data/example_amd_nvda_iv_spread_daily.csv'
    entries_path = 'slides/chart_data/example_amd_nvda_proposals_and_entries.csv'
    d = pub.frame(path); d.date = pd.to_datetime(d.date)
    d = d[d.date.between('2019-01-01', '2019-12-31')]
    e = pub.frame(entries_path)
    e = e[e.case.eq('10m_learned') & e.status.eq('entered')].copy()
    e.signal_date = pd.to_datetime(e.signal_date)
    e = e[e.signal_date.between('2019-01-01', '2019-12-31')]
    fig, ax = plt.subplots(figsize=(8.1, 2.35))
    ax.plot(d.date, d.iv_spread_vol_points, color=BLUE, lw=1.25, label='AMD minus NVDA IV')
    for col, label in [('spread_20th_percentile_vol_points', 'Prior 20th percentile'),
                       ('spread_80th_percentile_vol_points', 'Prior 80th percentile')]:
        ax.plot(d.date, d[col], color=GRAY, linestyle='--', lw=.9, label=label)
    ax.scatter(e.signal_date, e.signal_iv_spread_vol_points, marker='^', s=50, color=GREEN,
               zorder=4, label='Signal leading to learned entry')
    pub.style(ax, 'IV spread (volatility points)')
    ax.legend(loc='upper right', fontsize=7.5, frameon=False, ncol=2)
    ax.axhline(0, color=GRAY, lw=.6); fig.tight_layout()
    save('signal_example', fig,
         'AMD/NVDA, 2019: daily rolling selected-contract P25 IV spread and strictly prior rank boundaries. Green markers are signal dates; executions occur next session (12/13 August signals, 13/14 August fills). Gaps remain where either quote is unavailable; this is not the held-cohort mark series.', [path, entries_path], d)

    path = 'reports/p25_recency/comparison.csv'
    d = pub.frame(path).set_index('model').loc[['6M_floor', '9M_floor', '12M_floor', '18M_floor', '24M_floor', '5Y_reference']].reset_index()
    fig, ax = plt.subplots(figsize=(8.1, 1.95))
    vals = d.annualized_excess_sharpe
    ax.bar(range(len(d)), vals, color=[BLUE if x=='18M_floor' else GRAY for x in d.model], width=.6)
    ax.set_xticks(range(len(d)), ['6 months', '9 months', '12 months', '18 months', '24 months', '5 years'])
    for i, v in enumerate(vals): ax.text(i, v+.025, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.12); pub.style(ax, 'Annualized excess Sharpe'); fig.tight_layout()
    save('recency_search', fig,
         'Monthly positive-floor sizing with alternative training windows, original $345m research reference. The retained 18-month model scores 0.962 versus 0.826 for the five-year reference. These are selected development results; shorter history does not improve monotonically.', [path], d)

    boxes('allocation_model', ['Completed history', 'State partitions', 'Size allocation'],
          ['18-month rolling window\nOutcomes mature first', 'Observable 33 inputs\nTree: at most 8 leaves', 'CEM policy search\nJoint daily-path Sharpe'],
          'Retained contextual learner. Tree partitions map observable states to positive vega budgets; CEM searches allocations jointly. No Bellman continuation, learned exit, or direction reversal enters this checkpoint.',
          ['config/p25_variance_premium_candidate.json'])

    path = 'reports/p25_short_management/comparison.csv'
    d = pub.frame(path).set_index('case')
    qpath = 'reports/long_tenor_fitted_q/comparison.csv'
    q = pub.frame(qpath)
    q = q[q.tenor.eq(180)].set_index('case')
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.3))
    specs = [(axes[0], '30D: direct policy search',
              ['Hold', '18M manage', '36M manage'],
              [d.loc[k, 'annualized_excess_sharpe'] for k in ['30D_current_hold', '30D_manage_18M', '30D_manage_36M']]),
             (axes[1], '180D: fitted-Q management', ['Hold', 'Conservative Q'],
              [q.loc[k, 'annualized_excess_sharpe'] for k in ['fixed_50_hold', 'fixed50_fitted_q_conservative_aligned_runoff']])]
    plotted=[]
    for ax, title, names, vals in specs:
        ax.bar(range(len(vals)), vals, color=[GRAY]+[BLUE]*(len(vals)-1), width=.6)
        ax.set_xticks(range(len(vals)), names, fontsize=8)
        ax.set_ylim(0, 1.22); ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals): ax.text(i, v+.03, f'{v:.3f}', ha='center', fontsize=8)
        pub.style(ax, 'Annualized excess Sharpe')
        plotted.extend(dict(experiment=title, policy=k, sharpe=v) for k,v in zip(names,vals))
    fig.tight_layout()
    save('management_results', fig,
         '$10m live-admission replays. At 30D, 36-month management adds 0.027 Sharpe, while 18-month management reduces it. At 180D, conservative fitted Q adds 0.018. Algorithms, training schedules and strategies differ across panels; compare controls within each panel.', [path, qpath], pd.DataFrame(plotted))

    path = 'reports/p25_reversion_q/comparison.csv'
    d = pub.frame(path).set_index('case')
    cases=['frozen_10m_learned', 'matched_frozen_sized_q_managed_locked_v3_risk0',
           'universe_q_fixed_managed_v3_risk0', 'universe_q_sized_managed_v3_risk0']
    labels=['Retained expiry hold', 'Same 212 entries + Q exits', 'Broad Q timing, fixed', 'Broad Q timing + sizing']
    fig, ax = plt.subplots(figsize=(8.1, 2.15))
    vals = d.loc[cases, 'annualized_excess_sharpe'].to_numpy()
    ax.barh(range(4), vals, color=[BLUE, ORANGE, GRAY, GRAY]); ax.invert_yaxis()
    ax.set_yticks(range(4), labels, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v+(.025 if v>=0 else -.025), i, f'{v:.3f}', ha='left' if v>=0 else 'right', va='center', fontsize=9)
    ax.set_xlim(-.48, 1.2); ax.axvline(0, color=GRAY, lw=.8)
    pub.style(ax); ax.set_xlabel('Annualized cash-excess Sharpe'); fig.tight_layout()
    save('reversion_q', fig,
         'Paper-inspired actual-P&L fitted Q, plain-P&L reward. The locked exit control preserves original accepted entries/counts (a counterfactual attribution control). Broad cases preserve all 17,967 cached evaluation candidates; 698/729 trades execute. Final model/execution v3 only; risk-penalized variants are separately reported and also fail to beat the checkpoint.', [path], d.loc[cases].reset_index())

    path = 'reports/native_year_entry/replacement_blend/walk_forward_24M/comparison.csv'
    twopath = 'reports/native_year_entry/replacement_blend/walk_forward_24M_30D_180D/comparison.csv'
    a=pub.frame(path); b=pub.frame(twopath)
    a=a[a.period.eq('full_including_equal_warmup')].set_index('case')
    b=b[b.period.eq('full_including_equal_warmup')].set_index('case')
    fig, ax = plt.subplots(figsize=(8.1, 2.35))
    names=['30D managed', '30D + 180D', '30D + 180D + 365D']
    equal=[a.loc['30D_only','annualized_excess_sharpe'], b.loc['equal_weights','annualized_excess_sharpe'], a.loc['equal_weights','annualized_excess_sharpe']]
    rolling=[np.nan, b.loc['rolling24M_quarterly','annualized_excess_sharpe'], a.loc['rolling24M_quarterly','annualized_excess_sharpe']]
    x=np.arange(3)
    ax.bar(x-.16, equal, width=.3, color=BLUE, label='Single strategy / equal weights')
    ax.bar(x+.16, rolling, width=.3, color=ORANGE, label='Prior 24M allocation, quarterly')
    for positions, vals in [(x-.16,equal), (x+.16,rolling)]:
        for i,v in zip(positions,vals):
            if np.isfinite(v): ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=8)
    ax.set_xticks(x,names,fontsize=9); ax.set_ylim(0,1.3)
    pub.style(ax,'Annualized excess Sharpe'); ax.legend(frameon=False,fontsize=8,loc='lower left')
    fig.tight_layout()
    save('portfolio_results', fig,
         '2018–August 2025 including equal-weight warmup. Longer sleeves are the managed 180D and independently forecast 365D put strategies. Prior-performance allocation fails to beat equal weights. These are combinations of saved sleeve returns, not a new shared-inventory portfolio execution backtest.', [path,twopath], pd.DataFrame({'portfolio':names,'equal_or_single_sharpe':equal,'prior_data_sharpe':rolling}))

    # Export every numeric series used by the existing charts as editable CSVs.
    record('feature_ablations',pub.frame('reports/p25_feature_search/comparison.csv'))
    record('annual_results',pub.frame('reports/p25_capital_sensitivity/10m_learned_annual.csv'))
    record('matched_sizing',pub.frame('reports/p25_capital_sensitivity/comparison.csv').query("case in ['10m_fixed50','10m_fixed100','10m_learned']"))
    record('train_test',pub.frame('slides/chart_data/sources/initial_train_walk_forward.csv'))
    record('cash_relative_wealth',pub.frame('slides/chart_data/cumulative_performance_10m.csv'))
    pair_path='reports/p25_variance_years/annual_pairs.csv'
    pairs=pub.frame(pair_path)
    record('pair_contributions',pairs[pairs.model.eq('variance_premium')].groupby('label',as_index=False).calendar_net_before_shared_financing.sum())


def manuscript(cap, fit, annual):
    main_rows=[]
    for key,label in [('10m_fixed50','Fixed 50%'),('10m_fixed100','Fixed 100%'),('10m_learned','Learned 50/75/100%')]:
        r=cap.loc[key]
        main_rows.append([label,f'{r.annualized_excess_sharpe:.3f}',str(int(r.entries)),f'${r.net_excess_pnl:,.0f}'])
    native=pub.frame('reports/native_year_entry/comparison.csv').set_index('case')
    nrows=[[label,f'{native.loc[key,"annualized_excess_sharpe"]:.3f}',str(int(native.loc[key,'entries']))]
           for key,label in [('forecast_variance_fixed50','Forecast, fixed 50%'),
                             ('forecast_fixed50_same_sized_skips','Same learned skips, fixed 50%'),
                             ('forecast_rl_sized','Forecast, learned sizing')]]
    return [
      {'title':TITLE,'subtitle':'Signal discovery, adaptive exposure and the limits of mean-reversion learning','blocks':[
        p('Adam Kreitzman and Soo Xin Yang | MFE230Z | Research report, '+DATE),
        h('Abstract'),
        p('We investigate whether relative option pricing across related equities supports profitable delta-hedged trades, and whether learned allocation improves their risk-adjusted performance. Research begins with ATM implied-volatility proxies, proceeds to actual option contracts, and explores smile locations, maturities, market states and alternative data. The retained configuration uses approximately 30-day, 25-delta puts: historical spread ranks and completed profitability propose trades, while a contextual policy chooses 50%, 75% or 100% of a vega budget. In a $10m funded walk-forward replay from 2018 through August 2025, learned sizing achieves annualized cash-excess Sharpe 1.003 versus 0.769 for fixed full exposure. Longer-history, feature, management and portfolio experiments reveal that learning helps selectively. A paper-inspired Q-learning adaptation weakens the retained strategy. The findings support adaptive allocation within this researched options strategy; repeated selection and concentrated profits limit claims about future performance.'),
        chart('research_map',105),
        h('1. Research question and motivating paper'),
        p('Ning and Lee [1] organize equity statistical arbitrage into identification, spread formation and trading. Their empirical mean reversion time (EMRT) measures the time for a spread to return from historical extremes; formation coefficients are chosen to shorten this time. The trading learner then uses recent spread movements to select actions. The useful starting point for our extension is this separation: finding relative value and choosing exposure are distinct problems.'),
        p('Options make the distinction consequential. An unusually wide IV spread need not converge profitably through the contracts we hold. Delta-hedged returns also depend on realized movements, strike, time decay, costs and financing. We therefore ask three empirical questions: where is the signal, what improves exposure allocation, and which additional decisions should be learned? The report follows those questions, with unsuccessful experiments informing the retained design.') ]},
      {'title':'2. Original evidence and critical assessment','blocks':[
        p('The motivating paper combines EMRT-based formation with tabular Q-learning. Its empirical application forms ten equity spreads in 2022 and trades them in 2023. The published tables report higher daily Sharpe for the combined method than for Distance and Ornstein–Uhlenbeck (OU) alternatives. Figure 2 summarizes the published cross-pair averages using their original daily metric; these values are not directly comparable with our annualized funded-options Sharpe.'),
        chart('original_paper',145),
        h('What the comparison identifies'),
        p('Formation and trading change together. EMRT changes the spread coefficients, while reinforcement learning changes the trading rule; the resulting improvement does not isolate the learner without a fixed-spread control. One trading year and ten pairs also leave substantial regime uncertainty. Pairs exposed to common market factors do not provide ten independent replications. Broader histories, alternative formation dates and matched costs would strengthen the evidence.'),
        p('Although the formation method avoids fitting an OU model, the learner trains on synthetic mean-reverting paths with a reward involving a known global mean. That creates a transfer problem when real equilibria change or jumps interrupt convergence. Historical reversion speed describes the observed sample; selecting coefficients to minimize it can favor past behavior without establishing future profitability. We retain the formation/trading distinction while requiring actual contract and hedge P&L for our own evaluation.'),
        h('Replication details and our translation'),
        p('The paper describes four move categories and a four-observation state history, but later reports 16 states. Without another compression, these ingredients imply 256 combinations. Its formation discussion also describes a positive coefficient while the table includes a negative CVS/JNJ coefficient. These are details to resolve in a faithful replication. Our successful options checkpoint does not replicate EMRT formation or the paper’s Q-table: it uses return-based pair screens and contextual size allocation. Section 9 separately tests a paper-inspired sequential adaptation.'),
        p('The distinction is central to interpretation. Equity spread convergence and delta-hedged option profits are different outcomes. Equal initial vega does not equalize gamma, market-volatility beta or jump exposure. Our extension tests whether relative option pricing predicts favorable risk-taking, then whether learning improves its allocation. We do not presume that faster IV-spread reversion must imply a more profitable option pair.') ]},
      {'title':'3. Actual contracts, observable data and execution','blocks':[
        chart('workflow',110),
        p('Research began with AMD/NVDA ATM IV and expanded to stock pairs and constituent-versus-ETF ideas. The retained candidate roster contains 345 pair sleeves. The prepared P25 input panel has 36,327 candidate rows across 125 cached pairs and 161 assets; only 608 proposals qualify during the retained evaluation, and 42 pairs trade in the $10m learned replay. The roster, cached panel and executed portfolio are different universes. Coverage varies by date, name, smile location and expiry.'),
        h('Contract identity and trade proposal'),
        p('At the signal close, select puts near absolute delta 0.25 and approximately 30 calendar days to expiry. P25 denotes downside OTM puts at selection. Each proposed trade buys one name’s option and sells the other’s; backward return beta must be 0.85–1.15 and return correlation at least 0.5. The IV-spread rank over the preceding 252 sessions must be at the lower or upper quintile. Completed directional profitability memory over 91 and 365 days must be positive, with observation thresholds of 21 and 63. The inherited formation calendar remains retrospectively conditioned.'),
        p('Orders retain signal-date contract IDs and execute next session subject to quote and risk checks. The full-size target is $450 of vega per volatility point on each leg. Integer counts use execution Greeks, allowing at most 10% initial mismatch. The policy chooses a fraction of this budget. Both legs are puts, but expiries can differ; vega neutrality is established at entry and is not continuously rebalanced. The checkpoint holds each leg to expiry except audited corporate exits.'),
        h('Accounting and limits'),
        p('Daily stock hedges offset option delta, with stock exposures netted across shared names. Option and stock cash flows, dividends, cash interest and borrowing reconcile funded NAV. Evaluation charges $0.03 per option contract, 0.10 volatility point per leg at entry, and 1 bp on stock hedge turnover. Training labels retain $0.65 commissions. Midpoint execution with a penalty is an assumed fill model, not verified institutional execution. Quote/Greek fallbacks and corporate exits are audited.'),
        p('Entry admission uses inherited 5% stress, 80% reserve and 2× gross-hedge limits, alongside pair limits. These research proxies are not broker margin requirements. Held positions can breach limits after entry; early assignment, depth-dependent impact and forced liquidation are absent. Changing account equity can therefore change admissions and Sharpe, not just rescale percentage returns. Numerical results in the main comparison share the $10m account and the same constraints.') ]},
      {'title':'4. Experiment: locate a usable volatility signal','blocks':[
        p('The initial ATM proxy appeared promising, but actual contracts weakened the result. A proxy based on future realized volatility minus entry IV cannot reproduce strike-specific gamma, discrete hedges, mark dynamics or execution. Subsequent actual-ATM experiments reached a sentiment-enhanced 0.368 Sharpe versus a 0.318 baseline [3]; historical review vintages remain unverified. We then examined downside puts, the opposite wing, shorter maturities, and broad action menus. The retained P25 configuration emerged from this exploration; it was not selected before observing these results.'),
        chart('smile_transfer',160),
        p('A coverage-matched 30D/90D transfer supplies a useful comparison: 30D P25 positive-floor sizing scores 0.285, compared with −0.033 at ATM and −0.048 for P75. At 90D, all three weaken. The restricted P25 sample has 213 proposals, not the checkpoint’s 608; its lower Sharpe is not evidence that the saved checkpoint was lost. Quote availability, formation dates and proposal qualification must be recorded before interpreting performance differences.'),
        chart('signal_example',140),
        p('The retained direction signal buys relatively cheap and sells relatively expensive selected-bucket IV at historical rank extremes. Other screens can veto an extreme; the learner sizes qualifying proposals. The AMD/NVDA example shows the observable signal and subsequent entry dates without replacing unavailable quotes. Its rolling selected contracts are signal inputs; the actual held contract keeps its own strike and expiry. The economic hypothesis is relative richness, not a guarantee that spread convergence is the source of every profit.') ]},
      {'title':'5. Experiments: features and adaptation speed','blocks':[
        p('With the P25 trade proposals frozen, we compared feature blocks individually. The 27-input baseline covers oriented IV differences, ranks, realized/implied volatility, beta, correlation, DTE, VIX, costs and completed profitability. The retained addition supplies EWMA variance premium, its historical z-score and an expected-realized/implied variance ratio, each with an availability field. Missing observations are explicit rather than grounds for dropping a proposal.'),
        eq('Estimated variance = 252 × EWMA₂₀(r²);   VRP = IV² − estimated variance'),
        chart('feature_ablations',180),
        p('The EWMA uses a 20-session half-life and at least 63 observations; the historical z-score uses a shifted 252-session window. These backward measures are forecasting proxies, not demonstrated superiority over a dedicated RV model. Variance-premium inputs improve the research reference from 0.962 to 1.006. Term structure, smile, option activity and employee sentiment fail to improve that baseline; transcripts fall below the best fixed exposure. Sparse IV-factor coverage reproduces the baseline. The feature ranking is a development result, not independent validation.'),
        chart('recency_search',115),
        p('The training-window experiment retains monthly refits and positive sizing floors. Eighteen months improves the five-year reference from 0.826 to 0.962, but still shorter windows are worse. These results suggest a useful adaptation/stability balance for this representation; they do not establish a universal optimal window. Both figures use the original $345m research funding reference. Their progression must be separated from the controlled $10m sizing comparison.') ]},
      {'title':'6. Learning: allocate exposure to frozen proposals','blocks':[
        chart('allocation_model',100),
        p('A regression tree with at most eight leaves partitions states using completed outcomes. Cross-entropy method (CEM) policy search assigns 50%, 75% or 100% exposure to each leaf. Candidate assignments are evaluated jointly on aggregated daily historical P&L paths, allowing covariance across allocations to enter the objective. The retained search uses 50 iterations and 192 candidates per iteration. There is no learned skipping, direction reversal or within-trade management in this checkpoint.'),
        eq('Annualized cash-excess Sharpe = √252 × mean(daily excess return) / sample SD'),
        p('This is contextual allocation with direct policy search, not Q-learning: no Bellman update or discounted action continuation is estimated. It is the successful learned-allocation component of our research, but its result does not by itself establish sequential RL superiority. A profit-only allocation objective reaches 0.979 against 1.006 for direct Sharpe; a diagonal-covariance variant reaches an effectively tied 1.007. The common benefit is state-dependent exposure relative to constant size.'),
        h('Chronological training and initial-fit evidence'),
        p('Every month uses the preceding 18 months of proposals whose outcomes completed before the cutoff. The January 2018 fit has 74 completed examples from July 2016 onward. Later refits may use earlier evaluation outcomes once completed. Five CEM seeds reproduce the saved NAV with tree partitions fixed, checking search stability for that representation. They do not test alternative tree partitions or research selection.'),
        chart('train_test',135),
        p('Initial training improves from 1.573 to 2.015; subsequent $10m walk-forward performance improves from 0.769 to 1.003. Compare fixed versus learned within each panel. Training scores are optimized in sample on aggregated paths with $0.65 labels; evaluation uses the funded ledger and $0.03 commissions. The training panel is neither all pre-2018 history nor an executable pre-2018 account replay. Walk-forward chronology prevents direct outcome leakage, but the repeatedly researched evaluation history is not an untouched holdout.') ]},
      {'title':'7. Main result: learned size improves the portfolio','blocks':[
        table(['Policy','Excess Sharpe','Entries','P&L above cash'],main_rows,[180,92,65,175]),
        chart('cash_relative_wealth',175),
        p('The $10m learned replay improves full-size Sharpe by 0.234 and earns approximately $323,000 above cash, versus $272,000 at fixed 100% and $248,000 at fixed 50%. The strategies share all 608 proposals, directions, contract identities, costs and risk limits. They do not execute identical portfolios: size affects integer feasibility, overlapping inventory and risk admission. Fixed half size admits 263 entries, fixed full size 133 and learned size 212. The finding is an account-level allocation improvement under common rules.'),
        h('Return, capital and exposure'),
        p('Total funded CAGR is 2.963%, while wealth relative to the cash benchmark compounds at 0.342% annually. Maximum cash-relative drawdown is 0.491%. Cash interest accounts for most total account growth; the plotted excess performance isolates the trading contribution. Peak reserve usage is about 18% and peak gross stock hedge about 0.29× NAV. Reserve is only a research proxy for capital deployed, not actual margin.'),
        p('Lower equity raises percentage exposure but can alter constraints. At $3m, learned cash-relative CAGR is 0.991% and Sharpe 0.893; at $1m they are 1.135% and 0.552. A smaller denominator alone does not manufacture a higher excess Sharpe, and tighter account resources can reject profitable entries. Scaling studies are therefore separate replays rather than arithmetic changes to the headline return.'),
        h('Uncertainty in the gain'),
        p('A $10m paired bootstrap (126-session circular blocks, 1,000 draws) gives an unadjusted 95% interval of −0.031 to +0.416 for the 0.234 Sharpe gain over fixed full size [6]. The interval and repeated strategy selection prevent a strong significance claim. The historical point estimate nevertheless supplies a reproducible comparison of adaptive and constant allocation. The stronger evidence is in walk-forward portfolio performance, not the expected improvement in an optimized training objective.') ]},
      {'title':'8. Diagnostics: regimes, concentration and parity','blocks':[
        chart('annual_results',160),
        p('Performance is uneven. The strongest contributions occur in 2020, 2022 and 2023, while 2019, 2021 and 2024 are negative. Partial 2025 improves, but does not convert the preceding weak period into independent validation. Calendar returns above cash and annualized Sharpe answer different questions; neither is a total funded return. The 2025 observation ends 29 August after final June signals and contract runoff.'),
        chart('pair_contributions',150),
        p('In the original funding reference, 2022 contributes about 55% of cumulative excess dollars. Pair attribution also shows concentrated winners and losses. These figures motivate risk-context and completed-profitability states, but a retrospective list of winning names cannot become an entry rule without a chronological test. The pair chart excludes shared financing and uses the original research account; it is attribution, not the $10m account’s pair-level Sharpe.'),
        h('Why puts and calls need careful matching'),
        p('P25 puts and C25 calls occupy opposite wings; the same-strike call for P25 is approximately C75. On 64 identical 30D cohorts, the parity-Greek control finds similar pre-execution/borrow P&L ($64.6k puts, $65.3k calls), but net values differ ($53.1k, $46.5k). Extra hedge/borrow costs and mark/carry variation matter; return correlation is 0.972 monthly versus 0.470 daily [7].'),
        p('This subset is not the full checkpoint. Longer-expiry differences remain sensitive to European fits, declared dividends, American exercise and borrowing. The audit motivates implementation checks before attributing put/call differences to alpha; it does not establish arbitrage or independently identify stale quotes.') ]},
      {'title':'9. Extensions: management, maturity and Q-learning','blocks':[
        chart('management_results',110),
        p('At 30D, 36-month direct policy search raises Sharpe from 1.003 to 1.029, while 18-month management scores 0.963. The better controller trims five cohorts and closes four. Conservative fitted-Q management at 180D improves fixed-half-size Sharpe from 0.592 to 0.610. These modest selected gains involve different algorithms and strategies; compare within each panel.'),
        h('A separate one-year entry signal'),
        table(['365D forecast policy','Excess Sharpe','Entries'],nrows,[330,105,77]),
        p('A horizon-specific variance forecast fixes one-year direction before learning participation and size. Sizing scores 0.313 versus 0.254 at fixed 50%; fixing size with the same learned skips gives 0.273. Learning direction and size jointly is negative. The gain is statistically uncertain. This signal replaces the older one-year spread strategy.'),
        chart('reversion_q',120),
        p('Paper-inspired fitted Q uses recent IV moves, causal reversion history, valuation and inventory, monthly 18-month fits, and actual P&L rewards. It learns direction, flat/size, and hold/close. Broad runs retain 17,967 candidates without old rank/profit vetoes but lose money. On the original 212 entries/counts, Q exits lower Sharpe to 0.598 and excess P&L from $323k to $145k. Risk-penalized variants also fail. This is neither an exact paper replication nor a pure Q-versus-CEM comparison; this state/reward design does not improve the strategy.') ]},
      {'title':'10. Portfolio construction and conclusions','blocks':[
        chart('portfolio_results',130),
        p('Combining saved managed 30D and 180D return streams yields equal-weight Sharpe 1.029; adding the new 365D signal gives 1.006. Allocations estimated from the preceding 24 months and updated quarterly score 0.990 and 0.865, respectively, including warmup. A full-history optimized three-sleeve blend reaches 1.107, an exploratory hindsight reference. These combinations are return-stream experiments; shared capital, hedge netting and inventory require an integrated account backtest before claiming executable portfolio improvement.'),
        h('What the experiments establish'),
        p('Our strongest finding is a useful retained downside-put signal whose historical risk-adjusted performance improves under contextual sizing. Feature and training-window experiments explain how we arrived there. Actual-contract testing, weak smile/maturity transfers, unsuccessful broader policies and the Q-learning result define its limits. More information and more actions do not automatically create a better learner; returns depend on the economic signal, available data and allocation formulation.'),
        p('The next experiment should freeze independently validated signal proposals and compare equal allocation, risk-based allocation and learned portfolio allocation in one shared account. States should include signal strength, completed profitability, correlations, shared names, inventory and capital usage. Subsequent unseen dates and prospective pair formation are needed to distinguish adaptation from accumulated research selection. Better remaining-variance forecasts are also a natural input for position management.'),
        h('References and reproducibility'),
        p('[1] Ning, B. and Lee, K. (2024), Advanced Statistical Arbitrage with Reinforcement Learning. arXiv:2403.12180. https://arxiv.org/abs/2403.12180'),
        p('[2] Strategy/data: config/p25_variance_premium_candidate.json; data/p25_variance_premium_inputs.csv. [3] ATM research: reports/atm65_employee_sentiment.md. [4] Discovery: reports/p25_maturity_matched/; reports/p25_feature_search/; reports/p25_recency/.'),
        p('[5] Main results: reports/p25_capital_sensitivity/; reports/p25_variance_validation/. [6] Uncertainty: reports/long_tenor_search_summary/core_comparison.csv. [7] Parity: reports/long_expiry_parity/. [8] Extensions: reports/p25_short_management/; reports/long_tenor_fitted_q/; reports/native_year_entry/; reports/p25_reversion_q/.'),
        p('Every figure records source-file hashes in paper/source_manifest.json. Editable figure data and complete experiment tables accompany the report. No model or backtest was rerun for publication. Academic research use only; market and alternative inputs remain subject to their institutional/vendor licenses.') ]}
    ]


class ReportPage(pub.PaperPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for t in self.fig.texts:
            if 'RESEARCH DRAFT' in t.get_text():
                t.set_text('RESEARCH REPORT · '+DATE.upper())
    def paragraph(self, text, size=10.35, color='#202832', leading=13.25):
        # A compact readable report body; captions retain the original small type.
        if size == 10.35:
            size, leading = 10, 12.5
            if re.match(r'^\[\d\]',text):
                size, leading = 8.8, 11
        super().paragraph(text, size, color, leading)


def editable_text(pages):
    # Plain text is intentionally included for editing in Word/Docs without losing prose.
    lines=[TITLE, 'Adam Kreitzman and Soo Xin Yang | '+DATE, '']
    for i,page in enumerate(pages,1):
        lines += [f'PAGE {i}: '+page['title'], '']
        for b in page['blocks']:
            if b['type'] in ['paragraph','heading','equation']:
                lines += [b['text'], '']
            elif b['type']=='chart':
                lines += [f'Figure {pub.PAPER_FIGURES[b["name"]]}. '+pub.CHARTS[b['name']]['caption'],
                          f'Figure file: figures/{b["name"]}.png','']
            elif b['type']=='table':
                lines += [' | '.join(b['headers'])]+[' | '.join(r) for r in b['rows']]+['']
    (PAPER/'report_text.txt').write_text('\n'.join(lines))


def main():
    plt.rcParams['text.parse_math'] = False  # Currency pairs must never become math delimiters.
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'slides').glob('*.pptx')}
    protected=['reports/p25_capital_sensitivity/comparison.csv',
               'reports/p25_reversion_q/comparison.csv']
    before.update({k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in protected})
    archive=PAPER/'archive'/'report_before_2026-10-06_rewrite.zip'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in PAPER.rglob('*'):
                if f.is_file() and 'archive' not in f.parts:
                    z.write(f,f.relative_to(PAPER))
    FIG.mkdir(exist_ok=True)
    cap,fit,annual,features,overlay=pub.charts()
    extra_charts()
    for path in ['reports/atm65_employee_sentiment.md','reports/p25_variance_validation/research_summary.txt',
                 'reports/long_tenor_search_summary/core_comparison.csv',
                 'reports/long_expiry_parity/findings.txt','reports/p25_reversion_q/findings.txt',
                 'reports/p25_reversion_q/audit.json', 'data/p25_variance_premium_inputs.csv']:
        pub.source(path)
    for path in ['scripts/rewrite_research_report.py','scripts/build_research_publication.py',
                 'scripts/report_long_tenor_search.py']:
        pub.source(path)
    pages=manuscript(cap,fit,annual)
    figure_order=[b['name'] for page in pages for b in page['blocks'] if b['type']=='chart']
    assert len(figure_order)==len(set(figure_order))
    pub.PAPER_FIGURES={k:i+1 for i,k in enumerate(figure_order)}
    pub.PaperPage=ReportPage
    (PAPER/'manuscript.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
    checks=pub.render_paper(pages)
    file=PAPER/'options_relative_value.html'
    content=file.read_text().replace('Research draft · 4 October 2026','Research report · '+DATE)
    content=content.replace('https://arxiv.org/abs/2403.12180',
           '<a href="https://arxiv.org/abs/2403.12180">arXiv:2403.12180</a>')
    content=content.replace('paper/source_manifest.json','<a href="source_manifest.json">paper/source_manifest.json</a>')
    file.write_text(content)
    editable_text(pages)
    previews=[]
    for i in range(1,11):
        im=Image.open(PAPER/f'page_{i:02d}.png').convert('RGB'); im.thumbnail((306,396)); previews.append(im)
    sheet=Image.new('RGB',(5*306,2*396),'#d8dfe7')
    for i,im in enumerate(previews):sheet.paste(im,((i%5)*306,(i//5)*396))
    sheet.save(PAPER/'contact_sheet.png')
    reader=PdfReader(PAPER/'options_relative_value.pdf')
    extracted='\n'.join(p.extract_text() for p in reader.pages)
    for expected in ['1.003','0.769','0.598','0.313','0.610','1.029','0.865']:
        assert expected in extracted,expected
    assert len(reader.pages)==10
    for path,digest in before.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for path,digest in pub.SOURCES.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    manifest=dict(title=TITLE,draft_date='2026-10-06',data_end='2025-08-29',paper_pages=10,
                  sources=pub.SOURCES,charts={k:pub.CHARTS[k] for k in figure_order},
                  layout_checks=checks,models_or_backtests_rerun=False,slides_modified=False,
                  figure_numbers=pub.PAPER_FIGURES,protected_files=before)
    (PAPER/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    # Full experiment results accompany the concise main report without enlarging its ten pages.
    supplemental=['reports/p25_feature_search/comparison.csv','reports/p25_recency/comparison.csv',
                  'reports/p25_capital_sensitivity/comparison.csv','reports/p25_maturity_matched/comparison.csv',
                  'reports/p25_short_management/comparison.csv','reports/long_tenor_fitted_q/comparison.csv',
                  'reports/native_year_entry/comparison.csv','reports/p25_reversion_q/comparison.csv',
                  'reports/p25_reversion_q/annual.csv','reports/p25_reversion_q/findings.txt',
                  'reports/native_year_entry/replacement_blend/walk_forward_24M/comparison.csv',
                  'reports/native_year_entry/replacement_blend/walk_forward_24M_30D_180D/comparison.csv',
                  'reports/long_tenor_search_summary/core_comparison.csv']
    for path in supplemental:pub.source(path)
    manifest['sources']=pub.SOURCES
    manifest['supplemental_results']=supplemental
    (PAPER/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    words=sum(len(b.get('text','').split()) for page in pages for b in page['blocks'])
    audit=dict(status='complete',paper_pages=10,paper_word_count=words,figures=len(figure_order),
               source_hashes_verified=True,all_layout_checks_pass=True,
               prior_report_archived=True,presentation_unchanged=True,checkpoints_unchanged=True,
               main_values_verified=True,models_or_backtests_rerun=False)
    (PAPER/'publication_audit.json').write_text(json.dumps(audit,indent=2))
    (PAPER/'reproduction.txt').write_text('Rebuild only the report from saved results:\n'
       '.venv/bin/python -m scripts.run_signal_discovery_guarded --module scripts.rewrite_research_report --timeout 600 --duty-cycle 0.2 --rss-limit-mib 384 --min-free-percent 35 --warning-rss-limit-mib 384 --max-swap-growth-mib 64 --pause-low-headroom-seconds 60\n'
       'Install requirements-report.txt; restore submission caches first (see SUBMISSION.md).\n'
       'One worker;20% of one core; BLAS/Arrow threads1;384 MiB RSS stop. No fitting, backtest or deck editing.\n'
       'Main PDF: options_relative_value.pdf; editable HTML: options_relative_value.html; prose: report_text.txt.\n'
       'Every chart: figures/*.svg and *.png; numeric figure tables: figure_data/*.csv.\n')
    with zipfile.ZipFile(PAPER/'report_and_visuals.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['options_relative_value.pdf','options_relative_value.html','report_text.txt','manuscript.json',
                     'source_manifest.json','publication_audit.json','reproduction.txt']:
            z.write(PAPER/name,name)
        for k in figure_order:
            for ext in ['png','svg']:z.write(FIG/f'{k}.{ext}',f'figures/{k}.{ext}')
        for f in DATA.glob('*.csv'):z.write(f,f'figure_data/{f.name}')
        for path in supplemental:z.write(ROOT/path,'supplemental/'+path)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
