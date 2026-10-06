"""Render the ten-page research paper and updated deck from retained results.

No fitting or backtest execution. Run through run_signal_discovery_guarded.
"""
from pathlib import Path
import csv
import gc
import hashlib
import html
import json
import re
import shutil
import xml.etree.ElementTree as ET
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
FIG = PAPER / 'figures'
BLUE, ORANGE, GREEN, RED, GRAY = '#2463a6', '#d48236', '#337e65', '#b54545', '#66717e'
SOURCES = {}
CHARTS = {}
PAPER_FIGURES = {'original_paper':1, 'workflow':2, 'feature_ablations':3,
                 'train_test':4, 'cash_relative_wealth':5, 'annual_results':6,
                 'capital_scaling':7}


def source(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def frame(path):
    return pd.read_csv(source(path))


def savechart(name, fig, caption, paths):
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close(fig)
    CHARTS[name] = dict(caption=caption, sources=paths)


def style(ax, ylabel=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#d0d6dc')
    ax.grid(axis='y', color='#e8ecf0', linewidth=.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def charts():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'svg.fonttype': 'none'})
    cap_path = 'reports/p25_capital_sensitivity/comparison.csv'
    cap = frame(cap_path).set_index('case')
    fit_path = 'artifacts/contracts/p25_feature_search/models/variance_premium/2018-01-01.json'
    fit = json.loads(source(fit_path).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.65))
    values = [[fit['baseline_shadow_daily_sharpe'], fit['training_shadow_daily_sharpe']],
              [cap.loc['10m_fixed100', 'annualized_excess_sharpe'], cap.loc['10m_learned', 'annualized_excess_sharpe']]]
    for ax, vals, title in zip(axes, values, ['Initial training: Jul 2016–Dec 2017', 'Walk-forward: 2018–Aug 2025 ($10m)']):
        ax.bar(['Fixed full size', 'Learned sizing'], vals, color=[GRAY, BLUE], width=.55)
        ax.set_ylim(0, 2.45)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals): ax.text(i, v+.05, f'{v:.3f}', ha='center', weight='bold')
        style(ax, 'Annualized Sharpe')
    fig.tight_layout()
    savechart('train_test', fig, 'Learned sizing improves each within-period benchmark. Training uses aggregated P&L paths and $0.65 commissions; walk-forward uses the funded ledger and $0.03 commissions. The levels are not directly comparable.', [fit_path, cap_path])

    fig, ax = plt.subplots(figsize=(8.1, 2.7))
    for policy, label, color in [('fixed50', 'Fixed 50%', ORANGE), ('fixed100', 'Fixed 100%', GRAY), ('learned', 'Learned 50/75/100%', BLUE)]:
        path = f'artifacts/contracts/p25_capital_sensitivity/replays/10m_{policy}/daily_nav.parquet'
        nav = pd.read_parquet(source(path), columns=['nav', 'cash_benchmark'])
        ax.plot(nav.index, 100*(nav.nav/nav.cash_benchmark-1), label=label, color=color, linewidth=1.35)
    style(ax, 'Wealth relative to cash (%)')
    ax.set_xlim(nav.index[0], nav.index[-1])
    ax.axhline(0, color=GRAY, linewidth=.7)
    ax.legend(loc='upper left', frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    savechart('cash_relative_wealth', fig, 'Actual $10m funded portfolios, 2018–29 August 2025. Curves show 100 × (NAV / same-equity cash benchmark − 1), not total NAV returns. All controls share proposals and risk limits; sizing changes admission and inventory.', [f'artifacts/contracts/p25_capital_sensitivity/replays/10m_{p}/daily_nav.parquet' for p in ['fixed50', 'fixed100', 'learned']])

    annual_path = 'reports/p25_capital_sensitivity/10m_learned_annual.csv'
    annual = frame(annual_path)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.7))
    x = np.arange(len(annual))
    for ax, field, title, label in [(axes[0], 'annualized_excess_sharpe', 'Annualized excess Sharpe', 'Sharpe'), (axes[1], 'return_above_cash_pp', 'Return above cash', 'Percentage points')]:
        vals = annual[field].to_numpy()
        ax.bar(x, vals, color=[BLUE if v>=0 else RED for v in vals])
        ax.set_xticks(x, [str(y)+('*' if y==2025 else '') for y in annual.year], rotation=45)
        ax.axhline(0, color=GRAY, linewidth=.7)
        ax.set_title(title, fontsize=11)
        style(ax, label)
    fig.tight_layout()
    savechart('annual_results', fig, '$10m learned strategy. Sharpe is annualized; calendar-period return above cash is not. The 2025 observation is partial through 29 August. Positive total funded returns can coexist with negative trading excess returns.', [annual_path])

    feature_path = 'reports/p25_feature_search/comparison.csv'
    features = frame(feature_path)
    learned = features[features.kind.eq('learned')].sort_values('sharpe')
    names = {'variance_premium': 'EWMA variance premium', 'control': 'Prior learned baseline', 'iv_factor': 'IV factor', 'term_structure': 'Term structure', 'volume_oi': 'Volume / open interest', 'smile_shape': 'Smile shape', 'smile_price': 'Smile price', 'employee': 'Employee leadership', 'rv_risk': 'RV / IV risk', 'physical_tail': 'Realized tail moments', 'transcripts': 'Earnings transcripts'}
    fig, ax = plt.subplots(figsize=(8.1, 3.45))
    ax.barh([names[n] for n in learned.model], learned.sharpe, color=[BLUE if n=='variance_premium' else GRAY for n in learned.model])
    ax.axvline(float(features.loc[features.model.eq('fixed2'), 'sharpe'].iloc[0]), color=ORANGE, linestyle='--', label='Best fixed size: 0.770')
    ax.set_xlim(0, 1.13)
    for i, v in enumerate(learned.sharpe): ax.text(v+.012, i, f'{v:.3f}', va='center', fontsize=8)
    style(ax, 'Feature block')
    ax.set_xlabel('Annualized cash-excess Sharpe')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    fig.tight_layout()
    savechart('feature_ablations', fig, 'Original $345m research reference, identical 608 qualified proposals. Each block is added separately to the prior 27-input state. The retained variance-premium block reaches 1.006; most additional signals fail to improve the 0.962 learned baseline.', [feature_path])

    pair_path = 'reports/p25_variance_years/annual_pairs.csv'
    pairs = frame(pair_path)
    total = pairs[pairs.model.eq('variance_premium')].groupby('label').calendar_net_before_shared_financing.sum().sort_values()
    chosen = pd.concat([total.head(5), total.tail(5)])
    fig, ax = plt.subplots(figsize=(8.1, 3.05))
    ax.barh(chosen.index, chosen/1000, color=[RED if v<0 else BLUE for v in chosen])
    ax.axvline(0, color=GRAY, linewidth=.7)
    style(ax, 'Pair')
    ax.set_xlabel('Net contribution before shared financing ($ thousands)')
    fig.tight_layout()
    savechart('pair_contributions', fig, 'Five largest losing and winning pairs in the original $345m reference. Includes costs, hedges and dividends; excludes shared financing. This is retrospective attribution, not a rule for selecting or banning names.', [pair_path])

    overlay_path = 'reports/p25_capital_policy_matched/comparison.csv'
    overlay = frame(overlay_path).set_index('case')
    order = ['constant1', 'constant2', 'constant4', 'capital_policy']
    labels = ['1× expert', '2× expert', '4× expert', 'Capital-aware pilot']
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.75))
    for ax, field, title, suffix in [(axes[0], 'annualized_excess_sharpe', 'Sharpe', ''), (axes[1], 'relative_to_cash_cagr_pct', 'Cash-relative wealth CAGR', '%')]:
        vals = overlay.loc[order, field].to_numpy()
        ax.bar(np.arange(4), vals, color=[BLUE, GREEN, ORANGE, GRAY])
        ax.set_xticks(np.arange(4), labels, rotation=15, ha='right', fontsize=8)
        for i, v in enumerate(vals): ax.text(i, v+.025, f'{v:.3f}{suffix}', ha='center', fontsize=8)
        ax.set_ylim(0, max(vals)*1.2)
        ax.set_title(title, fontsize=11)
        style(ax)
    fig.tight_layout()
    savechart('capital_scaling', fig, '$10m equity with per-pair risk allowances scaled alongside requested trade size. All expert multipliers retain the underlying learned 50/75/100% sizing. Global limits remain fixed. The limited capital-aware pilot underperforms the 1× and 2× expert on Sharpe.', [overlay_path])

    fig, ax = plt.subplots(figsize=(8.1, 2.3))
    stages = [('Eligible pair', 'beta 0.85–1.15\ncorr ≥ 0.5; past profit'), ('Relative value', '252-session IV rank\n20 / 80 boundaries'), ('Learned size', '33 inputs → ≤8 leaves\n50 / 75 / 100%'), ('Actual positions', 'integer vega matching\ndaily delta hedge')]
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis('off')
    for i, (title, text) in enumerate(stages):
        x = .1 + 2.5*i
        ax.add_patch(FancyBboxPatch((x,.45),2.15,1.15,boxstyle='round,pad=.08',facecolor='#eef3f8',edgecolor='#b9cde0'))
        ax.text(x+1.075,1.31,title,ha='center',weight='bold',fontsize=10,color=BLUE)
        ax.text(x+1.075,.91,text,ha='center',va='center',fontsize=8)
        if i<3: ax.annotate('',xy=(x+2.4,1.02),xytext=(x+2.2,1.02),arrowprops={'arrowstyle':'->','color':GRAY})
    fig.tight_layout()
    savechart('workflow', fig, 'Retained P25 strategy: rules propose the direction; the learner adjusts size. Orders use prior-close selected contracts on the next session. Positions hold to each leg’s expiry, except audited corporate exits.', ['config/p25_variance_premium_candidate.json'])

    fig, ax = plt.subplots(figsize=(8.1,2.3))
    methods = ['Distance', 'OU', 'EMRT + RL']
    vals = [np.mean([.0955,.0213,.0535,-.0207,.0704,.0178,.1058,-.0764,-.0300,0]),
            np.mean([.0764,-.0049,.0261,.0985,.1007,.0149,.0983,0,-.0093,.0693]),
            np.mean([.1250,.0780,.1377,.1049,.1312,.0861,.1097,.1061,.0513,.0698])]
    ax.bar(methods,vals,color=[GRAY,ORANGE,BLUE],width=.55)
    for i,v in enumerate(vals):ax.text(i,v+.004,f'{v:.3f}',ha='center')
    ax.set_ylim(0,.125);style(ax,'Mean daily Sharpe (not annualized)')
    fig.tight_layout()
    savechart('original_paper',fig,'Ning and Lee (2024), Tables 3–4: arithmetic mean of reported daily Sharpe across ten pairs, formed in 2022 and traded in 2023. These published equity results use a different metric and setup from our annualized options results.', ['Advanced Statistical Arbitrage with Reinforcement Learning.pdf'])
    source('Advanced Statistical Arbitrage with Reinforcement Learning.pdf')
    return cap, fit, annual, features, overlay


def p(text): return {'type': 'paragraph', 'text': text}
def h(text): return {'type': 'heading', 'text': text}
def eq(text): return {'type': 'equation', 'text': text}
def chart(name, height=180): return {'type': 'chart', 'name': name, 'height': height}
def table(headers, rows, widths=None): return {'type':'table','headers':headers,'rows':rows,'widths':widths}


def manuscript(cap, fit, annual, features, overlay):
    yearly = [[str(r.year)+('*' if r.year==2025 else ''), f'{r.funded_return_pct:.3f}%', f'{r.return_above_cash_pp:+.3f}', f'{r.annualized_excess_sharpe:.3f}'] for r in annual.itertuples()]
    pages = [
        {'title':'From Equity Pairs to Options Relative Value', 'subtitle':'Learned sizing of delta-hedged, initially vega-neutral downside puts', 'blocks':[
            p('Adam Kreitzman and Soo Xin Yang | MFE230Z project | Research draft, 4 October 2026'),
            h('Abstract'),
            p('We extend an equity statistical-arbitrage framework to cross-sectional volatility trading. Early constant-maturity ATM experiments suggested useful relative value, but actual-contract backtests showed that proxy payoffs omit material execution, strike and inventory effects. The strongest retained candidate trades approximately 30-day, 25-delta puts, matches vega at entry, delta hedges daily and holds each leg to expiry. A contextual regression tree and direct Sharpe policy search select 50/75/100% trade sizes. At $10m equity, the walk-forward research sample from 2018 through August 2025 yields annualized cash-excess Sharpe 1.003, versus 0.769 for fixed full size. Initial pre-2018 training scores also improve, from 1.573 to 2.015. These are different accounting exercises, and all evaluation history has influenced research. Results support adaptive sizing in this sample rather than a universal advantage of reinforcement learning or a validated deployable strategy.'),
            h('1. The motivating equity-pairs paper'),
            p('Ning and Lee (2024) separate statistical arbitrage into asset identification, spread formation and trading decisions [1]. Their empirical mean reversion time (EMRT) measures how long a spread takes to return from significant extremes to its historical mean. They choose portfolio coefficients to reduce that time, replacing an assumed Ornstein–Uhlenbeck fit with a sample-based formation criterion. For two assets the spread is X = S_A − B S_B. Trading then becomes a sequential decision problem rather than a fixed deviation threshold.'),
            p('Their Q-learner discretizes recent price changes and chooses whether to buy, hold or sell the spread, subject to position restrictions. It learns on simulated mean-reverting paths because the reward depends on the true spread mean. In the real-data application, 2022 is the formation year and 2023 the trading year. Tables 3–4 report higher daily Sharpe for EMRT plus RL than the Distance and OU alternatives in all ten pairs. Figure 1 summarizes those published values; it does not compare them numerically with our options Sharpe.'),
            chart('original_paper',145)]},
        {'title':'2. Criticism and the economic translation', 'blocks':[
            p('The original paper offers a useful design pattern, but its evidence is narrower than a general claim that RL solves pairs trading. The real-market evaluation uses one trading year and ten representative pairs. A common market regime can affect every pair, so ten pair outcomes are not ten independent replications. Stability across formation dates, longer test histories, trading costs and alternative pair universes remains important. Our criticism is about what the experiment identifies, not a claim that its reported results are incorrect.'),
            p('Formation and trading change together: EMRT replaces the coefficient estimate while RL replaces the timing rule. A performance difference therefore cannot isolate the contribution of RL without controls holding the spread fixed. The reported daily Sharpe is also not an annualized funded-account Sharpe. We retain its original label and avoid placing it on the same numerical scale as our cash-excess performance.'),
            p('The framework is described as model-free, yet its trading learner uses synthetic mean-reverting paths and a reward involving a true global mean [1]. That creates a transfer question: a policy trained in a stable simulated environment must operate on markets with changing equilibria, jumps and costs. Historical extrema are useful formation statistics, but they do not certify that the next observed deviation will converge. Optimizing empirical reversion speed can itself select favorable historical paths.'),
            p('There is a reproducibility detail worth clarifying: the text defines four move categories and later specifies a four-observation lookback but reports 16 states. Those ingredients imply 4^4 = 256 combinations unless an additional compression is applied. The real-data formation discussion also calls the coefficient positive while Table 2 includes a negative CVS/JNJ coefficient. These inconsistencies should be reconciled in a replication; they do not by themselves invalidate the economic idea.'),
            h('What we borrow'),
            p('We borrow the separation between relative-value formation and learned actions. We do not replicate EMRT formation or the paper’s Q-table in the retained options strategy. Our pair screens use backward return beta and correlation, and our current successful learner performs contextual sizing. The question is whether economically interpretable option states lead to profitable trades, and whether learned size allocation improves their portfolio risk-adjusted result.'),
            h('Why price convergence is insufficient for options'),
            p('An equity spread earns from relative price convergence. A delta-hedged option position earns or loses through realized movements, time decay, changing implied volatility, discrete hedging and financing. Equal initial vega does not make two legs equal in gamma, jump exposure or market-volatility beta. Our motivating variance-premium proxy is IV squared minus an estimate of future annualized variance. It is a feature, not an exact payoff formula for a strike-specific option.'),
            eq('VRP_i,t = IV_i,t² − estimated variance_i,t;  pair state = VRP_long,t − VRP_short,t'),
            p('The extension consequently needs actual contract identities, daily marks and an inventory ledger. Otherwise a positive IV-spread proxy can be mistaken for executable relative-value profits. Our subsequent research tests this translation rather than assuming that the equity paper’s mechanism transfers unchanged.') ]},
        {'title':'3. Research progression and data', 'blocks':[
            p('Research began with constant-maturity ATM IV, including AMD/NVDA, and expanded to stock pairs and constituent-versus-ETF pairs. An earlier NN benchmark compared full-size long/flat/short decisions with direct Sharpe policy search using sizing. Its combined proxy Sharpe was 1.461 versus 1.036 for the NN. However, the same-feature full-size Sharpe policy reached 1.497, and excluding silver reduced the sized result to 0.188 [2]. These findings motivated further work but did not demonstrate broad, executable options profitability.'),
            table(['Stage','Main finding','Interpretation'],[
                ['ATM IV proxy','Combined sized Sharpe 1.461','Approximate payoff; silver concentrated'],
                ['Actual ATM contracts','Sentiment checkpoint 0.368','Daily cash-excess Sharpe; separate strategy'],
                ['Smile / maturity research','P25 retained; other transfers weak','Coverage and state complexity matter'],
                ['P25 learned sizing','1.006 research; 1.003 at $10m','Current candidate, not fresh validation']],[125,170,217]),
            p('Moving to actual ATM calls substantially weakened performance. Alternative-data experiments included employee leadership ratings, earnings-call language and option volume/open interest. The retained ATM sentiment checkpoint reached 0.368 annualized excess Sharpe, versus 0.318 for its baseline [3]. Historical vendor vintages were not independently archived, and that gain did not establish that the same signal would improve a different wing strategy.'),
            p('We next examined smile location and maturities, including roughly 14-, 30-, 45- and 90-day contracts, separate downside and opposite-wing ideas, and combinations. These were not one uniformly controlled tournament. Expanding menus introduced sparse states, inconsistent feature coverage and changes in selected contracts. Coverage-matched 30/90-day transfers remained weak, with P25 floor sizing at 0.285 for 30D and −0.363 for 90D in that particular restricted experiment [4]. These are not the final candidate’s performance.'),
            h('Data used by the retained candidate'),
            p('The retained universe has 345 candidate pair sleeves, with stock/ETF returns, option quotes and Greeks, dividends, rates and auxiliary market states. Historical option downloads were extended back to 2010, but the current model’s first training window begins in July 2016 and its executable evaluation begins in January 2018. Coverage is not identical across names, wings or expiries. The inputs-only P25 CSV contains 36,327 rows and 33 feature columns plus identifiers and eligibility; it excludes targets and forward rewards [5].'),
            p('Historical pair/formation calendars are inherited and have retrospective conditioning. Only 608 rule-qualified proposals enter the retained experiment; 42 pairs actually trade under the $10m learned replay. This is not a claim that an unrestricted options universe was evaluated prospectively. Source options and reward reconstruction remain subject to institutional and vendor licenses; the shared inputs CSV alone is not sufficient to reproduce the complete contract ledger.') ]},
        {'title':'4. Executable strategy and accounting', 'blocks':[
            chart('workflow',110),
            p('The retained strategy uses two puts, each selected near absolute delta 0.25 and approximately 30 calendar days to expiry at the previous close. The contracts are selected before the next-session execution check; they are not replaced using later quotes. P25 means an out-of-the-money downside put at selection, not upside volatility. Execution-day deltas can move materially away from 0.25, and the two expiries need not be identical.'),
            h('Entry direction and size'),
            p('The original 252-session IV-spread rank proposes a long/short direction at the lower or upper quintile. Backward pair beta must lie in 0.85–1.15 and return correlation must be at least 0.5. Completed directional profitability memory over 91 and 365 days must be positive, with minimum observation counts of 21 and 63. These are inherited rule screens. The learner never reverses the proposed direction and cannot create a new trade outside this eligible set [5].'),
            p('The full-size reference is $450 of vega per volatility point per leg. The policy chooses 50%, 75% or 100%, and integer contract counts are matched using execution Greeks with at most 10% entry vega imbalance. Sizing compounds with account NAV. Entry neutrality is not continuous neutrality: deltas, vegas and gammas evolve, and each leg is held to its own expiry except audited corporate-action exits.'),
            h('Daily hedge and cash ledger'),
            p('Stock hedges offset option delta daily, with shared-name stock positions netted in the portfolio ledger. Option marks, hedge gains, dividends, execution charges, cash interest and stock borrow reconcile daily NAV. Expiry uses the contract settlement convention implemented in the saved simulator. Finite-difference or Black–Scholes-based fallback Greeks and causal held-mark fallbacks are counted rather than silently concealed. The accounting audit reconciles cash flows; it does not prove every vendor quote is executable.'),
            p('Evaluation option commissions are $0.03 per contract. Entries use midpoint plus a 0.10 volatility-point penalty on each leg, and stock hedge turnover costs 1 basis point. Training uses the stricter $0.65 commission labels. Options are held to expiry, so the entry penalty is not repeatedly charged as if the strategy crossed the option spread daily. Liquidity depth and nonlinear market impact are not modeled.'),
            p('Inherited risk admission includes a 5% independent-name stress limit, an 80% reserve cap and a 2× gross stock-hedge cap. Stress shocks each underlying by ±20% and IV by ±10 volatility points. Reserve is gross option premium plus 50% of gross netted stock-hedge notional. These are research assumptions, not broker margin rules or user-specified limits. Held positions can breach limits after entry; forced liquidation and early assignment are absent.') ]},
        {'title':'5. Observable signals and feature exploration', 'blocks':[
            p('The initial state contains 27 fields, including IV levels and oriented gaps, spread-rank strength, trailing realized volatility versus implied volatility, beta, correlation, DTE, a cost proxy, VIX and completed profitability memory. Missing values receive a neutral value plus availability indicators. Features may be unavailable without becoming an additional learned veto. Inputs are oriented as long-minus-short so the same economic interpretation can apply across pair directions.'),
            p('The retained feature block adds three economic quantities and three availability fields: EWMA variance premium, its name-history z-score, and expected-realized/implied variance ratio. Annualized variance is 252 times the exponentially weighted mean of squared daily log returns, using a 20-session half-life and at least 63 observations. The historical VRP z-score uses a strictly prior, shifted 252-session window with at least 126 observations. The resulting state has 33 fields [5].'),
            eq('estimated variance = 252 × EWMA_20(r²);  VRP = IV² − estimated variance'),
            p('This backward estimator is a simple variance forecast proxy, not independently demonstrated forecast superiority over HAR or GARCH. Normalizing the premium helps distinguish unusually rich or cheap volatility from a name’s structural premium. No future realized variance, completed current-trade outcome or hindsight event label is an action input.'),
            chart('feature_ablations',200),
            p('Eleven declared feature configurations were compared on frozen proposals. Variance-premium additions improve the prior learned reference from 0.962 to 1.006 at the original research capital. Volume/open interest, term structure, smile shape, physical tail moments and employee sentiment do not improve that reference; transcripts fall below the best fixed size. The IV-factor state is sparse and reproduces the baseline. More inputs therefore did not generally improve the strategy. These ablations share a researched sample, and their ranking is subject to model selection.') ]},
        {'title':'6. Learning, retraining and experimental controls', 'blocks':[
            h('Contextual sizing rather than a sequential Q-policy'),
            p('A small regression tree, with at most eight leaves, partitions observable states using completed directional outcomes. Cross-entropy method (CEM) search assigns one of the three positive sizing levels to each leaf. Candidate allocations are evaluated jointly on aggregated daily training P&L paths, so covariance between state allocations enters the Sharpe objective. The search uses 50 iterations and 192 candidates per iteration with a retained seed of 2026. Elite allocations update the distribution over choices, and the best allocation is retained [5].'),
            eq('SR = √252 × mean(daily excess return) / sample SD(daily excess return)'),
            p('The successful learner is contextual regression-tree policy search with a direct risk-adjusted objective. It is related to a contextual-bandit decision formulation, but it does not estimate Q-values, bootstrap future action values or learn within-trade inventory transitions. Calling the result simply “RL beats a neural network” would conflate algorithms and experiments. The current strongest comparison is learned sizing versus constant sizing; the current P25 candidate does not have a freshly retrained same-state supervised NN benchmark.'),
            h('Monthly rolling fits'),
            p('At each monthly cutoff, the model uses an 18-month window of eligible proposals whose outcomes have fully completed before that cutoff. A proposal cannot train its own decision. The January 2018 model uses 74 completed examples beginning 1 July 2016, with the final outcome on 29 December 2017. Subsequent months refit using completed history, including earlier evaluation months once their outcomes mature. The test sequence is therefore walk-forward, not the application of a single fixed pre-2018 model throughout 2018–2025.'),
            p('Five CEM seeds reproduce the retained candidate’s daily NAV exactly with the tree partitions fixed. This checks optimizer stability for this representation, not robustness to alternative features, trees or universes. A profit-only allocation objective reaches Sharpe 0.979 versus the direct-Sharpe result 1.006; a diagonal covariance variant is effectively tied at 1.007. Adaptive size allocation is useful in the observed sample, but the evidence does not require a unique RL mechanism.'),
            h('What is held constant'),
            p('The fixed and learned replays share all 608 qualified proposals, directions, contract IDs, cost conventions, risk limits and funding assumptions. Learned sizes are compared with constant 50% and 100% vega budgets. Smaller trades can permit additional overlapping entries while larger sizes can block them. Executed trade lists therefore differ: at $10m, fixed 100% enters 133 cohorts, fixed 50% enters 263 and learned sizing enters 212. This is an economic sizing/inventory effect, not evidence that all policies saw different signal opportunities.'),
            h('Accounting validation and research selection'),
            p('Saved allocations, contract cash flows and the shared portfolio ledger were independently reconciled. The $10m books agree to less than $0.000001 in daily ledger checks. All retained training outcomes precede their fit dates. These audits establish implementation consistency, not statistical independence. Feature blocks and strategy design have repeatedly used the same history, with inherited retrospective pair calendars. There is no untouched holdout, and the bootstrap comparisons do not correct for the full research search.') ]},
        {'title':'7. Main findings: signal and learned sizing', 'blocks':[
            chart('train_test',145),
            p('The initial training period contains a positive signal before optimizing sizes: fixed full-size Sharpe is 1.573. Learned sizing raises the optimized score to 2.015, a gain of 0.442. The fitted policy subsequently starts trading in January 2018. This result is worth reporting, but training improvement is expected when Sharpe is the objective. It is not a full pre-2018 funded-portfolio replay and does not cover all available pre-2018 data.'),
            p('At $10m funded equity, the monthly walk-forward learned strategy reaches 1.003 annualized cash-excess Sharpe, versus 0.769 for fixed full size and 0.761 for fixed half size. It earns $322,637 above the same-equity cash benchmark, across 212 entries and 42 active pairs. Total funded CAGR is 2.963%; cash-relative wealth CAGR is 0.342%. Maximum drawdown of wealth relative to cash is 0.491%. Low capital utilization explains the modest percentage excess return; it does not mechanically invalidate the excess Sharpe.'),
            chart('cash_relative_wealth',145),
            p('The original $345m research reference, funded as $1m times 345 candidate sleeves rather than an estimated margin requirement, scores 1.006 with 213 entries. The $10m replay freezes those learned policies but replays actual account funding and admission; it is not a new fit. Training uses aggregated paths and $0.65 commission labels, while evaluation uses the full ledger and $0.03 commissions. Compare learned versus fixed within each period, not the absolute train and test levels.'),
            p('For the original research reference, a 126-session block bootstrap gives a Sharpe interval of [0.137, 1.660] and an improvement over the best fixed size of [0.008, 0.519]. Improvement over the prior learned policy includes zero. These descriptive intervals resample the realized selected policies, do not refit them and are not corrected for strategy selection [6]. They are not newly calculated $10m intervals.') ]},
        {'title':'8. Annual performance and the source of profits', 'blocks':[
            chart('annual_results',125),
            table(['Year','$10m funded return','Above cash (pp)','Excess Sharpe'],yearly,[70,160,145,137]),
            p('Calendar returns include cash interest; “above cash” subtracts the same-equity cash benchmark’s calendar return in percentage points. Sharpe uses daily cash-excess returns and √252. The 2025 row is partial through 29 August, with final new signals on 30 June and subsequent expiry runoff. Losses in 2019, 2021 and 2024 are therefore not hidden by positive funded returns.'),
            p('Attribution of the original research reference shows that 2022 contributes about 55% of total net excess dollars. SNAP/UBER, GPS/TJX and AMD/MU are major winners. SNAP’s May 2022 guidance warning produces large convex gains in long puts; overlapping cohorts share the same event, so trade count overstates independent successes. Removing 2022 descriptively lowers the retained Sharpe to 0.575 rather than eliminating the entire positive estimate [6,7].'),
            p('Loss mechanisms differ. In 2024, short AAL puts lose sharply after a guidance reduction, while UPST/COIN loses mainly through expensive long-UPST carry. In 2021, OIH/HAL loses despite correlation near 0.91: strike crossing and execution-day delta drift matter. Initial vega neutrality is not protection against asymmetric jumps. These are retrospective diagnostics, not causal name bans or hindsight features.'),
            p('Some held quote midpoints also generate large reversing daily marks near expiry. Their existence warrants a common, causal mark-quality sensitivity across all models and years. Selectively removing inconvenient observations would overstate performance. The same tail and event exposure creates both important profits and important losses [7].') ]},
        {'title':'9. Capital deployment and portfolio-level learning', 'blocks':[
            p('The reference capital of $345m diluted returns because most sleeves were inactive. Frozen-policy replays at $1m, $3m, $10m and $25m tested actual funding rather than dividing P&L by an arbitrary margin estimate. Learned Sharpe is approximately 0.552, 0.893, 1.003 and 1.001, respectively. Reduced equity does not mechanically raise Sharpe: return and volatility scale together, while risk admission changes the trades. At $3m, held positions breach global internal limits on 45 days, so that result does not establish margin feasibility [8].'),
            chart('capital_scaling',170),
            table(['$10m policy','Cash-relative CAGR','Sharpe','Mean / peak reserve'],[
                ['1× learned expert','0.342%','1.003','2.01% / 18.01%'],
                ['2× expert, matched allowance','0.674%','0.999','3.94% / 36.14%'],
                ['4× expert, matched allowance','1.032%','0.811','6.81% / 63.03%'],
                ['Capital-aware pilot','0.362%','0.854','2.20% / 31.32%']],[165,125,65,157]),
            p('The multipliers preserve the expert’s learned trade sizes. Per-pair risk allowances increase with requested size while global $10m limits stay fixed. Doubling trade sizes without increasing the per-pair allowance instead reduces Sharpe to 0.564 and entries to 91: larger trades are rejected, reducing deployment rather than increasing it. The matched 2× run retains approximately one Sharpe and roughly doubles cash-relative CAGR, but has five held-position global stress breaches. The 4× run has 54 [9].'),
            p('A separate nine-parameter capital-aware overlay sees prior-close stress, reserve and hedge utilization, shared-name inventory, cash-relative drawdown, borrowing and expert size. It chooses 0.5/1/2/4× multipliers through sequential CEM trajectory search, refitting semiannually on 18 months of completed proposals. This limited pilot reaches 0.854 Sharpe and does not improve on the simple 1× or 2× expert. It uses capital states, but the small search budget and sparse cohorts do not establish that RL cannot manage portfolios [10].'),
            p('Internal reserves are not broker margin, and average reserve is not a valid denominator for deployable return. Peak collateral, stress losses, liquidity and forced liquidation matter. The best current evidence supports learned trade sizing with simple overall scaling, subject to the unmodeled margin and impact constraints.') ]},
        {'title':'10. Interpretation, limitations and next experiments', 'blocks':[
            p('The economic finding is specific: this researched P25 universe contains a relative-value signal, and state-dependent size allocation improves its historical cash-excess Sharpe over fixed sizing. The original paper’s separation of formation and learned decisions is useful, but our retained method is a contextual tree plus direct Sharpe search rather than a replication of its Q-learning. We have not demonstrated that neural sequential RL is necessary, that every smile region works, or that the current P25 learner beats a newly trained supervised model.'),
            p('The approximately one-Sharpe result is a selected historical estimate. Pair formation and repeated signal searches introduce selection bias; 2022 dominates dollars; 2024 is negative; and 2024–partial 2025 jointly has approximately zero excess Sharpe at the original reference. Vendor historical revisions, earnings information availability, midpoint quote quality, Black–Scholes-based approximations for American options, execution-day Greek drift, missing paths, nonlinear impact, early assignment and broker margin remain material limitations.'),
            p('The next controlled experiment should preserve the retained entries, directions and initial sizes and allow a small policy to hold or close both legs together at executable quotes. State should include remaining life, current versus entry IV, moneyness, relative gamma and shared event/name inventory. Closing costs must be included in training and evaluation. Compare identical starting cohorts against expiry holding before expanding the action space. The stronger confirmation would then freeze the design and evaluate newly obtained, untouched dates with prospective pair formation and a consistent quote-quality policy.'),
            h('References and reproducibility sources'),
            p('[1] Ning, B. and Lee, K. (2024). Advanced Statistical Arbitrage with Reinforcement Learning. arXiv:2403.12180. https://doi.org/10.48550/arXiv.2403.12180'),
            p('[2] Project ATM proxy benchmark: reports/nn_rl_benchmark.md. [3] Actual ATM sentiment checkpoint: reports/atm65_employee_sentiment.md. [4] Maturity coverage control: reports/p25_maturity_matched/findings.txt.'),
            p('[5] Retained strategy: config/p25_variance_premium_candidate.json; inputs: data/p25_variance_premium_inputs.csv; feature comparison: reports/p25_feature_search/comparison.csv. [6] Validation, seed checks and descriptive uncertainty: reports/p25_variance_validation/research_summary.txt; bootstrap.csv; leave_year_out.csv; initial_train_walk_forward.csv.'),
            p('[7] Ledger and event attribution: reports/p25_variance_years/findings.txt; annual_pairs.csv; annual_components.csv. Event links: SNAP 8-K, 23 May 2022 (SEC accession 0001193125-22-157565); AAL 8-K, 28 May 2024 (0001193125-24-148138). [8] Funding and annual returns: reports/p25_capital_sensitivity/comparison.csv and 10m_learned_annual.csv.'),
            p('[9] Trade-scale controls: reports/p25_trade_scale/comparison.csv. [10] Capital-policy pilot: reports/p25_capital_policy_matched/research_summary.txt and comparison.csv. All figures are regenerated from saved tables or NAV files; paper/source_manifest.json records hashes. Run scripts.build_research_publication through the guarded runner. No models or backtests were rerun for this publication.'),
            p('Academic research only. Source market and alternative data remain subject to institutional/vendor licenses. Numerical results are rounded for exposition; source files retain full precision. The draft’s authorship follows the repository contributors and can be edited in paper/manuscript.json.') ]},
    ]
    return pages


class PaperPage:
    def __init__(self, number, title, subtitle=''):
        self.fig = plt.figure(figsize=(8.5,11), facecolor='white')
        self.renderer = self.fig.canvas.get_renderer()
        self.x, self.width, self.y = 50, 512, 733
        self.number = number
        self.text(50,764,'MFE230Z  |  OPTIONS RELATIVE VALUE',9,BLUE,'sans-serif','bold')
        self.text(50,746,'RESEARCH DRAFT · 4 OCTOBER 2026',8,GRAY)
        self.heading(title, size=19 if number==1 else 17)
        if subtitle: self.paragraph(subtitle,size=11,color=GRAY,leading=14)
        self.fig.add_artist(plt.Line2D([50/612,562/612],[37/792,37/792],transform=self.fig.transFigure,color='#d0d6dc',linewidth=.7))
        self.text(50,22,'Kreitzman & Yang | Cash-excess Sharpe; 2025 partial',8,GRAY)
        self.text(546,22,str(number),9,GRAY)

    def text(self,x,y,text,size=10.35,color='#202832',family='serif',weight='normal'):
        return self.fig.text(x/612,y/792,text,fontsize=size,color=color,family=family,weight=weight,va='top')

    def lines(self,text,size,family='serif',weight='normal',width=None):
        font=FontProperties(family=family,size=size,weight=weight)
        limit=(self.width if width is None else width)*self.fig.dpi/72
        lines=[];line=''
        for word in text.split():
            trial=(line+' '+word).strip()
            pixels=self.renderer.get_text_width_height_descent(trial,font,ismath=False)[0]
            if pixels>limit and line: lines.append(line);line=word
            else: line=trial
        if line:lines.append(line)
        return lines

    def paragraph(self,text,size=10.35,color='#202832',leading=13.25):
        for line in self.lines(text,size):
            self.text(self.x,self.y,line,size,color)
            self.y-=leading
        self.y-=7

    def heading(self,text,size=12):
        if size==12:self.y-=3
        for line in self.lines(text,size,'sans-serif','bold'):
            self.text(self.x,self.y,line,size,BLUE,'sans-serif','bold');self.y-=size*1.25
        self.y-=8

    def image(self,name,height):
        im=Image.open(FIG/f'{name}.png')
        w,h=im.size
        height=min(height,self.width*h/w)
        ax=self.fig.add_axes([self.x/612,(self.y-height)/792,self.width/612,height/792])
        ax.imshow(im);ax.axis('off')
        self.y-=height+2
        self.paragraph(f'Figure {PAPER_FIGURES[name]}. '+CHARTS[name]['caption'],size=8,leading=10.1,color=GRAY)

    def table(self,block):
        widths=block.get('widths') or [self.width/len(block['headers'])]*len(block['headers'])
        for index,row in enumerate([block['headers']]+block['rows']):
            texts=[self.lines(str(v),8.8,'sans-serif','bold' if index==0 else 'normal',width=w-10) for v,w in zip(row,widths)]
            height=max(len(x) for x in texts)*11.3+8
            rect=plt.Rectangle((self.x/612,(self.y-height)/792),self.width/612,height/792,transform=self.fig.transFigure,facecolor='#edf2f7' if index==0 else ('#f8fafc' if index%2 else 'white'),edgecolor='none')
            self.fig.add_artist(rect)
            x=self.x
            for lines,w in zip(texts,widths):
                for n,line in enumerate(lines): self.text(x+5,self.y-4-n*11.3,line,8.8,BLUE if index==0 else '#202832','sans-serif','bold' if index==0 else 'normal')
                x+=w
            self.y-=height
        self.y-=10


def render_paper(pages):
    checks=[]
    with PdfPages(PAPER/'options_relative_value.pdf') as pdf:
        pdf.infodict().update(Title='From Equity Pairs to Options Relative Value',Author='Adam Kreitzman and Soo Xin Yang',Subject='MFE230Z research report; ten pages')
        for i,page in enumerate(pages,1):
            canvas=PaperPage(i,page['title'],page.get('subtitle',''))
            for block in page['blocks']:
                if block['type']=='paragraph':canvas.paragraph(block['text'])
                elif block['type']=='heading':canvas.heading(block['text'])
                elif block['type']=='equation':canvas.paragraph(block['text'],size=10,color=BLUE,leading=13)
                elif block['type']=='chart':canvas.image(block['name'],block['height'])
                elif block['type']=='table':canvas.table(block)
            checks.append(dict(page=i,bottom_content_points=canvas.y,overflow=canvas.y<49))
            pdf.savefig(canvas.fig)
            canvas.fig.savefig(PAPER/f'page_{i:02d}.png',dpi=95)
            plt.close(canvas.fig)
            del canvas
            gc.collect()
    if any(c['overflow'] for c in checks):
        raise ValueError(f'Page layout overflow: {checks}')
    css='''@page{size:letter;margin:0}body{margin:0;background:#e9edf2;color:#202832;font:10.1pt/1.30 Georgia,serif}.page{box-sizing:border-box;width:8.5in;height:11in;padding:.55in .69in;background:white;margin:20px auto;page-break-after:always;position:relative}h1,h2{font-family:Arial,sans-serif;color:#2463a6}h1{font-size:17pt}h2{font-size:12pt;margin:12px 0 6px}p{margin:0 0 7px}.meta,figcaption{font:8pt/1.3 Arial,sans-serif;color:#66717e}figure{margin:6px 0 10px}figure img{width:100%;object-fit:contain}table{width:100%;border-collapse:collapse;font:9pt Arial,sans-serif;margin:10px 0}td,th{padding:5px;text-align:left;border-bottom:1px solid #e2e7ec}th{background:#edf2f7;color:#2463a6}.eq{color:#2463a6}footer{position:absolute;bottom:.3in;font:8pt Arial,sans-serif;color:#66717e}@media print{body{background:white}.page{margin:0}}'''
    sections=[]
    for i,page in enumerate(pages,1):
        blocks=[]
        for b in page['blocks']:
            if b['type']=='paragraph':blocks.append('<p>'+html.escape(b['text'])+'</p>')
            elif b['type']=='heading':blocks.append('<h2>'+html.escape(b['text'])+'</h2>')
            elif b['type']=='equation':blocks.append('<p class="eq">'+html.escape(b['text'])+'</p>')
            elif b['type']=='chart':blocks.append(f'<figure><img src="figures/{b["name"]}.svg" style="height:{b["height"]}pt"><figcaption>Figure {PAPER_FIGURES[b["name"]]}. {html.escape(CHARTS[b["name"]]["caption"])}</figcaption></figure>')
            elif b['type']=='table':blocks.append(htmltable(b['headers'],b['rows']))
        sections.append(f'<section class="page"><div class="meta">MFE230Z · Research draft · 4 October 2026</div><h1>{html.escape(page["title"])}</h1>'+('<p class="meta">'+html.escape(page['subtitle'])+'</p>' if page.get('subtitle') else '')+''.join(blocks)+f'<footer>Kreitzman &amp; Yang · {i} / 10</footer></section>')
    (PAPER/'options_relative_value.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>From Equity Pairs to Options Relative Value</title><style>'+css+'</style></head><body>'+''.join(sections)+'</body></html>')
    return checks


def htmltable(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def deck(cap,fit,annual):
    slides=[]
    def slide(kicker,title,subtitle,body,notes):slides.append(dict(kicker=kicker,title=title,subtitle=subtitle,body=body,notes=notes))
    def bullets(*items):return '<ul>'+''.join('<li>'+x+'</li>' for x in items)+'</ul>'
    def image(name):return f'<figure><img src="../paper/figures/{name}.svg"><figcaption>{html.escape(CHARTS[name]["caption"])}</figcaption></figure>'
    slide('MFE230Z','From equity pairs to options relative value','Learned sizing of daily delta-hedged, initially vega-neutral downside puts', '<div class="hero">1.003 <span>walk-forward excess Sharpe at $10m</span></div>'+bullets('Adam Kreitzman and Soo Xin Yang','2018–29 August 2025 · Actual option contracts · Research draft'), 'The project begins with an equity pairs paper. Our extension asks whether learned decisions improve volatility relative value. The headline is a selected historical cash-excess Sharpe, not a promised live result.')
    slide('Motivating paper','Ning & Lee: formation and trading are separate problems','Advanced Statistical Arbitrage with Reinforcement Learning (2024)',bullets('<b>Formation:</b> empirical mean reversion time (EMRT) selects spread coefficients.','<b>Trading:</b> a Q-policy responds to recent spread movements.','Ten equity pairs: formed in 2022, traded in 2023.')+'<div class="equation">X = S<sub>A</sub> − B S<sub>B</sub></div>','EMRT measures how quickly significant historical deviations return to the mean. The paper then learns trading decisions instead of fixing a threshold. We borrow that separation, rather than replicate its formation rule.')
    slide('Original paper','What the Q-learner learns','Synthetic mean-reverting training; real equity evaluation',bullets('Recent changes become discrete states; actions buy, hold or sell under position restrictions.','Reward uses deviation from the true spread mean, minus transaction cost.','Synthetic paths supply that known mean; transfer to changing market regimes remains a question.'), 'Explain state, action and reward briefly. The paper uses simulated paths to learn because its reward needs the true mean. Its definition of four move bins with a four-step lookback implies 256 combinations; a later statement of 16 states needs clarification.')
    slide('Published evidence','The paper reports higher daily Sharpe in all ten pairs','Keep its daily metric separate from our annualized excess Sharpe',image('original_paper'),'These are arithmetic means of the published daily Sharpes, not annualized account results. Both formation and trading change, so the table does not isolate RL from EMRT.')
    slide('Criticism → research design','What we need to test in the volatility extension','Longer evaluation, economically correct rewards and controlled sizing comparisons',bullets('One equity trading year does not establish robustness across regimes.','Formation and trading change together; isolate learned sizing with common proposals.','IV-spread convergence is not the same as profitable delta-hedged options.','Use contract marks, hedges, financing, costs and observable inputs.'),'The point is not that the paper is wrong. It gives us a useful framework, but options demand a different payoff and risk representation. This motivates our actual-contract simulator and fixed-size controls.')
    slide('Our research path','ATM proxy signal did not transfer unchanged','We moved from approximate ATM payoff to actual contracts and separate smile ideas',htmltable(['Stage','Finding','Meaning'],[['ATM IV proxy','Sized 1.461; NN 1.036','Silver concentrated; full-size policy 1.497'],['Actual ATM calls','Sentiment checkpoint 0.368','Daily funded cash-excess results'],['P25 puts + learned size','1.006 reference; 1.003 at $10m','Retained contract-level candidate']]),'The old 1.461 proxy number remains historical evidence, not our current performance. Silver dominates that basket and the full-size Sharpe policy was stronger than sizing there. The current research found a more useful signal in downside puts. These rows are not controlled ablations of the same strategy.')
    slide('Universe and data','Broad research universe; sparse eligible trades','345 candidate pair sleeves → 608 proposals → 212 entries / 42 pairs at $10m',bullets('Actual option IDs, quotes, Greeks, underlying histories, dividends and rates.','Inputs-only P25 CSV: 36,327 rows; 33 feature columns plus identifiers and eligibility.','Monthly completed-history fits; no forward reward is an action input.','Inherited retrospective pair calendars; no untouched holdout.'),'The universe was expanded substantially beyond AMD and NVDA. But the retained entry rule is selective and formation calendars are retrospective. Full proposal coverage means every qualified state is kept, not that every option pair trades.')
    slide('Retained trade','Approximately 30-day, 25-delta puts','P25 is downside volatility; neutrality is imposed at entry',image('workflow')+bullets('Same option type; integer contract counts match initial vega within 10%.','Daily stock delta hedge; each leg held to expiry except audited corporate exits.'),'Contracts are selected at the previous close and checked on the next session. Execution deltas can drift. Vega neutrality does not equal gamma or jump neutrality, and vega is not continually rebalanced.')
    slide('Execution and risk','Account for actual positions and costs','A shared cash-and-stock ledger, independently reconciled',htmltable(['Item','Retained assumption'],[['Option execution','$0.03 / contract + 0.10 vol-point entry penalty per leg'],['Stock hedging','Daily delta hedge; 1 bp turnover cost'],['Training labels','$0.65 / contract hurdle'],['Risk admission','5% stress; 80% reserve; 2× gross stock hedge']])+bullets('Reserves are internal estimates, not broker margin. Held positions can breach limits.','No early assignment, forced liquidation or nonlinear market impact.'),'We use midpoint plus a penalty rather than full bid-ask crossing. The option penalty is charged on entry for expiry-held positions, not daily. Stricter training costs and execution costs differ. The risk limits are inherited research assumptions.')
    slide('Trade signal','Keep relative value separate from size allocation','Frozen direction rule; learned 50 / 75 / 100% vega budget',bullets('252-session IV-spread rank: lower / upper quintile.','Backward beta 0.85–1.15 and return correlation ≥ 0.5.','Positive completed 91 / 365-day directional memory; count minima 21 / 63.','Full-size reference: $450 vega per vol point per leg.'),'The learner adjusts size; it does not invent the signal or reverse its direction. There is no learned skip action in the successful retained candidate. Risk and quote checks can still reject a proposed entry.')
    slide('Economic state','Relative expected variance premium improves the state','27 existing inputs + three economic quantities and availability flags','<div class="equation">VRP = IV² − estimated variance<br>estimated variance = 252 × EWMA<sub>20</sub>(r²)</div>'+bullets('Long-minus-short EWMA premium, prior-history premium z-score and realized/implied variance ratio.','20-session half-life; historical normalization uses strictly prior 252 sessions.','Existing IV / RV, beta, correlation, DTE, VIX and profitability memory remain.'),'These are backward observable forecast proxies, not future RV. Missing features are neutral-filled and flagged. We have not separately proven this forecast dominates HAR or GARCH.')
    slide('Learning mechanism','A contextual sizing policy optimized for daily Sharpe','Regression tree → discrete states → CEM size allocation',bullets('At most 8 leaves; actions 50 / 75 / 100%.','50 search iterations × 192 candidate allocations; joint training P&L covariance.','Rolling 18-month completed-outcome training; monthly refits.','Direct contextual policy search—not Q-learning or a neural actor-critic.'),'Use precise terminology. In our project this sits under the RL research direction, but the retained model is contextual tree plus CEM. It has no within-trade Bellman updates. State-dependent sizing already changes portfolio admission and overlap.')
    slide('Training and evaluation','Sizing benefit appears before and after 2018','Within-period comparisons; different training and evaluation accounting',image('train_test'),'The first January 2018 fit uses 74 completed examples from July 2016 onward, ending in December 2017. Training improves from 1.573 to 2.015. The subsequent monthly walk-forward result at ten million improves from .769 to 1.003. Training gains are expected under direct optimization. The stronger evidence is walk-forward improvement; it is researched history, not an untouched test.')
    slide('Main portfolio result','Learned sizing improves the fixed-vega benchmarks','2018–August 2025 at $10m equity',image('cash_relative_wealth')+htmltable(['Sizing','Sharpe','Entries'],[['Fixed 100%','0.769','133'],['Fixed 50%','0.761','263'],['Learned 50/75/100%','1.003','212']]),'All three see the same qualified proposals and contracts. Their executed portfolios differ because integer sizes and risk admission change overlapping inventory. At ten million, the learner earns about 323 thousand dollars above cash and cash-relative CAGR is .342 percent.')
    slide('Annual results','Profits vary sharply across regimes','Positive cash interest must not conceal trading losses',image('annual_results'),'2022 is the strongest year. 2019, 2021 and 2024 have negative excess Sharpe. Total funded returns can still be positive because cash earns interest. 2025 is partial. The annual table is in the appendix and the paper.')
    slide('Attribution','Convexity creates both major wins and major losses','Pair contributions in the original research reference, before shared financing',image('pair_contributions'),'About 55 percent of total net excess dollars arrive in 2022. Long SNAP puts benefit from the guidance shock; short AAL puts are hurt by a different guidance shock in 2024. UPST losses include expensive long carry. Several cohorts share events, so entries are not independent observations. Do not turn hindsight winners and losers into name bans.')
    slide('Feature exploration','More signals did not generally mean higher Sharpe','Frozen proposals; original $345m research reference',image('feature_ablations'),'Variance-premium features improve .962 to 1.006. The other added blocks do not improve the prior learned baseline. This includes many alternative-data and smile ideas. The chart preserves negative findings. The small improvement over prior learned sizing is statistically uncertain.')
    slide('Capital deployment','Simple scaling beats the first capital-aware pilot','All multipliers retain the underlying learned trade sizes',image('capital_scaling'),'At ten million, average reserve is about two percent and peak eighteen percent. Doubling size and per-pair allowance roughly doubles cash-relative CAGR while keeping about one Sharpe. Four times size lowers Sharpe. The capital-aware overlay gets .854; this is a limited pilot, not proof that RL cannot manage portfolios.')
    slide('Evidence and limitations','One Sharpe is a research finding, not a live guarantee','What the current experiment establishes—and what it does not',bullets('Learned sizing beats fixed sizing in this sample; current P25 has no fresh supervised-NN comparison.','126-session reference bootstrap: Sharpe [0.137, 1.660]; best-fixed gap [0.008, 0.519].','Intervals are not selection-adjusted; repeated research and retrospective formation remain.','2022 concentration, negative 2024, quote-quality and margin/assignment limitations.'),'Bootstrap intervals refer to the original research reference, not newly calculated ten-million intervals. The same-state profit-only policy is close at .979. We should claim adaptive sizing value, not universal RL superiority.')
    slide('Conclusion and next step','Preserve the signal; test sequential exits separately','Equity relative-value framework → actual options → learned sizing',bullets('<b>Finding:</b> a downside-put relative-value signal with a persistent historical sizing benefit.','<b>Next controlled test:</b> identical entries / initial sizes; learn hold versus close both legs.','Use remaining life, moneyness, relative gamma and event / name inventory.','Then freeze the design and evaluate untouched dates with prospective pair formation.'),'The narrow scope is a strength: identify the economic signal first, then test whether a richer control policy adds value. Keep costs and executable marks common. Do not widen instruments, features and actions all at once.')
    slide('Appendix','Returns and Sharpe by year at $10m','Funded returns include cash; above-cash figures are percentage points',htmltable(['Year','Funded return','Above cash (pp)','Excess Sharpe'],[[str(r.year)+('*' if r.year==2025 else ''),f'{r.funded_return_pct:.3f}%',f'{r.return_above_cash_pp:+.3f}',f'{r.annualized_excess_sharpe:.3f}'] for r in annual.itertuples()])+'<p class="small">*2025 through 29 August; calendar-period returns are not annualized.</p>','Use this appendix for detailed return questions. Funded return, above-cash percentage points and excess Sharpe are distinct. A positive funded year can underperform cash.')
    slide('References','Sources and reproducibility','Updated presentation accompanies a ten-page paper',bullets('<a href="https://arxiv.org/abs/2403.12180">Ning & Lee (2024), arXiv:2403.12180</a>','<a href="../paper/options_relative_value.pdf">Ten-page paper (PDF)</a> · <a href="../paper/options_relative_value.html">Editable paper (HTML)</a>','<a href="../reports/p25_variance_validation/research_summary.txt">Main findings</a> · <a href="../reports/p25_capital_sensitivity/comparison.csv">Funding controls</a>','<a href="../reports/p25_capital_policy_matched/research_summary.txt">Capital-policy pilot</a> · <a href="../paper/source_manifest.json">Figure source hashes</a>'),'The paper records exact strategy assumptions and the source files for every figure. Data ends in August 2025, even though this draft is dated October 2026. All source data remain subject to existing licenses.')
    old=ROOT/'slides/index.html'; archive=ROOT/'slides/archive/atm_proxy_2026-09-27.html'
    if not archive.exists():archive.write_text(old.read_text().replace('src="img/','src="../img/'))
    script=ROOT/'slides/script.md'; script_archive=ROOT/'slides/archive/atm_proxy_script_2026-09-27.md'
    if not script_archive.exists():shutil.copy2(script,script_archive)
    css='''*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:#202832;font:20px/1.45 system-ui,-apple-system,sans-serif}.slide{height:100vh;min-height:720px;padding:42px 6vw 50px;background:white;border-bottom:1px solid #dbe2e9;position:relative;display:flex;flex-direction:column}.kicker{color:#2463a6;font-size:14px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}h1{font-size:clamp(30px,3vw,43px);line-height:1.15;margin:12px 0 8px;max-width:1100px}h2{font-size:19px;font-weight:450;color:#66717e;margin:0 0 15px}.body{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;gap:10px}ul{margin:10px 0;padding-left:27px}li{margin:13px 0;max-width:1150px}b,a{color:#2463a6}figure{margin:0;min-height:0;display:flex;flex-direction:column;align-items:center}figure img{display:block;width:100%;max-width:1100px;max-height:44vh;object-fit:contain}figcaption{font-size:12px;color:#66717e;max-width:1120px;margin-top:3px;line-height:1.35}table{width:100%;border-collapse:collapse;font-size:17px;max-width:1150px}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #e3e8ee}th{background:#edf2f7;color:#2463a6}.equation{padding:24px;background:#edf2f7;border-radius:10px;font-size:29px;color:#2463a6}.hero{color:#2463a6;font-size:110px;font-weight:750;line-height:1.1}.hero span{font-size:25px;display:block;color:#66717e;margin:12px 0}.small{font-size:13px;color:#66717e}.foot{position:absolute;bottom:15px;left:6vw;right:6vw;display:flex;justify-content:space-between;color:#66717e;font-size:12px}nav{position:fixed;right:18px;top:12px;z-index:10;font-size:13px}nav button{border:1px solid #ccd5df;background:white;padding:5px 10px;border-radius:4px;color:#2463a6;cursor:pointer}.notes{display:none;font-size:15px;padding:15px;background:#fff8e9}body.show-notes .notes{display:block}body.show-notes .slide{height:auto;min-height:100vh}#progress{position:fixed;bottom:0;left:0;height:4px;background:#2463a6;z-index:20}@media(max-width:800px){.slide{height:auto;min-height:100vh;padding:30px 5vw 55px}body{font-size:17px}table{font-size:13px}figure img{max-height:none}.hero{font-size:80px}}@media print{@page{size:landscape;margin:0}.slide{width:100%;height:7.5in;min-height:0;padding:30px 50px 38px;page-break-after:always;break-inside:avoid}h1{font-size:30px}h2{font-size:16px}body{font-size:17px}figure img{max-height:3.4in}figcaption{font-size:9px}table{font-size:13px}li{margin:9px 0}nav,#progress,.notes{display:none!important}.hero{font-size:95px}.foot{left:50px;right:50px;bottom:12px;font-size:10px}}'''
    bodies=[]
    for i,s in enumerate(slides,1):
        bodies.append(f'<section class="slide" id="slide-{i}"><div class="kicker">{html.escape(s["kicker"])}</div><h1>{html.escape(s["title"])}</h1><h2>{html.escape(s["subtitle"])}</h2><div class="body">{s["body"]}</div><aside class="notes">{html.escape(s["notes"])}</aside><div class="foot"><span>Options relative value · Research draft · 2025 partial</span><span>{i} / {len(slides)}</span></div></section>')
    js='''const slides=[...document.querySelectorAll('.slide')];function current(){return slides.reduce((best,x,i)=>Math.abs(x.getBoundingClientRect().top)<Math.abs(slides[best].getBoundingClientRect().top)?i:best,0)}function move(d){slides[Math.max(0,Math.min(slides.length-1,current()+d))].scrollIntoView({behavior:'smooth'})}addEventListener('keydown',e=>{if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)){move(1);e.preventDefault()}if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){move(-1);e.preventDefault()}if(e.key==='Home'){slides[0].scrollIntoView();e.preventDefault()}if(e.key==='End'){slides.at(-1).scrollIntoView();e.preventDefault()}if(e.key.toLowerCase()==='n')document.body.classList.toggle('show-notes')});addEventListener('scroll',()=>document.getElementById('progress').style.width=(current()+1)/slides.length*100+'%');'''
    css+=' .body:has(figure):has(table) figure img{max-height:34vh}@media print{.body:has(figure):has(table) figure img{max-height:3in}}'
    old.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>From Equity Pairs to Options Relative Value</title><style>'+css+'</style></head><body><nav><button onclick="move(-1)">←</button> <button onclick="move(1)">→</button> <button onclick="document.body.classList.toggle(\'show-notes\')">Notes</button></nav>'+''.join(bodies)+'<div id="progress"></div><script>'+js+'</script></body></html>')
    script.write_text(('# Presentation script: equity pairs to options relative value\n\nUpdated research deck. Slides 1–20 form the main narrative; slides 21–22 are appendix and sources. Press N in the HTML deck to show notes.\n\n'+''.join(f'## Slide {i} — {s["title"]}\n\n{s["notes"]}\n\n' for i,s in enumerate(slides,1))).rstrip()+'\n')
    return slides


def render_deck_pdf(slides, out=None, first_slide_number=1, total_slides=None, figure_dir=None):
    """Portable slide PDF, using the same HTML content and chart artifacts."""
    out=Path(out) if out is not None else ROOT/'slides'
    figure_dir=Path(figure_dir) if figure_dir is not None else FIG
    total_slides=total_slides or len(slides)
    out.mkdir(parents=True,exist_ok=True)
    (out/'previews').mkdir(exist_ok=True)
    checks=[]
    with PdfPages(out/'options_relative_value_slides.pdf') as pdf:
        pdf.infodict().update(Title='From Equity Pairs to Options Relative Value',Author='Adam Kreitzman and Soo Xin Yang')
        for i,s in enumerate(slides,1):
            fig=plt.figure(figsize=(16,9),facecolor='white')
            renderer=fig.canvas.get_renderer()
            y=589
            def draw(x,y,t,size=18,color='#202832',weight='normal'):
                fig.text(x/1152,y/648,t,fontsize=size,family='sans-serif',weight=weight,color=color,va='top')
            def lines(t,size,width=1040,weight='normal'):
                font=FontProperties(family='sans-serif',size=size,weight=weight)
                limit=width*fig.dpi/72;result=[];line=''
                for word in t.split():
                    trial=(line+' '+word).strip()
                    if renderer.get_text_width_height_descent(trial,font,ismath=False)[0]>limit and line:result.append(line);line=word
                    else:line=trial
                if line:result.append(line)
                return result
            def paragraph(t,size=18,color='#202832',weight='normal',gap=12):
                nonlocal y
                for line in lines(t,size,weight=weight):draw(56,y,line,size,color,weight);y-=size*1.3
                y-=gap
            paragraph(s['kicker'].upper(),12,BLUE,'bold',8)
            paragraph(s['title'],28,'#202832','bold',8)
            paragraph(s['subtitle'],17,GRAY,gap=16)
            body=re.sub(r'<img([^>]*)>',r'<img\1/>',s['body']).replace('<br>','<br/>')
            body=re.sub(r'&(?![a-zA-Z0-9#]+;)', '&amp;', body)
            node=ET.fromstring('<root>'+body+'</root>')
            def text(n):return ''.join(n.itertext())
            for b in node:
                if b.tag=='ul':
                    for li in b:paragraph('• '+text(li),19,gap=12)
                elif b.tag=='figure':
                    img=b.find('img');name=Path(img.attrib['src']).stem
                    im=Image.open(figure_dir/f'{name}.png');w,h=im.size
                    height=min(245 if node.find('table') is not None else 270,1040*h/w)
                    ax=fig.add_axes([56/1152,(y-height)/648,1040/1152,height/648]);ax.imshow(im);ax.axis('off')
                    y-=height+4
                    paragraph(text(b.find('figcaption')),10,GRAY,gap=10)
                elif b.tag=='table':
                    trs=b.findall('.//tr');ncols=len(trs[0]);widths=[1040/ncols]*ncols
                    for rowno,tr in enumerate(trs):
                        wrapped=[lines(text(td),15,w-14,'bold' if rowno==0 else 'normal') for td,w in zip(tr,widths)]
                        height=max(len(v) for v in wrapped)*20+12
                        fig.add_artist(plt.Rectangle((56/1152,(y-height)/648),1040/1152,height/648,transform=fig.transFigure,facecolor='#edf2f7' if rowno==0 else ('#f8fafc' if rowno%2 else 'white'),edgecolor='none'))
                        x=56
                        for wrappedcell,w in zip(wrapped,widths):
                            for j,line in enumerate(wrappedcell):draw(x+7,y-6-j*20,line,15,BLUE if rowno==0 else '#202832','bold' if rowno==0 else 'normal')
                            x+=w
                        y-=height
                    y-=12
                elif b.tag=='div':
                    if b.attrib.get('class')=='hero':paragraph(text(b),38,BLUE,'bold',18)
                    else:
                        fragments=[]
                        if b.text:fragments.append(b.text)
                        for ch in b:
                            if ch.tag=='br':fragments.append('\n')
                            else:fragments.append(text(ch))
                            if ch.tail:fragments.append(ch.tail)
                        for line in ''.join(fragments).split('\n'):paragraph(line,24,BLUE,gap=10)
                else:paragraph(text(b),13,GRAY)
            draw(56,24,'Options relative value · Research draft · 2025 partial',10,GRAY)
            number=first_slide_number+i-1
            draw(1030,24,f'{number} / {total_slides}',10,GRAY)
            checks.append(dict(slide=number,bottom_content_points=y,overflow=y<42))
            pdf.savefig(fig)
            fig.savefig(out/f'previews/slide_{i:02d}.png',dpi=75)
            plt.close(fig)
    if any(c['overflow'] for c in checks):raise ValueError(f'Slide overflow: {checks}')
    sheet=Image.new('RGB',(4*384,((len(slides)+3)//4)*216),'#d8dfe7')
    for i in range(len(slides)):
        im=Image.open(out/f'previews/slide_{i+1:02d}.png').convert('RGB');im.thumbnail((384,216));sheet.paste(im,((i%4)*384,(i//4)*216))
    sheet.save(out/'previews/contact_sheet.png')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript',type=Path,help='Re-render edited ten-page manuscript JSON instead of rebuilding its text')
    parser.add_argument('--append-only',action='store_true',help='Preserve teammate slides and append current research; do not rebuild the paper')
    parser.add_argument('--refresh-extension',action='store_true',help='Keep paper discussion and replace the obsolete strategy/results section; do not rebuild the paper')
    args=parser.parse_args()
    if args.append_only and args.refresh_extension:
        parser.error('Choose either --append-only or --refresh-extension')
    if args.append_only or args.refresh_extension:
        from scripts.append_research_slides import build_append_only
        build_append_only(replace_old_extension=args.refresh_extension)
        return
    FIG.mkdir(parents=True,exist_ok=True)
    (ROOT/'slides/archive').mkdir(parents=True,exist_ok=True)
    cap,fit,annual,features,overlay=charts()
    for path in ['config/p25_variance_premium_candidate.json','reports/p25_variance_validation/bootstrap.csv','reports/p25_variance_validation/leave_year_out.csv','reports/p25_variance_validation/research_summary.txt','reports/p25_variance_years/findings.txt','reports/p25_maturity_matched/findings.txt','reports/atm65_employee_sentiment.md','reports/nn_rl_benchmark.md','reports/p25_trade_scale/comparison.csv','reports/p25_capital_policy_matched/research_summary.txt']:
        source(path)
    pages=json.loads(args.manuscript.read_text()) if args.manuscript else manuscript(cap,fit,annual,features,overlay)
    if len(pages)!=10:raise ValueError('The manuscript must contain exactly ten pages')
    (PAPER/'manuscript.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
    checks=render_paper(pages)
    from scripts.append_research_slides import build_append_only
    appended=build_append_only(replace_old_extension=True)
    deck_count=appended['total_slides']
    slide_checks=appended['appended_slide_layout_checks']
    previews=[]
    for i in range(1,11):
        im=Image.open(PAPER/f'page_{i:02d}.png').convert('RGB');im.thumbnail((306,396));previews.append(im)
    sheet=Image.new('RGB',(5*306,2*396),'#d8dfe7')
    for i,im in enumerate(previews):sheet.paste(im,((i%5)*306,(i//5)*396))
    sheet.save(PAPER/'contact_sheet.png')
    manifest=dict(title='From Equity Pairs to Options Relative Value',draft_date='2026-10-04',data_end='2025-08-29',paper_pages=10,deck_slides=deck_count,sources=SOURCES,charts=CHARTS,layout_checks=checks,slide_layout_checks=slide_checks,models_or_backtests_rerun=False)
    (PAPER/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({'paper_pages':len(pages),'deck_slides':deck_count,'sources':len(SOURCES),'charts':len(CHARTS),'layout_checks':checks},indent=2),flush=True)


if __name__=='__main__':main()
