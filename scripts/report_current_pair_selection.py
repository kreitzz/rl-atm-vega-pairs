"""Readable current VIX pair diagnosis from audited saved ledger summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('reports/current_pair_selection')


def main():
    pairs=pd.read_csv(ROOT/'pairs.csv')
    rl=pairs[(pairs.model=='rl_sized')&(pairs.entries>0)].sort_values('net_pnl',ascending=False)
    nn=pairs[pairs.model=='nn_fixed'].set_index('pair')
    summary=pd.read_csv(ROOT/'summary.csv').set_index('model').loc['rl_sized']
    exclusions=pd.read_csv(ROOT/'exclusions.csv');exclusions=exclusions[exclusions.model=='rl_sized']
    assoc=pd.read_csv(ROOT/'associations.csv');assoc=assoc[assoc.model=='rl_sized']
    years=pd.read_csv(ROOT/'annual.csv')
    # All 35 active RL pairs shown; NN is a comparison of different accepted trades.
    ordered=rl.sort_values('net_pnl');y=np.arange(len(ordered))
    fig,ax=plt.subplots(figsize=(11,12))
    ax.barh(y,ordered.net_pnl/1000,color=np.where(ordered.net_pnl>=0,'#287d86','#b65a35'),label='RL net excess P&L')
    ax.scatter(nn.loc[ordered.pair,'net_pnl']/1000,y,color='#292929',marker='x',s=28,label='NN net excess P&L')
    ax.set_yticks(y,ordered.label);ax.axvline(0,color='gray',lw=.8)
    ax.set_xlabel('Net P&L above cash ($ thousands), 2018–August 2025')
    ax.set_title('Current VIX policy: all active RL pairs\nNN uses its own direction, timing and fixed sizing')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right');fig.tight_layout()
    fig.savefig(ROOT/'pair_pnl.png',dpi=160);plt.close(fig)
    lines=['# Current VIX model: which pairs drive performance?','',
        f'The matched universe contains 346 pairs, but RL trades only {len(rl)}: {int(summary.winning_pairs)} finish profitable and {int(summary.losing_pairs)} lose, across {int(summary.entries)} accepted entries. Net excess P&L is ${rl.net_pnl.sum():,.0f}. The other 311 pairs stay in cash under the existing policy and gates; their inactivity is not evidence that their potential strategies are unprofitable.','',
        'This analysis uses the user-selected VIX-only reference, beta 0.85–1.15, unchanged execution costs and hold-to-expiry accounting. Pair labels are the saved research labels and can reflect historical tickers. All outcomes are exploratory and already inspected.','',
        '## Largest winners and losers','',
        '| Pair | RL entries | RL net excess P&L | RL annualized excess Sharpe | NN net excess P&L |',
        '|---|---:|---:|---:|---:|']
    for row in pd.concat([rl.head(5),rl.tail(5).sort_values('net_pnl')]).itertuples():
        lines.append(f'| {row.label} | {row.entries} | ${row.net_pnl:,.0f} | {row.sharpe:.3f} | ${nn.loc[row.pair,"net_pnl"]:,.0f} |')
    lines+=['','![All active RL pairs compared with NN](current_pair_selection/pair_pnl.png)','',
        '## Concentration','',
        f'Profitable RL pairs contribute ${summary.winner_profit:,.0f}, offset by ${-summary.loser_loss:,.0f} from losing pairs. GES/TJX contributes {summary.top1_share_of_positive_profit:.1%} of positive-pair profit; the top three contribute {summary.top3_share_of_positive_profit:.1%}. The top five account for essentially the entire final net profit.','',
        '| Diagnostic | Annualized excess Sharpe | Net excess P&L |',
        '|---|---:|---:|']
    for row in exclusions.itertuples():
        label='Full current portfolio' if row.top_pairs_replaced_with_cash==0 else f'Replace top {row.top_pairs_replaced_with_cash} pairs with cash'
        lines.append(f'| {label} | {row.sharpe:.3f} | ${row.net_pnl:,.0f} |')
    lines+=['','These are hindsight concentration diagnostics. They retain the original capital allocation and replace removed sleeves with their cash benchmarks. They are not proposed selection rules or retrained portfolios.','',
        '## Pair quality versus policy quality','',
        'The five largest RL winners also profit under the NN, which is useful agreement, although the models share inputs, costs, training history and gates. Across the 35 pairs traded by both models, 10 profit in both, 11 lose in both and 14 disagree in sign. THC/HCA is −$22.2k for RL but +$28.9k for NN; CSX/UNP is −$8.3k versus +$6.4k. RCL/NCLH loses in both. Pair identity alone therefore does not explain the losses; direction, entry timing and size matter.','',
        '## Do observable pair characteristics separate winners?','',
        '| Entry characteristic | Winner median | Loser median | Rank correlation with pair Sharpe |',
        '|---|---:|---:|---:|']
    labels={'beta_distance':'Absolute beta distance from one','correlation':'Stock-return correlation','mean_iv':'Mean contract IV (vol points)',
        'absolute_spread':'Absolute contract IV spread (vol points)','dte_gap':'Expiry gap (days)',
        'earnings_inside_30d':'Fraction of entries with earnings inside 30 days','short_higher_iv':'Fraction shorting the higher-IV leg',
        'vix_previous_close':'Previous-close VIX'}
    for row in assoc.itertuples():
        lines.append(f'| {labels[row.feature]} | {row.winner_median:.3f} | {row.loser_median:.3f} | {row.rank_correlation_with_sharpe:.3f} |')
    lines+=['','These are pair-level means measured only on accepted entry dates, then compared across active pairs. They are descriptive associations, not causal effects or independent significance tests. The narrow existing beta/correlation gates restrict what can be learned about excluded pairs. There is no strong monotonic separator here. In particular, RCL/NCLH loses despite roughly 0.929 stock-return correlation, while GES/TJX wins with roughly 0.598. Higher correlation alone is not sufficient.','',
        '## Costs, tails and stability','',
        f'Only {int(summary.gross_positive_net_negative)} of {int(summary.losing_pairs)} RL losers are gross-positive before execution, borrow and financing: BAC/C, C/WFC and AEO/TJX. The other 12 already lose before those charges. Costs are not the main explanation for most losing pairs.','',
        f'{int(summary.losers_positive_without_worst_day)} of the 15 losers would be positive if their worst daily mark were deleted, but only {int(summary.losers_positive_without_worst_year)} would be positive excluding their worst year. Deleting a day is not a tradable counterfactual: it changes the marked path and can remove a temporary quote move without changing terminal economics. RCL/NCLH still loses after excluding its worst day, and its six RL entries all occur in 2023. THC/HCA loses about $21.9k in 2022, but its NN profits overall, so it should not simply be blacklisted based on the RL result.','',
        f'Median accepted entries per active RL pair: {rl.entries.median():.0f}; {(rl.entries<=5).sum()} of {len(rl)} have five or fewer. Median number of entry years: {rl.entry_years.median():.0f}. Only {int(summary.pairs_traded_both_periods)} pairs have entries in both 2018–2021 and 2022–2025, too few for a reliable persistence conclusion. Counts are overlapping cohorts, not independent observations.','',
        'GES/TJX has 13 RL entries, all in 2022–2023. SPWR/CSIQ has five, all in 2021. URI/AER has seven across three entry years, and OKE/KMI is profitable in both models (its NN has 20 entries across five years). These are research candidates for validation, not established durable winners.','',
        'Net excess P&L includes interest on accumulated gains or losses relative to cash. A pair can therefore show small later-year P&L without new trades. The annual CSV separately reports net trading P&L before financing to distinguish this carry from new trading profit. Sharpe uses daily cash-excess returns and √252 over the full evaluation calendar, including flat periods.','',
        '## Next useful test','',
        'Keep the current universe while testing a causal pair-quality gate based only on completed historical shadow trades, with minimum independent-window coverage and performance stability. The existing directional 91/365-day profitability memory already filters entries; simply adding another past-profit mean would duplicate it. A useful extension would assess reliability across non-overlapping windows and penalize dependence on a few events, then validate subsequent performance on a fixed roster. Selecting the winners listed above using full-period results would overstate the opportunity.','',
        '[All pair metrics](current_pair_selection/pairs.csv) · [Year-by-year P&L](current_pair_selection/annual.csv) · [Entry characteristics](current_pair_selection/entries.csv) · [Concentration](current_pair_selection/exclusions.csv) · [Feature associations](current_pair_selection/associations.csv) · [Industry groups](current_pair_selection/industries.csv)','',
        'Verification: each pair’s daily excess P&L reconciles to terminal NAV less its cash benchmark; aggregate totals and Sharpe match the saved VIX comparison. No model was retrained and no trading policy was changed for this diagnosis.','']
    ROOT.with_suffix('.md').write_text('\n'.join(lines))
    funnel_path=ROOT/'entry_funnel.csv'
    if funnel_path.exists():
        f=pd.read_csv(funnel_path);f=f[f.model=='rl_sized']
        stages=[('covered_signal_dates','Usable dates within historical membership'),('beta_correlation','Beta/correlation eligible'),
            ('plus_memory_either_direction','Also passes memory in at least one direction'),
            ('plus_validation_enabled','Also has an enabled validation month'),('nonzero_final_action','Final nonzero RL action')]
        extra=['','## Why only 35 pairs trade','',
            '| Cumulative eligibility stage | Pairs with any opportunity | Pair-signal dates |','|---|---:|---:|']
        for key,label in stages:
            group=f[f.stage==key]
            extra.append(f'| {label} | {group.dates.gt(0).sum()} | {group.dates.sum():,} |')
        extra+=['','The initial matched roster is 346 pairs. Historical membership and usable state coverage leave 285 with evaluation signal dates; 61 have none. These counts are not a count of all stock-calendar dates. The beta range is 0.85–1.15 with correlation ≥0.5. RL additionally requires positive directional net shadow payoff over both 91 and 365 days, with at least 21 and 63 completed observations respectively. A separate preceding-year validation must produce positive daily excess Sharpe; minimum training history also applies. These are hard vetoes in addition to the learned policy.','',
            'Of 1,360 nonzero RL signals, 280 enter, 582 hit entry risk limits, 338 are blocked while the held portfolio breaches risk limits, 93 lack an eligible entry quote, and 67 fail integer-contract vega matching. All 35 pairs with nonzero final actions have at least one accepted entry: execution/risk gates reduce frequency here, while the earlier screens explain the narrow pair breadth. Since positions remain held to expiry, daily repeated signals are not independent new opportunities.','',
            'Across RL pair-month fits, 5,199 have no prior-validation opportunities passing eligibility and directional memory, 1,619 fail the validation gate, 1,525 lack sufficient training, and 549 are enabled. The final-action count also combines policy-flat decisions with chosen directions rejected by memory; the saved outputs do not identify those two causes separately.','',
            'A clean participation experiment would keep beta eligibility, costs and risk limits fixed, retain profitability memory as model inputs, and compare the current hard directional profitability veto with a state-only version. This would test whether the stacked filters are suppressing useful opportunities without using full-period winners to select pairs. It is a proposed experiment, not a result or a policy change.','',
            '[Entry funnel](current_pair_selection/entry_funnel.csv) · [Monthly gates](current_pair_selection/fit_gates.csv) · [Execution rejections](current_pair_selection/execution_gates.csv)','']
        with ROOT.with_suffix('.md').open('a') as stream:stream.write('\n'.join(extra))


if __name__=='__main__':main()
