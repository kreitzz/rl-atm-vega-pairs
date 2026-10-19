"""Regenerate the report, figures and tables from small frozen research outputs.

This is publication reproduction, not a market-data download or backtest.
Run from repository root: python -m scripts.strategy_economics
"""
from pathlib import Path
import hashlib
import html
import json
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/strategy_economics'
DATA = OUT/'data'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/strategy_economics/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN = '#255d79', '#c46b46', '#40816a'


def read(name):
    return pd.read_csv(DATA/name)


def figure(fig,name):
    fig.savefig(OUT/'figures'/name,dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    (OUT/'tables').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False})
    manifest=json.loads((DATA/'manifest.json').read_text())
    # Accept the manifest's file listing, but never an unchecked path traversal.
    listing=manifest.get('files',manifest.get('snapshots',{}))
    if isinstance(listing,list):listing={x.get('file',x.get('name')):x for x in listing}
    assert listing,'Input manifest must list snapshot files'
    for name,record in listing.items():
        p=(DATA/name).resolve()
        assert p.is_relative_to(DATA.resolve()) and p.is_file(),name
        expected=record.get('snapshot_sha256',record.get('sha256'))
        assert expected and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
    trades=read('trade_outcomes.csv')
    assert len(trades)==2599 and trades.cohort.is_unique
    np.testing.assert_allclose(trades.option_profit+trades.hedge_profit,trades.combined_profit,atol=1e-6)
    trades['path']=np.where(trades.persistence<=.2,'More reversing','More persistent')
    trades['gap']=np.where(trades.entry_iv>=trades.rv,'IV ≥ RV','IV < RV')
    trades['profit_per_premium']=trades.combined_profit/trades.initial_premium
    assert trades[['entry_iv','rv','persistence','combined_profit']].notna().all().all()
    cross=trades.groupby(['path','gap'],sort=True).agg(Trades=('cohort','size'),
        combined_profit=('combined_profit','sum'),mean_premium_return=('profit_per_premium','mean'))
    cross.to_csv(OUT/'tables/ivrv_groups.csv')
    grouped=trades.groupby('path')[['option_profit','hedge_profit','combined_profit']].sum().reindex(['More reversing','More persistent'])
    bridge=read('profit_bridge.csv')
    np.testing.assert_allclose(bridge.dollars.iloc[:2].to_numpy(),[trades.option_profit.sum(),trades.hedge_profit.sum()],atol=.001)
    assert abs(bridge.dollars.sum()-3748711.2606)<.01
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    labels=['Options after\noption charges','Stock hedge\nbefore costs','Hedge\ntrading costs','Interest above\ncash benchmark','Stock\nborrowing','Hedge\ndividends','Total above\ncash']
    vals=list(bridge.dollars/1e6)+[bridge.dollars.sum()/1e6]
    running=0.
    for i,value in enumerate(vals):
        base=0 if i==len(vals)-1 else running
        ax.bar(i,value,bottom=base,color=BLUE if i==len(vals)-1 else GREEN if value>=0 else ORANGE)
        ax.annotate(f'{value:+.2f}',(i,base+value),xytext=(0,6 if value>=0 else -16),textcoords='offset points',ha='center',fontsize=10)
        if i<len(vals)-1:running+=value
    ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Profit ($ millions)',title='1. The option and hedge components must be read together')
    ax.axhline(0,color='#999',lw=.7)
    ax.set_ylim(0,float(np.cumsum(vals[:-1]).max())+.75)
    figure(fig,'01_profit_bridge.png')
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for i,(col,label,color) in enumerate([('option_profit','Options',BLUE),('hedge_profit','Stock hedge',ORANGE),('combined_profit','Together',GREEN)]):
        ax.bar(np.arange(2)+(i-1)*.24,grouped[col]/1e6,.23,label=label,color=color)
    ax.set(xticks=[0,1],xticklabels=['More reversing paths (1,292 trades)','More persistent paths (1,307 trades)'],ylabel='Profit ($ millions)',title='2. A profitable stock hedge can accompany an overall losing trade')
    ax.axhline(0,color='#999',lw=.7);ax.legend(frameon=False);figure(fig,'02_path_offsets.png')
    fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
    matrix=cross.reset_index();positions=np.arange(len(matrix))
    ax.bar(positions,matrix.combined_profit/1e6,color=np.where(matrix.combined_profit>=0,GREEN,ORANGE))
    ax.axhline(0,color='#999',lw=.7)
    for i,row in matrix.iterrows():
        val=row.combined_profit/1e6
        ax.annotate(f'${val:+.2f}m\n{row.Trades:,} trades',(i,val),ha='center',xytext=(0,7 if val>=0 else -30),textcoords='offset points',fontsize=10)
    ax.set(xticks=positions,xticklabels=[f'{r.path}\n{r.gap}' for r in matrix.itertuples()],ylabel='Combined option + hedge profit ($ millions)',title='3. IV versus subsequent RV separates outcomes within both path groups')
    ax.margins(y=.22);figure(fig,'03_ivrv_groups.png')
    examples=read('trade_examples.csv');examples['date']=pd.to_datetime(examples.date)
    selected=['2020-02-21|single_101803|ATM_STRADDLE','2024-04-23|single_213114|ATM_STRADDLE']
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
    for j,cohort in enumerate(selected):
        b=examples[examples.cohort.eq(cohort)].sort_values('date');assert len(b)>2
        np.testing.assert_allclose((b.shares_carried*b.spot.diff()).iloc[1:],b.hedge_profit.iloc[1:],atol=1e-6)
        axes[0,j].plot(b.date,b.spot,color=BLUE,label='Stock price')
        axes[0,j].axhline(b.strike.iloc[0],color='#999',ls='--',label='Strike')
        axes[0,j].set(title=['Large hedge-gain example: security 101803','Large hedge-loss example: security 213114'][j],ylabel='Stock price ($)')
        axes[0,j].legend(frameon=False,fontsize=8)
        axes[1,j].step(b.date,b.shares_carried,where='pre',color=ORANGE)
        axes[1,j].set(ylabel='Shares carried into session')
        for field,label,color in [('option_profit','Options',BLUE),('hedge_profit','Stock hedge',ORANGE),('combined_profit','Together',GREEN)]:
            axes[2,j].plot(b.date,b[field].cumsum()/1000,label=label,color=color)
        axes[2,j].set(ylabel='Cumulative profit ($ thousands)');axes[2,j].legend(frameon=False,fontsize=8)
        for ax in axes[:,j]:
            locator=mdates.AutoDateLocator(minticks=3,maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',labelsize=9)
    figure(fig,'04_trade_examples.png')
    accounts=read('account_daily.csv');accounts['date']=pd.to_datetime(accounts.date)
    perf=read('account_performance.csv');perf=perf[perf.period.eq('later')].set_index('case')
    intervals=read('sharpe_intervals.csv')
    assert set(perf.index)=={'hold_control','conditional_half','constant_lower'}
    book={case:b.set_index('date').sort_index() for case,b in accounts.groupby('case')}
    for b in book.values():assert b.index.equals(book['hold_control'].index)
    for case,b in book.items():
        r=b.excess_return.loc['2022':]
        assert len(r)==int(perf.loc[case,'observations'])==918
        np.testing.assert_allclose(r.mean()/r.std(ddof=1)*np.sqrt(252),perf.loc[case,'annualized_excess_sharpe'],rtol=1e-10)
        np.testing.assert_allclose(r.mean()*252,perf.loc[case,'annual_excess_mean'],rtol=1e-10)
        np.testing.assert_allclose(r.std(ddof=1)*np.sqrt(252),perf.loc[case,'annual_excess_volatility'],rtol=1e-10)
    flags=read('warning.csv');flags['first_changed_market_return_date']=pd.to_datetime(flags.first_changed_market_return_date)
    flags=flags.dropna(subset=['first_changed_market_return_date']).set_index('first_changed_market_return_date')
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1,1.5,1]})
    score=flags.prediction/flags.prior_nav/flags.training_risk_cut_per_nav
    axes[0].plot(score.index,score,color=BLUE,lw=1)
    axes[0].axhline(1,color=ORANGE,ls='--',lw=1)
    axes[0].fill_between(score.index,0,score,where=score>1,color=ORANGE,alpha=.2)
    axes[0].set(ylabel='Risk score / cutoff',title='5. Risk cuts helped unevenly over time')
    labels={'hold_control':'Unchanged 90/10','conditional_half':'Conditional straddle cuts','constant_lower':'Constant 90/5 + cash'}
    for case,color in [('hold_control',BLUE),('conditional_half',GREEN),('constant_lower',ORANGE)]:
        b=book[case].loc['2021-12-31':]
        growth=(b.nav/b.nav.iloc[0])/(b.cash_benchmark/b.cash_benchmark.iloc[0])-1
        axes[1].plot(b.index,100*growth,label=labels[case],color=color,lw=1.6)
    axes[1].set(ylabel='Growth above cash (%)');axes[1].legend(frameon=False,fontsize=9,loc='upper left')
    delta=(book['conditional_half'].excess_return-book['hold_control'].excess_return).loc['2022':].cumsum()*100
    axes[2].plot(delta.index,delta,color=GREEN);axes[2].axhline(0,color='#999',lw=.7)
    axes[2].set(ylabel='Accumulated return difference\n(percentage points)',xlabel='Calendar date')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_xlim(delta.index[0],delta.index[-1])
    figure(fig,'05_risk_trial.png')
    display_cross=pd.DataFrame({'Price path':matrix.path,'Entry IV vs later RV':matrix.gap,'Trades':matrix.Trades,
        'Combined profit ($m)':matrix.combined_profit.map(lambda x:f'{x/1e6:+.2f}'),
        'Mean profit / entry premium':matrix.mean_premium_return.map(lambda x:f'{100*x:+.1f}%')})
    display_perf=pd.DataFrame([{'Account':labels[case],
        'Annualized mean excess return':f'{100*r.annual_excess_mean:.3f}%',
        'Volatility/year':f'{100*r.annual_excess_volatility:.3f}%',
        'Excess Sharpe':f'{r.annualized_excess_sharpe:.2f}',
        'Worst cash-relative drawdown':f'{100*r.relative_max_drawdown:.3f}%'} for case,r in perf.iterrows()])
    display_cross.to_csv(OUT/'tables/ivrv_display.csv',index=False)
    display_perf.to_csv(OUT/'tables/account_performance.csv',index=False)
    render_report(display_cross,display_perf)
    verify={'status':'passed','trade_count':len(trades),'combined_profit':float(trades.combined_profit.sum()),
        'profit_above_cash':float(bridge.dollars.sum()),'later_sessions':918,
        'snapshot_hashes_verified':True,'account_sharpes_recomputed':True,
        'reproduction_scope':'Figures and tables from frozen derived research outputs; not a raw-data or account replay.'}
    (OUT/'verification.json').write_text(json.dumps(verify,indent=2)+'\n')
    print(json.dumps(verify,indent=2))


def render_report(cross,performance):
    blocks=[]
    def h(text):blocks.append(('h',text))
    def p(text):blocks.append(('p',text))
    def img(name,caption):blocks.append(('img',(name,caption)))
    h('What drives the short-straddle returns?')
    p('A focused research extension: economic attribution and one risk-management experiment. The original repository strategy and paper remain the baseline. This report separates an explanation of historical profits from evidence of a new trading edge.')
    h('The strategy and the two accounts studied')
    p('The inherited system combines 25-delta put pairs with short at-the-money straddles. Put pairs buy one stock’s put and sell another’s, matching vega approximately at entry, then hedge delta daily. Their relative-volatility trade is different from selling both a call and a put on one stock. The evidence below concerns the straddle sleeve; it is not proof of the put-pair return mechanism.')
    p('The straddle shorts a matching call and put, carries a daily stock delta hedge and generally holds to expiry. Entry screening and contract/risk checks precede sizing. A ridge profit score is mapped to buckets with searched size assignments; an outer Q allocation exists in the inherited research. Neither learning layer was introduced by this attribution study. The final risk experiment instead uses a fixed 90% put / 10% straddle budget and half-size entries in both sleeves.')
    p('Two populations must stay separate: Figures 1–4 describe 2,599 actual straddle cohorts over 2018–29 August 2025 from the saved straddle account; Figure 5 and the performance table describe a separate $10 million joint executable account over 2022–29 August 2025. The joint account uses whole contracts, costs, funding and shared risk limits. Its Sharpe must not be compared as if it came from the full-history straddle account.')
    h('1. Positive option and hedge profits are not two separate edges')
    img('01_profit_bridge.png','Figure 1. Full-history straddle-account profit bridge, 2018–29 August 2025. Read the option and hedge components together; the other entries reconcile them to profit above an all-cash benchmark.')
    p('Options after their charges contributed $1.74 million and stock hedges before hedge costs contributed $1.75 million. Their combined $3.49 million becomes $3.75 million above cash after the remaining hedge costs, borrowing, dividends and interest relative to the cash benchmark. Option P&L includes time decay, price changes and volatility revaluation; it is not a pure theta attribution. Stock-hedge P&L is not an isolated gamma cost.')
    p('A short straddle has negative gamma. As the stock rises, its hedge tends to buy shares; as the stock falls, its hedge tends to sell shares. Continuing moves can make the held stock hedge profitable while the short option loses more. Reversals can hurt the hedge while the options retain more premium. The negative-gamma cost belongs to the combined option-and-hedge economics, so the sign of the stock component alone cannot identify an independent source of return.')
    h('2. Price paths explain the surprising hedge sign')
    img('02_path_offsets.png','Figure 2. Full-history cohorts grouped using their realized price paths. Options include option charges; combined P&L excludes remaining stock-hedge costs, dividends and financing.')
    p('More reversing paths contributed +$4.78 million together: option gains exceeded hedge losses. More persistent paths contributed −$1.29 million together: hedge gains did not cover option losses. Across the groups, stock-hedge gains of about $14.96 million outweighed stock-hedge losses of about $13.21 million, explaining the positive $1.75 million aggregate hedge contribution. This does not mean the strategy profits overall from persistent trends.')
    p('Persistence is |log(final price / entry price)| divided by the sum of absolute daily log returns over the holding period. Values at or below 0.20 are called more reversing; values above 0.20 more persistent. This is a descriptive boundary applied after observing the path, not a forecast, optimized entry condition or claim that ordinary Brownian motion belongs to one group.')
    h('3. IV versus subsequent RV supplies the economic explanation')
    img('03_ivrv_groups.png','Figure 3. Within both price-path groups, aggregate combined profits are positive when entry IV exceeds subsequent RV and negative when it does not. Unequal counts and sizes affect totals; the accompanying table also reports mean profit relative to collected entry premium.')
    blocks.append(('table',cross))
    p('Entry IV is the common annualized European/dividend-escrow implied volatility that reprices the combined call-and-put entry mid premiums. Subsequent RV is sqrt(252 × mean(daily log return squared)) from the entry close to actual exit close, without subtracting the sample mean. These are stock-specific measures; VIX is not substituted for either.')
    p('The split is consistent with selling volatility at a premium to the movement subsequently realized. But it is not an identity for individual trades: 220 trades lost despite entry IV exceeding subsequent RV. Discrete hedging, jumps, changing gamma, costs and different holding paths matter. Calendar-time annualization changes 69 gap classifications and a dividend adjustment changes six. Three trades exited early. This sensitivity limits any claim that a simple unweighted IV−RV comparison exactly reproduces option P&L.')
    p('Both subsequent RV and persistence use future observations. They explain realized outcomes but cannot be used as entry signals in this form. Mean profit divided by entry premium is also not a Sharpe or an equal-risk comparison. This evidence is consistent with volatility-risk compensation; it does not establish exposure-adjusted alpha or prove the return is fully explained by a risk premium.')
    h('4. Inspect actual trades rather than treating the components as abstractions')
    img('04_trade_examples.png','Figure 4. Two deliberately contrasting historical examples. Top: stock price against strike. Middle: shares carried into each session (positive means long, negative short). Bottom: cumulative option, hedge and combined profit. These are illustrations selected for a large hedge gain and a large hedge loss, not representative performance estimates.')
    p('Read each column vertically. The stock position is the daily delta hedge, not a separate trend strategy. Its profit is earned on shares already held during that session’s price move. The option and stock curves offset; the green combined curve is the first economic result to inspect. Security IDs are stable identifiers, rather than potentially renamed stock tickers.')
    p('The attribution uses saved holdings and recovered quote paths. One historical quote-availability convention was inferred; replaying with the observed recovered quotes changes the full-history hedge total by $367.52. This publication retains the reconciled saved-account attribution. It does not claim exact independent regeneration from raw quotes, and the difference is disclosed rather than treated as a new profit source.')
    h('5. Can managing accumulated exposure improve the account?')
    p('One bounded candidate predicts the magnitude of the next-session straddle-book market P&L using a nonnegative regression of gross straddle vega and the root-mean-square of its preceding 21 session P&Ls. It predicts absolute movement, not loss direction. Coefficients and a 75th-percentile risk-per-NAV cutoff were fitted on 2018–2021 and frozen before the management trial. The reference book remains unchanged alongside the managed account, so trimming cannot mechanically turn its warning off.')
    p('When the warning is high, new straddles are paused and existing cohorts receive a next-close target of half their original integer contracts, rounded down. One-contract cohorts close. Trims remain in place; only new entries resume when the warning clears. Put rules remain unchanged. The first possible action was 3 January 2022. The trial did not search new cutoffs or reduction sizes.')
    img('05_risk_trial.png','Figure 5. Top: warning score divided by its cutoff, aligned to the first market return a next-close trim could affect. Middle: all three executable accounts’ growth relative to cash. Bottom: accumulated daily excess-return difference between conditional cuts and holding unchanged; rising segments favor cuts. The difference is not an isolated causal attribution to particular fills.')
    blocks.append(('table',performance))
    p('The conditional account’s excess Sharpe rose from 1.37 to 1.58, mainly through lower volatility with little change in mean return. The constant-lower control has budgets of 90% put, 5% straddle and 5% idle cash; it does not match the conditional account’s average exposure. Its Sharpe reached 1.15, so the observed result is not equivalent to simply reducing the straddle budget at all times. Annualized mean excess return in the table is the arithmetic daily excess-return mean multiplied by 252, not CAGR. Nevertheless, the paired 126-session block-bootstrap 95% interval for the Sharpe gain versus holding is −0.22 to +0.66, and cash-relative maximum drawdown was slightly worse. The bootstrap conditions on the saved histories and does not adjust for previous research or selection. The gain is promising but unproven. All these historical dates have been researched; this is not a fresh holdout.')
    h('What the work establishes—and what it does not')
    p('The substantive contribution is a coherent explanation of the surprising hedge profits, a decomposition of outcomes by realized path and IV–RV, and one interpretable risk-management candidate. The evidence does not establish new alpha, an industry-selection edge or reliable improvement from the learning layers. A proper market/sector exposure attribution and independent validation of the risk rule remain open. No new allocation or entry policy is promoted from these descriptive splits.')
    p('Unsuccessful forecast, sizing and allocation variants were also investigated. Their detailed development outputs are excluded from this focused publication. They did not establish a replacement policy and the retained risk candidate should not be read as the only idea ever examined. Supporting audit and reconstruction work is likewise not counted as a strategy contribution.')
    h('Reproduction and provenance')
    p('From the repository root run: python -m scripts.strategy_economics. This verifies the retained input hashes, recomputes the displayed groupings and later account Sharpes, and regenerates this report, its figures and compact tables offline. It reproduces the publication from frozen derived research outputs; it does not download WRDS data, rerun the backtest, refit the models or certify the original market-data pipeline.')
    p('data/manifest.json records each selected snapshot’s source path and SHA256, the historical working commit, and the remote baseline. The original strategy paper and repository code remain available at their existing locations. No raw option quote history is included in this extension. See README.md for the source map, dependency versions and explicit limits.')
    markdown=[];web=[]
    for kind,value in blocks:
        if kind=='h':
            markdown.append(('## ' if markdown else '# ')+value)
            web.append(f'<h2>{html.escape(value)}</h2>' if web else f'<h1>{html.escape(value)}</h1>')
        elif kind=='p':markdown.append(value);web.append('<p>'+html.escape(value)+'</p>')
        elif kind=='img':
            name,caption=value;markdown.append(f'![{caption}](figures/{name})\n\n{caption}')
            web.append(f'<figure><img src="figures/{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        else:markdown.append(table(value));web.append(value.to_html(index=False,border=0))
    (OUT/'report.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
    style='body{font:17px/1.65 system-ui;color:#253b49;max-width:1080px;margin:50px auto;padding:0 25px}h1{font-size:38px;line-height:1.2}h2{margin-top:45px;font-size:26px}figure{margin:28px 0}img{width:100%}figcaption{font-size:14px;color:#586c76}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:10px;border-bottom:1px solid #d9e1e3;text-align:left}tr:nth-child(even){background:#f5f7f6}'
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Strategy economics</title><style>'+style+'</style></head><body>'+''.join(web)+'</body></html>',encoding='utf-8')


if __name__=='__main__':
    main()
