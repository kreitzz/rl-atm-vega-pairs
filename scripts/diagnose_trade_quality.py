"""Relative-value behavior and concentration on the saved common portfolio calendar."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_broad_memory import ROOT,OUT
from scripts.run_first_pass import payoff_sharpe
from scripts.run_walk_forward import fold_data
from scripts.report_nn_rl_benchmark import table

MODELS=['rl_no_memory','rl_memory']
REPORT=Path('reports/trade_quality')


def concentration(values,counts=(1,5,10,20)):
    ranked=values.sort_values(ascending=False);positive=ranked.clip(lower=0).sum();net=ranked.sum()
    return [dict(top=k,contribution=ranked.head(k).sum(),fraction_of_net=ranked.head(k).sum()/net,
                 fraction_of_positive_pool=ranked.head(k).clip(lower=0).sum()/positive) for k in counts]


def load_trades():
    common=pd.read_csv(OUT/'all_portfolio.csv',index_col=0,parse_dates=[0]).dropna().index
    screens=pd.read_csv('reports/broad_pairs/candidate_screens.csv');screens=screens[screens.year.ge(2018)]
    assets=pd.read_csv('reports/broad_pairs/assets.csv').set_index('asset')
    chunks=[]
    for year,roster in screens.groupby('year'):
        dates=common[(common.year>=year)&(common.year<year+2)]
        for row in roster.itertuples():
            path=OUT/row.pair/f'{year}.csv'
            if not path.exists():continue
            pred=pd.read_csv(path,index_col=0,parse_dates=[0]);p=pd.read_parquet(OUT/row.pair/'panel.parquet')
            index=dates.intersection(pred.index);q=p.loc[index]
            tr,_,_=fold_data(p,year);tr=tr.dropna(subset=['long_reward'])
            costs=.025*(tr.iv_amd+tr.iv_nvda)/2
            constant=max([-1,1],key=lambda a:payoff_sharpe(a*tr.long_reward-costs))
            for model in MODELS:
                a=pred.loc[index,model+'_action'];direction=np.sign(a);weight=a.abs()/len(roster)
                d=pd.DataFrame(index=index);d['model']=model;d['pair']=row.pair;d['label']=row.label
                d['division']=str(row.division);d['industry']=str(row.industry);d['fold']=year
                d['asset_a']=row.asset_a;d['asset_b']=row.asset_b
                d['permno_a']=int(assets.loc[row.asset_a,'permno']);d['permno_b']=int(assets.loc[row.asset_b,'permno'])
                d['ticker_a']=row.label.split('/')[0];d['ticker_b']=row.label.split('/')[1]
                d['action']=a;d['exposure']=weight;d['entry_cost']=.025*(q.iv_amd+q.iv_nvda)/2
                d['iv_carry']=(-a*q.spread/len(roster)).fillna(0.)
                d['realized_difference']=(a*(q.rv_amd-q.rv_nvda)/len(roster)).fillna(0.)
                d['cost']=weight*d.entry_cost;d['net']=pred.loc[index,model+'_reward']/len(roster)
                d['entry_value']=direction*q.rv_adjusted_value;d['reversion_agrees']=direction*q.spread_reversion>0
                d['short_higher_iv']=direction*q.spread<0
                d['extreme']=q[['iv_amd','iv_nvda']].max(axis=1)>100
                d['long_iv']=np.where(a>0,q.iv_amd,q.iv_nvda);d['short_iv']=np.where(a>0,q.iv_nvda,q.iv_amd)
                d['long_rv']=np.where(a>0,q.rv_amd,q.rv_nvda);d['short_rv']=np.where(a>0,q.rv_nvda,q.rv_amd)
                d['long_net']=(weight*(d.long_rv-d.long_iv-.0125*d.long_iv)).fillna(0.)
                d['short_net']=(weight*(d.short_iv-d.short_rv-.0125*d.short_iv)).fillna(0.)
                d['a_net']=(a*(q.rv_amd-q.iv_amd)/len(roster)-weight*.0125*q.iv_amd).fillna(0.)
                d['b_net']=(-a*(q.rv_nvda-q.iv_nvda)/len(roster)-weight*.0125*q.iv_nvda).fillna(0.)
                for name,sign in [('higher_iv_rule',-np.sign(q.spread)),('rv_value_rule',np.sign(q.rv_adjusted_value)),
                                  ('spread_rule',np.sign(q.spread_reversion)),('train_constant',constant),('reverse',-direction)]:
                    action=a.abs()*sign
                    d[name]=(action*q.long_reward/len(roster)-action.abs()*d.entry_cost/len(roster)).where(action.ne(0),0.)
                np.testing.assert_allclose(d.iv_carry+d.realized_difference-d.cost,d.net,atol=1e-9)
                np.testing.assert_allclose(d.long_net+d.short_net,d.net,atol=1e-9)
                np.testing.assert_allclose(d.a_net+d.b_net,d.net,atol=1e-9)
                chunks.append(d.rename_axis('date').reset_index())
    return pd.concat(chunks,ignore_index=True),common


def main():
    REPORT.mkdir(exist_ok=True);trades,common=load_trades()
    saved=pd.read_csv(OUT/'all_portfolio.csv',index_col=0,parse_dates=[0]).loc[common]
    summary=[];pairs=[];names=[];sectors=[];concentrations=[];removals=[];controls=[];years=[];buckets=[]
    for model,allrows in trades.groupby('model'):
        baseline=allrows.groupby('date').net.sum().reindex(common,fill_value=0.)
        np.testing.assert_allclose(baseline,saved[model],atol=1e-9)
        for sample,mask in [('all',pd.Series(True,index=allrows.index)),('iv_at_most_100',~allrows.extreme)]:
            rows=allrows[mask];active=rows[rows.action.ne(0)];w=active.exposure
            curve=rows.groupby('date').net.sum().reindex(common,fill_value=0.)
            summary.append(dict(model=model,sample=sample,trades=len(active),net_pnl=curve.sum(),sharpe=payoff_sharpe(curve),
                net_win_fraction=active.net.gt(0).mean(),vega_weighted_win_fraction=np.average(active.net.gt(0),weights=w),
                short_higher_iv_fraction=np.average(active.short_higher_iv,weights=w),rv_value_agrees_fraction=np.average(active.entry_value.gt(0),weights=w),
                reversion_agrees_fraction=np.average(active.reversion_agrees,weights=w),entry_value_per_vega=np.average(active.entry_value,weights=w),
                net_per_vega=active.net.sum()/w.sum(),iv_carry=active.iv_carry.sum(),realized_difference=active.realized_difference.sum(),cost=active.cost.sum(),
                long_net=active.long_net.sum(),short_net=active.short_net.sum(),mean_long_iv=np.average(active.long_iv,weights=w),mean_short_iv=np.average(active.short_iv,weights=w),
                mean_long_rv=np.average(active.long_rv,weights=w),mean_short_rv=np.average(active.short_rv,weights=w)))
            for name in ['net','higher_iv_rule','rv_value_rule','spread_rule','train_constant','reverse']:
                r=rows.groupby('date')[name].sum().reindex(common,fill_value=0.)
                controls.append(dict(model=model,sample=sample,control=name,sharpe=payoff_sharpe(r),net_pnl=r.sum()))
        active=allrows[allrows.action.ne(0)]
        for label,mask in [('value_agrees',active.entry_value>0),('value_disagrees',active.entry_value<=0),
                           ('both_value_and_reversion',active.entry_value.gt(0)&active.reversion_agrees),('iv_over_100',active.extreme),('iv_at_most_100',~active.extreme)]:
            g=active[mask];buckets.append(dict(model=model,bucket=label,trades=len(g),exposure_fraction=g.exposure.sum()/active.exposure.sum(),net_pnl=g.net.sum(),net_per_vega=g.net.sum()/g.exposure.sum(),win_fraction=g.net.gt(0).mean()))
        grouped=allrows.groupby('pair')
        pair_net=grouped.net.sum()
        for key,g in grouped:
            p=g[g.action.ne(0)];size=p.action.abs().sum();long_size=p.loc[p.action.gt(0),'action'].sum()
            dominant=max(long_size,size-long_size)/size if size else np.nan
            pairs.append(dict(model=model,pair=key,label=g.label.iloc[-1],net_pnl=g.net.sum(),share_of_net=g.net.sum()/baseline.sum(),active_entries=len(p),dominant_direction_share=dominant,extreme_net_pnl=g.loc[g.extreme,'net'].sum()))
        for dimension,values in [('pair',pair_net),('sector',allrows.groupby('division').net.sum()),('entry_date',baseline)]:
            concentrations.extend(dict(model=model,dimension=dimension,**x) for x in concentration(values))
        for k in [1,5,10,20]:
            excluded=set(pair_net.nlargest(k).index)
            curve=allrows.loc[~allrows.pair.isin(excluded)].groupby('date').net.sum().reindex(common,fill_value=0.)
            removals.append(dict(model=model,exclusion=f'top_{k}_pairs',sharpe=payoff_sharpe(curve),net_pnl=curve.sum()))
        legs=[]
        for leg in ['a','b']:
            d=allrows[['date',f'permno_{leg}',f'ticker_{leg}',f'{leg}_net']].rename(columns={f'permno_{leg}':'permno',f'ticker_{leg}':'ticker',f'{leg}_net':'net'})
            legs.append(d)
        legs=pd.concat(legs);name_net=legs.groupby('permno').net.sum()
        np.testing.assert_allclose(name_net.sum(),baseline.sum(),atol=1e-8)
        for key,g in legs.groupby('permno'):names.append(dict(model=model,permno=key,ticker=g.ticker.iloc[-1],net_pnl=g.net.sum(),share_of_net=g.net.sum()/baseline.sum()))
        concentrations.extend(dict(model=model,dimension='underlying_leg',**x) for x in concentration(name_net))
        for k in [1,3,5]:
            excluded=set(name_net.nlargest(k).index);keep=~allrows.permno_a.isin(excluded)&~allrows.permno_b.isin(excluded)
            curve=allrows.loc[keep].groupby('date').net.sum().reindex(common,fill_value=0.)
            removals.append(dict(model=model,exclusion=f'pairs_involving_top_{k}_profitable_names',sharpe=payoff_sharpe(curve),net_pnl=curve.sum()))
        for sector,g in allrows.groupby('division'):sectors.append(dict(model=model,division=sector,net_pnl=g.net.sum(),share_of_net=g.net.sum()/baseline.sum(),pairs=g.pair.nunique()))
        for k in [1,3]:
            excluded=set(allrows.groupby('division').net.sum().nlargest(k).index)
            curve=allrows.loc[~allrows.division.isin(excluded)].groupby('date').net.sum().reindex(common,fill_value=0.)
            removals.append(dict(model=model,exclusion=f'top_{k}_sectors',sharpe=payoff_sharpe(curve),net_pnl=curve.sum()))
        for year,r in baseline.groupby(baseline.index.year):years.append(dict(model=model,year=year,net_pnl=r.sum(),share_of_net=r.sum()/baseline.sum(),sharpe=payoff_sharpe(r)))
        for fraction in [.01,.05]:
            top=active.net.nlargest(max(1,int(np.ceil(len(active)*fraction))))
            concentrations.append(dict(model=model,dimension='individual_entry_cohort',top=fraction,contribution=top.sum(),fraction_of_net=top.sum()/baseline.sum(),fraction_of_positive_pool=top.sum()/active.net.clip(lower=0).sum()))
    for name,data in [('summary',summary),('pairs',pairs),('names',names),('sectors',sectors),('concentration',concentrations),('removals',removals),('direction_controls',controls),('annual',years),('value_buckets',buckets)]:pd.DataFrame(data).to_csv(REPORT/f'{name}.csv',index=False)
    s=pd.DataFrame(summary);p=pd.DataFrame(pairs);n=pd.DataFrame(names);c=pd.DataFrame(concentrations)
    r=pd.DataFrame(removals);y=pd.DataFrame(years);ctrl=pd.DataFrame(controls)
    best=s[s.model.eq('rl_no_memory')&s['sample'].eq('all')].iloc[0]
    pair_table=p[p.model.eq('rl_no_memory')].nlargest(10,'net_pnl')[['label','share_of_net','dominant_direction_share','net_pnl','extreme_net_pnl']]
    name_table=n[n.model.eq('rl_no_memory')].nlargest(10,'net_pnl')[['ticker','share_of_net','net_pnl']]
    control_table=ctrl.pivot(index=['model','sample'],columns='control',values='sharpe').reset_index()
    metadata=p[p.model.eq('rl_no_memory')]
    text=['# Relative-value capture and profit concentration','',
          'The trades capture a relative volatility premium within the simulation, but the evidence is stronger for selling the richer leg and paying for a hedge than for sophisticated convergence timing. The larger universe is not dependent on one winning pair; dependence on a handful of underlying names, recent years, and extreme standardized IV marks is more substantial.','',
          'This audit uses the existing 1,805 jointly complete entry dates in 2018–August 2025, identical to the broad-universe report. It retains the original per-fold pair budgets and idle allocations. The primary policy is the stronger sized RL model without the explicit profit-history state; the memory version is included for comparison. No policy was retrained.','',
          '## What the trades buy and sell','',
          f"For the primary policy, {best.short_higher_iv_fraction:.1%} of deployed entry vega shorts the higher-IV leg, {best.rv_value_agrees_fraction:.1%} agrees with the trailing-RV-adjusted value signal, and {best.reversion_agrees_fraction:.1%} agrees with reversion toward the historical spread mean. The trade-level net win rate is {best.net_win_fraction:.1%}. These fractions are not probabilities of future success.",'',
          '| Vega-weighted leg | Entry IV | Subsequent 30-day RV | Net contribution |',
          '| --- | ---: | ---: | ---: |',
          f'| Long | {best.mean_long_iv:.2f} | {best.mean_long_rv:.2f} | {best.long_net:.2f} |',
          f'| Short | {best.mean_short_iv:.2f} | {best.mean_short_rv:.2f} | {best.short_net:.2f} |','',
          f"The IV gap averages {best.mean_short_iv-best.mean_long_iv:.2f} vol points, while the subsequent RV gap averages {best.mean_short_rv-best.mean_long_rv:.2f}. Their difference leaves {best.net_per_vega:.2f} net proxy vol points per unit deployed entry vega after costs. The long side is relatively cheaper, but not absolutely cheap against subsequent RV on average. The short side earns {best.short_net/best.net_pnl:.1%} of net profit, with the long side subtracting the balance.",'',
          'The decomposition is algebraic, not causal attribution: `net = size × [(short IV − long IV) + (long forward RV − short forward RV)] − costs`. Costs are split by each leg\'s IV so the two legs reconcile exactly to the pair. Net contribution units are weighted proxy P&L per unit total maximum cohort budget, not a capital return percentage.','',
          'Trades agreeing with the RV-adjusted value signal earn 2,225.95 net proxy units; trades opposing it lose 125.90. The former account for 61.1% of exposure. This is a descriptive split on an existing input, not an independently tested exclusion rule.','',
          '## How much does the learned direction add?','',
          table(control_table,['Model','Sample']+list(control_table.columns[2:])),'',
          'These controls reuse each policy\'s selected entry dates and absolute sizes, replacing only direction. `higher_iv_rule` shorts the higher-IV leg; `rv_value_rule` follows the trailing-RV-adjusted value; `spread_rule` follows historical spread reversion. A zero control signal stays flat. `train_constant` chooses one fixed pair direction using only the corresponding historical training sample; `reverse` flips every actual trade. These are not standalone full-size strategies on every possible date.','',
          'For the primary policy, the higher-IV rule slightly exceeds learned-direction Sharpe (2.885 versus 2.788) on the same exposure schedule. With entries above 100 vol suppressed, learned direction does better (1.853 versus 1.704). A training-selected fixed pair direction is much weaker (0.522 on the full sample). Thus changing which leg is rich matters, but learned direction is not clearly superior to a simple IV comparison across the full book. The controls do not isolate the incremental benefit of entry timing or sizing.','',
          '## Concentration','',
          table(c[c.dimension.isin(['pair','sector','underlying_leg'])],['Model','Aggregation','Top N','Net contribution','Share of total net','Share of positive-contribution pool']),'',
          f"On the shared portfolio calendar, {int(metadata.net_pnl.gt(0).sum())} pairs contribute positively, {int(metadata.net_pnl.lt(0).sum())} negatively, and {int(metadata.net_pnl.eq(0).sum())} are flat. One of the 308 fitted pairs has no rows on this calendar, leaving 307 represented here. Pair histories vary in length. Shares of total net can exceed 100% because losses elsewhere offset positive contributions; the separate positive-pool denominator avoids that interpretation problem.",'',
          'Top pair contributions, primary policy:', '',
          table(pair_table,['Pair','Share of total net','Dominant-direction exposure share','Net contribution','Contribution when IV >100']),'',
          'Top underlying-leg contributions, primary policy:', '',
          table(name_table,['Underlying label','Share of total net','Net contribution']),'',
          'Names are grouped by permanent CRSP identifier; displayed ticker aliases can be historical. An underlying\'s contribution aggregates its long and short legs across every pair and allocates its own execution cost. It is not the profit of an unhedged standalone strategy. For removal tests, all pairs involving the named underlying are removed together so the retained trades remain paired.','',
          'The top five name contributions (KGC, INFN, HL, RIG, SFL) explain 41.5% of net profit, versus 22.1% for the top five pairs. The top three historical pair-industry groups are metal mining (18.6%), oil/gas extraction (16.0%), and water transportation (11.6%). ETF comparisons remain in a separate group. Industry labels use [OSHA\'s SIC manual](https://www.osha.gov/ords/imis/sic_manual.display?id=30&tab=group); the underlying historical codes come from the local security-name records.','',
          '## Does the remainder still earn money?','',
          table(r,['Model','Ex-post exclusion','Remaining Sharpe','Remaining net P&L']),'',
          'The primary policy retains Sharpe 2.429 without its ten best pairs, 2.057 without its best twenty, and 2.278 after removing every pair involving its five most profitable underlying legs. Rankings use the completed test history, so these are concentration diagnostics, not implementable selection rules. Budgets freed by exclusions remain idle. These separate removal tests still contain extreme-IV trades; they are not jointly cleaned results.','',
          '## Time and extreme-mark concentration','',
          table(y,['Model','Entry year','Net contribution','Share of total net','Sharpe']),'',
          'For the primary policy, 78.5% of net profit comes from 2022–August 2025 and 48.6% from 2024–August 2025. The twenty best portfolio entry dates explain only 5.6% of total net profit. However, the best 1% of individual active pair-entry cohorts explain 43.5%, and the best 5% explain 104.9% after losses elsewhere. Thirty-day positions overlap: these cohorts are not independent trades or market episodes.','',
          'Entries with either IV above 100 vol use just 5.0% of deployed entry vega but contribute 66.5% of net profit. Excluding them leaves 1.853 Sharpe for the primary policy and 1.520 for the memory policy. This is a sensitivity check, not a newly validated filter.','',
          'A concrete concern is INFN/COMM, the largest pair contributor. On January 27, 2025 the cache shows INFN IV of 254.69 vol and subsequent RV of 4.17; the policy is short INFN and long COMM. Infinera was already subject to Nokia\'s announced acquisition, which closed on February 28, 2025. [Nokia transaction announcement](https://www.nokia.com/newsroom/nokia-to-acquire-infinera-to-increase-scale-in-optical-networks-and-accelerate-product-roadmap/) · [Infinera closing filing](https://www.sec.gov/Archives/edgar/data/1138639/000119312525041368/d847449d8k.htm). This is a reason to inspect actual contracts and merger-event risk, not proof that the surface mark is erroneous or the trade executable.','',
          'The interpretation is conditional on the standardized-IV/forward-RV payoff proxy, assumed execution costs, and already inspected history. The 121 entry dates with unresolved active-trade outcomes remain excluded exactly as in the preceding report; no missing active payoff is counted as zero.','',
          '[Summary and leg decomposition](trade_quality/summary.csv) · [Direction controls](trade_quality/direction_controls.csv) · [Pair contributions](trade_quality/pairs.csv) · [Underlying contributions](trade_quality/names.csv) · [Industry contributions](trade_quality/sectors.csv) · [Concentration](trade_quality/concentration.csv) · [Removal sensitivities](trade_quality/removals.csv) · [Value buckets](trade_quality/value_buckets.csv)','',
          '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.diagnose_trade_quality','```','',
          'The script reconciles leg contributions and the IV/RV decomposition to each stored pair payoff. It publishes aggregate diagnostics only; the shared market-data CSV is unchanged.']
    Path('reports/trade_quality.md').write_text('\n'.join(text)+'\n')
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(pd.DataFrame(controls).round(3).to_string(index=False))
    print(pd.DataFrame(concentrations).query('dimension in ["pair","sector","underlying_leg"]').round(3).to_string(index=False))
    print(pd.DataFrame(removals).round(3).to_string(index=False))


if __name__=='__main__':main()
