"""Descriptive winner/loser attribution and entry-observable state diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_first_pass import payoff_sharpe
from scripts.explore_universe_gap import block_difference


def matured_history(panel, window=252, minimum=126):
    """Rolling moments of outcomes strictly completed before the entry date."""
    ordered = panel.sort_values('end')
    moments = ordered.long_reward.rolling(window, min_periods=minimum).agg(['mean', 'std'])
    positions = pd.DatetimeIndex(ordered.end).searchsorted(panel.index, side='left') - 1
    result = moments.iloc[np.maximum(positions, 0)].copy()
    result.index = panel.index
    result.loc[positions < 0] = np.nan
    return result


def entry_features(panel):
    memory = matured_history(panel)
    spread_history = panel.spread.shift(1).rolling(252, min_periods=126)
    cost = .025 * (panel.iv_amd + panel.iv_nvda) / 2
    # Positive values of directional signals predict long first / short second.
    f = pd.DataFrame(index=panel.index)
    f['matured_edge'] = memory['mean']
    f['edge_to_cost'] = memory['mean'] / cost
    f['edge_to_risk'] = memory['mean'] / memory['std'].replace(0, np.nan)
    f['rv_adjusted_value'] = (panel.trailing_rv_amd-panel.iv_amd) - (panel.trailing_rv_nvda-panel.iv_nvda)
    f['spread_reversion'] = (spread_history.mean()-panel.spread) / spread_history.std().replace(0, np.nan)
    f['cost_to_risk'] = cost / memory['std'].replace(0, np.nan)
    f['max_iv_to_rv'] = pd.concat([panel.iv_amd/panel.trailing_rv_amd, panel.iv_nvda/panel.trailing_rv_nvda], axis=1).max(axis=1)
    f['max_iv'] = panel[['iv_amd', 'iv_nvda']].max(axis=1)
    f['correlation_gap'] = panel.correlation_gap
    f['earnings_mismatch'] = panel.amd_earnings_days_capped.le(30).astype(int)-panel.nvda_earnings_days_capped.le(30).astype(int)
    return f.replace([np.inf, -np.inf], np.nan)


def main():
    output = Path('reports/winner_loser'); output.mkdir(exist_ok=True)
    summary, annual, signals, filters = [], [], [], []
    gate_returns = {}
    for group, root in [('stocks', Path('artifacts/universe_gap')), ('etfs', Path('artifacts/etf_exploration/run'))]:
        packed = pd.read_parquet(root/'panels.parquet')
        for pair in packed.index.get_level_values('pair').unique():
            panel = packed.xs(pair).sort_index()
            pred = pd.read_csv(root/f'{pair}.csv', index_col='date', parse_dates=['date'])
            q = panel.loc[pred.index]; f = entry_features(panel).loc[pred.index]
            a = pred.baseline_action; net = pred.baseline_reward
            cost = .025*(q.iv_amd+q.iv_nvda)/2
            gross = a*q.long_reward; paid = a.abs()*cost
            np.testing.assert_allclose(gross-paid, net, atol=1e-9)
            years = net.groupby(net.index.year).sum()
            denom = a.abs().sum()
            constant_long = q.long_reward-cost; constant_short = -q.long_reward-cost
            high = f.max_iv.gt(100)
            summary.append(dict(pair=pair,group=group,first=str(q.index.min().date()),entries=len(q),sharpe=payoff_sharpe(net),gross_sharpe=payoff_sharpe(gross),gross_per_size=gross.sum()/denom if denom else np.nan,cost_per_size=paid.sum()/denom if denom else np.nan,net_pnl=net.sum(),positive_years=int((years>0).sum()),years=len(years),worst_year=int(years.idxmin()),sharpe_without_worst=payoff_sharpe(net[net.index.year!=years.idxmin()]),always_long=payoff_sharpe(constant_long),always_short=payoff_sharpe(constant_short),short_higher_iv=payoff_sharpe(-np.sign(q.spread)*q.long_reward-cost),long_share_of_active=a.gt(0).sum()/a.ne(0).sum() if a.ne(0).any() else np.nan,extreme_iv_fraction=high.mean(),extreme_iv_net=net[high].sum(),sharpe_skip_extreme=payoff_sharpe(net.where(~high,0)),return_corr=q.stock_correlation.median(),iv_corr=q.iv_correlation.median()))
            for year in years.index:
                m=net.index.year==year
                annual.append(dict(pair=pair,year=year,net_pnl=net[m].sum(),gross_pnl=gross[m].sum(),cost=paid[m].sum(),sharpe=payoff_sharpe(net[m])))
            for feature in ['matured_edge','rv_adjusted_value','spread_reversion','correlation_gap','earnings_mismatch']:
                for period, mask in [('all',np.ones(len(q),dtype=bool)),('through_2017',q.index.year<2018),('2018_2021',(q.index.year>=2018)&(q.index.year<=2021)),('2022_on',q.index.year>=2022)]:
                    d=pd.concat([f[feature],q.long_reward,cost.rename('cost')],axis=1).loc[mask].dropna()
                    if len(d)<126:continue
                    rho=d[feature].corr(d.long_reward,method='spearman') if d[feature].nunique()>1 else np.nan
                    r=np.sign(d[feature])*d.long_reward-np.sign(d[feature]).abs()*d.cost
                    signals.append(dict(pair=pair,feature=feature,period=period,entries=len(d),rank_correlation=rho,sign_control_sharpe=payoff_sharpe(r)))
            # Fixed diagnostic gates; retain zero on skipped dates. No threshold search.
            ready=f[['matured_edge','edge_to_cost','edge_to_risk','rv_adjusted_value','spread_reversion']].notna().all(axis=1)
            gates={'unfiltered':pd.Series(True,index=q.index),
                   'matured_edge_agrees':a*f.matured_edge>0,
                   'matured_edge_covers_cost':np.sign(a)*f.matured_edge>cost,
                   'rv_value_agrees':a*f.rv_adjusted_value>0,
                   'spread_reversion_agrees':a*f.spread_reversion>0,
                   'both_or_neither_earnings':f.earnings_mismatch.eq(0)}
            gate_returns[pair] = pd.DataFrame({name:net.where(gate,0).loc[ready] for name,gate in gates.items()})
            for name,gate in gates.items():
                for period,mask in [('all',ready),('2018_2021',ready&(q.index.year>=2018)&(q.index.year<=2021)),('2022_on',ready&(q.index.year>=2022))]:
                    if mask.sum()<126:continue
                    r=net.where(gate,0).loc[mask]
                    filters.append(dict(pair=pair,filter=name,period=period,entries=len(r),sharpe=payoff_sharpe(r),net_pnl=r.sum(),retained_active=(gate&mask&a.ne(0)).sum()/(mask&a.ne(0)).sum() if (mask&a.ne(0)).any() else np.nan))
    for name,rows in [('summary',summary),('annual',annual),('signals',signals),('filters',filters)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    portfolios=[]
    definitions={'stocks':[x['pair'] for x in summary if x['group']=='stocks'],
                 'etf_stock':[x['pair'] for x in summary if x['group']=='etfs' and x['pair'] not in ['OIH_XOP','ARKK_TSLA','GSLC_SPYM']]}
    definitions['combined']=definitions['stocks']+definitions['etf_stock']
    definitions['combined_without_silver']=[p for p in definitions['combined'] if not p.startswith('SIL_')]
    for group,pairs in definitions.items():
        common=gate_returns[pairs[0]].index
        for pair in pairs[1:]:common=common.intersection(gate_returns[pair].index)
        portfolio=sum(gate_returns[pair].loc[common] for pair in pairs)/len(pairs)
        for period,mask in [('all',np.ones(len(common),dtype=bool)),('2018_2021',(common.year>=2018)&(common.year<=2021)),('2022_on',common.year>=2022)]:
            p=portfolio.loc[mask]
            for name in p:
                ci=block_difference(p[name].to_numpy(),p.unfiltered.to_numpy(),repeats=1000) if period=='all' and name!='unfiltered' else [np.nan,np.nan]
                portfolios.append(dict(group=group,filter=name,period=period,first=str(p.index.min().date()),last=str(p.index.max().date()),entries=len(p),pairs=len(pairs),sharpe=payoff_sharpe(p[name]),change=payoff_sharpe(p[name])-payoff_sharpe(p.unfiltered),change_ci_low=ci[0],change_ci_high=ci[1]))
    pd.DataFrame(portfolios).to_csv(output/'portfolios.csv',index=False)
    s=pd.DataFrame(summary)
    print(s[['pair','sharpe','gross_per_size','cost_per_size','positive_years','years','always_long','always_short','short_higher_iv']].round(3).to_string(index=False))
    z=pd.DataFrame(filters).query('period == "all"').pivot(index='pair',columns='filter',values='sharpe')
    print('\nFixed diagnostic gates, changes versus matched unfiltered baseline:')
    print(z.sub(z.unfiltered,axis=0).round(3).to_string())


if __name__=='__main__': main()
