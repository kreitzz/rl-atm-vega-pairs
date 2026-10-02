"""Pair-level concentration, stability and entry characteristics; no policy changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.diagnose_current_contract_losses import read_csv

ROOT=Path('artifacts/contracts/expiry_vix_momentum/vix')
OUT=Path('reports/current_pair_selection')


def sharpe(nav):
    r=nav.nav.pct_change().iloc[1:]-nav.cash_benchmark.pct_change().iloc[1:]
    return np.sqrt(252)*r.mean()/r.std() if r.std()>1e-12 else np.nan


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    members=json.loads(Path('reports/expiry_vix_momentum/matched_pairs.json').read_text())
    definitions={d['pair']:d for d in json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())}
    rows=[];annual=[];entries=[];books={};daily={}
    for model in ['rl_sized','nn_fixed']:
        book=None
        for pair in members:
            d=definitions[pair];folder=ROOT/pair/model
            nav=pd.read_parquet(folder/'daily_nav.parquet')
            own=nav[['nav','cash_benchmark']];daily[(model,pair)]=own
            book=own.copy() if book is None else book+own
            pnl=nav.pnl-nav.cash_benchmark.diff().fillna(0)
            gross=nav.pnl-nav.interest+nav.option_execution_cost+nav.hedge_cost+nav.stock_borrow_cost
            trading=gross-nav.option_execution_cost-nav.hedge_cost-nav.stock_borrow_cost
            excess_returns=nav.nav.pct_change()-nav.cash_benchmark.pct_change()
            np.testing.assert_allclose(pnl.sum(),nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1],atol=1e-6)
            dec=read_csv(folder/'decisions.csv',['signal_date','entry_date'])
            dec=dec[dec.status.eq('entered')] if len(dec) else dec
            state=pd.read_parquet(ROOT/pair/'states.parquet')
            used=state.reindex(dec.signal_date).copy() if len(dec) else state.iloc[:0].copy()
            used['mean_iv']=(used.iv_amd+used.iv_nvda)/2
            used['absolute_spread']=used.spread.abs()
            used['beta_distance']=(used.beta-1).abs()
            used['dte_gap']=(used.dte_a-used.dte_b).abs()
            used['earnings_inside_30d']=((used.amd_earnings_days_capped<=30)|(used.nvda_earnings_days_capped<=30)).astype(float)
            used['short_higher_iv']=(used.spread.to_numpy()*dec.action.to_numpy()<0).astype(float) if len(dec) else pd.Series(dtype=float)
            features=['beta','beta_distance','correlation','mean_iv','absolute_spread','dte_gap','earnings_inside_30d','short_higher_iv','vix_previous_close']
            for i,(_,r) in enumerate(used.iterrows()):
                entries.append(dict(model=model,pair=pair,label=d['label'],date=r.name,action=dec.action.iloc[i],
                    **{f:r[f] for f in features}))
            worst_year=pnl.groupby(pnl.index.year).sum().idxmin()
            row=dict(model=model,pair=pair,label=d['label'],industry=d['industry'],entries=len(dec),
                gross_pnl=gross.sum(),option_cost=nav.option_execution_cost.sum(),hedge_cost=nav.hedge_cost.sum(),
                borrow_cost=nav.stock_borrow_cost.sum(),net_pnl=pnl.sum(),net_trading_before_financing=trading.sum(),sharpe=sharpe(nav),
                worst_day=pnl.idxmin(),worst_day_pnl=pnl.min(),pnl_excluding_worst_day=pnl.sum()-pnl.min(),
                worst_year=worst_year,pnl_excluding_worst_year=pnl[pnl.index.year!=worst_year].sum(),
                early_pnl=pnl[pnl.index.year<2022].sum(),later_pnl=pnl[pnl.index.year>=2022].sum(),
                early_entries=int((dec.signal_date.dt.year<2022).sum()) if len(dec) else 0,
                later_entries=int((dec.signal_date.dt.year>=2022).sum()) if len(dec) else 0,
                **{f:used[f].mean() for f in features})
            active_years=sorted(set(dec.entry_date.dt.year)) if len(dec) else []
            row['entry_years']=len(active_years)
            row['positive_entry_years']=sum(pnl[pnl.index.year==y].sum()>0 for y in active_years)
            rows.append(row)
            for year,g in nav.groupby(nav.index.year):
                er=excess_returns.loc[g.index].dropna()
                annual.append(dict(model=model,pair=pair,label=d['label'],year=year,net_pnl=pnl.loc[g.index].sum(),
                    net_trading_before_financing=trading.loc[g.index].sum(),
                    entries=int((dec.entry_date.dt.year==year).sum()) if len(dec) else 0,
                    sharpe=np.sqrt(252)*er.mean()/er.std() if er.std()>1e-12 else np.nan))
        books[model]=book
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'pairs.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame(entries).to_csv(OUT/'entries.csv',index=False)
    summary=[];exclusions=[];associations=[];groups=[]
    for model,allpairs in pairs.groupby('model'):
        active=allpairs[allpairs.entries>0].sort_values('net_pnl',ascending=False)
        winners=active[active.net_pnl>0];losers=active[active.net_pnl<0]
        both=active[(active.early_entries>0)&(active.later_entries>0)]
        summary.append(dict(model=model,active_pairs=len(active),entries=int(active.entries.sum()),
            winning_pairs=len(winners),losing_pairs=len(losers),winner_profit=winners.net_pnl.sum(),loser_loss=losers.net_pnl.sum(),
            top1_share_of_positive_profit=winners.head(1).net_pnl.sum()/winners.net_pnl.sum(),
            top3_share_of_positive_profit=winners.head(3).net_pnl.sum()/winners.net_pnl.sum(),
            gross_positive_net_negative=int(((active.gross_pnl>0)&(active.net_pnl<0)).sum()),
            losers_positive_without_worst_day=int(losers.pnl_excluding_worst_day.gt(0).sum()),
            losers_positive_without_worst_year=int(losers.pnl_excluding_worst_year.gt(0).sum()),
            pairs_traded_both_periods=len(both),early_later_rank_correlation=both.early_pnl.corr(both.later_pnl,method='spearman'),
            early_winners=int(both.early_pnl.gt(0).sum()),early_winners_later_positive=int(((both.early_pnl>0)&(both.later_pnl>0)).sum())))
        for n in [0,1,3,5]:
            book=books[model].copy()
            for pair in active.head(n).pair:
                removed=daily[(model,pair)];book['nav']-=removed.nav-removed.cash_benchmark
            exclusions.append(dict(model=model,top_pairs_replaced_with_cash=n,sharpe=sharpe(book),
                net_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1],removed_labels=';'.join(active.head(n).label)))
        for feature in ['beta_distance','correlation','mean_iv','absolute_spread','dte_gap','earnings_inside_30d','short_higher_iv','vix_previous_close']:
            associations.append(dict(model=model,feature=feature,rank_correlation_with_sharpe=active[feature].corr(active.sharpe,method='spearman'),
                winner_median=winners[feature].median(),loser_median=losers[feature].median()))
        for industry,g in active.groupby('industry'):
            groups.append(dict(model=model,industry=industry,pairs=len(g),entries=int(g.entries.sum()),net_pnl=g.net_pnl.sum(),labels=';'.join(g.label)))
    for name,data in [('summary',summary),('exclusions',exclusions),('associations',associations),('industries',groups)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    agreement=pairs[pairs.entries>0].pivot(index='pair',columns='model',values='net_pnl').dropna()
    print('SUMMARY',pd.DataFrame(summary).to_string(index=False))
    print('EXCLUSIONS',pd.DataFrame(exclusions).to_string(index=False))
    print('ASSOCIATIONS',pd.DataFrame(associations).to_string(index=False))
    print('MODEL AGREEMENT',len(agreement),agreement.corr(method='spearman').iloc[0,1])
    for model in ['rl_sized','nn_fixed']:
        a=pairs[(pairs.model==model)&(pairs.entries>0)].sort_values('net_pnl',ascending=False)
        print(model,a[['label','entries','net_pnl','gross_pnl','sharpe','early_pnl','later_pnl','entry_years','positive_entry_years']].to_string(index=False))


if __name__=='__main__':main()
