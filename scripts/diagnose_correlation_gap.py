"""Causal stock-return versus IV-change dependence diagnostics, without policy retuning."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_four_pairs import PAIRS, IDS
from scripts.run_first_pass import payoff_sharpe
from scripts.run_walk_forward import fold_data


def rolling_dependence(a,b,window=252):
    correlation=a.rolling(window,min_periods=window).corr(b)
    beta=a.rolling(window,min_periods=window).cov(b)/b.rolling(window,min_periods=window).var()
    return pd.DataFrame({'correlation':correlation,'beta':beta})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--run',type=Path,default=Path('artifacts/four_pairs_independent_cost_0.025'))
    ap.add_argument('--output',type=Path,default=Path('reports/correlation_gap'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ids={k:v[1] for k,v in IDS.items()};ids.update(AMD=61241,NVDA=86580)
    frames=[]
    for path in sorted((args.source_root/'crsp_underlying_daily').glob('*.parquet')):
        d=pd.read_parquet(path,columns=['permno','date','retx']);frames.append(d[d.permno.isin(ids.values())])
    r=pd.concat([x for x in frames if len(x)]).drop_duplicates();r['ticker']=r.permno.map({v:k for k,v in ids.items()})
    r=r.pivot(index='date',columns='ticker',values='retx').sort_index().astype(float)
    pred=pd.read_csv(args.run/'predictions.csv',index_col='date',parse_dates=['date','end'])
    packed=pd.read_parquet(args.run/'panels.parquet');summaries=[];bins=[];annual=[];associations=[];features=[]
    for pair in PAIRS:
        a,b=pair.split('_');p=packed.xs(pair,level='pair').sort_index()
        # Surface change is aligned with the same close-to-close stock-return interval.
        stock=r[[a,b]].reindex(p.index)
        delta=p[['iv_amd','iv_nvda']].diff()
        sf=rolling_dependence(stock[a],stock[b]).add_prefix('stock_')
        vf=rolling_dependence(delta.iv_amd,delta.iv_nvda).add_prefix('iv_')
        d=p.join(sf).join(vf)
        d['correlation_gap']=d.stock_correlation-d.iv_correlation
        d['beta_gap']=d.stock_beta-d.iv_beta
        # Raw beta is asymmetric; a positive gap is not a valuation or directional signal.
        features.append(d[['stock_correlation','stock_beta','iv_correlation','iv_beta','correlation_gap','beta_gap']].assign(pair=pair))
        test=d.loc[pred.index];s=stock.loc[pred.index];v=delta.loc[pred.index]
        summaries.append(dict(pair=pair,stock_return_correlation=s[a].corr(s[b]),stock_beta_first_on_second=s[a].cov(s[b])/s[b].var(),iv_change_correlation=v.iv_amd.corr(v.iv_nvda),iv_beta_first_on_second=v.iv_amd.cov(v.iv_nvda)/v.iv_nvda.var(),median_trailing_stock_correlation=test.stock_correlation.median(),median_trailing_iv_correlation=test.iv_correlation.median(),median_trailing_correlation_gap=test.correlation_gap.median(),median_trailing_stock_beta=test.stock_beta.median(),median_trailing_iv_beta=test.iv_beta.median()))
        for name in ['correlation_gap','beta_gap']:
            for target,x in [('signed_gross_pair_payoff',test.long_reward),('absolute_gross_pair_payoff',test.long_reward.abs()),('learned_policy_net_payoff',pred[pair+'_solo_reward'])]:
                associations.append(dict(pair=pair,feature=name,target=target,spearman=test[name].corr(x,method='spearman')))
            tags=pd.Series(index=pred.index,dtype='object')
            for year in range(2010,2026,2):
                train,_,te=fold_data(d,year)
                threshold=train[name].dropna().median()
                tags.loc[te.index]=np.where(te[name]>threshold,'high','low')
            for label in ['low','high']:
                mask=tags.eq(label);gross=test.long_reward.loc[mask];net=pred.loc[mask,pair+'_solo_reward']
                bins.append(dict(pair=pair,feature=name,group=label,entries=int(mask.sum()),mean_gap=test.loc[mask,name].mean(),mean_signed_gross_payoff=gross.mean(),mean_absolute_gross_payoff=gross.abs().mean(),mean_policy_net_payoff=net.mean(),conditional_policy_sharpe=payoff_sharpe(net)))
                for year in sorted(pred.index.year.unique()):
                    m=mask&(pred.index.year==year)
                    annual.append(dict(pair=pair,feature=name,group=label,year=int(year),entries=int(m.sum()),mean_policy_net_payoff=pred.loc[m,pair+'_solo_reward'].mean(),mean_absolute_gross_payoff=test.loc[m,'long_reward'].abs().mean()))
        print(pair,'checked',flush=True)
    for name,rows in [('summary',summaries),('bins',bins),('associations',associations),('annual_bins',annual)]:pd.DataFrame(rows).to_csv(args.output/f'{name}.csv',index=False)
    pd.concat(features).to_parquet(args.run/'dependence_features.parquet')
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(bins).to_string(index=False));print(pd.DataFrame(associations).to_string(index=False))


if __name__=='__main__':main()
