"""Prespecified wider-universe test of independent dependence-gap policies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_first_pass import forward_rv, net_rewards, payoff_sharpe
from scripts.run_rv_vix_pass import trailing_rv, fit_tree_policy, leaf_states
from scripts.run_walk_forward import fold_data
from scripts.run_four_pairs import FEATURES, LADDER, entry_cost
from scripts.diagnose_correlation_gap import rolling_dependence

VARIANTS={'baseline':FEATURES,'correlation_gap':FEATURES+['correlation_gap'],'beta_gap':FEATURES+['beta_gap']}
EARNINGS_OVERRIDES=Path('config/universe_earnings_overrides.csv')


def correct_earnings(earnings):
    result=earnings.copy()
    overrides=pd.read_csv(EARNINGS_OVERRIDES,parse_dates=['datadate','earnings_date'])
    for row in overrides.itertuples():
        mask=result.permno.eq(row.permno)&pd.to_datetime(result.datadate).eq(row.datadate)
        if not mask.any():raise ValueError(f'Missing earnings record for override: {row.ticker} {row.datadate}')
        result.loc[mask,'rdq']=row.earnings_date
    return result


def build(root, universe, output):
    mapping={t:v for pair in universe for t,v in pair['names'].items()}
    files=sorted((root/'optionmetrics/atm_iv_compact').rglob('*.parquet'))+sorted((root/'optionmetrics/atm_implied_volatility').glob('*.parquet'))
    frames=[pd.read_parquet(f,columns=['secid','date','impl_volatility'],filters=[('secid','in',[v['secid'] for v in mapping.values()]),('days','=',30),('delta','=',50),('cp_flag','=','C')]) for f in files]
    iv=pd.concat([f for f in frames if len(f)]).drop_duplicates();iv['ticker']=iv.secid.map({v['secid']:t for t,v in mapping.items()})
    iv=iv.pivot(index='date',columns='ticker',values='impl_volatility').sort_index()*100
    iv=iv.where(iv>0)
    frames=[]
    for f in sorted((root/'crsp_underlying_daily').glob('*.parquet')):
        r=pd.read_parquet(f,columns=['permno','date','retx']);frames.append(r[r.permno.isin([v['permno'] for v in mapping.values()])])
    r=pd.concat([f for f in frames if len(f)]).drop_duplicates();r['ticker']=r.permno.map({v['permno']:t for t,v in mapping.items()})
    r=r.pivot(index='date',columns='ticker',values='retx').sort_index().astype(float)
    earnings=correct_earnings(pd.read_parquet(root/'compustat/underlying_quarterly_earnings.parquet'))
    supplement=pd.read_csv('config/earnings_supplement.csv',parse_dates=['earnings_date'])
    panels={};coverage=[]
    for definition in universe:
        pair=definition['pair'];a,b=pair.split('_')
        px=iv[[a,b]].rename(columns={a:'iv_amd',b:'iv_nvda'})
        ret=r[[a,b]].rename(columns={a:'AMD',b:'NVDA'})
        p=px.dropna().join(forward_rv(ret,px.dropna().index),how='inner').join(trailing_rv(ret))
        for ticker,prefix in [(a,'amd'),(b,'nvda')]:
            dates=pd.to_datetime(earnings.loc[earnings.permno.eq(mapping[ticker]['permno']),'rdq']).dropna()
            dates=pd.DatetimeIndex(pd.concat([dates,supplement.loc[supplement.ticker.eq(ticker),'earnings_date']],ignore_index=True).unique()).sort_values()
            if len(dates)<20:raise ValueError(f'Insufficient earnings for {ticker}')
            pos=dates.searchsorted(p.index,side='left');upcoming=pd.Series([dates[i] if i<len(dates) else pd.NaT for i in pos],index=p.index)
            p[prefix+'_earnings_days_capped']=(upcoming-pd.Series(p.index,index=p.index)).dt.days.clip(upper=31)
        # Use full local IV/return history before intersecting model dates; do not roll across dropped observations.
        changes=px.diff()
        stock=rolling_dependence(r[a],r[b],min_observations=200).add_prefix('stock_')
        vol=rolling_dependence(changes.iv_amd,changes.iv_nvda,min_observations=200).add_prefix('iv_')
        p=p.join(stock).join(vol)
        p['correlation_gap']=p.stock_correlation-p.iv_correlation
        p['beta_gap']=p.stock_beta-p.iv_beta
        p['spread']=p.iv_amd-p.iv_nvda
        p['long_reward']=(p.rv_amd-p.iv_amd)-(p.rv_nvda-p.iv_nvda)
        p=p.replace([np.inf,-np.inf],np.nan).dropna()
        p=p.loc[(p.index>='2002-12-31')&(p.index<='2025-08-29')]
        if len(p.loc[p.index>='2010-01-01'])<3000:raise ValueError(f'Insufficient evaluation coverage for {pair}')
        panels[pair]=p
        coverage.append(dict(pair=pair,industry=definition['industry'],original_pair=definition['original_pair'],rows=len(p),first=str(p.index.min().date()),last=str(p.index.max().date()),iv_dates=int(px.dropna().shape[0])))
        print('Prepared',pair,len(p),'rows',flush=True)
    common=panels[universe[0]['pair']].index
    for p in panels.values():common=common.intersection(p.index)
    panels={k:p.loc[common] for k,p in panels.items()}
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    pd.concat(panels,names=['pair','date']).to_parquet(output/'panels.parquet')
    print('Common panel:',len(common),'test dates:',sum(common>='2010-01-01'),flush=True)
    return panels


def block_difference(a,b,seed=2026,repeats=2000):
    rng=np.random.default_rng(seed);n=len(a);values=[]
    for _ in range(repeats):
        ix=((rng.integers(n,size=int(np.ceil(n/63)))[:,None]+np.arange(63))%n).ravel()[:n]
        values.append(payoff_sharpe(a[ix])-payoff_sharpe(b[ix]))
    return np.quantile(values,[.025,.975])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--universe',type=Path,default=Path('config/universe_gap_pairs.json'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/universe_gap'))
    ap.add_argument('--reports',type=Path,default=Path('reports/universe_gap'))
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.reports.mkdir(parents=True,exist_ok=True)
    universe=json.loads(args.universe.read_text())
    settings={'universe_sha256':hashlib.sha256(args.universe.read_bytes()).hexdigest(),'variants':VARIANTS,'rate':.025,'ladder':LADDER.tolist(),'window':252,'minimum_matched_observations':200,'minimum_test_rows':3000,'selection':'business similarity and data coverage; no payoff ranking','budget':'equal fixed budgets; independent pair policies; unused budget stays idle'}
    settings['earnings_overrides_sha256']=hashlib.sha256(EARNINGS_OVERRIDES.read_bytes()).hexdigest()
    settings_path=args.output/'settings.json'
    if args.resume and settings_path.exists():
        if json.loads(settings_path.read_text())!=settings:ap.error('Resume settings mismatch')
        packed=pd.read_parquet(args.output/'panels.parquet');panels={d['pair']:packed.xs(d['pair'],level='pair') for d in universe}
    else:
        panels=build(args.source_root,universe,args.output);settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    predictions={};configs=[]
    for definition in universe:
        pair=definition['pair'];panel=panels[pair];path=args.output/f'{pair}.csv';cpath=args.output/f'{pair}_configs.csv'
        if args.resume and path.exists() and cpath.exists():
            predictions[pair]=pd.read_csv(path,index_col='date',parse_dates=['date','end']);configs.extend(pd.read_csv(cpath).to_dict('records'));continue
        pieces=[];pair_configs=[]
        for year in range(2010,2026,2):
            tr,va,te=fold_data(panel,year)
            assert tr.end.max()<va.index.min() and va.end.max()<te.index.min()
            out=te[['end','correlation_gap','beta_gap']].copy();out['fold']=year
            for variant,features in VARIANTS.items():
                candidates=[]
                for leaves in [4,8,16]:
                    for minimum in [126,252]:
                        tree,ids,policy,score,_=fit_tree_policy(tr,features,leaves,minimum,entry_cost(tr,.025),action_values=LADDER)
                        vr=net_rewards(policy[leaf_states(tree,va,features,ids)],va.long_reward.to_numpy(),entry_cost(va,.025))
                        candidates.append((payoff_sharpe(vr),tree,ids,policy,leaves,minimum,score))
                vs,tree,ids,policy,leaves,minimum,score=max(candidates,key=lambda x:x[0])
                action=policy[leaf_states(tree,te,features,ids)]
                out[variant+'_action']=action;out[variant+'_reward']=net_rewards(action,te.long_reward.to_numpy(),entry_cost(te,.025))
                pair_configs.append(dict(pair=pair,fold=year,variant=variant,leaves=leaves,min_leaf=minimum,train_sharpe=score,validation_sharpe=vs))
            for feature in ['correlation_gap','beta_gap']:
                out[feature+'_high']=te[feature]>tr[feature].median()
            pieces.append(out)
        pred=pd.concat(pieces).sort_index();pred.to_csv(path);pd.DataFrame(pair_configs).to_csv(cpath,index=False)
        predictions[pair]=pred;configs.extend(pair_configs)
        print('Completed',pair,{v:round(payoff_sharpe(pred[v+'_reward']),3) for v in VARIANTS},flush=True)
    scores=[];annual=[];dependence=[];bins=[]
    for definition in universe:
        pair=definition['pair'];p=predictions[pair];d=panels[pair].loc[p.index]
        for variant in VARIANTS:
            r=p[variant+'_reward'];a=p[variant+'_action']
            scores.append(dict(pair=pair,original_pair=definition['original_pair'],industry=definition['industry'],variant=variant,entries=len(r),sharpe=payoff_sharpe(r),net_pnl=r.sum(),trade_fraction=a.ne(0).mean(),mean_size=a.abs().mean()))
            for year,g in p.groupby(p.index.year):annual.append(dict(pair=pair,year=year,variant=variant,sharpe=payoff_sharpe(g[variant+'_reward']),net_pnl=g[variant+'_reward'].sum()))
        dependence.append(dict(pair=pair,median_stock_correlation=d.stock_correlation.median(),median_iv_correlation=d.iv_correlation.median(),median_stock_beta=d.stock_beta.median(),median_iv_beta=d.iv_beta.median(),median_correlation_gap=d.correlation_gap.median(),median_beta_gap=d.beta_gap.median()))
        for feature in ['correlation_gap','beta_gap']:
            for high in [False,True]:
                mask=p[feature+'_high']==high;r=p.loc[mask,'baseline_reward']
                bins.append(dict(pair=pair,feature=feature,group='high' if high else 'low',entries=int(mask.sum()),conditional_sharpe=payoff_sharpe(r),mean_net_payoff=r.mean(),mean_absolute_gross_payoff=d.loc[mask,'long_reward'].abs().mean()))
    portfolio=[];portfolio_annual=[]
    for group,selected in [('all',[d['pair'] for d in universe]),('new',[d['pair'] for d in universe if not d['original_pair']]),('original',[d['pair'] for d in universe if d['original_pair']])]:
        returns={v:pd.concat([predictions[k][v+'_reward'] for k in selected],axis=1).mean(axis=1) for v in VARIANTS}
        for variant,r in returns.items():
            ci=block_difference(r.to_numpy(),returns['baseline'].to_numpy()) if variant!='baseline' else [0.,0.]
            portfolio.append(dict(group=group,pairs=len(selected),variant=variant,entries=len(r),sharpe=payoff_sharpe(r),delta_sharpe=payoff_sharpe(r)-payoff_sharpe(returns['baseline']),delta_ci_low=ci[0],delta_ci_high=ci[1],net_pnl=r.sum()))
            for year,g in r.groupby(r.index.year):portfolio_annual.append(dict(group=group,year=year,variant=variant,sharpe=payoff_sharpe(g),net_pnl=g.sum()))
    for name,rows in [('scores',scores),('annual',annual),('dependence',dependence),('bins',bins),('portfolio',portfolio),('portfolio_annual',portfolio_annual),('configs',configs)]:pd.DataFrame(rows).to_csv(args.reports/f'{name}.csv',index=False)
    for name in ['coverage.csv','settings.json']:(args.reports/name).write_bytes((args.output/name).read_bytes())
    print(pd.DataFrame(portfolio).to_string(index=False),flush=True)


if __name__=='__main__':main()
