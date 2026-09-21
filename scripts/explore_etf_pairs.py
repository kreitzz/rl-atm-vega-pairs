"""Independent ETF/stock and ETF/ETF policies using the local standardized-IV cache."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_first_pass import forward_rv, net_rewards, payoff_sharpe
from scripts.run_rv_vix_pass import trailing_rv, fit_tree_policy, leaf_states
from scripts.run_four_pairs import FEATURES, LADDER, entry_cost
from scripts.run_walk_forward import fold_data
from scripts.diagnose_correlation_gap import rolling_dependence
from scripts.explore_universe_gap import VARIANTS, block_difference


def add_countdown(panel, dates, prefix, is_etf):
    if is_etf:
        panel[prefix+'_earnings_days_capped']=31
        return
    dates=pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if len(dates)<20:raise ValueError('Insufficient stock earnings coverage')
    pos=dates.searchsorted(panel.index,side='left')
    nxt=pd.Series([dates[i] if i<len(dates) else pd.NaT for i in pos],index=panel.index)
    panel[prefix+'_earnings_days_capped']=(nxt-pd.Series(panel.index,index=panel.index)).dt.days.clip(upper=31)


def eligible_fold(train,val,test):
    return len(train)>=504 and len(val)>=252 and len(test)>=126


def build(root, config, output):
    assets=config['assets'];secids=[x['secid'] for x in assets.values()];permnos=[x['permno'] for x in assets.values()]
    om=root/'optionmetrics';files=sorted((om/'atm_iv_compact').rglob('*.parquet'))+sorted((om/'atm_implied_volatility').glob('*.parquet'))
    frames=[pd.read_parquet(f,columns=['secid','date','impl_volatility'],filters=[('secid','in',secids),('days','=',30),('delta','=',50),('cp_flag','=','C')]) for f in files]
    iv=pd.concat([x for x in frames if len(x)]).drop_duplicates();iv['ticker']=iv.secid.map({v['secid']:k for k,v in assets.items()});iv=iv.pivot(index='date',columns='ticker',values='impl_volatility').sort_index()*100;iv=iv.where(iv>0)
    frames=[]
    for f in sorted((root/'crsp_underlying_daily').glob('*.parquet')):
        d=pd.read_parquet(f,columns=['permno','date','retx']);frames.append(d[d.permno.isin(permnos)])
    r=pd.concat([x for x in frames if len(x)]).drop_duplicates();r['ticker']=r.permno.map({v['permno']:k for k,v in assets.items()});r=r.pivot(index='date',columns='ticker',values='retx').sort_index().astype(float)
    for t,v in assets.items():
        iv.loc[iv.index<pd.Timestamp(v['history_start']),t]=np.nan
        r.loc[r.index<pd.Timestamp(v['history_start']),t]=np.nan
    earnings=pd.read_parquet(root/'compustat/underlying_quarterly_earnings.parquet')
    panels={};coverage=[];audit=[]
    for t,v in assets.items():
        if not v['etf']:
            dates=pd.to_datetime(earnings.loc[earnings.permno.eq(v['permno']),'rdq']).dropna().drop_duplicates().sort_values()
            audit.append(dict(ticker=t,announcements=len(dates),max_gap_days=dates.diff().dt.days.max()))
    for definition in config['pairs']:
        pair=definition['pair'];a,b=pair.split('_');px=iv[[a,b]].rename(columns={a:'iv_amd',b:'iv_nvda'});ret=r[[a,b]].rename(columns={a:'AMD',b:'NVDA'})
        p=px.dropna().join(forward_rv(ret,px.dropna().index),how='inner').join(trailing_rv(ret))
        for t,prefix in [(a,'amd'),(b,'nvda')]:add_countdown(p,earnings.loc[earnings.permno.eq(assets[t]['permno']),'rdq'],prefix,assets[t]['etf'])
        changes=px.diff();p=p.join(rolling_dependence(r[a],r[b],min_observations=200).add_prefix('stock_')).join(rolling_dependence(changes.iv_amd,changes.iv_nvda,min_observations=200).add_prefix('iv_'))
        p['correlation_gap']=p.stock_correlation-p.iv_correlation;p['beta_gap']=p.stock_beta-p.iv_beta
        p['spread']=p.iv_amd-p.iv_nvda;p['long_reward']=(p.rv_amd-p.iv_amd)-(p.rv_nvda-p.iv_nvda)
        p=p.replace([np.inf,-np.inf],np.nan).dropna().loc[:'2025-08-29'];panels[pair]=p
        folds=[y for y in range(2010,2026,2) if eligible_fold(*fold_data(p,y))]
        coverage.append(dict(pair=pair,kind=definition['kind'],short_history=definition['short_history'],first=str(p.index.min().date()),last=str(p.index.max().date()),rows=len(p),test_folds=','.join(map(str,folds))))
        print('Prepared',pair,len(p),'rows; test folds',folds,flush=True)
    pd.concat(panels,names=['pair','date']).to_parquet(output/'panels.parquet');pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False);pd.DataFrame(audit).to_csv(output/'earnings_audit.csv',index=False)
    return panels


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=Path('config/etf_pairs.json'));ap.add_argument('--output',type=Path,default=Path('artifacts/etf_exploration/run'));ap.add_argument('--reports',type=Path,default=Path('reports/etf_exploration'));ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);args.reports.mkdir(parents=True,exist_ok=True);config=json.loads(args.config.read_text())
    settings={'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'variants':VARIANTS,'cost_rate':.025,'window':252,'min_observations':200,'min_train':504,'min_validation':252,'min_test':126,'etf_earnings':'constant 31: no corporate earnings countdown','sizing':'independent ladder; fixed equal pair budgets'}
    sp=args.output/'settings.json'
    if args.resume and sp.exists():
        if json.loads(sp.read_text())!=settings:ap.error('Resume settings differ')
        packed=pd.read_parquet(args.output/'panels.parquet');panels={x['pair']:packed.xs(x['pair'],level='pair') for x in config['pairs']}
    else:panels=build(args.source_root,config,args.output);sp.write_text(json.dumps(settings,indent=2)+'\n')
    predictions={};configs=[]
    for definition in config['pairs']:
        pair=definition['pair'];panel=panels[pair];path=args.output/f'{pair}.csv';cp=args.output/f'{pair}_configs.csv'
        if args.resume and path.exists() and cp.exists():predictions[pair]=pd.read_csv(path,index_col='date',parse_dates=['date','end']);configs.extend(pd.read_csv(cp).to_dict('records'));continue
        pieces=[];selected=[]
        for year in range(2010,2026,2):
            tr,va,te=fold_data(panel,year)
            if not eligible_fold(tr,va,te):continue
            assert tr.end.max()<va.index.min() and va.end.max()<te.index.min()
            out=te[['end','correlation_gap','beta_gap']].copy();out['fold']=year
            for variant,features in VARIANTS.items():
                candidates=[]
                for leaves in [4,8,16]:
                    for minimum in [126,252]:
                        tree,ids,policy,score,_=fit_tree_policy(tr,features,leaves,minimum,entry_cost(tr,.025),action_values=LADDER)
                        vr=net_rewards(policy[leaf_states(tree,va,features,ids)],va.long_reward.to_numpy(),entry_cost(va,.025))
                        candidates.append((payoff_sharpe(vr),tree,ids,policy,leaves,minimum,score))
                vs,tree,ids,policy,leaves,minimum,score=max(candidates,key=lambda x:x[0]);action=policy[leaf_states(tree,te,features,ids)]
                out[variant+'_action']=action;out[variant+'_reward']=net_rewards(action,te.long_reward.to_numpy(),entry_cost(te,.025))
                selected.append(dict(pair=pair,fold=year,variant=variant,leaves=leaves,min_leaf=minimum,train_sharpe=score,validation_sharpe=vs))
            for feature in ['correlation_gap','beta_gap']:out[feature+'_high']=te[feature]>tr[feature].median()
            pieces.append(out)
        if not pieces:print('No eligible folds:',pair,flush=True);continue
        pred=pd.concat(pieces).sort_index();pred.to_csv(path);pd.DataFrame(selected).to_csv(cp,index=False);predictions[pair]=pred;configs.extend(selected)
        print('Completed',pair,'entries',len(pred),{v:round(payoff_sharpe(pred[v+'_reward']),3) for v in VARIANTS},flush=True)
    rows=[];annual=[];profiles=[];controls=[];bins=[];legs=[]
    for definition in config['pairs']:
        pair=definition['pair']
        if pair not in predictions:continue
        p=predictions[pair];q=panels[pair].loc[p.index];cost=entry_cost(q,.025)
        profiles.append(dict(pair=pair,kind=definition['kind'],first=str(p.index.min().date()),last=str(p.index.max().date()),entries=len(p),median_return_correlation=q.stock_correlation.median(),median_iv_correlation=q.iv_correlation.median(),median_return_beta=q.stock_beta.median(),median_iv_beta=q.iv_beta.median(),median_correlation_gap=q.correlation_gap.median(),mean_iv_first=q.iv_amd.mean(),mean_iv_second=q.iv_nvda.mean()))
        for variant in VARIANTS:
            r=p[variant+'_reward'];a=p[variant+'_action'];rows.append(dict(pair=pair,kind=definition['kind'],short_history=definition['short_history'],variant=variant,entries=len(r),sharpe=payoff_sharpe(r),net_pnl=r.sum(),trade_fraction=a.ne(0).mean(),mean_size=a.abs().mean(),long_first_fraction=a.gt(0).mean(),short_first_fraction=a.lt(0).mean()))
            for year,g in p.groupby(p.index.year):annual.append(dict(pair=pair,variant=variant,year=year,sharpe=payoff_sharpe(g[variant+'_reward']),net_pnl=g[variant+'_reward'].sum()))
            for ticker,gap,iv,sign in [(pair.split('_')[0],q.rv_amd-q.iv_amd,q.iv_amd,1),(pair.split('_')[1],q.rv_nvda-q.iv_nvda,q.iv_nvda,-1)]:
                gross=sign*a*gap;lc=a.abs()*.025*iv/2;legs.append(dict(pair=pair,variant=variant,leg=ticker,gross_pnl=gross.sum(),cost=lc.sum(),net_pnl=(gross-lc).sum()))
        for name,a in [('always_long_first',np.ones(len(q))),('always_short_first',-np.ones(len(q))),('short_higher_iv',-np.sign(q.spread.to_numpy()))]:
            r=net_rewards(a,q.long_reward.to_numpy(),cost);controls.append(dict(pair=pair,policy=name,sharpe=payoff_sharpe(r),net_pnl=r.sum()))
        for feature in ['correlation_gap','beta_gap']:
            for high in [False,True]:
                mask=p[feature+'_high'].eq(high);r=p.loc[mask,'baseline_reward'];bins.append(dict(pair=pair,feature=feature,group='high' if high else 'low',entries=int(mask.sum()),conditional_sharpe=payoff_sharpe(r) if len(r)>1 else np.nan,mean_net_payoff=r.mean()))
    portfolio=[];portfolio_annual=[]
    groups={'core_etf_stock':[x['pair'] for x in config['pairs'] if x['kind']=='ETF_stock' and not x['short_history']],
            'energy_etf_stock':[x['pair'] for x in config['pairs'] if x['kind']=='ETF_stock' and x['pair'].split('_')[0] in ['XOP','OIH']],
            'pharma_etf_stock':[x['pair'] for x in config['pairs'] if x['pair'].startswith('PPH_')],
            'silver_etf_stock':[x['pair'] for x in config['pairs'] if x['pair'].startswith('SIL_')]}
    for group,selected in groups.items():
        selected=[k for k in selected if k in predictions]
        if not selected:continue
        common=predictions[selected[0]].index
        for k in selected:common=common.intersection(predictions[k].index)
        returns={v:pd.concat([predictions[k].loc[common,v+'_reward'] for k in selected],axis=1).mean(axis=1) for v in VARIANTS}
        for variant,r in returns.items():
            ci=block_difference(r.to_numpy(),returns['baseline'].to_numpy()) if variant!='baseline' else [0.,0.]
            portfolio.append(dict(group=group,pairs=len(selected),first=str(r.index.min().date()),last=str(r.index.max().date()),entries=len(r),variant=variant,sharpe=payoff_sharpe(r),delta_sharpe=payoff_sharpe(r)-payoff_sharpe(returns['baseline']),delta_ci_low=ci[0],delta_ci_high=ci[1]))
            for year,g in r.groupby(r.index.year):portfolio_annual.append(dict(group=group,variant=variant,year=year,sharpe=payoff_sharpe(g),net_pnl=g.sum()))
    for name,data in [('scores',rows),('annual',annual),('profiles',profiles),('controls',controls),('conditional_groups',bins),('legs',legs),('configs',configs),('portfolio',portfolio),('portfolio_annual',portfolio_annual)]:pd.DataFrame(data).to_csv(args.reports/f'{name}.csv',index=False)
    for name in ['coverage.csv','earnings_audit.csv','settings.json']:(args.reports/name).write_bytes((args.output/name).read_bytes())
    print(pd.DataFrame(portfolio).to_string(index=False),flush=True)


if __name__=='__main__':main()
