"""Audit independent pair generalization, annual loss concentration, and IV co-movement."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_four_pairs import PAIRS, FEATURES, LADDER, entry_cost
from scripts.run_rv_vix_pass import fit_tree_policy, leaf_states
from scripts.run_walk_forward import fold_data
from scripts.run_first_pass import net_rewards, payoff_sharpe


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=Path('artifacts/four_pairs_independent_cost_0.025'))
    ap.add_argument('--output',type=Path,default=Path('reports/independent_diagnostics'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if json.loads((args.run/'cost_settings.json').read_text())['cost_rate']!=.025:
        ap.error('This diagnostic expects the 0.025 proportional-cost run')
    pred=pd.read_csv(args.run/'predictions.csv',index_col='date',parse_dates=['date','end'])
    packed=pd.read_parquet(args.run/'panels.parquet')
    configs=pd.read_csv(args.run/'selected_configs.csv')
    summaries=[]; annual=[]; folds=[]; legs=[]; baselines=[]
    for pair in PAIRS:
        panel=packed.xs(pair,level='pair').sort_index();q=panel.loc[pred.index]
        a=pred[pair+'_solo_action'];r=pred[pair+'_solo_reward'];cost=a.abs()*pred[pair+'_cost']
        gross=r+cost; yearly=r.groupby(r.index.year).sum();worst=yearly.idxmin();remaining=r[r.index.year!=worst]
        for name,action in [('always_long_first',np.ones(len(q))),('always_short_first',-np.ones(len(q))),('short_higher_iv',-np.sign(q.spread.to_numpy()))]:
            reward=net_rewards(action,q.long_reward.to_numpy(),pred[pair+'_cost'].to_numpy())
            baselines.append(dict(pair=pair,policy=name,sharpe=payoff_sharpe(reward),net_pnl=reward.sum()))
        delta=panel[['iv_amd','iv_nvda']].diff().loc[pred.index].dropna()
        rho=delta.iv_amd.corr(delta.iv_nvda)
        rolling_rho=delta.iv_amd.rolling(252).corr(delta.iv_nvda).dropna()
        rolling_beta=delta.iv_amd.rolling(252).cov(delta.iv_nvda)/delta.iv_nvda.rolling(252).var()
        gated=r.copy();pair_folds=[]
        for cfg in configs[configs.model.eq(pair)].itertuples():
            tr,va,te=fold_data(panel,int(cfg.fold))
            tree,ids,policy,score,_=fit_tree_policy(tr,FEATURES,int(cfg.leaves),int(cfg.min_leaf),entry_cost(tr,.025),action_values=LADDER)
            actions=policy[leaf_states(tree,te,FEATURES,ids)]
            np.testing.assert_allclose(actions,pred.loc[te.index,pair+'_solo_action'],atol=1e-12)
            v=net_rewards(policy[leaf_states(tree,va,FEATURES,ids)],va.long_reward.to_numpy(),entry_cost(va,.025))
            np.testing.assert_allclose(payoff_sharpe(v),cfg.validation_sharpe,atol=1e-10)
            test=r.loc[te.index]
            rec=dict(pair=pair,fold=int(cfg.fold),train_sharpe=score,validation_sharpe=cfg.validation_sharpe,test_sharpe=payoff_sharpe(test),test_net_pnl=float(test.sum()))
            pair_folds.append(rec);folds.append(rec)
            if cfg.validation_sharpe<=0:gated.loc[te.index]=0
        f=pd.DataFrame(pair_folds)
        summaries.append(dict(pair=pair,iv_change_correlation=rho,iv_change_beta_first_on_second=delta.iv_amd.cov(delta.iv_nvda)/delta.iv_nvda.var(),rolling_252d_correlation_median=rolling_rho.median(),rolling_252d_correlation_p10=rolling_rho.quantile(.1),rolling_252d_correlation_p90=rolling_rho.quantile(.9),rolling_beta_median=rolling_beta.median(),positive_years=int((yearly>0).sum()),negative_years=int((yearly<0).sum()),flat_years=int((yearly==0).sum()),worst_year=int(worst),worst_year_pnl=yearly.min(),net_pnl=r.sum(),sharpe=payoff_sharpe(r),excluding_worst_year_pnl=remaining.sum(),excluding_worst_year_sharpe=payoff_sharpe(remaining),gross_pnl=gross.sum(),cost=cost.sum(),gross_per_size=gross.sum()/a.abs().sum(),cost_per_size=cost.sum()/a.abs().sum(),median_fold_train_sharpe=f.train_sharpe.median(),median_fold_validation_sharpe=f.validation_sharpe.median(),median_fold_test_sharpe=f.test_sharpe.median(),positive_test_folds=int((f.test_sharpe>0).sum()),validation_gated_sharpe=payoff_sharpe(gated),validation_gated_net_pnl=gated.sum()))
        for year in sorted(q.index.year.unique()):
            mask=q.index.year==year
            annual.append(dict(pair=pair,year=year,net_pnl=r.loc[mask].sum(),gross_pnl=gross.loc[mask].sum(),cost=cost.loc[mask].sum(),sharpe=payoff_sharpe(r.loc[mask]),trade_fraction=a.loc[mask].ne(0).mean()))
        for leg,leg_gross in [(pair.split('_')[0],a*(q.rv_amd-q.iv_amd)),(pair.split('_')[1],-a*(q.rv_nvda-q.iv_nvda))]:
            # Half each leg's modeled full width; sums to proportional pair cost.
            iv=q.iv_amd if leg==pair.split('_')[0] else q.iv_nvda
            leg_cost=a.abs()*.025*iv/2
            legs.append(dict(pair=pair,leg=leg,gross_pnl=leg_gross.sum(),cost=leg_cost.sum(),net_pnl=(leg_gross-leg_cost).sum()))
        print(pair,'verified selected policies',flush=True)
    for name,rows in [('summary',summaries),('annual',annual),('folds',folds),('legs',legs),('simple_baselines',baselines)]:
        pd.DataFrame(rows).to_csv(args.output/f'{name}.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':main()
