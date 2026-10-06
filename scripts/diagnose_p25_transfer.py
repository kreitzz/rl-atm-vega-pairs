"""Read-only attribution of frozen cross-wing transfer outputs."""
from pathlib import Path
import json
import resource
import numpy as np
import pandas as pd
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest

OUT=Path('reports/p25_signal_transfer_diagnostics')
BOOK=Path('artifacts/contracts/p25_signal_transfer/replays')
CACHE=Path('artifacts/contracts/uniform_smile')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    def read(path,**kwargs):
        sources[str(path)]=digest(path)
        return pd.read_parquet(path,**kwargs)
    def csv(path,**kwargs):
        sources[str(path)]=digest(path)
        return pd.read_csv(path,**kwargs)
    components=[];annual=[];exposure=[];decisions=[]
    for variant in ['fixed_100','floor']:
        for wing in ['ATM','P25','P75']:
            name=f'{variant}_{wing}'
            folder=BOOK/name
            nav=read(folder/'daily_nav.parquet')
            d=csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
            trades=csv(folder/'trades.csv',parse_dates=['date'])
            # option_pnl already includes all option commissions and model
            # penalties; add them back exactly once for the gross comparison.
            for year,g in [(None,nav),*list(nav.groupby(nav.index.year))]:
                gross=float((g.option_pnl+g.option_execution_cost+g.hedge_gross_pnl+g.dividend_cash).sum())
                option_cost=float(g.option_execution_cost.sum())
                hedge_cost=float(g.hedge_cost.sum())
                borrow=float(g.stock_borrow_cost.sum())
                funding=float((g.interest-nav.cash_benchmark.diff().fillna(0).reindex(g.index)).sum())
                excess=gross-option_cost-hedge_cost-borrow+funding
                expected=float((g.pnl-nav.cash_benchmark.diff().fillna(0).reindex(g.index)).sum())
                assert abs(excess-expected)<1e-4
                item=dict(model=name,year=year,gross_delta_hedged_pnl=gross,
                    option_execution_cost=option_cost,hedge_cost=hedge_cost,stock_borrow_cost=borrow,
                    excess_financing=funding,net_excess_pnl=excess)
                (components if year is None else annual).append(item)
            e=d.loc[d.status.eq('entered')].copy()
            exposure.append(dict(model=name,entries=len(e),active_pairs=int(e.pair.nunique()),
                mean_leg_entry_vega=float(e[['vega_a','vega_b']].to_numpy().mean()),
                mean_gross_hedge_notional=float(nav.gross_hedge_notional.mean()),
                mean_capital_reserve=float(nav.capital_reserve.mean()),
                daily_excess_pnl_sd=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).iloc[1:].std()),
                active_sessions=int(nav.active_cohorts.gt(0).sum()),
                greek_fallbacks=int(nav.greek_fallbacks.sum()),
                modeled_held_marks=int(nav.causal_held_mark_fallbacks.sum())))
            for status,count in d.status.value_counts().items():
                decisions.append(dict(model=name,status=status,count=int(count)))
    pd.DataFrame(components).to_csv(OUT/'pnl_components.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_pnl_components.csv',index=False)
    pd.DataFrame(exposure).to_csv(OUT/'exposures.csv',index=False)
    pd.DataFrame(decisions).to_csv(OUT/'decisions.csv',index=False)
    anchor=read(BOOK/'floor_P25/scored_allocations.parquet',columns=['date','pair','direction','optionid_a','optionid_b'])
    shadow=[];signals=[]
    for wing in ['ATM','P25','P75']:
        raw=read(CACHE/f'{wing}_rows.parquet')
        raw=raw.loc[raw.date.ge('2018-01-01')].copy()
        joined=anchor.rename(columns={'direction':'p25_direction','optionid_a':'p25_optionid_a','optionid_b':'p25_optionid_b'}).merge(raw,on=['date','pair'],how='inner',validate='one_to_one')
        side=joined.p25_direction.to_numpy()>0
        long_profit=np.where(side,joined.profit_long_91d,joined.profit_short_91d)
        annual_profit=np.where(side,joined.profit_long_365d,joined.profit_short_365d)
        short_count=np.where(side,joined.count_long_91d,joined.count_short_91d)
        annual_count=np.where(side,joined.count_long_365d,joined.count_short_365d)
        signals.append(dict(wing=wing,matched_observations=len(joined),
            native_same_direction=int(joined.rule_direction.eq(joined.p25_direction).sum()),
            native_opposite_direction=int(joined.rule_direction.eq(-joined.p25_direction).sum()),
            native_no_rank_signal=int(joined.rule_direction.eq(0).sum()),
            own_memory_pass_for_p25_direction=int(((long_profit>0)&(annual_profit>0)&(short_count>=21)&(annual_count>=63)).sum())))
        for pair,g in joined.groupby('pair',sort=False):
            path=CACHE/'pairs'/pair/f'{wing}_labels.npy'
            sources[str(path)]=digest(path)
            labels=np.load(path,mmap_mode='r')
            for row in g.itertuples():
                if row.finite_label:
                    value=labels[row.label_row,8 if row.p25_direction>0 else 1]
                    assert np.isfinite(value).all()
                    shadow.append(dict(date=row.date,pair=pair,wing=wing,
                        shadow_pnl_at_training_cost=float(value.sum()*450)))
            del labels
    panel=pd.DataFrame(shadow).pivot(index=['date','pair'],columns='wing',values='shadow_pnl_at_training_cost')
    panel.to_csv(OUT/'shadow_matched_outcomes.csv')
    summaries=[]
    for other in ['ATM','P75']:
        matched=panel[['P25',other]].dropna()
        for wing in ['P25',other]:
            summaries.append(dict(comparison=other,wing=wing,matched_complete_outcomes=len(matched),
                mean_shadow_pnl_at_training_cost=float(matched[wing].mean()),
                total_shadow_pnl_at_training_cost=float(matched[wing].sum()),
                profitable_fraction=float(matched[wing].gt(0).mean()),
                payout_correlation=float(matched.P25.corr(matched[other]))))
        bypair=matched.groupby('pair').sum()
        bypair.to_csv(OUT/f'{other}_matched_pair_shadow_pnl.csv')
    pd.DataFrame(summaries).to_csv(OUT/'matched_outcome_summary.csv',index=False)
    pd.DataFrame(signals).to_csv(OUT/'native_signal_agreement.csv',index=False)
    write_json(OUT/'audit.json',dict(status='complete',no_retraining=True,no_new_backtest=True,
        actual_pnl_components_reconcile=True,
        shadow_scope='Same pair/date completed outcomes with original contracts, fixed 450 vega, training .65 commissions, standalone financing and no portfolio inventory overlay; diagnostic only, not executable returns or trade selection',
        forward_outcomes_used_only_for_retrospective_diagnosis=True,
        sources=sources,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    for path,value in sources.items():
        assert digest(path)==value,path
    print(pd.DataFrame(components).to_string(index=False),flush=True)
    print(pd.DataFrame(signals).to_string(index=False),flush=True)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
