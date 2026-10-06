"""Exploratory agreement filters, compared with identical available IV cohorts."""
import numpy as np
import pandas as pd
from scripts.mine_atm_variance_signals import OUT,REPORT


def main():
    data=pd.read_parquet(OUT/'pair_signals.parquet')
    screen=pd.read_csv(REPORT/'signals.csv')
    threshold=float(screen[(screen.signal=='iv_spread')&screen.period.eq('early_completed')&screen['sample'].eq('all')].early_fixed_extreme_threshold.iloc[0])
    variants={'iv_factor_agreement':['iv_factor_reversion'],
        'downside_agreement':['downside_fraction_gap'],
        'both_agree':['iv_factor_reversion','downside_fraction_gap']}
    rows=[];annual=[];pairs=[]
    for name,signals in variants.items():
        for period,g in [('early_completed',data[(data.date<'2022-01-01')&(data.end<'2022-01-01')]),('later',data[data.date>='2022-01-01'])]:
            # Availability is matched BEFORE comparing filtered versus original.
            g=g.dropna(subset=['iv_spread',*signals]).copy()
            g['base']=g.iv_spread.abs()>=threshold
            g['agree']=np.logical_and.reduce([np.sign(g[s])==np.sign(g.iv_spread) for s in signals])
            g['selected_net']=np.where(g.iv_spread>0,g.long_net,g.short_net)
            for sample,h in [('all',g),('nonoverlap',g[g.nonoverlap])]:
                for rule,chosen in [('matched_iv_spread',h.base),('with_agreement',h.base&h.agree)]:
                    z=h[chosen]
                    rows.append(dict(filter=name,period=period,sample=sample,rule=rule,eligible_observations=len(h),
                        trades=len(z),active_pairs=z.pair.nunique(),mean_net=float(z.selected_net.mean()) if len(z) else None,
                        median_net=float(z.selected_net.median()) if len(z) else None,
                        net_per_opportunity=float(np.where(chosen,h.selected_net,0.).mean()) if len(h) else None))
                    if sample=='all':
                        for year,y in h.groupby(h.date.dt.year):
                            selected=y.base if rule=='matched_iv_spread' else y.base&y.agree
                            annual.append(dict(filter=name,rule=rule,year=year,trades=int(selected.sum()),
                                mean_net=float(y.loc[selected,'selected_net'].mean()) if selected.any() else None))
                    if sample=='all' and period=='later':
                        for pair,y in z.groupby('pair'):
                            pairs.append(dict(filter=name,rule=rule,pair=pair,label=y.label.iloc[0],trades=len(y),mean_net=float(y.selected_net.mean())))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'agreement_filters.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'agreement_annual.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'agreement_pairs.csv',index=False)
    note=REPORT/'summary.txt'
    note.write_text(note.read_text()+'\nExploratory IV-spread agreement filters:\n'
        'Follow-up hypotheses proposed after the initial screen, not predeclared or a fresh holdout. The original economic IV direction and early-only 90th-percentile absolute-spread threshold remain fixed. Each comparison matches signal availability. Payoffs are standalone terminal dollars per initial mean leg vega, not portfolio Sharpe.\n'
        +result.to_string(index=False)+'\n')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
