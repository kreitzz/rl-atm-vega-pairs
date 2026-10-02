"""Interpret reconciled diagnostics without changing or selecting policies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.diagnose_current_contract_losses import REPORT, POLICIES, read_csv, complete_nav
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar


def main():
    pairs=pd.read_csv(REPORT/'pairs.csv');components=pd.read_csv(REPORT/'components.csv').set_index('policy')
    cohort=pd.read_csv(REPORT/'cohorts.csv',parse_dates=['entry_date','exit_date'])
    planned=[]
    for (policy,pair),g in cohort.groupby(['policy','pair']):
        root,subdir,*_=POLICIES[policy]
        decisions=read_csv(root/pair/subdir/'decisions.csv',['signal_date','entry_date'])
        entered=decisions[decisions.status.eq('entered')].copy()
        lookup=entered.assign(cohort=entered.signal_date.dt.strftime('%Y-%m-%d')).set_index('cohort').exit_date
        p=g.copy();p['planned_exit_date']=pd.to_datetime(p.cohort.map(lookup))
        assert p.planned_exit_date.notna().all()
        p['planned_holding_days']=(p.planned_exit_date-p.entry_date).dt.days
        p['exited_early']=p.exit_date.lt(p.planned_exit_date)
        planned.append(p)
    cohort=pd.concat(planned,ignore_index=True);cohort.to_csv(REPORT/'cohorts_with_planned_exit.csv',index=False)
    holding=cohort.groupby('policy').agg(entries=('cohort','size'),average_actual_days=('holding_days','mean'),
        average_planned_days=('planned_holding_days','mean'),early_exits=('exited_early','sum'),early_exit_fraction=('exited_early','mean'))
    holding.to_csv(REPORT/'holding_periods.csv')
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2018-01-01')&(calendar<='2025-08-29')]
    market=Market(calendar);cfg=json.loads((REPORT/'manifest.json').read_text())['settings'];funding=[]
    for r in pairs[pairs.entries.gt(0)].itertuples():
        root,subdir,*_=POLICIES[r.policy]
        nav=complete_nav(pd.read_parquet(root/r.pair/subdir/'daily_nav.parquet'),cfg['capital'])
        dt=calendar.to_series().diff().dt.days.fillna(0)/365
        accrual=market.rates.shift(1).fillna(0)*dt
        prior_nav=nav.nav.shift(1).fillna(cfg['capital']);cash=nav.cash.shift(1).fillna(cfg['capital'])
        benchmark=nav.cash_benchmark.shift(1).fillna(cfg['capital'])
        exposure=(cash-prior_nav)*accrual
        compounding=(prior_nav-benchmark)*accrual
        surcharge=cash.clip(upper=0)*cfg['borrowing_spread']*dt
        observed=nav.interest-nav.cash_benchmark.diff().fillna(nav.cash_benchmark.iloc[0]-cfg['capital'])
        np.testing.assert_allclose(exposure+compounding+surcharge,observed,atol=1e-7,rtol=0)
        funding.append(dict(policy=r.policy,pair=r.pair,label=r.label,position_funding=exposure.sum(),
            prior_pnl_compounding=compounding.sum(),borrowing_surcharge=surcharge.sum(),total_financing_vs_cash=observed.sum()))
    funding=pd.DataFrame(funding);funding.to_csv(REPORT/'funding.csv',index=False)
    aggregate=funding.groupby('policy').sum(numeric_only=True)
    np.testing.assert_allclose(aggregate.total_financing_vs_cash,components.reindex(aggregate.index).financing_vs_cash,atol=1e-5,rtol=0)
    aggregate.to_csv(REPORT/'funding_summary.csv')
    normalized=[]
    for name,g in pairs.groupby('policy'):
        vega=g.entry_vega_sum.sum()
        normalized.append(dict(policy=name,entry_vega_sum=vega,**{c+'_per_entry_vega':g[c].sum()/vega for c in ['gross_hedged_pnl','option_commission','option_penalty','stock_execution','stock_borrow']}))
    pd.DataFrame(normalized).to_csv(REPORT/'vega_normalized_components.csv',index=False)
    report=Path('reports/current_contract_losses.md');text=report.read_text().split('\n## Interpretation')[0]
    lines=['','## Interpretation','',
        'The latest filtered policies have positive but small gross option-plus-hedge P&L. Costs overwhelm that edge. Removing execution costs and stock borrow gives full-period fixed-inventory Sharpe of only 0.093 for surface RL and 0.283 for surface NN. The later-period RL remains negative at −0.178 even in that diagnostic. The earlier filtered RL similarly reaches only 0.168 over the full period. Thus execution costs explain the sign flip over the full history, but do not conceal a strong persistent gross signal.', '',
        'For surface RL, gross hedged profit is 0.330 dollars per dollar of initial mean-leg vega (equivalently about 0.33 vol points). The deductions are 0.560 commissions, 0.287 option execution penalty, 0.280 stock execution, and 0.069 stock borrow, before the cash-benchmark financing difference. These are aggregate realized P&L/cost ratios, not predicted trade edges or actual quoted bid/ask widths.', '',
        'The current filtered RL loses $75,271 in AMD/NVDA and $44,282 in M/KSS over the full period. The five worst pairs contribute 58.5% of gross pair losses. Its 2022 gross hedged loss is $61,627 before costs, versus a $97,165 gross gain in 2020. The weakness is therefore also pair- and regime-specific.', '',
        'A concrete jump-risk example: on January 24, 2022 the filtered surface RL held long Macy’s calls and short Kohl’s calls. In our underlying data Macy’s rose 18.0% and Kohl’s 36.0% close-to-close. The KSS leg lost $73,326 after its stock hedge, while M gained $23,596, leaving a gross pair loss of $49,730 and net loss of $49,879. Kohl’s confirmed acquisition interest that day in its [company announcement](https://investors.kohls.com/news/news-details/2022/Kohls-Confirms-Receipt-of-Expressions-of-Interest/default.aspx). Daily delta hedging and initial vega neutrality did not offset this relative jump.', '',
        'There is also quote-mark noise. On November 20, 2024 a held PENN call’s bid/ask widened from $1.26/$1.30 to $1.04/$2.22. Its midpoint-fit IV jumped from 45.6% to 69.3%, and the PENN/CZR book lost $22,848. The next day the same call quoted $1.12/$1.18 and the book gained $22,990. This reversal shows why individual daily losses and Sharpe/drawdown estimates need quote-quality scrutiny; it does not explain the cumulative loss by itself. Held quotes are currently marked at mid without the 25% relative-width entry filter.', '',
        'The worst-day contract decomposition exactly reconstructs the saved prior hedge. It separates repricing at the old IV with current spot/time/rate/known dividends from the residual to the observed new midpoint. This is a path-dependent accounting decomposition, not a clean causal Greek regression. For rows without valid IV at both endpoints, the residual must not be called an IV-change effect.', '',
        '### Actual holding periods', '',
        '| Policy | Entries | Planned hold, days | Actual hold, days | Exited before planned date |','|---|---:|---:|---:|---:|']
    for name in ['earlier_filtered_rl','surface_filtered_rl','surface_filtered_nn']:
        h=holding.loc[name];lines.append(f'| {name} | {int(h.entries)} | {h.average_planned_days:.2f} | {h.average_actual_days:.2f} | {h.early_exit_fraction:.1%} |')
    lines+=['','Early exits above compare actual and scheduled dates. Exit reason alone is insufficient: a missing-Greek or dividend tag can coincide with the scheduled day. The standalone training labels include dividend and missing-Greek exits, but exclude overlapping-book risk limits; frequent portfolio risk exits create a remaining training/deployment reward mismatch.', '',
        'All invalid-Greek legs triggering exits in the three filtered policies have midpoint prices at or below the European model’s lower bound. For current surface RL there are 88 affected cohorts, with 98 invalid legs and median 7 days to expiry. Quote synchronization and the pricing/dividend approximation need investigation. These forced liquidations are a simulation assumption; whether alternative treatment improves P&L requires a controlled replay, not arbitrary replacement marks.', '',
        'The cash-interest difference is not entirely a borrowing charge. It also includes reduced interest earned after prior trading losses. The funding table separates position funding, compounding of previous P&L versus cash, and any negative-cash surcharge.', '',
        'Priorities supported by this diagnosis: audit wide held marks and invalid-Greek handling; align training rewards with actual overlapping-book exits; demand enough expected net edge to cover the complete cost budget; and examine the identified jump/relative-IV losses before adding further states. These findings do not justify deleting losing pairs or dates retrospectively.', '',
        '[Vega-normalized components](current_contract_losses/vega_normalized_components.csv) · [Actual versus planned holds](current_contract_losses/holding_periods.csv) · [Financing decomposition](current_contract_losses/funding_summary.csv) · [Greek-exit quote diagnosis](current_contract_losses/missing_greek_reasons.csv) · [Worst-day contracts](current_contract_losses/worst_day_contracts.csv)','',
        'Verification: 129 tests passed, including independent ledger fixtures with split adjustments. All diagnosed complete books reconcile daily to less than $0.000001. No policies, costs, marks or risk limits were changed by this analysis.','']
    report.write_text(text+'\n'.join(lines))
    print(holding.to_string());print(aggregate.to_string())


if __name__=='__main__':main()
