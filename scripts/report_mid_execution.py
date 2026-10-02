"""Compare quoted execution, fixed-trade repricing and a retrained mid model."""
import json
from pathlib import Path
import pandas as pd


def main():
    old=pd.read_csv('reports/contract_walkforward/portfolio.csv').set_index('model')
    sensitivity=pd.read_csv('reports/mid_execution_sensitivity/portfolio.csv').set_index(['model','penalty_vol_points'])
    costs=pd.read_csv('reports/mid_execution_sensitivity/pairs.csv')
    new=pd.read_csv('reports/contract_walkforward_mid_0p1/portfolio.csv').set_index('model')
    manifest=json.loads(Path('reports/contract_walkforward_mid_0p1/manifest.json').read_text())
    names={'rl_annual_base':'Annual RL, base states','rl_monthly_base':'Monthly RL, base states',
           'rl_monthly_memory':'Monthly RL + memory','rl_monthly_beta':'Monthly RL + beta',
           'rl_monthly_combined':'Monthly RL + memory + beta','nn_monthly_combined':'Fixed-size NN + memory + beta'}
    rows=[]
    for model,name in names.items():
        rows.append(dict(model=model,name=name,old_quoted_sharpe=old.loc[model,'annualized_excess_sharpe'],
                         fixed_mid_sharpe=sensitivity.loc[(model,0.),'annualized_excess_sharpe'],
                         fixed_mid_0p1_sharpe=sensitivity.loc[(model,.1),'annualized_excess_sharpe'],
                         retrained_mid_0p1_sharpe=new.loc[model,'annualized_excess_sharpe'] if new.loc[model,'status']=='complete' else float('nan'),
                         retrained_total_return=new.loc[model,'total_return'] if new.loc[model,'status']=='complete' else float('nan')))
    comparison=pd.DataFrame(rows);comparison.to_csv('reports/mid_execution_sensitivity/comparison.csv',index=False)
    lines=['# Mid-theoretical execution with explicit penalties','',
           '2013–August 2025, the same four pairs. Mid is the valuation and execution reference. A separate cash penalty replaces the quoted-spread charge:', '',
           '`execution penalty = abs(contracts) × 100 × vega per share per vol point × penalty in vol points`','',
           'The central assumption is **0.10 vol points per leg per fill**. Two $450-vega legs cost $90 to open, before commissions; closing uses the remaining vega. This is a research fill assumption, not a calibrated estimate of achievable execution. The 0/0.05/0.10/0.25 grid measures sensitivity.', '',
           'Commissions remain $0.65 per contract per execution; stock hedging costs 1 bp, with the existing financing and borrow assumptions. “Mid” below still includes those costs. Theoretical option values and Greeks continue to use the selected contracts and the existing BS fits. A forced exit without a current valid vega uses the last observed, split-adjusted vega; there is no future Greek fill.', '',
           '| Model | Old quoted cost | Fixed trades at mid | Fixed trades, mid + 0.10 | Retrained, mid + 0.10 |',
           '|---|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        lines.append(f'| {row.name} | {row.old_quoted_sharpe:.3f} | {row.fixed_mid_sharpe:.3f} | {row.fixed_mid_0p1_sharpe:.3f} | {row.retrained_mid_0p1_sharpe:.3f} |')
    lines+=['', 'All entries are annualized **daily excess-return Sharpe**. Old execution paid 25% of the full quoted spread per fill. Fixed-trade repricing preserves every contract, quantity, hedge and exit, and recomputes cash financing. It isolates the effect of costs; it does not choose new trades or resize.', '',
            'The retrained run rebuilds net actual-contract labels and profitability memory under the new penalty, retrains the policies, and reruns the original NAV sizing and portfolio risk limits. Historical quote-quality filters and contract selection are unchanged. No penalty was chosen by optimizing the test-period Sharpe.', '',
            '[Retrained returns, Sharpe and annual tables](contract_walkforward_mid_0p1.md) · [Comparison CSV](mid_execution_sensitivity/comparison.csv) · [All fixed-trade sensitivities](mid_execution_sensitivity/portfolio.csv) · [Cost breakdown](mid_execution_sensitivity/pairs.csv)', '',
            '## Fixed-trade cost curve', '',
            '| Penalty per leg/fill | Annual RL baseline Sharpe | Monthly RL + both Sharpe | NN + both Sharpe |',
            '|---:|---:|---:|---:|']
    for p in [0.,.05,.1,.25]:
        values=[sensitivity.loc[(m,p),'annualized_excess_sharpe'] for m in ['rl_annual_base','rl_monthly_combined','nn_monthly_combined']]
        lines.append(f'| {p:.2f} vol points | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |')
    central=costs[costs.penalty_vol_points.eq(.1)].groupby('model')[['old_option_slippage','new_option_penalty','option_commission','stock_execution']].sum()
    c=central.loc['rl_annual_base']
    lines += ['',f"For the annual RL baseline's original trades across all four accounts, quoted-spread slippage totaled ${c.old_option_slippage:,.0f}. The 0.10-vol-point penalty totals ${c.new_option_penalty:,.0f}, with commissions of ${c.option_commission:,.0f} retained separately.", '',
              f"The fixed-inventory 0.25-vol-point sensitivity has {int(costs.fixed_inventory_risk_breach_days.sum())} pair/model/session risk-limit breaches in total; that sensitivity is not a fresh risk-managed simulation. The 0/0.05/0.10 cases have {int(costs[costs.penalty_vol_points.le(.1)].fixed_inventory_risk_breach_days.sum())}. Retrained completed simulations independently enforce every limit.", '',
              f"Peak memory for retraining/evaluation: {manifest['peak_rss_bytes']/1024**3:.2f} GiB. Accounting, penalty units, zero-penalty commissions, missing-Greek fallback, borrowing and pre-trade risk-cost checks are tested. Sixteen real-quote shadow-label comparisons, including split dates, matched the cash-account simulator.", '',
              '## Reproduce', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.reprice_contract_execution',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_contract_walkforward --option-penalty-vol-points 0.1',
              'python -m scripts.report_contract_walkforward --root artifacts/contracts/walkforward_mid_0p1 --report reports/contract_walkforward_mid_0p1',
              'python -m scripts.report_mid_execution', '```','']
    Path('reports/mid_execution.md').write_text('\n'.join(lines))
    print(comparison.to_string(index=False))


if __name__=='__main__':main()
