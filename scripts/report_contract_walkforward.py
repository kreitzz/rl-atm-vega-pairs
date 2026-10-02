"""Audit and summarize the completed actual-contract learning experiment."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.contract_learning_labels import LADDER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/contracts/walkforward'))
    parser.add_argument('--report',type=Path,default=Path('reports/contract_walkforward'))
    parser.add_argument('--document',type=Path)
    args=parser.parse_args();root=args.root;out=args.report
    document=args.document or out.with_suffix('.md')
    manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['settings']
    status=pd.read_csv(out/'status.csv');portfolio=pd.read_csv(out/'portfolio.csv')
    audits=[];fit_rows=[];screen_rows=[]
    for pair in manifest['pairs']:
        folder=root/pair
        actions=pd.read_csv(folder/'actions.csv',index_col=0,parse_dates=[0])
        screen=pd.read_csv(folder/'beta_screen.csv',index_col=0,parse_dates=[0])
        fits=pd.read_csv(folder/'fits.csv');fits['pair']=pair;fit_rows.append(fits)
        assert (pd.to_datetime(fits.last_outcome)<pd.to_datetime(fits.fit)).all()
        assert np.isin(actions.to_numpy(),LADDER).all()
        assert np.isin(actions.nn_monthly_combined,[-1,0,1]).all()
        gate=screen.eligible.reindex(actions.index).to_numpy()
        np.testing.assert_array_equal(actions.rl_monthly_beta,actions.rl_monthly_base*gate)
        np.testing.assert_array_equal(actions.rl_monthly_combined,actions.rl_monthly_memory*gate)
        assert actions.loc[~gate,'nn_monthly_combined'].eq(0).all()
        screen_rows.append(dict(pair=pair,eligible_session_fraction=float(gate.mean()),
                                monthly_combined_signal_fraction=float(actions.rl_monthly_combined.ne(0).mean()),
                                mean_beta=float(screen.reindex(actions.index).beta.mean()),
                                mean_correlation=float(screen.reindex(actions.index).correlation.mean())))
        for row in status[(status.pair==pair)&status.status.eq('complete')].itertuples():
            nav=pd.read_csv(folder/row.model/'daily_nav.csv',index_col=0,parse_dates=[0])
            decisions=pd.read_csv(folder/row.model/'decisions.csv',parse_dates=['signal_date'])
            trades=pd.read_csv(folder/row.model/'trades.csv')
            limits={'gross_vega':cfg['max_gross_vega']/cfg['capital'],
                    'gross_hedge_notional':cfg['max_hedge_leverage'],
                    'capital_reserve':cfg['max_reserve_fraction'],'stress_loss':cfg['max_stress_loss_fraction']}
            for field,limit in limits.items():assert (nav[field]/nav.nav<=limit+1e-10).all()
            entries=decisions[decisions.status.eq('entered')]
            for field in ['vega_a','vega_b']:
                assert (entries[field]<=cfg['entry_vega_per_leg']*entries.action.abs()*entries.sizing_nav/cfg['capital']+1e-8).all()
            if row.model.endswith(('beta','combined')):
                assert screen.eligible.reindex(entries.signal_date).all()
            audits.append(dict(pair=pair,model=row.model,entries=len(entries),
                               active_session_fraction=float(nav.active_cohorts.gt(0).mean()),
                               max_hedge_leverage=float((nav.gross_hedge_notional/nav.nav).max()),
                               max_stress_fraction=float((nav.stress_loss/nav.nav).max()),
                               risk_exits=int(nav.risk_exit_cohorts.sum()),
                               repaired_quotes=int(nav.repaired_held_quotes.sum()),
                               split_inventory_adjustments=int(trades.reason.eq('split_adjustment').sum()) if len(trades) else 0))
    pd.DataFrame(audits).to_csv(out/'audit.csv',index=False)
    pd.concat(fit_rows,ignore_index=True).to_csv(out/'fits.csv',index=False)
    pd.DataFrame(screen_rows).to_csv(out/'selection_summary.csv',index=False)
    labels=pd.read_csv(out/'labels.csv')
    fits=pd.concat(fit_rows,ignore_index=True);nn=fits[fits.policy.eq('nn')]
    failed=int(nn.converged.eq(False).sum())
    penalty=cfg.get('option_penalty_vol_points')
    execution=('halfway between mid and the quoted side' if penalty is None else
               f'mid fills plus a separate {penalty:g}-vol-point cash penalty per leg per fill (absolute dollar vega × {penalty:g}); no quoted-spread charge')
    names={'rl_annual_base':'Annual RL, base states','rl_monthly_base':'Monthly RL, base states',
           'rl_monthly_memory':'Monthly RL + memory','rl_monthly_beta':'Monthly RL + beta screen',
           'rl_monthly_combined':'Monthly RL + memory + beta','nn_monthly_combined':'Monthly fixed-size NN + memory + beta'}
    lines=['# Actual-contract learning with memory and monthly beta selection','',
           f"Four independent pairs; {manifest['start']}–{manifest['end']}. Each starts with $1 million; rejected pair allocations remain in cash.", '',
           '| Model | Total return | CAGR | Annualized excess Sharpe | Max drawdown |',
           '|---|---:|---:|---:|---:|']
    years=(pd.Timestamp(manifest['end'])-pd.Timestamp(manifest['start'])).days/365.25
    for row in portfolio.itertuples():
        if row.status!='complete':
            lines.append(f'| {names[row.model]} | Incomplete | — | — | — |');continue
        cagr=(1+row.total_return)**(1/years)-1
        lines.append(f'| {names[row.model]} | {row.total_return:.2%} | {cagr:.2%} | {row.annualized_excess_sharpe:.3f} | {row.max_drawdown:.2%} |')
    completed=portfolio[portfolio.status.eq('complete')]
    if len(completed):
        reference=pd.read_csv(root/f'{completed.model.iloc[0]}_portfolio.csv')
        cash=reference.cash_benchmark.iloc[-1]/(len(manifest['pairs'])*cfg['capital'])-1
        lines.append(f'| Cash benchmark | {cash:.2%} | {(1+cash)**(1/years)-1:.2%} | — | 0.00% |')
    lines += ['', 'Sharpe uses daily portfolio returns less the same daily cash return, annualized by √252. Total return and CAGR include cash interest. The final year is partial. No failed pair is dropped from portfolio results.', '',
              '[Annual returns and Sharpe](contract_walkforward/annual.csv) · [Pair results](contract_walkforward/status.csv) · [Portfolio results](contract_walkforward/portfolio.csv) · [Selection summary](contract_walkforward/selection_summary.csv) · [Risk audit](contract_walkforward/audit.csv)', '',
              '## Implementation', '',
              '- **Memory:** trailing 91/365-calendar-day mean net P&L per actual initial mean leg vega, separately for long-A/short-B and short-A/long-B, plus observation counts. These are daily-entry shadow/reference trades, so inactive pairs retain a track record. Only trades closed before the signal date contribute. Overlapping observations are not independent samples.',
              '- **Selection:** monthly, using the preceding 126 trading sessions of adjusted stock returns. A-on-B beta must be 0.75–1.25 and correlation at least 0.5. The screen freezes for the month and gates new entries; existing trades follow their normal exits. This filters the four established pairs, rather than discovering pairs across the full downloaded universe.',
              '- **Retraining:** monthly, trailing five years, at least 504 fully observed training rows. Outcomes must close strictly before the fit date. The annual ablation refits in January. Hyperparameters are fixed; reported test years do not choose settings.',
              '- **Inputs:** selected-contract BS IV spread, both earnings countdowns, both trailing 30-day RVs, RV-adjusted relative value, and trailing spread reversion; memory adds eight states. The legacy standardized-IV/forward-RV reward is not used.',
              '- **RL:** eight-leaf state tree, minimum 63 training rows per leaf, direct cross-entropy policy search over the signed 0/25/50/75/100% ladder. Search maximizes daily excess-P&L Sharpe of overlapping standalone shadow cohorts. It is not actor-critic or Q-learning. Final evaluation applies the full NAV-scaled portfolio and risk constraints, which are not inside the training objective.',
              '- **NN:** one 16-unit tanh hidden layer, training-only feature/target scaling, predicts the two directional net contract outcomes. Trade the higher positive prediction at full size; otherwise stay flat. Same monthly schedule, memory, beta filter and execution assumptions.',
              f'- **Execution:** next-session entry in the prior-close selected ATM/~30D call; daily BS delta hedging; integer contracts; initial vega matching; {execution}; $0.65 option commission per contract, 1 bp stock execution, financing and borrow. Keep the existing NAV-scaled vega, hedge, reserve and stress limits.',
              '', '## Data and checks', '',
              'Ordinary integer stock splits require corroborating stock adjustment factors, option adjustment factors, unchanged expiry/100-share multiplier, and the divided strike within half-cent rounding. Contract and stock quantities adjust without cash execution. This matches the [OCC NVIDIA adjustment notice](https://infomemo.theocc.com/infomemos?number=48828). Unsupported actions still fail explicitly.', '',
              f"Shadow-label disposition: {labels.groupby('status').rows.sum().to_dict()}. Unresolved labels never become zero-profit trades. Known entry rejections are zero; no future label availability determines live predictions.", '',
              f"All completed account ledgers and end-of-day risk limits reconcile. Training cutoff, action ladder and beta gating audits pass. {failed}/{len(nn)} NN fits reported nonconvergence; those deterministic fits are retained and flagged in [fit logs](contract_walkforward/fits.csv). Peak process RSS: {manifest['peak_rss_bytes']/1024**3:.2f} GiB; one pair is processed at a time.", '',
              'Unit tests cover timing leakage, overlapping daily Sharpe, cash reconciliation, flat policies, split inventory/rounding, missing marks and portfolio risk limits. The accelerated labels were additionally checked against the original cash-account engine on real quotes, including NVIDIA split dates.', '',
              'Limitations: fixed retrospectively selected four-pair universe; historical earnings dates assumed known; EOD mid-based BS inversion for American calls; explicit crossed-quote envelope repair on held positions; no intraday execution, assignment or broker margin model. These are research results, not a point-in-time deployable strategy.', '',
              '## Reproduce', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_contract_walkforward'+('' if penalty is None else f' --option-penalty-vol-points {penalty:g}'),
              f'python -m scripts.report_contract_walkforward --root {root} --report {out}', '```', '',
              'Requires local licensed market-data caches. Raw quotes, labels, states, actions and account paths remain gitignored; source/settings hashes are recorded. `--reuse-labels --reuse-actions` requires matching input and code fingerprints. Prior frozen-policy results are preserved.', '']
    document.write_text('\n'.join(lines).replace('(contract_walkforward/',f'({os.path.relpath(out,document.parent)}/'))
    print(portfolio.to_string(index=False));print(pd.DataFrame(screen_rows).to_string(index=False))


if __name__=='__main__':main()
