"""Render the matched NN versus sized-policy experiment and verify accounting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_nn_rl_benchmark import MODELS, add_features
from scripts.run_first_pass import net_rewards, payoff_sharpe
from scripts.run_four_pairs import entry_cost, LADDER

LABELS={'rl_base_sized':'Original states, RL sizing','nn_both_fixed':'Both features, NN full-size/flat',
        'rl_both_fixed':'Both features, RL full-size/flat','rl_both_sized':'Both features, RL sizing'}


def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    root=Path('reports/nn_rl_benchmark');out=Path('artifacts/nn_rl_benchmark')
    scores=pd.read_csv(root/'scores.csv'); configs=pd.read_csv(root/'configs.csv');p=pd.read_csv(root/'portfolio.csv')
    contrasts=pd.read_csv(root/'contrasts.csv');yearly=pd.read_csv(root/'portfolio_annual.csv')
    for source in [Path('artifacts/universe_gap'),Path('artifacts/etf_exploration/run')]:
        packed=pd.read_parquet(source/'panels.parquet')
        for pair in packed.index.get_level_values('pair').unique():
            pred=pd.read_csv(out/f'{pair}.csv',index_col='date',parse_dates=['date'])
            panel=add_features(packed.xs(pair)).loc[pred.index]
            for model in MODELS:
                action=pred[model+'_action'].to_numpy()
                assert np.isin(action,LADDER if model.endswith('sized') else [-1.,0.,1.]).all()
                expected=net_rewards(action,panel.long_reward.to_numpy(),entry_cost(panel,.025))
                np.testing.assert_allclose(pred[model+'_reward'],expected,atol=1e-9)
                saved=scores[scores.pair.eq(pair)&scores.model.eq(model)].iloc[0]
                np.testing.assert_allclose(saved.sharpe,payoff_sharpe(expected),atol=1e-9)
                np.testing.assert_allclose(saved.net_pnl,expected.sum(),atol=1e-8)
    assert (pd.to_datetime(configs.train_last_outcome)<pd.to_datetime(configs.validation_first)).all()
    assert (pd.to_datetime(configs.validation_last_outcome)<pd.to_datetime(configs.test_first)).all()
    nn=configs[configs.model.eq('nn_both_fixed')]
    failed=nn[~nn.converged.eq(True)]
    wide=scores.pivot(index='pair',columns='model',values='sharpe')[MODELS]
    comparison=p.pivot(index='group',columns='model',values='sharpe')[MODELS]
    counts={m:int((wide.rl_both_sized>wide[m]).sum()) for m in MODELS if m!='rl_both_sized'}
    text=['# Simple NN versus RL sizing with both relative-value features','',
          'The requested benchmark compares a small supervised neural network choosing full-size long, flat, or short against the existing independent Sharpe-policy search using its 25% sizing ladder. Both receive the same seven inputs and use identical training, validation, test dates, and costs. Added features are inputs, not hard trading filters.','',
          f"The combined sized policy scores {comparison.loc['combined','rl_both_sized']:.3f} versus {comparison.loc['combined','nn_both_fixed']:.3f} for the NN and {comparison.loc['combined','rl_base_sized']:.3f} for the matched original-state sized reference. However, the new-feature full-size Sharpe policy scores {comparison.loc['combined','rl_both_fixed']:.3f}: sizing does not win on every basket. Excluding silver, the sized result is {comparison.loc['combined_without_silver','rl_both_sized']:.3f}; the remaining broad edge is modest.",'',
          '## Portfolio results','',
          table(comparison.reset_index(),['Basket']+[LABELS[m] for m in MODELS]),'',
          'Stock baskets span 2010–August 2025; ETF/stock and combined baskets span 2018–August 2025. All models within a basket use its common calendar. Portfolios use fixed equal pair budgets with unused allocation idle. Combined includes 12 stock pairs and 10 longer-history ETF/stock pairs. ETF/ETF comparisons and the two 2024-start pairs are reported individually below. All Sharpes are annualized 30-day proxy-payoff Sharpes, not daily marked option-and-hedge returns.','',
          '## Per-pair results','',table(wide.reset_index(),['Pair']+[LABELS[m] for m in MODELS]),'',
          'The sized policy with both features beats the NN in '+str(counts['nn_both_fixed'])+'/25 pairs, the original-state sized policy in '+str(counts['rl_base_sized'])+'/25, and the both-feature full-size policy in '+str(counts['rl_both_fixed'])+'/25. Counts include the short-history comparisons and are descriptive, not independent trials. Each pair uses its own available test history.','',
          '## Annual combined-basket Sharpe','',
          table(yearly[yearly.group.eq('combined')].pivot(index='year',columns='model',values='sharpe')[MODELS].reset_index(),['Year']+[LABELS[m] for m in MODELS]),'',
          'Excluding silver:', '',
          table(yearly[yearly.group.eq('combined_without_silver')].pivot(index='year',columns='model',values='sharpe')[MODELS].reset_index(),['Year']+[LABELS[m] for m in MODELS]),'',
          '2025 ends in August. The sized basket excluding silver loses in each of 2021–24, despite a positive full-period score. This is not a consistently profitable broad-universe result.','',
          '## Paired differences','',table(contrasts,['Basket','Left model','Right model','Sharpe difference','95% low','95% high']),'',
          'Intervals use 2,000 paired circular block resamples with blocks of 63 entry dates. They preserve cross-pair alignment within the basket, but do not account for model selection, repeated examination of history, or retraining uncertainty.','',
          '## Benchmark definitions','',
          '- **State:** IV spread, both capped earnings countdowns, both trailing 30-calendar-day RVs, `(trailing_RV_A-IV_A)-(trailing_RV_B-IV_B)`, and `(past_mean_spread-current_spread)/past_std_spread`. Spread moments use the prior 252 available panel observations, excluding today, with minimum 126. ETF countdowns remain constant 31. The RV feature is a transformation of existing inputs; the historical-spread feature adds history.',
          '- **NN:** one hidden layer of 16 tanh units; supervised mean-squared-error forecast of the next 30-day gross relative payoff. Features and target are standardized using training data only. L-BFGS training uses seed 2026, maximum 10,000 iterations / 200,000 function calls, tolerance 1e-4. Training fits two regularization strengths (0.01, 1.0); validation chooses among those and cost multipliers (1, 2, 4), for six candidates. Go long if predicted payoff exceeds multiplier × entry cost; short if below its negative; otherwise flat. Actions are strictly -1, 0, +1. There is no randomly held-out early-stopping sample.',
          '- **RL sizing:** the existing training-reward regression tree plus cross-entropy policy search maximizing annualized payoff Sharpe. This is direct historical policy search, not a neural actor-critic or TD learner. Tree candidates use 4/8/16 leaves × minimum 126/252 observations, selected by validation Sharpe. Five search seeds, 80 iterations, and population 256 are unchanged. Actions are -1 through +1 in 0.25 steps. Each sizing search is warm-started from the best training full-size policy.',
          '- **Sizing control:** the same new-feature tree/policy method with only -1/0/+1 actions. Its tree configuration is independently selected on validation. This helps distinguish sizing benefits from architecture/objective differences between the NN forecast and Sharpe policy. It is not the sized policy forcibly rounded to full size.',
          '- **Original-state reference:** fresh sized-policy fits with the five original inputs, on exactly the same feature-complete samples as the new models. The new historical feature removes 126 initial panel observations. SIL/HL\'s 2016 fold then falls below 504 training rows, so all four models start its test in 2018; other retained folds preserve their original test dates, which are asserted. Reference scores can therefore differ from previously saved experiments because initial training history changes.',
          '- **Validation:** expanding training, preceding two-year validation, following two-year test; forward 30-day outcomes are purged at boundaries. At least 504 training, 252 validation, and 126 test rows per fold. Test outcomes do not choose configurations, directions, sizes, or NN thresholds.',
          '- **Payoff/cost:** action × `[(forward_RV_A-IV_A)-(forward_RV_B-IV_B)]` minus `abs(action) × 0.025 × (IV_A+IV_B)/2`. Sharpe is `sqrt(365/30) × mean / sample_std`, retaining flat dates. Costs are paid per new entry; budgets are per cohort, not the sum of overlapping holdings.',
          '', '## Fit and data checks','',
          f'{len(nn)} NN selections across {scores.pair.nunique()} pairs; {len(failed)} selected fits reported an optimizer convergence warning. The NN iteration budget was increased from an initial 2,000-iteration run after inspecting training convergence, before comparing test performance. All NN fits were rerun under the final common settings, with tree-policy actions preserved. This is an optimization check, not a test-score-based choice of NN budget.','',
          'Training-sample sensitivity is material: the original-state stock basket is 0.145 here versus 0.317 in the prior run; AMD/NVDA is 0.369 versus 0.635. The model, search, test dates, and costs are unchanged, but the historical-feature warmup removes early training observations. Use the freshly matched reference to assess the new features; the sensitivity itself is a limitation of the current tree/policy approach.','',
          'Saved predictions were checked against the payoff formula and action menus for every pair/model. All training/validation outcome boundaries were verified. The benchmark tests also check NN cost thresholds and historical-feature causality.','',
          'The underlying universe and period have already been explored. Silver extreme-IV concentration and GSLC liquidity limitations remain; retraining does not validate execution or create a fresh holdout. See the [ETF input audit](etf_exploration.md) and [winner/loser review](winner_loser.md).','',
          '[Pair scores and exposure](nn_rl_benchmark/scores.csv) · [Annual pair results](nn_rl_benchmark/annual.csv) · [Annual portfolio results](nn_rl_benchmark/portfolio_annual.csv) · [Selected configurations](nn_rl_benchmark/configs.csv) · [All candidates](nn_rl_benchmark/candidates.csv) · [Paired differences](nn_rl_benchmark/contrasts.csv)','',
          '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_nn_rl_benchmark --workers 3 --resume','python -m scripts.report_nn_rl_benchmark','python -m unittest discover -s tests -q','```','',
          'Detailed predictions/checkpoints remain in ignored `artifacts/nn_rl_benchmark/`; no additional raw market data or reward columns are published in `data/`.']
    if len(failed):text.extend(['','Selected NN optimizer warnings:',table(failed[['pair','fold','iterations']],['Pair','Fold','Iterations'])])
    Path('reports/nn_rl_benchmark.md').write_text('\n'.join(text)+'\n')
    print(comparison.round(3).to_string()); print('Wins:',counts);print('Selected NN convergence warnings:',len(failed));print('All payoff/action/purge audits passed.')


if __name__=='__main__':main()
