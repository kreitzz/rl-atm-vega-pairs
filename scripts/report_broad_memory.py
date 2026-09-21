"""Audit and summarize the broad universe and lagged OOS-profit state."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_broad_memory import ROOT,OUT,REPORTS,MODELS,MEMORY,account,history_state
from scripts.run_four_pairs import LADDER
from scripts.report_nn_rl_benchmark import table
from scripts.run_first_pass import payoff_sharpe


def quality_checks(screen):
    common=pd.read_csv(OUT/'all_portfolio.csv',index_col=0,parse_dates=[0]).dropna(subset=MODELS).index
    total=pd.DataFrame(0.,index=common,columns=MODELS);skip=total.copy();contributions=[]
    for year,roster in screen.groupby('year'):
        dates=common[(common.year>=year)&(common.year<year+2)]
        for row in roster.itertuples():
            path=OUT/row.pair/f'{year}.csv'
            if not path.exists():continue
            p=pd.read_csv(path,index_col=0,parse_dates=[0]).reindex(dates)
            q=pd.read_parquet(OUT/row.pair/'panel.parquet').reindex(dates)
            extreme=q[['iv_amd','iv_nvda']].max(axis=1).gt(100)
            for m in MODELS:
                r=p[m+'_reward'].fillna(0.)/len(roster)
                total.loc[dates,m]+=r;skip.loc[dates,m]+=r.where(~extreme,0.)
                contributions.append(dict(pair=row.pair,label=row.label,year=year,model=m,net_pnl=r.sum(),extreme_net_pnl=r[extreme].sum(),max_iv=q[['iv_amd','iv_nvda']].max().max()))
    rows=[]
    for m in MODELS:
        rows.append(dict(model=m,baseline_sharpe=payoff_sharpe(total[m]),skip_iv_over_100_sharpe=payoff_sharpe(skip[m]),net_pnl=total[m].sum(),net_pnl_from_iv_over_100=(total[m]-skip[m]).sum(),extreme_profit_fraction=(total[m]-skip[m]).sum()/total[m].sum()))
    result=pd.DataFrame(rows);result.to_csv(REPORTS/'quality_sensitivity.csv',index=False)
    pd.DataFrame(contributions).to_csv(REPORTS/'pair_contributions.csv',index=False)
    return result


def main():
    scores=pd.read_csv(REPORTS/'scores.csv');coverage=pd.read_csv(REPORTS/'coverage.csv')
    cfg=pd.read_csv(REPORTS/'configs.csv');alloc=pd.read_csv(REPORTS/'allocations.csv')
    portfolios=pd.read_csv(REPORTS/'portfolio.csv');annual=pd.read_csv(REPORTS/'portfolio_annual.csv')
    contrasts=pd.read_csv(REPORTS/'contrasts.csv');screens=pd.read_csv(REPORTS/'candidate_screens.csv')
    scores.loc[scores.entries.lt(2),'sharpe']=np.nan
    scores.loc[scores.entries.eq(0),'net_pnl']=np.nan
    scores.to_csv(REPORTS/'scores.csv',index=False)
    alloc.loc[alloc.entries.lt(2),'sharpe']=np.nan
    alloc.to_csv(REPORTS/'allocations.csv',index=False)
    screen=screens[screens.year.ge(2018)];definitions={d['pair']:d for d in json.loads((ROOT/'pairs.json').read_text())}
    selected=cfg[cfg.model.ne('ineligible')]
    assert (pd.to_datetime(selected.train_last_outcome)<pd.to_datetime(selected.validation_first)).all()
    assert (pd.to_datetime(selected.validation_last_outcome)<pd.to_datetime(selected.test_first)).all()
    audit=[];annual_counts={}
    for key in scores.pair.unique():
        panel=pd.read_parquet(OUT/key/'panel.parquet');reference=pd.read_parquet(OUT/key/'reference.parquet')
        assert (pd.to_datetime(reference.fit_cutoff)<=reference.index).all()
        pd.testing.assert_frame_equal(panel[MEMORY],history_state(reference,panel.index))
        for year in screen.loc[screen.pair.eq(key),'year']:
            path=OUT/key/f'{year}.csv'
            if not path.exists():continue
            pred=pd.read_csv(path,index_col=0,parse_dates=[0]);q=panel.loc[pred.index]
            valid=pred[[m+'_reward' for m in MODELS]].notna().all(axis=1)
            for calendar_year,g in pred.loc[valid].groupby(pred.loc[valid].index.year):annual_counts[(key,calendar_year)]=len(g)
            for m in MODELS:
                a=pred[m+'_action'].to_numpy()
                assert np.isin(a,[-1.,0.,1.] if m in ['nn_memory','rl_memory_fixed'] else LADDER).all()
                np.testing.assert_allclose(pred[m+'_reward'],account(a,q),atol=1e-8,equal_nan=True)
                extreme=q[['iv_amd','iv_nvda']].max(axis=1).gt(100)
                audit.append(dict(pair=key,year=year,model=m,extreme_dates=int(extreme.sum()),entries=len(q),net_pnl=pred[m+'_reward'].sum(),extreme_net_pnl=pred.loc[extreme,m+'_reward'].sum()))
    pd.DataFrame(audit).to_csv(REPORTS/'extreme_iv_attribution.csv',index=False)
    pair_annual=pd.read_csv(REPORTS/'annual.csv')
    pair_annual['entries']=[annual_counts.get((r.pair,r.year),0) for r in pair_annual.itertuples()]
    pair_annual.loc[pair_annual.entries.lt(2),'sharpe']=np.nan
    pair_annual.to_csv(REPORTS/'annual.csv',index=False)
    behavior=[]
    for (model,state),g in alloc.groupby(['model','state']):
        weights=g.entries
        behavior.append(dict(model=model,state=state,pair_entries=int(weights.sum()),trade_fraction=np.average(g.trade_fraction,weights=weights),mean_size=np.average(g.mean_size,weights=weights)))
    behavior=pd.DataFrame(behavior);behavior.to_csv(REPORTS/'behavior.csv',index=False)
    within=[]
    for model,g in alloc.groupby('model'):
        p=g.pivot(index='pair',columns='state',values='mean_size')
        d=(p.positive_history-p.negative_history).dropna()
        within.append(dict(model=model,pairs=len(d),pairs_larger_after_positive=int(d.gt(0).sum()),median_positive_minus_negative_size=d.median(),mean_positive_minus_negative_size=d.mean()))
    within=pd.DataFrame(within);within.to_csv(REPORTS/'within_pair_behavior.csv',index=False)
    by_pair=scores.pivot(index='pair',columns='model',values='sharpe')
    eligible=scores[scores.entries.ge(126)].pair.unique()
    winners=by_pair.loc[eligible,'rl_memory'].sort_values(ascending=False).head(10).index
    leaders=scores[scores.pair.isin(winners)].pivot(index=['pair','label','entries'],columns='model',values='sharpe').reset_index().drop(columns='pair')
    leaders=leaders.sort_values('rl_memory',ascending=False)
    nn=selected[selected.model.eq('nn_memory')];warnings=nn[~nn.converged.eq(True)]
    quality=quality_checks(screen)
    complete=coverage[coverage.group.eq('all')]
    used=int(complete.joint_complete_dates.sum());available=int(complete.calendar_dates.sum())
    text=['# Broad universe: does prior profitability help RL allocate its pair budget?','',
          'The policy, rather than an external profit-ranking rule, decides whether each pair deserves a trade and how much of its fixed budget to use. The new state summarizes completed out-of-sample trades from a reference RL strategy for that same pair. A poorly performing history can lead to flat or smaller positions, but no such behavior is imposed.','',
          f'The input-only historical screens produce {screens.pair.nunique()} distinct pairs across 2008–24. The 2018–August 2025 experiment considers {screen.pair.nunique()} unique pairs in {len(screen)} pair/fold combinations; {scores.pair.nunique()} pairs have at least one eligible fitted evaluation. The candidate roster changes every two years.','',
          '**Finding:** the added profit-history state does not improve the broad basket: sized RL falls from 2.788 to 2.606 annualized proxy Sharpe. Its change interval includes zero. Nor does it consistently allocate more after profits: mean size is about 42.0% after positive history versus 42.4% after negative history; only 101 of 230 pairs observed in both regimes have larger average size after profits. These are descriptive associations, not proof that historical profitability cannot be useful.','',
          '## Matched annualized proxy Sharpe','',
          'See the [trade-quality and profit-concentration audit](trade_quality.md) for leg attribution, direction controls, and concentration sensitivities.','',
          table(portfolios.pivot(index='group',columns='model',values='sharpe')[MODELS].reset_index(),['Basket','RL without profit state','NN full-size/flat + profit state','RL full-size/flat + profit state','RL sizing + profit state']),'',
          '`without_metals` is a concentration sensitivity excluding SIC groups 104, 106, 109 and SIL/stock pairs; it is not an optimized pair-selection rule or an exclusion of every mining business. Pair budgets are equal within each fold. Insufficient prior model history and unfilled allocation remain cash, with no reweighting to the remaining pairs.','',
          '## Extreme-mark sensitivity','',
          table(quality,['Model','Baseline Sharpe','Skip IV >100 Sharpe','Net P&L','Net P&L from IV >100','Extreme-date profit fraction']),'',
          'This diagnostic suppresses an entry if either leg\'s standardized IV exceeds 100 vol, retains idle budget, and uses the same 1,805 complete-outcome dates. It is not a newly adopted or independently validated rule. High IV can be real; this flags reliance on extreme marks for actual-contract quote checks. Contributions use the actual per-fold equal pair budgets. [Weighted pair contributions](broad_pairs/pair_contributions.csv).','',
          '## Does RL reduce exposure after losses?','',
          'Positive/negative history means the reference strategy\'s trailing mean net payoff is above/below zero, with at least 126 completed entry cohorts. Exposure is absolute size, not direction. The entry-weighted table describes behavior; differences can also reflect other changing state variables.','',
          table(behavior,['Model','History state','Pair entries','Trade fraction','Mean absolute size']),'',
          'Within-pair comparisons reduce differences caused solely by which names populate the positive and negative groups:', '',
          table(within,['Model','Pairs with both states','Pairs sizing more after profits','Median size difference','Mean size difference']),'',
          '## Annual all-candidate results','',
          table(annual[annual.group.eq('all')].pivot(index='year',columns='model',values='sharpe')[MODELS].reset_index(),['Year','RL without profit state','NN full-size/flat','RL full-size/flat','RL sizing']),'',
          '2025 is partial through August. Sharpe uses sqrt(365/30) on net entry-cohort proxy payoffs, including flat entries. It is not the Sharpe of a daily marked option portfolio.','',
          '## Paired Sharpe differences','',table(contrasts,['Basket','Compared with','RL-memory Sharpe change','95% low','95% high']),'',
          'Intervals use 2,000 paired circular block resamples of 63 entry dates. They do not correct for model selection, repeated research on this history, changing universe construction, or missing outcomes.','',
          '## Highest full-period fitted scores, for diagnosis only','',
          table(leaders[['label','entries']+MODELS],['Pair','Entries','RL without profit state','NN full-size/flat','RL full-size/flat','RL sizing']),'',
          'This retrospective table is not the universe selector and requires at least 126 scored entry dates for display. Different pairs enter different folds and have different sample lengths. The model does not receive full-period profitability or the list above. Per-pair and annual Sharpes with fewer than two scored observations are left undefined in the CSVs.','',
          '## Universe and timing','',
          '- At each two-year boundary, use the preceding 252 available return sessions. Require at least 200 complete observations of IV, return, trailing RV, earnings state, and call activity; median aggregate call volume at least 100 and median call open interest at least 500.',
          '- Use the latest security-name SIC classification effective strictly before that boundary. Within each three-digit SIC group, choose up to four names by historical median call open interest, deduplicating CRSP identifiers. Form all pairs whose matched trailing return correlation is at least 0.4. Cap at 180 stock pairs by rotating across industries, with higher minimum-leg open interest first within each industry. Add previously specified ETF comparisons only when they pass the same input screen. No profit measure enters this screen.',
          '- CRSP returns are restricted to their OptionMetrics link intervals. Securities with different permanent return identifiers are not spliced. Historical candidates include acquired/delisted names; ticker labels in aggregated pair tables are display labels and can differ across historical screens.',
          '- This remains the locally downloaded universe, not a complete point-in-time U.S. options universe. Current security-type metadata and source-history/earnings coverage influence which assets are available. At preparation, an asset needs 504 IV observations and 20 recorded stock earnings dates. These availability constraints and non-point-in-time earnings-calendar assumptions remain limitations; historical screening does not eliminate all selection bias.',
          '', '## Profitability state and benchmark','',
          '- Reference RL: the seven IV/RV/value inputs from the prior benchmark; a fixed eight-leaf tree, minimum 126 rows; the usual five-seed, 80-iteration, 256-population Sharpe search and 25% sizing ladder. It refits every two years from 2006 using only outcomes completed before the fit date. No in-sample fitted trade rewards enter the profitability state.',
          '- For each new entry, collect the last 252 reference trade cohorts whose 30-day end date is strictly earlier than entry. Four inputs are their mean net payoff, annualized payoff Sharpe, active-trade fraction, and observation count divided by 252. Flat reference trades contribute zero. Unknown history has zero values plus count zero. A zero variance history has Sharpe input zero.',
          '- These are shadow trades from a common reference RL policy, not endogenous feedback from each newly augmented policy\'s own changing decisions. The augmented RL and NN both observe this same historical track record. Modeling feedback from the newly learned policy itself would require a different sequential training setup.',
          '- Each candidate is trained independently. Compare a seven-input sized RL policy, an eleven-input sized RL policy, its independently validation-selected full-size/flat control, and a full-size/flat NN receiving the same eleven inputs. No joint allocation policy or cross-pair profit-ranking gate is introduced.',
          '- Training expands; the two preceding years are used for model validation, followed by two test years. Thirty-day outcomes are purged at both boundaries. Eligibility requires 504 prior training and 252 prior validation rows, with no minimum future test-length requirement.',
          '- RL uses the established six tree configurations and training-seed selection. The NN has one 16-unit tanh layer, training-only feature/target scaling, L-BFGS maximum 10,000 iterations, regularization 0.01/1, and validation-chosen entry-cost thresholds 1/2/4. NN actions are strictly -1/0/+1; sized RL actions range from -1 to +1 in 0.25 increments.',
          '- Pair payoff and costs are unchanged: action times the difference of forward RV-minus-entry-IV across legs, less absolute size times 2.5% of average entry IV. Initial vega neutrality, daily delta hedging, standardized IV marks, and fixed cohort entry budgets remain assumptions.',
          '', '## Missing outcomes and checks','',
          f'The full screened calendar has {available} entry dates; {used} have jointly complete portfolio outcomes across all four models. {available-used} dates are excluded from all comparisons because at least one active candidate trade has no complete forward RV outcome. These missing payoffs are not set to zero. This complete-outcome analysis can still be biased, particularly around delistings or corporate actions, and is not an executable portfolio backtest.','',
          table(coverage,['Basket','Fold','Candidate pairs','Fitted pairs','Calendar dates','Joint complete dates','Unresolved pair dates']),'',
          f'{len(warnings)} of {len(nn)} selected NN fits report an optimizer convergence warning. All saved action menus, net reward calculations, reference-history states, and model purge boundaries were audited. Tests cover forward-RV equivalence, strict maturity timing, future-outcome independence, and preservation of unknown active-trade payoffs.','',
          'Aggregate activity is across all call strikes/expiries; it does not verify executable ATM 30-day spreads. Extremely high IV marks can drive the proxy. [Extreme-IV contribution](broad_pairs/extreme_iv_attribution.csv) is descriptive, not a new trading rule.','',
          '[Candidate screens](broad_pairs/candidate_screens.csv) · [Pair scores](broad_pairs/scores.csv) · [Annual results](broad_pairs/portfolio_annual.csv) · [Exposure by past-profit state](broad_pairs/behavior.csv) · [Within-pair exposure differences](broad_pairs/within_pair_behavior.csv) · [Selected model configurations](broad_pairs/configs.csv)','',
          '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.build_broad_pairs --source-root /path/to/wrds','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_broad_memory --workers 4 --resume','python -m scripts.report_broad_memory','python -m unittest discover -s tests -q','```','',
          'Market panels, reference trades, and predictions stay under ignored `artifacts/broad_pairs/`. The shared market-data CSV is unchanged.']
    Path('reports/broad_pairs.md').write_text('\n'.join(text)+'\n')
    print(portfolios.to_string(index=False));print(behavior.to_string(index=False));print(within.to_string(index=False));print(quality.to_string(index=False));print('Matched complete calendar:',used,'/',available)


if __name__=='__main__':main()
