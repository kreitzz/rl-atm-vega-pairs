"""Report frozen additive-sizing results; no fitting, selection, or replay.

Run only after all five funded cases and the diagnostics have been written.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scripts.diagnose_p25_pca import Budget, digest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/p25_additive_sizing'
ARTIFACTS = ROOT/'artifacts/contracts/p25_additive_sizing'
CASES = ['tree_cem', 'fixed100', 'linear_score', 'additive_main', 'additive_pairs']
NEW = ['linear_score', 'additive_main', 'additive_pairs']
NAMES = {'tree_cem':'Original tree + CEM', 'fixed100':'Fixed 100%',
         'linear_score':'Linear score', 'additive_main':'Additive main effects',
         'additive_pairs':'Additive + 3 interactions'}
COLORS = {'tree_cem':'#566573', 'fixed100':'#9c9c9c', 'linear_score':'#3579aa',
          'additive_main':'#16826c', 'additive_pairs':'#ae6b24'}
FEATURE_NAMES = {'intercept':'Intercept', 'rank_strength':'Spread rank strength',
    'correlation':'Pair correlation', 'vix':'VIX', 'profit_365d':'Past profit, 365 days',
    'count_365d':'Observations, 365 days', 'profit_91d':'Past profit, 91 days',
    'count_91d':'Observations, 91 days', 'iv_mean':'Mean implied volatility',
    'rv_mean':'Mean realized volatility', 'oriented_iv_gap':'Oriented IV gap',
    'oriented_vrp_gap':'Oriented volatility-premium gap', 'beta':'Pair beta',
    'commission_vol_points':'Commission in volatility points', 'dte_mean':'Mean days to expiry',
    'oriented_dte_gap':'Oriented expiry gap'}


def read_tables(report):
    paths = {'periods':'funded/periods.csv', 'annual':'funded/annual.csv',
             'intervals':'funded/sharpe_intervals.csv', 'matched':'matched_intervals.csv',
             'fits':'fits.csv', 'perturbation':'perturbation.csv', 'refit':'refit_stability.csv'}
    frames = {name:pd.read_csv(report/path) for name,path in paths.items()}
    periods = frames['periods']
    fits=frames['fits']
    for case in NEW:
        own=fits.loc[fits.case.eq(case)]
        if len(own)!=90 or own.fit.nunique()!=90:
            raise ValueError(f'{case}: expected90 unique monthly fits')
        if 'latest_train_end' in own and not (pd.to_datetime(own.latest_train_end)<pd.to_datetime(own.fit)).all():
            raise ValueError(f'{case}: training outcomes do not precede fitting')
    for case in CASES:
        for period in ['validation','evaluation','full']:
            row = periods.loc[periods.case.eq(case)&periods.period.eq(period)]
            if len(row)!=1:
                raise ValueError(f'Missing or duplicate funded result: {case}, {period}')
            if not np.isfinite(row[['annualized_excess_sharpe','profit_above_cash','entries']]).all().all():
                raise ValueError('Nonfinite funded results')
        if int(periods.loc[periods.case.eq(case)&periods.period.eq('full'),'proposals'].iloc[0])!=608:
            raise ValueError(f'{case}: expected the same608 proposals')
    for key in ['intervals','matched']:
        required = {'case','reference','period','block_sessions','point','lower95','upper95'}
        if not required.issubset(frames[key]):
            raise ValueError(f'{key}: missing interval fields {required-set(frames[key])}')
        if (frames[key].lower95>frames[key].upper95).any():
            raise ValueError('Reversed interval endpoints')
    for candidate,reference in [('additive_main','linear_score'),('additive_pairs','additive_main')]:
        for period in ['validation','evaluation']:
            subset = frames['matched'].loc[frames['matched'].case.eq(candidate)&frames['matched'].reference.eq(reference)&frames['matched'].period.eq(period)]
            if len(subset)!=2 or set(subset.block_sessions)!={63,126}:
                raise ValueError(f'Missing matched comparison {candidate} vs {reference}, {period}')
            expected = (periods.loc[periods.case.eq(candidate)&periods.period.eq(period),'annualized_excess_sharpe'].iloc[0]
                        -periods.loc[periods.case.eq(reference)&periods.period.eq(period),'annualized_excess_sharpe'].iloc[0])
            if not np.allclose(subset.point,expected,atol=1e-10,rtol=0):
                raise ValueError('Interval point estimate differs from funded Sharpe difference')
    return frames, {str(report/path):digest(report/path) for path in paths.values()}


def summarize_stability(frame, denominator, grouping):
    required = set(grouping)|{denominator,'changed','total_abs_fraction_change'}
    if not required.issubset(frame):
        raise ValueError(f'Missing stability fields: {required-set(frame)}')
    if (frame[denominator]<0).any() or (frame.changed<0).any() or (frame.changed>frame[denominator]).any():
        raise ValueError('Invalid stability observation counts')
    if (frame.total_abs_fraction_change < -1e-12).any() or (frame.total_abs_fraction_change > .5*frame[denominator]+1e-9).any():
        raise ValueError('Size changes exceed the50–100% ladder range')
    result = frame.groupby(grouping,as_index=False).agg(observations=(denominator,'sum'),
        changed=('changed','sum'),total_abs_fraction_change=('total_abs_fraction_change','sum'))
    result['changed_fraction'] = np.divide(result.changed,result.observations,
        out=np.full(len(result),np.nan),where=result.observations.gt(0))
    result['mean_absolute_size_change'] = np.divide(result.total_abs_fraction_change,result.observations,
        out=np.full(len(result),np.nan),where=result.observations.gt(0))
    return result


def conclusion(intervals):
    if len(intervals)!=2 or set(intervals.block_sessions)!={63,126}:
        raise ValueError('Conclusion requires both63- and126-session blocks')
    if (intervals.lower95>0).all():
        return 'Both conditional intervals favor the candidate.'
    if (intervals.upper95<0).all():
        return 'Both conditional intervals favor the reference.'
    if ((intervals.lower95<=0)&(intervals.upper95>=0)).all():
        return 'Both intervals include zero; the difference remains uncertain.'
    return 'The interval conclusion changes with block length.'


def interval_lines(frame,candidate,reference,period):
    own = frame.loc[frame.case.eq(candidate)&frame.reference.eq(reference)&frame.period.eq(period)].sort_values('block_sessions')
    wording = conclusion(own)
    lines=[f'{NAMES[candidate]} minus {NAMES[reference]}, {period}: {wording}']
    for row in own.itertuples():
        lines.append(f'  {row.block_sessions}-session blocks: Sharpe change {row.point:+.3f}, 95% interval [{row.lower95:+.3f}, {row.upper95:+.3f}].')
    return lines


def build_findings(tables,local,refit,report):
    periods=tables['periods']; fits=tables['fits']; matched=tables['matched']
    main=matched.loc[matched.case.eq('additive_main')&matched.reference.eq('linear_score')&matched.period.eq('evaluation')]
    interactions=matched.loc[matched.case.eq('additive_pairs')&matched.reference.eq('additive_main')&matched.period.eq('evaluation')]
    lines=['ADDITIVE POSITION-SIZING EXPERIMENT','',
        'The matched comparisons isolate nonlinear main effects and then three declared pairwise interactions within the same score-to-size framework.',
        f'Nonlinear main effects versus a linear score: {conclusion(main)}',
        f'Three interactions beyond additive main effects: {conclusion(interactions)}','',
        'WHAT WAS TESTED',
        'All five cases receive the same608 original proposals and the unchanged20% rank gate. This experiment changes entry sizing, with the existing contract selection, capital, risk rules and daily hedge accounting.',
        'The three new models produce a continuous prediction score, followed by the same discrete 50%/75%/100% sizing head. Cutoffs are terciles of each outer training window\'s fitted scores; exact cutoff ties select the higher size. A smooth score does not create continuous option-position sizing: crossing a sizing boundary can still change the requested size abruptly.',
        'Sizing depends on where a score falls relative to its training-score cutoffs, not whether the predicted dollar payoff is positive. Every qualified proposal receives at least a50% requested size, with subsequent admission controlled by the existing funded engine.',
        'Linear score versus additive main effects uses the same input identities, training population, target, regularization selection and sizing head. Additive main effects versus additive plus three interactions adds the declared interaction terms. The original tree plus CEM differs in partitioning and action optimization, so its comparison tests the whole sizing pipeline rather than nonlinear features alone.',
        'The new sizing head assigns larger sizes to higher predicted full-size payoffs. It does not separately estimate conditional risk or portfolio covariance when assigning sizes. The original CEM searches leaf allocations using aggregated training return paths and a Sharpe objective, so a difference against that baseline cannot be attributed solely to replacing the tree.',
        'Fixed variables means that the feature identities are fixed. Their coefficients or response curves, regularization and sizing calibration are refitted monthly on the preceding18 months of completed outcomes. There is no permanent set of fixed states.',
        'The legacy target is full-size standalone lifetime dollar P&L, with $0.65 option commission; funded evaluation uses $0.03 commission. Prediction/CV target fit is not equivalent to optimizing funded portfolio Sharpe.',
        'The three declared interactions are rank strength × pair correlation, rank strength × VIX, and 365-day past profitability × 365-day observation count. Interaction bases are residualized against their participating main-effect bases in the training window; correlated inputs can still prevent a unique economic interpretation.',
        '', 'FUNDED RESULTS',
        'Earlier:2018–2021. Later:2022–August2025. Both are chronological evaluations, but both histories had already been inspected before this experiment.',
        'Model | Earlier Sharpe | Later Sharpe | Later profit above cash | Later entries | Later cash-relative drawdown',
    ]
    for case in CASES:
        early=periods.loc[periods.case.eq(case)&periods.period.eq('validation')].iloc[0]
        late=periods.loc[periods.case.eq(case)&periods.period.eq('evaluation')].iloc[0]
        lines.append(f'{NAMES[case]} | {early.annualized_excess_sharpe:.3f} | {late.annualized_excess_sharpe:.3f} | ${late.profit_above_cash:,.2f} | {int(late.entries)} | {late.max_cash_relative_drawdown_pct:.3f}%')
    lines += ['', 'MATCHED COMPARISONS AND UNCERTAINTY']
    for candidate,reference in [('additive_main','linear_score'),('additive_pairs','additive_main')]:
        for period in ['validation','evaluation']:
            lines += interval_lines(matched,candidate,reference,period)
    lines += ['', 'COMPARISON WITH THE ORIGINAL TREE/CEM PIPELINE']
    for case in NEW+['fixed100']:
        lines += interval_lines(tables['intervals'],case,'tree_cem','evaluation')
    lines += [
        'Intervals resample common calendar blocks and are conditional on the frozen fitted policies. They do not refit the models inside each resample and do not correct for earlier research, choice of model family or multiple comparisons. Agreement across two block lengths is sensitivity evidence, not an untouched confirmation sample.',
        'Different sizing policies change existing inventory, future admission, integer contract counts, hedging and financing. Identical proposals therefore do not imply identical executed portfolios. Funded P&L and entries must be interpreted together.',
        '', 'TRAINING AND FALLBACKS',
        'Regularization is chosen from multipliers 0.1, 1 and 10 using two chronological three-month validation folds, with ridge penalty equal to training sample count times the multiplier. If both inner folds do not have sufficient completed data, the multiplier is fixed at 1 and the outer-window score model is still fitted; this fallback does not mean switching to fixed 100% sizing.',
    ]
    for case in NEW:
        own=fits.loc[fits.case.eq(case)]
        if own.empty:
            raise ValueError(f'Missing fit diagnostics for{case}')
        details=f'{NAMES[case]}: {len(own)} monthly fits; training examples {int(own.n_train.min())}–{int(own.n_train.max())}.'
        if 'basis_columns' in own:
            valid=pd.to_numeric(own.basis_columns,errors='coerce').dropna()
            if len(valid):
                details+=f' Basis columns {int(valid.min())}–{int(valid.max())}.'
        lines.append(details)
        if 'cv_status' in own:
            counts=own.cv_status.fillna('missing').value_counts().sort_index()
            statuses={'two_fold_selected':'two-fold selection','insufficient_inner_data_fallback':'insufficient inner data, alpha=1 fallback'}
            lines.append('  Recorded CV/fallback statuses: '+', '.join(f'{statuses.get(key,key)}: {value}' for key,value in counts.items())+'.')
        if 'alpha_multiplier' in own:
            counts=own.alpha_multiplier.value_counts(dropna=False).sort_index()
            lines.append('  Selected regularization multipliers: '+', '.join(f'{key}={value}' for key,value in counts.items())+'.')
    lines += ['', 'MEASURED SIZING STABILITY',
        'The following rates are weighted by the number of evaluated observations; they are not unweighted averages of monthly percentages. Local perturbation and month-to-month refit checks answer different questions.',
        'Model | Perturbed decisions changed | Mean absolute size change, percentage points | Refit decisions changed | Mean absolute refit size change, percentage points',
    ]
    for case in CASES:
        a=local.loc[local.case.eq(case)]; b=refit.loc[refit.case.eq(case)]
        if case=='fixed100':
            lines.append('Fixed 100% | 0% by construction | 0 | 0% by construction | 0')
            continue
        if a.empty and b.empty:
            continue
        def values(own):
            return 'unavailable | unavailable' if own.empty else f'{own.changed_fraction.iloc[0]:.1%} | {100*own.mean_absolute_size_change.iloc[0]:.2f}'
        lines.append(f'{NAMES[case]} | {values(a)} | {values(b)}')
    lines += [
        'Each local check perturbs one economic input by plus and minus 5% of its outer-training population standard deviation on each actual proposal, keeping the fitted model and sizing head frozen. Zero-standard-deviation inputs are excluded. Observations therefore count proposal × feature × perturbation-direction cases, not distinct proposals.',
        'The other observed inputs are held fixed. With correlated inputs, such a point can be unusual or outside the training data distribution. This is local numerical sensitivity, not causal feature importance or a realistic market-shock scenario.',
        'Refit stability applies successive monthly models to common historical training examples; an example can recur in several adjacent-window comparisons. It measures changes in recommendations when the fitted model changes, not prospective out-of-sample stability or realized turnover of held option positions. Existing option quantities remain fixed after entry under this strategy.',
        'A lower switching rate can reflect stronger shrinkage, compressed predictions or a sizing head that places most scores in the same bucket. It is not sufficient evidence of a better economic model. Aggregate rates can conceal differences across features and months; inspect stability_by_feature.csv and the original monthly files.',
        'Additive curves can help interpretation, but correlated predictors can redistribute estimated effects among themselves. Individual curves describe the fitted score conditional on the other terms; they are not independently identified economic relationships.',
        '', 'ANNUAL RESULTS',
        'Year | Model | Profit above cash | Excess Sharpe | Entries',
    ]
    annual=tables['annual'].copy()
    for year in sorted(annual.year.unique()):
        for case in CASES:
            own=annual.loc[annual.year.eq(year)&annual.case.eq(case)]
            if len(own)!=1:
                raise ValueError('Annual results incomplete')
            r=own.iloc[0]
            lines.append(f'{int(year)} | {NAMES[case]} | ${r.profit_above_cash:,.2f} | {r.annualized_excess_sharpe:.3f} | {int(r.entries)}')
    lines += ['', 'LIMITS AND NEXT DECISION',
        'The observed score/size comparison can support or weaken the case for this specific additive specification. It does not show that all nonlinear models, all interactions, or the full input set are optimal. No proposal gate or pair universe was redesigned here.',
        'Use the matched uncertainty comparisons to judge the extra nonlinear complexity and use the original tree/CEM comparison to judge the resulting policy. A small improvement in prediction or stability alone does not establish improved funded Sharpe. No model is automatically promoted by this report.',
        'The next confirmatory evaluation requires a frozen design and genuinely new outcomes; further comparisons on this history remain exploratory.',
        '', 'FILES',
        'funded/periods.csv; funded/annual.csv; funded/sharpe_intervals.csv; matched_intervals.csv; fits.csv; perturbation.csv; refit_stability.csv; stability_summary.csv; stability_by_feature.csv; comparison.png; report_manifest.json.',
    ]
    # Keep generated prose readable even when compact code constants carry numbers.
    replacements={'same608':'same 608','unchanged20%':'unchanged 20%', 'discrete50%':'discrete 50%',
                  'preceding18':'preceding 18','with$':'with $','uses$':'uses $','Earlier:2018':'Earlier: 2018',
                  'Later:2022':'Later: 2022','August2025':'August 2025'}
    replacements['a50%']='a 50%'
    text='\n'.join(lines)+'\n'
    for source,target in replacements.items():
        text=text.replace(source,target)
    (report/'Findings.txt').write_text(text,encoding='utf-8')


def plot_comparison(tables,local,refit,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    labels=['Tree + CEM','Fixed 100%','Linear','Additive','Additive +\ninteractions']
    x=np.arange(len(CASES))
    for period,offset,color,name in [('validation',-.18,'#9aa9b4','Earlier: 2018–2021'),('evaluation',.18,'#277e96','Later: 2022–2025')]:
        own=tables['periods'].loc[tables['periods'].period.eq(period)].set_index('case').reindex(CASES)
        axes[0,0].bar(x+offset,own.annualized_excess_sharpe,width=.34,color=color,label=name)
    axes[0,0].set_xticks(x,labels)
    axes[0,0].set_ylabel('Annualized excess Sharpe')
    axes[0,0].axhline(0,color='#666',lw=.7)
    axes[0,0].legend(fontsize=9)
    axes[0,0].set_title('Funded performance on the same proposal stream')
    own=tables['periods'].loc[tables['periods'].period.eq('evaluation')].set_index('case').reindex(CASES)
    bars=axes[0,1].bar(x,own.profit_above_cash/1000,color=[COLORS[c] for c in CASES])
    for bar,count in zip(bars,own.entries):
        axes[0,1].annotate(f'{int(count)} entries',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[0,1].set_xticks(x,labels)
    axes[0,1].set_ylabel('Later profit above cash, $ thousands')
    axes[0,1].set_title('Sizing also changes admission and exposure')
    axes[0,1].margins(y=.18)
    for own,offset,color,label in [(local,-.18,'#367cac','One-input perturbation'),(refit,.18,'#d39248','Monthly refit')]:
        values=own.set_index('case').reindex(CASES).changed_fraction
        values.loc['fixed100']=0.
        axes[1,0].bar(x+offset,100*values,width=.34,color=color,label=label)
    axes[1,0].set_xticks(x,labels)
    axes[1,0].set_ylabel('Evaluated size recommendations changed, %')
    axes[1,0].set_title('Measured stability of the discrete sizing decision')
    axes[1,0].legend(fontsize=9)
    comparisons=[('additive_main','linear_score','Nonlinear main effects\nminus linear'),('additive_pairs','additive_main','Three interactions\nminus main effects')]
    ax=axes[1,1]
    for j,(case,reference,label) in enumerate(comparisons):
        rows=tables['matched'].loc[tables['matched'].case.eq(case)&tables['matched'].reference.eq(reference)&tables['matched'].period.eq('evaluation')]
        for row in rows.itertuples():
            y=j+(-.11 if row.block_sessions==63 else .11)
            color='#367cac' if row.block_sessions==63 else '#ae6b24'
            ax.plot([row.lower95,row.upper95],[y,y],color=color,lw=2)
            ax.scatter([row.point],[y],color=color,s=28,label=f'{row.block_sessions}-session blocks' if j==0 else None)
    ax.axvline(0,color='#777',linestyle='--',lw=1)
    ax.set_yticks(range(2),[r[2] for r in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel('Later Sharpe change and conditional 95% interval')
    ax.set_title('Matched score-to-size comparisons')
    ax.legend(fontsize=9,loc='best')
    fig.suptitle('Additive entry sizing: continuous scores, discrete 50% / 75% / 100% sizes\nExploratory history; no policy automatically promoted',fontsize=14)
    fig.savefig(report/'comparison.png',dpi=160)
    plt.close(fig)


def whitebox_example(report,artifacts):
    """Expose one frozen proposal score and its discrete size assignment."""
    import joblib
    import matplotlib.pyplot as plt
    allocation_path=artifacts/'allocations.parquet'
    contribution_path=artifacts/'contributions.parquet'
    allocations=pd.read_parquet(allocation_path,use_threads=False)
    contributions=pd.read_parquet(contribution_path,use_threads=False)
    chosen=allocations.loc[allocations.case.eq('additive_pairs')].sort_values(
        ['date','pair'],ascending=[False,True],kind='stable').iloc[0]
    own=contributions.loc[contributions.case.eq(chosen.case)&contributions.date.eq(chosen.date)&contributions.pair.eq(chosen.pair)]
    if len(own)!=1:
        raise ValueError('White-box example must identify exactly one contribution row')
    terms=own.drop(columns=['date','pair','case']).iloc[0].dropna().astype(float)
    if not np.isclose(terms.sum(),chosen.prediction,rtol=1e-10,atol=1e-7):
        raise ValueError('Stored terms do not sum to the stored prediction')
    date=pd.Timestamp(chosen.date)
    model_path=artifacts/'models/additive_pairs'/f'{date.to_period("M").start_time.date()}.joblib'
    model=joblib.load(model_path)
    edges=np.asarray(model.training_score_edges,float)
    code=int(np.searchsorted(edges,float(chosen.prediction),side='right')+2)
    if code!=int(chosen.learned_level):
        raise ValueError('White-box score does not reproduce the saved sizing decision')
    decision_path=artifacts/'replays/additive_pairs/decisions.csv'
    decisions=pd.read_csv(decision_path,parse_dates=['signal_date','entry_date'])
    decision=decisions.loc[decisions.signal_date.eq(date)&decisions.pair.eq(chosen.pair)]
    if len(decision)!=1:
        raise ValueError('Missing funded admission decision for the example')
    d=decision.iloc[0]
    if not np.isclose(abs(d.action),code/4,atol=1e-12):
        raise ValueError('Saved admission action differs from score-derived size')
    table=terms.rename_axis('term').rename('predicted_lifetime_profit_contribution').reset_index()
    table['kind']=np.where(table.term.eq('intercept'),'intercept',np.where(table.term.str.contains(':'),'interaction','main'))
    table['label']=table.term.map(lambda term:' × '.join(FEATURE_NAMES.get(part,part) for part in term.split(':')))
    table.to_csv(report/'example_contributions.csv',index=False)
    details=dict(case=chosen.case,signal_date=str(date.date()),pair=chosen.pair,fit=str(date.to_period('M').start_time.date()),
        predicted_full_size_lifetime_profit=float(chosen.prediction),sum_of_terms=float(terms.sum()),
        training_score_edges=edges.tolist(),size_fraction=code/4,entry_date=str(d.entry_date.date()),admission_status=d.status,
        units='Legacy-label dollars of predicted full-size standalone lifetime profit, not realized or funded portfolio P&L.',
        attribution='Fitted-model score decomposition; not causal economic contribution or realized P&L attribution.',
        selection='Latest actual qualified proposal for additive_pairs; ascending pair identifier breaks same-date ties.')
    (report/'example_decision.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    # All terms are shown; centered contributions may have either sign.
    ordered=table.loc[~table.term.eq('intercept')].sort_values('predicted_lifetime_profit_contribution')
    ordered=pd.concat([table.loc[table.term.eq('intercept')],ordered],ignore_index=True)
    fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
    values=ordered.predicted_lifetime_profit_contribution
    colors=np.where(ordered.kind.eq('intercept'),'#78828b',np.where(values>=0,'#238772','#bf6862'))
    bars=ax.barh(range(len(ordered)),values,color=colors)
    ax.set_yticks(range(len(ordered)),ordered.label)
    ax.axvline(0,color='#777',lw=.8)
    ax.set_xlabel('Contribution to predicted full-size lifetime profit, legacy-label dollars')
    ax.set_title(f'One actual proposal: {date.date()} | pair {chosen.pair}\n'
        f'Terms sum to ${chosen.prediction:,.2f} → requested size {code/4:.0%}; admission: {d.status.replace("_", " ")}',fontsize=12)
    span=max(float(values.max()-values.min()),1.)
    for bar,value in zip(bars,values):
        ax.text(value+(.015*span if value>=0 else -.015*span),bar.get_y()+bar.get_height()/2,
                f'{value:,.2f}',va='center',ha='left' if value>=0 else 'right',fontsize=8)
    ax.margins(x=.2)
    fig.savefig(report/'example_contributions.png',dpi=160)
    plt.close(fig)
    with (report/'Findings.txt').open('a',encoding='utf-8') as stream:
        stream.write('\nONE REVIEWABLE SIZING DECISION\n')
        stream.write(f'The latest qualified proposal (same-date ties use ascending pair ID) is {chosen.pair} on {date.date()}, scored by the {details["fit"]} additive-interaction model.\n')
        stream.write(f'Intercept plus all main and interaction terms sum to predicted full-size lifetime profit ${chosen.prediction:,.2f}. Training-score boundaries are ${edges[0]:,.2f} and ${edges[1]:,.2f}, so the recorded requested size is {code/4:.0%}. Exact boundary ties select the higher size.\n')
        stream.write(f'The funded engine\'s next-session admission on {d.entry_date.date()} is {d.status}. A sizing recommendation does not guarantee an executed trade.\n')
        stream.write('These are predicted legacy-label dollar contributions to one fitted score, not realized P&L, portfolio allocations or causal effects. With correlated predictors, effect attribution can move among terms even when the total prediction is similar.\n')
        stream.write('The table and chart show every nonmissing term. Columns absent from a model after concatenating cases are omitted; zero-valued terms remain visible. Files: example_contributions.csv, example_decision.json, example_contributions.png.\n')
    return {str(path):digest(path) for path in [allocation_path,contribution_path,model_path,decision_path]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--artifacts',type=Path,default=ARTIFACTS)
    args=parser.parse_args()
    os.environ['MPLCONFIGDIR']=str(args.artifacts/'matplotlib')
    budget=Budget()
    started=time.monotonic()
    tables,sources=read_tables(args.report)
    local=summarize_stability(tables['perturbation'],'n',['case'])
    byfeature=summarize_stability(tables['perturbation'],'n',['case','feature'])
    refit=summarize_stability(tables['refit'],'n_common',['case'])
    combined=pd.concat([local.assign(diagnostic='local_perturbation'),refit.assign(diagnostic='monthly_refit')],ignore_index=True)
    combined.to_csv(args.report/'stability_summary.csv',index=False)
    byfeature.to_csv(args.report/'stability_by_feature.csv',index=False)
    build_findings(tables,local,refit,args.report)
    plot_comparison(tables,local,refit,args.report)
    sources.update(whitebox_example(args.report,args.artifacts))
    budget.rest(started)
    sources[str(Path(__file__))]=digest(Path(__file__))
    for path in [args.report/'protocol.json',args.report/'funded/analysis_protocol.json']:
        if path.exists():
            sources[str(path)]=digest(path)
    manifest={'status':'complete','report_only':True,'model_fitted':False,'backtest_replayed':False,
        'source_hashes':sources,'threads':1,'duty_cycle':.2,'rss_limit_mib':384,'peak_rss_mib':budget.peak,
        'outputs':{name:digest(args.report/name) for name in ['Findings.txt','comparison.png','stability_summary.csv','stability_by_feature.csv',
            'example_contributions.csv','example_decision.json','example_contributions.png']}}
    (args.report/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Additive sizing report complete; peak RSS{budget.peak:.1f} MiB; {args.report}',flush=True)


if __name__=='__main__':
    main()
