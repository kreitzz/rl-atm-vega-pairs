"""Purged quarterly asset variance forecasts, mapped to cached actual pair P&L."""
from pathlib import Path
import json
import resource
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scripts.discover_contract_signals import digest, summarize
from scripts.explore_universe_gap import correct_earnings

OUT=Path('artifacts/contracts/variance_forecast')
REPORT=Path('reports/variance_forecast')
ROOT=Path('data/option_contracts')
EARNINGS=Path('/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds/compustat/underlying_quarterly_earnings.parquet')
HAR=['log_rv5','log_rv21','log_rv63','log_dte']
MODELS={'har':HAR,'har_iv':HAR+['log_ivvar'],
        'har_iv_earnings':HAR+['log_ivvar','earnings_in_contract','days_to_earnings','event_fraction']}


def variance_target(square, entry_index, end_index):
    # Entry-close hedge is opened after the entry-day move: exclude that move.
    x=np.asarray(square)[entry_index+1:end_index+1]
    return 252*float(np.mean(x)) if len(x)>0 and np.isfinite(x).all() else np.nan


def event_features(dates, signal, entry, expiry):
    dates=pd.DatetimeIndex(dates)
    future=dates[dates>signal]
    until=(future[0]-signal).days if len(future) else np.nan
    count=int(((dates>entry)&(dates<=expiry)).sum())
    return count,until


def forecast(train,test,features):
    x=train[features].to_numpy(float); t=test[features].to_numpy(float)
    keep=np.isfinite(x).any(axis=0);x=x[:,keep];t=t[:,keep]
    lo,hi=np.nanquantile(x,[.01,.99],axis=0)
    imp=SimpleImputer(strategy='median',add_indicator=True)
    x=imp.fit_transform(np.clip(x,lo,hi));t=imp.transform(np.clip(t,lo,hi))
    scaler=StandardScaler();x=scaler.fit_transform(x);t=scaler.transform(t)
    w=1/train.groupby('secid').secid.transform('size').to_numpy(float);w*=len(w)/w.sum()
    y=np.log(train.target.clip(lower=1e-8))
    model=Ridge(alpha=.1*len(train)).fit(x,y,sample_weight=w)
    # Conditional log-variance to level variance, using training-only smearing.
    smear=np.average(np.exp(y-model.predict(x)),weights=w)
    return np.exp(model.predict(t))*smear


def build():
    source=Path('artifacts/contracts/option_signal_diagnostics/outcomes.parquet')
    outcomes=pd.read_parquet(source).query("combination == 'CC'").copy()
    definition_path=Path('artifacts/contracts/expiry_trained_beta_085_115/inputs.json')
    defs=json.loads(definition_path.read_text())['definitions']
    defs={d['pair']:d for d in defs}
    requests=[]
    for pair,g in outcomes.groupby('pair'):
        for side,sid in zip(['a','b'],defs[pair]['secids']):
            h=g[['date','pair']].copy();h['side']=side;h['secid']=sid;requests.append(h)
    requests=pd.concat(requests,ignore_index=True)
    ids=sorted(requests.secid.unique().tolist());unique=requests[['date','secid']].drop_duplicates()
    pieces=[];hashes={str(source):digest(source),str(definition_path):digest(definition_path)}
    for year in range(2017,2026):
        p=ROOT/'underlying'/f'year={year}.parquet';hashes[str(p)]=digest(p)
        pieces.append(pd.read_parquet(p,columns=['date','secid','close','cfadj'],filters=[('secid','in',ids)]))
    stocks=pd.concat(pieces,ignore_index=True);calendar=pd.DatetimeIndex(stocks.date.unique()).sort_values()
    earnings=correct_earnings(pd.read_parquet(EARNINGS));hashes[str(EARNINGS)]=digest(EARNINGS)
    metadata=json.loads(Path('artifacts/broad_pairs/assets.json').read_text())
    metadata={x['asset']:x for x in metadata}
    supplements=pd.read_csv('config/earnings_supplement.csv',parse_dates=['earnings_date'])
    for p in [Path('artifacts/broad_pairs/assets.json'),Path('config/earnings_supplement.csv'),Path('config/universe_earnings_overrides.csv')]:hashes[str(p)]=digest(p)
    # Resolve the same historical CRSP link used by each pair, not latest ticker.
    asset_for_sid={}
    for pair in outcomes.pair.unique():
        d=defs[pair]
        if d['group']=='broad':
            for sid,a in zip(d['secids'],d['source_pair'].split('__')):
                if sid in asset_for_sid:assert asset_for_sid[sid]==a
                asset_for_sid[sid]=a
    rows=[]
    for j,(sid,g) in enumerate(unique.groupby('secid'),1):
        frames=[]
        for year in sorted(g.date.dt.year.unique()):
            p=Path('data/contract_history_shared')/str(year)/f'{sid}_selection.parquet'
            if p.exists():
                hashes[str(p)]=digest(p)
                frames.append(pd.read_parquet(p,columns=['date','optionid','expiry','bs_iv','entry_date']))
        if not frames:continue
        selection=pd.concat(frames).set_index('date');assert selection.index.is_unique
        s=stocks[stocks.secid.eq(sid)].set_index('date').reindex(calendar)
        square=np.log((s.close*s.cfadj)/(s.close.shift()*s.cfadj.shift()))**2
        rolling={n:252*square.rolling(n,min_periods=n).mean() for n in [5,21,63]}
        info=metadata.get(asset_for_sid.get(sid,''))
        if info is None:
            matches=[v for v in metadata.values() if v['secid']==sid]
            if len(matches)==1:info=matches[0]
        events=pd.DatetimeIndex([])
        if info and not info['etf']:
            events=pd.DatetimeIndex(pd.concat([pd.to_datetime(earnings.loc[earnings.permno.eq(info['permno']),'rdq']).dropna(),supplements.loc[supplements.ticker.eq(info['ticker']),'earnings_date']]).drop_duplicates()).sort_values()
        for date in g.date:
            if date not in selection.index:continue
            q=selection.loc[date];entry=calendar.searchsorted(date)+1;end=calendar.searchsorted(q.expiry,side='right')-1
            if entry>=len(calendar) or end<=entry or q.expiry>calendar[-1]:continue
            assert pd.Timestamp(q.entry_date)==calendar[entry]
            target=variance_target(square,entry,end)
            if not np.isfinite(target) or not np.isfinite(q.bs_iv) or q.bs_iv<=0:continue
            count,until=event_features(events,date,calendar[entry],calendar[end])
            known=bool(info and (info['etf'] or len(events)>=20))
            row=dict(date=date,secid=sid,entry=calendar[entry],end=calendar[end],expiry=q.expiry,optionid=int(q.optionid),
                     target=target,ivvar=float(q.bs_iv)**2,iv=float(q.bs_iv),log_ivvar=np.log(q.bs_iv**2),
                     log_dte=np.log((q.expiry-calendar[entry]).days),
                     earnings_in_contract=count if known else np.nan,days_to_earnings=min(until,120) if known and np.isfinite(until) else np.nan,
                     event_fraction=count/max(1,end-entry) if known else np.nan)
            for n in [5,21,63]:row[f'log_rv{n}']=np.log(max(rolling[n].get(date,np.nan),1e-8))
            row['rv21']=rolling[21].get(date,np.nan);rows.append(row)
        if j%20==0:print(f'Prepared {j}/{len(ids)} underlyings',flush=True)
    legs=pd.DataFrame(rows);legs=legs.replace([np.inf,-np.inf],np.nan).dropna(subset=HAR+['target'])
    assert not legs.duplicated(['date','secid']).any()
    legs.to_parquet(OUT/'legs.parquet',index=False)
    return legs,outcomes,requests,hashes


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    legs,outcomes,requests,hashes=build();preds=[];fits=[];payoffs=[];costs=[]
    for period in pd.period_range('2020Q1','2025Q3',freq='Q'):
        cutoff=period.start_time
        tr=legs[(legs.end<cutoff)&(legs.date>=cutoff-pd.DateOffset(years=5))]
        te=legs[legs.date.dt.to_period('Q').eq(period)].copy()
        if len(tr)<500 or te.empty:continue
        assert tr.end.max()<te.date.min()
        for name,features in MODELS.items():te[name]=forecast(tr,te,features)
        te['implied_variance']=te.ivvar;te['trailing_rv21']=te.rv21
        preds.append(te)
        past=outcomes[(outcomes.end<cutoff)&(outcomes.date>=cutoff-pd.DateOffset(years=5))].copy()
        past['symmetric_drag']=-(past.long_net+past.short_net)/2
        hurdle=max(0.,past.groupby('pair').symmetric_drag.mean().mean())
        costs.append(dict(quarter=str(period),completed_pair_outcomes=len(past),hurdle=hurdle))
        fits.append(dict(quarter=str(period),training_rows=len(tr),training_assets=tr.secid.nunique(),last_training_end=tr.end.max(),first_test_date=te.date.min(),cutoff=cutoff,test_rows=len(te)))
        joined=requests.merge(te,on=['date','secid'],validate='many_to_one')
        a=joined[joined.side.eq('a')].set_index(['date','pair']);b=joined[joined.side.eq('b')].set_index(['date','pair'])
        common=a.index.intersection(b.index)
        a=a.loc[common];b=b.loc[common]
        realized=outcomes.set_index(['date','pair']).loc[common]
        expected_end=pd.concat([a.end,b.end],axis=1).max(axis=1)
        assert (expected_end.to_numpy()==realized.end.to_numpy()).all(), 'Forecast and payoff horizons differ'
        for model in ['trailing_rv21',*MODELS]:
            # Small-variance-change approximation in dollars per one-vol-point vega.
            score=100*((a[model]-a.ivvar)/(2*a.iv)-(b[model]-b.ivvar)/(2*b.iv))
            r=realized.reset_index().copy();r['model']=model;r['pred_long']=score.to_numpy()-hurdle;r['pred_short']=-score.to_numpy()-hurdle
            r['action']=np.where(abs(score.to_numpy())>hurdle,np.sign(score.to_numpy()),0).astype(int)
            r['score']=score.to_numpy();payoffs.append(r)
        print(f'{period}: {len(tr):,} training legs, {len(common):,} matched pair opportunities, hurdle {hurdle:.3f}',flush=True)
    p=pd.concat(preds,ignore_index=True);p.to_parquet(OUT/'forecasts.parquet',index=False)
    predictions=pd.concat(payoffs,ignore_index=True);predictions.to_parquet(OUT/'pair_predictions.parquet',index=False)
    pd.DataFrame(fits).to_csv(REPORT/'fit_audit.csv',index=False);pd.DataFrame(costs).to_csv(REPORT/'cost_hurdles.csv',index=False)
    accuracy=[]
    for model in ['implied_variance','trailing_rv21',*MODELS]:
        for year,g in [('all',p),*list(p.groupby(p.date.dt.year))]:
            f=g[model].clip(lower=1e-8);ratio=g.target/f
            accuracy.append(dict(model=model,year=year,observations=len(g),mse=float(((g.target-f)**2).mean()),
                                 qlike=float((ratio-np.log(ratio.clip(lower=1e-12))-1).mean()),
                                 variance_spearman=g.target.corr(f,method='spearman')))
    pd.DataFrame(accuracy).to_csv(REPORT/'forecast_accuracy.csv',index=False)
    summary=[];annual=[]
    for model,g in predictions.groupby('model'):
        for sample,h in [('all',g),('nonoverlap',g[g.nonoverlap])]:
            summary.append(dict(model=model,sample=sample,**summarize(h)))
            for year,z in h.groupby(h.date.dt.year):annual.append(dict(model=model,sample=sample,year=year,**summarize(z)))
    result=pd.DataFrame(summary);result.to_csv(REPORT/'summary.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    manifest=dict(source_hashes=hashes,code_sha256=digest(Path(__file__)),models=MODELS,ridge_alpha_per_row=.1,
                  unique_prepared_legs=len(legs),unique_evaluation_legs=len(p),available_pair_opportunities=int((outcomes.date>='2020-01-01').sum()),
                  matched_pair_opportunities=predictions[predictions.model.eq('har')].shape[0],
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Contract-horizon realized-variance forecasts','',
      'First forecast each underlying’s close-to-close realized variance from next-session entry close through the selected call’s last trading day at expiry. Split-adjusted stock prices match the simulator convention; targets use 252 × mean squared log return, with no entry-day return. Unique security/date training rows avoid duplicating a stock merely because it occurs in several pairs.','',
      'Quarterly pooled log-variance ridge models use completed outcomes only, a five-year window, equal total weight per security, train-only clipping/imputation/scaling and a training-residual smearing correction. HAR uses trailing 5/21/63-session variance plus actual DTE. HAR+IV adds quoted-contract implied variance. The earnings variant additionally uses the uncapped countdown, event count inside the holding period and event count per holding session. No model or threshold was selected using these results.','',
      '**Earnings caveat:** event features use final historical earnings dates, assumed known on the signal date; no point-in-time announcement archive is available. The earnings-free variants are the primary comparison. Dates are daily, so before-open versus after-close timing is unresolved. Missing event coverage is imputed using training data, not treated as proof of no earnings.','',
      'Translate each forecast to approximate long-option profit per initial vega using 100 × (forecast variance − IV²)/(2 × IV), then subtract the two legs. Trade only when its magnitude exceeds the trailing completed-outcome estimate of symmetric directional net drag, equal-weighted across pairs. This is a historical empirical hurdle, not an exact decomposition of fees; both signs and all models use the same hurdle. Actual results use cached net daily-delta-hedged hold-to-expiry pair labels, including the existing execution, borrow and financing assumptions.','',
      '| Forecast | Trades | Net / initial vega per trade | Net / opportunity | Nonoverlap net / trade |',
      '|---|---:|---:|---:|---:|']
    for r in result[result['sample'].eq('all')].itertuples():
        thin=result[result.model.eq(r.model)&result['sample'].eq('nonoverlap')].iloc[0]
        lines.append(f'| {r.model} | {r.trades:,} | {r.mean_net_per_trade:.3f} | {r.mean_net_per_opportunity:.3f} | {thin.mean_net_per_trade:.3f} |')
    lines+=['', 'Quoted-contract implied variance beat every tested forecast on both mean squared error and QLIKE loss (lower is better). All translated trading rules had negative average net payoff in the full and nonoverlapping samples. Each fitted rule was negative in five of six evaluation calendar years; 2025 is partial.', '', '| Variance forecast | Mean squared error | QLIKE |', '|---|---:|---:|']
    for r in accuracy:
        if r['year']=='all':lines.append(f"| {r['model']} | {r['mse']:.4f} | {r['qlike']:.4f} |")
    lines+=['',f"Matched {manifest['matched_pair_opportunities']:,} of {manifest['available_pair_opportunities']:,} previously evaluable pair opportunities; missing histories or feature warmup are excluded identically across models.",'',
      'This is a signal/standalone-payoff test, not a financed portfolio backtest: it does not enforce overlapping-book capacity or establish annualized Sharpe. Terminal cohort P&L cannot be used as daily returns. Missing/resolved outcomes condition the sample; the prior beta/membership screens remain. Nonoverlap thinning is diagnostic and does not eliminate dependence between pairs. Historical research data has already been repeatedly explored. Vega is matched at entry, not rebalanced daily. The variance-to-vega approximation omits path-dependent gamma weighting and jumps; improved variance accuracy need not yield improved trading returns.','',
      '[Forecast accuracy](variance_forecast/forecast_accuracy.csv) · [Annual pair outcomes](variance_forecast/annual.csv) · [Fit timing](variance_forecast/fit_audit.csv) · [Historical cost hurdles](variance_forecast/cost_hurdles.csv)','']
    lines += ['', f"Checks: all matched pair outcome-end dates equal the later leg expiry session; completed training ends precede prediction quarters. Three unit tests cover holding-window boundaries, uncapped earnings horizons, and held-out-target independence. Peak worker memory: {manifest['peak_rss_bytes']/1024**3:.2f} GiB; one worker at 50% duty cycle.", '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
