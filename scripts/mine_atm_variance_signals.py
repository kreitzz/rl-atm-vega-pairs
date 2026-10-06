"""Current-cost, pre-policy economic signal mining with purged RV forecasts.

One guarded worker. No RL/NN refits, cached observed prices only. Terminal
cohort payoffs are diagnostics, never daily returns or portfolio Sharpe.
"""
import argparse
from dataclasses import replace,asdict
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scripts.run_atm65_employee_sentiment import OUT as EMPLOYEE
from scripts.run_atm65_ablation import write_json
from scripts.curve_trading.config import config
from scripts.run_atm_cost_hurdles import make_units,reward_and_memory
from scripts.single_leg_policy import single_labels
from scripts.replay_frozen_atm_commission import load_pair
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar
from scripts.run_curve_trading import screen_dates,RUNOFF

OUT=Path('artifacts/contracts/atm_variance_mining')
REPORT=Path('reports/atm_variance_mining')
HAR=['log_rv5_iv','log_rv21_iv','log_rv63_iv','log_dte']
EXTRA=['log_iv','log_ewma10_iv','log_ewma42_iv','log_range21_iv','overnight_fraction',
       'downside_fraction','jump_fraction','rv_acceleration','iv_change21',
       'iv_vol_of_vol','market_rv_ratio','vix_level']
SIGNALS=['iv_spread','trailing_vrp','ewma_vrp','range_vrp','har_anchor',
         'expanded_anchor','horizon_gamma_edge','historical_vrp_z','factor_vrp_residual',
         'iv_factor_reversion','predicted_leg_pnl','rv_acceleration_gap',
         'overnight_fraction_gap','downside_fraction_gap','jump_fraction_gap']


def holding_variance(square,positions,ends):
    """Entry-close hedge excludes both the signal and entry-day stock moves."""
    square=np.asarray(square,float);positions=np.asarray(positions,int);ends=np.asarray(ends,int)
    good=np.isfinite(square);s=np.r_[0.,np.cumsum(np.where(good,square,0.))];bad=np.r_[0,np.cumsum(~good)]
    start=positions+2;stop=ends+1;length=stop-start
    valid=(length>0)&(start>=0)&(stop<=len(square))
    a=np.clip(start,0,len(square));b=np.clip(stop,0,len(square))
    valid&=(bad[b]-bad[a])==0
    return np.divide(252*(s[b]-s[a]),length,out=np.full(len(length),np.nan),where=valid)


def past_z(series):
    h=series.shift(1).rolling(252,min_periods=126)
    return (series-h.mean())/h.std().where(h.std()>1e-8)


def price_features(prices,calendar):
    s=prices.set_index('date').reindex(calendar)
    p=(s.close*s.cfadj).where(s.close.gt(0)&s.cfadj.gt(0));ret=np.log(p/p.shift(1));square=ret**2
    result=pd.DataFrame(index=calendar)
    for n in [1,5,21,63]:result[f'rv{n}']=252*square.rolling(n,min_periods=n).mean()
    for n in [10,42]:result[f'ewma{n}']=252*square.ewm(halflife=n,min_periods=63,adjust=False).mean()
    range_valid=(s.high>=s.low)&(s.low>0)&(s.high>0)
    range_var=(np.log(s.high/s.low)**2/(4*np.log(2))).where(range_valid)
    opening=(s.open*s.cfadj).where(s.open.gt(0)&s.cfadj.gt(0))
    overnight=np.log(opening/p.shift(1))**2
    result['range21']=252*(range_var+overnight).rolling(21,min_periods=21).mean()
    total=square.rolling(21,min_periods=21).sum().where(lambda x:x>0)
    result['overnight_fraction']=overnight.rolling(21,min_periods=21).sum()/total
    result['downside_fraction']=square.where(ret<0,0.).where(ret.notna()).rolling(21,min_periods=21).sum()/total
    normal=square.rolling(63,min_periods=63).mean().shift(1)
    result['jump_fraction']=square.where(square>9*normal,0.).where(normal.notna()).rolling(21,min_periods=21).sum()/total
    result['rv_acceleration']=np.log(result.rv5/result.rv63.where(result.rv63>0))
    return result.replace([np.inf,-np.inf],np.nan),square


def deduplicate_legs(frame):
    keys=['date','secid','optionid']
    # Duplicated assets must not receive extra weight because of pair membership.
    for field in ['iv','target','long_net','short_net']:
        g=frame.groupby(keys)[field].agg(['min','max'])
        assert (g['max']-g['min']).abs().dropna().le(1e-7).all(),field
    result=frame.drop_duplicates(keys).sort_values(['secid','date']).reset_index(drop=True)
    assert not result.duplicated(['date','secid']).any()
    return result


def prepare(defs,market,cfg):
    start=time.monotonic();cal=market.calendar;coverage=[]
    bars=pd.concat([pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','open','high','low']) for year in cal.year.unique()],ignore_index=True)
    bars['date']=pd.to_datetime(bars.date)
    market.stocks=market.stocks.merge(bars,on=['date','secid'],how='left',validate='one_to_one');del bars
    for i,d in enumerate(defs,1):
        dest=OUT/'pairs'/d['pair'];dest.mkdir(parents=True,exist_ok=True)
        source=EMPLOYEE/d['pair']/'states.parquet'
        if not source.exists():
            coverage.append(dict(pair=d['pair'],label=d['label'],state_rows=0,eligible_rows=0,resolved_rows=0,unresolved_rows=0));continue
        if (dest/'coverage.json').exists():coverage.append(json.loads((dest/'coverage.json').read_text()));continue
        panel=pd.read_parquet(source);q,s,selections,divs=load_pair(market,d['secids'])
        units=make_units(panel,q,s,selections,divs,cal,cfg)
        daily,ends,vegas,status,_,_=reward_and_memory(units,cfg)
        eligible=panel.eligible.to_numpy(bool)&screen_dates(d,panel.index)&(panel.index<=cal[-1]-pd.Timedelta(days=RUNOFF))
        entered=(vegas[:,0]>0)&(vegas[:,-1]>0)
        resolved=eligible&entered&np.isfinite(daily[:,[0,-1]]).all(axis=(1,2))
        row=dict(pair=d['pair'],label=d['label'],state_rows=len(panel),eligible_rows=int(eligible.sum()),
            resolved_rows=int(resolved.sum()),unresolved_rows=int((eligible&entered&~resolved).sum()),
            ineligible_or_unfilled_rows=int((~eligible|~entered).sum()))
        coverage.append(row)
        pair=panel.copy();pair['date']=panel.index;pair['pair']=d['pair'];pair['label']=d['label']
        pair['secid_a'],pair['secid_b']=d['secids'];pair['end']=cal[ends]
        pair['long_net']=np.divide(daily[:,-1].sum(axis=1)*cfg.entry_vega_per_leg,vegas[:,-1],out=np.full(len(panel),np.nan),where=vegas[:,-1]>0)
        pair['short_net']=np.divide(daily[:,0].sum(axis=1)*cfg.entry_vega_per_leg,vegas[:,0],out=np.full(len(panel),np.nan),where=vegas[:,0]>0)
        pair['directional_edge']=(pair.long_net-pair.short_net)/2
        pair.loc[resolved].reset_index(drop=True).to_parquet(dest/'outcomes.parquet',index=False)
        leg_frames=[]
        for k,(sid,suffix) in enumerate(zip(d['secids'],['amd','nvda'])):
            unit=units[k];ids=unit.ids(30,'C','atm');solo,finish,v,_=single_labels(unit,cfg)
            sel=selections[selections.secid.eq(sid)&selections.tenor.eq(30)&selections.bucket.eq('atm')].set_index('date').reindex(panel.index)
            stock=s[s.secid.eq(sid)].copy()
            # OHLC used for predictors only; the hedge ledger keeps the original closes.
            f,square=price_features(stock,cal);leg=f.reindex(panel.index).copy()
            leg['date']=panel.index;leg['secid']=sid;leg['optionid']=sel.optionid
            leg['iv']=panel['iv_'+suffix]/100;leg['dte']=panel['dte_'+('a' if k==0 else 'b')]
            scheduled=cal.searchsorted(pd.DatetimeIndex(sel.expiry),side='right')-1
            complete_schedule=sel.expiry.notna().to_numpy()&(sel.expiry<=cal[-1]).to_numpy()&(scheduled>unit.positions+1)
            leg['expiry']=sel.expiry
            leg['end']=pd.Series(cal[np.clip(scheduled,0,len(cal)-1)],index=panel.index).where(complete_schedule)
            leg['entry']=cal[np.minimum(unit.positions+1,len(cal)-1)]
            leg['target']=np.where(complete_schedule,holding_variance(square,unit.positions,np.clip(scheduled,0,len(cal)-1)),np.nan)
            leg['horizon_sessions']=scheduled-unit.positions-1
            leg['calendar_t']=(leg.end-leg.entry).dt.days/365
            leg['long_net']=np.divide(solo[:,-1].sum(axis=1)*cfg.entry_vega_per_leg,v[:,-1],out=np.full(len(panel),np.nan),where=v[:,-1]>0)
            leg['short_net']=np.divide(solo[:,0].sum(axis=1)*cfg.entry_vega_per_leg,v[:,0],out=np.full(len(panel),np.nan),where=v[:,0]>0)
            leg['vix_level']=panel.vix_previous_close/100
            # Sector metadata is the historical roster's industry, a research grouping.
            leg['industry']=str(d['industry'])[:2] if str(d['industry']).isdigit() else 'other'
            # Prediction/factor membership uses ONLY signal-date information.
            # Outcome labels remain NaN when entry or later hedged paths are unavailable.
            ok=leg.iv.gt(0).to_numpy()&leg.optionid.notna().to_numpy()&sel.expiry.notna().to_numpy()&(unit.positions+1<len(cal))
            leg_frames.append(leg.loc[ok].reset_index(drop=True))
        pd.concat(leg_frames,ignore_index=True).to_parquet(dest/'legs.parquet',index=False)
        write_json(dest/'coverage.json',row)
        if i%10==0:print(f'Prepared {i}/{len(defs)} sleeves; {time.monotonic()-start:.0f}s; RSS peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB',flush=True)
        del panel,q,s,selections,units,daily,pair,leg_frames,solo;gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    frames=[pd.read_parquet(p) for p in (OUT/'pairs').glob('*/legs.parquet')]
    legs=deduplicate_legs(pd.concat(frames,ignore_index=True))
    # Resolve sector grouping without using outcome performance.
    grouping=legs.groupby('secid').industry.agg(lambda x:sorted(set(x)))
    ambiguous=set(grouping.index[grouping.map(len)>1]);legs.loc[legs.secid.isin(ambiguous),'industry']='other'
    legs.to_parquet(OUT/'legs.parquet',index=False)
    pd.concat([pd.read_parquet(p) for p in (OUT/'pairs').glob('*/outcomes.parquet')],ignore_index=True).to_parquet(OUT/'outcomes.parquet',index=False)
    write_json(OUT/'prepared.json',dict(unique_legs=len(legs),ambiguous_sector_assets=len(ambiguous),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def forecast(train,test,columns,target='target',anchor=True):
    x=train[columns].to_numpy(float);t=test[columns].to_numpy(float)
    keep=np.isfinite(x).any(axis=0);x=x[:,keep];t=t[:,keep]
    lo,hi=np.nanquantile(x,[.01,.99],axis=0)
    imp=SimpleImputer(strategy='median',add_indicator=True)
    x=imp.fit_transform(np.clip(x,lo,hi));t=imp.transform(np.clip(t,lo,hi))
    scale=StandardScaler();x=scale.fit_transform(x);t=scale.transform(t)
    weights=1/train.groupby('secid').secid.transform('size').to_numpy(float);weights*=len(weights)/weights.sum()
    y=np.log(train.target.clip(lower=1e-8)/train.iv.pow(2)) if anchor else train[target].to_numpy(float)
    model=Ridge(alpha=.1*len(train)).fit(x,y,sample_weight=weights)
    prediction=model.predict(t)
    if anchor:
        smear=np.average(np.exp(np.clip(y-model.predict(x),-10,10)),weights=weights)
        return test.iv.to_numpy()**2*np.exp(np.clip(prediction,-10,10))*smear
    return prediction


def predictions(legs,cal):
    frame=legs.copy();ivvar=frame.iv**2
    for col in ['rv5','rv21','rv63','ewma10','ewma42','range21']:
        frame['log_'+col+'_iv']=np.log(frame[col].clip(lower=1e-8)/ivvar)
    frame['log_dte']=np.log(frame.dte.clip(lower=1))
    frame['log_iv']=np.log(frame.iv)
    frame['leg_directional_edge']=(frame.long_net-frame.short_net)/2
    # Past signal marks can be missing because these legs require resolved payoffs.
    # IV history below instead comes from ALL original signal states, not this outcome-filtered frame.
    marks=[]
    definitions={d['pair']:d for d in json.loads((EMPLOYEE/'inputs.json').read_text())['definitions']}
    for path in EMPLOYEE.glob('*/states.parquet'):
        pair=path.parent.name;a,b=definitions[pair]['secids'];p=pd.read_parquet(path,columns=['iv_amd','iv_nvda','trailing_rv_amd','trailing_rv_nvda'])
        for sid,suffix in [(a,'amd'),(b,'nvda')]:
            marks.append(pd.DataFrame(dict(date=p.index,secid=sid,iv=p['iv_'+suffix].to_numpy()/100,rv=p['trailing_rv_'+suffix].to_numpy()/100)))
    marks=pd.concat(marks,ignore_index=True).drop_duplicates(['date','secid'])
    iv=marks.pivot(index='date',columns='secid',values='iv').reindex(cal).where(lambda x:x>0)
    rv=marks.pivot(index='date',columns='secid',values='rv').reindex(cal)
    premium=iv**2-rv**2
    market_rv=rv.pow(2).median(axis=1).where(rv.notna().sum(axis=1)>=20)
    history=[]
    for sid in iv.columns:
        own=iv[sid];change=own.diff();factor=iv.drop(columns=sid).diff().median(axis=1)
        h=pd.DataFrame(index=cal);h['secid']=sid;h['iv_change21']=own.diff(21)/own.shift(21)
        h['iv_vol_of_vol']=change.rolling(21,min_periods=21).std()/own
        v=factor.shift(1).rolling(252,min_periods=126).var()
        beta=change.shift(1).rolling(252,min_periods=126).cov(factor.shift(1))/v.where(v>1e-10)
        h['iv_factor_residual']=change-beta*factor
        h['iv_factor_reversion']=-h.iv_factor_residual.rolling(21,min_periods=21).sum()/change.rolling(252,min_periods=126).std()
        h['trailing_vrp_z']=past_z(premium[sid]);h['market_rv']=market_rv
        h['date']=cal;history.append(h)
    history=pd.concat(history,ignore_index=True)
    frame=frame.merge(history.drop(columns='iv_factor_residual'),on=['date','secid'],validate='many_to_one')
    frame['market_rv_ratio']=np.log(frame.rv21.clip(lower=1e-8)/frame.market_rv.clip(lower=1e-8))
    frame=frame.replace([np.inf,-np.inf],np.nan)
    output=[];fits=[]
    for quarter in pd.period_range('2012Q1','2025Q3',freq='Q'):
        cutoff=quarter.start_time
        train=frame[(frame.end<cutoff)&(frame.date>=cutoff-pd.DateOffset(years=5))&frame.target.gt(0)]
        test=frame[frame.date.dt.to_period('Q').eq(quarter)].copy()
        if len(train)<1000 or test.empty:continue
        assert train.end.max()<test.date.min()
        test['har_forecast']=forecast(train,test,HAR)
        test['expanded_forecast']=forecast(train,test,HAR+EXTRA)
        direct=train.dropna(subset=['leg_directional_edge'])
        test['predicted_leg_pnl']=forecast(direct,test,HAR+EXTRA,'leg_directional_edge',False) if len(direct)>=1000 else np.nan
        output.append(test)
        fits.append(dict(quarter=str(quarter),cutoff=str(cutoff.date()),training_rows=len(train),test_rows=len(test),
            last_training_end=str(train.end.max().date()),first_test_date=str(test.date.min().date()),unique_training_assets=train.secid.nunique()))
        if quarter.quarter==1:print(f'{quarter}: {len(train):,} completed training legs / {len(test):,} predictions',flush=True)
    frame=pd.concat(output,ignore_index=True)
    # Relative-value scores: larger means a more attractive LONG in this leg.
    for name,var in [('trailing_vrp',frame.rv21),('ewma_vrp',frame.ewma10),('range_vrp',frame.range21),
                     ('har_anchor',frame.har_forecast),('expanded_anchor',frame.expanded_forecast)]:
        frame[name]=100*(var-frame.iv**2)/(2*frame.iv)
    frame['iv_spread']=-100*frame.iv
    frame['horizon_gamma_edge']=100*(frame.expanded_forecast*frame.horizon_sessions/252-frame.iv**2*frame.calendar_t)/(2*frame.iv*frame.calendar_t)
    frame['historical_vrp_z']=-frame.trailing_vrp_z
    frame['forecast_vrp']=frame.iv**2-frame.expanded_forecast
    frame=factor_residuals(frame,cal)
    frame.rename(columns={'rv_acceleration':'rv_acceleration_gap','overnight_fraction':'overnight_fraction_gap',
        'downside_fraction':'downside_fraction_gap','jump_fraction':'jump_fraction_gap'},inplace=True)
    frame.to_parquet(OUT/'forecasts.parquet',index=False);pd.DataFrame(fits).to_csv(REPORT/'fit_audit.csv',index=False)
    return frame


def rolling_factor_residual(y,x):
    """Joint past-only market/sector/DTE regression with fixed ridge shrinkage."""
    n=len(y);values=x.to_numpy(float);target=y.to_numpy(float)
    valid=np.isfinite(values).all(axis=1)&np.isfinite(target)
    x=x.where(pd.Series(valid,index=x.index),axis=0);y=y.where(valid)
    means=x.shift(1).rolling(252,min_periods=126).mean();ym=y.shift(1).rolling(252,min_periods=126).mean()
    covariance=np.zeros((n,3,3));cross=np.zeros((n,3))
    for j in range(3):
        cross[:,j]=(x.iloc[:,j]*y).shift(1).rolling(252,min_periods=126).mean()-means.iloc[:,j]*ym
        for k in range(3):
            covariance[:,j,k]=(x.iloc[:,j]*x.iloc[:,k]).shift(1).rolling(252,min_periods=126).mean()-means.iloc[:,j]*means.iloc[:,k]
    ready=np.isfinite(covariance).all(axis=(1,2))&np.isfinite(cross).all(axis=1)&valid
    diagonal=np.diagonal(covariance,axis1=1,axis2=2).copy()
    for j in range(3):covariance[:,j,j]+=.1*np.maximum(diagonal[:,j],1e-8)+1e-10
    beta=np.zeros((n,3));beta[ready]=np.linalg.solve(covariance[ready],cross[ready,:,None])[:,:,0]
    residual=target-ym.to_numpy()-np.sum(beta*(values-means.to_numpy()),axis=1)
    residual[~ready]=np.nan
    return pd.Series(residual,index=y.index)


def factor_residuals(frame,cal):
    # These are lagged-model forecast values, not future realized premiums.
    p=frame.pivot(index='date',columns='secid',values='forecast_vrp').reindex(cal)
    dt=frame.pivot(index='date',columns='secid',values='dte').reindex(cal)
    groups=frame.groupby('secid').industry.first();rows=[]
    for sid in p.columns:
        others=p.drop(columns=sid);market=others.median(axis=1).where(others.notna().sum(axis=1)>=20)
        peers=[int(s) for s in groups.index if s!=sid and groups[s]==groups[sid] and groups[sid]!='other']
        sector=p[peers].median(axis=1).where(p[peers].notna().sum(axis=1)>=3) if peers else pd.Series(np.nan,index=cal)
        # Missing sector factor: zero with a separate coverage output, no future filling.
        x=pd.DataFrame(dict(market=market,sector=sector.fillna(0),dte=dt[sid]/365),index=cal)
        residual=rolling_factor_residual(p[sid],x)
        scale=p[sid].shift(1).rolling(252,min_periods=126).std().where(lambda x:x>1e-8)
        rows.append(pd.DataFrame(dict(date=cal,secid=sid,factor_vrp_residual=-residual/scale,sector_factor_available=sector.notna())))
    return frame.merge(pd.concat(rows,ignore_index=True),on=['date','secid'],validate='one_to_one')


def pair_signals(outcomes,forecasts):
    f=forecasts.set_index(['date','secid']);result=outcomes.copy()
    for suffix in ['a','b']:
        index=pd.MultiIndex.from_arrays([result.date,result['secid_'+suffix]])
        part=f.reindex(index).reset_index(drop=True)
        for signal in SIGNALS:result[signal+'_'+suffix]=part[signal].to_numpy()
        result['leg_end_'+suffix]=part.end.to_numpy()
    # Pair and component outcomes must have the SAME selected contract horizons.
    present=result.leg_end_a.notna()&result.leg_end_b.notna()
    assert (result.loc[present,['leg_end_a','leg_end_b']].max(axis=1).to_numpy()==result.loc[present,'end'].to_numpy()).all()
    for signal in SIGNALS:result[signal]=result[signal+'_a']-result[signal+'_b']
    result=result[result.date.ge('2018-01-01')].sort_values(['pair','date']).reset_index(drop=True)
    result['nonoverlap']=False
    for _,g in result.groupby('pair'):
        after=pd.Timestamp.min
        for row in g.itertuples():
            if row.date>after:result.loc[row.Index,'nonoverlap']=True;after=row.end
    result.to_parquet(OUT/'pair_signals.parquet',index=False)
    return result


def within_pair_corr(g,signal):
    g=g.dropna(subset=[signal,'directional_edge']).copy()
    g=g[g.groupby('pair').pair.transform('size')>=20]
    if len(g)<100:return np.nan
    a=g.groupby('pair')[signal].rank(pct=True);b=g.groupby('pair').directional_edge.rank(pct=True)
    a-=a.groupby(g.pair).transform('mean');b-=b.groupby(g.pair).transform('mean')
    w=1/g.groupby('pair').pair.transform('size')
    den=np.sqrt((w*a*a).sum()*(w*b*b).sum())
    return float((w*a*b).sum()/den) if den>1e-12 else np.nan


def diagnostics(data,forecasts):
    rows=[];annual=[];bins=[];accuracy=[];rules=[];pair_rows=[]
    early=data[(data.date<'2022-01-01')&(data.end<'2022-01-01')]
    for signal in SIGNALS:
        training=early.dropna(subset=[signal]);threshold=training[signal].abs().quantile(.9) if len(training)>=100 else np.nan
        for period,g in [('early_completed',early),('later',data[data.date>='2022-01-01']),('all',data)]:
            for sample,h in [('all',g),('nonoverlap',g[g.nonoverlap])]:
                h=h.dropna(subset=[signal]);positive=h[signal]>0
                pnl=np.where(positive,h.long_net,h.short_net)
                selected=h[signal].abs()>=threshold
                rows.append(dict(signal=signal,period=period,sample=sample,observations=len(h),pairs=h.pair.nunique(),
                    pooled_rank_corr=h[signal].corr(h.directional_edge,method='spearman'),
                    within_pair_rank_corr=within_pair_corr(h,signal),
                    all_direction_mean_net=float(np.mean(pnl)) if len(h) else None,
                    early_fixed_extreme_threshold=threshold,extreme_trades=int(selected.sum()),
                    extreme_mean_net=float(np.mean(pnl[selected])) if selected.any() else None,
                    extreme_mean_net_per_opportunity=float(np.where(selected,pnl,0.).mean()) if len(h) else None))
                if sample=='all' and period=='later':
                    for year,v in h.groupby(h.date.dt.year):
                        chosen=v[signal].abs()>=threshold;net=np.where(v[signal]>0,v.long_net,v.short_net)
                        annual.append(dict(signal=signal,year=year,observations=len(v),within_pair_rank_corr=within_pair_corr(v,signal),
                            extreme_trades=int(chosen.sum()),extreme_mean_net=float(np.mean(net[chosen])) if chosen.any() else None))
                    for pair,v in h.groupby('pair'):
                        chosen=v[signal].abs()>=threshold;net=np.where(v[signal]>0,v.long_net,v.short_net)
                        pair_rows.append(dict(signal=signal,pair=pair,label=v.label.iloc[0],observations=len(v),extreme_trades=int(chosen.sum()),
                            extreme_mean_net=float(np.mean(net[chosen])) if chosen.any() else None))
                    # Thresholds, bin boundaries and economic signs frozen using early outcomes only.
                    edges=np.unique(training[signal].quantile(np.linspace(0,1,6)).to_numpy())
                    if len(edges)>=3:
                        boundaries=edges[1:-1];groups=np.searchsorted(boundaries,h[signal],side='right')
                        for k in range(len(boundaries)+1):
                            z=h.iloc[np.flatnonzero(groups==k)]
                            bins.append(dict(signal=signal,bin=k,observations=len(z),mean_directional_edge=float(z.directional_edge.mean()),
                                mean_long_net=float(z.long_net.mean()),mean_short_net=float(z.short_net.mean())))
    for period,g in [('early',forecasts[forecasts.date.between('2018-01-01','2021-12-31')]),('later',forecasts[forecasts.date>='2022-01-01'])]:
        g=g[g.target.gt(0)]
        for model,col in [('iv','iv'),('rv21','rv21'),('ewma10','ewma10'),('har_anchor','har_forecast'),('expanded_anchor','expanded_forecast')]:
            f=g[col]**2 if model=='iv' else g[col];valid=np.isfinite(f)&(f>0)
            z=g.loc[valid];f=f.loc[valid];ratio=z.target/f
            accuracy.append(dict(period=period,forecast=model,legs=len(z),mse=float(((z.target-f)**2).mean()),
                qlike=float((ratio-np.log(ratio)-1).mean()),mean_forecast_variance=float(f.mean()),mean_realized_variance=float(z.target.mean()),
                within_name_rank_corr=within_pair_corr(z.rename(columns={'secid':'pair','target':'directional_edge'}),col)))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'signals.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(bins).to_csv(REPORT/'early_fixed_bins.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT/'pairs.csv',index=False);pd.DataFrame(accuracy).to_csv(REPORT/'forecast_accuracy.csv',index=False)
    fits=pd.read_csv(REPORT/'fit_audit.csv')
    assert (pd.to_datetime(fits.last_training_end)<pd.to_datetime(fits.first_test_date)).all()
    later=result[(result.period=='later')&result['sample'].eq('all')].sort_values('within_pair_rank_corr',ascending=False)
    stable=[]
    years=pd.DataFrame(annual)
    for signal in SIGNALS:
        r=later[later.signal.eq(signal)].iloc[0]
        e=result[(result.signal==signal)&result.period.eq('early_completed')&result['sample'].eq('all')].iloc[0]
        thin=result[(result.signal==signal)&result.period.eq('later')&result['sample'].eq('nonoverlap')].iloc[0]
        y=years[years.signal==signal]
        stable.append(dict(signal=signal,early_within_pair_corr=e.within_pair_rank_corr,later_within_pair_corr=r.within_pair_rank_corr,
            later_nonoverlap_corr=thin.within_pair_rank_corr,positive_later_years=int(y.within_pair_rank_corr.gt(0).sum()),
            later_years=len(y),later_extreme_mean_net=r.extreme_mean_net,nonoverlap_extreme_mean_net=thin.extreme_mean_net,
            later_extreme_trades=r.extreme_trades))
    stability=pd.DataFrame(stable);stability.to_csv(REPORT/'stability.csv',index=False)
    notes=[
        'Current-cost ATM signal mining, before RL selects trades',
        'No RL or NN policy refits. All original 345 sleeves counted in coverage; only evaluable eligible actual-contract cohorts enter payoff diagnostics.',
        'Actual costs: $0.03/option contract, mid + 0.1-vol-point entry penalty, original stock costs, borrow and financing. ATM30 calls, next-session execution, daily delta hedging and own expiry settlement.',
        'Forecasts: quarterly, preceding five years, outcomes completed strictly before the fit, unique asset/date/contract rows, equal training weight per name, fixed ridge shrinkage and train-only transformations.',
        'RV target excludes the entry-day move. IV-anchored models predict log(realized variance / IV squared), with training-only smearing. The direct leg model forecasts symmetric directional actual P&L per initial vega; it is a diagnostic regression, not an NN policy.',
        'OHLC predictors: trailing variance, EWMA, Parkinson range plus overnight variance, overnight/downside/jump fractions. These are daily bars, not intraday realized volatility.',
        'Gamma-horizon edge adjusts annualized variance for known remaining trading sessions and calendar tenor; it is an approximation, not exact gamma attribution.',
        'Historical VRP uses each name history. Joint factor residuals use past-only 252-session market/industry/DTE fits; market means leave-one-name-out cross-sectional median, not SPX IV. Sector grouping uses historical roster industry; missing factor marked unavailable, no future filling.',
        'Within-pair correlations remove between-pair ranking effects and weight pairs equally. Economic directions remain fixed; signs are not flipped to make later results positive.',
        'Extreme thresholds and quintile boundaries use only completed 2018-2021 observations, then remain frozen for 2022-Aug2025 diagnostics. Earlier training models may use 2010 onwards.',
        'Forecast/factor membership uses all signal-ready names, independent of future entry/quote/return availability. Labels remain missing for unavailable outcomes; their resolution conditions only the evaluation sample. Nonoverlap thinning does not remove shared-name dependence.',
        'This history has already been inspected; the later period is not a new untouched holdout. No winner is automatically promoted; no annualized portfolio Sharpe is computed from terminal payoffs.',
        '',stability.to_string(index=False),'',pd.DataFrame(accuracy).to_string(index=False)]
    (REPORT/'summary.txt').write_text('\n'.join(notes)+'\n')
    print(stability.to_string(index=False),flush=True);print(pd.DataFrame(accuracy).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    source=EMPLOYEE/'inputs.json';defs=json.loads(source.read_text())['definitions'];cfg=replace(config(),option_commission=.03)
    signature=dict(code_sha256=digest(Path(__file__)),reference_inputs_sha256=digest(source),settings=asdict(cfg),signals=SIGNALS,
        fixed_ridge_penalty=.1,fit_frequency='quarterly',screen_cutoff='2022-01-01',workers=1,
        source_states={str(p):digest(p) for p in EMPLOYEE.glob('*/states.parquet')})
    path=OUT/'inputs.json'
    if path.exists():assert json.loads(path.read_text())==signature,'Mining inputs changed'
    else:write_json(path,signature)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    if not (OUT/'prepared.json').exists():prepare(defs,Market(cal),cfg)
    if args.prepare_only:return
    legs=pd.read_parquet(OUT/'legs.parquet')
    forecasts=predictions(legs,cal) if not (OUT/'forecasts.parquet').exists() else pd.read_parquet(OUT/'forecasts.parquet')
    outcomes=pd.read_parquet(OUT/'outcomes.parquet')
    signals=pair_signals(outcomes,forecasts);diagnostics(signals,forecasts)
    write_json(OUT/'status.json',dict(status='complete',original_roster=len(defs),signal_cohorts=len(signals),unique_asset_cohorts=len(legs),
        rl_refits=0,nn_refits=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
