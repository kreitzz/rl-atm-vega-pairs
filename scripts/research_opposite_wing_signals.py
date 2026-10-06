"""Independent opposite-wing signal discovery and fixed-size rule benchmarks."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.analyze_tail_universe_change import BASE, working_inputs as frozen_working_inputs, memory_filter
from scripts.explore_opposite_wing import SILOS as OLD_SILOS
from scripts.curve_trading.config import DATA, config, COMBOS, RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.retrain_atm_commission import fast_memory, cheaper_units
from scripts.repair_tail_put25_pooled import EVENTS, save
from scripts.curve_trading.policy import Replay
from scripts.option_flex.pooled_tail_engine import simulate
from scripts.run_tail_put25_legacy_rl import CONTROL
from scripts.tail_feature_exploration import past_z, scheduled_nonoverlap, ranked_statistics
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.select_atm_contracts import load_calendar
from scripts.run_contract_walkforward import digest
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit, scores

OUT = Path('artifacts/contracts/opposite_wing_signals')
REPORT = Path('reports/opposite_wing_signals')
SILOS = [('P',75),('C',25)]
RULES = ['spread', 'spread_native_memory']
FEATURES = ['wing_iv_gap', 'wing_iv_rank', 'wing_atm_gap', 'wing_atm_z_gap',
    'right_steepness_gap', 'wing_term_gap', 'risk_reversal_gap', 'butterfly_gap',
    'har_vrp_gap', 'vol_factor_residual_gap', 'vol_factor_residual_rank',
    'upside_variance_share_gap', 'realized_skew_gap', 'realized_kurt_gap']
CONTROLS = ['beta','correlation','dte_gap','delta_gap','execution_cost_vol']
FX=Path('data/raw/opposite_wing_DEXCAUS.csv')
CORPORATE_EVENTS=[dict(e,secid=114726) if e['secid']==109956 and e['effective']=='2020-04-15' else dict(e) for e in EVENTS]


class AccountingReplay(Replay):
    def run(self,codes,calendar):
        import scripts.curve_trading.policy as policy
        original=policy.simulate
        policy.simulate=lambda *args,**kwargs:simulate(*args,**kwargs,corporate_events=CORPORATE_EVENTS)
        try:return super().run(codes,calendar)
        finally:policy.simulate=original


def working_inputs(assets,secids):
    q,stock,sel,divs=frozen_working_inputs(assets,secids)
    # Additional cases discovered by the opposite wing, leaving all frozen
    # experiments untouched. FX is fixed at its last observation strictly
    # before declaration; no future exchange rate enters a price or cash flow.
    if not set(secids).intersection({105003,127149}):return q,stock,sel,divs
    from scripts.option_flex.black_scholes import fit_quotes
    dist=pd.read_parquet(ROOT/'distributions.parquet').copy()
    for col in ['declare_date','ex_date']:dist[col]=pd.to_datetime(dist[col])
    exact=dist.cancel_flag.eq('0')&dist.liquid_flag.eq('0')&dist.approx_flag.eq('0')&dist.adj_factor.eq(1)&dist.amount.ge(0)
    supplemental=exact&dist.secid.eq(105003)&dist.distr_type.eq('5')&dist.currency.eq('USD')
    foreign=exact&dist.secid.eq(127149)&dist.distr_type.isin(['1','5'])&dist.currency.eq('CAD')&dist.declare_date.ge('2010-01-01')
    dist.loc[supplemental,'distr_type']='1'
    fx=pd.read_csv(FX);fx['observation_date']=pd.to_datetime(fx.observation_date)
    fx['DEXCAUS']=pd.to_numeric(fx.DEXCAUS,errors='coerce');fx=fx.dropna().sort_values('observation_date')
    events=dist.loc[foreign,['declare_date']].sort_values('declare_date')
    if len(events):
        rates=pd.merge_asof(events,fx,left_on='declare_date',right_on='observation_date',
            direction='backward',allow_exact_matches=False)
        assert rates.DEXCAUS.gt(0).all() and (rates.observation_date<rates.declare_date).all()
        dist.loc[events.index,'amount']=dist.loc[events.index,'amount'].to_numpy()/rates.DEXCAUS.to_numpy()
        dist.loc[foreign,'currency']='USD';dist.loc[foreign,'distr_type']='1'
    additions=dist.loc[supplemental|foreign].copy()
    divs=pd.concat([divs,additions],ignore_index=True)
    rows=q.secid.isin([105003,127149])&q.bs_status.eq('unsupported_distribution') if len(q) else pd.Series(dtype=bool)
    if rows.any():
        fitted=fit_quotes(q.loc[rows].drop(columns=['close','dte'],errors='ignore'),stock,
            pd.read_parquet(ROOT/'zero_curve.parquet'),dist)
        for col in ['bs_iv','bs_rate','known_dividend_pv','bs_status']:q.loc[rows,col]=fitted[col].to_numpy()
        q.loc[rows,'delta']=fitted.bs_delta.to_numpy();q.loc[rows,'vega']=fitted.bs_vega.to_numpy()
    return q,stock,sel,divs


def har_forecast(returns, horizon=20):
    """Monthly HAR-style variance regression, purging incomplete forward RV."""
    returns = returns.astype(float); cal = returns.index
    square = returns**2 * 252
    x = pd.DataFrame({'rv1':square,'rv5':square.rolling(5,min_periods=5).mean(),
                      'rv20':square.rolling(20,min_periods=20).mean()})
    target = square.rolling(horizon,min_periods=horizon).mean().shift(-horizon)
    positions = np.arange(len(cal)); forecast = x.rv20.copy(); fitted = pd.Series(False,index=cal)
    records = []
    for month in cal.to_period('M').unique():
        cutoff = month.start_time; cutoff_position = cal.searchsorted(cutoff)
        known = (cal >= cutoff-pd.DateOffset(years=5)) & (positions+horizon < cutoff_position)
        good = known & np.isfinite(x).all(axis=1) & np.isfinite(target)
        test = cal.to_period('M') == month
        if good.sum() < 252: continue
        train = x.loc[good].to_numpy(); mean = train.mean(axis=0)
        scale = train.std(axis=0); scale[scale<1e-10] = 1.
        z = (train-mean)/scale; y = target.loc[good].to_numpy(); intercept = y.mean()
        coef = np.linalg.solve(z.T@z+np.eye(3), z.T@(y-intercept))
        forecast.loc[test] = np.maximum(0., intercept + ((x.loc[test].to_numpy()-mean)/scale)@coef)
        fitted.loc[test] = np.isfinite(x.loc[test]).all(axis=1)
        last = int(np.flatnonzero(good)[-1])+horizon
        assert cal[last] < cutoff
        records.append(dict(fit=str(cutoff.date()),training_rows=int(good.sum()),last_target_end=str(cal[last].date())))
    return forecast, fitted, records


def past_vol_beta(iv, vix):
    di, dv = iv.diff().shift(1), vix.diff().shift(1)
    return di.rolling(126,min_periods=42).cov(dv) / dv.rolling(126,min_periods=42).var().where(lambda v:v>1e-10)


def rule_codes(panel, cp, bucket, native_memory=False):
    idx = COMBOS.index((30,cp,bucket,bucket)); short, long = 1+idx*8, 1+idx*8+7
    rank = panel.wing_iv_rank
    good = panel.observed & panel.eligible & panel.legacy
    if native_memory: good &= memory_filter(panel)
    return pd.Series(np.where(good & rank.le(.2),long,np.where(good & rank.gt(.8),short,0)),
                     index=panel.index,name='code',dtype=int)


class SignalMarket:
    def __init__(self,market,cfg,sources,vix):
        self.market,self.cfg,self.sources,self.vix = market,cfg,sources,vix; self.cache={}

    def retain(self,secids):
        for sid in list(self.cache):
            if sid not in secids: del self.cache[sid]
        gc.collect()

    def asset(self,sid):
        if sid in self.cache: return self.cache[sid]
        cal=self.market.calendar; selected=[]; quotes=[]
        for cp,buckets in [('P',['atm','25','75']),('C',['atm','10','25'])]:
            parts=[]
            for year in cal.year.unique():
                path=DATA/'selection'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    if str(path) not in self.sources:self.sources[str(path)]=digest(path)
                    s=pd.read_parquet(path)
                    parts.append(s[s.bucket.isin(buckets) & s.tenor.isin([14,30,45])])
            if not parts:continue
            sel=pd.concat(parts,ignore_index=True);selected.append(sel)
            trade=sel[sel.tenor.eq(30)&sel.bucket.eq('75' if cp=='P' else '25')]
            ids=trade.optionid.unique().tolist()
            for year in cal.year.unique():
                path=DATA/'history'/cp/str(year)/f'{sid}.parquet'
                if ids and path.exists():
                    if str(path) not in self.sources:self.sources[str(path)]=digest(path)
                    quotes.append(pd.read_parquet(path,filters=[('optionid','in',ids)]))
        columns=['date','expiry','strike','bucket','cp_flag','tenor','secid','optionid','bs_iv','bs_delta','bs_vega','dte','bid','ask']
        sel=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(columns=columns)
        for c in ['date','expiry']:sel[c]=pd.to_datetime(sel[c])
        q=pd.concat(quotes,ignore_index=True) if quotes else pd.DataFrame()
        stock=self.market.stocks[self.market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        stock['secid']=sid;stock['rate']=self.market.rates;stock.index.name='date'
        divs=self.market.divs[self.market.divs.secid.eq(sid)]
        q,stock,sel,divs=working_inputs([dict(q=q,s=stock.reset_index(),sel=sel,divs=divs)],[sid])
        adjusted=stock.set_index('date').close*stock.set_index('date').cfadj
        returns=adjusted.pct_change(fill_method=None); square=returns**2
        forecast,fitted,fits=har_forecast(returns)
        frames={}
        for tenor in [14,30,45]:
            for cp,buckets in [('P',['atm','25','75']),('C',['atm','10','25'])]:
                for bucket in buckets:
                    f=sel[sel.tenor.eq(tenor)&sel.cp_flag.eq(cp)&sel.bucket.eq(bucket)].set_index('date').reindex(cal)
                    assert not f.index.has_duplicates
                    f['bs_iv']=f.bs_iv.where(f.bs_iv.gt(0));frames[tenor,cp,bucket]=f
        atm=frames[30,'C','atm']; sigma=atm.bs_iv
        def iv(cp,bucket):
            f=frames[30,cp,bucket];return f.bs_iv.where(f.expiry.eq(atm.expiry))
        common=dict(risk_reversal=(iv('C','25')-iv('P','25'))/sigma,
            butterfly=((iv('C','25')+iv('P','25'))/2-sigma)/sigma,
            right_steepness=(iv('C','10')-iv('C','25'))/sigma,
            upside_variance_share=square.where(returns>0,0.).where(returns.notna()).rolling(63,min_periods=63).sum()/square.rolling(63,min_periods=63).sum().where(lambda x:x>0),
            realized_skew=returns.rolling(63,min_periods=63).skew(),
            realized_kurt=returns.rolling(63,min_periods=63).kurt())
        wing={}
        for cp,bucket in SILOS:
            f=frames[30,cp,str(bucket)]; front=frames[14,cp,str(bucket)];back=frames[45,cp,str(bucket)]
            premium=iv(cp,str(bucket))/sigma-1; span=back.dte-front.dte
            bracket=front.expiry.lt(f.expiry)&back.expiry.gt(f.expiry)&span.gt(0)
            beta=past_vol_beta(f.bs_iv,self.vix)
            wing[cp,bucket]=pd.DataFrame(dict(iv=f.bs_iv,wing_atm=premium,wing_atm_z=past_z(premium),
                wing_term=((back.bs_iv-front.bs_iv)*30/span/sigma).where(bracket),
                har_vrp=f.bs_iv**2-forecast,vol_factor_residual=f.bs_iv-beta*self.vix,
                dte=f.dte,delta=f.bs_delta.abs(),expiry=f.expiry,
                quoted_width_vol=(f.ask-f.bid)/f.bs_vega.where(f.bs_vega>0),
                execution_cost_vol=.1+.03/(100*f.bs_vega.where(f.bs_vega>0)),
                **common),index=cal)
        trade=sel[sel.tenor.eq(30)&((sel.cp_flag.eq('P')&sel.bucket.eq('75'))|(sel.cp_flag.eq('C')&sel.bucket.eq('25')))]
        unit=UnitPaths(q,stock,trade,cal,cal,divs,self.cfg) if len(q) else None
        result=dict(q=q,s=stock,sel=trade,divs=divs,unit=unit,wing=wing,fits=fits,
            har_variance=forecast,har_fitted=fitted,returns=returns,vix=self.vix)
        self.cache[sid]=result;return result


def pair_state(a,b,cp,bucket,screen,legacy):
    fa,fb=a['wing'][cp,bucket],b['wing'][cp,bucket]
    out=pd.DataFrame(index=fa.index)
    out['wing_iv_gap']=fa.iv-fb.iv
    out['wing_iv_rank']=out.wing_iv_gap.rolling(252,min_periods=63).rank(pct=True)
    for feature in FEATURES:
        if feature.endswith('_gap') and feature!='wing_iv_gap':
            name=feature[:-4];out[feature]=fa[name]-fb[name]
    out['vol_factor_residual_rank']=out.vol_factor_residual_gap.rolling(252,min_periods=63).rank(pct=True)
    out['observed']=fa.iv.gt(0)&fb.iv.gt(0)
    for c in ['beta','correlation','eligible']:out[c]=screen[c]
    out['eligible']=out.eligible.fillna(False).astype(bool);out['legacy']=out.index.isin(legacy)
    out['dte_gap']=abs(fa.dte-fb.dte);out['delta_gap']=abs(fa.delta-fb.delta)
    out['execution_cost_vol']=fa.execution_cost_vol+fb.execution_cost_vol
    out['quoted_width_vol_max']=pd.concat([fa.quoted_width_vol,fb.quoted_width_vol],axis=1).max(axis=1,skipna=False)
    out['planned_expiry']=pd.concat([fa.expiry,fb.expiry],axis=1).max(axis=1,skipna=False)
    out['har_fitted_a']=a['har_fitted'];out['har_fitted_b']=b['har_fitted']
    out['vix']=a['vix'];out['iv_mean']=(fa.iv+fb.iv)/2
    return out.replace([np.inf,-np.inf],np.nan)


def prepare_replay(defs,market,cfg,sources,vix):
    cal=market.calendar;evalcal=cal[cal>='2018-01-01'];stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    loader=SignalMarket(market,cfg,sources,vix);cash=cash_book(evalcal,market.rates,cfg.capital)
    coverage=[];forecast_logs=json.loads((OUT/'har_fits.json').read_text()) if (OUT/'har_fits.json').exists() else {}
    training=replace(cfg,option_commission=.65)
    for n,d in enumerate(defs,1):
        folder=OUT/'pairs'/d['pair'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'coverage.json').exists():
            coverage.extend(json.loads((folder/'coverage.json').read_text()));continue
        source=BASE/d['pair']/'states.parquet'
        old=CONTROL/d['pair']/'legacy_calendar_memory/actions.csv'
        legacy=read_actions(old).index if old.exists() else pd.DatetimeIndex([])
        loader.retain(d['secids']);a,b=[loader.asset(sid) for sid in d['secids']]
        for sid,x in zip(d['secids'],[a,b]):forecast_logs[str(sid)]=x['fits']
        if source.exists():screen=pd.read_parquet(source,columns=['beta','correlation','eligible']).reindex(cal)
        else:
            from scripts.contract_learning_labels import beta_screen
            screen=beta_screen(pd.concat([a['returns'],b['returns']],axis=1),cal,beta_tolerance=.15)
        engine=stock=None;rows=[]
        for cp,bucket in SILOS:
            silo=f'{cp}{bucket}';dest=folder/silo;dest.mkdir(exist_ok=True)
            panel=pair_state(a,b,cp,bucket,screen,legacy)
            panel['long_net']=np.nan;panel['short_net']=np.nan;panel['resolved']=False
            panel['end']=pd.NaT;panel['status']='missing_history'
            for days in [91,365]:
                panel[f'profit_{days}d']=0.;panel[f'count_{days}d']=0
            if a['unit'] is not None and b['unit'] is not None:
                combo=(30,cp,bucket,bucket)
                daily,ends,vegas,status=pair_labels(a['unit'],b['unit'],combo,cfg)
                for name,j in [('long_net',-1),('short_net',0)]:
                    panel[name]=np.divide(daily[:,j].sum(axis=1)*cfg.entry_vega_per_leg,vegas[:,j],out=np.full(len(cal),np.nan),where=vegas[:,j]>0)
                panel['resolved']=np.isfinite(panel[['long_net','short_net']]).all(axis=1)&(vegas[:,[0,-1]]>0).all(axis=1)
                panel['end']=cal[ends];panel['status']=status;del daily
                units=[cheaper_units(x['unit'],training) for x in [a,b]]
                tdaily,tends,tvegas,tstatus=pair_labels(*units,combo,training)
                reward=np.divide(tdaily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,tvegas[:,[0,-1]],out=np.full((len(cal),2),np.nan),where=tvegas[:,[0,-1]]>0)
                memory=fast_memory(cal,cal[tends],reward,tvegas[:,[0,-1]]>0)
                long=panel.wing_iv_rank.le(.2)
                for days in [91,365]:
                    for name in ['profit','count']:
                        panel[f'{name}_{days}d']=np.where(long,memory[f'{name}_long_{days}d'],memory[f'{name}_short_{days}d'])
                del tdaily,units,memory
            panel['directional_edge']=(panel.long_net-panel.short_net)/2
            panel['nonoverlap']=scheduled_nonoverlap(cal,panel.planned_expiry,panel.observed&panel.eligible&panel.legacy)
            panel.to_parquet(dest/'states.parquet')
            wanted=(cal>='2018-01-01')&(cal<=stop)
            rows.append(dict(pair=d['pair'],label=d['label'],silo=silo,
                observed_rows=int((panel.observed&wanted).sum()),
                eligible_rows=int((panel.observed&panel.eligible&wanted).sum()),
                legacy_rows=int((panel.observed&panel.eligible&panel.legacy&wanted).sum()),
                resolved_legacy=int((panel.resolved&panel.eligible&panel.legacy&wanted).sum()),
                unresolved_evaluation=int((panel.status.eq('unresolved')&wanted).sum())))
            for rule in RULES:
                target=dest/rule;full=target/'full';full.mkdir(parents=True,exist_ok=True)
                actions=rule_codes(panel,cp,bucket,rule.endswith('memory')).loc[wanted]
                actions.to_csv(target/'actions.csv')
                if (full/'status.json').exists():continue
                if not actions.ne(0).any():
                    write_json(full/'status.json',dict(status='complete',cash_only=True,entries=0));continue
                if engine is None:
                    quotes,stock,selections,divs=[pd.concat([x[c] for x in [a,b]],ignore_index=True) for c in ['q','s','sel','divs']]
                    engine=AccountingReplay(quotes,stock,selections,d['secids'],divs,cfg)
                nav,trades,decisions=engine.run(actions,evalcal);mark_book(nav,market.rates,cfg.capital)
                if trades.empty:
                    checks=audit(nav,trades,stock,cfg,d['secids']);np.testing.assert_allclose(nav.nav,cash.nav,atol=1e-7,rtol=0)
                    decisions.to_csv(full/'decisions.csv',index=False)
                    write_json(full/'status.json',dict(status='complete',cash_only=True,entries=0,**checks))
                else:save(full,nav,trades,decisions,cfg,stock,d)
        write_json(folder/'coverage.json',rows);coverage.extend(rows)
        write_json(OUT/'market_sources.json',sources);write_json(OUT/'har_fits.json',forecast_logs)
        write_json(OUT/'progress.json',dict(stage='prepare_and_replay',pair=n,roster=len(defs),label=d['label'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if n%10==0:print(f'Opposite signal/replay {n}/{len(defs)} {d["label"]}',flush=True)
        del engine,stock,a,b,panel;gc.collect()
    loader.cache.clear();gc.collect();pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)


def discover(defs):
    correlations=[];pairs=[];quantiles=[];availability=[];profitability=[]
    stop=pd.Timestamp('2025-08-29')-pd.Timedelta(days=RUNOFF)
    for cp,bucket in SILOS:
        silo=f'{cp}{bucket}';parts=[]
        for d in defs:
            f=pd.read_parquet(OUT/'pairs'/d['pair']/silo/'states.parquet')
            f=f.loc[(f.index>='2018-01-01')&(f.index<=stop)].copy()
            f['date']=f.index;f['pair']=d['pair'];f['label']=d['label'];parts.append(f.reset_index(drop=True))
        data=pd.concat(parts,ignore_index=True);del parts
        for period,mask in [('pre2022_completed',data.end.lt('2022-01-01')),('2022plus_entries',data.date.ge('2022-01-01'))]:
            frame=data[mask&data.resolved&data.eligible]
            for sample,subset in [('broad',frame),('legacy',frame[frame.legacy]),
                                   ('nonoverlap_legacy',frame[frame.legacy&frame.nonoverlap])]:
                for feature in FEATURES:
                    stats,bypair=ranked_statistics(subset,feature,controls=CONTROLS)
                    correlations.append(dict(silo=silo,period=period,sample=sample,feature=feature,**stats))
                    pairs.extend(dict(silo=silo,period=period,sample=sample,feature=feature,**r) for r in bypair)
                    availability.append(dict(silo=silo,period=period,sample=sample,feature=feature,
                        total_rows=len(subset),available_rows=int(subset[feature].notna().sum()),
                        available_pairs=int(subset.loc[subset[feature].notna(),'pair'].nunique())))
                    valid=subset.dropna(subset=[feature,'directional_edge']).copy()
                    if len(valid):
                        pct=valid.groupby('pair')[feature].rank(pct=True)
                        valid['bin']=np.minimum(4,np.floor(pct*5).astype(int))
                        for q,g in valid.groupby('bin'):
                            quantiles.append(dict(silo=silo,period=period,sample=sample,feature=feature,quintile=int(q)+1,
                                observations=len(g),pairs=int(g.pair.nunique()),
                                mean_directional_edge=float(g.groupby('pair').directional_edge.mean().mean())))
                # Entry quality is a different question from choosing direction.
                extreme=subset[(subset.wing_iv_rank.le(.2)|subset.wing_iv_rank.gt(.8))].copy()
                direction=np.where(extreme.wing_iv_rank.le(.2),1.,-1.)
                extreme['directional_edge']=np.where(direction>0,extreme.long_net,extreme.short_net)
                for feature in ['wing_iv_gap','wing_atm_gap','right_steepness_gap','wing_term_gap',
                                'har_vrp_gap','vol_factor_residual_gap','upside_variance_share_gap']:
                    extreme['quality_signal']=-direction*extreme[feature]
                    stats,_=ranked_statistics(extreme,'quality_signal',controls=CONTROLS,min_observations=12)
                    profitability.append(dict(silo=silo,period=period,sample=sample,feature='oriented_'+feature,
                        entry_rows=len(extreme),**stats))
                for feature in ['vix','iv_mean']:
                    stats,_=ranked_statistics(extreme,feature,controls=CONTROLS,min_observations=12)
                    profitability.append(dict(silo=silo,period=period,sample=sample,feature=feature,
                        entry_rows=len(extreme),**stats))
        print(f'Signal discovery {silo} complete',flush=True)
        del data;gc.collect()
    result=pd.DataFrame(correlations);result.to_csv(REPORT/'signal_correlations.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'signals_by_pair.csv',index=False)
    pd.DataFrame(quantiles).to_csv(REPORT/'signal_quintiles.csv',index=False)
    pd.DataFrame(availability).to_csv(REPORT/'feature_coverage.csv',index=False)
    pd.DataFrame(profitability).to_csv(REPORT/'entry_profitability_signals.csv',index=False)
    return result


def summarize(defs,market,cfg):
    cal=market.calendar[market.calendar>='2018-01-01'];cash=cash_book(cal,market.rates,cfg.capital)
    rows=[];annual=[];pairs=[];books={};decision_rows=[];concentration=[]
    reference=pd.read_csv('reports/tail_put25_residual_rl/context_residual_rl_full_portfolio.csv',index_col=0,parse_dates=[0])
    for cp,bucket in SILOS:
        silo=f'{cp}{bucket}'
        for rule in RULES:
            model=f'{silo}_{rule}';book=cash[['nav','cash_benchmark']]*len(defs)
            entries=active=signals=breaches=0;costs=dict(commission=0.,option_cost=0.,hedge_cost=0.)
            for d in defs:
                folder=OUT/'pairs'/d['pair']/silo/rule/'full'
                status=json.loads((folder/'status.json').read_text());assert status['status']=='complete'
                nav=cash if status.get('cash_only') else pd.read_parquet(folder/'daily_nav.parquet')
                assert nav.index.equals(cal)
                book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                entries+=status['entries'];active+=int(status['entries']>0);breaches+=status.get('risk_breach_sessions',0)
                signals+=read_actions(folder.parent/'actions.csv').ne(0).sum()
                for key in costs:costs[key]+=status.get(key,0.)
                pairs.append(dict(model=model,pair=d['pair'],label=d['label'],entries=status['entries'],
                    net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                for name,count in status.get('decision_counts',{}).items():
                    decision_rows.append(dict(model=model,pair=d['pair'],decision=name,count=count))
            book['return_']=book.nav.pct_change().fillna(0)
            book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT/f'{model}_full_portfolio.csv');books[model]=book.excess_return
            excess=book.nav-book.cash_benchmark
            rows.append(dict(model=model,entries=entries,active_pairs=active,signals=int(signals),
                net_excess_pnl=float(excess.iloc[-1]),max_excess_drawdown_dollars=float((excess-excess.cummax()).min()),
                risk_breach_pair_sessions=breaches,correlation_with_put25_rl=float(book.excess_return.corr(reference.excess_return)),
                **costs,**scores(book,cfg.capital*len(defs))))
            for year,g in book.groupby(book.index.year):
                e=g.excess_return
                annual.append(dict(model=model,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*e.mean()/e.std()) if e.std()>1e-12 else None,
                    total_return_percent=100*((1+g.return_).prod()-1),excess_return_percent=100*((1+g.return_).prod()-(1+cash.return_.loc[g.index]).prod())))
            detail=pd.DataFrame(pairs);leaders=detail[detail.model.eq(model)].nlargest(3,'net_excess_pnl')
            stripped=book[['nav','cash_benchmark']].copy()
            for i,r in enumerate(leaders.itertuples(),1):
                if r.entries==0:continue
                nav=pd.read_parquet(OUT/'pairs'/r.pair/silo/rule/'full/daily_nav.parquet')
                stripped-=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                if i in [1,3]:
                    stripped['return_']=stripped.nav.pct_change().fillna(0)
                    stripped['excess_return']=stripped.return_-stripped.cash_benchmark.pct_change().fillna(0)
                    concentration.append(dict(model=model,winners_removed=i,
                        net_excess_pnl=float(stripped.nav.iloc[-1]-stripped.cash_benchmark.iloc[-1]),**scores(stripped,cfg.capital*len(defs))))
    pd.DataFrame(books).assign(put25_rl=reference.excess_return).corr().to_csv(REPORT/'policy_daily_correlations.csv')
    pd.DataFrame(rows).to_csv(REPORT/'strategy_comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'strategy_annual.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'strategy_pairs.csv',index=False)
    pd.DataFrame(decision_rows).to_csv(REPORT/'strategy_decisions.csv',index=False)
    pd.DataFrame(concentration).to_csv(REPORT/'strategy_concentration.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return pd.DataFrame(rows)


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((BASE/'inputs.json').read_text());defs=frozen['definitions']
    sources=dict(frozen['sources']);sources.update(json.loads((BASE/'market_sources.json').read_text()))
    for study in ['opposite_wing_exploration','tail_put25_residual_rl']:
        sources.update(json.loads(Path(f'reports/{study}/output_hashes.json').read_text()))
    for path in [Path(__file__),Path('tests/test_opposite_wing_signals.py'),Path('data/raw/VIX_History.csv'),FX]:sources[str(path)]=digest(path)
    for path,sha in sources.items():assert digest(path)==sha,f'Previous data changed: {path}'
    protocol='Separate per-pair 30D P75 and C25 attempts, not a joint strike/direction menu. Scan all 345 roster pairs, retaining each missing-feature coverage. Fourteen predeclared causal signals, completed actual-contract directional outcomes and entry-quality associations; report broad and original-calendar/ nonoverlapping samples before2022 and2022+. HAR-style variance forecast: monthly five-year regression of future20session variance on1/5/20session backward variance; minimum252 completed observations then ridge1 after standardization; prior RV20 fallback before sufficient global name history. Vol beta: prior126session IV/VIX change covariance, minimum42; signal-close VIX available before next-session execution. Two predefined fixed-size rules per wing: bottom/top252session spread-rank quintiles, min63 quotes, with/without native0.65-commission completed-profit history (positive91/365days, min21/63 completed observations). Actual commission0.03, mid+.1volpoint perleg, all original financing/hedge costs and inventory risk limits. Exact legacy pair/date calendar for trading comparisons; broad sample retained for discovery, no per-pair fitted-model veto. No RL fitting or strategy combination in this signal stage. Preserved0.756 rule/0.792 sizing references. No early assignment or ongoing option-vega rebalancing; each leg holds to own expiry except preannounced corporate-action paired exits. Shared-name risk not consolidated between separate pair sleeves. Inspected history, overlapping cohorts, exploratory multiple comparisons, no new holdout. A1Sharpe combined strategy is a research target, not a selected result.'
    write_json(OUT/'inputs.json',dict(features=FEATURES,silos=SILOS,rules=RULES,sources=sources,protocol=protocol))
    reference_audit=json.loads((OUT/'corporate_reference_audit.json').read_text())
    assert len(reference_audit)==14 and all(r['held_before_close']==0 and r['action_at_close']==0 for r in reference_audit)
    write_json(OUT/'corporate_accounting.json',dict(events=CORPORATE_EVENTS,
        correction='The inherited OIH reverse-split event incorrectly named secid109956 (SLB); OIH is secid114726 in the roster. Correct only this study; recompute all affected SLB and OIH pair books. Both frozen .756 rule and .792 sizing references were flat with zero incoming actions in all affected pairs at the corrected event close, so their reported NAVs are unaffected.',
        reference_audit='corporate_reference_audit.json',source='https://www.vaneck.com/globalassets/home/us/insights/media-coverage/press-releases/vaneck-announces-reverse-share-split-of-five-etfs.pdf'))
    write_json(OUT/'dividend_accounting.json',dict(
        repairs='FCX exact noncancelled USD supplemental cash dividends; TECK exact noncancelled CAD cash dividends converted to USD at the last available FX observation strictly before declaration. Apply the same constant declared USD proxy to option dividend PV and hedge-stock cash flows. These new secids first occur at roster position143, after the saved prefix, so the resumed prefix uses unchanged accounting.',
        limitation='Declared-rate CAD/USD conversion is a fixed-FX approximation; FX changes through actual payment and withholding are not modeled. FRED is a historical market-rate proxy, not a vintage-certified feed.',
        sources=['https://www.teck.com/news/news-releases/2022/teck-announces-dividend-en_ca',
                 'https://investors.fcx.com/investors/news-releases/news-release-details/2022/Freeport-McMoRan-Declares-Quarterly-Cash-Dividends-on-Common-Stock-6443e01de/default.aspx',
                 'https://fred.stlouisfed.org/series/DEXCAUS']))
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    raw=pd.read_csv('data/raw/VIX_History.csv');raw['DATE']=pd.to_datetime(raw.DATE)
    vix=raw.drop_duplicates('DATE').set_index('DATE').CLOSE.reindex(cal).ffill(limit=3)/100
    cfg=replace(config(),option_commission=.03);market=Market(cal)
    prepare_replay(defs,market,cfg,sources,vix)
    signal=discover(defs);strategies=summarize(defs,market,cfg)
    for fits in json.loads((OUT/'har_fits.json').read_text()).values():
        for r in fits:assert pd.Timestamp(r['last_target_end'])<pd.Timestamp(r['fit'])
    for path,sha in sources.items():assert digest(path)==sha,f'Previous data changed: {path}'
    record=dict(status='complete',roster_pairs=len(defs),strategy_books=len(defs)*len(SILOS)*len(RULES),
        source_files_verified=len(sources),rl_models_trained=0,strategies_combined=False,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(REPORT/'audit.json',record)
    (REPORT/'findings.txt').write_text(strategies.to_string(index=False)+'\n\n'+json.dumps(record,indent=2)+'\n\n'+protocol)
    hashes={str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
    hashes[str(Path(__file__))]=digest(Path(__file__));write_json(REPORT/'output_hashes.json',hashes)


if __name__=='__main__':main()
