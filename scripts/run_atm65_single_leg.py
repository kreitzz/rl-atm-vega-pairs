"""RL pair versus single-leg experiments on the frozen employee-sentiment best.

Same monthly fits, completed-outcome cutoffs and contract accounting. Single
guarded worker; baseline enabled refits are recovered before candidate replay.
"""
import argparse
from dataclasses import asdict,replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.run_atm65_ablation import features,read_actions,write_json,NN,CURVE,FROZEN,PERIODS
from scripts.run_atm65_employee_sentiment import OUT as EMPLOYEE,REPORT as EMPLOYEE_REPORT,ADDED as EMPLOYEE_ADDED,CODES
from scripts.single_leg_policy import ADDED,add_vrp,single_labels,single_conditions,action_masks,fit as fit_candidate,predict as predict_candidate,gate as gate_candidate,Replay as FlexibleReplay,MAX_SINGLE_NET_VEGA
from scripts.curve_trading.config import config,COMBOS
from scripts.curve_trading.policy import ATM,Replay,fit,predict,gate,memory_gates
from scripts.run_atm_cost_hurdles import make_units,reward_and_memory
from scripts.replay_frozen_atm_commission import load_pair,cash_book
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.run_curve_trading import RUNOFF,screen_dates
from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import scores,audit
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.retrain_atm_commission import check_fits

OUT=Path('artifacts/contracts/atm65_single_leg')
REPORT=Path('reports/atm65_single_leg')
MODELS=['rl_pair_vrp','rl_single_historical','rl_single_absolute']
WORKFLOW=Path('config/research_training.json')
CHECKPOINT=Path('config/atm30_fee065_employee_2026-10-01_checkpoint.json')


def save_result(dest,nav,trades,decisions,cfg,stocks,d):
    checks=audit(nav,trades,stocks,cfg,d['secids'])
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    if not trades.empty:
        assert set(trades.reason).issubset({'entry','split_adjustment','expiry_settlement'})
        paid=trades.reason.eq('entry')
        np.testing.assert_allclose(trades.loc[paid,'commission'],abs(trades.loc[paid,'quantity'])*cfg.option_commission)
        assert trades.loc[~paid,'commission'].eq(0).all()
    entered=decisions[decisions.status.eq('entered')] if len(decisions) else decisions
    if len(entered):
        paired=entered[entered.entry_kind.eq('pair')]
        assert (abs(paired.vega_a-paired.vega_b)/paired[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
        singles=entered[entered.entry_kind.ne('pair')]
        assert (singles[['vega_a','vega_b']].gt(0).sum(axis=1)==1).all()
        assert abs(singles.proposed_net_vega).le(MAX_SINGLE_NET_VEGA*singles.sizing_nav/cfg.capital+1e-8).all()
    nav.to_parquet(dest/'daily_nav.parquet');trades.to_csv(dest/'trades.csv',index=False)
    decisions.to_csv(dest/'decisions.csv',index=False)
    result=dict(status='complete',pair=d['pair'],label=d['label'],entries=len(entered),
        commission=float(trades.commission.sum()) if len(trades) else 0.,
        option_cost=float(nav.option_execution_cost.sum()),hedge_cost=float(nav.hedge_cost.sum()),
        **scores(nav,cfg.capital),**checks)
    write_json(dest/'status.json',result)
    return result


def configure():
    workflow=json.loads(WORKFLOW.read_text())
    if workflow['retrain_models'] != ['rl'] or workflow['retrain_nn']:
        raise ValueError('This runner supports RL-only retraining')


def employee_roster():
    c=pd.read_csv(EMPLOYEE_REPORT/'coverage.csv')
    return set(c.loc[c.retrain,'pair'])


def baseline_actions(pair,model,employee):
    assert model.startswith('rl')
    if employee:
        return EMPLOYEE/pair/'rl_sentiment'/'actions.csv'
    return CURVE/pair/'atm_control/actions.csv'


def baseline_results(pair,model,period,employee):
    assert model.startswith('rl')
    if employee: return EMPLOYEE/pair/'rl_sentiment'/period
    return FROZEN/pair/(period+'_003')



def prepare(defs,cal,employees):
    assets=[];panels={}
    for d in defs:
        path=EMPLOYEE/d['pair']/'states.parquet'
        if not path.exists():continue
        panel=pd.read_parquet(path);panels[d['pair']]=panel
        for sid,suffix in zip(d['secids'],['amd','nvda']):
            premium=(panel['iv_'+suffix]/100)**2-(panel['trailing_rv_'+suffix]/100)**2
            valid=panel['iv_'+suffix].gt(0)&panel['trailing_rv_'+suffix].ge(0)
            assets.append(pd.DataFrame(dict(date=panel.index,secid=sid,premium=premium.where(valid).to_numpy())))
    asset=pd.concat(assets,ignore_index=True).dropna(subset=['premium'])
    # Only agreeing same-name histories may contribute to a name's normalizer.
    sizes=asset.groupby(['date','secid']).premium.agg(['min','max'])
    ambiguous=sizes.index[(sizes['max']-sizes['min']).abs()>1e-8]
    bad=pd.MultiIndex.from_frame(asset[['date','secid']]).isin(ambiguous)
    asset=asset.loc[~bad].drop_duplicates(['date','secid'])
    normalizers={}
    for sid,g in asset.groupby('secid'):
        series=g.set_index('date').premium.reindex(cal)
        history=series.shift(1).rolling(252,min_periods=126)
        normalizers[sid]=pd.DataFrame(dict(mean=history.mean(),std=history.std()),index=cal)
    rows=[]
    for d in defs:
        folder=OUT/d['pair'];folder.mkdir(exist_ok=True)
        row=dict(pair=d['pair'],label=d['label'],employee_features=d['pair'] in employees,state_rows=0,
            observed_test_rows=0,observed_permitted_test_rows=0,retrain=False)
        if d['pair'] in panels:
            panel=panels[d['pair']];a,b=d['secids']
            state=add_vrp(panel,(normalizers[a],normalizers[b]));state.to_parquet(folder/'states.parquet')
            test=(panel.index>='2018-01-01')&(panel.index<=cal[-1]-pd.Timedelta(days=RUNOFF))&screen_dates(d,panel.index)
            observed=state.vrp_z_a_available.eq(1)&state.vrp_z_b_available.eq(1)
            permitted=panel.eligible.to_numpy(bool)
            row.update(state_rows=len(state),observed_test_rows=int((test&observed).sum()),
                observed_permitted_test_rows=int((test&observed&permitted).sum()),retrain=bool((test&permitted).any()))
        rows.append(row)
    coverage=pd.DataFrame(rows);coverage.to_csv(REPORT/'coverage.csv',index=False)
    write_json(OUT/'prepared.json',dict(pairs=len(defs),retrained_pairs=int(coverage.retrain.sum()),
        added_features=ADDED,normalization_sessions=252,minimum_prior_observations=126,
        ambiguous_name_dates_excluded=len(ambiguous),available_lag='Signal-close values; normalizers use strictly earlier sessions',
        reference='Frozen employee-sentiment 0.368 checkpoint; original 28 employee sleeves unchanged'))
    print(f'{coverage.retrain.sum()}/{len(defs)} sleeves have eligible dates for the controlled experiment.',flush=True)


def fit_labels(daily,mask,rule=None,conditions=None):
    labels=daily[mask].copy()
    if rule is not None:
        cheap,rich=conditions
        for code in range(9,25):
            offset=(code-9)%8;allowed=cheap if offset>=4 else rich
            labels[~allowed[mask],code,:]=0
    return labels


def train_pair(d,panel,daily,pair_finish,finish,gates,available,masks,cal,replay,original_replay,rates,folder,employee):
    stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted=(panel.index>='2018-01-01')&(panel.index<=stop)&screen_dates(d,panel.index)
    idx=panel.index[wanted];original=features(panel)+(EMPLOYEE_ADDED if employee else [])
    assert len(original)==(49 if employee else 47)
    path=baseline_actions(d['pair'],'rl',employee)
    saved=read_actions(path) if path.exists() else pd.Series(0,index=idx,dtype=int)
    assert saved.index.equals(idx)
    status_path=EMPLOYEE/d['pair']/'fits.csv' if employee else CURVE/d['pair']/'atm_control/fits.csv'
    try:
        baseline_table=pd.read_csv(status_path).set_index('fit')
        baseline_status=baseline_table['rl_sentiment_status' if employee else 'status']
    except (pd.errors.EmptyDataError,FileNotFoundError):baseline_status=pd.Series(dtype=str)
    actions=pd.DataFrame(0,index=idx,columns=MODELS,dtype=int);logs=[]
    checkpoint=folder/'month_checkpoint.json'
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text());assert old['dates']==[str(t) for t in idx]
        actions.iloc[:,:]=np.asarray(old['actions'],int);logs=old['fits']
    completed={r['fit'] for r in logs};positions=cal.get_indexer(panel.index)
    completion=cal[finish];pair_completion=cal[pair_finish]
    finite=np.isfinite(daily).all(axis=(1,2));pair_finite=np.isfinite(daily[:,:9]).all(axis=(1,2))
    columns=original+ADDED
    conditions={r:single_conditions(panel,r) for r in ['historical','absolute']}
    candidate_masks={'rl_pair_vrp':masks['historical'].copy(),
        'rl_single_historical':masks['historical'],'rl_single_absolute':masks['absolute']}
    candidate_masks['rl_pair_vrp'][:,9:]=False
    for month in pd.period_range('2018-01-01',stop,freq='M'):
        cutoff=month.start_time;fitdate=str(cutoff.date());test=(idx>=cutoff)&(idx<(month+1).start_time)
        if not test.any() or fitdate in completed:continue
        vstart=cutoff-pd.DateOffset(years=1)
        use=(panel.index>=vstart-pd.DateOffset(years=5))&(completion<vstart)&finite
        recent=(panel.index>=cutoff-pd.DateOffset(years=5))&(completion<cutoff)&finite
        vcal=cal[(cal>=vstart)&(cal<cutoff)]
        val=(panel.index>=vcal[0])&(panel.index<=vcal[-1]-pd.Timedelta(days=RUNOFF))
        dates=idx[test]
        row=dict(fit=fitdate,training_rows=int(use.sum()),recent_rows=int(recent.sum()),
            validation_start=str(vcal[0].date()),validation_end=str(vcal[-1].date()))
        # Recover the frozen enabled refit with its original pair-only cutoff.
        if baseline_status.get(fitdate)=='enabled':
            old_recent=(panel.index>=cutoff-pd.DateOffset(years=5))&(pair_completion<cutoff)&pair_finite
            own=daily[old_recent,:9].copy();own[~panel.loc[old_recent,'eligible'].to_numpy(bool)]=0
            model,_=fit(panel.loc[old_recent,original],own,positions[old_recent],pair_finish[old_recent],CODES)
            recovered=gate(predict(model,panel.loc[dates,original]),dates,panel,gates,available)
        else:recovered=pd.Series(0,index=dates,dtype=int)
        np.testing.assert_array_equal(recovered.to_numpy(),saved.loc[dates].to_numpy(),err_msg=f'{d["label"]} {fitdate}: baseline not recovered')
        row['baseline_reproduced']=True
        for name in MODELS:
            mask=candidate_masks[name];rule=None if name=='rl_pair_vrp' else name.removeprefix('rl_single_')
            codes=list(range(9 if rule is None else 25))
            opportunity=bool(mask[val,1:len(codes)].any())
            row[name+'_status']='insufficient_training' if use.sum()<504 or len(vcal)<200 else 'flat_no_validation_opportunities'
            if use.sum()<504 or len(vcal)<200 or not opportunity:continue
            labels=fit_labels(daily,use,rule,None if rule is None else conditions[rule])
            labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
            model,details=fit_candidate(panel.loc[use,columns],labels,positions[use],finish[use],codes)
            row['last_training_outcome']=str(completion[use].max().date());row.update({name+'_proposal_'+k:v for k,v in details.items()})
            proposal=gate_candidate(predict_candidate(model,panel.loc[val,columns]),panel.index[val],panel,mask)
            score=None;error=None
            try:
                nav,trades,_=replay.run(proposal,vcal);mark_book(nav,rates,replay.cfg.capital)
                audit(nav,trades,replay.s,replay.cfg,d['secids']);score=scores(nav,replay.cfg.capital)['annualized_excess_sharpe']
            except (UnresolvedPosition,ValueError) as exc:error=str(exc)
            enabled=score is not None and score>0 and recent.sum()>=504
            row[name+'_status']='enabled' if enabled else 'flat_validation_gate'
            row[name+'_validation_sharpe']=score;row[name+'_validation_error']=error
            if enabled:
                own=fit_labels(daily,recent,rule,None if rule is None else conditions[rule])
                own[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                model,details=fit_candidate(panel.loc[recent,columns],own,positions[recent],finish[recent],codes)
                actions.loc[dates,name]=gate_candidate(predict_candidate(model,panel.loc[dates,columns]),dates,panel,mask)
                row['last_recent_outcome']=str(completion[recent].max().date());row.update({name+'_refit_'+k:v for k,v in details.items()})
        logs.append(row);write_json(checkpoint,dict(dates=[str(t) for t in idx],actions=actions.to_numpy().tolist(),fits=logs))
        if month.month in [1,7]:print(f'  {d["label"]} {month}: '+', '.join(name+' '+row[name+'_status'] for name in MODELS),flush=True)
    table=pd.DataFrame(logs);check_fits(table);table.to_csv(folder/'fits.csv',index=False)
    for name in MODELS:
        dest=folder/name;dest.mkdir(exist_ok=True);a=actions[name].rename('code')
        pd.testing.assert_series_equal(a,gate_candidate(a,a.index,panel,candidate_masks[name]),check_names=False);a.to_csv(dest/'actions.csv')
    # Replay the frozen action path using the extension; require identical NAV.
    actual_cfg=replace(replay.cfg,option_commission=.03);old_cfg=replay.cfg;replay.cfg=actual_cfg
    local=saved.where(saved.eq(0),saved-ATM*8).astype(int)
    for period,start in PERIODS:
        folder_base=baseline_results(d['pair'],'rl',period,employee)
        status=json.loads((folder_base/'status.json').read_text());calendar=cal[cal>=start]
        if status.get('cash_only'):expected=cash_book(calendar,rates,actual_cfg.capital)
        else:expected=pd.read_parquet(status.get('nav_source',folder_base/'daily_nav.parquet'))
        nav,trades,_=replay.run(local[local.index>=start],calendar);mark_book(nav,rates,actual_cfg.capital)
        audit(nav,trades,replay.s,actual_cfg,d['secids'])
        np.testing.assert_allclose(nav.nav,expected.nav,atol=1e-7,rtol=0)
    replay.cfg=old_cfg;write_json(folder/'reference_replay_audit.json',dict(baseline_actions_reproduced=True,full_and_recent_nav_reproduced=True))
    return actions



def summary(defs,coverage,market,cfg,employees):
    selected=set(coverage.loc[coverage.retrain,'pair']);models=['rl_baseline',*MODELS]
    def source(d,m,p):
        return OUT/d['pair']/m/p if m in MODELS and d['pair'] in selected else baseline_results(d['pair'],m,p,d['pair'] in employees)
    statuses={(d['pair'],m,p):json.loads((source(d,m,p)/'status.json').read_text()) for d in defs for m in models for p,_ in PERIODS}
    complete=[d for d in defs if all(statuses[d['pair'],m,p]['status']=='complete' for m in models for p,_ in PERIODS)]
    unresolved=[dict(pair=d['pair'],model=m,period=p,error=statuses[d['pair'],m,p].get('error')) for d in defs for m in models for p,_ in PERIODS if statuses[d['pair'],m,p]['status']!='complete']
    write_json(REPORT/'accounting_coverage.json',dict(requested_pairs=len(defs),common_complete_pairs=len(complete),unresolved=unresolved))
    rows=[];annual=[];pairs=[]
    for universe in ['all','refit_covered']:
        roster=complete if universe=='all' else [d for d in complete if d['pair'] in selected]
        if not roster:continue
        for m in models:
            for p,start in PERIODS:
                cal=market.calendar[market.calendar>=start];cash=cash_book(cal,market.rates,cfg.capital)
                book=cash[['nav','cash_benchmark']]*len(roster);entries=active=0
                for d in roster:
                    status=statuses[d['pair'],m,p];folder=source(d,m,p)
                    nav=cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'),columns=['nav','cash_benchmark'])
                    assert nav.index.equals(cal);book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                    entries+=status['entries'];active+=int(status['entries']>0)
                    pairs.append(dict(universe=universe,model=m,period=p,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
                book.to_csv(REPORT/f'{universe}_{m}_{p}_portfolio.csv')
                rows.append(dict(universe=universe,model=m,period=p,pairs=len(roster),entries=entries,active_pairs=active,
                    net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),**scores(book,cfg.capital*len(roster))))
                for year,g in book.groupby(book.index.year):
                    sd=g.excess_return.std()
                    annual.append(dict(universe=universe,model=m,period=p,year=year,return_percent=100*((1+g.return_).prod()-1),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    if len(complete)==len(defs):
        saved=pd.read_csv(EMPLOYEE_REPORT/'comparison.csv')
        for m,old in [('rl_baseline','rl_sentiment')]:
            for p,_ in PERIODS:
                got=comparison[(comparison.universe=='all')&(comparison.model==m)&(comparison.period==p)].iloc[0]
                expected=saved[(saved.universe=='all')&(saved.model==old)&(saved.period==p)].iloc[0]
                np.testing.assert_allclose(got.annualized_excess_sharpe,expected.annualized_excess_sharpe,atol=1e-12);assert got.entries==expected.entries
    fit_tables=[pd.read_csv(OUT/d['pair']/'fits.csv').assign(pair=d['pair']) for d in defs if d['pair'] in selected]
    if fit_tables:pd.concat(fit_tables,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    notes=[
        'Controlled RL experiment: relative versus absolute volatility trades',
        'Frozen reference: annualized excess Sharpe 0.368 full / 0.428 recent.',
        'Add each leg IV squared minus trailing-RV squared, and its standardized deviation from its own prior 252-session history (minimum 126 observations), plus availability flags.',
        'Name histories are deduplicated across sleeves; ambiguous same-name observations are omitted from the normalizer. Each signal uses only earlier history.',
        'Trailing RV is an existing forecast proxy, not a new calibrated HAR forecast. No market/sector residualization has been added in this first ablation.',
        'Pair-only and both expanded menus share the same original eight-leaf relative-value tree partition, original state features and common purged training rows.',
        'Pair-only: flat or the original eight paired ladder choices. Expanded menus add eight signed sizes in either leg individually.',
        'Historical singles: both z-scores negative for a long, positive for a short. Absolute singles: both forecast-variance-minus-IV-squared edges positive for a long, negative for a short.',
        'Pairs retain their original completed-profitability and beta/correlation gates. Each single leg has its own completed-profitability gates (91/365-day windows, minimum 21/63 outcomes).',
        'The same masked data and monthly prior-year validation decide whether each candidate may trade; recent-five-year refits use only fully completed outcomes.',
        'Single-only label payoffs are zero on dates disallowed by its known signal condition. The pair label definition remains unchanged.',
        'ATM30 calls, daily delta hedging, held to each actual expiry. No new NN fits.',
        'Same $0.65 training/$0.03 execution commission, 0.1-vol-point entry penalty, stock costs, funding and stress/reserve/gross-vega limits.',
        'Standalone entry target is at most $450 per vol point, using the original 25/50/75/100% ladder. A new single entry is rejected if total projected net vega exceeds $900 per vol point per sleeve, scaled by NAV.',
        'The additional net-vega limit is an entry limit: held Greeks can drift; hold-to-expiry accounting is unchanged. Pair entries retain the original rules.',
        'All 345 original sleeves remain in the requested universe; both full (2018) and recent (2022) books independently start flat through August 2025.',
        'Expanded policies can earn or lose from outright volatility exposure. A Sharpe increase does not alone establish better relative-value signal quality.',
        'This history has been inspected repeatedly. No checkpoint is automatically promoted.',
        '',comparison.to_string(index=False),'',
        'Files: comparison.csv, annual.csv, pairs.csv, coverage.csv, fits.csv, accounting_coverage.json']
    (REPORT/'summary.txt').write_text('\n'.join(notes)+'\n')
    print(comparison[['universe','model','period','pairs','entries','annualized_excess_sharpe']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--limit',type=int)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    configure()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((EMPLOYEE/'inputs.json').read_text());defs=inputs['definitions'];employees=employee_roster()
    actual=replace(config(),option_commission=.03);training=replace(actual,option_commission=.65)
    assert asdict(actual)==inputs['actual']
    files=[Path(__file__),WORKFLOW,Path('scripts/single_leg_policy.py'),Path('scripts/option_flex/single_leg_engine.py'),CHECKPOINT,
        EMPLOYEE/'inputs.json',EMPLOYEE_REPORT/'coverage.csv',EMPLOYEE_REPORT/'comparison.csv',
        Path('scripts/curve_trading/policy.py'),Path('scripts/curve_trading/unit_paths.py'),
        Path('scripts/run_atm_cost_hurdles.py'),Path('scripts/run_atm_nn_cost_hurdles.py'),
        Path('scripts/run_contract_walkforward.py'),Path('scripts/option_flex/contract_engine.py')]
    for d in defs:
        files.extend(p for p in [EMPLOYEE/d['pair']/'states.parquet',NN/d['pair']/'fee_065/gates.npz',
            baseline_actions(d['pair'],'rl',d['pair'] in employees),
            EMPLOYEE/d['pair']/'fits.csv' if d['pair'] in employees else CURVE/d['pair']/'atm_control/fits.csv'] if p.exists())
    signature=dict(models=MODELS,workflow=json.loads(WORKFLOW.read_text()),definitions=defs,baseline_checkpoint=str(CHECKPOINT),actual=asdict(actual),training=asdict(training),
        added_features=ADDED,max_single_net_vega=MAX_SINGLE_NET_VEGA,code_and_sources={str(p):digest(p) for p in files})
    if (OUT/'inputs.json').exists():assert json.loads((OUT/'inputs.json').read_text())==signature,'Inputs changed'
    else:write_json(OUT/'inputs.json',signature)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    if not (OUT/'prepared.json').exists():prepare(defs,cal,employees)
    if args.prepare:return
    coverage=pd.read_csv(REPORT/'coverage.csv');selected=set(coverage.loc[coverage.retrain,'pair'])
    market=Market(cal);started=time.monotonic();newly=0
    if not args.report_only:
        for n,d in enumerate([x for x in defs if x['pair'] in selected],1):
            folder=OUT/d['pair']
            if all((folder/m/p/'status.json').exists() for m in MODELS for p,_ in PERIODS):continue
            pair_started=time.monotonic();print(f'{n}/{len(selected)} {d["label"]}: pair versus historical/absolute single legs',flush=True)
            panel=pd.read_parquet(folder/'states.parquet');q,s,selections,divs=load_pair(market,d['secids'])
            with np.load(NN/d['pair']/'fee_065/gates.npz') as z:
                gates=np.zeros((len(COMBOS),len(panel),2),bool);gates[ATM]=z['gates']
                available=np.zeros((len(COMBOS),len(panel)),bool);available[ATM]=z['available'];saved_ends=z['ends'].copy()
            units=make_units(panel,q,s,selections,divs,cal,training)
            labels,pair_finish,vegas,statuses,rewards,_=reward_and_memory(units,training)
            np.testing.assert_array_equal(pair_finish,saved_ends)
            np.testing.assert_array_equal(gates[ATM],memory_gates(panel.index,cal[pair_finish],rewards,vegas[:,[0,-1]]>0))
            singles=[single_labels(u,training) for u in units]
            daily=np.zeros((len(panel),25,labels.shape[-1]),dtype=np.float32);daily[:,1:9]=labels
            daily[:,9:17]=singles[0][0];daily[:,17:25]=singles[1][0]
            finish=np.maximum.reduce([pair_finish,singles[0][1],singles[1][1]])
            single_gates=[singles[0][3],singles[1][3]]
            masks={rule:action_masks(panel,gates[ATM],available[ATM],single_gates,rule) for rule in ['historical','absolute']}
            del units,labels,singles;gc.collect()
            replay=FlexibleReplay(q,s,selections,d['secids'],divs,training)
            original_replay=Replay(q,s,selections,d['secids'],divs,training)
            actions=train_pair(d,panel,daily,pair_finish,finish,gates,available,masks,cal,replay,original_replay,market.rates,folder,d['pair'] in employees)
            replay.cfg=actual
            for m in MODELS:
                for p,start in PERIODS:
                    dest=folder/m/p;dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'status.json').exists():continue
                    a=actions.loc[actions.index>=start,m]
                    if not a.ne(0).any():result=dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True)
                    else:
                        try:
                            nav,trades,decisions=replay.run(a,cal[cal>=start]);mark_book(nav,market.rates,actual.capital)
                            result=save_result(dest,nav,trades,decisions,actual,s,d)
                        except UnresolvedPosition as exc:result=dict(status='incomplete',pair=d['pair'],label=d['label'],error=str(exc))
                    result.update(training_hurdle=.65,execution_commission=.03);write_json(dest/'status.json',result)
            newly+=1
            write_json(OUT/'progress.json',dict(completed_retrained_pairs=n,retrained_pairs=len(selected),last_pair=d['label'],
                last_pair_seconds=time.monotonic()-pair_started,elapsed_seconds=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: completed in {time.monotonic()-pair_started:.1f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB',flush=True)
            del panel,q,s,selections,divs,replay,original_replay,daily,actions,masks,gates,available;gc.collect()
            if args.limit and newly>=args.limit:return
    summary(defs,coverage,market,actual,employees)
    write_json(OUT/'task_status.json',dict(status='complete',models=MODELS,pairs=len(defs),retrained_pairs=len(selected),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
