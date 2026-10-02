"""Two predeclared causal feature families and saved-policy exit diagnostics.

One pair at a time. Monthly resumable fits, reusable labels, unchanged reference.
Run only through the existing resource supervisor.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import resource
import time
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.atm65_mining_features import DYNAMICS, GAMMA, names, engineer, thin_completed
from scripts.run_atm65_ablation import (features, read_actions, write_json, NN, CURVE,
                                      FROZEN, PERIODS)
from scripts.curve_trading.config import config, COMBOS, ACTIONS
from scripts.curve_trading.policy import ATM, Replay, fit, predict, gate, memory_gates
from scripts.run_atm_cost_hurdles import make_units, reward_and_memory
from scripts.replay_frozen_atm_commission import load_pair, cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_curve_trading import RUNOFF, screen_dates
from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import scores, audit
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.option_flex.exit_engine import simulate as simulate_exits

OUT=Path('artifacts/contracts/atm65_signal_mining')
REPORT=Path('reports/atm65_signal_mining')
FAMILIES={'rl_dynamics':DYNAMICS, 'rl_dynamics_gamma':DYNAMICS+GAMMA}
EXITS={'exit_reverse':'reverse', 'exit_flat_or_reverse':'flat_or_reverse'}
MODELS=list(FAMILIES)+list(EXITS)
CODES=[0,*range(1+ATM*8,1+ATM*8+8)]

def fit_masks(index, completion, finite, cutoff):
    """Purge all outcomes crossing either the validation or refit boundary."""
    validation_start=cutoff-pd.DateOffset(years=1)
    proposal=(index>=validation_start-pd.DateOffset(years=5))&(completion<validation_start)&finite
    recent=(index>=cutoff-pd.DateOffset(years=5))&(completion<cutoff)&finite
    return proposal,recent

def train_pair(d,panel,daily,finish,gates,available,cal,replay,rates,folder):
    stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted=(panel.index>='2018-01-01')&(panel.index<=stop)&screen_dates(d,panel.index)
    idx=panel.index[wanted]
    saved=read_actions(CURVE/d['pair']/'atm_control/actions.csv')
    assert saved.index.equals(idx)
    actions=pd.DataFrame(0,index=idx,columns=list(FAMILIES),dtype=int)
    logs=[];checkpoint=folder/'month_checkpoint.json'
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text());assert old['dates']==[str(t) for t in idx]
        actions.loc[:,:]=np.asarray(old['actions'],dtype=int);logs=old['fits']
    completed={r['fit'] for r in logs}
    positions=cal.get_indexer(panel.index);completion=cal[finish]
    finite=np.isfinite(daily).all(axis=(1,2))
    for month in pd.period_range('2018-01-01',stop,freq='M'):
        cutoff=month.start_time;fitdate=str(cutoff.date())
        test=(idx>=cutoff)&(idx<(month+1).start_time)
        if not test.any() or fitdate in completed:continue
        use,recent=fit_masks(panel.index,completion,finite,cutoff)
        vcal=cal[(cal>=cutoff-pd.DateOffset(years=1))&(cal<cutoff)]
        val=(panel.index>=vcal[0])&(panel.index<=vcal[-1]-pd.Timedelta(days=RUNOFF))
        row=dict(fit=fitdate,training_rows=int(use.sum()),recent_rows=int(recent.sum()),
                 validation_start=str(vcal[0].date()),validation_end=str(vcal[-1].date()))
        opportunity=(panel.eligible.to_numpy(bool)&available[ATM]&gates[ATM].any(axis=1)&val).any()
        for name,new in FAMILIES.items():
            columns=features(panel)+names(new);x=panel[columns]
            row[name+'_status']='flat_no_validation_opportunities'
            if use.sum()<504 or len(vcal)<200:
                row[name+'_status']='insufficient_training';continue
            if not opportunity:continue
            labels=daily[use].copy();labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
            model,detail=fit(x.loc[use],labels,positions[use],finish[use],CODES)
            row['last_training_outcome']=str(completion[use].max().date())
            proposal=gate(predict(model,x.loc[val]),panel.index[val],panel,gates,available)
            score=None;error=None
            if proposal.ne(0).any():
                try:
                    nav,_,_=replay.run(proposal,vcal);mark_book(nav,rates,replay.cfg.capital)
                    score=scores(nav,replay.cfg.capital)['annualized_excess_sharpe']
                except UnresolvedPosition as exc:error=str(exc)
            enabled=score is not None and score>0 and recent.sum()>=504
            row[name+'_status']='enabled' if enabled else 'flat_validation_gate'
            row[name+'_validation_sharpe']=score;row[name+'_validation_error']=error
            row[name+'_proposal_shadow_sharpe']=detail['training_shadow_daily_sharpe']
            if enabled:
                own=daily[recent].copy();own[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                model,detail=fit(x.loc[recent],own,positions[recent],finish[recent],CODES)
                dates=idx[test]
                actions.loc[dates,name]=gate(predict(model,x.loc[dates]),dates,panel,gates,available)
                row['last_recent_outcome']=str(completion[recent].max().date())
                row[name+'_refit_shadow_sharpe']=detail['training_shadow_daily_sharpe']
        logs.append(row)
        write_json(checkpoint,dict(dates=[str(t) for t in idx],actions=actions.to_numpy().tolist(),fits=logs))
    for name in FAMILIES:
        dest=folder/name;dest.mkdir(exist_ok=True)
        a=actions[name].rename('code')
        pd.testing.assert_series_equal(a,gate(a,a.index,panel,gates,available),check_names=False)
        a.to_csv(dest/'actions.csv')
    for name in EXITS:
        dest=folder/name;dest.mkdir(exist_ok=True);saved.rename('code').to_csv(dest/'actions.csv')
        actions[name]=saved
    fits=pd.DataFrame(logs) if logs else pd.DataFrame(columns=['fit','validation_start'])
    fits.to_csv(folder/'fits.csv',index=False)
    return actions

def rank_early(panel,daily,finish,statuses,available,cal,pair):
    ends=cal[finish]
    eligible=(panel.index>='2013-01-01')&(ends<'2018-01-01')&panel.eligible.to_numpy(bool)
    eligible &= np.asarray(statuses)=='complete'
    eligible &= available & np.isfinite(daily).all(axis=(1,2))
    rows=thin_completed(panel.index,ends,eligible)
    target=(daily[rows,-1].sum(axis=1)-daily[rows,1].sum(axis=1))/2
    result=[]
    for feature in DYNAMICS+GAMMA:
        observed=panel[f'mine_{feature}_available'].to_numpy()[rows]>0
        x=panel[f'mine_{feature}'].to_numpy()[rows][observed];y=target[observed]
        rho=None
        if len(x)>=10 and np.std(x)>1e-12 and np.std(y)>1e-12:rho=float(spearmanr(x,y).statistic)
        result.append(dict(pair=pair,feature=feature,rows=len(x),spearman=rho,
                           last_outcome=str(ends[rows].max().date()) if len(rows) else None))
    return result

def exit_replay(replay,codes,calendar,rule):
    sizes=pd.Series([ACTIONS[int(c)][1] if c else 0. for c in codes],index=codes.index,dtype=float)
    selected={}
    for date,code in codes[codes.ne(0)].items():
        combo,_=ACTIONS[int(code)];tenor,cp,ba,bb=COMBOS[combo]
        for sid,bucket in zip(replay.secids,[ba,bb]):
            oid=replay.lookup.get((date,sid,tenor,cp,str(bucket)))
            if oid is not None:selected[(date,sid)]=oid
    prepared=copy.copy(replay.prepared);prepared.selection_ids=selected
    selection=pd.DataFrame(columns=['date','secid','optionid']);prepared.sources=(replay.q,replay.s,selection)
    return simulate_exits(replay.q,replay.s,sizes,calendar,replay.secids,replay.cfg,replay.divs,
                          selection,prepared,exit_signals=sizes,exit_rule=rule)

def save_exit(dest,nav,trades,decisions,cfg,stocks,d):
    checks=audit(nav,trades,stocks,cfg,d['secids'])
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    if len(trades):
        paid=trades.reason.isin(['entry','signal_reversal_exit','signal_flat_exit'])
        assert set(trades.reason).issubset({'entry','signal_reversal_exit','signal_flat_exit','split_adjustment','expiry_settlement'})
        np.testing.assert_allclose(trades.loc[paid,'commission'],abs(trades.loc[paid,'quantity'])*cfg.option_commission)
        assert trades.loc[~paid,'commission'].eq(0).all()
        exited=trades.reason.isin(['signal_reversal_exit','signal_flat_exit'])
        if exited.any():assert (pd.to_datetime(trades.loc[exited,'exit_signal_date'])<pd.to_datetime(trades.loc[exited,'date'])).all()
    entered=decisions[decisions.status.eq('entered')] if len(decisions) else decisions
    if len(entered):assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
    nav.to_parquet(dest/'daily_nav.parquet');trades.to_csv(dest/'trades.csv',index=False);decisions.to_csv(dest/'decisions.csv',index=False)
    return dict(status='complete',pair=d['pair'],label=d['label'],entries=len(entered),
        exited_cohorts=int(nav.policy_exit_cohorts.sum()),commission=float(trades.commission.sum()) if len(trades) else 0.,
        option_cost=float(nav.option_execution_cost.sum()),hedge_cost=float(nav.hedge_cost.sum()),**scores(nav,cfg.capital),**checks)

def source_folder(pair,model,period):
    return FROZEN/pair/(period+'_003') if model=='rl_sized' else OUT/pair/model/period

def summary(defs,market,cfg):
    models=['rl_sized']+MODELS
    statuses={(d['pair'],m,p):json.loads((source_folder(d['pair'],m,p)/'status.json').read_text())
              for d in defs for m in models for p,_ in PERIODS}
    complete=[d for d in defs if all(statuses[d['pair'],m,p]['status']=='complete' for m in models for p,_ in PERIODS)]
    unresolved=[dict(pair=d['pair'],label=d['label'],model=m,period=p,error=statuses[d['pair'],m,p].get('error'))
                for d in defs for m in models for p,_ in PERIODS if statuses[d['pair'],m,p]['status']!='complete']
    write_json(REPORT/'coverage.json',dict(requested_pairs=len(defs),common_complete_pairs=len(complete),unresolved=unresolved))
    rows=[];annual=[];pairs=[]
    for m in models:
        for p,start in PERIODS:
            cal=market.calendar[market.calendar>=start];cash=cash_book(cal,market.rates,cfg.capital)
            book=cash[['nav','cash_benchmark']]*len(complete);entries=active=exits=0
            for d in complete:
                status=statuses[d['pair'],m,p];folder=source_folder(d['pair'],m,p)
                nav=cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'),columns=['nav','cash_benchmark'])
                assert nav.index.equals(cal);book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                entries+=status['entries'];active+=int(status['entries']>0);exits+=status.get('exited_cohorts',0)
                pairs.append(dict(model=m,period=p,pair=d['pair'],label=d['label'],entries=status['entries'],
                                  net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
            book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT/f'{m}_{p}_portfolio.csv')
            rows.append(dict(model=m,period=p,pairs=len(complete),entries=entries,active_pairs=active,exited_cohorts=exits,
                        net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),**scores(book,cfg.capital*len(complete))))
            for year,g in book.groupby(book.index.year):
                annual.append(dict(model=m,period=p,year=year,return_percent=100*((1+g.return_).prod()-1),
                    annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    rank=[];fits=[]
    for d in defs:
        folder=OUT/d['pair']
        if (folder/'early_rank.csv').exists():rank.append(pd.read_csv(folder/'early_rank.csv'))
        if (folder/'fits.csv').exists():fits.append(pd.read_csv(folder/'fits.csv').assign(pair=d['pair']))
    ranks=pd.concat(rank,ignore_index=True) if rank else pd.DataFrame(columns=['feature','spearman','rows'])
    ranks.to_csv(REPORT/'early_rank_pairs.csv',index=False)
    usable=ranks.dropna(subset=['spearman'])
    screen=usable.groupby('feature').agg(mean_spearman=('spearman','mean'),median_spearman=('spearman','median'),
        pairs=('spearman','size'),rows=('rows','sum'),positive_fraction=('spearman',lambda x:float((x>0).mean())))
    screen['absolute_mean']=screen.mean_spearman.abs();screen=screen.sort_values('absolute_mean',ascending=False)
    screen.to_csv(REPORT/'early_rank.csv')
    if fits:pd.concat(fits,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    lines=['# Causal signal mining and exit diagnostics','',
      'Reference training hurdle $0.65; executed commission $0.03. Same 345 requested pairs, option selection, sizing ladder, risk limits and monthly walk-forward procedure. All comparisons use the common complete roster; unsupported paths are recorded in coverage rather than treated as cash. Original checkpoint unchanged.','',
      '| Model | Full Sharpe | Recent Sharpe | Full entries |','|---|---:|---:|---:|']
    for m in models:
        a=comparison[comparison.model.eq(m)&comparison.period.eq('full')].iloc[0]
        b=comparison[comparison.model.eq(m)&comparison.period.eq('recent')].iloc[0]
        sa='cash' if pd.isna(a.annualized_excess_sharpe) else f'{a.annualized_excess_sharpe:.3f}'
        sb='cash' if pd.isna(b.annualized_excess_sharpe) else f'{b.annualized_excess_sharpe:.3f}'
        lines.append(f'| {m} | {sa} | {sb} | {int(a.entries)} |')
    lines+=['','Full: 2018–August 2025. Recent: 2022–August 2025, fresh start. Daily cash-excess Sharpe × √252. Actual costs include 0.10 vol point per option leg per execution, $0.03 commission, 1 bp stock execution and original financing/borrow.','',
      'Two feature families were specified before these runs: dynamics adds 5/21/63-session variance value, short RV acceleration, IV changes, spread z-scores, IV/return correlation gap and beta-scaled IV ratio. Dynamics+gamma also adds gamma per vega, gamma-weighted variance value, quoted width, open interest and expiry mismatch. Each value has a missingness flag. IV correlation requires 32 paired observations within 63 calendar trading sessions; spread z-scores require 11/32 prior observations in 21/63-session windows. Missing IV is not forward-filled. All 47 original inputs remain unchanged. Each candidate receives its own prior-year validation gate at the $0.65 training hurdle and five-year refit, with completed outcomes strictly purged at both boundaries.','',
      'Signal ranks use only 2013–2017 completed, eligible, non-overlapping hypothetical training cohorts. The target is half the difference between full-size long-pair and short-pair net payoff per target entry vega. Average within-pair Spearman correlations are descriptive; the two predeclared families are not selected using 2018–2025 performance. This remains an historically screened universe and previously inspected evaluation period.','',
      'Exit overlays reuse saved baseline entry recommendations. Reversal closes both live legs together on the next session; flat-or-reverse additionally closes when the recommendation becomes flat. Missing signal dates cause no exit. A flat recommendation can reflect eligibility or memory gates. These are diagnostic rules, not a learned exit policy, and do not forecast the held contracts. Option quantities remain fixed between entry and closure (except splits); neutrality is checked at entry, with daily delta hedging but no daily option-vega rebalancing. Early exits pay the same modeled commission and vol penalty; natural expiries do not. Freed risk capacity can change subsequent realized entries.','',
      '[Comparison](atm65_signal_mining/comparison.csv) · [Annual](atm65_signal_mining/annual.csv) · [Pairs](atm65_signal_mining/pairs.csv) · [Early ranks](atm65_signal_mining/early_rank.csv) · [Coverage](atm65_signal_mining/coverage.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['model','period','pairs','annualized_excess_sharpe','entries','exited_cohorts']].to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    previous=json.loads(Path('artifacts/contracts/atm65_ablation/inputs.json').read_text());defs=previous['definitions']
    actual=replace(config(),option_commission=.03);training=replace(actual,option_commission=.65)
    assert asdict(actual)==previous['actual']
    paths=[Path(__file__),Path('scripts/atm65_mining_features.py'),Path('scripts/option_flex/exit_engine.py'),
           Path('scripts/audit_atm65_signal_mining.py'),Path('artifacts/contracts/atm65_ablation/inputs.json'),
           Path('config/atm30_fee065_checkpoint.json'),Path('scripts/run_atm65_ablation.py'),
           Path('scripts/run_atm_cost_hurdles.py'),Path('scripts/replay_frozen_atm_commission.py'),
           Path('scripts/option_flex/contract_engine.py'),Path('scripts/run_curve_trading.py')]
    paths+=list(Path('scripts/curve_trading').glob('*.py'))
    for d in defs:
        for path in [NN/d['pair']/'fee_065/states.parquet',NN/d['pair']/'fee_065/gates.npz',CURVE/d['pair']/'atm_control/actions.csv']:
            if path.exists():paths.append(path)
    signature=dict(definitions=defs,actual=asdict(actual),training=asdict(training),families=FAMILIES,exit_rules=EXITS,
      early_rank_cutoff='2018-01-01',sources={str(p):digest(p) for p in paths})
    if (OUT/'inputs.json').exists():assert json.loads((OUT/'inputs.json').read_text())==json.loads(json.dumps(signature)),'Inputs changed'
    else:write_json(OUT/'inputs.json',signature)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=Market(cal);started=time.monotonic();newly=0
    if not args.report_only:
        for n,d in enumerate(defs,1):
            folder=OUT/d['pair'];folder.mkdir(exist_ok=True)
            if all((folder/m/p/'status.json').exists() for m in MODELS for p,_ in PERIODS):continue
            original=json.loads((CURVE/d['pair']/'atm_control/status.json').read_text())
            source=NN/d['pair']/'fee_065'
            base=pd.read_parquet(source/'states.parquet') if (source/'states.parquet').exists() else None
            if original.get('structural_cash') or base is None or base.empty:
                for m in MODELS:
                    for p,_ in PERIODS:write_json(folder/m/p/'status.json',dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True,training_hurdle=.65,execution_commission=.03))
                continue
            pair_started=time.monotonic();print(f'{n}/{len(defs)} {d["label"]}: dynamics, gamma and exits',flush=True)
            q,s,selections,divs=load_pair(market,d['secids'])
            with np.load(source/'gates.npz') as z:
                gates=np.zeros((len(COMBOS),len(base),2),bool);gates[ATM]=z['gates']
                available=np.zeros((len(COMBOS),len(base)),bool);available[ATM]=z['available'];saved_ends=z['ends'].copy()
            if (folder/'states.parquet').exists():panel=pd.read_parquet(folder/'states.parquet')
            else:panel=engineer(base,q,s,selections,d['secids']);panel.to_parquet(folder/'states.parquet')
            pd.testing.assert_frame_equal(panel[base.columns],base)
            cache=folder/'labels.npz'
            if cache.exists():
                with np.load(cache) as z:daily=z['daily'];finish=z['finish'];statuses=z['statuses'];vegas=z['vegas'];rewards=z['rewards']
            else:
                units=make_units(base,q,s,selections,divs,cal,training)
                labels,finish,vegas,statuses,rewards,_=reward_and_memory(units,training)
                daily=np.zeros((len(base),9,labels.shape[-1]),dtype=np.float32);daily[:,1:]=labels
                np.savez(cache,daily=daily,finish=finish,statuses=np.asarray(statuses,dtype='U32'),vegas=vegas,rewards=rewards)
                del units,labels;gc.collect()
            np.testing.assert_array_equal(finish,saved_ends)
            np.testing.assert_array_equal(gates[ATM],memory_gates(base.index,cal[finish],rewards,vegas[:,[0,-1]]>0))
            pd.DataFrame(rank_early(panel,daily,finish,statuses,available[ATM],cal,d['pair'])).to_csv(folder/'early_rank.csv',index=False)
            replay=Replay(q,s,selections,d['secids'],divs,training)
            actions=train_pair(d,panel,daily,finish,gates,available,cal,replay,market.rates,folder);replay.cfg=actual
            for m in MODELS:
                for p,start in PERIODS:
                    dest=folder/m/p;dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'status.json').exists():continue
                    a=actions.loc[actions.index>=start,m]
                    if not a.ne(0).any():result=dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True)
                    else:
                        try:
                            nav,trades,decisions=exit_replay(replay,a,cal[cal>=start],EXITS[m]) if m in EXITS else replay.run(a,cal[cal>=start])
                            mark_book(nav,market.rates,actual.capital)
                            result=save_exit(dest,nav,trades,decisions,actual,s,d) if m in EXITS else save_result(dest,nav,trades,decisions,actual,s,d)
                        except UnresolvedPosition as exc:result=dict(status='incomplete',pair=d['pair'],label=d['label'],error=str(exc))
                    result.update(training_hurdle=.65,execution_commission=.03);write_json(dest/'status.json',result)
            newly+=1;elapsed=time.monotonic()-pair_started
            write_json(OUT/'progress.json',dict(roster_position=n,pairs=len(defs),last_pair=d['label'],last_pair_seconds=elapsed,
                elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: {elapsed:.0f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GiB',flush=True)
            del base,panel,q,s,selections,divs,replay,daily,actions;gc.collect()
            if args.limit and newly>=args.limit:return
    summary(defs,market,actual)
    write_json(OUT/'task_status.json',dict(status='complete',pairs=len(defs),training_hurdle=.65,execution_commission=.03))

if __name__=='__main__':main()
