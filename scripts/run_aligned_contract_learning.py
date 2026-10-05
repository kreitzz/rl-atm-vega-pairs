"""Independent monthly contract-trained RL sizing versus fixed-size NN.

Proposals learn net standalone contract/hedge trajectories. A separate,
preceding one-year validation book applies the identical portfolio simulator,
costs, beta eligibility and risk limits before allowing a proposal to trade.
No proxy IV or forward-RV reward is used. One pair is processed at a time.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, UnresolvedPosition, PreparedMarket, simulate
from scripts.contract_learning_labels import ShadowLabels, profitability_memory, beta_screen
from scripts.batched_contract_labels import build_labels
from scripts.run_broad_contract_replay import Market, universe, CACHE, ROOT
from scripts.run_contract_walkforward import digest, fit_rl, fit_nn, predict
from scripts.run_nn_rl_benchmark import NEW_FEATURES, add_features
from scripts.run_selected_pair_backtest import audit, scores
from scripts.select_atm_contracts import load_calendar

MODELS=['rl_sized','nn_fixed']
EXTRA=['beta','correlation','commission_vol_roundtrip','dte_a','dte_b','moneyness_a','moneyness_b']


def asset_paths(definition):
    if definition['group']=='broad':
        return [Path('artifacts/broad_pairs/assets')/f'{a}.parquet' for a in definition['source_pair'].split('__')]
    paths=[]
    for sid in definition['secids']:
        found=list(Path('artifacts/broad_pairs/assets').glob(f's{sid}p*.parquet'))
        if len(found)!=1:raise ValueError(f'Ambiguous return history for {sid}')
        paths.append(found[0])
    return paths


def contract_states(definition,selection,stocks,calendar,cfg,beta_tolerance=.25):
    """As-of signal-close contract marks; return/earnings histories only from assets."""
    panel=pd.DataFrame(index=calendar)
    assets=[pd.read_parquet(p,columns=['ret','trailing_rv','earnings']) for p in asset_paths(definition)]
    for sid,asset,suffix,side in zip(definition['secids'],assets,['amd','nvda'],['a','b']):
        chosen=selection[selection.secid.eq(sid)].set_index('date')
        panel[f'iv_{suffix}']=chosen.get('bs_iv',pd.Series(dtype=float))*100
        panel[f'trailing_rv_{suffix}']=asset.trailing_rv
        panel[f'{suffix}_earnings_days_capped']=asset.earnings
        panel[f'dte_{side}']=(chosen['expiry']-chosen.index.to_series()).dt.days if 'expiry' in chosen else np.nan
        spot=stocks[stocks.secid.eq(sid)].set_index('date').close
        panel[f'moneyness_{side}']=chosen['strike']/spot-1 if 'strike' in chosen else np.nan
        panel[f'commission_{side}']=cfg.option_commission/(chosen.bs_vega*100) if 'bs_vega' in chosen else np.nan
    panel['spread']=panel.iv_amd-panel.iv_nvda
    panel['commission_vol_roundtrip']=2*(panel.commission_a+panel.commission_b)
    returns=pd.concat([a.ret.rename(s) for a,s in zip(assets,definition['secids'])],axis=1)
    screen=beta_screen(returns,calendar,beta_tolerance=beta_tolerance)
    panel=panel.join(screen[['beta','correlation','eligible']])
    panel=add_features(panel).replace([np.inf,-np.inf],np.nan).dropna(subset=EXTRA)
    return panel,screen


def completed_training_mask(panel,ends,daily,validation_start):
    return ((panel.index>=validation_start-pd.DateOffset(years=5)) &
            (ends<validation_start) & np.isfinite(daily).all(axis=(1,2)))


def screen_dates(definition,dates):
    if 'screen_quarters' in definition:
        return pd.DatetimeIndex(dates).to_period('Q').astype(str).isin(definition['screen_quarters'])
    years={y+k for y in definition['screen_years'] for k in [0,1]}
    return np.isin(dates.year,list(years))


def mark_book(nav,rates,capital):
    elapsed=nav.index.to_series().diff().dt.days.fillna(0)/365
    cash_return=rates.reindex(nav.index).shift(1).fillna(0)*elapsed
    nav['cash_benchmark']=capital*(1+cash_return).cumprod()
    nav['excess_return']=nav.return_-cash_return
    return nav


def validate_policy(actions,q,stocks,selection,divs,secids,cal,cfg,rates,prepared=None):
    """Out-of-fit full-book score; an unresolved book cannot pass the gate."""
    try:
        nav,trades,_=simulate(q,stocks,actions,cal,secids,cfg,divs,selection,prepared)
        mark_book(nav,rates,cfg.capital)
        audit(nav,trades,stocks,cfg,secids)
        score=scores(nav,cfg.capital)['annualized_excess_sharpe']
        return (score is not None and score>0),score,None
    except (UnresolvedPosition,ValueError) as exc:
        return False,None,str(exc)


def train_pair(definition,panel,daily,ends,calendar,q,stocks,selection,divs,cfg,rates,start,stop,folder,
               feature_columns=None,memory_filters=None,recent_refit=False):
    wanted=(panel.index>=start)&(panel.index<=stop)&screen_dates(definition,panel.index)
    actions=pd.DataFrame(0.,index=panel.index[wanted],columns=MODELS)
    memory=[c for c in panel if c.startswith(('profit_','count_'))]
    features=NEW_FEATURES+EXTRA+memory if feature_columns is None else feature_columns
    prepared=PreparedMarket(q,stocks,selection)
    rows=[]
    for month in pd.period_range(start,stop,freq='M'):
        cutoff=month.start_time
        test=(actions.index>=cutoff)&(actions.index<(month+1).start_time)
        if not test.any():continue
        validation_start=cutoff-pd.DateOffset(years=1)
        train=completed_training_mask(panel,ends,daily,validation_start)
        val_cal=calendar[(calendar>=validation_start)&(calendar<cutoff)]
        if train.sum()<504 or len(val_cal)<200:
            rows.append(dict(fit=str(cutoff.date()),model='both',status='insufficient_training',training_rows=int(train.sum())))
            continue
        # All validation positions close before the current prediction month.
        runoff=cfg.max_dte+5 if cfg.hold_to_expiry else 35
        val=(panel.index>=val_cal[0])&(panel.index<=val_cal[-1]-pd.Timedelta(days=runoff))
        if not val.any():continue
        assert ends[train].max()<validation_start and val_cal[-1]<cutoff
        x=panel.loc[train,features]
        # Beta eligibility is the same in training proposals, validation and deployment.
        labels=daily[train].copy();labels[~panel.loc[train,'eligible'].to_numpy(bool)]=0
        positions=calendar.get_indexer(panel.index[train])
        end_positions=calendar.get_indexer(ends[train])
        for name in MODELS:
            neural=name=='nn_fixed'
            if memory_filters:
                from scripts.test_contract_entry_gates import gate_actions
                eligibility=panel.loc[val,'eligible'].astype(float)
                possible=any(gate_actions(eligibility*direction,panel,memory_filters[name]).ne(0).any()
                             for direction in [-1,1])
                if not possible:
                    rows.append(dict(fit=str(cutoff.date()),model=name,status='flat_no_validation_opportunities',
                        training_rows=int(train.sum()),last_training_outcome=str(ends[train].max().date()),
                        validation_start=str(val_cal[0].date()),validation_end=str(val_cal[-1].date()),
                        validation_sharpe=None,validation_error=None))
                    continue
            model,details=fit_nn(x,labels) if neural else fit_rl(x,labels,positions,end_positions)
            proposal=predict(model,panel.loc[val,features],neural)*panel.loc[val,'eligible'].to_numpy()
            proposal=pd.Series(proposal,index=panel.index[val])
            if memory_filters:
                from scripts.test_contract_entry_gates import gate_actions
                proposal=gate_actions(proposal,panel,memory_filters[name])
            permitted,score,error=validate_policy(proposal,q,stocks,selection,divs,
                                                 definition['secids'],val_cal,cfg,rates,prepared)
            dates=actions.index[test]
            if permitted:
                if recent_refit:
                    use=(panel.index>=cutoff-pd.DateOffset(years=5))&(ends<cutoff)&np.isfinite(daily).all(axis=(1,2))
                    if use.sum()<504:
                        rows.append(dict(fit=str(cutoff.date()),model=name,status='insufficient_recent_training',
                            training_rows=int(train.sum()),recent_training_rows=int(use.sum()),validation_sharpe=score))
                        continue
                    assert ends[use].max()<cutoff
                    fresh=daily[use].copy();fresh[~panel.loc[use,'eligible'].to_numpy(bool)]=0
                    model,refit_details=(fit_nn(panel.loc[use,features],fresh) if neural else
                        fit_rl(panel.loc[use,features],fresh,calendar.get_indexer(panel.index[use]),calendar.get_indexer(ends[use])))
                    details.update(recent_training_rows=int(use.sum()),last_recent_outcome=str(ends[use].max().date()),
                                   **{'recent_'+k:v for k,v in refit_details.items()})
                deployed=pd.Series(predict(model,panel.loc[dates,features],neural)*panel.loc[dates,'eligible'].to_numpy(),index=dates)
                if memory_filters:deployed=gate_actions(deployed,panel,memory_filters[name])
                actions.loc[dates,name]=deployed
            rows.append(dict(fit=str(cutoff.date()),model=name,status='enabled' if permitted else 'flat_validation_gate',
                             training_rows=int(train.sum()),last_training_outcome=str(ends[train].max().date()),
                             validation_start=str(val_cal[0].date()),validation_end=str(val_cal[-1].date()),
                             validation_sharpe=score,validation_error=error,**details))
        pd.DataFrame(rows).to_csv(folder/'fits.csv',index=False)
        actions.to_csv(folder/'actions.partial.csv')
        if month.month in [1,7]:print(f"  {definition['label']} fit {month}",flush=True)
    assert np.isin(actions.rl_sized,np.arange(-1,1.01,.25)).all()
    assert np.isin(actions.nn_fixed,[-1,0,1]).all()
    return actions,pd.DataFrame(rows)


def load_labels(definition,panel,q,stocks,selection,calendar,divs,cfg,folder,sources):
    path=folder/'labels.npz';meta=folder/'labels_metadata.json'
    fingerprint=dict(settings=asdict(cfg),sources=sources,dates=digest_array(panel.index.values),
                     label_code=digest('scripts/contract_learning_labels.py'),batch_code=digest('scripts/batched_contract_labels.py'),
                     engine_code=digest('scripts/contract_engine.py'))
    if path.exists() and meta.exists() and json.loads(meta.read_text())==fingerprint:
        with np.load(path,allow_pickle=False) as z:
            return z['daily'],z['ends'],z['vegas'],z['statuses']
    maker=ShadowLabels(q,stocks,selection,calendar,definition['secids'],cfg,divs)
    positions=calendar.get_indexer(panel.index)
    daily,end_positions,vegas,statuses=build_labels(maker,positions)
    ends=calendar[end_positions].values;statuses=np.asarray(statuses,dtype='U40')
    np.savez_compressed(path,daily=daily,ends=ends,vegas=vegas,statuses=statuses)
    meta.write_text(json.dumps(fingerprint,indent=2)+'\n')
    return daily,ends,vegas,statuses


def digest_array(values):
    import hashlib
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/contracts/aligned_learning'))
    parser.add_argument('--report',type=Path,default=Path('reports/aligned_contract_learning'))
    parser.add_argument('--start',default='2018-01-01')
    parser.add_argument('--end',default='2025-08-29')
    parser.add_argument('--pairs',nargs='+',help='Exact labels or canonical pair IDs; omitted means all 358')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.report.mkdir(parents=True,exist_ok=True)
    definitions=universe()
    if args.pairs:
        definitions=[d for d in definitions if d['pair'] in args.pairs or d['label'] in args.pairs]
        if len(definitions)!=len(set(args.pairs)):raise ValueError('Unknown or duplicate pair argument')
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2010-01-01')&(calendar<=args.end)]
    evaluation=calendar[calendar>=args.start];stop=calendar[-1]-pd.Timedelta(days=35)
    cfg=Settings(option_spread_fraction=0,option_penalty_vol_points=.1,risk_controls=True,
                 repair_crossed_held_quotes=True,support_integer_splits=True,allow_missing_flat_underlying=True)
    code=[Path(__file__),Path('scripts/run_contract_walkforward.py'),Path('scripts/contract_learning_labels.py'),
          Path('scripts/batched_contract_labels.py'),
          Path('scripts/contract_engine.py'),Path('scripts/contract_risk.py'),Path('scripts/run_broad_contract_replay.py'),
          Path('scripts/run_selected_pair_backtest.py'),Path('scripts/run_nn_rl_benchmark.py')]
    fingerprint=dict(settings=asdict(cfg),start=args.start,end=args.end,definitions=definitions,
                     cache_manifest_sha256=digest(CACHE/'manifest.json'),code={str(p):digest(p) for p in code},
                     assets={str(p):digest(p) for d in definitions for p in asset_paths(d)})
    metadata=args.output/'inputs.json'
    if args.resume and metadata.exists() and json.loads(metadata.read_text())!=fingerprint:raise ValueError('Resume inputs changed')
    metadata.write_text(json.dumps(fingerprint,indent=2)+'\n')
    (args.report/'universe.json').write_text(json.dumps(definitions,indent=2)+'\n')
    market=Market(calendar);records=[];began=time.monotonic()
    for n,d in enumerate(definitions,1):
        folder=args.output/d['pair'];folder.mkdir(parents=True,exist_ok=True)
        if args.resume and (folder/'status.json').exists():
            records.extend(json.loads((folder/'status.json').read_text()));continue
        print(f"{n}/{len(definitions)} Loading {d['label']}",flush=True)
        q,stocks,selection,divs=market.pair(d['secids'])
        panel,screen=contract_states(d,selection,stocks,calendar,cfg)
        panel=panel[panel.index<=stop]
        sources={'cache':fingerprint['cache_manifest_sha256'],
                 **{str(p):fingerprint['assets'][str(p)] for p in asset_paths(d)}}
        daily,ends,vegas,statuses=load_labels(d,panel,q,stocks,selection,calendar,divs,cfg,folder,sources)
        ends=pd.DatetimeIndex(ends)
        rewards=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
                          out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
        panel=panel.join(profitability_memory(panel.index,ends,rewards,vegas[:,[0,-1]]>0))
        panel.to_parquet(folder/'states.parquet');screen.to_csv(folder/'beta_screen.csv')
        pd.Series(statuses).value_counts().rename_axis('status').to_csv(folder/'label_status.csv')
        print(f"  labels {pd.Series(statuses).value_counts().to_dict()}",flush=True)
        actions,fits=train_pair(d,panel,daily,ends,calendar,q,stocks,selection,divs,cfg,market.rates,
                               pd.Timestamp(args.start),stop,folder)
        actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
        pair_records=[]
        for model in MODELS:
            dest=folder/model;dest.mkdir(exist_ok=True)
            common=dict(pair=d['pair'],label=d['label'],group=d['group'],model=model,signal_dates=len(actions),
                        nonzero_signals=int(actions[model].ne(0).sum()),feature_rows=len(panel))
            try:
                nav,trades,decisions=simulate(q[q.date.isin(evaluation)],stocks[stocks.date.isin(evaluation)],
                    actions[model],evaluation,d['secids'],cfg,divs,selection)
                mark_book(nav,market.rates,cfg.capital)
                checks=audit(nav,trades,stocks,cfg,d['secids'])
                nav.to_parquet(dest/'daily_nav.parquet');trades.to_csv(dest/'trades.csv',index=False)
                decisions.to_csv(dest/'decisions.csv',index=False)
                row=dict(**common,status='complete',entries=int(decisions.status.eq('entered').sum()) if len(decisions) else 0,
                         **scores(nav,cfg.capital),**checks)
            except (UnresolvedPosition,ValueError) as exc:
                row=dict(**common,status='incomplete',error=str(exc))
            pair_records.append(row);print(json.dumps(row),flush=True)
        (folder/'status.json').write_text(json.dumps(pair_records,indent=2)+'\n');records.extend(pair_records)
        pd.DataFrame(records).to_csv(args.report/'status.csv',index=False)
        (args.report/'progress.json').write_text(json.dumps(dict(completed_pairs=n,total_pairs=len(definitions),
            elapsed_seconds=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    pd.DataFrame(records).to_csv(args.report/'status.csv',index=False)
    manifest=dict(settings=asdict(cfg),pairs=len(definitions),models=MODELS,start=str(evaluation[0].date()),end=str(evaluation[-1].date()),
                  training='Monthly: five-year standalone net-contract trajectory proposal fit; minimum 504 known rows; separate one-year full-book validation; positive validation excess Sharpe required. No refit on validation.',
                  memory='91/365-day completed net directional shadow outcomes, not proxy RV outcomes',
                  eligibility='original historical pair screen membership in evaluation; monthly preceding-126-session beta 0.75-1.25 and correlation >=0.5',
                  limitations='Proposal objective uses standalone cohorts, not an exact overlapping-book optimizer; validation and evaluation use the full identical risk-controlled engine. Unresolved histories remain explicit.',
                  elapsed_seconds=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  inputs_sha256=digest(metadata))
    (args.report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
