"""Bounded matched experiments: identical rules across ATM/P25/P75."""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import uniform_smile_policy as policy
from scripts import uniform_smile_data as data
from scripts.joint_smile_data import registry
from scripts.uniform_smile_engine import simulate
from scripts import uniform_smile_engine as shared_engine
from scripts.uniform_smile_risk import book_risk as cached_risk
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.curve_trading.config import config,RUNOFF
from scripts.tail_put25_pooled import completed_mask
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.research_opposite_wing_signals import CORPORATE_EVENTS

OUT,REPORT=data.OUT,data.REPORT
SIZING_CONTROL='spread_flex_selected_full'
UNIFORM_CORPORATE_EVENTS=CORPORATE_EVENTS+[
    dict(secid=111459,announced='2020-03-30',effective='2020-04-03',
         reason='UTX Carrier/Otis separation and RTX merger',
         source='https://investors.rtx.com/news-releases/news-release-details/raytheon-and-united-technologies-obtain-all-regulatory-approvals'),
    dict(secid=106625,announced='2021-12-10',effective='2021-12-14',
         reason='KSU acquisition by Canadian Pacific',
         source='https://www.sec.gov/Archives/edgar/data/54480/000119312521354084/d151013dex991.htm'),
    dict(secid=110962,announced='2018-09-24',effective='2018-10-01',
         reason='Andeavor acquisition by Marathon Petroleum',
         source='https://ir.marathonpetroleum.com/investor/news-releases/news-details/2018/Shareholders-of-Marathon-Petroleum-Corp-and-Andeavor-approve-strategic-combination/default.aspx'),
    dict(secid=106905,announced='2023-01-05',effective='2023-01-17',
         reason='Jefferies Vitesse distribution after January13 close',
         source='https://www.sec.gov/Archives/edgar/data/96223/000119312523002005/d398713dex991.htm'),
    dict(secid=125179,announced='2020-08-04',effective='2020-08-27',
         reason='SunPower Maxeon distribution after August26 close',
         source='https://mediaroom.maxeon.com/2020-08-04-Maxeon-Solar-Technologies-Form-20-F-Registration-Statement-Declared-Effective-by-SEC'),
    dict(secid=101556,announced='2023-09-12',effective='2023-09-25',
         reason='AngloGold ADS termination and primary listing reorganisation',
         source='https://www.sec.gov/Archives/edgar/data/1067428/000162828023032020/finalisationannouncementin.htm'),
    dict(secid=101322,announced='2025-07-18',effective='2025-07-18',exit_same_day=True,
         reason='Hess conversion to Chevron before the open; close paired inventory at that close',
         source='https://infomemo.theocc.com/infomemos?number=56907',
         missing_option_exit_mark='existing last-known-IV causal mark on published1.025CVX deliverable'),
]
PROTOCOL=('Nine predeclared matched variants across independent ATM/P25/P75 wing policies, all345 roster pairs retained. '
 'Same35 raw/directional-history features, optional neutral fill/availability flags, monthly five-year completed-outcome training, eight-leaf direct-Sharpe CEM seed2026. '
 'Common monthly beta0.85-1.15/correlation>=0.5 and original paired formation/calendar context for all wings. '
 'Spread-rule bottom/top252-session IV-gap rank quintiles min63 quotes: fullsize,50/75/100 sizing,0/25/50/75/100 sizing. '
 'Repeat those three with a common directional completed-profit filter: positive91/365day means and counts>=21/63. '
 'RL entry/direction on every observed eligible contextual date: cash/long/short fullsize, cash/long/short50/75/100 sizing, plus force the sized policy nonzero outputs to fullsize. '
 'No per-pair validation/model activation or optional-feature veto; one global full-size spread-rule fallback when completed training rows<16. '
 'Identical0.65 training commission,0.03 actual, mid+.1volpoint perleg, stock1bp, original financing/dividend/borrow conventions. '
 'Actual options at approximately30D, ATM calls/P25 puts/P75 puts. Integer entry-vega match<=10%, daily delta hedge, hold own expiry/corporate exits. '
 'Equal pair-wing capital1m andgrossvega20k risk reservation, pair3m/60k; shared cash and net stock hedges, identical summed global risk caps. '
 'Standalone labels approximate actual shared-book funding/risk; this is direct policy search, not sequential inventory-trained RL. '
 'Replay exits paired inventory on causal unsupported-distribution declarations and blocks known corporate effective-session reentry. '
 'Hess July18 stock deliverable follows public OCC56907 (1.025CVX); absent option exit quotes use the existing last-IV model fallback that session only. '
 'Training targets and historical profit memory retain only available historical paths; unsupported standalone labels are excluded at completed training cutoffs, never used to filter evaluated candidates. '
 'Inspected 2018-Aug2025 research history: every variant reported, no new holdout or claim that a selected maximum is confirmatory. '
 'Oneworker,50%of onecore,oneBLAS/Arrowthread,3GiB RSS stop; original checkpoints and old experiments preserved.')


def aggregate(book,states,labels,positions,first):
    days=positions[:,None]+1+np.arange(labels.shape[-1])[None,:]-first;valid=(days>=0)&(days<len(book));na=labels.shape[1]
    for leaf in np.unique(states):
        rows=np.flatnonzero(states==leaf)
        for action in range(na):np.add.at(book[:,int(leaf)*na+action],days[rows][valid[rows]],labels[rows,action][valid[rows]])


def train(cal):
    logs=[]
    for wing in data.WINGS:
        meta=pd.read_parquet(OUT/f'{wing}_rows.parquet')
        for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=RUNOFF),freq='M'):
            cutoff=month.start_time;name=str(cutoff.date())
            known=completed_mask(meta.date,meta.end,cutoff)&meta.finite_label.to_numpy(bool)
            broad=meta.loc[known].reset_index(drop=True)
            for variant in policy.TRAINED:
                folder=OUT/'models'/wing/variant;folder.mkdir(parents=True,exist_ok=True);stamp=folder/f'{name}.json'
                if stamp.exists():logs.append(json.loads(stamp.read_text()));continue
                rows=broad.loc[policy.candidate_mask(broad,variant)].reset_index(drop=True)
                tree=None;leaves=np.array([0]);choices=np.array([4]);details={};menu=policy.action_menu(variant)
                if len(rows)>=16:
                    direction=rows.rule_direction.to_numpy(int)
                    target=(rows.target_long-rows.target_short)/2 if variant.startswith('rl_entry') else np.where(direction>0,rows.target_long,rows.target_short)
                    tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[policy.FEATURES].to_numpy(float),target)
                    leaves=np.unique(tree.apply(rows[policy.FEATURES].to_numpy(float)));states=np.searchsorted(leaves,tree.apply(rows[policy.FEATURES].to_numpy(float)))
                    first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                    book=np.zeros((last-first+1,len(leaves)*len(menu)))
                    for pair,idx in rows.groupby('pair',sort=False).groups.items():
                        idx=np.asarray(idx);disk=np.load(OUT/'pairs'/pair/f'{wing}_labels.npy',mmap_mode='r')
                        labels=disk[rows.loc[idx,'label_row'].to_numpy(int)]
                        action_paths=policy.reward_choices(labels,direction[idx],variant)
                        aggregate(book,states[idx],action_paths,rows.loc[idx,'position'].to_numpy(int),first)
                        del disk,labels,action_paths
                    choices,details=policy.choose(book,menu);del book
                record=dict(wing=wing,variant=variant,fit=name,training_rows=len(rows),training_pairs=int(rows.pair.nunique()),leaves=len(leaves),
                            status='enabled' if tree is not None else 'common_global_fullsize_spread_fallback',
                            last_training_outcome=str(pd.Timestamp(rows.end.max()).date()) if len(rows) else None,choices=choices.tolist(),**details)
                assert not len(rows) or pd.Timestamp(rows.end.max())<cutoff
                joblib.dump(dict(tree=tree,leaves=leaves,choices=choices,fit=name),folder/f'{name}.joblib');write_json(stamp,record);logs.append(record)
                del rows,tree;gc.collect()
            data.progress('train',wing=wing,fit=name,completed_rows=len(broad))
            if month.month in [1,7]:print(f'Uniform fits {wing} {name}: {len(broad)} completed rows',flush=True)
            del broad;gc.collect()
        del meta;gc.collect()
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(variant,cal):
    pieces=[];coverage=[]
    for wing in data.WINGS:
        rows=pd.read_parquet(OUT/f'{wing}_rows.parquet')
        rows=rows.loc[(rows.date>='2018-01-01')&(rows.date<=cal[-1]-pd.Timedelta(days=RUNOFF))].copy().reset_index(drop=True)
        allowed=policy.candidate_mask(rows,variant);direction=rows.rule_direction.to_numpy(int);level=np.zeros(len(rows),int)
        if variant.endswith('full') and not variant.startswith('rl_entry'):level[allowed]=4
        else:
            trained='rl_entry_sized' if variant=='rl_entry_sized_full' else variant
            for month,idx in rows.loc[allowed].groupby(rows.loc[allowed,'date'].dt.to_period('M')).groups.items():
                model=joblib.load(OUT/'models'/wing/trained/f'{month.start_time.date()}.joblib')
                assert pd.Timestamp(model['fit'])<=rows.loc[idx,'date'].min()
                values=policy.apply(model,rows.loc[idx],trained)
                d,s=policy.decode(values,direction[idx],variant);direction[idx]=d;level[idx]=s
        rows['direction']=direction;rows['base_level']=level;rows['rule_allowed']=allowed
        coverage.append(dict(variant=variant,wing=wing,roster_pairs=345,candidate_dates=len(rows),allowed_dates=int(allowed.sum()),
                             nonzero_proposals=int((level>0).sum()),learned_cash=int((allowed&(level==0)).sum())))
        pieces.append(rows)
    return pd.concat(pieces,ignore_index=True),coverage


def verify(nav,trades,decisions,signals,cfg):
    assert len(decisions)==len(signals)
    entered=decisions.loc[decisions.status.eq('entered')] if len(decisions) else pd.DataFrame(columns=['vega_a','vega_b','pair'])
    assert ((entered.vega_a-entered.vega_b).abs()/entered[['vega_a','vega_b']].max(axis=1)).le(.1+1e-12).all()
    if len(trades):
        paid=trades.reason.isin(['entry','announced_corporate_action_exit'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],trades.loc[paid,'quantity'].abs()*.03,rtol=0,atol=1e-10)
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        cashflow=(-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
    else:cashflow=pd.Series(0.,index=nav.index)
    expected=cashflow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
    error=float((expected-nav.pnl).abs().max());assert error<1e-5
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    return dict(entries=len(entered),active_pairs=int(entered.pair.nunique()),independent_ledger_error=error,
                commission=float(trades.commission.sum()) if len(trades) else 0.,all_inventory_closed=True,**scores(nav,cfg.capital))


def replay(cal,sizing_control=False):
    market=data.ReplayMarket();cfg=replace(config(),option_commission=.03,capital=3_000_000*345,max_gross_vega=60_000*345)
    evaluation=cal[cal>='2018-01-01']
    coverage=pd.read_csv(REPORT/'scoring_coverage.csv').to_dict('records') if sizing_control else []
    for variant in ([SIZING_CONTROL] if sizing_control else policy.VARIANTS):
        folder=OUT/'replays'/variant;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():
            coverage.extend(json.loads((folder/'scoring_coverage.json').read_text()));continue
        rows,counts=allocations('spread_flex' if sizing_control else variant,cal)
        if sizing_control:
            rows['base_level']=np.where(rows.base_level.gt(0),4,0)
            for c in counts:c['variant']=variant
        coverage.extend(counts)
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        signals=rows.loc[rows.base_level.gt(0)].copy();assert np.isfinite(signals[policy.FEATURES]).all().all()
        print(f'Uniform replay {variant}: {len(signals):,} proposals',flush=True);data.progress('replay',variant=variant,proposals=len(signals))
        market.variant=variant;market.started=time.monotonic();market.reported_year=None
        original_risk=shared_engine.book_risk
        try:
            shared_engine.book_risk=cached_risk
            nav,trades,decisions,audit=simulate(market,signals,evaluation,cfg,3_000_000,corporate_events=UNIFORM_CORPORATE_EVENTS,wing_capital=1_000_000)
        finally:shared_engine.book_risk=original_risk
        mark_book(nav,market.rates,cfg.capital);checks=verify(nav,trades,decisions,signals,cfg)
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
        write_json(folder/'scoring_coverage.json',counts);write_json(folder/'status.json',dict(status='complete',variant=variant,proposals=len(signals),**checks,**audit))
        print(f'{variant}: Sharpe {checks["annualized_excess_sharpe"]:.4f}, {checks["entries"]} entries',flush=True)
        del nav,trades,decisions,rows,signals;gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT/'scoring_coverage.csv',index=False)


def summarize(cal):
    rows=[];annual=[];wings=[];books={};cfg=replace(config(),capital=3_000_000*345)
    variants=policy.VARIANTS+([SIZING_CONTROL] if (OUT/'replays'/SIZING_CONTROL/'status.json').exists() else [])
    if SIZING_CONTROL in variants:
        source=pd.read_parquet(OUT/'replays/spread_flex/scored_allocations.parquet')
        control=pd.read_parquet(OUT/'replays'/SIZING_CONTROL/'scored_allocations.parquet')
        pd.testing.assert_frame_equal(source.drop(columns='base_level'),control.drop(columns='base_level'))
        np.testing.assert_array_equal(control.base_level,np.where(source.base_level.gt(0),4,0))
        write_json(REPORT/'sizing_control_identity.json',dict(status='verified',same_signal_dates=True,same_directions=True,
                   same_cash_decisions=True,same_state_inputs=True,nonzero_proposals=int(source.base_level.gt(0).sum()),
                   only_difference='positive allocation forced to100%; integer feasibility and risk admission can consequently differ'))
        del source,control
    for variant in variants:
        folder=OUT/'replays'/variant;status=json.loads((folder/'status.json').read_text());assert status['status']=='complete'
        nav=pd.read_parquet(folder/'daily_nav.parquet');books[variant]=nav
        try:decisions=pd.read_csv(folder/'decisions.csv')
        except pd.errors.EmptyDataError:decisions=pd.DataFrame(columns=['wing','status','pair','level'])
        rows.append(dict(model=variant,net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),
                         option_cost=float(nav.option_execution_cost.sum()),hedge_cost=float(nav.hedge_cost.sum()),**status))
        for year,g in nav.groupby(nav.index.year):
            sd=g.excess_return.std();annual.append(dict(model=variant,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None))
        for wing,g in decisions.groupby('wing'):
            e=g.loc[g.status.eq('entered')];wings.append(dict(model=variant,wing=wing,entries=len(e),active_pairs=int(e.pair.nunique()),mean_entry_size=float(e.level.mean()/4) if len(e) else 0))
        nav.to_csv(REPORT/f'{variant}_portfolio.csv')
    result=pd.DataFrame(rows);result.to_csv(REPORT/'comparison.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(wings).to_csv(REPORT/'wings.csv',index=False)
    matched=[]
    pairs=[('spread_floor','spread_full'),('spread_flex','spread_full'),('memory_floor','memory_full'),('memory_flex','memory_full'),
           ('rl_entry_sized','rl_entry_fixed'),('rl_entry_sized','rl_entry_sized_full')]
    if SIZING_CONTROL in books:pairs.append(('spread_flex',SIZING_CONTROL))
    for candidate,baseline in pairs:
        a,b=books[candidate].excess_return.iloc[1:].to_numpy(),books[baseline].excess_return.iloc[1:].to_numpy()
        sharpe=lambda x:np.sqrt(252)*x.mean()/x.std(ddof=1) if x.std(ddof=1)>1e-12 else 0
        rng=np.random.default_rng(2026);deltas=[];n=len(a);block=60;offset=np.arange(block)
        for _ in range(1000):
            starts=rng.integers(0,n,size=int(np.ceil(n/block)));idx=((starts[:,None]+offset)%n).ravel()[:n]
            deltas.append(sharpe(a[idx])-sharpe(b[idx]))
        matched.append(dict(candidate=candidate,baseline=baseline,sharpe_uplift=sharpe(a)-sharpe(b),
                            bootstrap_95_low=float(np.percentile(deltas,2.5)),bootstrap_95_high=float(np.percentile(deltas,97.5)),
                            bootstrap_fraction_uplift_positive=float(np.mean(np.asarray(deltas)>0)),block_sessions=block,draws=1000))
    pd.DataFrame(matched).to_csv(REPORT/'matched_rl_uplift.csv',index=False)
    print(result[['model','annualized_excess_sharpe','entries','net_excess_pnl']].sort_values('annualized_excess_sharpe',ascending=False).to_string(index=False),flush=True)
    print(pd.DataFrame(matched).to_string(index=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','train','replay','summarize','sizing_control','all'],default='all');args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_uniform_smile*.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources=registry();sources.expected.update(json.loads(Path('reports/joint_smile/output_hashes.json').read_text()))
    if (OUT/'sources.json').exists():
        for path,value in json.loads((OUT/'sources.json').read_text()).items():assert digest(path)==value,f'Uniform source changed: {path}';sources.track(path)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    defs=json.loads(sources.track(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    for path in [ROOT/'zero_curve.parquet',ROOT/'distributions.parquet',Path('data/raw/opposite_wing_DEXCAUS.csv')]+[ROOT/'underlying'/f'year={year}.parquet' for year in cal.year.unique()]:sources.track(path)
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,variants=policy.VARIANTS,features=policy.FEATURES,wings=data.WINGS,training_commission=.65,actual_commission=.03,models_per_wing_same_rules=True,corporate_events=UNIFORM_CORPORATE_EVENTS))
    if args.phase in ['prepare','all']:
        market=Market(cal);data.assets(defs,market,sources);data.prepare_pairs(defs,market,sources);data.compact(defs,market)
        write_json(OUT/'sources.json',sources);del market;gc.collect()
    if args.phase in ['train','all']:train(cal)
    if args.phase in ['replay','all']:replay(cal)
    if args.phase=='sizing_control':replay(cal,sizing_control=True)
    history_path=OUT/'phase_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));write_json(history_path,history)
    if args.phase in ['replay','summarize','sizing_control','all']:
        result=summarize(cal);fits=pd.read_csv(REPORT/'fits.csv')
        known=fits.last_training_outcome.notna();assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
        assert len(fits)==90*3*len(policy.TRAINED)
        for path,value in sources.items():assert digest(path)==value,f'Frozen source changed: {path}'
        audit=dict(status='complete',roster_pairs=len(defs),variants=len(result),predeclared_variants=len(policy.VARIANTS),
                   post_selection_sizing_controls=len(result)-len(policy.VARIANTS),monthly_models=len(fits),common_features=len(policy.FEATURES),
                   actual_commission=.03,training_commission=.65,frozen_sources_verified=len(sources.verified),causal_completed_training_verified=True,
                   full_roster_retained=True,sequential_inventory_training=False,elapsed_seconds=sum(h['elapsed_seconds'] for h in history),
                   peak_rss_bytes=max(h['peak_rss_bytes'] for h in history),phase=args.phase)
        audit['sizing_control_signal_identity_verified']=SIZING_CONTROL in result.model.values
        write_json(REPORT/'audit.json',audit);data.progress('complete',**audit)
        best=result.loc[result.annualized_excess_sharpe.idxmax()]
        uplift=pd.read_csv(REPORT/'matched_rl_uplift.csv')
        config_path=Path('config/uniform_smile_research.json')
        checkpoint=json.loads(config_path.read_text())
        checkpoint.update(status='complete',results_status=f'All1620 model fits and{len(result)} matched portfolio replays complete',
                          additional_sizing_control=SIZING_CONTROL if SIZING_CONTROL in result.model.values else None,
                          additional_sizing_control_status='Post-selection diagnostic: frozen spread-flex cash/direction decisions forced to100% size; no retraining',
                          best_uniform_variant=best.model,best_uniform_annualized_excess_sharpe=float(best.annualized_excess_sharpe),
                          results='reports/uniform_smile/comparison.csv',audit='reports/uniform_smile/audit.json',
                          causal_corporate_exits=UNIFORM_CORPORATE_EVENTS,
                          evidence_limit='Previously inspected history; selected maximum across nine variants. All paired95% block-bootstrap Sharpe-uplift intervals include zero. No supervised NN comparator in this experiment.')
        write_json(config_path,checkpoint)
        archived=pd.read_csv('reports/smile_flat_size/comparison.csv')
        conclusion=dict(best_uniform_variant=best.model,annualized_excess_sharpe=float(best.annualized_excess_sharpe),
                        uniform_rule_fixed_size_sharpe=float(result.set_index('model').loc['spread_full','annualized_excess_sharpe']),
                        matched_sizing_evidence=uplift.to_dict('records'),
                        archived_same_signal_sizing_comparison=archived.to_dict('records'),
                        archived_comparability='Archived sizing controls share the same frozen wing-specific entry/direction policies with each other. The new uniform study changes feature definitions, entry policy and pooled learner scope, so its maximum is not a rerun of the0.778 checkpoint.',
                        pure_sizing_control='spread_flex_selected_full preserves every spread-flex cash decision, signal date, direction and feature; only positive sizes become100%. Risk admission may consequently differ.',
                        sizing_control_timing='One post-selection diagnostic added after the nine predeclared variants; no model retraining or parameter search.',
                        interpretation='A spread rule plus learned allocation beats the uniform fixed-size control modestly. The common positive-history veto and RL entry/direction are weaker. The older0.778 portfolio remains the strongest archived combined portfolio. These are exploratory sizing gains, not conclusive evidence of general RL superiority.',
                        tests='11 uniform policy, risk, corporate lifecycle and accounting tests pass')
        write_json(REPORT/'conclusions.json',conclusion)
        (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+PROTOCOL+'\n\n'+json.dumps(audit,indent=2)+'\n')
        paths=[config_path,Path('scripts/run_signal_discovery_guarded.py'),Path('tests/test_uniform_smile_policy.py'),Path('tests/test_uniform_smile_risk.py'),Path('scripts/run_uniform_smile.py'),Path('scripts/uniform_smile_policy.py'),Path('scripts/uniform_smile_data.py'),Path('scripts/uniform_smile_risk.py'),Path('scripts/uniform_smile_engine.py'),Path('scripts/joint_smile_engine.py')]
        hashes={str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
        for path in paths:hashes[str(path)]=digest(path)
        write_json(REPORT/'output_hashes.json',hashes)
    else:data.progress(args.phase+'_complete',elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':main()
