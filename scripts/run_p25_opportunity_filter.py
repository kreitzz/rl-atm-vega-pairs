"""Contextual P25 entry selection: broad shadow discovery and covered funded gates.

Run with .venv-pca312: python -m scripts.run_p25_opportunity_filter.
Existing strategy/data remain read-only. One worker, one thread,20% CPU duty.
"""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_representation as base
from scripts import p25_opportunity_dataset as data
from scripts.p25_opportunity_models import fit_month, best_per_pair_date, SETTINGS
from scripts.analyze_p25_opportunity_filter import analyze, prediction_diagnostics, paired_sharpe_interval, sharpe
from scripts.option_flex.contract_engine import MarketRow

ROOT = base.ROOT
OUT = ROOT/'artifacts/contracts/p25_opportunity_filter'
REPORT = ROOT/'reports/p25_opportunity_filter'
CASES = ['broad_ridge','broad_spline','rank_ridge','rank_spline']
MEMORY_ARCHIVE = ROOT/'data/p25_memory_ablation_experiment.zip'
CONFIG = base.BUNDLE/'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json'


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():
            raise InterruptedError('Opportunity experiment STOP requested')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    audit = json.loads((OUT/'dataset/dataset_audit.json').read_text())
    protocol = dict(
        cases=CASES, features=data.FEATURES, settings=SETTINGS,
        training='Monthly18month completed outcomes; two past3month validation folds; alpha chosen by OOF MSE; nonnegative hurdle chosen by signal-date lifetime payoff mean/sd surrogate.',
        broad='All11977 inherited eligible pair-dates, both directions; choose at most one direction or skip. No rank or profit-memory gates.',
        rank='All4744 extreme-rank candidates in inherited direction, learned skip/fullsize gate replaces profit-memory screen. Separate models trained on historical rank-selected directions only.',
        target=data.LABEL_CONVENTION,
        shadow='Sum archived standalone daily net excess P&L paths at fixed reference450vega; zero paths only if stored finite; unresolved paths never zero-imputed. Incomplete books labeled observed-path diagnostics.',
        funded='Rank arms only; same10m cash ledger, entry450vega per leg scaled with account, hold to expiry, daily hedge, .03 commission, risk limits and carry. No sizing optimization.',
        controls=['current_fixed100','rank_all_fixed100'],
        comparisons='2018-2021 earlier and2022-Aug2025 later; both models reported; no automatic promotion or model chosen on later results.',
        uncertainty='Paired circular calendar blocks63/126,2000resamples; conditional descriptive intervals, no research-selection adjustment.',
        broad_funded_blocker=audit['funded_quote_contract_coverage'],
        untouched_holdout=False, inherited_universe_bias=True,
        dataset_sha256=base.digest(OUT/'dataset/candidates.parquet'),
        audit_sha256=base.digest(OUT/'dataset/dataset_audit.json'),
        protected_sources={str(p.relative_to(ROOT)):base.digest(p) for p in [
            ROOT/'config/p25_submission.json',base.CONFIG,CONFIG,base.SOURCE,data.SOURCE,
            ROOT/'scripts/uniform_smile_engine.py',ROOT/'scripts/uniform_smile_risk.py']},
        scripts={p:base.digest(ROOT/'scripts'/p) for p in ['p25_opportunity_dataset.py','p25_opportunity_models.py','run_p25_opportunity_filter.py','analyze_p25_opportunity_filter.py']},
        resources=dict(threads=1,duty=.2,rss_limit_mib=384))
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Protocol/source changed; use a separate experiment folder')
    base.write(path,protocol)
    return protocol


def read_candidates():
    columns=list(dict.fromkeys(data.FEATURES+[
        'date','end','pair','direction','target','finite_label','train_eligible','evaluation_eligible',
        'candidate_id','label_row','label_action','position','wing','secid_a','secid_b','optionid_a',
        'optionid_b','base_level','legacy_selected','legacy_rank_selected','rule_direction','iv_rank',
        'archived_path_status']))
    return pd.read_parquet(OUT/'dataset/candidates.parquet',columns=columns)


def train(rows,cal,budget,limit=None):
    signature=base.digest(REPORT/'protocol.json')
    months=pd.period_range('2018-01','2025-06',freq='M')
    if limit: months=months[:limit]
    for case in CASES:
        source=rows if case.startswith('broad') else rows.loc[rows.legacy_rank_selected]
        live=source.loc[source.evaluation_eligible]
        for month in months:
            folder=OUT/'models'/case; folder.mkdir(parents=True,exist_ok=True)
            stamp=str(month.start_time.date()); path=folder/f'{stamp}.joblib'
            if path.exists():
                rule=joblib.load(path)
                assert rule.metadata['protocol_sha256']==signature
            else:
                rule=fit_month(source,data.FEATURES,month.start_time,kind=case.split('_')[1],calendar=cal,budget=budget)
                rule.metadata['protocol_sha256']=signature
                joblib.dump(rule,path)
                base.write(folder/f'{stamp}.json',rule.metadata)
            frame=live.loc[live.date.dt.to_period('M').eq(month)]
            if len(frame):
                scored=rule.select(frame)
                scored.to_parquet(folder/f'{stamp}_scores.parquet',index=False)
                selected=int(scored.selected.sum())
            else: selected=0
            print(f'{case} {stamp}: {rule.metadata["training_rows"]} training directions; {selected} selected; {rule.metadata["status"]}',flush=True)
            del rule
            gc.collect(); budget.rest()
        if not limit:
            pieces=[pd.read_parquet(p) for p in sorted((OUT/'models'/case).glob('*_scores.parquet'))]
            scored=pd.concat(pieces,ignore_index=True)
            assert len(scored)==len(live)
            assert not scored.duplicated(['date','pair','direction']).any()
            folder=OUT/'predictions'; folder.mkdir(exist_ok=True)
            scored.to_parquet(folder/f'{case}.parquet',index=False)
            del pieces,scored
            gc.collect();budget.rest()


def shadow(rows,cal,budget):
    """Archived paths in actual holding dates; incomplete selected paths tracked."""
    evalrows=rows.loc[rows.evaluation_eligible]
    cases={
        'current_fixed100':evalrows.loc[evalrows.legacy_selected].copy(),
        'rank_all_fixed100':evalrows.loc[evalrows.legacy_rank_selected].copy(),
    }
    predictions=[]
    for case in CASES:
        frame=pd.read_parquet(OUT/'predictions'/f'{case}.parquet')
        winners=best_per_pair_date(frame)
        winners['case']=case
        predictions.append(winners)
        cases[case]=frame.loc[frame.selected].copy()
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_parquet(OUT/'prediction_winners.parquet',index=False)
    bins,coverage=prediction_diagnostics(pred)
    bins.to_csv(REPORT/'prediction_bins.csv',index=False)
    coverage.to_csv(REPORT/'prediction_coverage.csv',index=False)
    calendar=cal[cal>='2018-01-01']
    books={}; periods=[]; pairs=[]; missing=[]; annual=[]
    for case,chosen in cases.items():
        chosen=chosen.reset_index(drop=True)
        paths=data.load_paths(chosen,check=budget.rest)
        available=np.isfinite(paths).all(axis=1)
        missing.append(dict(case=case,selected=len(chosen),finite_paths=int(available.sum()),unresolved_paths=int((~available).sum()),
            complete_training_labels=int(chosen.finite_label.sum()),zero_finite_paths=int((available & (np.nansum(abs(paths),axis=1)==0)).sum()),
            selected_middle_rank=int(chosen.rule_direction.eq(0).sum()),chosen_pairs=int(chosen.pair.nunique())))
        pairbook=pd.DataFrame(0.,index=cal,columns=sorted(chosen.pair.unique()))
        for i,row in enumerate(chosen.itertuples()):
            if not available[i]: continue
            positions=int(row.position)+1+np.arange(paths.shape[1])
            ok=positions<len(cal)
            pairbook.loc[cal[positions[ok]],row.pair]+=paths[i,ok].astype(float)*450
        pairbook=pairbook.loc[calendar]
        pnl=pairbook.sum(axis=1);books[case]=pnl
        folder=OUT/'shadow';folder.mkdir(exist_ok=True)
        pairbook.to_parquet(folder/f'{case}_pair_daily.parquet')
        pnl.rename('observed_shadow_excess_pnl').to_frame().to_parquet(folder/f'{case}_daily.parquet')
        chosen['path_available']=available
        chosen.to_parquet(folder/f'{case}_selected.parquet',index=False)
        for period,start,end in [('earlier','2018-01-01','2021-12-31'),('later','2022-01-01','2025-08-29'),('full','2018-01-01','2025-08-29')]:
            p=pnl.loc[start:end];group=chosen.loc[chosen.date.between(start,end)]
            periods.append(dict(case=case,period=period,observed_shadow_sharpe=sharpe(p),observed_shadow_net_pnl=p.sum(),
                proposals=len(group),pairs=int(group.pair.nunique()),middle_rank=int(group.rule_direction.eq(0).sum()),
                unresolved_selected=int((~group.path_available).sum()),complete_book=bool(available.all())))
            totals=pairbook.loc[start:end].sum()
            for pair,value in totals.items(): pairs.append(dict(case=case,period=period,pair=pair,observed_shadow_net_pnl=value))
        for year in range(2018,2026):
            p=pnl.loc[str(year)]
            annual.append(dict(case=case,year=year,observed_shadow_sharpe=sharpe(p),observed_shadow_net_pnl=p.sum()))
        del paths,pairbook
        gc.collect();budget.rest()
    pd.DataFrame(missing).to_csv(REPORT/'shadow_coverage.csv',index=False)
    pd.DataFrame(periods).to_csv(REPORT/'shadow_periods.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'shadow_pairs.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'shadow_annual.csv',index=False)
    intervals=[]
    for case in CASES:
        for block in [63,126]:
            stats=paired_sharpe_interval(books[case].loc['2022':],books['current_fixed100'].loc['2022':],block=block,rest=budget.rest)
            intervals.append(dict(case=case,reference='current_fixed100',scope='observed_shadow_paths_only',**stats))
    pd.DataFrame(intervals).to_csv(REPORT/'shadow_sharpe_intervals.csv',index=False)
    print(pd.DataFrame(periods).to_string(index=False),flush=True)


def extract_member(z,member,destination,manifest):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        with z.open(member) as source,destination.open('wb') as target: shutil.copyfileobj(source,target,1024*1024)
    if base.digest(destination)!=manifest[member]:
        raise ValueError(f'Archive checksum mismatch {member}')


class QuoteMarket(base.FrozenMarket):
    def __init__(self,budget,secids):
        ids=set(secids)
        if 101322 in ids: ids.add(102968)
        stock=pd.read_parquet(base.MARKET/'stocks.parquet',columns=['date','secid','close','cfadj'],filters=[('secid','in',sorted(ids))])
        self.spots=stock.pivot(index='date',columns='secid',values='close')
        self.adjustments=stock.pivot(index='date',columns='secid',values='cfadj')
        self.rates=pd.read_parquet(base.MARKET/'rates.parquet').rate
        self.divs=pd.read_parquet(base.MARKET/'dividends.parquet')
        self.reported_year=None;self.last_check=None;self.budget=budget;self.days=OrderedDict()
        self.db=sqlite3.connect(f'file:{(OUT/"quotes.sqlite").as_posix()}?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-2048')

    def rate(self,date):
        if date!=self.last_check:
            self.budget.rest();self.last_check=date
        return super().rate(date)

    def quote(self,date,oid):
        if date not in self.days:
            daily={}
            for (payload,) in self.db.execute('SELECT payload FROM quotes WHERE date=?',(str(date.date()),)):
                row=json.loads(payload)
                for key,value in row.items():
                    if key in ['date','expiry'] and value is not None: row[key]=pd.Timestamp(value)
                    elif value is None: row[key]=np.nan
                daily[int(row['optionid'])]=MarketRow(row)
            self.days[date]=daily
            while len(self.days)>2:self.days.popitem(last=False)
        self.days.move_to_end(date)
        return self.days[date].get(int(oid))


def replay(rows,cal,budget):
    manifest=json.loads((ROOT/'reports/p25_memory_ablation/bundle_manifest.json').read_text())['files']
    prefix='artifacts/contracts/p25_memory_ablation/'
    with zipfile.ZipFile(MEMORY_ARCHIVE) as z:
        extract_member(z,prefix+'quotes.sqlite',OUT/'quotes.sqlite',manifest)
        folder=OUT/'replays/rank_all_fixed100'
        for name in ['daily_nav.parquet','trades.csv','decisions.csv']:
            extract_member(z,prefix+'replays/no_memory_gate_fixed100/'+name,folder/name,manifest)
    folder=OUT/'replays/current_fixed100';folder.mkdir(parents=True,exist_ok=True)
    for name in ['daily_nav.parquet','trades.csv','decisions.csv']:
        source=base.BUNDLE/'artifacts/contracts/p25_capital_sensitivity/replays/10m_fixed100'/name
        shutil.copyfile(source,folder/name)
    cfg=base.Settings(**json.loads(CONFIG.read_text()))
    checkpoint=json.loads((ROOT/'config/p25_submission.json').read_text())
    for case in ['rank_ridge','rank_spline']:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        signals=pd.read_parquet(OUT/'predictions'/f'{case}.parquet')
        signals['learned_level']=np.where(signals.selected,4,0)
        assert len(signals)==4744 and signals.legacy_rank_selected.all()
        signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),predictions_sha256=base.digest(OUT/'predictions'/f'{case}.parquet'))
        if (folder/'status.json').exists():
            saved=json.loads((folder/'status.json').read_text())
            assert all(saved[k]==v for k,v in signature.items())
            continue
        signals.to_parquet(folder/'scored_allocations.parquet',index=False)
        market=QuoteMarket(budget,set(signals.secid_a.astype(int))|set(signals.secid_b.astype(int)))
        original=base.engine.book_risk
        try:
            base.engine.book_risk=base.book_risk
            nav,trades,decisions,ledger=base.engine.simulate(market,signals,cal[cal>='2018-01-01'],cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=checkpoint['corporate_events'],wing_capital=1_000_000)
        finally:base.engine.book_risk=original
        base.mark_book(nav,market.rates,cfg.capital)
        audit=base.verify_ledger(nav,trades,decisions,signals)
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
        base.write(folder/'status.json',dict(status='complete',**signature,**audit,**ledger))
        market.db.close();del market,nav,trades,decisions,signals
        gc.collect();budget.rest()
        print(f'{case}: funded ledger reconciled; {audit["entries"]} entries',flush=True)


def report(budget):
    folders={c:OUT/'replays'/c for c in ['current_fixed100','rank_all_fixed100','rank_ridge','rank_spline']}
    pred=pd.read_parquet(OUT/'prediction_winners.parquet')
    analyze(folders,REPORT/'funded',reference='current_fixed100',predictions=pred.loc[pred.case.str.startswith('rank')],rest=budget.rest)
    fits=[]
    for case in CASES:
        for path in sorted((OUT/'models'/case).glob('*.json')):
            item=json.loads(path.read_text())
            fits.append(dict(case=case,cutoff=item['cutoff'],training_rows=item['training_rows'],training_pair_dates=item['training_pair_dates'],
                status=item['status'],alpha=item['alpha_multiplier'],hurdle=item['hurdle'],latest_outcome=item['latest_training_outcome']))
    pd.DataFrame(fits).to_csv(REPORT/'fits.csv',index=False)
    protocol=json.loads((REPORT/'protocol.json').read_text())
    for path,expected in protocol['protected_sources'].items():assert base.digest(ROOT/path)==expected
    base.write(REPORT/'completion.json',dict(status='tests_complete_broad_funded_unavailable',monthly_models=len(fits),
        funded_new_replays=2,broad_funded_unavailable_missing_quotes=5712,protected_sources_unchanged=True,peak_rss_mib=budget.peak))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','train','shadow','replay','report'],default='all')
    parser.add_argument('--limit-fits',type=int)
    args=parser.parse_args()
    if args.limit_fits and args.phase!='train':parser.error('Fit limit is training preflight only')
    os.chdir(ROOT);prepare();budget=Budget()
    rows=read_candidates();cal=pd.read_parquet(base.MARKET/'rates.parquet').index
    assert (cal.get_indexer(rows.date)==rows.position.to_numpy()).all()
    budget.rest()
    if args.phase in ['all','train']:train(rows,cal,budget,args.limit_fits)
    if args.phase in ['all','shadow']:shadow(rows,cal,budget)
    if args.phase in ['all','replay']:replay(rows,cal,budget)
    if args.phase in ['all','report']:report(budget)


if __name__=='__main__':main()
