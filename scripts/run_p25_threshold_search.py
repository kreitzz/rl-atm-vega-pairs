"""Direct, causal search over IV-rank cutoffs, using actual funded returns.

Separate counterfactual candidate books provide past-only selection evidence.
Every deployed selection has its own continuous cash/inventory replay.
"""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scripts import run_p25_representation as base
from scripts import run_p25_opportunity_filter as prior
from scripts.p25_threshold_policy import ThresholdPolicy, proposal_mask, select_policy
from scripts.analyze_p25_opportunity_filter import analyze, period_metrics

ROOT = base.ROOT
OUT = ROOT/'artifacts/contracts/p25_threshold_search'
REPORT = ROOT/'reports/p25_threshold_search'
DEPLOYED = ['q20', 'selected_fixed', 'selected_rolling', 'selected_context']


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():
            raise InterruptedError('Threshold search STOP requested')


class ThresholdMarket(prior.QuoteMarket):
    """Decode only contracts this replay can hold; keep the same quote values."""
    def __init__(self,budget,signals,quotes):
        super().__init__(budget,set(signals.secid_a.astype(int))|set(signals.secid_b.astype(int)))
        self.wanted=set(signals.optionid_a.astype(int))|set(signals.optionid_b.astype(int))
        self.db.close()
        self.db=sqlite3.connect(f'file:{quotes.as_posix()}?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-2048')

    def quote(self,date,oid):
        if date not in self.days:
            daily={}
            for optionid,payload in self.db.execute('SELECT optionid,payload FROM quotes WHERE date=?',(str(date.date()),)):
                if optionid not in self.wanted:
                    continue
                row=json.loads(payload)
                for key,value in row.items():
                    if key in ['date','expiry'] and value is not None:
                        row[key]=pd.Timestamp(value)
                    elif value is None:
                        row[key]=np.nan
                daily[int(optionid)]=prior.MarketRow(row)
            self.days[date]=daily
            while len(self.days)>2:
                self.days.popitem(last=False)
        self.days.move_to_end(date)
        return self.days[date].get(int(oid))


def families(max_tail):
    grid = [q for q in [.05,.10,.15,.20,.25,.30] if q <= max_tail+1e-10]
    fixed = {f'q{round(q*100):02d}': ThresholdPolicy(f'q{round(q*100):02d}',q) for q in grid}
    context_grid = [.10,.20,.30] if max_tail >= .30 else [.10,.20]
    context = {key: p for key,p in fixed.items() if p.q_low in context_grid}
    for low in context_grid:
        for high in context_grid:
            if low == high:
                continue
            key = f'vix{round(low*100):02d}_{round(high*100):02d}'
            context[key] = ThresholdPolicy(key,low,high)
    return fixed, context


def prepare(max_tail, quotes, rows):
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    fixed,context = families(max_tail)
    all_policies = fixed | context
    # Identical diagonal context policies can share fixed-policy books only
    # after establishing that the already-lagged VIX is available everywhere.
    assert rows.vix.gt(0).all() and np.isfinite(rows.vix).all()
    baseline = rows.loc[proposal_mask(rows,fixed['q20'])]
    assert set(zip(baseline.date,baseline.pair,baseline.direction)) == set(zip(
        rows.loc[rows.legacy_selected,'date'],rows.loc[rows.legacy_selected,'pair'],rows.loc[rows.legacy_selected,'direction']))
    assert len(baseline)==608
    with sqlite3.connect(f'file:{quotes.as_posix()}?mode=ro',uri=True) as db:
        ids={int(x[0]) for x in db.execute('SELECT DISTINCT optionid FROM quotes')}
    coverage=[]
    for key,p in all_policies.items():
        selected=rows.loc[proposal_mask(rows,p)]
        covered=selected.optionid_a.isin(ids)&selected.optionid_b.isin(ids)
        coverage.append(dict(case=key,proposals=len(selected),missing_contract_proposals=int((~covered).sum())))
        if not covered.all():
            raise ValueError(f'{key}: missing contracts; cannot coverage-filter proposals')
    pd.DataFrame(coverage).to_csv(REPORT/'funded_input_coverage.csv',index=False)
    files=[prior.OUT/'dataset/candidates.parquet',prior.CONFIG,ROOT/'config/p25_submission.json',
        ROOT/'scripts/uniform_smile_engine.py',ROOT/'scripts/uniform_smile_risk.py',
        ROOT/'scripts/run_p25_threshold_search.py',ROOT/'scripts/p25_threshold_policy.py',quotes]
    protocol=dict(
        question='Does direct past-only selection of the IV-rank opportunity cutoff improve funded returns?',
        policies={key:asdict(p) for key,p in all_policies.items()},fixed_family=list(fixed),context_family=list(context),
        max_tail=max_tail,quotes=str(quotes),
        isolation='Change IVrank gate only; retain directional91/365day profit>0 and count>=21/63, inherited universe/calendar/beta screens, fixed100% sizing, all costs/risk/expiry/dailyhedge rules.',
        candidate_books='Continuous funded2018-Aug2025 ledgers, .03 option commission. Past marked daily excess returns including cash days; not lifetime label targets.',
        fixed='Choose max funded excess Sharpe in2018-2021 with >=252 sessions and>=20 entries; deterministic proximity-to20% ties. Follow baseline until2022, then locked choice.',
        rolling='Each signal month select from fixed candidates on trailing18calendar months strictly before month, >=252 sessions and>=20 entries per eligible candidate; fallback20%. Replay actual changing decisions, never splice NAV.',
        context='Prior VIX/100 >=.20 defines high regime, no additional lag; low/high IVrank tails chosen jointly by whole-policy2018-2021 funded Sharpe. Baseline until2022 then locked policy; no rolling contextual search.',
        deployment='Separate continuous actual inventory for each selection; fixed/context carry baseline holdings and NAV through2022 boundary.',
        later='2022-01-03 through2025-08-29 with signals ending2025-06-30; exploratory previously inspected history, not untouched holdout.',
        inference='Paired circular63/126session blocks,2000 replications; conditional descriptive intervals, not adjusted for repeated research or multiple comparisons.',
        limitations=['Inherited retrospective pair roster and formation calendar','Memory gates unchanged and based on archived cost convention','Grid and VIX regime family restricted in advance; not global optimality'],
        resources=dict(threads=1,duty=.2,rss_limit_mib=384),
        hashes={str(p.relative_to(ROOT)):base.digest(p) for p in files})
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Frozen protocol changed; use separate experiment directory')
    base.write(path,protocol)
    return fixed,context,protocol


def replay(case, signals, cal, quotes, budget):
    folder=OUT/'replays'/case; folder.mkdir(parents=True,exist_ok=True)
    signals=signals.sort_values(['date','pair'],kind='stable').reset_index(drop=True)
    assert not signals.duplicated(['date','pair','wing']).any()
    signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),
        signal_identity_sha256=__import__('hashlib').sha256(signals[['date','pair','direction']].to_csv(index=False).encode()).hexdigest())
    if (folder/'status.json').exists():
        status=json.loads((folder/'status.json').read_text())
        assert all(status[k]==v for k,v in signature.items())
        return
    print(f'Starting {case}: {len(signals)} proposals',flush=True)
    signals['learned_level']=4
    signals.to_parquet(folder/'scored_allocations.parquet',index=False)
    cfg=base.Settings(**json.loads(prior.CONFIG.read_text()))
    checkpoint=json.loads((ROOT/'config/p25_submission.json').read_text())
    market=ThresholdMarket(budget,signals,quotes)
    original=base.engine.book_risk
    try:
        base.engine.book_risk=base.book_risk
        nav,trades,decisions,ledger=base.engine.simulate(market,signals,cal[cal>='2018-01-01'],cfg,1_000_000,
            policy=lambda row,context:4,corporate_events=checkpoint['corporate_events'],wing_capital=1_000_000)
    finally:
        base.engine.book_risk=original
    base.mark_book(nav,market.rates,cfg.capital)
    audit=base.verify_ledger(nav,trades,decisions,signals)
    if case=='q20':
        reference=pd.read_parquet(prior.OUT/'replays/current_fixed100/daily_nav.parquet')
        np.testing.assert_allclose(nav.nav,reference.nav,rtol=0,atol=1e-7)
        audit['baseline_max_nav_difference']=float((nav.nav-reference.nav).abs().max())
    nav.to_parquet(folder/'daily_nav.parquet')
    trades.to_csv(folder/'trades.csv',index=False); decisions.to_csv(folder/'decisions.csv',index=False)
    base.write(folder/'status.json',dict(status='complete',**signature,**audit,**ledger))
    market.db.close();del market,nav,trades,decisions
    gc.collect();budget.rest()
    print(f'Finished {case}: {audit["entries"]} entries; ledger reconciled',flush=True)


def select_and_replay(rows,cal,quotes,fixed,context,budget):
    policies=fixed|context
    books={key:pd.read_parquet(OUT/'replays'/key/'daily_nav.parquet').excess_return for key in policies}
    decisions={key:pd.read_csv(OUT/'replays'/key/'decisions.csv',parse_dates=['signal_date','entry_date']) for key in policies}
    fits=[];scores=[]
    schedule=[]
    for family,case in [(fixed,'selected_fixed'),(context,'selected_context')]:
        result=select_policy(books,decisions,family,'2022-01-01',lookback_months=None,start='2018-01-01')
        base.write(OUT/'selections'/f'{case}.json',result)
        fits.append(dict(case=case,**{k:v for k,v in result.items() if k!='candidates'}))
        scores.extend(dict(case=case,asof=result['asof'],**r) for r in result['candidates'])
        chosen=family[result['policy']]
        before=rows.date.lt('2022-01-01')
        selected=(before&proposal_mask(rows,fixed['q20']))|(~before&proposal_mask(rows,chosen))
        replay(case,rows.loc[selected].copy(),cal,quotes,budget)
        print(f'{case}: past-only choice {result["policy"]}',flush=True)
    parts=[]
    for month in pd.period_range('2018-01','2025-06',freq='M'):
        result=select_policy(books,decisions,fixed,month.start_time)
        base.write(OUT/'selections/rolling'/f'{month}.json',result)
        fits.append(dict(case='selected_rolling',**{k:v for k,v in result.items() if k!='candidates'}))
        scores.extend(dict(case='selected_rolling',asof=result['asof'],**r) for r in result['candidates'])
        group=rows.loc[rows.date.dt.to_period('M').eq(month)]
        chosen=fixed[result['policy']]
        part=group.loc[proposal_mask(group,chosen)].copy()
        part['selected_policy']=result['policy'];parts.append(part)
        schedule.append(dict(month=str(month),policy=result['policy'],tail=chosen.q_low,status=result['status'],proposals=len(part)))
    pd.DataFrame(fits).to_csv(REPORT/'selection_fits.csv',index=False)
    pd.DataFrame(scores).to_csv(REPORT/'selection_scores.csv',index=False)
    pd.DataFrame(schedule).to_csv(REPORT/'rolling_schedule.csv',index=False)
    replay('selected_rolling',pd.concat(parts,ignore_index=True),cal,quotes,budget)


def report(policies,protocol,budget):
    # Every predeclared candidate is disclosed, including later losers.
    records=[];annual=[]
    for case in policies:
        folder=OUT/'replays'/case
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        for name,start,stop in [('validation','2018-01-01','2021-12-31'),('evaluation','2022-01-01','2025-08-29')]:
            records.append(dict(case=case,period=name,**period_metrics(nav,decisions,start,stop)))
        for year in range(2018,2026):
            annual.append(dict(case=case,year=year,**period_metrics(nav,decisions,f'{year}-01-01',f'{year}-12-31')))
        budget.rest()
    pd.DataFrame(records).to_csv(REPORT/'candidate_periods.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'candidate_annual.csv',index=False)
    analyze({key:OUT/'replays'/key for key in DEPLOYED},REPORT/'funded',reference='q20',rest=budget.rest)
    for p,expected in protocol['hashes'].items():
        assert base.digest(ROOT/p)==expected
    base.write(REPORT/'completion.json',dict(status='complete',candidate_books=len(policies),deployment_books=3,
        protected_sources_unchanged=True,peak_rss_mib=budget.peak))
    print(pd.read_csv(REPORT/'funded/periods.csv').to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','candidates','deploy','report'],default='all')
    parser.add_argument('--max-tail',type=float,choices=[.20,.30],required=True)
    parser.add_argument('--quotes',type=Path,default=prior.OUT/'quotes.sqlite')
    args=parser.parse_args()
    budget=Budget()
    rows=prior.read_candidates(); rows=rows.loc[rows.evaluation_eligible].copy()
    cal=pd.read_parquet(base.MARKET/'rates.parquet').index
    fixed,context,protocol=prepare(args.max_tail,args.quotes.resolve(),rows)
    budget.rest()
    if args.phase in ['all','candidates']:
        policies=fixed|context
        for key in ['q20']+[k for k in policies if k!='q20']:
            p=policies[key]
            replay(key,rows.loc[proposal_mask(rows,p)].copy(),cal,args.quotes.resolve(),budget)
    if args.phase in ['all','deploy']:
        select_and_replay(rows,cal,args.quotes.resolve(),fixed,context,budget)
    if args.phase in ['all','report']:
        report(fixed|context,protocol,budget)


if __name__=='__main__':
    main()
