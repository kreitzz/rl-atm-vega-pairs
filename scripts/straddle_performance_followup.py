"""Transfer portfolio performance context to actual funded straddle entry size.

Uses a causal fixed-half benchmark's contemporaneous marks as external market
state. This is not feedback from the learner's own inventory. Every proposal,
cost, reward archive, and execution limit remains unchanged.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import straddle_context_followup as study

CONFIG=Path('config/straddle_performance_followup.json')
CFG=json.loads(CONFIG.read_text())
OUT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
SOURCE=Path(CFG['feature_source'])
PERFORMANCE=['benchmark_sharpe21','benchmark_sharpe63','benchmark_sharpe126',
    'benchmark_vol63','benchmark_drawdown63','benchmark_reserve','benchmark_available']
CAPACITY=['commission_vol_points','quoted_vol_width','put_call_iv_basis','dte',
    'half_contract_count','half_vega_utilization','full_vega_utilization','signal_vega_available']
BASE=study.CORE+['vix_change21','iv_change5']
FEATURES=dict(performance=BASE+PERFORMANCE,capacity=BASE+PERFORMANCE+CAPACITY)


def benchmark_features(nav):
    r=nav.excess_return
    frame=pd.DataFrame(index=nav.index)
    for n,minimum in [(21,15),(63,42),(126,63)]:
        average=r.rolling(n,min_periods=minimum).mean()
        std=r.rolling(n,min_periods=minimum).std()
        frame[f'benchmark_sharpe{n}']=(np.sqrt(252)*average/std.where(std>1e-10)).clip(-5,5)
    frame['benchmark_vol63']=np.sqrt(252)*r.rolling(63,min_periods=42).std()
    relative=nav.nav/nav.cash_benchmark
    frame['benchmark_drawdown63']=relative/relative.rolling(63,min_periods=42).max()-1
    frame['benchmark_reserve']=nav.capital_reserve/nav.nav
    frame['benchmark_available']=frame.notna().all(axis=1).astype(float)
    return frame.replace([np.inf,-np.inf],np.nan).fillna(0)


def contract_features(commission_vol_points, fee=.03):
    vega=2*fee/commission_vol_points.where(commission_vol_points>0)
    half=np.floor(225/vega);full=np.floor(450/vega)
    return pd.DataFrame(dict(half_contract_count=np.log1p(half),
        half_vega_utilization=half*vega/225,full_vega_utilization=full*vega/450,
        signal_vega_available=vega.notna().astype(float))).replace([np.inf,-np.inf],np.nan).fillna(0)


def prepare_features(_limit):
    # Assemble directly in the next phase; no new raw-data requests or labels.
    pass


def assemble():
    needed=list(dict.fromkeys(study.PARENT.unit.bandit.EXECUTION+['secid']+BASE))
    rows=pd.read_parquet(SOURCE/'proposals.parquet',columns=needed,use_threads=False)
    native=pd.read_parquet(Path(CFG['source_root'])/'proposals.parquet',
        columns=['date','secid','commission_vol_points','quoted_vol_width','put_call_iv_basis','dte'],use_threads=False)
    rows=rows.merge(native,on=['date','secid'],how='left',validate='one_to_one')
    nav=pd.read_parquet('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/daily_nav.parquet',
        columns=['excess_return','nav','cash_benchmark','capital_reserve'],use_threads=False)
    context=benchmark_features(nav).reindex(rows.date).reset_index(drop=True)
    for c in PERFORMANCE:rows[c]=context[c].to_numpy()
    rounding=contract_features(rows.commission_vol_points,study.PARENT.unit.base.execution_settings()['option_commission'])
    for c in rounding:rows[c]=rounding[c]
    names=list(dict.fromkeys(sum(FEATURES.values(),[])))
    assert len(rows)==35774 and np.isfinite(rows[names]).all().all()
    rows.to_parquet(OUT/'proposals.parquet',index=False)
    p=OUT/'base_proposals.parquet'
    if not p.exists():p.symlink_to((SOURCE/'base_proposals.parquet').resolve())
    meta=pd.read_parquet(SOURCE/'training.parquet',use_threads=False)
    meta=meta.drop(columns=[c for c in names if c in meta])
    meta=meta.merge(rows[['date','secid']+names],on=['date','secid'],how='left',validate='many_to_one')
    assert int(meta.finite_label.sum())==4116 and np.isfinite(meta.loc[meta.finite_label,names]).all().all()
    meta.to_parquet(OUT/'training.parquet',index=False)
    study.PARENT.core.dump(OUT/'training_manifest.json',{str(p):study.previous.digest(p) for p in
        [OUT/'training.parquet',OUT/'proposals.parquet',Path(CFG['source_root'])/'training_manifest.json']})
    study.PARENT.core.dump(REPORT/'protocol.json',dict(configuration=CFG,features=FEATURES,
        decisions='Signal-close benchmark marks and signal-close quote vegas; next-session option execution.',
        benchmark='Fixed50 funded expert, which can be maintained as an external paper portfolio. Not the learner own inventory, not future completed profits.',
        rounding='Signal-close estimates of whole-contract counts and utilized vega under225/450 dollar budgets; actual next-session quantities remain determined by the unchanged simulator.',
        proposals=len(rows),valid_labels=int(meta.finite_label.sum()),unchanged_execution=True))
    print(json.dumps(dict(phase='assembled',proposals=len(rows),features=len(names))),flush=True)


def sources():
    files=[CONFIG,Path(__file__),Path(study.__file__),SOURCE/'training_manifest.json',
        Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/daily_nav.parquet'),
        Path(study.PARENT.core.__file__),Path(study.PARENT.unit.engine.__file__),Path('paper_code/execution.json')]
    record={str(p):study.previous.digest(p) for p in files};p=OUT/'sources.json'
    if p.exists():assert json.loads(p.read_text())==record,'Changed source; create a new study root'
    else:study.PARENT.core.dump(p,record)


def main():
    study.CONFIG=CONFIG;study.CFG=CFG;study.OUT=OUT;study.REPORT=REPORT;study.SOURCE=SOURCE;study.FEATURES=FEATURES
    study.prepare_features=prepare_features;study.assemble=assemble;study.sources=sources
    # Keep the preceding tests and add benchmark-specific causality checks.
    import sys
    if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase')+1]=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_performance_followup','tests.test_research_atomic_io','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:study.main()


if __name__=='__main__':main()
