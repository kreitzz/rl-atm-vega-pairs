"""50/75/100% straddle entry sizes, reusing exact unit-ledger machinery.

Keep both previous 50/100/200% experiments untouched. Recompute the 75% reward
with integer contracts and exact cash accrual, rather than interpolate returns.
No daily live proposals are removed by training sample availability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from scripts import atm_rv_straddle_bandit as original

CFG_PATH=Path('config/straddle_capped_books.json');CFG=json.loads(CFG_PATH.read_text())
OUT=Path(CFG['output_root']);REPORT=Path(CFG['report_root'])
PREVIOUS=Path('artifacts/contracts/atm_rv_straddle_bandit')


def configure():
    original.OUT=OUT;original.REPORT=REPORT;original.CFG=CFG;original.CFG_PATH=CFG_PATH


def check_sources():
    paths=[Path(__file__),CFG_PATH,Path(original.__file__),Path(original.bandit.__file__),
        Path(original.engine.__file__),Path(original.base.__file__),Path(CFG['source']),
        PREVIOUS/'sample.parquet',PREVIOUS/'training.parquet',Path('paper_code/execution.json'),
        Path('config/atm_rv_persistence.json')]
    database=Path(CFG['quote_root'])/'quotes.sqlite'
    record=dict(hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        quote_signature=[database.stat().st_size,database.stat().st_mtime_ns])
    marker=OUT.parent/'source_manifest.json'
    if marker.exists():assert json.loads(marker.read_text())==record,'Changed source: preserve outputs and use a new root'
    else:original.base.dump(marker,record)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    for name in ['sample.parquet','proposals.parquet']:
        if not (OUT/name).exists():shutil.copy2(PREVIOUS/name,OUT/name)
    folder=OUT/'unit_labels';folder.mkdir(exist_ok=True)
    meta=pd.read_parquet(PREVIOUS/'training.parquet',use_threads=False)
    # Unit entry eligibility is unchanged across sizing ladders. Reuse cached
    # unexecutable episodes, retaining every one of their live daily proposals.
    for i in meta.loc[~meta.finite_label,'sample_id']:
        for suffix in ['npz','parquet']:
            dest=folder/f'{i}.{suffix}'
            if not dest.exists():shutil.copy2(PREVIOUS/'unit_labels'/dest.name,dest)
    original.base.dump(REPORT/'protocol.json',dict(configuration=CFG,
        reused_unexecutable_units=int((~meta.finite_label).sum()),
        exact_75_percent_integer_reward_replay=True,unchanged_proposal_count=35774,
        selection='Entry learner inherited unchanged at 18 months; no entry hyperparameter selection.',
        scope='New funded straddle75 and entry-bandit books; daily delta hedge, same costs and risk limits. Previously inspected dates.'))


def full_size_target(record,paths):
    if record['finite_label']:record['full_profit']=float(paths[2].sum())
    return record,paths


def labels(limit):
    unit=original.label_one
    def corrected(*args,**kwargs):return full_size_target(*unit(*args,**kwargs))
    original.label_one=corrected
    try:original.labels(limit)
    finally:original.label_one=unit


def assemble():
    original.assemble()
    new=pd.read_parquet(OUT/'training.parquet',use_threads=False).sort_values('sample_id')
    old=pd.read_parquet(PREVIOUS/'training.parquet',use_threads=False).sort_values('sample_id')
    np.testing.assert_array_equal(new.sample_id,old.sample_id)
    np.testing.assert_array_equal(new.finite_label,old.finite_label)
    np.testing.assert_allclose(new.full_profit,old.full_profit,atol=1e-8,rtol=0,equal_nan=True)
    for path in sorted((OUT/'labels').glob('*.npz')):
        with np.load(path) as n,np.load(PREVIOUS/'labels'/path.name) as o:
            np.testing.assert_allclose(n['paths'][:,0],o['paths'][:,0],atol=1e-8,rtol=0)
            np.testing.assert_allclose(n['paths'][:,2],o['paths'][:,1],atol=1e-8,rtol=0)
    original.base.dump(REPORT/'ladder_audit.json',dict(samples=len(new),
        unchanged_unit_eligibility=True,unchanged_50_and_100_percent_reward_paths=True,
        full_profit_target_is_100_percent=True))


def book(case,months):
    if case=='fixed75':
        root=OUT.parent/'fixed75';alloc=root/'allocations/tree18_test.parquet'
        if not alloc.exists():
            alloc.parent.mkdir(parents=True,exist_ok=True)
            rows=pd.read_parquet(OUT/'proposals.parquet',use_threads=False)
            rows['learned_level']=3;rows.to_parquet(alloc,index=False)
        original.OUT=root
    else:root=OUT
    original.replay(months)
    status=root/'replays/tree18/status.json'
    if status.exists():
        record=json.loads(status.read_text());record.update(case=case,allowed_entry_size_percent=[50,75,100] if case=='entry_bandit' else [75])
        original.base.dump(status,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['test','prepare','labels','assemble','allocations','book','report'])
    parser.add_argument('--case',choices=['fixed75','entry_bandit']);parser.add_argument('--limit',type=int,default=32)
    args=parser.parse_args();configure()
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_capped_books','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    check_sources()
    if args.phase=='labels':labels(min(32,args.limit))
    elif args.phase=='book':
        if not args.case:parser.error('--case required')
        book(args.case,min(8,args.limit))
    else:dict(prepare=prepare,assemble=assemble,allocations=original.allocations,report=original.report)[args.phase]()


if __name__=='__main__':main()
