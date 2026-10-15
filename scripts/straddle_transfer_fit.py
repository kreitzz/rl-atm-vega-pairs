"""Fit monthly straddle policies without materializing the full feature matrix.

Read one signal month at a time and retain only execution keys in the final
allocation file. This preserves the policy calculation while keeping the
35-feature state within the original 256 MiB worker limit.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from scripts import straddle_feature_transfer as study


def scatter_levels(keys, frame, values, levels):
    indices=keys.get_indexer(pd.MultiIndex.from_frame(frame[['date','secid']]))
    if (indices<0).any() or len(np.unique(indices))!=len(indices):
        raise ValueError('Monthly predictions must match unique live proposals')
    levels[indices]=values


def allocations(name,later):
    study.verify_training()
    if later:
        selected=json.loads((study.REPORT/'selection.json').read_text())['selected']
        control=json.loads((study.REPORT/'evaluation_plan.json').read_text())['control']
        assert name in [selected,control]
    marker=study.OUT/'cases'/name/('evaluation_allocations.json' if later else 'validation_allocations.json')
    if marker.exists():return
    spec=next(s for s in study.CFG['specifications'] if s['name']==name)
    columns=list(dict.fromkeys(study.unit.bandit.EXECUTION+['secid']))
    roster=pd.read_parquet(study.OUT/'base_proposals.parquet',columns=columns,use_threads=False)
    keys=pd.MultiIndex.from_frame(roster[['date','secid']])
    assert keys.is_unique
    levels=np.full(len(roster),study.CFG['warmup_action'],int)
    meta,paths=study.core.load_labels();cal=study.unit.base.calendar()
    end=pd.Timestamp(study.CFG['evaluation_end'] if later else study.CFG['validation_end'])
    names=study.core.FEATURES[spec['features']]
    needed=list(dict.fromkeys(['date','secid']+names))
    for month in pd.period_range(roster.date.min(),min(end,roster.date.max()),freq='M'):
        mpath=study.OUT/'models'/name/f'{month}.json'
        if mpath.exists():model=json.loads(mpath.read_text())
        else:
            model=study.core.fit_model(meta,paths,month.start_time,spec,cal)
            study.core.dump(mpath,model)
        frame=pd.read_parquet(study.OUT/'proposals.parquet',columns=needed,
            filters=[('date','>=',month.start_time),('date','<',(month+1).start_time)],use_threads=False)
        if len(frame):scatter_levels(keys,frame,study.core.predict(model,frame),levels)
        assert model['last_completed'] is None or pd.Timestamp(model['last_completed'])<month.start_time
        if month.month in [1,7]:print(json.dumps(dict(case=name,month=str(month),completed_examples=model['training_samples'])),flush=True)
        del frame;gc.collect();pa.default_memory_pool().release_unused()
    roster['learned_level']=levels
    dest=study.OUT/'cases'/name/'allocations/tree18_test.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
    roster.to_parquet(dest,index=False)
    study.core.dump(marker,dict(complete=True,case=name,later=later,
        builder_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unchanged_proposals=len(roster),sizes=roster.learned_level.value_counts().to_dict()))
    print(json.dumps(dict(case=name,phase='allocations_complete',rows=len(roster))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=[s['name'] for s in study.CFG['specifications']])
    p.add_argument('--later',action='store_true');p.add_argument('--test',action='store_true');a=p.parse_args()
    study.configure()
    if a.test:
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_transfer_fit','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    if not a.case:p.error('--case required')
    study.check_sources()
    provenance=study.OUT/'fit_builder.json'
    record={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if provenance.exists():assert json.loads(provenance.read_text())==record
    else:study.core.dump(provenance,record)
    allocations(a.case,a.later)


if __name__=='__main__':main()
