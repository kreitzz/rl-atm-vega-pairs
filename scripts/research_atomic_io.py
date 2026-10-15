"""Atomic publication of this study's restart markers and JSON model files.

Only filesystem publication changes. Reward data, model fitting, policies and
resource limits are unaffected. A killed writer can leave a pending file but
cannot publish that partial file as a completed checkpoint.
"""
import os
from pathlib import Path


def pending_path(path):
    return path.with_name('.'+path.name+f'.pending-{os.getpid()}')


def atomic_dump(value,path,original,*args,**kwargs):
    path=Path(path);pending=pending_path(path)
    files=original(value,pending,*args,**kwargs)
    if len(files)!=1:raise RuntimeError('Atomic checkpoint requires a single-file joblib format')
    os.replace(pending,path)
    return [str(path)]


def atomic_text(path,text,original,*args,**kwargs):
    path=Path(path);pending=pending_path(path)
    result=original(pending,text,*args,**kwargs)
    os.replace(pending,path)
    return result


def install():
    import joblib
    if getattr(joblib.dump,'_research_atomic',False):return
    roots=[(Path.cwd()/p).resolve() for p in [
        'artifacts/contracts/straddle_overnight_study','reports/straddle_overnight_study',
        'artifacts/contracts/straddle_context_followup','reports/straddle_context_followup',
        'artifacts/contracts/straddle_performance_followup','reports/straddle_performance_followup',
        'artifacts/contracts/straddle_ensemble_search','reports/straddle_ensemble_search',
        'artifacts/contracts/straddle_policy_refinement','reports/straddle_policy_refinement',
        'artifacts/contracts/put_straddle_q_frontier','reports/put_straddle_q_frontier',
        'artifacts/contracts/put_straddle_q_frontier_regularized','reports/put_straddle_q_frontier_regularized']]
    original_dump,original_text=joblib.dump,Path.write_text
    def owned(path):return any(Path(path).resolve().is_relative_to(root) for root in roots)
    def dump(value,filename,*args,**kwargs):
        if isinstance(filename,(str,os.PathLike)) and Path(filename).name=='checkpoint.joblib' and owned(filename):
            return atomic_dump(value,filename,original_dump,*args,**kwargs)
        return original_dump(value,filename,*args,**kwargs)
    def write(path,text,*args,**kwargs):
        if path.suffix=='.json' and owned(path):return atomic_text(path,text,original_text,*args,**kwargs)
        return original_text(path,text,*args,**kwargs)
    dump._research_atomic=True;joblib.dump=dump;Path.write_text=write
