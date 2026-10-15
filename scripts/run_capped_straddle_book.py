"""Replay the native 50/75/100 ladder without the legacy scale-hook menu.

The legacy capital-scale hook accepts 0.5/1/2/4 only. The simulator's native
policy already supports 25% increments, including 75%; use that API and keep
all inventory/risk budgets unchanged. Existing experiment sources stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts import straddle_capped_books as books


def native_ladder(simulate,*args,**kwargs):
    kwargs['policy']=lambda row,context:int(row['learned_level'])
    kwargs['scale_policy']=None
    return simulate(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['fixed75','entry_bandit']);parser.add_argument('--months',type=int,default=8,choices=range(1,9))
    parser.add_argument('--test',action='store_true');args=parser.parse_args()
    if args.test:
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_native_straddle_ladder','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    if not args.case:parser.error('--case required')
    books.configure();books.check_sources()
    record=dict(code=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        change='Native sizing policy replaces scale hook; risk budget scaling remains false.')
    marker=books.OUT.parent/'native_ladder_driver.json'
    if marker.exists():assert json.loads(marker.read_text())==record
    else:books.original.base.dump(marker,record)
    original=books.original.engine.simulate
    def simulate(*a,**kw):return native_ladder(original,*a,**kw)
    books.original.engine.simulate=simulate
    try:books.book(args.case,args.months)
    finally:books.original.engine.simulate=original


if __name__=='__main__':main()
