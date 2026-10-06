"""Guarded entry point for the joint frozen-signal allocation experiment."""
import argparse
import unittest


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','train','backtest','finalize','all'],default='all');a=p.parse_args()
    if a.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_joint_smile*.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    from scripts.joint_smile_research import run
    run(a.phase)


if __name__=='__main__':main()
