"""Rebuild saved-result paper artifacts, one memory-guarded worker at a time.

Run with the project research interpreter. No training or data downloads occur.
Stages are separate so figures can be inspected before page rendering.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['figures','pages','finalize'],required=True)
    parser.add_argument('--pages',type=int,nargs='+')
    parser.add_argument('--figures',nargs='+')
    args=parser.parse_args()
    destination=Path('paper/unified_learning_2026_10_16')
    meta=json.loads((destination/'provenance.json').read_text())
    if args.stage=='figures':
        names=args.figures or meta['figures']
        assert all(name in meta['figures'] for name in names)
        jobs=[['--phase','figure','--figure',name] for name in names]
    elif args.stage=='pages':
        indices=args.pages or range(1,meta['main_pages']+meta['appendix_pages']+1)
        jobs=[['--phase','render','--page',str(n)] for n in indices]
    else:
        jobs=[['--phase','assemble'],['--phase','docx']]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
             MPLCONFIGDIR='/tmp/options_paper_figure_revision_mpl')
    overflows=[]
    for job in jobs:
        print('Building: '+' '.join(job),flush=True)
        command=[sys.executable,'scripts/run_signal_discovery_guarded.py',
                 '--module','scripts.build_unified_options_paper',
                 '--timeout','180','--duty-cycle','0.2','--rss-limit-mib','256',
                 '--warning-rss-limit-mib','256','--min-free-percent','30',
                 '--warning-min-free-percent','30','--max-swap-growth-mib','256',
                 '--pause-low-headroom-seconds','60',
                 '--stop-file','/tmp/options_paper_revision_STOP','--']+job
        result=subprocess.run(command,env=env)
        result.check_returncode()
        if args.stage=='pages':
            # Review all layout failures; resource or other errors stop above.
            layout=destination/'rendered_pages'/f'layout_{int(job[-1]):02d}.json' if args.stage=='pages' else None
            if layout and layout.exists() and json.loads(layout.read_text())['overflow']:
                overflows.append(int(job[-1]))
    if overflows:
        raise SystemExit('Pages requiring layout edits: '+str(overflows))


if __name__=='__main__':main()
