"""Resume preparation while the bounded download runs, then execute the experiment."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('data/call_put');jobs=json.loads((root/'requests.json').read_text());last=-1
    while True:
        completed=sum((root/j['path']).with_suffix('.json').exists() for j in jobs)
        print(f'Put download: {completed}/{len(jobs)} partitions',flush=True)
        if completed==len(jobs) or completed>=last+20:
            subprocess.run([sys.executable,'-m','scripts.prepare_matched_puts'],check=True);last=completed
        if completed==len(jobs):break
        time.sleep(30)
    # First bounded validation run, then resume the unchanged full roster.
    subprocess.run([sys.executable,'-m','scripts.run_option_flex','--limit','3'],check=True)
    subprocess.run([sys.executable,'-m','scripts.run_option_flex'],check=True)
    subprocess.run([sys.executable,'-m','scripts.report_option_flex'],check=True)


if __name__=='__main__':main()
