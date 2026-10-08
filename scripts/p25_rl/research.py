"""Run the bounded IV/RV reinforcement-learning study serially.

Use the pinned Python environment:
    python -m scripts.p25_rl.research --phase all

Individual validation, evaluation and report phases resume existing verified
artifacts. Evaluation verifies the already frozen combined selection; it never
reselects a model. This parent imports only the standard library and starts one
child at a time, preserving the children's numerical resource controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / 'reports/p25_rl_research/selection.json'
SELECTION_SOURCES = {
    f'reports/{family}/{filename}'
    for family in ['p25_inventory_q', 'p25_action_q', 'p25_inventory_q_management',
                   'p25_temporal_q']
    for filename in ['protocol.json', 'validation.csv']
}
DECLARED_CANDIDATES = {
    'q_capacity', 'q_risk', 'q_conservative', 'q_portfolio',
    'action_q_capacity', 'action_q_risk', 'action_q_conservative',
    'manage_q', 'manage_q_risk',
}
STEPS = {
    'validation': [
        ('scripts.p25_rl.run', 'baseline', 'Verify original threshold accounting'),
        ('scripts.p25_rl.run', 'behavior', 'Generate entry allocation histories'),
        ('scripts.p25_rl.run', 'validation', 'Validate shared entry Q policies'),
        ('scripts.p25_rl.action_q', 'validation', 'Validate separate action Q policies'),
        ('scripts.p25_rl.management', 'behavior', 'Generate held-book management histories'),
        ('scripts.p25_rl.management', 'validation', 'Validate hold, trim and exit Q policies'),
        ('scripts.p25_rl.temporal', 'validation', 'Validate matched immediate-reward controls'),
        ('scripts.p25_rl.summary', 'select', 'Freeze combined earlier-period selection'),
    ],
    'evaluation': [
        ('scripts.p25_rl.run', 'evaluation', 'Evaluate shared entry Q policies'),
        ('scripts.p25_rl.action_q', 'evaluation', 'Evaluate separate action Q policies'),
        ('scripts.p25_rl.management', 'evaluation', 'Evaluate held-book management Q policies'),
        ('scripts.p25_rl.temporal', 'evaluation', 'Evaluate matched immediate-reward controls'),
    ],
    'report': [
        ('scripts.p25_rl.report', None, 'Verify and analyze entry allocation results'),
        ('scripts.p25_rl.summary', 'report', 'Verify all families and write concise findings'),
    ],
}


def _digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_frozen_selection():
    """Check the frozen identity and hashes without importing numeric modules.

    Scientific selection is reproduced independently in the child verifiers.
    This guard prevents an absent, stale or changed selection from authorizing
    any later-history evaluation step.
    """
    if not SELECTION.exists():
        raise ValueError('Combined selection is missing; run --phase validation first.')
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    if selection.get('validation_only') is not True:
        raise ValueError('Combined selection must use earlier validation only.')
    if (selection.get('validation_start') != '2020-01-01'
            or selection.get('validation_end') != '2021-12-31'
            or selection.get('proposal_cutoff') != '2021-10-31'):
        raise ValueError('Combined selection has a different validation window.')
    sources = selection.get('source_hashes')
    normalized_sources = ({name.replace('\\', '/') for name in sources}
                          if isinstance(sources, dict) else set())
    if not isinstance(sources, dict) or normalized_sources != SELECTION_SOURCES:
        raise ValueError('Combined selection does not identify all four validation families.')
    candidates = selection.get('candidates')
    if (not isinstance(candidates, list) or not candidates
            or any(not isinstance(case, str) for case in candidates)
            or len(candidates) != len(set(candidates))
            or set(candidates) != DECLARED_CANDIDATES
            or selection.get('selected') not in set(candidates) | {'fixed100'}
            or selection.get('best_q_by_validation') not in candidates):
        raise ValueError('Combined selection candidate identity is invalid.')
    for name, digest in sources.items():
        if _digest(ROOT / name.replace('\\', '/')) != digest:
            raise ValueError(f'Combined validation input changed: {name}')
    return selection


def run_phase(phase):
    if phase in ['evaluation', 'report']:
        check_frozen_selection()
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    for index, (module, child_phase, description) in enumerate(STEPS[phase], start=1):
        # Recheck before every later-history child. No selection file is written
        # here, so evaluation cannot silently revise the validation decision.
        if phase == 'evaluation':
            check_frozen_selection()
        print(f'{phase.capitalize()} {index}/{len(STEPS[phase])}: {description}', flush=True)
        command = [sys.executable, '-m', module]
        if child_phase is not None:
            command.extend(['--phase', child_phase])
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validation', 'evaluation', 'report', 'all'],
                        default='all')
    args = parser.parse_args(argv)
    phases = ['validation', 'evaluation', 'report'] if args.phase == 'all' else [args.phase]
    try:
        for phase in phases:
            run_phase(phase)
    except subprocess.CalledProcessError as error:
        print('Research stopped because a child stage failed. Completed verified stages remain resumable.',
              file=sys.stderr, flush=True)
        return error.returncode
    except (ValueError, KeyError, OSError) as error:
        print(f'Research stopped: {error}', file=sys.stderr, flush=True)
        return 2
    if args.phase == 'validation':
        print('Validation complete; combined selection is frozen. Next: --phase evaluation.', flush=True)
    elif args.phase == 'evaluation':
        print('Evaluation complete. Next: --phase report.', flush=True)
    else:
        print('Research complete. Findings: reports/p25_rl_research/Findings.txt', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
