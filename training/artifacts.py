"""Shared locations for experiment evidence and the canonical findings record."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'out/experiments'
HISTORY = ROOT / 'EXPERIMENT_HISTORY.md'


def unused_run_path(parent, description):
    """Choose a descriptive path; callers still create it exclusively."""
    name = re.sub(r'[^a-z0-9.-]+', '-', description.lower()).strip('-.') or 'experiment'
    parent = Path(parent)
    candidate = parent / name
    repeat = 2
    while candidate.exists():
        candidate = parent / f'{name}-repeat{repeat}'
        repeat += 1
    return candidate


def append_findings(section):
    """Keep new generated findings in the SSOT instead of per-run Markdown reports."""
    with HISTORY.open('a') as output:
        output.write('\n\n' + section.rstrip() + '\n')
