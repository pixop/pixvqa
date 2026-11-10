#!/usr/bin/env python3
"""Compare CVQA-HDR-NR sampled outputs under two isolated Python environments.

No package installation, no model modification. Both subprocesses use the same
source tree and checked model/video inputs; detailed JSON persists in --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(raw: str, config: Path) -> Path:
    path = Path(os.path.expanduser(raw))
    return (path if path.is_absolute() else config.parent / path).resolve()


def compare(a: dict, b: dict, tolerance: float) -> dict:
    assert a['metric'] == b['metric'] == 'cvqa-hdr-nr'
    fields = ['checkpoint', 'calibration_profile']
    same_assets = all(a[field]['sha256'] == b[field]['sha256'] for field in fields)
    sa, sb = a['temporal']['samples'], b['temporal']['samples']
    aligned = len(sa) == len(sb) and all(
        (x['frame_index'], x['time_seconds']) == (y['frame_index'], y['time_seconds'])
        for x, y in zip(sa, sb)
    )
    samples = []
    if aligned:
        for x, y in zip(sa, sb):
            samples.append({'frame': x['frame_index'], 'time': x['time_seconds'],
                            'raw_delta': y['raw_model_score'] - x['raw_model_score'],
                            'calibrated_delta': y['quality_score'] - x['quality_score']})
    delta = b['quality_score'] - a['quality_score']
    return {'assets_match': same_assets, 'schedule_matches': aligned,
            'samples': len(sa), 'raw_delta': b['raw_model_score'] - a['raw_model_score'],
            'calibrated_delta': delta,
            'max_abs_sample_raw_delta': max((abs(x['raw_delta']) for x in samples), default=None),
            'max_abs_sample_calibrated_delta': max((abs(x['calibrated_delta']) for x in samples), default=None),
            'per_sample': samples,
            'pass': same_assets and aligned and abs(delta) <= tolerance}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--baseline-python', type=Path, required=True)
    p.add_argument('--candidate-python', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=1e-4)
    args = p.parse_args()
    config = args.config.expanduser().resolve()
    c = tomllib.loads(config.read_text(encoding='utf-8'))['cvqa-hdr-nr']
    assets = {key: resolve(c[key], config) for key in ('repository','checkpoint','profile')}
    video = args.video.expanduser().resolve()
    for name, path in {**assets, 'video': video}.items():
        if not path.exists():
            p.error(f'{name} does not exist: {path}')
    args.out.mkdir(parents=True, exist_ok=True)
    for label, py in (('cu128',args.baseline_python), ('cu130',args.candidate_python)):
        out_json = (args.out / f'{label}.json').resolve()
        command = [str(py.expanduser().resolve()), '-m', 'pixvqa_cvqahdr',
                   '--repo', str(assets['repository']), '--checkpoint', str(assets['checkpoint']),
                   '--profile', str(assets['profile']), '--test', str(video),
                   '--device', 'cuda', '--output-json', str(out_json)]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'python') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        proc = subprocess.run(command, env=env, text=True, capture_output=True)
        (args.out / f'{label}.stdout').write_text(proc.stdout)
        (args.out / f'{label}.stderr').write_text(proc.stderr)
        print(label, 'exit', proc.returncode, flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-1500:], file=sys.stderr)
            return 1
    a = json.loads((args.out / 'cu128.json').read_text())
    b = json.loads((args.out / 'cu130.json').read_text())
    result = compare(a, b, args.tolerance)
    result['video_sha256'] = digest(video)
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Score delta:', result['calibrated_delta'])
    print('Raw delta:', result['raw_delta'])
    print('Max raw sample delta:', result['max_abs_sample_raw_delta'])
    print('Assets identical:', result['assets_match'])
    print('Schedule aligned:', result['schedule_matches'])
    print('Tolerance:', args.tolerance)
    print('VERDICT:', 'PASS' if result['pass'] else 'REVIEW')
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
