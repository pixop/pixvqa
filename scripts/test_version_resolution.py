#!/usr/bin/env python3
"""Smoke-test the Git-derived build version without touching the real repository."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def version(root: Path, override: str | None = None) -> str:
    env = dict(os.environ)
    env.pop('PIXVQA_VERSION_OVERRIDE', None)
    if override is not None:
        env['PIXVQA_VERSION_OVERRIDE'] = override
    return run('bash', 'scripts/version.sh', cwd=root, env=env)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='pixvqa-version-') as tmp:
        root = Path(tmp) / 'test'
        (root / 'scripts').mkdir(parents=True)
        shutil.copy2(ROOT / 'scripts/version.sh', root / 'scripts/version.sh')
        assert version(root) == '0.3.0-dev+unknown'
        assert version(root, '0.3.0-rc1') == '0.3.0-rc1'
        run('git', 'init', '-q', cwd=root)
        run('git', 'config', 'user.name', 'PixVQA Test', cwd=root)
        run('git', 'config', 'user.email', 'test@example.invalid', cwd=root)
        run('git', 'add', 'scripts/version.sh', cwd=root)
        run('git', 'commit', '-qm', 'test base', cwd=root)
        sha = run('git', 'rev-parse', '--short=10', 'HEAD', cwd=root)
        assert version(root) == f'0.3.0-dev+{sha}'
        run('git', 'tag', '-a', 'v0.3.0-rc1', '-m', 'rc1', cwd=root)
        assert version(root) == '0.3.0-rc1'
        run('git', 'tag', '-a', 'v0.3.0', '-m', 'stable', cwd=root)
        assert version(root) in ('0.3.0', '0.3.0-rc1')  # both tags point to HEAD
        run('git', 'commit', '--allow-empty', '-qm', 'development', cwd=root)
        sha = run('git', 'rev-parse', '--short=10', 'HEAD', cwd=root)
        assert version(root) == f'0.3.0-dev+{sha}'
        env = dict(os.environ, PIXVQA_VERSION_OVERRIDE='bad\"version')
        proc = subprocess.run(['bash', 'scripts/version.sh'], cwd=root, env=env, capture_output=True, text=True)
        assert proc.returncode == 2
        # A Git worktree in a parent directory must not be mistaken for this archive.
        archive = root / 'untracked-archive'
        (archive / 'scripts').mkdir(parents=True)
        shutil.copy2(ROOT / 'scripts/version.sh', archive / 'scripts/version.sh')
        assert version(archive) == '0.3.0-dev+unknown'
    print('PASS: archive, override, tagged RC/stable, dev SHA, invalid override, nested archive')


if __name__ == '__main__':
    main()
