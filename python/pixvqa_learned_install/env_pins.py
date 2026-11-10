"""Verify a reused learned-metric venv matches python/requirements/learned-env.txt pins."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

# ColorVideoVDP is installed from a pinned export (--no-deps); the commit stamp is authoritative.
CVVDP_SKIP = frozenset({"cvvdp", "colorvideovdp", "pycvvdp"})

# Torch 2.13+cu130 wheels install the CUDA stack as separate distributions with pip
# metadata. Older torch builds (for example cu128) bundle the same libraries inside the
# torch wheel without per-package dist-info. When the lock torch pin does not match the
# installed interpreter, skip checks on those bundled-only names to avoid false
# "not installed" noise; torch/torchvision pins still reject the environment.
TORCH_STACK_PREFIXES = ("nvidia-", "cuda-")


def parse_lock_pins(lock_path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def normalize_dist_name(dist: str) -> str:
    return dist.strip().lower().replace("_", "-")


def installed_version(dist: str) -> str | None:
    canonical = normalize_dist_name(dist)
    try:
        return metadata.version(canonical)
    except metadata.PackageNotFoundError:
        pass
    # Lock pins use PyPI names; some wheels register alternate metadata names.
    for installed in metadata.distributions():
        name = installed.metadata.get("Name") or ""
        if normalize_dist_name(name) == canonical:
            return installed.version
    return None


def is_torch_stack_package(dist: str) -> bool:
    return dist == "triton" or dist.startswith(TORCH_STACK_PREFIXES)


def torch_matches_lock(pins: dict[str, str]) -> bool:
    expected = pins.get("torch")
    if expected is None:
        return False
    actual = installed_version("torch")
    return actual is not None and actual == expected


def verify_python() -> str | None:
    if sys.version_info[:2] != (3, 12):
        major, minor = sys.version_info[:2]
        return f"interpreter is Python {major}.{minor}, required 3.12"
    return None


def verify_pins(pins: dict[str, str]) -> list[str]:
    errors: list[str] = []
    verify_torch_stack = torch_matches_lock(pins)
    for dist, expected in sorted(pins.items()):
        if dist in CVVDP_SKIP:
            continue
        if is_torch_stack_package(dist) and not verify_torch_stack:
            continue
        actual = installed_version(dist)
        if actual is None:
            errors.append(f"{dist}: not installed (expected {expected})")
        elif actual != expected:
            errors.append(f"{dist}: installed {actual}, expected {expected}")
    return errors


def migration_hint(env_dir: str = "") -> str:
    lines = [
        "Learned setup: the existing environment does not match pinned versions.",
        "Do not upgrade or reinstall packages in-place.",
        "Preserve the current tree and create a new environment instead, for example:",
        "  export PIXVQA_LEARNED_ENV_DIR=/absolute/path/to/new-learned-venv",
        "  bash scripts/install_learned_metrics.sh",
    ]
    if env_dir:
        lines.insert(1, f"  Rejected environment: {env_dir}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify learned env matches learned-env.txt (fail closed on reuse)",
    )
    parser.add_argument("lockfile", type=Path)
    parser.add_argument("--env-dir", default="", help="Rejected venv path for error text")
    args = parser.parse_args(argv)

    py_err = verify_python()
    if py_err:
        print(f"Learned env pins: {py_err}", file=sys.stderr)
        print(migration_hint(args.env_dir), file=sys.stderr)
        return 2

    if not args.lockfile.is_file():
        print(f"Learned env pins: lockfile missing: {args.lockfile}", file=sys.stderr)
        return 2

    errors = verify_pins(parse_lock_pins(args.lockfile))
    if errors:
        print("Learned env pins: version mismatches:", file=sys.stderr)
        for entry in errors:
            print(f"  - {entry}", file=sys.stderr)
        print(migration_hint(args.env_dir), file=sys.stderr)
        return 2

    print("Learned env pins: lockfile package versions match", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
