"""CLI: python -m pixvqa_learned_install"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install learned-metric upstream assets for PixVQA")
    parser.add_argument(
        "--root",
        type=Path,
        default=_repo_root(),
        help="PixVQA repository root (default: parent of python/)",
    )
    parser.add_argument("--learned-python", required=True, help="Shared learned-metrics interpreter")
    parser.add_argument("--feature-python", required=True, help="HDR-ChipQA feature extraction interpreter")
    parser.add_argument("--inference-python", required=True, help="HDR-ChipQA SVR inference interpreter (py3.10)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download; only verify existing assets and update config",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip torch/joblib load probes",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="install",
        choices=("install", "validate"),
        help="install (default) or validate only",
    )
    args = parser.parse_args(argv)
    offline = args.offline or os.environ.get("PIXVQA_LEARNED_INSTALL_OFFLINE", "") not in ("", "0")

    # Ensure python/ is importable when invoked bare.
    py_dir = args.root / "python"
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))

    from pixvqa_learned_install.install import run_install
    from pixvqa_learned_install.validate import validate_readiness

    if args.command == "validate":
        ready, issues = validate_readiness(
            args.root.resolve(),
            learned_python=args.learned_python,
            feature_python=args.feature_python,
            inference_python=args.inference_python,
            skip_load=args.skip_load,
        )
        for issue in issues:
            print(issue, flush=True)
        return 0 if ready else 1

    return run_install(
        args.root.resolve(),
        learned_python=args.learned_python,
        feature_python=args.feature_python,
        inference_python=args.inference_python,
        offline=offline,
        skip_load=args.skip_load,
    )


if __name__ == "__main__":
    raise SystemExit(main())
