#!/usr/bin/env python3
"""UHD SDR/PQ/HLG numerical conformance (real metrics, auto fixtures)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.baselines import approve_candidate  # noqa: E402
from numerical_conformance.runner import run_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PixVQA numerical SDR/PQ/HLG conformance",
        epilog=(
            "Compressed scenarios require a decoder-enabled binary "
            "(build/swdec or build/full). Set PIXVQA_BIN; when it points at "
            "build/cpu the suite auto-falls back to build/swdec if present."
        ),
    )
    parser.add_argument("--profile", choices=("smoke", "comprehensive"), default="smoke")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--case-id", action="append", default=[],
                        help="Run one specified numerical case; repeatable.")
    parser.add_argument("--json", type=Path, default=ROOT / "reports" / "numerical_conformance.json")
    parser.add_argument("--force-fixtures", action="store_true")
    parser.add_argument(
        "--compare-candidate",
        type=Path,
        default=None,
        help="Read-only compare run results to an unapproved candidate JSON (does not approve baseline).",
    )
    parser.add_argument("--approve-baseline", action="store_true")
    args = parser.parse_args()
    if args.approve_baseline:
        path = approve_candidate(args.profile)
        print(f"Approved baseline: {path}")
        return 0
    import os

    strict = args.strict or os.environ.get("PIXVQA_NUMERICAL_STRICT", "") in ("1", "true", "yes")
    if args.case_id and args.json == ROOT / "reports" / "numerical_conformance.json":
        args.json = ROOT / "reports" / "numerical_conformance_partial.json"
    return run_suite(
        profile=args.profile,
        strict=strict,
        json_report=args.json,
        force_fixtures=args.force_fixtures,
        compare_candidate=args.compare_candidate,
        case_ids=tuple(args.case_id) if args.case_id else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
