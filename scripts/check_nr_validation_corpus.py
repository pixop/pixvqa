#!/usr/bin/env python3
"""Read-only validation of external NR video source paths and raw geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config/nr_validation_corpus.example.json"
LOCAL = ROOT / "config/nr_validation_corpus.local.json"
DEFAULT = LOCAL if LOCAL.is_file() else EXAMPLE


def validate(data: dict, *, check_files: bool) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    rows: list[dict] = []
    if data.get("schema_version") != 1 or not isinstance(data.get("sources"), list):
        return [], ["invalid schema_version or sources list"]
    seen: set[str] = set()
    for i, item in enumerate(data["sources"]):
        try:
            ident = item["id"]
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError("missing/duplicate id")
            seen.add(ident)
            if item["kind"] != "raw":
                raise ValueError("only raw fixtures are currently supported")
            width = item["width"]
            height = item["height"]
            depth = item["bit_depth"]
            chroma = item["chroma"]
            fps_num = item["fps_num"]
            fps_den = item["fps_den"]
            if any(type(v) is not int or v <= 0 for v in (width, height, depth, fps_num, fps_den)):
                raise ValueError("invalid geometry, depth, or fps")
            if chroma != "420" or width % 2 or height % 2 or depth not in (8, 10):
                raise ValueError("unsupported raw format")
            if item["transfer"] not in ("sdr", "pq", "hlg"):
                raise ValueError("unsupported transfer")
            path = Path(item["path"])
            if not path.is_absolute():
                raise ValueError("fixture path must be absolute")
            frame_bytes = width * height * 3 // 2 * (1 if depth == 8 else 2)
            row = {"id": ident, "transfer": item["transfer"], "path": str(path), "frame_bytes": frame_bytes}
            if check_files:
                if not path.is_file():
                    errors.append(f"{ident}: missing source {path}")
                    row["status"] = "MISSING"
                else:
                    size = path.stat().st_size
                    frames, remainder = divmod(size, frame_bytes)
                    row.update(frames=frames, remainder_bytes=remainder, duration_seconds=frames * fps_den / fps_num)
                    row["status"] = "PASS" if frames > 0 and remainder == 0 else "FAIL"
                    if row["status"] != "PASS":
                        errors.append(f"{ident}: incomplete frames or empty file")
            rows.append(row)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"source #{i}: {exc}")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--check-files", action="store_true", help="stat local paths (never read frame contents)")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()
    try:
        rows, errors = validate(json.loads(args.manifest.read_text(encoding="utf-8")), check_files=args.check_files)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"sources": rows, "errors": errors}, indent=2))
    else:
        for row in rows:
            print(row["id"], row.get("status", "SCHEMA"), row.get("frames", ""), row.get("duration_seconds", ""))
        for err in errors:
            print("FAIL", err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
