#!/usr/bin/env python3
"""Locate first differing SimpleVQA tensor stage across independent trace files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_trace(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    stages = [row["stage"] for row in rows]
    if len(stages) != len(set(stages)):
        raise ValueError(f"duplicate stages in {path}")
    if not rows or stages[-1] != "score":
        raise ValueError(f"incomplete SimpleVQA trace in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="PIXVQA_SIMPLEVQA_TRACE_DIR")
    args = parser.parse_args()
    paths = sorted(args.directory.glob("simplevqa-*.jsonl"), key=lambda p: p.stat().st_mtime_ns)
    if len(paths) < 2:
        parser.error("need at least two SimpleVQA trace files")
    records = [read_trace(path) for path in paths]
    reference = records[0]
    stage_order = [item["stage"] for item in reference]
    print("traces:")
    for path, rows in zip(paths, records):
        print(f"  {path.name}: {len(rows)} stages, deterministic={rows[0].get('deterministic')}")
        if [item["stage"] for item in rows] != stage_order:
            raise ValueError(f"stage order/count mismatch in {path}")
    first = None
    for index, base in enumerate(reference):
        hashes = [rows[index]["sha256"] for rows in records]
        if len(set(hashes)) > 1:
            first = base["stage"]
            print(f"FIRST DIFFERENCE: {first}")
            print("  hashes:", " ".join(hash_value[:16] for hash_value in hashes))
            break
    if first is None:
        print("PASS: all traced inputs, features, and scores are byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
