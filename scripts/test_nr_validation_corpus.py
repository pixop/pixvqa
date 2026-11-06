#!/usr/bin/env python3
"""No media, network, GPU, or model assets required."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from check_nr_validation_corpus import EXAMPLE, validate

source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
rows, errors = validate(source, check_files=False)
assert len(rows) == 6 and not errors, errors
assert {row["transfer"] for row in rows} == {"sdr", "pq", "hlg"}
assert len({row["id"] for row in rows}) == 6
with tempfile.TemporaryDirectory() as tmp:
    fixture = Path(tmp) / "small.yuv"
    fixture.write_bytes(bytes(4 * 4 * 3 // 2 * 2 * 3))
    unit = {"schema_version": 1, "sources": [{
        "id": "test", "kind": "raw", "path": str(fixture), "width": 4,
        "height": 4, "chroma": "420", "bit_depth": 10,
        "fps_num": 25, "fps_den": 1, "transfer": "sdr",
    }]}
    r, e = validate(unit, check_files=True)
    assert not e and r[0]["frames"] == 3 and r[0]["remainder_bytes"] == 0
    fixture.write_bytes(fixture.read_bytes() + b"x")
    r, e = validate(unit, check_files=True)
    assert e and r[0]["remainder_bytes"] == 1
    unit["sources"].append(dict(unit["sources"][0]))
    _, e = validate(unit, check_files=False)
    assert e and "duplicate" in e[0]
print("PASS: corpus schema, transfers, raw sizes, truncated inputs, duplicate IDs")
