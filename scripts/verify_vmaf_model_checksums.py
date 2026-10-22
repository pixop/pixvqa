#!/usr/bin/env python3
"""Verify vendored model/*.json are byte-identical to Netflix/vmaf @ f85a853.

Downloads each upstream file and compares SHA-256 to the local copy.
Does not use digests embedded in PixVQA source as the sole authority.
"""
from __future__ import annotations

import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "f85a853692a8c730d0270cd733c8bb30b5b93b7c"
FILES = {
    "vmaf_v1.0.16_3d0h.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json",
    "vmaf_v1.0.16_5d0h.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json",
    "vmaf_v1.0.16_1d5h_2160.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json",
    "vmaf_v1.0.16_3d0h_2160.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json",
    "vmaf_v1.0.16_hfr_3d0h.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json",
    "vmaf_v1.0.16_hfr_5d0h.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_5d0h.json",
    "vmaf_v1.0.16_hfr_1d5h_2160.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json",
    "vmaf_v1.0.16_hfr_3d0h_2160.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h_2160.json",
}


def main() -> int:
    ctx = ssl.create_default_context()
    ok = True
    rows = []
    for local_name, remote_path in FILES.items():
        local = ROOT / "model" / local_name
        if not local.is_file():
            print(f"MISSING {local}", file=sys.stderr)
            ok = False
            continue
        local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
        url = f"https://raw.githubusercontent.com/Netflix/vmaf/{PIN}/{remote_path}"
        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
            remote = resp.read()
        remote_sha = hashlib.sha256(remote).hexdigest()
        match = local_sha == remote_sha and local.read_bytes() == remote
        status = "MATCH" if match else "MISMATCH"
        print(f"{status} {local_name}")
        print(f"  local  {local_sha}")
        print(f"  remote {remote_sha}")
        print(f"  url    {url}")
        rows.append((local_name, local_sha, remote_sha, match))
        if not match:
            ok = False
    report = ROOT / "reports" / "VMAF_MODEL_CHECKSUMS.md"
    lines = [
        "# VMAF v1.0.16 model SHA-256 (exact byte identity)",
        "",
        f"Upstream pin: Netflix/vmaf `{PIN}` (tag `v3.2.1`).",
        "",
        "Verified by `scripts/verify_vmaf_model_checksums.py` downloading each",
        "upstream file and comparing SHA-256 / bytes to `model/`.",
        "",
        f"Result: **{'exact match for all eight files' if ok else 'MISMATCH DETECTED'}**.",
        "",
        "| File | SHA-256 |",
        "|---|---|"]
    for name, lsha, _, match in rows:
        mark = "" if match else " **MISMATCH**"
        lines.append(f"| `{name}` | `{lsha}`{mark} |")
    lines.extend(
        [
            "",
            "PixVQA prints `sha256=` on the `VMAF:` metadata line using the digest of the",
            "**actual loaded file** (resolved path), not a preset name.",
            "",
            "Reproduce:",
            "",
            "```bash",
            "python3 scripts/verify_vmaf_model_checksums.py",
            "```",
            ""]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
