#!/usr/bin/env python3
"""Verify vendored model JSON files are byte-identical to Netflix/vmaf.

SDR files are compared to tag v3.2.1. HDR files are compared to commit 74f930d.
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
SDR_PIN = "f85a853692a8c730d0270cd733c8bb30b5b93b7c"
HDR_PIN = "74f930d17de8c58e7c2ce55e2b897b0030ebd8ef"
GROUPS = {
    SDR_PIN: {
        "vmaf_v1.0.16_3d0h.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json",
        "vmaf_v1.0.16_5d0h.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json",
        "vmaf_v1.0.16_1d5h_2160.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json",
        "vmaf_v1.0.16_3d0h_2160.json": "model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json",
        "vmaf_v1.0.16_hfr_3d0h.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json",
        "vmaf_v1.0.16_hfr_5d0h.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_5d0h.json",
        "vmaf_v1.0.16_hfr_1d5h_2160.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json",
        "vmaf_v1.0.16_hfr_3d0h_2160.json": "model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h_2160.json",
    },
    HDR_PIN: {
        "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json": "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json",
        "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json": "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json",
        "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json": "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json",
        "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json": "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json",
        "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json": "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json",
        "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json": "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json",
        "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json": "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json",
        "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json": "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json",
    },
}


def main() -> int:
    ctx = ssl.create_default_context()
    ok = True
    rows = []
    for pin, files in GROUPS.items():
        for local_name, remote_path in files.items():
            local = ROOT / "model" / local_name
            if not local.is_file():
                print(f"MISSING {local}", file=sys.stderr)
                ok = False
                continue
            local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
            url = f"https://raw.githubusercontent.com/Netflix/vmaf/{pin}/{remote_path}"
            with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
                remote = resp.read()
            remote_sha = hashlib.sha256(remote).hexdigest()
            match = local_sha == remote_sha and local.read_bytes() == remote
            status = "MATCH" if match else "MISMATCH"
            print(f"{status} {local_name}")
            print(f"  local  {local_sha}")
            print(f"  remote {remote_sha}")
            print(f"  url    {url}")
            rows.append((pin, local_name, local_sha, match))
            if not match:
                ok = False
    report = ROOT / "reports" / "VMAF_MODEL_CHECKSUMS.md"
    lines = [
        "# VMAF model SHA-256 (exact byte identity)",
        "",
        f"SDR upstream: Netflix/vmaf `{SDR_PIN}` (tag `v3.2.1`).",
        f"HDR upstream: Netflix/vmaf `{HDR_PIN}` (not a release tag).",
        "",
        "Verified by `scripts/verify_vmaf_model_checksums.py` downloading each",
        "upstream file and comparing SHA-256 / bytes to `model/`.",
        "",
        f"Result: **{'exact match for all sixteen files' if ok else 'MISMATCH DETECTED'}**.",
        "",
        "| Pin | File | SHA-256 |",
        "|---|---|---|"]
    for pin, name, lsha, match in rows:
        mark = "" if match else " **MISMATCH**"
        short = pin[:7]
        lines.append(f"| `{short}` | `{name}` | `{lsha}`{mark} |")
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
