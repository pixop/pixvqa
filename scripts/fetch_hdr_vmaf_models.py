#!/usr/bin/env python3
"""Download the eight HDR VMAF v1.1.8 JSON files from the pinned commit.

Writes them under model/hdr_vmaf_v1.1.8 and model/hdr_vmaf_v1.1.8_hfr.
Does not touch the eight SDR v1.0.16 files. Prints SHA-256 and a short
diff of each HFR file against its standard twin.
"""
from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "74f930d17de8c58e7c2ce55e2b897b0030ebd8ef"
PAIRS = [
    (
        "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json",
        "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json",
    ),
    (
        "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json",
        "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json",
    ),
    (
        "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json",
        "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json",
    ),
    (
        "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json",
        "model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json",
    ),
]


def get(url: str) -> bytes:
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
        return resp.read()


def feature_opts(doc: dict) -> list:
    return doc.get("model_dict", {}).get("feature_opts_dicts", [])


def main() -> None:
    for std_rel, hfr_rel in PAIRS:
        for rel in (std_rel, hfr_rel):
            url = f"https://raw.githubusercontent.com/Netflix/vmaf/{PIN}/{rel}"
            data = get(url)
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            print(f"{digest}  {rel}  bytes={len(data)}")
        std = json.loads((ROOT / std_rel).read_text(encoding="utf-8"))
        hfr = json.loads((ROOT / hfr_rel).read_text(encoding="utf-8"))
        print(f"  clip std={std.get('model_dict', {}).get('score_clip')} "
              f"hfr={hfr.get('model_dict', {}).get('score_clip')}")
        print(f"  conversion std={std.get('model_dict', {}).get('conversion_target')}")
        so = feature_opts(std)
        ho = feature_opts(hfr)
        if so == ho:
            print("  feature opts identical")
        else:
            for i, (xa, xb) in enumerate(zip(so, ho)):
                if xa != xb:
                    print(f"  opts[{i}] std={xa}")
                    print(f"  opts[{i}] hfr={xb}")


if __name__ == "__main__":
    main()
