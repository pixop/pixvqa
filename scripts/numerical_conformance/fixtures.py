"""Build and verify per-family UHD fixture bundles."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import CACHE_ROOT, FRAMES, HEIGHT, WIDTH
from .ffmpeg_util import decode_to_yuv, ffprobe_color, raw_to_hevc_lossless, raw_to_hevc_lossy
from .scene import all_scene_frames
from .yuv_io import read_yuv420p10, rgb_frames_to_yuv, write_yuv420p10


@dataclass
class FamilyFixtures:
    family: str
    root: Path
    reference_yuv: Path
    distorted_yuv: Path
    reference_lossless: Path
    distorted_lossy: Path

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


def family_dir(family: str) -> Path:
    return CACHE_ROOT / "uhd60" / family


def build_family(family: str, ffmpeg: str, *, force: bool = False) -> FamilyFixtures:
    root = family_dir(family)
    root.mkdir(parents=True, exist_ok=True)
    fx = FamilyFixtures(
        family=family,
        root=root,
        reference_yuv=root / "reference.yuv",
        distorted_yuv=root / "distorted.yuv",
        reference_lossless=root / "reference_lossless.mkv",
        distorted_lossy=root / "distorted_lossy.mp4",
    )
    manifest = fx.manifest_path()
    if not force and manifest.is_file():
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        paths = stored.get("files", {})
        if stored.get("verified") and all(Path(paths[k]).is_file() for k in paths):
            return fx

    linear = all_scene_frames()
    write_yuv420p10(fx.reference_yuv, rgb_frames_to_yuv(linear, family))

    raw_to_hevc_lossy(ffmpeg, fx.reference_yuv, fx.distorted_lossy, family=family, qp=34)
    tmp_dist = root / "_dist_dec.raw"
    decode_to_yuv(ffmpeg, fx.distorted_lossy, tmp_dist)
    fx.distorted_yuv.write_bytes(tmp_dist.read_bytes())
    tmp_dist.unlink(missing_ok=True)

    raw_to_hevc_lossless(ffmpeg, fx.reference_yuv, fx.reference_lossless, family=family)
    tmp_ref = root / "_ref_dec.raw"
    decode_to_yuv(ffmpeg, fx.reference_lossless, tmp_ref)
    if tmp_ref.read_bytes() != fx.reference_yuv.read_bytes():
        raise RuntimeError(f"{family}: lossless decode != reference.yuv")
    tmp_ref.unlink(missing_ok=True)

    tmp_check = root / "_dist_check.raw"
    decode_to_yuv(ffmpeg, fx.distorted_lossy, tmp_check)
    if tmp_check.read_bytes() != fx.distorted_yuv.read_bytes():
        raise RuntimeError(f"{family}: decode(distorted_lossy) != distorted.yuv")
    tmp_check.unlink(missing_ok=True)

    files = {
        "reference_yuv": str(fx.reference_yuv),
        "distorted_yuv": str(fx.distorted_yuv),
        "reference_lossless": str(fx.reference_lossless),
        "distorted_lossy": str(fx.distorted_lossy),
    }
    meta = {
        "family": family,
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAMES,
        "verified": True,
        "files": files,
        "sha256": {k: sha256_file(Path(v)) for k, v in files.items()},
        "lossless_container": "mkv",
        "lossy_container": "mp4",
        "reference_lossless_probe": ffprobe_color(ffmpeg, fx.reference_lossless),
        "distorted_lossy_probe": ffprobe_color(ffmpeg, fx.distorted_lossy),
    }
    manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return fx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_all_families(ffmpeg: str, *, force: bool = False) -> dict[str, FamilyFixtures]:
    return {fam: build_family(fam, ffmpeg, force=force) for fam in ("sdr", "pq", "hlg")}
