"""Build and verify per-family UHD fixture bundles."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import CACHE_ROOT, FRAMES, FRAMES_NR, HEIGHT, WIDTH
from .ffmpeg_util import decode_to_yuv, ffprobe_color, raw_to_hevc_lossless, raw_to_hevc_lossy
from .scene import all_scene_frames
from .yuv_io import rgb_frames_to_yuv, write_yuv420p10


@dataclass
class FamilyFixtures:
    family: str
    root: Path
    reference_yuv: Path
    distorted_yuv: Path
    reference_lossless: Path
    distorted_lossy: Path
    reference_yuv_60: Path
    reference_lossless_60: Path
    distorted_lossy_60: Path

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


def family_dir(family: str) -> Path:
    return CACHE_ROOT / "uhd60" / family


def _build_yuv_bundle(
    ffmpeg: str,
    family: str,
    root: Path,
    *,
    frame_count: int,
    reference_yuv: Path,
    distorted_yuv: Path,
    reference_lossless: Path,
    distorted_lossy: Path,
) -> None:
    linear = all_scene_frames(frame_count)
    write_yuv420p10(reference_yuv, rgb_frames_to_yuv(linear, family))
    raw_to_hevc_lossy(ffmpeg, reference_yuv, distorted_lossy, family=family, qp=34)
    tmp_dist = root / f"_dist_dec_{frame_count}.raw"
    decode_to_yuv(ffmpeg, distorted_lossy, tmp_dist)
    distorted_yuv.write_bytes(tmp_dist.read_bytes())
    tmp_dist.unlink(missing_ok=True)
    raw_to_hevc_lossless(ffmpeg, reference_yuv, reference_lossless, family=family)
    tmp_ref = root / f"_ref_dec_{frame_count}.raw"
    decode_to_yuv(ffmpeg, reference_lossless, tmp_ref)
    if tmp_ref.read_bytes() != reference_yuv.read_bytes():
        raise RuntimeError(f"{family}: lossless decode != {reference_yuv.name}")
    tmp_ref.unlink(missing_ok=True)
    tmp_check = root / f"_dist_check_{frame_count}.raw"
    decode_to_yuv(ffmpeg, distorted_lossy, tmp_check)
    if tmp_check.read_bytes() != distorted_yuv.read_bytes():
        raise RuntimeError(f"{family}: decode({distorted_lossy.name}) != {distorted_yuv.name}")
    tmp_check.unlink(missing_ok=True)


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
        reference_yuv_60=root / "reference_60f.yuv",
        reference_lossless_60=root / "reference_lossless_60f.mkv",
        distorted_lossy_60=root / "distorted_lossy_60f.mp4",
    )
    manifest = fx.manifest_path()
    required_keys = (
        "reference_yuv",
        "distorted_yuv",
        "reference_lossless",
        "distorted_lossy",
        "reference_yuv_60",
        "reference_lossless_60",
        "distorted_lossy_60",
    )
    if not force and manifest.is_file():
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        paths = stored.get("files", {})
        if stored.get("verified") and all(Path(paths[k]).is_file() for k in required_keys):
            return fx

    _build_yuv_bundle(
        ffmpeg,
        family,
        root,
        frame_count=FRAMES,
        reference_yuv=fx.reference_yuv,
        distorted_yuv=fx.distorted_yuv,
        reference_lossless=fx.reference_lossless,
        distorted_lossy=fx.distorted_lossy,
    )
    _build_yuv_bundle(
        ffmpeg,
        family,
        root,
        frame_count=FRAMES_NR,
        reference_yuv=fx.reference_yuv_60,
        distorted_yuv=root / "distorted_60f.yuv",
        reference_lossless=fx.reference_lossless_60,
        distorted_lossy=fx.distorted_lossy_60,
    )

    files = {k: str(getattr(fx, k)) for k in required_keys}
    meta = {
        "family": family,
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAMES,
        "frames_nr": FRAMES_NR,
        "verified": True,
        "files": files,
        "sha256": {k: sha256_file(Path(v)) for k, v in files.items()},
        "lossless_container": "mkv",
        "lossy_container": "mp4",
        "reference_lossless_probe": ffprobe_color(ffmpeg, fx.reference_lossless),
        "distorted_lossy_probe": ffprobe_color(ffmpeg, fx.distorted_lossy),
        "distorted_lossy_60_probe": ffprobe_color(ffmpeg, fx.distorted_lossy_60),
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
