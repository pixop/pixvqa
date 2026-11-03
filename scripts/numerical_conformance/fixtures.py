"""Build and verify per-family UHD fixture bundles."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CACHE_ROOT, FRAMES, FRAMES_NR, HEIGHT, WIDTH
from .ffmpeg_util import (
    decode_to_yuv,
    ffprobe_color,
    ffprobe_stream,
    raw_to_hevc_lossless,
    raw_to_hevc_lossy,
)
from .yuv_io import files_identical, raw_yuv_frame_count, stream_reference_yuv

MANIFEST_VERSION = 2
CORE_KEYS = ("reference_yuv", "distorted_yuv", "reference_lossless", "distorted_lossy")
NR_KEYS = ("reference_yuv_60", "reference_lossless_60", "distorted_lossy_60")
ALL_KEYS = CORE_KEYS + NR_KEYS


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

    def path_map(self) -> dict[str, Path]:
        return {k: getattr(self, k) for k in ALL_KEYS}


def family_dir(family: str) -> Path:
    return CACHE_ROOT / "uhd60" / family


def _lock_path(root: Path) -> Path:
    return root / ".generating"


def _acquire_lock(root: Path) -> None:
    lock = _lock_path(root)
    if lock.is_file():
        age = time.time() - lock.stat().st_mtime
        if age > 3600:
            lock.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"fixture generation already in progress ({lock})")
    lock.write_text(str(os.getpid()), encoding="utf-8")


def _release_lock(root: Path) -> None:
    _lock_path(root).unlink(missing_ok=True)


def _default_files(fx: FamilyFixtures) -> dict[str, str]:
    return {k: str(getattr(fx, k)) for k in ALL_KEYS}


def _merge_manifest_files(stored: dict[str, Any], fx: FamilyFixtures) -> dict[str, str]:
    files = _default_files(fx)
    files.update(stored.get("files") or {})
    return files


def _paths_exist(files: dict[str, str], keys: tuple[str, ...]) -> bool:
    return all(Path(files[k]).is_file() for k in keys)


def _verify_raw_yuv(path: Path, expected_frames: int) -> None:
    if raw_yuv_frame_count(path) != expected_frames:
        raise RuntimeError(f"{path}: expected {expected_frames} frames")


def _verify_compressed(ffmpeg: str, path: Path, expected_frames: int, family: str) -> None:
    info = ffprobe_stream(ffmpeg, path)
    w = int(info.get("width", "0") or "0")
    h = int(info.get("height", "0") or "0")
    if w != WIDTH or h != HEIGHT:
        raise RuntimeError(f"{path}: expected {WIDTH}x{HEIGHT}, got {w}x{h}")
    nb = info.get("nb_read_frames") or info.get("nb_frames") or ""
    if nb.isdigit() and int(nb) != expected_frames:
        raise RuntimeError(f"{path}: expected {expected_frames} frames, ffprobe {nb}")


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
    stream_reference_yuv(reference_yuv, family=family, frame_count=frame_count)
    _verify_raw_yuv(reference_yuv, frame_count)

    raw_to_hevc_lossy(ffmpeg, reference_yuv, distorted_lossy, family=family, qp=34)
    _verify_compressed(ffmpeg, distorted_lossy, frame_count, family)

    dist_tmp = distorted_yuv.with_suffix(distorted_yuv.suffix + ".tmp")
    decode_to_yuv(ffmpeg, distorted_lossy, dist_tmp)
    dist_tmp.replace(distorted_yuv)
    _verify_raw_yuv(distorted_yuv, frame_count)

    raw_to_hevc_lossless(ffmpeg, reference_yuv, reference_lossless, family=family)
    _verify_compressed(ffmpeg, reference_lossless, frame_count, family)

    tmp_ref = root / f"_ref_dec_{frame_count}.raw"
    decode_to_yuv(ffmpeg, reference_lossless, tmp_ref)
    if not files_identical(tmp_ref, reference_yuv):
        raise RuntimeError(f"{family}: lossless decode != {reference_yuv.name}")
    tmp_ref.unlink(missing_ok=True)

    tmp_check = root / f"_dist_check_{frame_count}.raw"
    decode_to_yuv(ffmpeg, distorted_lossy, tmp_check)
    if not files_identical(tmp_check, distorted_yuv):
        raise RuntimeError(f"{family}: decode({distorted_lossy.name}) != {distorted_yuv.name}")
    tmp_check.unlink(missing_ok=True)


def _write_manifest(
    fx: FamilyFixtures,
    ffmpeg: str,
    files: dict[str, str],
    *,
    include_nr: bool,
    prior: dict[str, Any] | None,
) -> None:
    keys = ALL_KEYS if include_nr else CORE_KEYS
    sha: dict[str, str] = {}
    if prior and prior.get("sha256"):
        sha.update(prior["sha256"])
    for k in keys:
        sha[k] = sha256_file(Path(files[k]))
    meta: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "family": fx.family,
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAMES,
        "frames_nr": FRAMES_NR,
        "verified": True,
        "files": {k: files[k] for k in keys},
        "sha256": {k: sha[k] for k in keys},
        "lossless_container": "mkv",
        "lossy_container": "mp4",
        "reference_lossless_probe": ffprobe_color(ffmpeg, Path(files["reference_lossless"])),
        "distorted_lossy_probe": ffprobe_color(ffmpeg, Path(files["distorted_lossy"])),
    }
    if include_nr:
        meta["distorted_lossy_60_probe"] = ffprobe_color(ffmpeg, Path(files["distorted_lossy_60"]))
    tmp = fx.manifest_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(fx.manifest_path())


def build_family(
    family: str,
    ffmpeg: str,
    *,
    force: bool = False,
    include_nr: bool = False,
) -> FamilyFixtures:
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
    stored: dict[str, Any] = {}
    if manifest.is_file():
        try:
            stored = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stored = {}

    files = _merge_manifest_files(stored, fx)
    core_ok = _paths_exist(files, CORE_KEYS)
    nr_ok = _paths_exist(files, NR_KEYS)

    if force:
        _lock_path(root).unlink(missing_ok=True)

    if not force and stored.get("verified") and core_ok:
        if not include_nr or nr_ok:
            return fx

    need_core = force or not core_ok
    need_nr = include_nr and (force or not nr_ok)

    if not need_core and not need_nr:
        return fx

    _acquire_lock(root)
    try:
        if need_core:
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
        if need_nr:
            nr_dist_yuv = root / "distorted_60f.yuv"
            _build_yuv_bundle(
                ffmpeg,
                family,
                root,
                frame_count=FRAMES_NR,
                reference_yuv=fx.reference_yuv_60,
                distorted_yuv=nr_dist_yuv,
                reference_lossless=fx.reference_lossless_60,
                distorted_lossy=fx.distorted_lossy_60,
            )
        files = _default_files(fx)
        nr_present = _paths_exist(files, NR_KEYS)
        _write_manifest(
            fx,
            ffmpeg,
            files,
            include_nr=nr_present,
            prior=stored,
        )
    finally:
        _release_lock(root)

    return fx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_all_families(
    ffmpeg: str,
    *,
    force: bool = False,
    include_nr: bool = False,
) -> dict[str, FamilyFixtures]:
    return {
        fam: build_family(fam, ffmpeg, force=force, include_nr=include_nr)
        for fam in ("sdr", "pq", "hlg")
    }
