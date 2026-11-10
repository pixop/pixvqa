"""HDR-ChipQA unified NR adapter: SDR (BT.709), PQ HDR, and experimental HLG via upstream hdr_chipqa.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pixvqa_hdr_chipqa import FIXTURE_LABEL, PINNED_UPSTREAM_COMMIT
from pixvqa_hdr_chipqa.inference import (
    InferenceError,
    SetupError,
    features_to_npy,
    git_revision,
    predict_from_npy,
    resolve_python,
    validate_repo,
)
from pixvqa_video.ffmpeg_bins import (
    FFmpegBinError,
    resolve_ffmpeg_bin,
    resolve_ffprobe_legacy,
)

_HLG_TRANSFER_OK = frozenset({"arib-std-b67", "hlg", "std-b67", "bt2020-hlg"})
_PQ_TRANSFER_OK = frozenset({"smpte2084", "pq", "st2084"})
_UNSUPPORTED_HDR_TRANSFER = frozenset({"smpte428", "smpte240", "iec61966-2-4"})
_SDR_TRANSFER_OK = frozenset({
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "iec61966-2-1",
    "gamma22",
    "gamma28",
    "sdr",
    "unknown",
    "unspecified",
    "",
})

_HLG_EXPERIMENTAL_WARNING = (
    "WARNING: HDR-ChipQA on HLG is experimental: HLG performance was not "
    "experimentally validated in the HDR-ChipQA paper; scores are uncalibrated "
    "and should not be treated as established subjective-quality predictions."
)

_hlg_warning_emitted = False


def emit_hlg_experimental_warning() -> None:
    global _hlg_warning_emitted
    if not _hlg_warning_emitted:
        print(_HLG_EXPERIMENTAL_WARNING, file=sys.stderr)
        _hlg_warning_emitted = True


def frame_bytes_yuv420(width: int, height: int, bit_depth: int) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise SetupError("YUV420 requires positive even width and height")
    if bit_depth > 8:
        return width * height * 3
    return width * height * 3 // 2


def _run_ffprobe(ffprobe: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [ffprobe, "-v", "error", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SetupError(f"could not launch ffprobe ({ffprobe!r}): {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise SetupError(f"ffprobe failed: {tail[-500:] or proc.returncode}")
    return proc


def _norm_tag(value: object) -> str:
    if value in (None, ""):
        return "unknown"
    return str(value).strip().lower()


def is_hlg_transfer_norm(tr: str) -> bool:
    return tr in _HLG_TRANSFER_OK or "hlg" in tr or "b67" in tr


def is_pq_transfer_norm(tr: str) -> bool:
    return tr in _PQ_TRANSFER_OK or tr == "bt2020"


def transfer_norm_to_mode(tr: str) -> str | None:
    if is_hlg_transfer_norm(tr):
        return "hlg"
    if is_pq_transfer_norm(tr):
        return "pq"
    if tr in _SDR_TRANSFER_OK:
        return "sdr"
    if tr in _UNSUPPORTED_HDR_TRANSFER:
        return "unsupported_hdr"
    return None


def classify_transfer_token(transfer: str | None) -> str:
    tr = (transfer or "").strip().lower()
    if not tr:
        return "unknown"
    mode = transfer_norm_to_mode(tr)
    if mode is not None:
        return mode
    return "unknown"


def _probe_10bit(meta: dict) -> bool:
    pix = _norm_tag(meta.get("pix_fmt"))
    profile = _norm_tag(meta.get("profile"))
    if "10" in pix or pix.endswith("p10le"):
        return True
    if "main10" in profile or "main 10" in profile:
        return True
    bits = meta.get("bits_per_raw_sample")
    if bits not in (None, "", "unknown", "unspecified"):
        try:
            return int(bits) >= 10
        except (TypeError, ValueError):
            pass
    return False


def _validate_hlg_compressed_tags(meta: dict, *, label: str = "compressed HLG") -> None:
    tr = meta["transfer_norm"]
    if not is_hlg_transfer_norm(tr):
        raise SetupError(f"{label} requires arib-std-b67 / HLG color_transfer (got {meta.get('transfer')!r})")
    prim = _norm_tag(meta.get("color_primaries"))
    if prim not in ("unknown", "unspecified", "bt2020"):
        raise SetupError(
            f"{label} requires BT.2020 primaries when tagged (got {meta.get('color_primaries')!r})"
        )
    matrix = _norm_tag(meta.get("color_space"))
    if matrix not in ("unknown", "unspecified", "bt2020nc", "bt2020"):
        raise SetupError(
            f"{label} requires BT.2020 NCL matrix when tagged (got {meta.get('color_space')!r})"
        )
    rng = _norm_tag(meta.get("color_range"))
    if rng not in ("unknown", "unspecified", "tv", "mpeg", "limited"):
        raise SetupError(f"{label} requires limited range when tagged (got {meta.get('color_range')!r})")
    if not _probe_10bit(meta):
        raise SetupError(f"{label} requires 10-bit video (pix_fmt/profile); got {meta.get('pix_fmt')!r}")


def _reject_compressed_hdr_unknown(meta: dict) -> None:
    tr = meta["transfer_norm"]
    mode = transfer_norm_to_mode(tr)
    if mode == "unsupported_hdr":
        raise SetupError(
            f"hdr-chipqa rejects unsupported HDR transfer {meta.get('transfer')!r}; "
            "use SDR BT.709, PQ raw YUV, or experimental HLG"
        )
    if mode == "pq":
        raise SetupError(
            f"hdr-chipqa compressed PQ transfer ({meta.get('transfer')}) rejected; "
            "use raw 10-bit PQ YUV for HDR scoring"
        )
    if mode is None and tr not in ("unknown", "unspecified"):
        raise SetupError(f"hdr-chipqa rejects unrecognized transfer {meta.get('transfer')!r}")


def _resolved_ffprobe(ffprobe: str | None) -> str:
    if ffprobe is not None:
        return ffprobe
    try:
        return resolve_ffprobe_legacy("PIXVQA_HDR_CHIPQA_FFPROBE")
    except FFmpegBinError as exc:
        raise SetupError(str(exc)) from exc


def _resolved_ffmpeg() -> str:
    try:
        return resolve_ffmpeg_bin()
    except FFmpegBinError as exc:
        raise SetupError(str(exc)) from exc


def inspect_compressed(path: Path, ffprobe: str | None = None) -> dict:
    ffprobe = _resolved_ffprobe(ffprobe)
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SetupError(f"video not found: {path}")
    if path.suffix.lower() in (".yuv", ".raw"):
        raise SetupError("compressed inspect expected a container file, not raw YUV")
    proc = _run_ffprobe(
        ffprobe,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,bits_per_raw_sample,"
            "color_transfer,color_primaries,color_space,color_range",
            "-of",
            "json",
            str(path),
        ],
    )
    try:
        streams = json.loads(proc.stdout).get("streams") or []
    except json.JSONDecodeError as exc:
        raise SetupError("ffprobe stream output is not valid JSON") from exc
    if not streams:
        raise SetupError("video has no video stream")
    stream = streams[0]
    tr_raw = stream.get("color_transfer")
    tr_norm = _norm_tag(tr_raw)
    return {
        "codec": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pix_fmt": stream.get("pix_fmt"),
        "bits_per_raw_sample": stream.get("bits_per_raw_sample"),
        "transfer": tr_raw if tr_raw is not None else "unknown",
        "transfer_norm": tr_norm,
        "color_primaries": stream.get("color_primaries"),
        "color_space": stream.get("color_space"),
        "color_range": stream.get("color_range"),
        "storage": "compressed",
    }


def _assert_transfer_consistency(cli_token: str, detected_mode: str) -> None:
    if cli_token in ("unknown", ""):
        return
    if cli_token == detected_mode:
        return
    raise SetupError(
        f"hdr-chipqa transfer mismatch: CLI declares {cli_token!r} but input resolves to "
        f"{detected_mode!r}"
    )


def resolve_pipeline_mode(
    path: Path,
    *,
    transfer: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bit_depth: int | None = None,
    ffprobe: str | None = None,
) -> dict:
    path = path.expanduser().resolve()
    is_raw = path.suffix.lower() in (".yuv", ".raw")
    token = classify_transfer_token(transfer)
    if is_raw:
        if width is None or height is None or bit_depth is None:
            raise SetupError("raw YUV requires width, height, and bit depth")
        if bit_depth not in (8, 10):
            raise SetupError("hdr-chipqa raw YUV supports 8-bit (SDR) or 10-bit (PQ/HLG HDR)")
        if token in ("unsupported_hdr", "unknown") and transfer:
            raise SetupError(f"hdr-chipqa rejects unsupported or unrecognized transfer {transfer!r}")
        if token == "hlg":
            if bit_depth != 10:
                raise SetupError("HLG raw input requires 10-bit YUV420")
            return {
                "mode": "hlg",
                "color_space": "BT2020",
                "bit_depth": 10,
                "path": str(path),
                "compressed": False,
                "transfer": transfer or "hlg-coded-yuv-domain",
                "experimental": True,
            }
        if token == "pq":
            if bit_depth != 10:
                raise SetupError("PQ HDR raw input requires 10-bit YUV420")
            return {
                "mode": "pq",
                "color_space": "BT2020",
                "bit_depth": 10,
                "path": str(path),
                "compressed": False,
                "transfer": transfer or "pq-coded-yuv-domain",
            }
        return {
            "mode": "sdr",
            "color_space": "BT709",
            "bit_depth": bit_depth,
            "path": str(path),
            "compressed": False,
            "transfer": transfer or "bt709-gamma-yuv",
        }
    vmeta = inspect_compressed(path, ffprobe=ffprobe)
    detected = transfer_norm_to_mode(vmeta["transfer_norm"])
    if detected == "hlg":
        _validate_hlg_compressed_tags(vmeta)
        _assert_transfer_consistency(token, "hlg")
        return {
            "mode": "hlg",
            "color_space": "BT2020",
            "bit_depth": 10,
            "path": str(path),
            "compressed": True,
            "transfer": vmeta["transfer"],
            "width": vmeta.get("width"),
            "height": vmeta.get("height"),
            "experimental": True,
            "video_metadata": vmeta,
        }
    if detected == "sdr" or detected is None:
        if token in ("pq", "hlg"):
            raise SetupError(
                f"hdr-chipqa {token.upper()} transfer requested but compressed stream is SDR "
                f"({vmeta.get('transfer')!r})"
            )
        _reject_compressed_hdr_unknown(vmeta)
        if token not in ("sdr", "unknown"):
            _assert_transfer_consistency(token, "sdr")
        return {
            "mode": "sdr",
            "color_space": "BT709",
            "bit_depth": 8,
            "path": str(path),
            "compressed": True,
            "transfer": vmeta["transfer"],
            "width": vmeta.get("width"),
            "height": vmeta.get("height"),
        }
    _reject_compressed_hdr_unknown(vmeta)
    raise SetupError(f"hdr-chipqa could not classify compressed transfer {vmeta.get('transfer')!r}")


def decode_sdr_to_yuv(video: Path, dest: Path, ffprobe: str | None = None) -> tuple[int, int]:
    video = video.expanduser().resolve()
    meta = inspect_compressed(video, ffprobe=ffprobe)
    w, h = meta.get("width"), meta.get("height")
    if not w or not h:
        raise SetupError("could not read compressed SDR dimensions via ffprobe")
    cmd = [
        _resolved_ffmpeg(),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SetupError(f"could not decode SDR video for hdr-chipqa: {exc}") from exc
    if proc.returncode != 0:
        raise InferenceError(f"SDR decode failed: {(proc.stderr or proc.stdout or '')[-800:]}")
    frame_bytes = frame_bytes_yuv420(int(w), int(h), 8)
    size = dest.stat().st_size
    if size <= 0 or size % frame_bytes:
        raise InferenceError("decoded SDR YUV size is not an exact multiple of frame size")
    return int(w), int(h)


def decode_hlg_to_yuv10(video: Path, dest: Path, ffprobe: str | None = None) -> tuple[int, int]:
    video = video.expanduser().resolve()
    meta = inspect_compressed(video, ffprobe=ffprobe)
    detected = transfer_norm_to_mode(meta["transfer_norm"])
    if detected != "hlg":
        raise SetupError(f"HLG decode refused: stream transfer is {meta.get('transfer')!r}")
    _validate_hlg_compressed_tags(meta)
    w, h = meta.get("width"), meta.get("height")
    if not w or not h:
        raise SetupError("could not read compressed HLG dimensions via ffprobe")
    cmd = [
        _resolved_ffmpeg(),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-pix_fmt",
        "yuv420p10le",
        "-f",
        "rawvideo",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SetupError(f"could not decode HLG video for hdr-chipqa: {exc}") from exc
    if proc.returncode != 0:
        raise InferenceError(f"HLG decode failed: {(proc.stderr or proc.stdout or '')[-800:]}")
    frame_bytes = frame_bytes_yuv420(int(w), int(h), 10)
    size = dest.stat().st_size
    if size <= 0 or size % frame_bytes:
        raise InferenceError("decoded HLG YUV size is not an exact multiple of frame size")
    return int(w), int(h)


def validate_yuv(
    path: Path,
    width: int,
    height: int,
    *,
    bit_depth: int = 10,
    color_space: str = "BT2020",
    transfer: str | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SetupError(f"raw YUV not found: {path}")
    token = classify_transfer_token(transfer)
    if token in ("unsupported_hdr", "unknown") and transfer:
        raise SetupError(f"hdr-chipqa rejects unsupported or unrecognized transfer {transfer!r}")
    if token == "hlg":
        if color_space != "BT2020" or bit_depth != 10:
            raise SetupError("HLG requires 10-bit YUV420 with BT2020 matrix (no EOTF)")
    if token == "pq" and (color_space != "BT2020" or bit_depth != 10):
        raise SetupError("PQ HDR requires 10-bit YUV420 with BT2020 matrix")
    if color_space == "BT2020" and bit_depth != 10:
        raise SetupError("BT2020 matrix path requires 10-bit YUV420")
    if color_space == "BT709" and bit_depth not in (8, 10):
        raise SetupError("SDR BT709 path requires 8- or 10-bit YUV420")
    if color_space == "BT709" and token == "hlg":
        raise SetupError("HLG transfer incompatible with BT709 color matrix")
    if color_space == "BT709" and token == "pq":
        raise SetupError("PQ transfer incompatible with BT709 color matrix")
    if color_space not in ("BT2020", "BT709"):
        raise SetupError("hdr-chipqa color matrix must be BT709 (SDR) or BT2020 (PQ/HLG HDR)")
    frame_bytes = frame_bytes_yuv420(width, height, bit_depth)
    size = path.stat().st_size
    if size <= 0 or size % frame_bytes:
        raise SetupError(
            f"{path} is not an exact multiple of {width}x{height} "
            f"YUV420 {'p10le' if bit_depth > 8 else 'p'} frame size ({frame_bytes} B)"
        )
    frames = size // frame_bytes
    if frames < 5:
        raise SetupError("hdr-chipqa needs at least 5 frames for temporal buffer warmup")
    fps = fps_num / fps_den if fps_num and fps_den and fps_num > 0 and fps_den > 0 else None
    storage = (
        "little-endian uint16, 10-bit codes in low bits (upstream yuv_read)"
        if bit_depth > 8
        else "planar YUV420p 8-bit (gamma-coded SDR luma)"
    )
    mode = token if token in ("hlg", "pq", "sdr") else ("pq" if color_space == "BT2020" else "sdr")
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "frames_in_file": frames,
        "bit_depth": bit_depth,
        "color_space": color_space,
        "transfer": transfer or (
            "hlg-coded-yuv-domain" if mode == "hlg"
            else "pq-coded-yuv-domain" if color_space == "BT2020"
            else "bt709-gamma-yuv"
        ),
        "fps": fps,
        "storage": storage,
        "fixture_note": FIXTURE_LABEL if color_space == "BT2020" and mode == "pq" else None,
        "experimental": mode == "hlg",
    }


def _inline_extract_script(
    repo: Path,
    argv: list[str],
    parent_pythonpath: str,
) -> str:
    return f"""
import os, runpy, sys, tempfile
from pathlib import Path
import pixvqa_hdr_chipqa.numpy_legacy_shim  # noqa: F401
hdr = Path({str(repo)!r})
tmp_pkg = hdr / "ChipQA" / "ChipQA_tmp"
td = Path(tempfile.mkdtemp(prefix="pixvqa-hdr-chipqa-path-"))
(td / "ChipQA").symlink_to(tmp_pkg, target_is_directory=True)
for p in (str(td), str(hdr), {parent_pythonpath!r}):
    if p not in sys.path:
        sys.path.insert(0, p)
import ChipQA.save_stats as _ss
if not hasattr(_ss, "extract_subband_feats"):
    _ss.extract_subband_feats = _ss._extract_subband_feats
import pixvqa_hdr_chipqa.feature_accel_shim as _fa
_fa.install()
os.chdir(hdr)
sys.argv = [{", ".join(repr(x) for x in argv)}]
runpy.run_path(str(hdr / "hdr_chipqa.py"), run_name="__main__")
"""


def extract_features(
    repo: Path,
    yuv: Path,
    features_z: Path,
    width: int,
    height: int,
    *,
    bit_depth: int,
    color_space: str,
    feature_python: str | None = None,
    timeout: int = 3600,
) -> None:
    repo = repo.expanduser().resolve()
    yuv = yuv.expanduser().resolve()
    py = resolve_python(feature_python, "PIXVQA_HDR_CHIPQA_FEATURE_PYTHON")
    parent = str(Path(__file__).resolve().parent.parent)
    argv = [
        "hdr_chipqa.py",
        "--input_file",
        str(yuv),
        "--results_file",
        str(features_z),
        "--width",
        str(width),
        "--height",
        str(height),
        "--bit_depth",
        str(bit_depth),
        "--color_space",
        color_space,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [py, "-c", _inline_extract_script(repo, argv, parent)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise InferenceError(f"HDR-ChipQA feature extraction timed out after {timeout}s") from exc
    except OSError as exc:
        raise SetupError(f"could not start HDR-ChipQA feature extraction: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if proc.returncode < 0:
            import signal
            signum = -proc.returncode
            try:
                signal_name = signal.Signals(signum).name
            except ValueError:
                signal_name = f"signal {signum}"
            reason = f"child terminated by {signal_name} ({signum})"
            if signum == signal.SIGKILL:
                reason += "; possible system/cgroup OOM kill or external termination; inspect dmesg/journalctl"
        else:
            reason = f"child exited with status {proc.returncode}"
        details = err[-3000:] if err else "no child stderr/stdout (no Python traceback)"
        raise InferenceError(f"HDR-ChipQA feature extraction failed: {reason}; {details}")
    if not features_z.is_file() or features_z.stat().st_size == 0:
        raise InferenceError("HDR-ChipQA feature extraction produced no output")


def validate(
    repo: Path,
    video: Path,
    width: int | None = None,
    height: int | None = None,
    *,
    bit_depth: int | None = None,
    transfer: str | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
    feature_python: str | None = None,
    inference_python: str | None = None,
    ffprobe: str | None = None,
    check_input: bool = True,
) -> dict:
    meta = validate_repo(repo)
    feat_py = resolve_python(feature_python, "PIXVQA_HDR_CHIPQA_FEATURE_PYTHON")
    inf_py = resolve_python(inference_python, "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON", required=False)
    pipe = resolve_pipeline_mode(
        video,
        transfer=transfer,
        width=width,
        height=height,
        bit_depth=bit_depth,
        ffprobe=ffprobe,
    )
    if pipe.get("mode") == "hlg":
        emit_hlg_experimental_warning()
    ymeta: dict = pipe
    if check_input and not pipe["compressed"]:
        ymeta = validate_yuv(
            Path(pipe["path"]),
            width or 0,
            height or 0,
            bit_depth=pipe["bit_depth"],
            color_space=pipe["color_space"],
            transfer=transfer,
            fps_num=fps_num,
            fps_den=fps_den,
        )
    elif check_input:
        inspect_compressed(video, ffprobe=ffprobe)
    return {
        **meta,
        "video": str(video.expanduser().resolve()),
        "input": ymeta,
        "pipeline": pipe["mode"],
        "feature_python": feat_py,
        "inference_python": inf_py or None,
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "experimental": pipe.get("mode") == "hlg",
    }


def run(
    repo: Path,
    video: Path,
    width: int | None = None,
    height: int | None = None,
    *,
    bit_depth: int | None = None,
    transfer: str | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
    feature_python: str | None = None,
    inference_python: str | None = None,
    ffprobe: str | None = None,
    timeout: int = 3600,
) -> dict:
    info = validate(
        repo,
        video,
        width,
        height,
        bit_depth=bit_depth,
        transfer=transfer,
        fps_num=fps_num,
        fps_den=fps_den,
        feature_python=feature_python,
        inference_python=inference_python,
        ffprobe=ffprobe,
    )
    repo = Path(info["repo"])
    pipe = info["input"]
    feat_py = info["feature_python"]
    color_space = pipe["color_space"]
    bd = pipe["bit_depth"]
    mode = info.get("pipeline") or ("sdr" if color_space == "BT709" else "pq")

    with tempfile.TemporaryDirectory(prefix="pixvqa_hdr_chipqa_") as tmp:
        work_yuv = Path(pipe["path"])
        cleanup_yuv: Path | None = None
        w, h = width, height
        if pipe.get("compressed"):
            if mode == "hlg":
                decoded = Path(tmp) / "decoded_hlg.yuv"
                w, h = decode_hlg_to_yuv10(Path(pipe["path"]), decoded, ffprobe=ffprobe)
            else:
                decoded = Path(tmp) / "decoded_sdr.yuv"
                w, h = decode_sdr_to_yuv(Path(pipe["path"]), decoded, ffprobe=ffprobe)
            work_yuv = decoded
            cleanup_yuv = decoded
        elif w is None or h is None:
            raise SetupError("raw YUV requires width and height")
        features_z = Path(tmp) / "features.z"
        features_npy = Path(tmp) / "features.npy"
        extract_features(
            repo,
            work_yuv,
            features_z,
            w,
            h,
            bit_depth=bd,
            color_space=color_space,
            feature_python=feat_py,
            timeout=timeout,
        )
        features_to_npy(features_z, features_npy, feat_py)
        score, infer_meta = predict_from_npy(features_npy, repo, inference_python)
        if cleanup_yuv is not None:
            infer_meta["decoded_from_compressed"] = pipe["path"]

    if mode == "hlg":
        transfer_label = (
            "HLG-coded YUV domain (no EOTF; experimental; upstream color_space=BT2020 "
            "feature path — not paper-validated for HLG)"
        )
    elif color_space == "BT2020":
        transfer_label = "PQ-coded YUV domain (no EOTF in reader)"
    else:
        transfer_label = "BT.709 gamma-coded YUV (no PQ EOTF; upstream color_space=BT709)"

    return {
        "metric": "hdr-chipqa",
        "pipeline": mode,
        "quality_score": score,
        "video": str(video.expanduser().resolve()),
        "video_metadata": pipe,
        "upstream_repo": str(repo),
        "upstream_commit": git_revision(repo) or PINNED_UPSTREAM_COMMIT,
        "backend": "upstream-hdr_chipqa.py+testing.py",
        "temporal_semantics": "pooled-video",
        "color_space": color_space,
        "transfer": transfer_label,
        "experimental": mode == "hlg",
        "feature_python": feat_py,
        **infer_meta,
    }
