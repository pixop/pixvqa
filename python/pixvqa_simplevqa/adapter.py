"""Run the upstream SimpleVQA single-video demo with explicit model provenance.

No PyTorch code is vendored. The official demo is executed in a temporary,
isolated working directory to satisfy its fixed relative ckpts/ path.
CPU runs a staged overlay (no DataParallel; ``map_location`` on load) because
upstream unconditionally wraps the model in DataParallel. The default CUDA
path stages a bounded-window single-decode overlay. Optional ``--use-worker``
amortizes imports for batch CUDA scoring. An experimental streaming mode
is opt-in pending full numerical-parity validation. This is a parity baseline, not a native implementation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


class SetupError(RuntimeError):
    """Missing/invalid inputs or required runtime dependencies."""


class InferenceError(RuntimeError):
    """The upstream implementation did not complete with a valid score."""


REQUIRED_IMPORTS = ("torch", "torchvision", "pytorchvideo", "cv2", "PIL", "numpy")
# Pip distribution names for metadata.version (no import of torch / cv2 required).
_RUNTIME_DIST_NAMES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("pytorchvideo", "pytorchvideo"),
    ("numpy", "numpy"),
    ("opencv_python", "opencv-python"),
    ("pillow", "Pillow"),
)
MODEL_FILES = ("test_demo.py", "model/UGC_BVQA_model.py")
SCORE_RE = re.compile(r"^\s*(.+),\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")

# Upstream test_demo.py main() loads a CUDA-saved checkpoint through DataParallel even
# when --is_gpu is absent. CPU hosts need a staged overlay (parity-reviewed; no math change).
_CPU_OVERLAY_OLD_BLOCK = """    model = UGC_BVQA_model.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.load_state_dict(torch.load('ckpts/UGC_BVQA_model.pth'))"""
_CPU_OVERLAY_NEW_BLOCK = """    model = UGC_BVQA_model.resnet50(pretrained=False)
    _ckpt = torch.load('ckpts/UGC_BVQA_model.pth', map_location=device)
    if any(_k.startswith('module.') for _k in _ckpt):
        _ckpt = {_k.replace('module.', '', 1): _v for _k, _v in _ckpt.items()}
    model.load_state_dict(_ckpt)
    model = model.to(device=device)"""
_CPU_OVERLAY_EXTRA = (
    ("with torch.no_grad():", "with torch.inference_mode():"),
    (
        "feature_motion = torch.zeros([n_clip, 2048+256])",
        "feature_motion = torch.zeros([n_clip, 2048+256], device=device)",
    ),
)

# CUDA single-decode overlay: one OpenCV pass for spatial keyframes + motion clips.
# Parity-reviewed on official 960x540/8s clip (exact FP32 score 2.9507508277893066).
_CUDA_SINGLE_DECODE_MAIN_OLD = """        video_dist_spatial, video_name = video_processing_spatial(config.dist)
        video_dist_motion, video_name = video_processing_motion(config.dist)"""
_CUDA_SINGLE_DECODE_MAIN_NEW = """        if os.environ.get('PIXVQA_SIMPLEVQA_EXPERIMENTAL_STREAMING') == '1':
            video_dist_spatial, video_name = video_processing_spatial(config.dist)
            video_dist_motion, video_name = video_processing_motion_stream(config.dist)
        else:
            video_dist_spatial, video_dist_motion, video_name = video_processing_unified(config.dist)"""
_CUDA_UNIFIED_FN_ANCHOR = "def pack_pathway_output(frames, device):"
_CUDA_UNIFIED_FN = '''

def video_processing_unified(dist):
    """Single OpenCV decode pass for spatial keyframes and motion clip tensors."""
    video_name = dist
    video_name_dis = video_name
    video_capture = cv2.VideoCapture()
    video_capture.open(dist)
    video_channel = 3
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
    if video_frame_rate <= 0:
        video_capture.release()
        raise ValueError("SimpleVQA: OpenCV reported an invalid frame rate")
    video_length_read = int(video_length / video_frame_rate)
    video_clip = int(video_length / video_frame_rate)
    video_clip_min = 8
    video_length_clip = 32

    spatial_tf = transforms.Compose([
        transforms.Resize(520), transforms.CenterCrop(448), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    motion_tf = transforms.Compose([
        transforms.Resize([224, 224]), transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    # Keep only the actual 32-frame SlowFast windows, not a tensor of every
    # decoded frame. Each clip retains the upstream order, shape, and FP32 data.
    transformed_video = torch.zeros([video_length_read, video_channel, 448, 448])
    motion_clips = [
        torch.zeros([video_length_clip, video_channel, 224, 224])
        for _ in range(video_clip)
    ]
    video_read_index = 0
    frame_idx = 0
    spatial_idx = 0
    last_decoded_frame = None

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if not has_frames:
            continue
        # Retain the last decoded BGR frame for exact upstream tail padding.
        # It may lie beyond the final 32-frame motion window.
        last_decoded_frame = frame
        t = video_read_index
        # A frame can belong to multiple 32-frame windows when fps < 32.
        first_clip = max(0, (t - video_length_clip + video_frame_rate) // video_frame_rate)
        last_clip = min(video_clip - 1, t // video_frame_rate)
        need_motion = first_clip <= last_clip
        need_spatial = (spatial_idx < video_length_read) and (frame_idx % video_frame_rate == 0)
        if need_motion or need_spatial:
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if need_motion:
                motion_frame = motion_tf(read_frame)
                for clip_idx in range(first_clip, last_clip + 1):
                    motion_clips[clip_idx][t - clip_idx * video_frame_rate] = motion_frame
            if need_spatial:
                transformed_video[spatial_idx] = spatial_tf(read_frame)
                spatial_idx += 1
        video_read_index += 1
        frame_idx += 1

    if spatial_idx < video_length_read:
        for i in range(spatial_idx, video_length_read):
            transformed_video[i] = transformed_video[spatial_idx - 1]

    # The upstream reference pads a truncated decode with its final frame,
    # then pads any incomplete 32-frame window with the same last frame.
    # Convert that tail frame only if one of the selected windows needs it.
    needs_padding = any(
        video_read_index < clip_idx * video_frame_rate + video_length_clip
        for clip_idx in range(video_clip)
    )
    if last_decoded_frame is not None and needs_padding:
        last_image = Image.fromarray(cv2.cvtColor(last_decoded_frame, cv2.COLOR_BGR2RGB))
        last_motion_frame = motion_tf(last_image)
        for clip_idx, clip_tensor in enumerate(motion_clips):
            start = clip_idx * video_frame_rate
            padded_from = min(video_length_clip, max(0, video_read_index - start))
            if padded_from < video_length_clip:
                clip_tensor[padded_from:] = last_motion_frame

    video_capture.release()
    if video_clip < video_clip_min:
        for i in range(video_clip, video_clip_min):
            motion_clips.append(motion_clips[video_clip - 1])
    return transformed_video, motion_clips, video_name_dis
'''

# Experimental streaming alternative to the current bounded-window CUDA path.
# It preserves all selected full-frame spatial samples and exactly the same
# SlowFast motion windows. Activation is opt-in pending authentic-score parity.
_CUDA_STREAMING_MOTION_FN = '''
def video_processing_motion_stream(dist):
    """Yield upstream SlowFast windows while keeping at most 32 decoded tensors."""
    from collections import deque

    cap = cv2.VideoCapture(dist)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    if fps <= 0:
        cap.release()
        raise ValueError("SimpleVQA: OpenCV reported an invalid frame rate")
    n_clip = int(video_length / fps)
    if n_clip < 1:
        cap.release()
        raise ValueError("SimpleVQA requires at least one full second of video")
    tf = transforms.Compose([
        transforms.Resize([224, 224]), transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    class MotionWindows:
        def __len__(self):
            return max(8, n_clip)

        def __iter__(self):
            # OpenCV capture belongs to this iterator; it is closed on exhaustion
            # or generator finalization (including exceptions).
            window = deque(maxlen=32)
            decoded = 0
            next_clip = 0
            last_frame = None
            last_clip = None
            try:
                for _ in range(video_length):
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    # Keep only a single decoded-frame index, even on corrupt inputs.
                    idx = decoded
                    decoded += 1
                    last_frame = frame
                    # We need the transformed frame only if a motion window uses it.
                    first_clip = max(0, (idx - 31 + fps - 1) // fps)
                    last_including = min(n_clip - 1, idx // fps)
                    if first_clip > last_including:
                        continue
                    tensor = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    window.append((idx, tensor))
                    while next_clip < n_clip and idx >= next_clip * fps + 31:
                        start = next_clip * fps
                        selected = [value for pos, value in window if start <= pos < start + 32]
                        if len(selected) != 32:
                            raise RuntimeError("SimpleVQA streaming motion window incomplete")
                        last_clip = torch.stack(selected)
                        yield last_clip
                        next_clip += 1

                if last_frame is None:
                    raise ValueError("SimpleVQA: no decoded frames")
                # Upstream pads unread frames with the final successfully decoded
                # frame and pads the last incomplete 32-frame motion window.
                tail = tf(Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)))
                while next_clip < n_clip:
                    start = next_clip * fps
                    selected = [value for pos, value in window if start <= pos < start + 32]
                    # Missing indices are contiguous after the last decoded frame.
                    first_missing = max(start, decoded)
                    if len(selected) != max(0, min(decoded, start + 32) - start):
                        raise RuntimeError("SimpleVQA streaming tail window incomplete")
                    selected.extend([tail] * (32 - len(selected)))
                    last_clip = torch.stack(selected)
                    yield last_clip
                    next_clip += 1
                for _ in range(n_clip, 8):
                    yield last_clip
            finally:
                cap.release()

    return MotionWindows(), dist
'''

# PQ / HLG and common alias strings from ffprobe color_transfer (SDR-only reference).
_HDR_TRANSFER_REJECT = frozenset({
    "smpte2084",
    "arib-std-b67",
    "pq",
    "hlg",
    "smpte428",
})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_stack_versions() -> dict[str, str | None]:
    """Best-effort versions for experiment manifests (pip metadata only; no heavy imports).

    Values are None when the distribution is not installed in the current interpreter
    (e.g. minimal CI). CUDA / compiled feature flags are not queried here.
    """
    versions: dict[str, str | None] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    for key, dist in _RUNTIME_DIST_NAMES:
        try:
            versions[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _run_ffprobe(ffprobe: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke ffprobe; OSError and failures become SetupError."""
    cmd = [ffprobe, "-v", "error", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SetupError(f"could not launch ffprobe ({ffprobe!r}): {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        if len(tail) > 500:
            tail = tail[-500:]
        raise SetupError(f"ffprobe failed ({' '.join(cmd[:6])}…): {tail or proc.returncode}")
    return proc


def _parse_fps(stream: dict) -> float:
    rate = str(stream.get("avg_frame_rate") or "").split("/")
    if len(rate) != 2:
        raise ValueError("avg_frame_rate")
    fps = float(rate[0]) / float(rate[1])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps")
    return fps


def _parse_duration_seconds(raw: object) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "n/a":
        return None
    try:
        d = float(s)
    except ValueError:
        return None
    if not math.isfinite(d) or d < 0:
        return None
    return d


def _parse_nb_frames(raw: object) -> int | None:
    """Frame count from stream metadata, or None if unknown / N/A."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().upper() == "N/A":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    return n


def _format_duration_seconds(format_block: dict) -> float | None:
    """Prefer floating `format.duration` from ffprobe JSON."""
    val = format_block.get("duration")
    got = _parse_duration_seconds(val)
    if got is not None:
        return got
    tags = format_block.get("tags")
    if isinstance(tags, dict):
        for tag in ("DURATION", "duration"):
            if tag not in tags:
                continue
            s = str(tags[tag]).strip()
            if not s:
                continue
            # e.g. 0:00:01.23 or plain seconds
            if s.count(":") >= 2:
                parts = s.split(":")
                try:
                    sec = float(parts[-1])
                    total = sec + int(parts[-2]) * 60
                    if len(parts) >= 3:
                        total += int(parts[-3]) * 3600
                    if math.isfinite(total) and total >= 0:
                        return total
                except (ValueError, TypeError, IndexError):
                    pass
            else:
                got2 = _parse_duration_seconds(s)
                if got2 is not None:
                    return got2
    return None


def inspect_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SetupError(f"video not found: {path}")
    suffix = path.suffix.lower()
    # Upstream reads via OpenCV; extension-only raw containers are unsupported (ambiguous transport).
    if suffix in (".yuv", ".raw"):
        raise SetupError(
            "upstream SimpleVQA demo expects compressed video with a known codec, not raw YUV/raw pixel"
        )

    stream_proc = _run_ffprobe(
        ffprobe,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate,codec_name,width,height,color_transfer",
            "-of",
            "json",
            str(path),
        ],
    )
    try:
        stream_root = json.loads(stream_proc.stdout)
    except json.JSONDecodeError as exc:
        raise SetupError("ffprobe stream output is not valid JSON") from exc
    if not isinstance(stream_root, dict):
        raise SetupError("ffprobe stream output JSON root is not an object")
    streams = stream_root.get("streams")
    if not isinstance(streams, list) or not streams:
        raise SetupError("video has no video stream for SimpleVQA preflight")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise SetupError("ffprobe stream record is not a JSON object")

    format_proc = _run_ffprobe(
        ffprobe,
        ["-show_entries", "format=duration", "-of", "json", str(path)],
    )
    try:
        fmt_root = json.loads(format_proc.stdout)
    except json.JSONDecodeError as exc:
        raise SetupError("ffprobe format output is not valid JSON") from exc
    if not isinstance(fmt_root, dict):
        raise SetupError("ffprobe format output JSON root is not an object")
    fmt_val = fmt_root.get("format")
    if fmt_val is None:
        fmt: dict = {}
    elif isinstance(fmt_val, dict):
        fmt = fmt_val
    else:
        raise SetupError("ffprobe format record is not a JSON object")
    duration_s = _format_duration_seconds(fmt)

    try:
        fps = _parse_fps(stream)
    except (KeyError, IndexError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise SetupError("video metadata has no valid video stream/frame rate") from exc

    rounded_fps = max(1, int(round(fps)))
    nb_parsed = _parse_nb_frames(stream.get("nb_frames"))
    if stream.get("nb_frames") not in (None, "N/A") and nb_parsed is None:
        raise SetupError("invalid or malformed nb_frames in video stream metadata")

    # Reject sub-second or invalid clips when duration is known.
    if duration_s is not None:
        if duration_s <= 0:
            raise SetupError("invalid video duration (non-positive)")
        if duration_s < 1.0:
            raise SetupError("upstream SimpleVQA requires at least one full second of video (duration < 1s)")

    if nb_parsed is not None:
        if nb_parsed < rounded_fps:
            raise SetupError("upstream SimpleVQA requires at least one full second of video (frame count too low)")
    elif duration_s is not None:
        approx_frames = duration_s * fps
        if approx_frames < float(rounded_fps):
            raise SetupError("upstream SimpleVQA requires at least one full second of video (estimated frames too low)")
    else:
        raise SetupError(
            "cannot verify video length for SimpleVQA preflight (missing nb_frames and duration)"
        )

    tr_raw = stream.get("color_transfer")
    tr_norm = str(tr_raw).strip().lower() if tr_raw not in (None, "") else None
    if tr_norm in _HDR_TRANSFER_REJECT:
        raise SetupError("SimpleVQA reference backend supports SDR only; PQ/HLG rejected")
    # Missing/unknown transfer: still allow SDR workflows; HDR must not be labeled as unknown in practice.

    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "frame_count": nb_parsed,
        "duration_seconds": duration_s,
        "transfer": tr_raw if tr_raw is not None else "unknown",
    }


def validate(repo: Path, checkpoint: Path, video: Path, *,
             check_imports: bool = True, probe_video: bool = True,
             ffprobe: str = "ffprobe") -> dict:
    repo, checkpoint, video = (p.expanduser().resolve() for p in (repo, checkpoint, video))
    if not repo.is_dir():
        raise SetupError(f"upstream SimpleVQA checkout not found: {repo}")
    for name in MODEL_FILES:
        if not (repo / name).is_file():
            raise SetupError(f"upstream SimpleVQA missing {name}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SetupError(f"SimpleVQA trained checkpoint missing/empty: {checkpoint}")
    if check_imports:
        missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
        if missing:
            raise SetupError("missing required Python modules: " + ", ".join(missing))
    meta = inspect_video(video, ffprobe=ffprobe) if probe_video else {}
    tr = meta.get("transfer")
    if isinstance(tr, str) and tr.strip().lower() in _HDR_TRANSFER_REJECT:
        raise SetupError("SimpleVQA reference backend supports SDR only; PQ/HLG rejected")
    out: dict[str, object] = {
        "repo": str(repo),
        "checkpoint": str(checkpoint),
        "video": str(video),
        "input": meta,
        "runtime_stack": runtime_stack_versions(),
    }
    return out


def cpu_overlay_test_demo(src: Path) -> str:
    """Return CPU-safe test_demo.py text (no DataParallel; map_location honors device).

    Fails explicitly when upstream layout differs from the pinned block (commit 017edda).
    """
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        raise SetupError(f"could not read upstream test_demo.py: {exc}") from exc
    if _CPU_OVERLAY_OLD_BLOCK not in text:
        raise SetupError(
            "upstream test_demo.py layout changed; CPU overlay block not found "
            "(expected DataParallel + torch.load without map_location at main())"
        )
    text = text.replace(_CPU_OVERLAY_OLD_BLOCK, _CPU_OVERLAY_NEW_BLOCK, 1)
    for old, new in _CPU_OVERLAY_EXTRA:
        if old not in text:
            raise SetupError(
                f"upstream test_demo.py layout changed; expected snippet missing: {old!r}"
            )
        text = text.replace(old, new, 1)
    return text


def cuda_single_decode_overlay_test_demo(src: Path) -> str:
    """Return CUDA test_demo.py with single OpenCV decode (parity-reviewed; no math change).

    Upstream DataParallel, checkpoint, sampling, color pipeline, and tensor shapes
    are unchanged. Fails explicitly when upstream layout differs from pinned anchors.
    """
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        raise SetupError(f"could not read upstream test_demo.py: {exc}") from exc
    if _CUDA_SINGLE_DECODE_MAIN_OLD not in text:
        raise SetupError(
            "upstream test_demo.py layout changed; single-decode main() block not found "
            "(expected separate video_processing_spatial/motion calls)"
        )
    if _CUDA_UNIFIED_FN_ANCHOR not in text:
        raise SetupError(
            "upstream test_demo.py layout changed; pack_pathway_output anchor not found"
        )
    if "def video_processing_unified(" in text:
        raise SetupError("upstream test_demo.py already contains video_processing_unified")
    text = text.replace(_CUDA_UNIFIED_FN_ANCHOR, _CUDA_STREAMING_MOTION_FN + "\n" + _CUDA_UNIFIED_FN + "\n" + _CUDA_UNIFIED_FN_ANCHOR, 1)
    text = text.replace(_CUDA_SINGLE_DECODE_MAIN_OLD, _CUDA_SINGLE_DECODE_MAIN_NEW, 1)
    return text


def parse_score(text: str, expected_video: Path) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise InferenceError("upstream result must contain exactly one video score line")
    match = SCORE_RE.fullmatch(lines[0])
    if not match:
        raise InferenceError("upstream result line is not valid path,score CSV")
    try:
        score = float(match.group(2))
    except ValueError as exc:
        raise InferenceError("upstream score is not a valid number") from exc
    if not math.isfinite(score):
        raise InferenceError("upstream predicted a non-finite score")
    path_a = Path(match.group(1)).expanduser().resolve()
    path_b = expected_video.expanduser().resolve()
    if path_a != path_b:
        raise InferenceError("upstream result does not match the requested video path")
    return score


def run(repo: Path, checkpoint: Path, video: Path, *,
        device: str = "cpu", ffprobe: str = "ffprobe",
        timeout: int = 3600, use_worker: bool = False) -> dict:
    if device not in ("cpu", "cuda"):
        raise SetupError("device must be cpu or cuda")
    if timeout <= 0:
        raise SetupError("timeout must be positive")
    info = validate(repo, checkpoint, video, ffprobe=ffprobe)
    repo, checkpoint, video = Path(info["repo"]), Path(info["checkpoint"]), Path(info["video"])
    video_in = str(video)
    if use_worker:
        if device != "cuda":
            raise SetupError("persistent worker backend requires device=cuda")
        from .worker import run_via_worker

        score = run_via_worker(
            repo, checkpoint, video,
            timeout=timeout,
        )
        return {
            "metric": "simplevqa", "quality_score": score, "device": device,
            "checkpoint_sha256": digest(checkpoint),
            "upstream_repo": str(repo), "upstream_commit": git_revision(repo),
            "upstream_test_demo": "cuda-single-decode-worker",
            "video": str(video), "video_metadata": info["input"],
            "backend": "persistent-worker", "temporal_semantics": "pooled-video",
            "slowfast_pretrained_source": "upstream PyTorchVideo pretrained cache/download",
            "runtime_stack": info.get("runtime_stack") or runtime_stack_versions(),
        }
    # Check dependencies before launching the heavyweight upstream demo.
    # The demo internally loads slowfast_r50(pretrained=True); that call may
    # trigger PyTorchVideo/hub download and substantial model allocations — not
    # initiated by this wrapper. Never suppress
    # failures or silently accept a randomly initialized backbone.
    with tempfile.TemporaryDirectory(prefix="pixvqa_simplevqa_") as tmp:
        root = Path(tmp)
        upstream_demo = repo / "test_demo.py"
        if device == "cpu":
            try:
                (root / "test_demo.py").write_text(
                    cpu_overlay_test_demo(upstream_demo), encoding="utf-8"
                )
            except OSError as exc:
                raise SetupError(f"could not stage CPU overlay test_demo.py: {exc}") from exc
            upstream_test_demo = "cpu-overlay"
        else:
            try:
                (root / "test_demo.py").write_text(
                    cuda_single_decode_overlay_test_demo(upstream_demo), encoding="utf-8"
                )
            except OSError as exc:
                raise SetupError(f"could not stage CUDA overlay test_demo.py: {exc}") from exc
            upstream_test_demo = "cuda-bounded-motion-overlay"
        try:
            (root / "model").symlink_to(repo / "model", target_is_directory=True)
        except OSError as exc:
            raise SetupError(f"could not stage upstream model symlink: {exc}") from exc
        (root / "ckpts").mkdir()
        try:
            (root / "ckpts" / "UGC_BVQA_model.pth").symlink_to(checkpoint)
        except OSError as exc:
            raise SetupError(f"could not stage checkpoint symlink: {exc}") from exc
        out = root / "result.txt"
        argv = [sys.executable, str(root / "test_demo.py"),
                "--method_name", "single-scale", "--dist", video_in,
                "--output", str(out)]
        if device == "cuda":
            argv.append("--is_gpu")
        try:
            proc = subprocess.run(argv, cwd=root, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            err = (exc.stderr or "").strip()
            if len(err) > 800:
                err = err[-800:]
            raise InferenceError(
                f"SimpleVQA upstream timed out after {timeout}s" + (f": {err}" if err else "")
            ) from exc
        except OSError as exc:
            raise SetupError(f"could not start upstream SimpleVQA process: {exc}") from exc
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            out_snip = (proc.stdout or "").strip()
            if len(err) > 1200:
                err = err[-1200:]
            if not err and out_snip:
                if len(out_snip) > 600:
                    out_snip = out_snip[-600:]
                err = f"(stderr empty; stdout tail) {out_snip}"
            raise InferenceError(
                f"SimpleVQA upstream failed (exit {proc.returncode}): {err or 'no error output'}")
        if not out.is_file():
            raise InferenceError("SimpleVQA upstream produced no result file")
        try:
            score = parse_score(out.read_text(encoding="utf-8"), video)
        except (OSError, UnicodeError) as exc:
            raise InferenceError(f"could not read upstream result file: {exc}") from exc
    return {
        "metric": "simplevqa", "quality_score": score, "device": device,
        "checkpoint_sha256": digest(checkpoint),
        "upstream_repo": str(repo), "upstream_commit": git_revision(repo),
        "upstream_test_demo": upstream_test_demo,
        "video": str(video), "video_metadata": info["input"],
        "backend": "upstream-test_demo", "temporal_semantics": "pooled-video",
        "slowfast_pretrained_source": "upstream PyTorchVideo pretrained cache/download",
        "runtime_stack": info.get("runtime_stack") or runtime_stack_versions(),
    }


def git_revision(repo: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None
