"""Run the official CompressedVQA-HDR NR implementation without modifying it."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

from pixvqa_cvqahdr import PINNED_COMMIT, PINNED_UPSTREAM
from pixvqa_cvqahdr.mos_calibration import (
    apply_mos_calibration,
    gate_reviewed_mos_outcome,
    minimal_unavailable_mos_record,
    mos_calibration_record,
)
from pixvqa_device import (
    DEVICE_CHOICES, DeviceError, auto_notice, device_record, peak_rss_bytes, resolve_device,
)
from pixvqa_video.ffmpeg_bins import resolve_ffprobe_legacy


def configure_deterministic_inference() -> bool:
    """Production-default deterministic execution; configure before CUDA initialization.

    May raise if a GPU operator has no deterministic implementation.
    """
    if os.environ.get("PIXVQA_LEARNED_DETERMINISTIC", "1").lower() in ("0", "false", "no"):
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return True


class AdapterError(RuntimeError):
    """A preflight or official-model failure."""


class PreflightError(AdapterError):
    """The environment, assets, or input were rejected before inference started."""


TEMPORAL_NOTE = (
    "Per-sample scores apply the official logistic to each sampled frame's raw quality1 "
    "prediction. This is a PixVQA extension of the published video-level model, not an "
    "officially validated per-frame predictor. QCS from these samples describes sampled-frame "
    "consistency at the configured sampling interval, not frame-by-frame consistency."
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AdapterError(f"could not read {path}: {exc}") from exc
    return digest.hexdigest()


def version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_upstream(repository: str) -> dict:
    repository = os.path.abspath(repository)
    if not os.path.isfile(os.path.join(repository, "NR", "NR_model.py")):
        raise PreflightError(f"{repository} does not contain NR/NR_model.py")
    try:
        head = subprocess.run(
            ["git", "-C", repository, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", repository, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"could not inspect upstream repository {repository}: {exc}") from exc
    return {
        "repository": repository,
        "expected_commit": PINNED_COMMIT,
        "actual_commit": head,
        "commit_verified": head == PINNED_COMMIT,
        "working_tree_clean": not bool(status.strip()),
    }


def runtime_sampling_options() -> tuple[float, float]:
    """Options supplied by PixVQA to both preflight and inference subprocesses."""
    try:
        interval = float(os.environ.get("PIXVQA_CVQA_NR_SAMPLE_INTERVAL_SEC", "1"))
        chunk_sec = float(os.environ.get("PIXVQA_CVQA_NR_CHUNK_SIZE_SEC", "8"))
    except ValueError as exc:
        raise PreflightError("invalid CVQA-HDR-NR sampling/chunk duration") from exc
    if not all(math.isfinite(v) and v > 0 for v in (interval, chunk_sec)):
        raise PreflightError("CVQA-HDR-NR sampling interval and chunk duration must be positive and finite")
    return interval, chunk_sec


def sampling_plan(frame_count: int, fps: float, interval_sec: float = 1.0) -> dict:
    """Plan midpoint samples in complete intervals; 1.0 s matches patched upstream."""
    if frame_count <= 0:
        raise PreflightError("OpenCV reported no video frames")
    if not math.isfinite(fps) or fps <= 0:
        raise PreflightError(f"OpenCV reported invalid frame rate {fps!r}")
    rounded_fps = int(round(fps))
    if rounded_fps <= 0:
        raise PreflightError(f"OpenCV rounded frame rate {fps!r} to zero")
    # The default interval reproduces the patched official midpoint sampler.
    interval_frames = max(1, int(math.floor(interval_sec * rounded_fps + 0.5)))
    sampled_frames = frame_count // interval_frames
    tail_frames = frame_count % interval_frames
    sample_offset = interval_frames // 2
    if sampled_frames == 0:
        raise PreflightError(
            f"clip has {frame_count} frames but the official NR sampler needs at least "
            f"{interval_frames} frames at rounded {rounded_fps} fps"
        )
    if sampled_frames < 2:
        raise PreflightError(
            f"NR sampling buffer: clip has only {sampled_frames} complete interval(s); need at least 2 "
            f"({2 * interval_frames} frames at rounded {rounded_fps} fps)"
        )
    return {
        "opencv_frame_count": frame_count,
        "opencv_fps": fps,
        "rounded_fps": rounded_fps,
        "sample_offset_frames": sample_offset,
        "sample_interval_sec_requested": interval_sec,
        "sample_interval_frames": interval_frames,
        "sample_interval_sec_effective": interval_frames / fps,
        "sampled_frames": sampled_frames,
        "tail_frames": tail_frames,
    }


def sample_schedule(sampling: dict) -> list[dict]:
    """Midpoint frame indices of complete sampling intervals, in order."""
    rounded = sampling["sample_interval_frames"]
    offset = sampling["sample_offset_frames"]
    fps = sampling["opencv_fps"]
    schedule = []
    for index in range(sampling["sampled_frames"]):
        frame_index = offset + index * rounded
        schedule.append({"index": index, "frame_index": frame_index, "time_seconds": frame_index / fps})
    return schedule


def reject_unsupported_cl(path: str) -> None:
    """The upstream NR preprocessing has no BT.2020 CL color conversion."""
    ffprobe = resolve_ffprobe_legacy("PIXVQA_CVQAHDR_FFPROBE")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_space", "-of", "json", path],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams") or []
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise PreflightError(f"could not verify input color matrix with ffprobe: {exc}") from exc
    if not streams:
        raise PreflightError("ffprobe found no video stream for NR input")
    if streams[0].get("color_space") == "bt2020c":
        raise PreflightError("bt2020c is unsupported by CVQA-HDR-NR (no validated CL conversion)")


def preflight_video(path: str, max_frames: int = -1) -> dict:
    reject_unsupported_cl(path)
    import cv2
    from pixvqa_video.ffmpeg_bgr import FFmpegCapture
    capture = FFmpegCapture(path)
    try:
        if not capture.isOpened():
            raise PreflightError(f"OpenCV could not open {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if max_frames == 0 or max_frames < -1:
        raise PreflightError("--max-frames expects a positive integer or -1")
    effective_frames = min(frame_count, max_frames) if max_frames > 0 else frame_count
    interval_sec, chunk_sec = runtime_sampling_options()
    plan = sampling_plan(effective_frames, fps, interval_sec)
    plan["source_frame_count"] = frame_count
    plan["chunk_size_sec_requested"] = chunk_sec
    plan["chunk_max_sampled_frames"] = max(1, int(math.floor(chunk_sec * fps / plan["sample_interval_frames"] + 1e-9)))
    plan["path"] = path
    return plan


def count_decoded_frames(path: str, limit: int) -> int:
    """Count frames the way VQA_NR.video_processing reads them: up to limit successful reads."""
    import cv2

    from pixvqa_video.ffmpeg_bgr import FFmpegCapture
    capture = FFmpegCapture(path)
    decoded = 0
    try:
        for _ in range(limit):
            ok = capture.grab()
            if ok:
                decoded += 1
    finally:
        capture.release()
    return decoded


def import_official(repository: str):
    nr_directory = os.path.join(os.path.abspath(repository), "NR")
    if nr_directory not in sys.path:
        sys.path.insert(0, nr_directory)
    # The upstream checkout stays read-only: no __pycache__ beside NR_model.py.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import NR_model
        import VQA_NR
    except ImportError as exc:
        raise PreflightError(f"could not import upstream NR implementation: {exc}") from exc
    finally:
        sys.dont_write_bytecode = previous
    return NR_model, VQA_NR


def load_calibration(profile: str):
    import numpy as np

    try:
        calibration = np.load(profile)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"could not load calibration profile {profile}: {exc}") from exc
    if calibration.shape != (4,):
        raise PreflightError(f"calibration profile must have shape (4,), got {calibration.shape}")
    if not all(math.isfinite(float(value)) for value in calibration):
        raise PreflightError("calibration profile contains a non-finite value")
    return calibration


def calibrate(raw_values, calibration, logistic) -> list[float]:
    """Apply the official logistic to each raw sample independently."""
    return [float(logistic(raw, *calibration)) for raw in raw_values]


def temporal_record(sampling: dict, raw_values: list[float], calibrated: list[float],
                    calibration_dtype: str, pooled_parity: bool,
                    decoded_frame_count: int | None) -> dict:
    schedule = sample_schedule(sampling)
    if len(raw_values) != len(schedule) or len(calibrated) != len(schedule):
        raise AdapterError(
            f"captured {len(raw_values)} quality1 samples; the sampler planned {len(schedule)}"
        )
    padded = None
    if decoded_frame_count is not None:
        padded = sum(1 for entry in schedule if entry["frame_index"] >= decoded_frame_count)
    samples = []
    for entry, raw, score in zip(schedule, raw_values, calibrated):
        sample = {**entry, "raw_model_score": raw, "quality_score": score}
        if decoded_frame_count is not None:
            sample["padded"] = entry["frame_index"] >= decoded_frame_count
        samples.append(sample)
    return {
        "basis": "sampled-frames",
        "sample_count": len(samples),
        "sampling_density_hz": sampling["opencv_fps"] / sampling["sample_interval_frames"],
        "raw_dtype": "float32",
        "calibration_dtype": calibration_dtype,
        "pooled_parity": pooled_parity,
        "decoded_frame_count": decoded_frame_count,
        "padded_samples": padded,
        "qcs_eligible": pooled_parity and padded == 0,
        "note": TEMPORAL_NOTE,
        "samples": samples,
    }


def forward_with_samples(model, batch, torch):
    """Run the unmodified forward pass and observe quality1 through a hook."""
    captured = []

    def observe(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.quality1.register_forward_hook(observe)
    try:
        with torch.no_grad():
            outputs = model(batch)
    finally:
        handle.remove()
    raw_score = outputs[0].item()
    if len(captured) != 1:
        raise AdapterError(f"quality1 ran {len(captured)} times; expected once")
    per_sample = captured[0].reshape(batch.shape[0], batch.shape[1])
    pooled_parity = torch.mean(per_sample, dim=1)[0].item() == raw_score
    raw_values = per_sample[0].to("cpu").tolist()
    return outputs, raw_score, raw_values, pooled_parity




def preprocess_custom_interval(test: str, sampling: dict, torch):
    """Use upstream-equivalent transforms with the requested nondefault schedule.

    The official preprocessing function has a fixed ~1 Hz sampler. This path
    changes only the midpoint selection, not the spatial transforms.
    """
    import cv2
    from PIL import Image
    from torchvision import transforms

    from pixvqa_video.ffmpeg_bgr import FFmpegCapture
    capture = FFmpegCapture(test)
    if not capture.isOpened():
        raise AdapterError(f"FFmpeg could not open {test}")
    size = 384
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if width <= 0 or height <= 0:
        capture.release()
        raise AdapterError("invalid decoded video geometry")
    if height > width:
        rw = size
        rh = int(rw / width * height)
    else:
        rh = size
        rw = int(rh / height * width)
    dim = (rw, rh)
    transform = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    selected = [entry["frame_index"] for entry in sample_schedule(sampling)]
    video = torch.zeros((len(selected), 3, size, size))
    sample_index = 0
    frame_index = 0
    try:
        for _ in range(sampling["opencv_frame_count"]):
            ok, frame = capture.read()
            if ok:
                if sample_index < len(selected) and frame_index == selected[sample_index]:
                    resized = cv2.resize(frame, dim)
                    rgb = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                    video[sample_index] = transform(rgb)
                    sample_index += 1
                frame_index += 1
    finally:
        capture.release()
    if sample_index == 0:
        raise AdapterError("no sampled video frames could be decoded")
    # Mirror the upstream behavior when the container's advertised frame count
    # exceeds the number of successfully decoded frames.
    if sample_index < len(selected):
        video[sample_index:] = video[sample_index - 1]
    return video, test


class GpuSampler:
    """Optional device-wide nvidia-smi sampler; metrics include other GPU processes."""

    def __init__(self):
        self.process = None
        self.thread = None
        self.samples = []

    def start(self):
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                bufsize=1,
            )
        except OSError as exc:
            print(f"[CVQA-NR] GPU monitoring unavailable: {exc}", file=sys.stderr, flush=True)
            return

        def collect():
            for line in self.process.stdout:
                # A multi-GPU system emits one line per device, in device index order.
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        self.samples.append((int(parts[0]), int(parts[1])))
                    except ValueError:
                        pass
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            if self.thread is not None:
                self.thread.join(timeout=3)
        if not self.samples:
            return None
        return {
            "samples": len(self.samples),
            "peak_device_memory_mib": max(x[0] for x in self.samples),
            "peak_gpu_utilization_percent": max(x[1] for x in self.samples),
            "mean_gpu_utilization_percent": sum(x[1] for x in self.samples) / len(self.samples),
            "note": "Device-wide nvidia-smi samples (not process-exclusive); on multi-GPU systems "
                    "samples aggregate all GPU rows and must be interpreted accordingly.",
        }


def forward_chunked(model, video, device, torch, chunk_size: int, profile: bool = False):
    """Keep the model's raw per-frame predictions; calibrate only their global mean."""
    raw_values = []
    parity = True
    chunks = []
    expected = int(video.shape[0])
    total_chunks = math.ceil(expected / chunk_size)
    for index, start in enumerate(range(0, expected, chunk_size), 1):
        stop = min(start + chunk_size, expected)
        t0 = time.perf_counter()
        batch = video[start:stop].to(device).unsqueeze(0)
        _, _, values, consistent = forward_with_samples(model, batch, torch)
        if len(values) != stop - start:
            raise AdapterError(f"chunk {index}/{total_chunks}: expected {stop-start} samples, got {len(values)}")
        raw_values.extend(values)
        parity = parity and consistent
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - t0
            chunks.append({"index": index, "start": start, "stop": stop,
                           "sample_count": len(values), "seconds": seconds})
            print(f"[CVQA-NR] Chunk {index}/{total_chunks}: samples={len(values)} "
                  f"({start}:{stop}), inference={seconds:.3f}s",
                  file=sys.stderr, flush=True)
        del batch
    if len(raw_values) != expected:
        raise AdapterError(f"incomplete inference: processed {len(raw_values)}/{expected} sampled frames")
    if profile:
        print(f"[CVQA-NR] Processed samples: {len(raw_values)}/{expected}; "
              f"chunks: {len(chunks)}/{total_chunks}", file=sys.stderr, flush=True)
    # Mirror the official float32 temporal mean; the different reduction
    # grouping may introduce minor floating-point differences.
    raw_score = torch.tensor(raw_values, dtype=torch.float32).mean().item()
    return raw_score, raw_values, parity, chunks


def preflight(test: str, repository: str, checkpoint: str, profile: str, device: str,
              report_auto_fallback: bool = False, max_frames: int = -1) -> dict:
    if device not in DEVICE_CHOICES:
        raise PreflightError(f"unsupported device {device!r}; use auto, cpu, or cuda")
    if not os.path.isfile(test):
        raise PreflightError(f"video not found: {test}")
    for label, path in (("checkpoint", checkpoint), ("calibration profile", profile)):
        if not os.path.isfile(path):
            raise PreflightError(f"{label} not found: {path}")
        try:
            with open(path, "rb") as handle:
                handle.read(1)
        except OSError as exc:
            raise PreflightError(f"{label} is not readable: {path}: {exc}") from exc
    upstream = verify_upstream(repository)
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import torch
        import torchvision  # noqa: F401
        import open_clip  # noqa: F401
    except ImportError as exc:
        raise PreflightError(f"missing CompressedVQA-HDR runtime dependency: {exc}") from exc
    try:
        effective = resolve_device(device, torch)
    except DeviceError as exc:
        raise PreflightError(str(exc)) from exc
    notice = auto_notice(device, effective)
    if notice and report_auto_fallback:
        print(f"pixvqa-cvqa-hdr-nr: warning: {notice}", file=sys.stderr)
    sampling = preflight_video(test, max_frames=max_frames)
    calibration = load_calibration(profile)
    NR_model, VQA_NR = import_official(repository)
    return {
        "upstream": upstream, "sampling": sampling, "calibration": calibration,
        "NR_model": NR_model, "VQA_NR": VQA_NR, "device": effective,
    }


def run(test: str, repository: str, checkpoint: str, profile: str, device: str, output_json: str,
        verify_decoded_frames: bool = False, max_frames: int = -1) -> dict:
    deterministic = configure_deterministic_inference()
    context = preflight(test, repository, checkpoint, profile, device,
                        max_frames=max_frames)
    upstream = context["upstream"]
    sampling = context["sampling"]
    calibration = context["calibration"]
    NR_model, VQA_NR = context["NR_model"], context["VQA_NR"]
    import cv2
    import numpy as np
    import torch
    import torchvision

    effective = context["device"]
    notice = auto_notice(device, effective)
    if notice:
        print(f"pixvqa-cvqa-hdr-nr: warning: {notice}", file=sys.stderr)
    selected_device = torch.device(effective)
    profiling_enabled = os.environ.get("PIXVQA_CVQA_NR_PROFILE", "0").lower() in ("1", "true", "yes")
    sampler = GpuSampler() if profiling_enabled and effective == "cuda" else None
    if sampler is not None:
        sampler.start()
    if profiling_enabled:
        print(f"[CVQA-NR] Video: {sampling['opencv_frame_count']} frames, "
              f"{sampling['opencv_fps']:.3f} fps; planned samples: {sampling['sampled_frames']}; "
              f"interval: {sampling['sample_interval_sec_effective']:.3f}s; "
              f"chunk size: {sampling['chunk_max_sampled_frames']} sampled frames",
              file=sys.stderr, flush=True)
    if effective == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    t0 = time.perf_counter()
    try:
        model = NR_model.SigLIP2_384_multi_dataset().to(selected_device)
        state_dict = torch.load(checkpoint, map_location=selected_device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        t_loaded = time.perf_counter()
        # Preserve the pinned upstream midpoint selection and image transforms
        # without mutating cv2.VideoCapture process-wide. The default 1 Hz
        # schedule uses the exact same integer FPS and midpoint indices.
        video, video_name = preprocess_custom_interval(test, sampling, torch)
        if int(video.shape[0]) != sampling["sampled_frames"]:
            raise AdapterError(
                f"official NR preprocessing returned {int(video.shape[0])} frames; "
                f"OpenCV preflight expected {sampling['sampled_frames']}"
            )
        t_preprocessed = time.perf_counter()
        if any(p.device.type != selected_device.type for p in model.parameters()):
            raise AdapterError(f"model parameters are not all on {effective}")
        chunk_samples = sampling["chunk_max_sampled_frames"]
        raw_score, raw_values, pooled_parity, chunks = forward_chunked(
            model, video, selected_device, torch, chunk_samples, profile=profiling_enabled
        )
        if effective == "cuda":
            torch.cuda.synchronize(selected_device)
        t_inferred = time.perf_counter()
        score = float(VQA_NR.logistic_func(raw_score, *calibration))
        calibrated = calibrate(raw_values, calibration, VQA_NR.logistic_func)
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"official NR inference failed: {exc}") from exc
    finally:
        gpu_samples = sampler.stop() if sampler is not None else None
    decoded = None
    if verify_decoded_frames:
        decoded = count_decoded_frames(test, sampling["opencv_frame_count"])
    temporal = temporal_record(sampling, raw_values, calibrated, str(calibration.dtype),
                               pooled_parity, decoded)
    devices = device_record(device, effective, torch)
    gpu = devices.get("gpu")
    performance = {
        "model_load_seconds": t_loaded - t0,
        "preprocessing_seconds": t_preprocessed - t_loaded,
        "inference_seconds": t_inferred - t_preprocessed,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_cuda_allocated_bytes":
            torch.cuda.max_memory_allocated(selected_device) if effective == "cuda" else None,
        "peak_cuda_reserved_bytes":
            torch.cuda.max_memory_reserved(selected_device) if effective == "cuda" else None,
    }
    if profiling_enabled:
        performance["gpu_monitor"] = gpu_samples
        performance["chunks"] = chunks
        print(f"[CVQA-NR] Peak CUDA allocated: "
              f"{(performance['peak_cuda_allocated_bytes'] or 0) / 1048576:.1f} MiB; "
              f"reserved: {(performance['peak_cuda_reserved_bytes'] or 0) / 1048576:.1f} MiB",
              file=sys.stderr, flush=True)
        if gpu_samples is not None:
            print(f"[CVQA-NR] Device-wide GPU utilization: "
                  f"mean={gpu_samples['mean_gpu_utilization_percent']:.1f}%, "
                  f"peak={gpu_samples['peak_gpu_utilization_percent']}%; "
                  f"device VRAM peak={gpu_samples['peak_device_memory_mib']} MiB",
                  file=sys.stderr, flush=True)
    record = {
        "adapter": "pixvqa_cvqahdr",
        "deterministic_inference_requested": deterministic,
        "metric": "cvqa-hdr-nr",
        "decoder": "ffmpeg-bgr24",
        "upstream": {"url": PINNED_UPSTREAM, **upstream},
        "device": effective,
        "device_requested": device,
        "device_info": devices,
        "gpu": gpu,
        "performance": performance,
        "test": test,
        "video_name": video_name,
        "sampling": sampling,
        "inference": {"mode": "chunked", "chunk_max_sampled_frames": sampling["chunk_max_sampled_frames"],
                      "chunk_size_sec_requested": sampling["chunk_size_sec_requested"],
                      "chunk_count": math.ceil(sampling["sampled_frames"] / sampling["chunk_max_sampled_frames"])},
        "quality_score": score,
        "raw_model_score": raw_score,
        "temporal": temporal,
        "calibration_profile": {
            "path": profile,
            "sha256": sha256(profile),
            "values": [float(value) for value in calibration],
        },
        "checkpoint": {"path": checkpoint, "sha256": sha256(checkpoint)},
        "runtime": {
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "open_clip_torch": version("open-clip-torch"),
            "timm": version("timm"),
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    cvqa_upstream = {
        "upstream": upstream,
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "profile_sha256": record["calibration_profile"]["sha256"],
    }
    try:
        mos_outcome = apply_mos_calibration(score, test)
        mos_outcome = gate_reviewed_mos_outcome(mos_outcome, cvqa_upstream)
        record["mos_calibration"] = mos_calibration_record(
            mos_outcome, cvqa_upstream=cvqa_upstream
        )
        if mos_outcome.available and mos_outcome.mos_calibrated_cvqa is not None:
            record["mos_calibrated_cvqa"] = mos_outcome.mos_calibrated_cvqa
    except Exception as exc:
        # Narrow boundary: only the optional MOS stage; canonical scoring is complete.
        print(f"[CVQA-NR] Optional MOS calibration unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        record["mos_calibration"] = minimal_unavailable_mos_record(
            score, f"calibration_error:{type(exc).__name__}"
        )
    destination = Path(output_json)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    except OSError as exc:
        raise AdapterError(f"could not write JSON result {output_json}: {exc}") from exc
    return record
