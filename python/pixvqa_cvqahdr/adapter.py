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
from pathlib import Path

from pixvqa_cvqahdr import PINNED_COMMIT, PINNED_UPSTREAM
from pixvqa_device import (
    DEVICE_CHOICES, DeviceError, auto_notice, device_record, peak_rss_bytes, resolve_device,
)


class AdapterError(RuntimeError):
    """A preflight or official-model failure."""


class PreflightError(AdapterError):
    """The environment, assets, or input were rejected before inference started."""


TEMPORAL_NOTE = (
    "Per-sample scores apply the official logistic to each sampled frame's raw quality1 "
    "prediction. This is a PixVQA extension of the published video-level model, not an "
    "officially validated per-frame predictor. QCS from these samples describes sampled-frame "
    "consistency at about one sample per second, not frame-by-frame consistency."
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


def sampling_plan(frame_count: int, fps: float) -> dict:
    """Validate the original VQA_NR.py fixed buffer sampler before it can overflow."""
    if frame_count <= 0:
        raise PreflightError("OpenCV reported no video frames")
    if not math.isfinite(fps) or fps <= 0:
        raise PreflightError(f"OpenCV reported invalid frame rate {fps!r}")
    rounded_fps = int(round(fps))
    if rounded_fps <= 0:
        raise PreflightError(f"OpenCV rounded frame rate {fps!r} to zero")
    sampled_frames = frame_count // rounded_fps
    tail_frames = frame_count % rounded_fps
    sample_offset = rounded_fps // 2
    if sampled_frames == 0:
        raise PreflightError(
            f"clip has {frame_count} frames but the official NR sampler needs at least "
            f"{rounded_fps} frames at rounded {rounded_fps} fps"
        )
    if tail_frames > sample_offset:
        raise PreflightError(
            "clip triggers the official NR sampling-buffer bug: "
            f"{frame_count} frames at rounded {rounded_fps} fps produces an extra sample "
            f"after its {sampled_frames}-frame buffer"
        )
    return {
        "opencv_frame_count": frame_count,
        "opencv_fps": fps,
        "rounded_fps": rounded_fps,
        "sample_offset_frames": sample_offset,
        "sampled_frames": sampled_frames,
        "tail_frames": tail_frames,
    }


def sample_schedule(sampling: dict) -> list[dict]:
    """Frames the official sampler keeps: decoded index % R == R // 2, in order."""
    rounded = sampling["rounded_fps"]
    offset = sampling["sample_offset_frames"]
    fps = sampling["opencv_fps"]
    schedule = []
    for index in range(sampling["sampled_frames"]):
        frame_index = offset + index * rounded
        schedule.append({"index": index, "frame_index": frame_index, "time_seconds": frame_index / fps})
    return schedule


def preflight_video(path: str) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise PreflightError("opencv-python is required by the CompressedVQA-HDR NR adapter") from exc
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise PreflightError(f"OpenCV could not open {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    plan = sampling_plan(frame_count, fps)
    plan["path"] = path
    return plan


def count_decoded_frames(path: str, limit: int) -> int:
    """Count frames the way VQA_NR.video_processing reads them: up to limit successful reads."""
    import cv2

    capture = cv2.VideoCapture(path)
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
        "sampling_density_hz": sampling["opencv_fps"] / sampling["rounded_fps"],
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


def preflight(test: str, repository: str, checkpoint: str, profile: str, device: str,
              report_auto_fallback: bool = False) -> dict:
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
        print(f"pixvqa-cvqahdr: {notice}", file=sys.stderr)
    sampling = preflight_video(test)
    calibration = load_calibration(profile)
    NR_model, VQA_NR = import_official(repository)
    return {
        "upstream": upstream, "sampling": sampling, "calibration": calibration,
        "NR_model": NR_model, "VQA_NR": VQA_NR, "device": effective,
    }


def run(test: str, repository: str, checkpoint: str, profile: str, device: str, output_json: str,
        verify_decoded_frames: bool = False) -> dict:
    context = preflight(test, repository, checkpoint, profile, device)
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
        print(f"pixvqa-cvqahdr: {notice}", file=sys.stderr)
    selected_device = torch.device(effective)
    if effective == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    t0 = time.perf_counter()
    try:
        model = NR_model.SigLIP2_384_multi_dataset().to(selected_device)
        state_dict = torch.load(checkpoint, map_location=selected_device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        t_loaded = time.perf_counter()
        video, video_name = VQA_NR.video_processing(test)
        if int(video.shape[0]) != sampling["sampled_frames"]:
            raise AdapterError(
                f"official NR preprocessing returned {int(video.shape[0])} frames; "
                f"OpenCV preflight expected {sampling['sampled_frames']}"
            )
        t_preprocessed = time.perf_counter()
        batch = video.to(selected_device).unsqueeze(0)
        used = {str(p.device.type) for p in model.parameters()} | {batch.device.type}
        if used != {selected_device.type}:
            raise AdapterError(f"model or input is on {sorted(used)}, not {effective}")
        _, raw_score, raw_values, pooled_parity = forward_with_samples(model, batch, torch)
        if effective == "cuda":
            torch.cuda.synchronize(selected_device)
        t_inferred = time.perf_counter()
        score = float(VQA_NR.logistic_func(raw_score, *calibration))
        calibrated = calibrate(raw_values, calibration, VQA_NR.logistic_func)
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"official NR inference failed: {exc}") from exc
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
    }
    record = {
        "adapter": "pixvqa_cvqahdr",
        "metric": "cvqa-hdr-nr",
        "upstream": {"url": PINNED_UPSTREAM, **upstream},
        "device": effective,
        "device_requested": device,
        "device_info": devices,
        "gpu": gpu,
        "performance": performance,
        "test": test,
        "video_name": video_name,
        "sampling": sampling,
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
    destination = Path(output_json)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:
        raise AdapterError(f"could not write JSON result {output_json}: {exc}") from exc
    return record
