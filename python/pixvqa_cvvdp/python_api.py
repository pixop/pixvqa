"""Run ColorVideoVDP through predict_video_source (mixed inputs)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from pixvqa_cvvdp import PINNED_COMMIT, PINNED_UPSTREAM, PINNED_VERSION
from pixvqa_cvvdp.adapter import AdapterError, runtime_metadata
from pixvqa_cvvdp.temporal import features_path
from pixvqa_device import device_record, peak_rss_bytes


def run_mixed_predict(
    prepared: dict,
    interpretation: str,
    display: str,
    nframes: int,
    device: str,
    output_json: str,
    features_dir: Path | None = None,
    gpu_mem_gb: float | None = None,
) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise AdapterError(f"CVVDP mixed input needs torch in this interpreter: {exc}") from exc

    import pycvvdp
    from pixvqa_cvvdp.mixed_video_source import PixvqaMixedVideoSource

    raw = prepared["raw_record"]
    if raw is None:
        raise AdapterError("mixed CVVDP run without raw_record")
    frame_cap = nframes if nframes > 0 else -1
    vs = PixvqaMixedVideoSource(
        prepared["test_path"],
        prepared["ref_path"],
        ref_is_raw=bool(raw["reference_is_raw"]),
        test_is_raw=bool(raw["test_is_raw"]),
        display_photometry=display,
        frames=frame_cap,
        fps=float(raw["fps_requested"]),
    )
    metric_kwargs: dict = {
        "display_name": display,
        "device": torch.device(device),
        "temp_padding": "replicate",
        "quiet": True,
    }
    if gpu_mem_gb is not None:
        metric_kwargs["gpu_mem"] = gpu_mem_gb
    metric = pycvvdp.cvvdp(**metric_kwargs)

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            q_pred, stats = metric.predict_video_source(vs)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record = _failure_record(
            prepared, interpretation, display, nframes, device, gpu_mem_gb, elapsed, torch, str(exc),
        )
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise AdapterError(f"CVVDP mixed-input Python run failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    jod = float(q_pred.item())
    write_official_csv(csv_path, prepared["test_path"], prepared["ref_path"], jod)
    if features_dir is not None:
        if stats is None:
            raise AdapterError("CVVDP returned no stats for --temporal / QCS features")
        dest = features_path(features_dir, prepared["test_path"])
        metric.write_features_to_json(stats, str(dest))

    record = {
        "adapter": "pixvqa_cvvdp",
        "backend": "pycvvdp.predict_video_source",
        "upstream": PINNED_UPSTREAM,
        "upstream_commit": PINNED_COMMIT,
        "upstream_version_expected": PINNED_VERSION,
        "runtime": runtime_metadata(),
        "cvvdp_executable": None,
        "command": None,
        "interpretation": interpretation,
        "display": display,
        "display_model": prepared["display_model"],
        "device": device,
        "device_requested": prepared.get("device_requested", device),
        "gpu_mem_gb": gpu_mem_gb,
        "device_logged_by_cvvdp": device,
        "device_info": device_record(prepared.get("device_requested", device), device, torch),
        "nframes_requested": nframes,
        "frames_evaluated": int(vs.frames),
        "reference": prepared["ref_meta"],
        "test": prepared["test_meta"],
        "exit_code": 0,
        "elapsed_seconds": elapsed,
        "performance": {
            "predict_video_source_seconds": elapsed,
            "peak_child_rss_bytes": peak_rss_bytes(children=False),
            "peak_cuda_allocated_bytes": None,
            "note": "Mixed input uses in-process predict_video_source; no cvvdp subprocess.",
        },
        "stdout": f"cvvdp={jod:.17g}\n",
        "stderr": f"[INFO] Running on device: {device}\n",
        "raw_yuv": raw,
        "jod": jod,
        "result_csv": str(csv_path),
    }
    return record


def write_official_csv(csv_path: Path, test_path: str, ref_path: str, jod: float) -> None:
    # Preserve the official column names and order while quoting arbitrary paths.
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["test", "reference", "cvvdp"])
        writer.writerow([test_path, ref_path, str(jod)])


def _failure_record(prepared, interpretation, display, nframes, device, gpu_mem_gb, elapsed, torch, err):
    return {
        "adapter": "pixvqa_cvvdp",
        "backend": "pycvvdp.predict_video_source",
        "interpretation": interpretation,
        "display": display,
        "device": device,
        "device_requested": prepared.get("device_requested", device),
        "gpu_mem_gb": gpu_mem_gb,
        "device_info": device_record(prepared.get("device_requested", device), device, torch),
        "nframes_requested": nframes,
        "reference": prepared["ref_meta"],
        "test": prepared["test_meta"],
        "exit_code": 1,
        "elapsed_seconds": elapsed,
        "stderr": err,
        "raw_yuv": prepared.get("raw_record"),
        "command": None,
    }
