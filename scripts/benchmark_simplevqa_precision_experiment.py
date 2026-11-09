#!/usr/bin/env python3
"""Isolated SimpleVQA FP16/BF16 experiment — not shipped; FP32 adapter default unchanged.

Stages the validated CUDA streaming overlay from pixvqa_simplevqa.adapter, then
applies a temporary inference-only autocast patch in a private temp directory.
Records FP32 reference scores and compares slowfast-only, regressor-only, and
full-model autocast against FP32 on authorized SDR clips.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "pr22_validation_20261109"
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_simplevqa.adapter import (  # noqa: E402
    SetupError,
    cuda_single_decode_overlay_test_demo,
    inspect_video,
    parse_score,
)

PYTHON = Path("/home/sergio/pixvqa-simplevqa-env/bin/python")
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
if not NVIDIA_SMI.is_file():
    NVIDIA_SMI = Path("nvidia-smi")

GNU_TIME = Path("/usr/bin/time")
LOCK_FILE = REPORT_DIR / ".precision_benchmark.lock"

MAX_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

DEFAULT_REPO = Path("/mnt/d/projects/pixvqa-public/deps/SimpleVQA")
DEFAULT_CHECKPOINT = DEFAULT_REPO / "ckpts" / "UGC_BVQA_model.pth"
DEFAULT_CLIPS = {
    "official_8s_960x540": DEFAULT_REPO / "videos" / "2999049224_original_centercrop_960x540_8s.mp4",
    "basketball_10s_1080p50": Path(
        "/mnt/d/projects/pixvqa-public/BasketballDrive_1920x1080_50_F0-499_420_10bit_"
        "sdr_bt709l_yuv_tbr10000_ip60_bfauto_420_10b_auto_hq_hrd2000_aqs0_aqt0_"
        "laauto_sg0_brefauto_wpauto_hevc_nvenc_4070_sdk13.mp4"
    ),
    "basketball_60s_loop": REPORT_DIR / "clips" / "BasketballDrive_1080p50_60s_loop6x_streamcopy.mp4",
}

# Inference loop in streaming overlay (must match adapter output exactly).
_INFERENCE_ANCHOR_START = "        with torch.no_grad():\n            model.eval()"
_INFERENCE_ANCHOR_END = "            y_val = outputs.item()"

_FP32_INFERENCE = """        with torch.no_grad():
            model.eval()
         
            video_dist_spatial = video_dist_spatial.to(device)
            video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)
            
            n_clip = len(video_dist_motion)
            feature_motion = torch.zeros([n_clip, 2048+256])
            
            for idx, ele in enumerate(video_dist_motion):
                ele = ele.unsqueeze(dim=0)
                ele = ele.permute(0, 2, 1, 3, 4)
                ele = pack_pathway_output(ele, device)
                ele_slow_feature, ele_fast_feature = model_motion(ele)

                ele_slow_feature = ele_slow_feature.squeeze()
                ele_fast_feature = ele_fast_feature.squeeze()

                ele_feature_motion = torch.cat([ele_slow_feature, ele_fast_feature])
                ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

                feature_motion[idx] = ele_feature_motion

            feature_motion = feature_motion.unsqueeze(dim=0)

            outputs = model(video_dist_spatial, feature_motion)
            
            y_val = outputs.item()"""

_SLOWFAST_ONLY = """        with torch.no_grad():
            model.eval()
            _ac_dtype = torch.{dtype}
         
            video_dist_spatial = video_dist_spatial.to(device)
            video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)
            
            n_clip = len(video_dist_motion)
            feature_motion = torch.zeros([n_clip, 2048+256])
            
            for idx, ele in enumerate(video_dist_motion):
                ele = ele.unsqueeze(dim=0)
                ele = ele.permute(0, 2, 1, 3, 4)
                ele = pack_pathway_output(ele, device)
                with torch.autocast(device_type='cuda', dtype=_ac_dtype):
                    ele_slow_feature, ele_fast_feature = model_motion(ele)
                ele_slow_feature = ele_slow_feature.squeeze().float()
                ele_fast_feature = ele_fast_feature.squeeze().float()

                ele_feature_motion = torch.cat([ele_slow_feature, ele_fast_feature])
                ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

                feature_motion[idx] = ele_feature_motion

            feature_motion = feature_motion.unsqueeze(dim=0)

            outputs = model(video_dist_spatial, feature_motion)
            
            y_val = outputs.item()"""

_REGRESSOR_ONLY = """        with torch.no_grad():
            model.eval()
            _ac_dtype = torch.{dtype}
         
            video_dist_spatial = video_dist_spatial.to(device)
            video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)
            
            n_clip = len(video_dist_motion)
            feature_motion = torch.zeros([n_clip, 2048+256])
            
            for idx, ele in enumerate(video_dist_motion):
                ele = ele.unsqueeze(dim=0)
                ele = ele.permute(0, 2, 1, 3, 4)
                ele = pack_pathway_output(ele, device)
                ele_slow_feature, ele_fast_feature = model_motion(ele)

                ele_slow_feature = ele_slow_feature.squeeze()
                ele_fast_feature = ele_fast_feature.squeeze()

                ele_feature_motion = torch.cat([ele_slow_feature, ele_fast_feature])
                ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

                feature_motion[idx] = ele_feature_motion

            feature_motion = feature_motion.unsqueeze(dim=0)

            with torch.autocast(device_type='cuda', dtype=_ac_dtype):
                outputs = model(video_dist_spatial, feature_motion)
            
            y_val = outputs.item()"""

_FULL_MODEL = """        with torch.no_grad():
            model.eval()
            _ac_dtype = torch.{dtype}
            with torch.autocast(device_type='cuda', dtype=_ac_dtype):
                video_dist_spatial = video_dist_spatial.to(device)
                video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)
                
                n_clip = len(video_dist_motion)
                feature_motion = torch.zeros([n_clip, 2048+256])
                
                for idx, ele in enumerate(video_dist_motion):
                    ele = ele.unsqueeze(dim=0)
                    ele = ele.permute(0, 2, 1, 3, 4)
                    ele = pack_pathway_output(ele, device)
                    ele_slow_feature, ele_fast_feature = model_motion(ele)

                    ele_slow_feature = ele_slow_feature.squeeze().float()
                    ele_fast_feature = ele_fast_feature.squeeze().float()

                    ele_feature_motion = torch.cat([ele_slow_feature, ele_fast_feature])
                    ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

                    feature_motion[idx] = ele_feature_motion

                feature_motion = feature_motion.unsqueeze(dim=0)

                outputs = model(video_dist_spatial, feature_motion)
            
            y_val = outputs.item()"""


@dataclass(frozen=True)
class PrecisionMode:
    name: str
    scope: str  # fp32 | slowfast_only | regressor_only | full
    dtype: str  # float32 | float16 | bfloat16


MODES: tuple[PrecisionMode, ...] = (
    PrecisionMode("fp32_baseline", "fp32", "float32"),
    PrecisionMode("slowfast_fp16", "slowfast_only", "float16"),
    PrecisionMode("slowfast_bf16", "slowfast_only", "bfloat16"),
    PrecisionMode("regressor_fp16", "regressor_only", "float16"),
    PrecisionMode("regressor_bf16", "regressor_only", "bfloat16"),
    PrecisionMode("full_fp16", "full", "float16"),
    PrecisionMode("full_bf16", "full", "bfloat16"),
)


def available_ram_mib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0.0


def extract_inference_block(text: str) -> str:
    start = text.find(_INFERENCE_ANCHOR_START)
    end = text.find(_INFERENCE_ANCHOR_END, start)
    if start < 0 or end < 0:
        raise SetupError("streaming overlay inference block not found; adapter layout changed?")
    return text[start : end + len(_INFERENCE_ANCHOR_END)]


def apply_precision_patch(base_text: str, mode: PrecisionMode) -> str:
    current = extract_inference_block(base_text)
    if mode.scope == "fp32":
        replacement = _FP32_INFERENCE
    elif mode.scope == "slowfast_only":
        replacement = _SLOWFAST_ONLY.format(dtype=mode.dtype)
    elif mode.scope == "regressor_only":
        replacement = _REGRESSOR_ONLY.format(dtype=mode.dtype)
    elif mode.scope == "full":
        replacement = _FULL_MODEL.format(dtype=mode.dtype)
    else:
        raise ValueError(f"unknown scope: {mode.scope}")
    if current not in base_text:
        raise SetupError("could not locate inference block for replacement")
    return base_text.replace(current, replacement, 1)


class GpuSampler:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [str(NVIDIA_SMI), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                used = int(out.split("\n")[0])
                self.peak_mib = max(self.peak_mib, used)
            except (subprocess.CalledProcessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.peak_mib


def run_mode(
    *,
    repo: Path,
    checkpoint: Path,
    video: Path,
    mode: PrecisionMode,
    timeout: int,
    staging_root: Path,
) -> dict:
    upstream_demo = repo / "test_demo.py"
    overlay = cuda_single_decode_overlay_test_demo(upstream_demo)
    overlay = apply_precision_patch(overlay, mode)

    work = staging_root / mode.name
    work.mkdir(parents=True, exist_ok=True)
    (work / "test_demo.py").write_text(overlay, encoding="utf-8")
    (work / "model").symlink_to(repo / "model", target_is_directory=True)
    (work / "ckpts").mkdir(exist_ok=True)
    (work / "ckpts" / "UGC_BVQA_model.pth").symlink_to(checkpoint)
    result_path = work / "result.txt"
    time_path = work / "time.txt"

    env = os.environ.copy()
    env.pop("PIXVQA_SIMPLEVQA_LEGACY_BOUNDED", None)
    env.pop("PIXVQA_SIMPLEVQA_PRECISION_EXPERIMENT", None)

    cmd = [
        str(PYTHON),
        "test_demo.py",
        "--method_name",
        "single-scale",
        "--dist",
        str(video),
        "--output",
        str(result_path),
        "--is_gpu",
    ]
    if not GNU_TIME.is_file():
        raise RuntimeError("/usr/bin/time required for peak RSS measurement")
    wrapped = [str(GNU_TIME), "-v", "-o", str(time_path), *cmd]

    gpu = GpuSampler()
    gpu.start()
    started = time.monotonic()
    record: dict = {
        "mode": mode.name,
        "scope": mode.scope,
        "dtype": mode.dtype,
        "status": "ok",
    }
    try:
        proc = subprocess.run(
            wrapped,
            cwd=work,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        record["wall_seconds"] = time.monotonic() - started
        record["gpu_vram_peak_mib"] = gpu.stop()
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if len(err) > 2000:
                err = err[-2000:]
            record["status"] = "failed"
            record["error"] = err or f"exit {proc.returncode}"
            return record
        if not result_path.is_file():
            record["status"] = "failed"
            record["error"] = "no result file"
            return record
        score = parse_score(result_path.read_text(encoding="utf-8"), video)
        record["quality_score"] = score
        rss_text = time_path.read_text(encoding="utf-8", errors="replace")
        rss_match = MAX_RSS_RE.search(rss_text)
        if rss_match is None:
            record["status"] = "failed"
            record["error"] = "GNU time did not report peak RSS"
            return record
        record["peak_rss_mib"] = int(rss_match.group(1)) / 1024
    except subprocess.TimeoutExpired as exc:
        record["status"] = "failed"
        record["error"] = f"timeout after {timeout}s: {(exc.stderr or '')[-500:]}"
        record["wall_seconds"] = time.monotonic() - started
        record["gpu_vram_peak_mib"] = gpu.stop()
    return record


def acquire_lock() -> object:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SystemExit(
            "Another precision benchmark is running (lock: "
            f"{LOCK_FILE}). Refusing concurrent execution."
        ) from exc
    handle.write(f"pid={os.getpid()} started={time.time()}\n")
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--clips",
        nargs="*",
        choices=tuple(DEFAULT_CLIPS.keys()),
        default=("official_8s_960x540", "basketball_10s_1080p50", "basketball_60s_loop"),
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "precision_benchmark.json")
    args = parser.parse_args()

    lock = acquire_lock()
    try:
        repo = args.repo.resolve()
        checkpoint = args.checkpoint.resolve()
        if not (repo / "test_demo.py").is_file() or not checkpoint.is_file():
            parser.error("SimpleVQA repo/checkpoint missing")

        ram_avail = available_ram_mib()
        if ram_avail < 2048:
            raise SystemExit(f"MemAvailable too low ({ram_avail:.0f} MiB); refusing run")

        upstream_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()

        clip_paths: dict[str, Path] = {}
        for name in args.clips:
            path = DEFAULT_CLIPS[name].resolve()
            if not path.is_file():
                parser.error(f"clip missing: {path}")
            inspect_video(path)
            clip_paths[name] = path

        all_results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="pixvqa_svqa_prec_") as tmp:
            staging = Path(tmp)
            for clip_name, video in clip_paths.items():
                clip_runs: list[dict] = []
                fp32_score: float | None = None
                for mode in MODES:
                    print(f"[{clip_name}] running {mode.name}...", flush=True)
                    run = run_mode(
                        repo=repo,
                        checkpoint=checkpoint,
                        video=video,
                        mode=mode,
                        timeout=args.timeout,
                        staging_root=staging / clip_name,
                    )
                    if run.get("status") == "ok":
                        if mode.name == "fp32_baseline":
                            fp32_score = float(run["quality_score"])
                        elif fp32_score is not None:
                            run["abs_error_vs_fp32"] = abs(float(run["quality_score"]) - fp32_score)
                            run["within_tolerance"] = run["abs_error_vs_fp32"] <= args.tolerance
                    clip_runs.append(run)
                    time.sleep(1)  # let GPU memory settle between runs

                entry = {
                    "clip": clip_name,
                    "video": str(video),
                    "fp32_reference_score": fp32_score,
                    "runs": clip_runs,
                }
                if fp32_score is not None:
                    ok_scores = [
                        r
                        for r in clip_runs
                        if r.get("status") == "ok" and r.get("mode") != "fp32_baseline"
                    ]
                    if ok_scores:
                        entry["max_abs_error_vs_fp32"] = max(
                            float(r["abs_error_vs_fp32"]) for r in ok_scores
                        )
                        entry["modes_within_tolerance"] = sum(
                            1 for r in ok_scores if r.get("within_tolerance")
                        )
                        entry["modes_tested_ok"] = len(ok_scores)
                all_results.append(entry)

        summary = {
            "validation_date": "2026-11-09",
            "experiment": "simplevqa_fp16_bf16_autocast",
            "gpu": "NVIDIA GeForce RTX 5080",
            "python": str(PYTHON),
            "upstream_repo": str(repo),
            "upstream_commit": upstream_commit,
            "overlay": "cuda-streaming-motion-overlay + temp inference autocast patch",
            "adapter_default_unchanged": True,
            "methodology": {
                "fp32_baseline": "streaming overlay, no autocast (matches production default path)",
                "autocast_scopes": ["slowfast_only", "regressor_only", "full"],
                "dtypes": ["float16", "bfloat16"],
                "slowfast_outputs": "cast to float32 before feature concat when scope=slowfast_only",
                "peak_rss": "GNU /usr/bin/time -v",
                "gpu_vram": f"{NVIDIA_SMI} memory.used polled 250ms (global)",
                "execution": "sequential; lock file prevents concurrent jobs",
                "tolerance": args.tolerance,
            },
            "mem_available_mib_at_start": ram_avail,
            "clips": all_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
