#!/usr/bin/env python3
"""Compare pinned upstream SimpleVQA against PixVQA CUDA memory optimizations.

Run from PixVQA checkout with the learned .venv Python. All three variants use
the same reference checkpoint, SlowFast cache, input files and FP32 inference.
The experimental streaming variant is opt-in and must pass score parity before
promotion. GNU time reports peak RSS for its command and descendants.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_simplevqa.adapter import SetupError, inspect_video  # noqa: E402

MAX_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def invoke(command: list[str], *, cwd: Path, environment: dict[str, str],
           timeout: int, label: str, output: Path) -> dict:
    time_file = cwd / (label + "_time.txt")
    timer = Path("/usr/bin/time")
    if not timer.is_file():
        raise RuntimeError("/usr/bin/time (GNU time) is required to profile peak RSS")
    wrapped = [str(timer), "-v", "-o", str(time_file), *command]
    started = time.monotonic()
    try:
        result = subprocess.run(wrapped, cwd=cwd, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label}: timed out after {timeout}s") from exc
    duration = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(
            f"{label}: exited {result.returncode}\n"
            f"stdout tail: {result.stdout[-1000:]}\n"
            f"stderr tail: {result.stderr[-2000:]}"
        )
    resource_text = time_file.read_text(encoding="utf-8", errors="replace")
    rss_match = MAX_RSS_RE.search(resource_text)
    if rss_match is None:
        raise RuntimeError(f"{label}: GNU time did not report maximum RSS")
    if label == "upstream":
        record = output.read_text(encoding="utf-8").strip().rsplit(",", 1)
        if len(record) != 2:
            raise RuntimeError("upstream score CSV missing or malformed")
        score = float(record[1])
    else:
        try:
            record = json.loads(result.stdout.strip().splitlines()[-1])
            score = float(record["quality_score"])
        except (ValueError, KeyError, IndexError) as exc:
            raise RuntimeError(f"{label}: unable to read SimpleVQA result JSON") from exc
    return {
        "quality_score": score,
        "wall_seconds": duration,
        "peak_rss_mib": int(rss_match.group(1)) / 1024,
        "runner": label,
    }


def evaluate(video: Path, *, repo: Path, checkpoint: Path, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="pixvqa_svqa_parity_") as tmp:
        base = Path(tmp)
        upstream = base / "upstream"
        upstream.mkdir()
        (upstream / "test_demo.py").symlink_to(repo / "test_demo.py")
        (upstream / "model").symlink_to(repo / "model", target_is_directory=True)
        (upstream / "ckpts").mkdir()
        (upstream / "ckpts" / "UGC_BVQA_model.pth").symlink_to(checkpoint)
        score_file = upstream / "score.csv"
        common = os.environ.copy()
        upstream_cmd = [
            sys.executable, "test_demo.py", "--method_name", "single-scale",
            "--dist", str(video), "--output", str(score_file), "--is_gpu",
        ]
        upstream_result = invoke(upstream_cmd, cwd=upstream, environment=common,
                                 timeout=timeout, label="upstream", output=score_file)
        module_env = common.copy()
        module_env["PYTHONPATH"] = str(ROOT / "python") + (
            os.pathsep + module_env["PYTHONPATH"] if module_env.get("PYTHONPATH") else ""
        )
        cmd = [
            sys.executable, "-m", "pixvqa_simplevqa",
            "--repo", str(repo), "--checkpoint", str(checkpoint),
            "--test", str(video), "--device", "cuda",
        ]
        module_env.pop("PIXVQA_SIMPLEVQA_EXPERIMENTAL_STREAMING", None)
        bounded = invoke(cmd, cwd=base, environment=module_env,
                         timeout=timeout, label="bounded", output=score_file)
        module_env["PIXVQA_SIMPLEVQA_EXPERIMENTAL_STREAMING"] = "1"
        streaming = invoke(cmd, cwd=base, environment=module_env,
                           timeout=timeout, label="streaming", output=score_file)
        for candidate in (bounded, streaming):
            candidate["abs_score_delta_vs_upstream"] = abs(
                candidate["quality_score"] - upstream_result["quality_score"]
            )
            candidate["rss_reduction_percent_vs_upstream"] = (
                100 * (1 - candidate["peak_rss_mib"] / upstream_result["peak_rss_mib"])
            )
        return {"video": str(video), "upstream": upstream_result,
                "bounded": bounded, "streaming": streaming}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT / "deps" / "SimpleVQA")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "deps" / "SimpleVQA" /
                        "ckpts" / "UGC_BVQA_model.pth")
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-abs-delta", type=float, default=1e-4,
                        help="Initial engineering tolerance; adjust only with justification.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_abs_delta < 0:
        parser.error("timeout must be positive and max-abs-delta nonnegative")
    repo, checkpoint = args.repo.resolve(), args.checkpoint.resolve()
    if not (repo / "test_demo.py").is_file() or not checkpoint.is_file():
        parser.error("SimpleVQA upstream checkout and checkpoint must exist")
    results = []
    for raw in args.videos:
        video = raw.resolve()
        if not video.is_file():
            parser.error(f"video not found: {video}")
        try:
            inspect_video(video)
        except SetupError as exc:
            parser.error(f"{video}: preflight failed ({exc})")
        results.append(evaluate(video, repo=repo, checkpoint=checkpoint,
                                timeout=args.timeout))
    worst = max(
        entry[mode]["abs_score_delta_vs_upstream"]
        for entry in results for mode in ("bounded", "streaming")
    )
    summary = {"results": results, "max_abs_score_delta": worst,
               "tolerance": args.max_abs_delta,
               "within_tolerance": worst <= args.max_abs_delta,
               "notes": "RSS is from GNU time; GPU VRAM must be measured separately. "
                        "Repeat runs for stable memory/runtime comparisons."}
    result_json = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)
    return 0 if summary["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
