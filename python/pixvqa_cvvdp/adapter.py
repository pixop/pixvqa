"""Run the official cvvdp CLI and collect numerical and runtime metadata."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pixvqa_cvvdp import (
    DISPLAY_MODELS,
    PINNED_COMMIT,
    PINNED_UPSTREAM,
    PINNED_VERSION,
    PINNED_TORCH_VERSION,
    PINNED_TORCHVISION_VERSION,
    PQ_DISPLAYS,
    SDR_DISPLAYS,
)

FFPROBE = os.environ.get("PIXVQA_CVVDP_FFPROBE", "ffprobe")


class AdapterError(RuntimeError):
    """A preflight or official-CVVDP failure."""


def installed_distribution(*names: str) -> dict:
    for name in names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        direct_url = None
        raw_direct_url = distribution.read_text("direct_url.json")
        if raw_direct_url:
            try:
                direct_url = json.loads(raw_direct_url)
            except json.JSONDecodeError:
                direct_url = {"invalid": True}
        return {"distribution": distribution.metadata["Name"], "version": distribution.version,
                "direct_url": direct_url}
    return {"distribution": None, "version": None, "direct_url": None}


def runtime_metadata() -> dict:
    cvvdp = installed_distribution("cvvdp", "ColorVideoVDP")
    torch = installed_distribution("torch")
    torchvision = installed_distribution("torchvision")
    commit = None
    if isinstance(cvvdp["direct_url"], dict):
        commit = cvvdp["direct_url"].get("vcs_info", {}).get("commit_id")
    version_matches = cvvdp["version"] == PINNED_VERSION
    commit_verified = commit == PINNED_COMMIT
    return {
        "cvvdp": {
            **cvvdp, "expected_version": PINNED_VERSION, "version_matches_expected": version_matches,
            "expected_commit": PINNED_COMMIT, "actual_commit": commit,
            "commit_verified": commit_verified,
            "commit_note": None if commit_verified else
            "The installed package did not provide a matching VCS commit in direct_url.json.",
        },
        "torch": {**torch, "expected_version": PINNED_TORCH_VERSION,
                  "version_matches_expected": torch["version"] == PINNED_TORCH_VERSION},
        "torchvision": {**torchvision, "expected_version": PINNED_TORCHVISION_VERSION,
                        "version_matches_expected": torchvision["version"] == PINNED_TORCHVISION_VERSION},
    }


def probe_video(path: str) -> dict:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,color_transfer,color_primaries,"
        "color_space,color_range,nb_frames,duration,r_frame_rate", "-of", "json", path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise AdapterError(f"could not execute ffprobe ({FFPROBE}): {exc}") from exc
    if proc.returncode != 0:
        raise AdapterError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    try:
        streams = json.loads(proc.stdout).get("streams") or []
    except json.JSONDecodeError as exc:
        raise AdapterError(f"ffprobe returned invalid JSON for {path}") from exc
    if not streams:
        raise AdapterError(f"no video stream in {path}")
    s = streams[0]
    return {
        "path": path, "codec": s.get("codec_name"), "profile": s.get("profile"),
        "width": int(s["width"]), "height": int(s["height"]), "pix_fmt": s.get("pix_fmt"),
        "color_transfer": s.get("color_transfer") or "unspecified",
        "color_primaries": s.get("color_primaries") or "unspecified",
        "color_space": s.get("color_space") or "unspecified",
        "color_range": s.get("color_range") or "unspecified",
        "nb_frames": s.get("nb_frames"), "duration": s.get("duration"),
        "r_frame_rate": s.get("r_frame_rate"),
    }


def check_pair(ref: dict, test: dict, interpretation: str, nframes: int) -> None:
    if (ref["width"], ref["height"]) != (test["width"], test["height"]):
        raise AdapterError("resolution mismatch between reference and test")
    if ref["r_frame_rate"] != test["r_frame_rate"]:
        raise AdapterError("frame rate mismatch between reference and test")
    if nframes < 0 and ref["nb_frames"] and test["nb_frames"] and ref["nb_frames"] != test["nb_frames"]:
        raise AdapterError("frame count mismatch between reference and test")
    for label, meta in (("reference", ref), ("test", test)):
        if interpretation == "pq":
            if meta["color_transfer"] != "smpte2084" or meta["color_primaries"] != "bt2020":
                raise AdapterError(
                    f"{label} is not explicitly BT.2020 PQ; PQ is not inferred from bit depth, "
                    "resolution, or filename"
                )
        elif interpretation == "sdr":
            if meta["color_transfer"] not in ("bt709", "iec61966-2-1"):
                raise AdapterError(f"{label} is not explicitly SDR (bt709 or iec61966-2-1)")
        else:
            raise AdapterError(f"unsupported interpretation {interpretation!r}; use sdr or pq")


def check_display(interpretation: str, display: str) -> dict:
    allowed = PQ_DISPLAYS if interpretation == "pq" else SDR_DISPLAYS
    if display not in allowed:
        raise AdapterError(
            f"display {display!r} is not allowed for {interpretation}; allowed: {', '.join(allowed)}"
        )
    return DISPLAY_MODELS[display]


def parse_result_csv(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    if len(rows) < 2:
        raise AdapterError(f"result CSV has no cvvdp score: {path}")
    # Official CVVDP 0.5.7 writes "test, reference, cvvdp". Skip the spaces
    # after commas, then strip so a padded header still matches.
    header = [cell.strip() for cell in rows[0]]
    if "cvvdp" not in header:
        raise AdapterError(f"result CSV has no cvvdp score: {path}")
    return float(rows[1][header.index("cvvdp")].strip())


def run(ref: str, test: str, interpretation: str, display: str, nframes: int, device: str,
        output_json: str, cvvdp_bin: str | None = None) -> dict:
    ref_meta, test_meta = probe_video(ref), probe_video(test)
    check_pair(ref_meta, test_meta, interpretation, nframes)
    display_model = check_display(interpretation, display)
    binary = cvvdp_bin or shutil.which("cvvdp")
    if not binary:
        raise AdapterError("cvvdp executable not found; see docs/COLORVIDEOVDP.md")
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    cmd = [
        binary, "--test", test, "--ref", ref, "--display", display, "--device", device,
        "--metric", "cvvdp", "--temp-padding", "replicate", "--result", str(csv_path),
    ]
    if nframes > 0:
        cmd.extend(["--nframes", str(nframes)])
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise AdapterError(f"could not execute cvvdp ({binary}): {exc}") from exc
    runtime = runtime_metadata()
    record = {
        "adapter": "pixvqa_cvvdp", "upstream": PINNED_UPSTREAM, "upstream_commit": PINNED_COMMIT,
        "upstream_version_expected": PINNED_VERSION, "runtime": runtime, "cvvdp_executable": binary, "command": cmd,
        "interpretation": interpretation, "display": display, "display_model": display_model,
        "device": device, "nframes_requested": nframes, "reference": ref_meta, "test": test_meta,
        "exit_code": proc.returncode, "elapsed_seconds": time.perf_counter() - t0,
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise AdapterError(f"cvvdp exited {proc.returncode}; see {out}")
    if not csv_path.exists():
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise AdapterError(f"cvvdp wrote no result CSV; see {out}")
    record["jod"] = parse_result_csv(csv_path)
    record["result_csv"] = str(csv_path)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
