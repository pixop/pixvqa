"""Run the official cvvdp CLI and collect numerical and runtime metadata."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pixvqa_device import (
    DEVICE_CHOICES, DeviceError, auto_notice, device_record, peak_rss_bytes, resolve_device,
)
from pixvqa_cvvdp import (
    DISPLAY_MODELS,
    PINNED_COMMIT,
    PINNED_UPSTREAM,
    PINNED_VERSION,
    PINNED_TORCH_VERSION,
    PINNED_TORCHVISION_VERSION,
    HLG_DISPLAYS,
    PQ_DISPLAYS,
    SDR_DISPLAYS,
)

FFPROBE = os.environ.get("PIXVQA_CVVDP_FFPROBE", "ffprobe")


class AdapterError(RuntimeError):
    """A preflight or official-CVVDP failure."""


class PreflightError(AdapterError):
    """Inputs, display, or the cvvdp executable were rejected before scoring started."""


def frames_evaluated(test_meta: dict, nframes: int) -> int | None:
    """Frames the official run covers: --nframes caps the test stream length."""
    available = test_meta.get("nb_frames")
    available = int(available) if isinstance(available, str) and available.isdigit() else None
    if nframes > 0:
        return nframes if available is None else min(nframes, available)
    return available


def resolve_cvvdp_binary(cvvdp_bin: str | None) -> str:
    if cvvdp_bin:
        if not (os.path.isfile(cvvdp_bin) and os.access(cvvdp_bin, os.X_OK)):
            raise PreflightError(f"could not execute cvvdp ({cvvdp_bin}): not an executable file")
        return cvvdp_bin
    # The entry point installed with this interpreter matches its pycvvdp and torch.
    for name in ("cvvdp.exe", "cvvdp") if os.name == "nt" else ("cvvdp",):
        beside = Path(sys.executable).parent / name
        if beside.is_file() and os.access(beside, os.X_OK):
            return str(beside)
    binary = shutil.which("cvvdp")
    if not binary:
        raise PreflightError("cvvdp executable not found; see docs/COLORVIDEOVDP.md")
    return binary


def check_device(requested: str, report_auto_fallback: bool) -> str:
    """Resolve auto here: upstream get_best_device turns an unavailable cuda into cpu."""
    if requested not in DEVICE_CHOICES:
        raise PreflightError(f"unsupported device {requested!r}; use auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception as exc:
        raise PreflightError(f"CVVDP needs torch in this interpreter: {exc}") from exc
    try:
        effective = resolve_device(requested, torch)
    except DeviceError as exc:
        raise PreflightError(str(exc)) from exc
    notice = auto_notice(requested, effective)
    if notice and report_auto_fallback:
        print(f"pixvqa-cvvdp: {notice}", file=sys.stderr)
    return effective


def logged_device(stderr: str) -> str | None:
    """Official run_cvvdp logs "[INFO] Running on device: X" unless --quiet."""
    found = re.findall(r"Running on device: (\S+)", stderr)
    return found[-1] if found else None


def check_temporal_runtime() -> None:
    """The frame sequence reuses pycvvdp's pooling in this interpreter."""
    try:
        import pycvvdp  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        raise PreflightError(f"CVVDP QCS needs pycvvdp and torch in this interpreter: {exc}") from exc


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
        elif interpretation == "hlg":
            if (meta["color_transfer"] != "arib-std-b67" or meta["color_primaries"] != "bt2020"
                    or meta["color_space"] != "bt2020nc" or meta["color_range"] not in ("tv", "mpeg")):
                raise AdapterError(
                    f"{label} is not explicitly limited-range BT.2020 HLG "
                    "(arib-std-b67, bt2020, bt2020nc, tv); HLG is not inferred from PQ or SDR tags"
                )
        elif interpretation == "sdr":
            if meta["color_transfer"] not in ("bt709", "iec61966-2-1"):
                raise AdapterError(f"{label} is not explicitly SDR (bt709 or iec61966-2-1)")
        else:
            raise AdapterError(f"unsupported interpretation {interpretation!r}; use sdr, pq, or hlg")


def check_display(interpretation: str, display: str) -> dict:
    allowed = {"pq": PQ_DISPLAYS, "hlg": HLG_DISPLAYS}.get(interpretation, SDR_DISPLAYS)
    if display not in allowed:
        raise AdapterError(
            f"display {display!r} is not allowed for {interpretation}; allowed: {', '.join(allowed)}"
        )
    return DISPLAY_MODELS[display]


def frame_bytes_yuv420p10le(width: int, height: int) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise AdapterError("YUV420 requires positive even width and height")
    return width * height * 3


def encode_fps(num: int, den: int) -> float:
    if num <= 0 or den <= 0:
        raise AdapterError("fps must be a positive rational")
    requested = num / den
    encoded = float(int(round(requested))) if round(requested) == requested else round(requested, 3)
    if abs(requested - encoded) > 5e-4:
        raise AdapterError(
            f"fps {num}/{den} cannot be represented by the official YUV filename (3 decimal places)"
        )
    return encoded


def fps_token(encoded: float) -> str:
    if float(encoded).is_integer():
        return str(int(encoded))
    return f"{encoded:.3f}".rstrip("0").rstrip(".")


def yuv_metadata_name(role: str, width: int, height: int, encoded_fps: float) -> str:
    if role not in ("reference", "test"):
        raise AdapterError(f"unsupported raw role {role!r}")
    return f"{role}_{width}x{height}_10b_420_2020_{fps_token(encoded_fps)}fps.yuv"


def strict_decode_video_props(fname: str) -> dict:
    """Parse official YUV filename tokens without accepting upstream defaults."""
    fields = os.path.splitext(os.path.basename(fname))[0].split("_")
    found: dict = {}
    resolution = re.compile(r"(\d+)x(\d+)p?(\d+)?$")
    for field in fields:
        lower = field.lower()
        match = resolution.fullmatch(field)
        if match:
            found["width"] = int(match.group(1))
            found["height"] = int(match.group(2))
            if match.group(3):
                found["fps"] = float(match.group(3))
            continue
        if lower.endswith("fps"):
            found["fps"] = float(field[:-3])
            continue
        if field in ("444", "420", "422"):
            found["chroma_ss"] = field
            continue
        if lower in ("444p", "420p", "422p", "444i", "420i", "422i"):
            found["chroma_ss"] = field[:-1]
            continue
        if field == "10" or lower in ("10b", "10bit"):
            found["bit_depth"] = 10
            continue
        if field == "8" or lower in ("8b", "8bit"):
            found["bit_depth"] = 8
            continue
        if field in ("2020", "709"):
            found["color_space"] = field
            continue
        if lower in ("bt709", "sdr"):
            found["color_space"] = "709"
            continue
        if lower in ("ct2020", "pq2020", "hdr", "pq", "hlg"):
            found["color_space"] = "2020"
            continue
    missing = [key for key in ("width", "height", "fps", "bit_depth", "color_space", "chroma_ss")
               if key not in found]
    if missing:
        raise AdapterError(
            "YUV filename is missing " + ", ".join(missing) + "; refusing upstream defaults"
        )
    return found


def load_upstream_decode_video_props():
    try:
        from pycvvdp.video_source_yuv import decode_video_props
    except ImportError:
        return None
    return decode_video_props


def props_match(decoded: dict, expected: dict) -> bool:
    try:
        return (
            int(decoded["width"]) == expected["width"]
            and int(decoded["height"]) == expected["height"]
            and int(decoded["bit_depth"]) == expected["bit_depth"]
            and str(decoded["chroma_ss"]) == expected["chroma_ss"]
            and str(decoded["color_space"]) == expected["color_space"]
            and abs(float(decoded["fps"]) - float(expected["fps"])) <= 1e-6
        )
    except (KeyError, TypeError, ValueError):
        return False


def verify_yuv_filename(path: str, expected: dict, decode_video_props=None) -> dict:
    parsed = strict_decode_video_props(path)
    if not props_match(parsed, expected):
        raise AdapterError(f"YUV filename metadata {parsed} does not match {expected}")
    decoder = load_upstream_decode_video_props() if decode_video_props is None else decode_video_props
    verified_by = "strict"
    upstream = None
    if decoder is not None:
        upstream = decoder(path)
        if not props_match(upstream, expected):
            raise AdapterError(
                f"pycvvdp decode_video_props returned {upstream}, expected {expected}"
            )
        verified_by = "strict+pycvvdp.decode_video_props"
    return {"parsed": parsed, "upstream": upstream, "verified_by": verified_by}


def samples_are_low_10bit(path: str) -> bool:
    """True when every sample is a little-endian uint16 in 0..1023."""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return True
            if len(chunk) % 2:
                return False
            if any(chunk[index] > 3 for index in range(1, len(chunk), 2)):
                return False


def validate_raw_yuv(path: str, width: int, height: int, nframes: int) -> int:
    if not os.path.isfile(path):
        raise AdapterError(f"raw YUV not found: {path}")
    frame_bytes = frame_bytes_yuv420p10le(width, height)
    size = os.path.getsize(path)
    if size <= 0 or size % frame_bytes:
        raise AdapterError(
            f"{path} is {size} bytes, not an exact positive number of "
            f"{width}x{height} YUV420p10le frames ({frame_bytes} bytes each)"
        )
    frames = size // frame_bytes
    if nframes > frames:
        raise AdapterError(f"{path} has {frames} frames, fewer than requested {nframes}")
    if not samples_are_low_10bit(path):
        raise AdapterError(
            f"{path} is not little-endian uint16 with 10-bit codes in the low bits (0..1023)"
        )
    return frames


class TemporarySymlinkDir:
    def __init__(self) -> None:
        self.path = tempfile.mkdtemp(prefix="pixvqa-cvvdp-yuv-")
        self.links: list[str] = []

    def symlink(self, source: str, name: str) -> str:
        destination = os.path.join(self.path, name)
        try:
            os.symlink(os.path.abspath(source), destination)
        except OSError as symlink_error:
            # Windows refuses symlinks without Developer Mode; a hard link on
            # the same volume needs no privilege. The YUV is never copied.
            try:
                os.link(os.path.abspath(source), destination)
            except OSError as link_error:
                raise AdapterError(
                    f"could not link {source} under {self.path} (symlink: {symlink_error}; "
                    f"hard link: {link_error}); set TMPDIR/TEMP to a directory on the same volume"
                ) from link_error
        self.links.append(destination)
        return destination

    def cleanup(self) -> None:
        for link in self.links:
            if os.path.lexists(link):
                os.unlink(link)
        if os.path.isdir(self.path):
            os.rmdir(self.path)


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


def raw_metadata(path: str, width: int, height: int, frames: int, num: int, den: int,
                 interpretation: str) -> dict:
    # The YUV filename carries the matrix (2020). The EOTF is the display model's:
    # standard_hdr_pq applies PQ, standard_hdr_hlg applies HLG. ffprobe spells
    # limited range "tv"; the raw reader expands limited-range codes.
    return {
        "path": path, "codec": "rawvideo", "pix_fmt": "yuv420p10le", "width": width, "height": height,
        "color_transfer": "arib-std-b67" if interpretation == "hlg" else "smpte2084",
        "color_primaries": "bt2020", "color_space": "bt2020nc",
        "color_range": "tv", "nb_frames": str(frames), "r_frame_rate": f"{num}/{den}",
        "storage": "little-endian uint16, 10-bit codes in the low 10 bits",
    }


def run(ref: str, test: str, interpretation: str, display: str, nframes: int, device: str,
        output_json: str, cvvdp_bin: str | None = None, raw_width: int | None = None,
        raw_height: int | None = None, raw_fps_num: int | None = None,
        raw_fps_den: int | None = None, decode_video_props=None, check_only: bool = False,
        temporal: bool = False, gpu_mem_gb: float | None = None) -> dict:
    prepared = None
    try:
        if gpu_mem_gb is not None and gpu_mem_gb <= 0:
            raise PreflightError("--gpu-mem expects a positive number of GB")
        # Preflight is intentionally quiet: execution emits the one auto->CPU
        # notice for this backend, after both preflights have passed.
        effective = check_device(device, report_auto_fallback=not check_only)
        try:
            prepared = prepare_inputs(ref, test, interpretation, display, nframes, cvvdp_bin, raw_width,
                                      raw_height, raw_fps_num, raw_fps_den, decode_video_props)
        except PreflightError:
            raise
        except AdapterError as exc:
            raise PreflightError(str(exc)) from exc
        if check_only:
            if temporal:
                check_temporal_runtime()
            return {"check": "ok", "frames_evaluated": frames_evaluated(prepared["test_meta"], nframes),
                    "device": effective}
        prepared["device_requested"] = device
        prepared["gpu_mem_gb"] = gpu_mem_gb
        return execute(prepared, interpretation, display, nframes, effective, output_json, temporal)
    finally:
        if prepared is not None and prepared["links"] is not None:
            prepared["links"].cleanup()


def prepare_inputs(ref, test, interpretation, display, nframes, cvvdp_bin, raw_width, raw_height,
                   raw_fps_num, raw_fps_den, decode_video_props) -> dict:
    raw_values = (raw_width, raw_height, raw_fps_num, raw_fps_den)
    if any(value is not None for value in raw_values) and any(value is None for value in raw_values):
        raise PreflightError("raw YUV width, height, and fps must be supplied together")
    links = None
    raw_record = None
    try:
        if raw_width is None:
            ref_path, test_path = ref, test
            ref_meta, test_meta = probe_video(ref), probe_video(test)
        else:
            if interpretation not in ("pq", "hlg"):
                raise PreflightError(
                    "raw YUV SDR is unsupported; the official raw reader applies BT.601 coefficients "
                    "on its BT.709 path"
                )
            encoded_fps = encode_fps(raw_fps_num, raw_fps_den)
            ref_frames = validate_raw_yuv(ref, raw_width, raw_height, max(nframes, 0))
            test_frames = validate_raw_yuv(test, raw_width, raw_height, max(nframes, 0))
            links = TemporarySymlinkDir()
            expected = {
                "width": raw_width, "height": raw_height, "bit_depth": 10, "chroma_ss": "420",
                "color_space": "2020", "fps": encoded_fps,
            }
            ref_path = links.symlink(ref, yuv_metadata_name("reference", raw_width, raw_height, encoded_fps))
            test_path = links.symlink(test, yuv_metadata_name("test", raw_width, raw_height, encoded_fps))
            ref_verified = verify_yuv_filename(ref_path, expected, decode_video_props)
            test_verified = verify_yuv_filename(test_path, expected, decode_video_props)
            ref_meta = raw_metadata(ref, raw_width, raw_height, ref_frames, raw_fps_num, raw_fps_den,
                                    interpretation)
            test_meta = raw_metadata(test, raw_width, raw_height, test_frames, raw_fps_num, raw_fps_den,
                                     interpretation)
            raw_record = {
                "pix_fmt": "yuv420p10le", "range": "limited", "matrix": "bt2020nc",
                "transfer": interpretation, "width": raw_width, "height": raw_height,
                "fps_requested": raw_fps_num / raw_fps_den, "fps_encoded": encoded_fps,
                "storage": "little-endian uint16 with 10-bit codes in the low bits (0..1023)",
                "assumption": "Official YUVReader expands limited-range codes; full-range samples are not converted.",
                "reference_frames": ref_frames, "test_frames": test_frames,
                "reference_symlink": ref_path, "test_symlink": test_path,
                "filename_verification": ref_verified["verified_by"],
                "reference_props": ref_verified["parsed"], "test_props": test_verified["parsed"],
            }
        check_pair(ref_meta, test_meta, interpretation, nframes)
        display_model = check_display(interpretation, display)
        binary = resolve_cvvdp_binary(cvvdp_bin)
    except BaseException:
        if links is not None:
            links.cleanup()
        raise
    return {
        "links": links, "ref_path": ref_path, "test_path": test_path, "ref_meta": ref_meta,
        "test_meta": test_meta, "raw_record": raw_record, "display_model": display_model,
        "binary": binary,
    }


def execute(prepared: dict, interpretation: str, display: str, nframes: int, device: str,
            output_json: str, temporal: bool = False) -> dict:
    binary = prepared["binary"]
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    cmd = [
        binary, "--test", prepared["test_path"], "--ref", prepared["ref_path"], "--display", display,
        "--device", device, "--metric", "cvvdp", "--temp-padding", "replicate", "--result", str(csv_path),
    ]
    if prepared.get("gpu_mem_gb") is not None:
        cmd.extend(["--gpu-mem", repr(prepared["gpu_mem_gb"])])
    if nframes > 0:
        cmd.extend(["--nframes", str(nframes)])
    features_dir = None
    if temporal:
        # --features only serializes stats['Q_per_ch'] after the JOD is computed.
        features_dir = Path(tempfile.mkdtemp(prefix="pixvqa-cvvdp-features-"))
        cmd.extend(["--features", "--output-dir", str(features_dir)])
    try:
        return execute_command(prepared, cmd, csv_path, out, interpretation, display, nframes, device,
                               features_dir)
    finally:
        if features_dir is not None:
            shutil.rmtree(features_dir, ignore_errors=True)


def execute_command(prepared: dict, cmd: list, csv_path: Path, out: Path, interpretation: str,
                    display: str, nframes: int, device: str, features_dir: Path | None) -> dict:
    try:
        import torch
    except ImportError:
        torch = None
    binary = prepared["binary"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise AdapterError(f"could not execute cvvdp ({binary}): {exc}") from exc
    elapsed = time.perf_counter() - t0
    runtime = runtime_metadata()
    used = logged_device(proc.stderr)
    record = {
        "adapter": "pixvqa_cvvdp", "upstream": PINNED_UPSTREAM, "upstream_commit": PINNED_COMMIT,
        "upstream_version_expected": PINNED_VERSION, "runtime": runtime, "cvvdp_executable": binary,
        "command": cmd, "interpretation": interpretation, "display": display,
        "display_model": prepared["display_model"], "device": device,
        "device_requested": prepared.get("device_requested", device),
        # Upstream defines --gpu-mem as a decimal-Gigabyte cap. Null means its
        # normal free-memory heuristic was deliberately left unchanged.
        "gpu_mem_gb": prepared.get("gpu_mem_gb"),
        "device_logged_by_cvvdp": used,
        "device_info": device_record(prepared.get("device_requested", device), device, torch),
        "nframes_requested": nframes,
        "frames_evaluated": frames_evaluated(prepared["test_meta"], nframes),
        "reference": prepared["ref_meta"], "test": prepared["test_meta"],
        "exit_code": proc.returncode, "elapsed_seconds": elapsed,
        "performance": {
            "official_cli_seconds": elapsed,
            "peak_child_rss_bytes": peak_rss_bytes(children=True),
            "peak_cuda_allocated_bytes": None,
            "note": "The official CLI runs in a child process; its CUDA allocator peak is not visible here.",
        },
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
    if prepared["raw_record"] is not None:
        record["raw_yuv"] = prepared["raw_record"]
    if proc.returncode != 0:
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise AdapterError(f"cvvdp exited {proc.returncode}; see {out}")
    if used is None or used.split(":", 1)[0] != device:
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise AdapterError(f"cvvdp reported device {used!r}, not {device}; see {out}")
    if not csv_path.exists():
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise AdapterError(f"cvvdp wrote no result CSV; see {out}")
    record["jod"] = parse_result_csv(csv_path)
    record["result_csv"] = str(csv_path)
    if features_dir is not None:
        from pixvqa_cvvdp.temporal import features_path, frame_sequence

        features = features_path(features_dir, prepared["test_path"])
        if not features.is_file():
            out.write_text(json.dumps(record, indent=2), encoding="utf-8")
            raise AdapterError(f"cvvdp wrote no --features file {features}; see {out}")
        try:
            t_temporal = time.perf_counter()
            record["temporal"] = frame_sequence(features, display, device, record["jod"])
            record["performance"]["temporal_pooling_seconds"] = time.perf_counter() - t_temporal
        except Exception as exc:
            out.write_text(json.dumps(record, indent=2), encoding="utf-8")
            raise AdapterError(f"could not derive the CVVDP frame sequence: {exc}") from exc
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
