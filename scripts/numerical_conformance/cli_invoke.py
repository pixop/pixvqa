"""Metric-aware pixvqa argv and environment for numerical conformance."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import CHROMA, FPS, FRAMES, FRAMES_NR, ROOT, SIZE
from .metric_caps import GROUP_METRICS, expand_metrics, includes_cvqa, includes_cvvdp, includes_learned, includes_native, includes_vmaf, selected_caps
from .registry import CVVDP_DISPLAY, Scenario, needs_nr_temporal

Family = Literal["sdr", "pq", "hlg"]

SDR_VMAF_2160 = "model/vmaf_v1.0.16_3d0h_2160.json"
HDR_VMAF_2160 = "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json"


@dataclass
class InvokeContext:
    full_bin: Path
    learned_config: Path | None = None
    learned_python: str | None = None
    vmaf_model: Path | None = None


@dataclass
class PixvqaInvoke:
    argv: list[str]
    env: dict[str, str]
    expected_frames: int | None
    metric_names: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _readable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def locate_beside_executable(full_bin: Path, rel: str) -> Path | None:
    rel_path = Path(rel)
    if _readable(ROOT / rel_path):
        return (ROOT / rel_path).resolve()
    exe = full_bin.resolve()
    for base in (exe.parent, exe.parent.parent, ROOT):
        cand = (base / rel_path).resolve()
        if _readable(cand):
            return cand
        cand = (base.parent / rel_path).resolve()
        if _readable(cand):
            return cand
    return None


def parse_learned_toml_python(config_path: Path) -> str | None:
    if not _readable(config_path):
        return None
    text = config_path.read_text(encoding="utf-8")
    section = ""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section not in ("learned", "cvvdp", "cvqa-hdr-nr"):
            continue
        m = re.match(r'python\s*=\s*"([^"]+)"', line)
        if m and section == "learned":
            val = m.group(1)
            if val.startswith("~/"):
                return str(Path.home() / val[2:])
            p = Path(val)
            if not p.is_absolute():
                p = (config_path.parent / p).resolve()
            return str(p)
    return None


def discover_learned_context(full_bin: Path) -> tuple[Path | None, str | None]:
    cfg_env = os.environ.get("PIXVQA_LEARNED_CONFIG", "").strip()
    if cfg_env.lower() == "none":
        config_path = None
    elif cfg_env:
        config_path = Path(cfg_env).expanduser()
    else:
        config_path = locate_beside_executable(full_bin, "config/learned_metrics.local.toml")

    for key in ("PIXVQA_LEARNED_PYTHON", "PIXVQA_CVVDP_PYTHON", "PIXVQA_CVQAHDR_PYTHON"):
        if os.environ.get(key):
            return config_path, os.environ[key]

    if config_path and _readable(config_path):
        py = parse_learned_toml_python(config_path)
        if py and Path(py).is_file():
            return config_path, py
    return config_path, None


def resolve_vmaf_model(full_bin: Path, family: Family) -> Path | None:
    if family == "hlg":
        return None
    rel = HDR_VMAF_2160 if family == "pq" else SDR_VMAF_2160
    override = os.environ.get("PIXVQA_VMAF_MODEL")
    if override:
        p = Path(override).expanduser()
        return p if _readable(p) else None
    found = locate_beside_executable(full_bin, rel)
    return found


def build_invoke_context(full_bin: Path) -> InvokeContext:
    cfg, py = discover_learned_context(full_bin)
    return InvokeContext(
        full_bin=full_bin,
        learned_config=cfg if cfg and _readable(cfg) else None,
        learned_python=py,
        vmaf_model=None,
    )


def scenario_uses_raw(scenario: Scenario) -> bool:
    return scenario in ("raw_raw", "raw_comp")


def needs_explicit_hdr_transfer(
    family: Family,
    scenario: Scenario,
    metric: str,
    *,
    force_explicit: bool,
) -> bool:
    if force_explicit:
        return True
    if not scenario_uses_raw(scenario):
        return False
    names = expand_metrics(metric)
    if any(n.startswith("pu21-") for n in names):
        return True
    if "cvvdp" in names:
        return True
    return False


def cvvdp_display_for(family: Family) -> str | None:
    if family == "sdr":
        return "standard_4k"
    return CVVDP_DISPLAY.get(family)


def frame_limit_for(metric: str) -> int | None:
    caps = selected_caps(metric)
    if any(not c.honors_frames for c in caps):
        return None
    return FRAMES


def build_pixvqa_command(
    ctx: InvokeContext,
    *,
    family: Family,
    scenario: Scenario,
    metric: str,
    ref: str,
    test: str,
    no_reference: bool = False,
    metric_device: str | None = None,
    decode: str | None = None,
    explicit_hdr_transfer: bool = False,
) -> PixvqaInvoke:
    mval = metric.split(":", 1)[1] if metric.startswith("group:") else metric
    names = expand_metrics(metric)
    notes: list[str] = []

    argv = [str(ctx.full_bin), "--metrics", mval]
    if not no_reference:
        argv += ["-r", ref, "-s", SIZE, "-f", CHROMA, "-t", test, "-s", SIZE, "-f", CHROMA]
    else:
        argv += ["-t", test]

    argv += ["--fps", FPS, "-v", "1"]

    nframes = frame_limit_for(metric)
    if nframes is not None:
        argv += ["-n", str(nframes)]
    else:
        notes.append("omit --frames (cvqa-hdr-nr official sampler)")

    if needs_explicit_hdr_transfer(family, scenario, metric, force_explicit=explicit_hdr_transfer):
        tf = "sdr" if family == "sdr" else family
        argv += ["--hdr-transfer", tf]
        notes.append(f"explicit hdr-transfer {tf} for raw-side inputs")

    if includes_vmaf(metric, family):
        model = resolve_vmaf_model(ctx.full_bin, family)
        if model is None:
            raise RuntimeError(f"VMAF model not found for family {family}")
        argv += ["--model", str(model)]
        notes.append(f"vmaf model {model.name}")

    if includes_cvvdp(metric):
        disp = cvvdp_display_for(family)
        if disp:
            argv += ["--cvvdp-display", disp]

    if metric_device:
        argv += ["--metric-device", metric_device]
    if decode:
        argv += ["--decode", decode]

    env = os.environ.copy()
    if includes_learned(metric):
        if ctx.learned_config and os.environ.get("PIXVQA_LEARNED_CONFIG", "").lower() != "none":
            env.setdefault("PIXVQA_LEARNED_CONFIG", str(ctx.learned_config))
        if ctx.learned_python:
            if includes_cvvdp(metric):
                env["PIXVQA_CVVDP_PYTHON"] = ctx.learned_python
                argv += ["--cvvdp-python", ctx.learned_python]
            if includes_cvqa(metric):
                env["PIXVQA_CVQAHDR_PYTHON"] = ctx.learned_python
                argv += ["--cvqahdr-python", ctx.learned_python]
    else:
        env["PIXVQA_LEARNED_CONFIG"] = "none"

    expected = nframes
    if expected is None and needs_nr_temporal(metric, scenario):
        expected = FRAMES_NR
    return PixvqaInvoke(argv=argv, env=env, expected_frames=expected, metric_names=names, notes=tuple(notes))


def classify_failure(proc_text: str, returncode: int, metric: str) -> tuple[str, str]:
    """Return (FAIL|SKIP, detail). Configuration/setup errors are never SKIP."""
    text = proc_text
    config_markers = (
        "requires --model",
        "transfer is unspecified",
        "--frames is incompatible",
        "No module named pixvqa_cvvdp",
        "No module named pixvqa_cvqahdr",
        "could not execute CVVDP interpreter",
        "could not execute CVQA-HDR-NR interpreter",
        "--cvvdp-* options require --metrics cvvdp",
        "--cvqahdr-* options require --metrics cvqa-hdr-nr",
        "preflight failed",
    )
    for marker in config_markers:
        if marker in text:
            return "FAIL", text[-800:]

    if includes_learned(metric) and returncode == 2:
        low = text.lower()
        if "checkpoint not found" in low or "checkpoint not fou" in low:
            return "SKIP", "cvqa-hdr-nr checkpoint/profile missing or path wrong in learned config"
        asset_markers = ("--cvqahdr-repo", "--cvqahdr-checkpoint", "--cvqahdr-profile")
        if any(m in text for m in asset_markers):
            return "SKIP", text[-500:]
    return "FAIL", text[-800:]
