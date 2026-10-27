"""Public CLI argument checks. A rejected configuration is a pass when the message is the documented one."""
from __future__ import annotations

import random
from pathlib import Path

from qa.harness import Case, Result, World, run_cmd, scored, write_yuv
from qa.inventory import check_inventory


def _case(id_, group, profiles, tags, func, optional=False) -> Case:
    return Case(id_, group, frozenset(profiles), tuple(tags), func, optional)


def _pair(world: World, w=64, h=64, chroma="420", bits=10, frames=2, bias=20):
    fmt = f"{chroma}p:{bits}"
    ref = world.tmp / f"r_{w}x{h}_{fmt}_{frames}_{bias}.yuv"
    dist = world.tmp / f"d_{w}x{h}_{fmt}_{frames}_{bias}.yuv"
    if not ref.exists():
        write_yuv(ref, w, h, chroma, bits, frames, 0)
        write_yuv(dist, w, h, chroma, bits, frames, bias)
    return ref, dist, f"{w}x{h}", fmt


def _base(world: World, metrics="psnr", binary="cpu"):
    ref, dist, size, fmt = _pair(world)
    return [
        "--metrics", metrics, "-v", "0",
        "-r", str(ref), "-s", size, "-f", fmt,
        "-t", str(dist), "-s", size, "-f", fmt,
    ]


def _ok(proc, command, detail=""):
    if proc.returncode != 0 or not scored(proc.stdout):
        return Result("FAIL", f"rc={proc.returncode} {proc.stderr[-400:]} {proc.stdout[-200:]}", command)
    return Result("PASS", detail or f"rc=0", command)


def _reject(proc, needle, command):
    text = proc.stderr + proc.stdout
    if proc.returncode == 0 or scored(proc.stdout) or needle not in text:
        return Result(
            "FAIL",
            f"rc={proc.returncode} missing {needle!r} scored={scored(proc.stdout)} {text[-500:]}",
            command,
        )
    if proc.returncode < 0:
        return Result("FAIL", f"signal {-proc.returncode}", command)
    return Result("PASS", f"rc={proc.returncode} {needle}", command)


def _run(world, binary, args, timeout=30):
    path = world.binary(binary)
    cmd = [str(path), *args]
    proc = run_cmd(cmd, timeout=timeout)
    return proc, " ".join(cmd)


def cases() -> list:
    out = []

    def inventory(_w: World) -> Result:
        ok, detail = check_inventory(_w.root)
        return Result("PASS" if ok else "FAIL", detail, "inventory")

    out.append(_case("inventory", "cli", ["smoke"], ["inventory"], inventory))

    def help_version(world: World) -> Result:
        path = world.binary("cpu")
        if path is None:
            return Result("FAIL", "build/cpu/pixvqa is absent", "")
        help_run = run_cmd([str(path), "--help"])
        short = run_cmd([str(path), "-h"])
        version = run_cmd([str(path), "--version"])
        hdr = run_cmd([str(path), "--help-hdr"])
        command = f"{path} --help"
        if help_run.returncode != 0 or short.stdout != help_run.stdout:
            return Result("FAIL", "help disagreement", command)
        for needle in ("--metrics", "--decode", "--metric-device", "--threads", "--qcs", "pu21-psnr"):
            if needle not in help_run.stdout:
                return Result("FAIL", f"help missing {needle}", command)
        if "NVIDIA" in help_run.stdout or "GeForce" in help_run.stdout:
            return Result("FAIL", "help printed a device name", command)
        if version.returncode != 0 or "pixvqa version 0.3.0-dev" not in version.stdout:
            return Result("FAIL", version.stdout, f"{path} --version")
        if "decode: none" not in version.stdout:
            return Result("FAIL", version.stdout, f"{path} --version")
        if hdr.returncode != 0 or "per-input option, shared option, stream metadata" not in hdr.stdout:
            return Result("FAIL", "help-hdr missing precedence", f"{path} --help-hdr")
        sw = world.binary("swdec")
        if sw is not None:
            swv = run_cmd([str(sw), "--version"])
            if swv.returncode != 0 or "decode: software" not in swv.stdout:
                return Result("FAIL", swv.stdout, f"{sw} --version")
        return Result("PASS", "help, version, and help-hdr", command)

    out.append(_case("help-version", "cli", ["smoke"], ["help", "version"], help_version))

    def missing(world: World) -> Result:
        proc, cmd = _run(world, "cpu", [])
        return _reject(proc, "Usage:", cmd)

    out.append(_case("missing-args", "cli", ["smoke"], ["cli-missing"], missing))

    def unknown(world: World) -> Result:
        proc, cmd = _run(world, "cpu", ["--not-a-real-option"])
        return _reject(proc, "PARSE ERROR", cmd)

    out.append(_case("unknown-option", "cli", ["smoke"], ["cli-unknown"], unknown))

    def missing_value(world: World) -> Result:
        proc, cmd = _run(world, "cpu", ["--metrics"])
        return _reject(proc, "missing value", cmd)

    out.append(_case("missing-value", "cli", ["smoke"], ["cli-missing"], missing_value))

    def one_metric(name, profiles):
        def run(world: World) -> Result:
            binary = "swdec" if name == "vmaf" else "cpu"
            if world.binary(binary) is None:
                return Result("FAIL", f"{binary} binary is absent", name)
            if name == "vmaf":
                if not world.model.is_file():
                    return Result("FAIL", f"model missing {world.model}", name)
                ref, dist, size, _fmt = _pair(world, 320, 240, "420", 8, 3)
                args = [
                    "--metrics", "vmaf", "--model", str(world.model), "-v", "0", "--threads", "1",
                    "-r", str(ref), "-s", size, "-f", "420p:8",
                    "-t", str(dist), "-s", size, "-f", "420p:8",
                ]
                proc, cmd = _run(world, binary, args, timeout=120)
                if proc.returncode != 0 or "VMAF" not in proc.stdout:
                    return Result("FAIL", proc.stderr[-400:] + proc.stdout[-200:], cmd)
                return Result("PASS", "vmaf scored", cmd)
            args = _base(world, name)
            if name == "xpsnr":
                args = ["--fps", "24", *args]
            if name.startswith("pu21-"):
                args = ["--hdr-transfer", "pq", *args]
            proc, cmd = _run(world, binary, args, timeout=90)
            return _ok(proc, cmd, name)

        return run

    for metric in ("psnr", "ssim", "msssim", "xpsnr", "pu21-psnr", "pu21-ssim", "pu21-msssim"):
        profiles = ["smoke", "full"] if metric in ("psnr", "ssim", "pu21-psnr") else ["full"]
        out.append(_case(f"metric-{metric}", "metrics", profiles, [f"metric:{metric}"], one_metric(metric, profiles)))
    out.append(_case("metric-vmaf", "metrics", ["full"], ["metric:vmaf"], one_metric("vmaf", ["full"])))

    def unknown_metric(world: World) -> Result:
        args = _base(world, "no-such-metric")
        proc, cmd = _run(world, "cpu", args)
        return _reject(proc, "unsupported metric", cmd)

    out.append(_case("unknown-metric", "cli", ["smoke"], ["cli-metric"], unknown_metric))

    def empty_metric(world: World) -> Result:
        args = _base(world, "")
        # _base still inserts the empty name. Force the flag.
        args[1] = ","
        proc, cmd = _run(world, "cpu", args)
        return _reject(proc, "empty --metrics entry", cmd)

    out.append(_case("empty-metric", "cli", ["full"], ["cli-metric"], empty_metric))

    def duplicate_metric(world: World) -> Result:
        args = _base(world, "psnr,psnr,ssim")
        proc, cmd = _run(world, "cpu", args, timeout=60)
        if proc.returncode != 0 or "[ PSNR" not in proc.stdout or "[ SSIM" not in proc.stdout:
            return Result("FAIL", proc.stdout[-300:] + proc.stderr[-200:], cmd)
        return Result("PASS", "duplicate name is idempotent and both metrics score", cmd)

    out.append(_case("duplicate-metric", "cli", ["full"], ["cli-metric"], duplicate_metric))

    def duplicate_option(world: World) -> Result:
        args = ["--metrics", "ssim", "--metrics", "psnr", *_base(world)[2:]]
        proc, cmd = _run(world, "cpu", args, timeout=60)
        if proc.returncode != 0 or "[ PSNR" not in proc.stdout or "[ SSIM" not in proc.stdout:
            return Result("FAIL", "second --metrics replaced the first list\n" + proc.stdout[-300:], cmd)
        return Result("PASS", "repeated --metrics adds names", cmd)

    out.append(_case("duplicate-metrics-flag", "cli", ["full"], ["cli-duplicate"], duplicate_option))

    def bad_values(world: World) -> Result:
        ref, dist, size, fmt = _pair(world)
        common = ["-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt]
        checks = [
            (["--metrics", "psnr", "--fps", "nan", *common], "invalid --fps"),
            (["--metrics", "psnr", "--fps", "inf", *common], "invalid --fps"),
            (["--metrics", "psnr", "--fps", "0", *common], "invalid --fps"),
            (["--metrics", "psnr", "--fps", "-1", *common], "invalid --fps"),
            (["--metrics", "psnr", "--threads", "0", *common], "--threads"),
            (["--metrics", "psnr", "--threads", "nan", *common], "--threads"),
            (["--metrics", "psnr", "--threads", "1000001", *common], "--threads"),
            (["--metrics", "psnr", "--frames", "abc", *common], "--frames"),
            (["--metrics", "psnr", "--frames", "0", *common], "--frames"),
            (["--metrics", "psnr", "--frames", "nan", *common], "--frames"),
            (["--metrics", "psnr", "-v", "nope", *common], "-v"),
            (["--metrics", "psnr", "--seek", "abc", *common], "--seek"),
            (["--metrics", "psnr", "--seek", "3", *common], "nonzero --seek"),
            (["--metrics", "psnr", "--decode", "both", *common], "unsupported --decode"),
            (["--metrics", "psnr", "--metric-device", "metal", *common], "unsupported --metric-device"),
            (["--metrics", "psnr", "--input", "tape", *common], "unsupported --input"),
            (["--metrics", "psnr", "-f", "nv12", "-r", str(ref), "-s", size, "-t", str(dist), "-s", size, "-f", fmt], "unsupported -f"),
            (["--metrics", "psnr", "-r", str(ref), "-s", size, "-f", "rgb24", "-t", str(dist), "-s", size, "-f", "rgb24"], "unsupported -f"),
            (["--metrics", "psnr", "-r", str(ref), "-s", size, "-f", "420p:12", "-t", str(dist), "-s", size, "-f", "420p:12"], "unsupported -f"),
            (["--metrics", "psnr", "-r", str(ref), "-s", size, "-f", "420p:abc", "-t", str(dist), "-s", size, "-f", "420p:abc"], "unsupported -f"),
            (["--metrics", "psnr", "-r", str(ref), "-s", "0x0", "-f", fmt, "-t", str(dist), "-s", "0x0", "-f", fmt], "bad -s"),
            (["--metrics", "psnr", "-r", str(ref), "-s", "axb", "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt], "bad -s"),
            (["--metrics", "psnr", "--qcs-range", "psnr=40:30", *common], "lower < upper"),
            (["--metrics", "psnr", "--qcs-range", "psnr=nan:1", *common], "invalid --qcs-range"),
            (["--metrics", "vmaf", "--qcs", *common], "--model"),
            (["--metrics", "psnr", "--model", "model.json", *common], "require vmaf"),
            (["--metrics", "psnr", "--tf", "st2084", "--hdr-transfer", "pq", *common], "--tf and --hdr-transfer both set"),
            (["--metrics", "psnr", "--hdr-peak-nits", "nan", *common], "expects auto or a number"),
            (["--metrics", "psnr", "--hdr-peak-nits", "inf", *common], "expects auto or a number"),
            (["--metrics", "psnr", "--hdr-peak-nits", "0", *common], "(0, 10000]"),
            (["--metrics", "psnr", "--hdr-peak-nits", "10001", *common], "(0, 10000]"),
            (["--metrics", "pu21-psnr", "--hdr-transfer", "hlg", "--hdr-black-nits", "-1", *common], "black-nits"),
            (["--metrics", "pu21-psnr", "--hdr-transfer", "hlg", "--hdr-hlg-gamma", "0", *common], "gamma"),
            (["--metrics", "pu21-psnr", "--hdr-transfer", "hlg", "--hdr-hlg-gamma", "11", *common], "gamma"),
            (["--metrics", "psnr", "--psnr", *common], "was removed"),
            (["--metrics", "psnr", "--cuda-prec", "fp64", *common], "was removed"),
            (["--metrics", "psnr", "--wsssim", *common], "unsupported argument"),
        ]
        failed = []
        last = ""
        for args, needle in checks:
            proc, cmd = _run(world, "cpu", args)
            last = cmd
            got = _reject(proc, needle, cmd)
            if got.status != "PASS":
                failed.append(f"{needle}: {got.detail}")
        if failed:
            return Result("FAIL", " | ".join(failed[:6]), last)
        return Result("PASS", f"{len(checks)} invalid values rejected", last)

    out.append(_case("invalid-values", "cli", ["smoke"], ["cli-invalid", "layout:nv12", "layout:rgb24", "layout:12bit"], bad_values))

    def defaults_and_edges(world: World) -> Result:
        ref, dist, size, fmt = _pair(world, bits=8, chroma="420")
        threads = run_cmd([
            str(world.cpu), "--metrics", "psnr", "--threads", "auto", "-v", "0",
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ])
        one = run_cmd([
            str(world.cpu), "--metrics", "psnr", "--threads", "1", "--frames", "1", "-v", "0",
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ])
        fps = run_cmd([
            str(world.cpu), "--metrics", "xpsnr", "--fps", "60000/1001", "-v", "0",
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ])
        qcs = run_cmd([
            str(world.cpu), "--metrics", "psnr", "--qcs", "-v", "0",
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ])
        command = "defaults"
        if threads.returncode != 0 or "threads:" not in threads.stdout or "(auto;" not in threads.stdout:
            return Result("FAIL", threads.stdout[-300:], command)
        if one.returncode != 0 or "Total frames:1" not in one.stdout or "threads: 1" not in one.stdout:
            return Result("FAIL", one.stdout[-300:], command)
        if fps.returncode != 0 or "[ XPSNR" not in fps.stdout:
            return Result("FAIL", fps.stderr[-300:], command)
        if qcs.returncode != 0 or "QCS" not in qcs.stdout:
            return Result("FAIL", qcs.stdout[-300:], command)
        return Result("PASS", "auto threads, one frame, rational fps, default qcs", command)

    out.append(_case("defaults-edges", "cli", ["smoke"], ["threads:auto", "threads:1", "qcs"], defaults_and_edges))

    def paths(world: World) -> Result:
        ref, dist, size, fmt = _pair(world)
        spaced = world.tmp / "ref clip.yuv"
        if not spaced.exists():
            spaced.write_bytes(ref.read_bytes())
        ok = run_cmd([
            str(world.cpu), "--metrics", "psnr", "-v", "0",
            "-r", str(spaced), "-s", size, "-f", fmt,
            "-t", str(dist), "-s", size, "-f", fmt,
        ])
        missing = run_cmd([
            str(world.cpu), "--metrics", "psnr", "-v", "0",
            "-r", str(world.tmp / "no-such.yuv"), "-s", size, "-f", fmt,
            "-t", str(dist), "-s", size, "-f", fmt,
        ])
        directory = run_cmd([
            str(world.cpu), "--metrics", "psnr", "-v", "0",
            "-r", str(world.tmp), "-s", size, "-f", fmt,
            "-t", str(dist), "-s", size, "-f", fmt,
        ])
        command = "paths"
        if ok.returncode != 0 or "[ PSNR" not in ok.stdout:
            return Result("FAIL", "path with a space was rejected", command)
        if missing.returncode == 0 or scored(missing.stdout):
            return Result("FAIL", "missing file scored", command)
        if directory.returncode == 0 or scored(directory.stdout):
            return Result("FAIL", "directory path scored", command)
        return Result("PASS", "space path scored; missing and directory paths did not", command)

    out.append(_case("paths", "cli", ["full"], ["paths"], paths))

    def disagree(world: World) -> Result:
        ref, dist, size, fmt = _pair(world)
        proc, cmd = _run(world, "cpu", [
            "--metrics", "psnr",
            "-r", str(ref), "-s", "64x64", "-f", "420p:10",
            "-t", str(dist), "-s", "32x32", "-f", "420p:10",
        ])
        return _reject(proc, "disagree", cmd)

    out.append(_case("geometry-disagree", "cli", ["smoke"], ["geometry"], disagree))

    def odd_420(world: World) -> Result:
        ref = world.tmp / "odd.yuv"
        ref.write_bytes(b"\x00" * 64)
        proc, cmd = _run(world, "cpu", [
            "--metrics", "psnr",
            "-r", str(ref), "-s", "3x4", "-f", "420p:8",
            "-t", str(ref), "-s", "3x4", "-f", "420p:8",
        ])
        return _reject(proc, "even width", cmd)

    out.append(_case("odd-420", "cli", ["full"], ["geometry"], odd_420))

    def model_conflict(world: World) -> Result:
        if not world.model.is_file() or world.binary("swdec") is None:
            return Result("SKIP", "swdec or the VMAF model is absent", "model")
        ref, dist, size, _fmt = _pair(world, 320, 240, "420", 8, 2)
        proc, cmd = _run(world, "swdec", [
            "--metrics", "vmaf", "--model", str(world.model), "--vmaf-model", str(world.tmp / "other.json"),
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ])
        return _reject(proc, "disagree", cmd)

    out.append(_case("model-disagree", "cli", ["full"], ["metric:vmaf"], model_conflict))

    def cpu_rejects_gpu_metric(world: World) -> Result:
        args = ["--metric-device", "gpu", *_base(world)]
        proc, cmd = _run(world, "cpu", args)
        return _reject(proc, "no GPU metrics", cmd)

    out.append(_case("cpu-build-rejects-gpu-metric", "backend", ["smoke"], ["metric-device:gpu"], cpu_rejects_gpu_metric))

    def fuzz(world: World) -> Result:
        rng = random.Random(20261022)
        ref, dist, size, fmt = _pair(world)
        base = [
            "--metrics", "psnr", "--threads", "1", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt,
            "-t", str(dist), "-s", size, "-f", fmt,
        ]
        junk = ["", "nan", "inf", "-inf", "NaN", "1e999", "--decode", "gpu", "--frames", "abc", "-f", "420p:abc"]
        crashes = []
        for i in range(40):
            args = list(base)
            op = i % 4
            if op == 0 and len(args) > 2:
                del args[rng.randrange(len(args))]
            elif op == 1:
                args[rng.randrange(len(args))] = rng.choice(junk)
            elif op == 2:
                args.insert(rng.randrange(len(args) + 1), rng.choice(junk))
            else:
                args.append("--not-a-real-option")
            try:
                proc = run_cmd([str(world.cpu), *args], timeout=15)
            except Exception as exc:
                crashes.append(f"{i}:{exc}")
                continue
            if proc.returncode < 0:
                crashes.append(f"{i}:signal {-proc.returncode}")
        command = "fuzz seed 20261022 x40"
        if crashes:
            return Result("FAIL", "; ".join(crashes[:5]), command)
        return Result("PASS", "40 CLI mutations produced no signal", command)

    out.append(_case("cli-fuzz", "fuzz", ["full"], ["fuzz"], fuzz))
    return out
