"""Codec, HDR, combinatorial, negative, stability, and hardware cases."""
from __future__ import annotations

import itertools
import random
from pathlib import Path

from qa.harness import Case, Result, World, frame_bytes, run_cmd, scored, write_yuv


def _case(id_, group, profiles, tags, func, optional=False) -> Case:
    return Case(id_, group, frozenset(profiles), tuple(tags), func, optional)


def _reject(proc, needle) -> bool:
    text = proc.stderr + proc.stdout
    return proc.returncode != 0 and proc.returncode > 0 and needle in text and not scored(proc.stdout)


def _ff(world: World, args: list, timeout: int = 90):
    return run_cmd([world.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args], timeout=timeout)


def _has_codec(world: World, kind: str, name: str) -> bool:
    done = run_cmd([world.ffmpeg, "-hide_banner", f"-{kind}"], timeout=30)
    blob = "\n" + done.stdout + "\n"
    return f"\n{name} " in blob or f" {name} " in done.stdout


def _pix(world: World, binary: str, args: list, timeout: int = 90):
    path = world.binary(binary)
    cmd = [str(path), *args]
    return run_cmd(cmd, timeout=timeout), " ".join(cmd)


def _pair(world: World, w, h, chroma, bits, frames, bias=30):
    fmt = f"{chroma}p:{bits}"
    ref = world.tmp / f"m_{w}x{h}_{fmt}_{frames}.yuv"
    dist = world.tmp / f"m_{w}x{h}_{fmt}_{frames}_d.yuv"
    if not ref.exists():
        write_yuv(ref, w, h, chroma, bits, frames, 0)
        write_yuv(dist, w, h, chroma, bits, frames, bias)
    return ref, dist, f"{w}x{h}", fmt


def _script(world: World, script: str, env: dict, timeout: int, binary: str) -> Result:
    path = world.root / "scripts" / script
    bin_path = world.binary(binary)
    if bin_path is None:
        return Result("FAIL", f"{binary} binary is absent", script)
    merged = {
        "PIXVQA_BIN": str(bin_path),
        "FFMPEG": world.ffmpeg or "",
        "VVDEC": world.vvdec or "",
    }
    if world.vmaf_cli:
        merged["VMAF_CLI"] = str(world.vmaf_cli)
    merged.update(env)
    try:
        proc = run_cmd(["python3", str(path)], timeout=timeout, env=merged)
    except Exception as exc:
        return Result("FAIL", str(exc), script)
    command = f"PIXVQA_BIN={bin_path} python3 scripts/{script}"
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-800:]
        return Result("FAIL", tail, command)
    return Result("PASS", "existing numerical script passed", command)


def _combinatorial() -> list:
    out = []
    factors = {
        "fmt": ["420p:8", "420p:10", "422p:8", "422p:10", "444p:8", "444p:10"],
        "metric": ["psnr", "ssim", "msssim", "xpsnr", "pu21-psnr", "pu21-ssim", "pu21-msssim"],
        "threads": ["auto", "1", "4"],
        "limit": ["all", "1"],
        "verbosity": ["0", "1"],
        "input": ["auto", "raw"],
    }
    names = list(factors)
    needed = set()
    for a, b in itertools.combinations(names, 2):
        for va in factors[a]:
            for vb in factors[b]:
                needed.add((a, va, b, vb))
    rng = random.Random(20261022)
    rows = []
    while needed:
        best = None
        best_cover = -1
        for _ in range(60):
            row = {n: rng.choice(factors[n]) for n in names}
            cover = sum((a, row[a], b, row[b]) in needed for a, b in itertools.combinations(names, 2))
            if cover > best_cover:
                best, best_cover = row, cover
        rows.append(best)
        for a, b in itertools.combinations(names, 2):
            needed.discard((a, best[a], b, best[b]))

    def make(index, row):
        def run(world: World) -> Result:
            chroma = row["fmt"][:3]
            bits = int(row["fmt"].split(":")[1])
            ref, dist, size, fmt = _pair(world, 64, 64, chroma, bits, 2)
            args = ["--metrics", row["metric"], "--threads", row["threads"], "-v", row["verbosity"],
                    "--input", row["input"]]
            if row["metric"] == "xpsnr":
                args += ["--fps", "24"]
            if row["metric"].startswith("pu21-"):
                args += ["--hdr-transfer", "pq"]
            if row["limit"] == "1":
                args += ["--frames", "1"]
            args += ["-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt]
            proc, cmd = _pix(world, "cpu", args, timeout=90)
            if proc.returncode != 0 or not scored(proc.stdout):
                return Result("FAIL", proc.stderr[-300:] + proc.stdout[-200:], cmd)
            return Result("PASS", str(row), cmd)
        return run

    for i, row in enumerate(rows):
        out.append(_case(
            f"pair-{i:03d}", "combinatorial", ["full"],
            [f"layout:{row['fmt']}", f"metric:{row['metric']}", f"threads:{row['threads']}"],
            make(i, row),
        ))

    fast = ["psnr", "xpsnr", "ssim", "msssim", "pu21-psnr", "pu21-ssim", "pu21-msssim"]

    def subset_run(chosen):
        def run(world: World) -> Result:
            ref, dist, size, fmt = _pair(world, 64, 64, "420", 10, 2)
            args = ["--metrics", ",".join(chosen), "--threads", "1", "-v", "0"]
            if "xpsnr" in chosen:
                args += ["--fps", "24"]
            if any(name.startswith("pu21-") for name in chosen):
                args += ["--hdr-transfer", "pq"]
            args += ["-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt]
            proc, cmd = _pix(world, "cpu", args, timeout=90)
            if proc.returncode != 0:
                return Result("FAIL", proc.stderr[-300:], cmd)
            for name in chosen:
                token = {
                    "psnr": "[ PSNR", "xpsnr": "[ XPSNR", "ssim": "[ SSIM", "msssim": "[ MS-SSIM",
                    "pu21-psnr": "PU21-PSNR", "pu21-ssim": "PU21-SSIM", "pu21-msssim": "PU21-MS-SSIM",
                }[name]
                if token not in proc.stdout:
                    return Result("FAIL", f"missing {token}", cmd)
            return Result("PASS", ",".join(chosen), cmd)
        return run

    for mask in range(1, 1 << len(fast)):
        chosen = [fast[i] for i in range(len(fast)) if mask & (1 << i)]
        out.append(_case(
            "subset-" + "-".join(chosen), "subsets", ["full"],
            [f"metric:{name}" for name in chosen],
            subset_run(chosen),
        ))

    def vmaf_with(extra):
        def run(world: World) -> Result:
            if world.binary("swdec") is None or not world.model.is_file():
                return Result("FAIL", "swdec or VMAF model is absent", "vmaf-subset")
            ref, dist, size, _fmt = _pair(world, 320, 240, "420", 8, 3)
            names = ["vmaf", *extra]
            args = ["--metrics", ",".join(names), "--model", str(world.model), "--threads", "1", "-v", "0"]
            if "xpsnr" in extra:
                args += ["--fps", "24"]
            args += ["-r", str(ref), "-s", size, "-f", "420p:8", "-t", str(dist), "-s", size, "-f", "420p:8"]
            proc, cmd = _pix(world, "swdec", args, timeout=180)
            if proc.returncode != 0 or "VMAF" not in proc.stdout:
                return Result("FAIL", proc.stderr[-400:], cmd)
            return Result("PASS", ",".join(names), cmd)
        return run

    for extra in ([], ["psnr"], ["ssim"], ["msssim"], ["xpsnr"], ["psnr", "ssim", "msssim", "xpsnr"]):
        label = "vmaf" if not extra else "vmaf-" + "-".join(extra)
        out.append(_case(label, "subsets", ["full"], ["metric:vmaf", *[f"metric:{n}" for n in extra]], vmaf_with(extra)))
    return out


def _codec_cases() -> list:
    specs = [
        ("h264-ip", "420", 8, "yuv420p", "mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "0", "-bf", "0"], "h264", 4, None),
        ("h264-b", "420", 8, "yuv420p", "mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "18", "-bf", "2", "-g", "4"], "h264", 8, None),
        ("h264-long", "420", 8, "yuv420p", "mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "0", "-bf", "0"], "h264", 16, None),
        ("h264-one", "420", 8, "yuv420p", "mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "0", "-bf", "0"], "h264", 1, None),
        ("hevc", "420", 10, "yuv420p10le", "mov", ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"], "hevc", 4, None),
        ("av1-8", "420", 8, "yuv420p", "mp4", ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-cpu-used", "8"], "libdav1d", 4, ["-c:v", "libdav1d"]),
        ("av1-10", "420", 10, "yuv420p10le", "mp4", ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le", "-cpu-used", "8",
            "-aom-params", "color-primaries=9:transfer-characteristics=16:matrix-coefficients=9",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"], "libdav1d", 4, ["-c:v", "libdav1d"]),
        ("vvc", "420", 10, "yuv420p10le", "mp4", ["-c:v", "libvvenc", "-preset", "faster", "-qp", "32", "-period", "1"], "vvc", 4, ["-c:v", "vvc"]),
        ("vp9", "420", 8, "yuv420p", "webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-lossless", "1", "-cpu-used", "8"], "vp9", 4, None),
        ("ffv1-422", "422", 10, "yuv422p10le", "mkv", ["-c:v", "ffv1", "-pix_fmt", "yuv422p10le", "-level", "3"], "ffv1", 4, None),
        ("ffv1-444", "444", 8, "yuv444p", "mkv", ["-c:v", "ffv1", "-pix_fmt", "yuv444p", "-level", "3"], "ffv1", 4, None),
        ("mpeg2", "420", 8, "yuv420p", "ts", ["-c:v", "mpeg2video", "-pix_fmt", "yuv420p", "-q:v", "2", "-f", "mpegts"], "mpeg2video", 4, None),
        ("mjpeg", "420", 8, "yuvj420p", "avi", ["-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", "3"], "mjpeg", 4, None),
        ("prores", "422", 10, "yuv422p10le", "mov", ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "prores", 4, None),
    ]

    def make(spec):
        label, chroma, bits, pix, ext, codec_args, decoder, frames, forced = spec

        def run(world: World) -> Result:
            if not world.ffmpeg or world.binary("swdec") is None:
                return Result("FAIL", "ffmpeg or swdec is absent", label)
            w = h = 64
            src = world.tmp / f"{label}.src.yuv"
            enc = world.tmp / f"{label}.{ext}"
            oracle = world.tmp / f"{label}.oracle.yuv"
            second = world.tmp / f"{label}.second.yuv"
            fmt_chroma = "420" if pix.startswith("yuvj") else chroma
            write_yuv(src, w, h, fmt_chroma, bits, frames)
            enc_cmd = ["-f", "rawvideo", "-pix_fmt", pix if not pix.startswith("yuvj") else "yuv420p",
                       "-s", f"{w}x{h}", "-r", "24", "-i", str(src), "-frames:v", str(frames), *codec_args, str(enc)]
            if pix.startswith("yuvj"):
                enc_cmd = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "24",
                           "-i", str(src), "-frames:v", str(frames), *codec_args, str(enc)]
            encoded = _ff(world, enc_cmd, timeout=180)
            if encoded.returncode != 0 or not enc.is_file():
                return Result("SKIP", f"encoder failed: {encoded.stderr[-200:]}", " ".join(codec_args))
            dec_args = ["-i", str(enc)]
            if forced:
                dec_args = [*forced, *dec_args]
            raw_pix = "yuv420p" if pix == "yuvj420p" else pix
            one = _ff(world, [*dec_args, "-f", "rawvideo", "-pix_fmt", raw_pix, str(oracle)])
            two = _ff(world, ["-i", str(enc), "-f", "rawvideo", "-pix_fmt", raw_pix, str(second)])
            if one.returncode != 0 or two.returncode != 0:
                return Result("FAIL", (one.stderr + two.stderr)[-300:], label)
            if oracle.read_bytes() != second.read_bytes():
                return Result("FAIL", "two independent decodes disagree", label)
            expected = frame_bytes(w, h, fmt_chroma, bits) * frames
            if oracle.stat().st_size != expected:
                return Result("FAIL", f"oracle bytes {oracle.stat().st_size} != {expected}", label)
            fmt = f"{fmt_chroma}p:{bits}"
            proc, cmd = _pix(world, "swdec", [
                "--metrics", "psnr", "--decode", "cpu", "-v", "0",
                "-r", str(oracle), "--input", "raw", "-s", f"{w}x{h}", "-f", fmt,
                "-t", str(enc), "--input", "compressed", "-s", f"{w}x{h}", "-f", fmt,
            ], timeout=180)
            text = proc.stdout + proc.stderr
            if proc.returncode != 0 or "999.99" not in proc.stdout:
                return Result("FAIL", text[-500:], cmd)
            if f"Total frames:{frames}" not in proc.stdout or "decoder threads: 8" not in proc.stdout:
                return Result("FAIL", f"frame count or decoder threads\n{proc.stdout[-400:]}", cmd)
            if f"decoder={decoder}" not in proc.stderr or "backend=software" not in proc.stderr:
                return Result("FAIL", proc.stderr[-400:], cmd)
            if "reference: raw" not in proc.stdout or "distorted: cpu" not in proc.stdout:
                return Result("FAIL", proc.stdout[-300:], cmd)
            if decoder == "vvc" and world.vvdec:
                elem = world.tmp / f"{label}.266"
                extracted = _ff(world, ["-i", str(enc), "-c", "copy", "-f", "vvc", str(elem)])
                yuv = world.tmp / f"{label}.vvdec.yuv"
                app = run_cmd([world.vvdec, "-b", str(elem), "-o", str(yuv)], timeout=60)
                if extracted.returncode != 0 or app.returncode != 0 or yuv.read_bytes() != oracle.read_bytes():
                    return Result("FAIL", "VVdeC bytes differ from the FFmpeg vvc oracle", cmd)
            return Result("PASS", f"exact oracle bytes and PSNR 999.99, {frames} frames, {decoder}", cmd)
        return run

    out = []
    for spec in specs:
        out.append(_case(f"codec-{spec[0]}", "codec", ["full"], [f"codec:{spec[0]}", f"layout:{spec[1]}p:{spec[2]}"], make(spec)))
    return out


def _negative_cases() -> list:
    out = []

    def empty(world: World) -> Result:
        path = world.tmp / "empty.yuv"
        path.write_bytes(b"")
        proc, cmd = _pix(world, "cpu", [
            "--metrics", "psnr", "-v", "0",
            "-r", str(path), "-s", "16x16", "-f", "420p:8",
            "-t", str(path), "-s", "16x16", "-f", "420p:8",
        ])
        if not _reject(proc, "no frames read"):
            return Result("FAIL", proc.stderr[-200:] + proc.stdout[-200:], cmd)
        return Result("PASS", "empty file", cmd)

    def short(world: World) -> Result:
        path = world.tmp / "short.yuv"
        path.write_bytes(b"\x10" * 30)
        proc, cmd = _pix(world, "cpu", [
            "--metrics", "psnr", "-v", "0",
            "-r", str(path), "-s", "16x16", "-f", "420p:8",
            "-t", str(path), "-s", "16x16", "-f", "420p:8",
        ])
        if not _reject(proc, "short YUV"):
            return Result("FAIL", proc.stderr[-200:], cmd)
        return Result("PASS", "truncated raw frame", cmd)

    def mismatch(world: World) -> Result:
        a = world.tmp / "frames4.yuv"
        b = world.tmp / "frames2.yuv"
        write_yuv(a, 32, 32, "420", 8, 4)
        write_yuv(b, 32, 32, "420", 8, 2)
        proc, cmd = _pix(world, "cpu", [
            "--metrics", "psnr", "-v", "0",
            "-r", str(a), "-s", "32x32", "-f", "420p:8",
            "-t", str(b), "-s", "32x32", "-f", "420p:8",
        ])
        if not _reject(proc, "frame count mismatch"):
            return Result("FAIL", proc.stderr[-200:], cmd)
        return Result("PASS", "raw frame-count mismatch", cmd)

    def fake_mp4(world: World) -> Result:
        path = world.tmp / "fake.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
        proc, cmd = _pix(world, "cpu", [
            "--metrics", "psnr", "--decode", "cpu", "--input", "compressed", "-v", "0",
            "-r", str(path), "-s", "16x16", "-f", "420p:8",
            "-t", str(path), "-s", "16x16", "-f", "420p:8",
        ])
        if not _reject(proc, "no CPU decoder"):
            return Result("FAIL", proc.stderr[-300:], cmd)
        return Result("PASS", "cpu build refuses a compressed request", cmd)

    def corrupt(world: World) -> Result:
        if not world.ffmpeg or world.binary("swdec") is None:
            return Result("FAIL", "ffmpeg or swdec is absent", "corrupt")
        src = world.tmp / "neg.src.yuv"
        enc = world.tmp / "neg.mp4"
        write_yuv(src, 64, 64, "420", 8, 4)
        made = _ff(world, ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x64", "-r", "24", "-i", str(src),
                           "-frames:v", "4", "-c:v", "libx264", "-preset", "veryfast", "-qp", "0", str(enc)])
        if made.returncode != 0:
            return Result("FAIL", made.stderr[-200:], "encode")
        data = bytearray(enc.read_bytes())
        if len(data) < 64:
            return Result("FAIL", "encoded file is too small to corrupt", "encode")
        for i in range(32, 64):
            data[i] ^= 0xFF
        broken = world.tmp / "broken.mp4"
        broken.write_bytes(data)
        truncated = world.tmp / "trunc.mp4"
        truncated.write_bytes(bytes(data[:24]))
        empty = world.tmp / "empty.mp4"
        empty.write_bytes(b"")
        declared = _pix(world, "swdec", [
            "--metrics", "psnr", "--decode", "cpu", "--input", "compressed", "-v", "0",
            "-r", str(enc), "-s", "48x48", "-f", "420p:8",
            "-t", str(enc), "-s", "48x48", "-f", "420p:8",
        ])
        bad = []
        for label, path in (("corrupt", broken), ("truncated", truncated), ("empty-container", empty)):
            proc, cmd = _pix(world, "swdec", [
                "--metrics", "psnr", "--decode", "cpu", "--input", "compressed", "-v", "0",
                "-r", str(path), "-s", "64x64", "-f", "420p:8",
                "-t", str(path), "-s", "64x64", "-f", "420p:8",
            ])
            if proc.returncode == 0 or scored(proc.stdout) or proc.returncode < 0:
                bad.append(label)
        if declared[0].returncode == 0 or scored(declared[0].stdout):
            bad.append("wrong-size")
        if bad:
            return Result("FAIL", "accepted " + ",".join(bad), "negative-compressed")
        return Result("PASS", "corrupt, truncated, empty, and wrong declared size produced no score", "negative-compressed")

    def swdec_gpu(world: World) -> Result:
        if not world.ffmpeg or world.binary("swdec") is None:
            return Result("FAIL", "ffmpeg or swdec is absent", "swdec-gpu")
        src = world.tmp / "gpu-req.yuv"
        enc = world.tmp / "gpu-req.mp4"
        if not enc.exists():
            write_yuv(src, 64, 64, "420", 8, 2)
            made = _ff(world, ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x64", "-i", str(src),
                               "-frames:v", "2", "-c:v", "libx264", "-preset", "veryfast", "-qp", "0", str(enc)])
            if made.returncode != 0:
                return Result("FAIL", made.stderr[-200:], "encode")
        proc, cmd = _pix(world, "swdec", [
            "--metrics", "psnr", "--decode", "gpu", "-v", "0",
            "-r", str(enc), "-s", "64x64", "-f", "420p:8",
            "-t", str(enc), "-s", "64x64", "-f", "420p:8",
        ])
        text = proc.stderr + proc.stdout
        if proc.returncode == 0 or scored(proc.stdout) or "not compiled" not in text or "backend=software" in text:
            return Result("FAIL", text[-400:], cmd)
        return Result("PASS", "explicit GPU on the software binary does not fall back", cmd)

    out.append(_case("empty-raw", "negative", ["smoke"], ["negative:empty"], empty))
    out.append(_case("short-raw", "negative", ["full"], ["negative:truncated"], short))
    out.append(_case("frame-mismatch", "negative", ["smoke"], ["negative:mismatch"], mismatch))
    out.append(_case("cpu-build-no-decoder", "negative", ["full"], ["negative:missing-decoder"], fake_mp4))
    out.append(_case("corrupt-bitstream", "negative", ["full"], ["negative:corrupt", "negative:truncated"], corrupt))
    out.append(_case("swdec-rejects-gpu", "backend", ["full"], ["decode:gpu"], swdec_gpu))
    return out


def _hdr_cases() -> list:
    def run(world: World) -> Result:
        ref, dist, size, fmt = _pair(world, 64, 64, "420", 10, 2, bias=40)
        plain, _ = _pix(world, "cpu", [
            "--metrics", "psnr", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        over, cmd = _pix(world, "cpu", [
            "--metrics", "psnr", "--hdr-transfer", "pq", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        missing, _ = _pix(world, "cpu", [
            "--metrics", "pu21-psnr", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        split, _ = _pix(world, "cpu", [
            "--metrics", "pu21-psnr,psnr", "--ref-hdr-transfer", "pq", "--dist-hdr-transfer", "hlg",
            "--hdr-peak-nits", "1000", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        color, _ = _pix(world, "cpu", [
            "--metrics", "pu21-psnr", "--hdr-transfer", "pq", "--hdr-primaries", "bt2020",
            "--hdr-matrix", "bt709", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        black, _ = _pix(world, "cpu", [
            "--metrics", "pu21-psnr", "--hdr-transfer", "hlg", "--hdr-peak-nits", "100",
            "--hdr-black-nits", "100", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        full, _ = _pix(world, "cpu", [
            "--metrics", "psnr,pu21-psnr", "--hdr-transfer", "pq", "--hdr-range", "full", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        limited, _ = _pix(world, "cpu", [
            "--metrics", "psnr,pu21-psnr", "--hdr-transfer", "pq", "--hdr-range", "limited", "-v", "0",
            "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt,
        ])
        problems = []
        if plain.returncode != 0 or over.returncode != 0:
            problems.append("override run failed")
        else:
            plain_psnr = [line for line in plain.stdout.splitlines() if "[ PSNR" in line]
            over_psnr = [line for line in over.stdout.splitlines() if "[ PSNR" in line]
            if plain_psnr != over_psnr or "transfer=pq (override)" not in over.stdout:
                problems.append("PQ override changed PSNR or hid provenance")
        if missing.returncode == 0 or "unspecified" not in (missing.stderr + missing.stdout):
            problems.append("missing transfer was scored")
        if split.returncode != 0 or "reference: transfer=pq (override)" not in split.stdout:
            problems.append("split transfer failed")
        elif "distorted: transfer=hlg (override)" not in split.stdout or "conventional:" not in split.stdout:
            problems.append("split provenance")
        elif scored(split.stdout) is False:
            problems.append("split produced no score")
        if color.returncode == 0 or "not a supported PQ/HLG color system" not in color.stderr:
            problems.append("bt2020/bt709 was accepted")
        if black.returncode == 0 or "display black" not in black.stderr:
            problems.append("black equal to peak was accepted")
        if full.returncode != 0 or limited.returncode != 0:
            problems.append("range run failed")
        else:
            def grab(text, token):
                for line in text.splitlines():
                    if token in line:
                        return line
                return ""
            if grab(full.stdout, "[ PSNR") != grab(limited.stdout, "[ PSNR"):
                problems.append("range changed PSNR")
            if grab(full.stdout, "[ PU21-PSNR:") == grab(limited.stdout, "[ PU21-PSNR:"):
                problems.append("range did not change PU21")
            if "range=full (override)" not in full.stdout or "range=limited (override)" not in limited.stdout:
                problems.append("range provenance")
        if problems:
            return Result("FAIL", "; ".join(problems), cmd)
        return Result("PASS", "override, split, range, and rejected HDR combinations", cmd)

    return [_case("hdr-matrix", "hdr", ["smoke"], ["hdr:pq", "hdr:hlg", "hdr:missing", "hdr:override", "hdr:range"], run)]


def _stable(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("Total frames:"):
            lines.append(line.split()[0])
        else:
            lines.append(line)
    return "\n".join(lines)


def _stability_cases() -> list:
    def repeat(world: World) -> Result:
        ref, dist, size, fmt = _pair(world, 64, 64, "420", 10, 2)
        args = [str(world.cpu), "--metrics", "psnr,ssim", "--threads", "2", "-v", "0",
                "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt]
        first = run_cmd(args, timeout=60)
        if first.returncode != 0:
            return Result("FAIL", first.stderr[-200:], "repeat")
        for _ in range(4):
            nxt = run_cmd(args, timeout=60)
            if nxt.returncode != 0 or _stable(nxt.stdout) != _stable(first.stdout):
                return Result("FAIL", "repeated stdout differed", "repeat")
        return Result("PASS", "5 sequential runs matched", " ".join(args))

    def concurrent(world: World) -> Result:
        import subprocess
        ref, dist, size, fmt = _pair(world, 64, 64, "420", 8, 2)
        args = [str(world.cpu), "--metrics", "psnr", "--threads", "1", "-v", "0",
                "-r", str(ref), "-s", size, "-f", "420p:8", "-t", str(dist), "-s", size, "-f", "420p:8"]
        procs = [subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        outs = []
        for proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                return Result("FAIL", "concurrent run timed out", "concurrent")
            outs.append((proc.returncode, stdout, stderr))
        if any(rc != 0 for rc, _, _ in outs) or _stable(outs[0][1]) != _stable(outs[1][1]):
            return Result("FAIL", "concurrent results differed", "concurrent")
        return Result("PASS", "2 concurrent processes matched", " ".join(args))

    def valgrind(world: World) -> Result:
        from shutil import which
        tool = which("valgrind")
        if not tool:
            return Result("SKIP", "valgrind is not installed", "valgrind")
        ref, dist, size, fmt = _pair(world, 32, 32, "420", 8, 1)
        proc = run_cmd([
            tool, "--error-exitcode=99", "--leak-check=full", str(world.cpu),
            "--metrics", "psnr", "-v", "0",
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ], timeout=180)
        if proc.returncode != 0:
            return Result("FAIL", (proc.stderr + proc.stdout)[-500:], "valgrind")
        return Result("PASS", "valgrind leak-check exit 0", "valgrind")

    def asan(world: World) -> Result:
        path = world.root / "build" / "swdec-asan" / "pixvqa"
        if not path.is_file():
            return Result("SKIP", "build/swdec-asan/pixvqa was not built", "asan")
        ref, dist, size, fmt = _pair(world, 32, 32, "420", 8, 1)
        proc = run_cmd([
            str(path), "--metrics", "psnr", "-v", "0",
            "-r", str(ref), "-s", size, "-f", "420p:8",
            "-t", str(dist), "-s", size, "-f", "420p:8",
        ], timeout=120, env={"ASAN_OPTIONS": "halt_on_error=1:detect_leaks=1"})
        if proc.returncode != 0:
            return Result("FAIL", (proc.stderr + proc.stdout)[-600:], "asan")
        return Result("PASS", "ASan software binary scored one frame", "asan")

    return [
        _case("repeat", "stability", ["full"], ["stability"], repeat),
        _case("concurrent", "stability", ["full"], ["stability"], concurrent),
        _case("valgrind", "stability", ["full"], ["sanitizer"], valgrind, optional=True),
        _case("asan", "stability", ["full"], ["sanitizer"], asan, optional=True),
    ]


def _regression_cases() -> list:
    scripts = [
        ("regression-pu21", "test_pu21_numeric.py", "cpu", {}, 180, ["metric:pu21-psnr", "metric:pu21-ssim", "metric:pu21-msssim"]),
        ("regression-hdr", "test_hdr_metadata.py", "swdec", {}, 240, ["hdr:stream", "hdr:conflict"]),
        ("regression-ffmpeg", "test_ffmpeg_input.py", "swdec", {}, 240, ["codec:regression"]),
        ("regression-dual-hevc", "test_dual_hevc.py", "swdec", {}, 180, ["codec:hevc"]),
        ("regression-vmaf", "test_vmaf_smoke.py", "swdec", {}, 240, ["metric:vmaf"]),
        ("regression-av1-vvc", "test_av1_vvc.py", "swdec", {}, 240, ["codec:av1", "codec:vvc"]),
    ]
    out = []

    def make(script, binary, env, timeout):
        def run(world: World) -> Result:
            if script == "test_vmaf_smoke.py" and world.vmaf_cli is None:
                return Result("FAIL", "VMAF CLI is absent", script)
            if not world.ffmpeg and binary == "swdec":
                return Result("FAIL", "ffmpeg is absent", script)
            return _script(world, script, env, timeout, binary)
        return run

    for id_, script, binary, env, timeout, tags in scripts:
        out.append(_case(id_, "regression", ["full"], tags, make(script, binary, env, timeout)))
    return out


def _hardware_cases() -> list:
    def gpu_metrics(world: World) -> Result:
        full = world.binary("full")
        cpu = world.binary("cpu")
        if full is None or cpu is None:
            return Result("FAIL", "cpu or full binary is absent", "gpu-metrics")
        ref, dist, size, fmt = _pair(world, 176, 176, "420", 10, 2)
        ref8, dist8, size8, fmt8 = _pair(world, 320, 240, "420", 8, 2)
        refv, distv, sizev, fmtv = _pair(world, 320, 240, "420", 10, 2)
        cpu_run = run_cmd([str(cpu), "--metrics", "psnr,ssim,msssim", "--metric-device", "cpu", "-v", "0",
                           "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt], timeout=90)
        gpu_run = run_cmd([str(full), "--metrics", "psnr,ssim,msssim", "--metric-device", "gpu", "-v", "0",
                           "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt], timeout=90)
        eight = run_cmd([str(full), "--metrics", "psnr,ssim", "--qcs", "--metric-device", "gpu", "-v", "0",
                         "-r", str(ref8), "-s", size8, "-f", fmt8, "-t", str(dist8), "-s", size8, "-f", fmt8], timeout=60)
        only_vmaf = run_cmd([str(full), "--metrics", "vmaf", "--model", str(world.model), "--metric-device", "gpu",
                             "-v", "0", "-r", str(ref8), "-s", size8, "-f", fmt8,
                             "-t", str(dist8), "-s", size8, "-f", fmt8], timeout=30)
        mixed = run_cmd([str(full), "--metrics", "psnr,vmaf", "--model", str(world.model), "--metric-device", "gpu",
                         "-v", "0", "-r", str(refv), "-s", sizev, "-f", fmtv, "-t", str(distv), "-s", sizev, "-f", fmtv], timeout=180)
        problems = []
        if cpu_run.returncode != 0 or gpu_run.returncode != 0:
            problems.append(gpu_run.stderr[-300:])
        else:
            if "PSNR: gpu" not in gpu_run.stdout or "SSIM: gpu" not in gpu_run.stdout or "MS-SSIM: gpu" not in gpu_run.stdout:
                problems.append("GPU summary did not select the GPU")
            for token in ("[ PSNR", "[ SSIM", "[ MS-SSIM"):
                c = next((ln for ln in cpu_run.stdout.splitlines() if token in ln), "")
                g = next((ln for ln in gpu_run.stdout.splitlines() if token in ln), "")
                if token == "[ PSNR" and c != g:
                    problems.append("PSNR CPU/GPU text differs")
                if token != "[ PSNR" and c != g:
                    # fp64 SSIM and MS-SSIM are allowed a 1e-4 absolute difference on the printed values.
                    import re
                    nums_c = [float(v) for v in re.findall(r"[-0-9.]+", c)]
                    nums_g = [float(v) for v in re.findall(r"[-0-9.]+", g)]
                    if len(nums_c) != len(nums_g) or any(abs(a - b) > 1e-4 for a, b in zip(nums_c, nums_g)):
                        problems.append(f"{token} CPU/GPU differ beyond 1e-4")
        eight_cpu = run_cmd([str(cpu), "--metrics", "psnr,ssim", "--qcs", "-v", "0",
                             "-r", str(ref8), "-s", size8, "-f", fmt8, "-t", str(dist8), "-s", size8, "-f", fmt8], timeout=60)
        if eight.returncode != 0 or eight_cpu.returncode != 0:
            problems.append("8-bit GPU PSNR/SSIM failed: " + (eight.stderr or eight_cpu.stderr)[-200:])
        else:
            for token in ("[ PSNR", "[ SSIM", "PSNR-QCS", "SSIM-QCS"):
                c = next((ln for ln in eight_cpu.stdout.splitlines() if token in ln), "")
                g = next((ln for ln in eight.stdout.splitlines() if token in ln), "")
                if c != g:
                    problems.append(f"8-bit {token} CPU/GPU text differs")
        if only_vmaf.returncode != 0 or "using CPU libvmaf" not in only_vmaf.stderr or "VMAF: cpu libvmaf" not in only_vmaf.stdout:
            problems.append("VMAF-only GPU did not fall back to CPU libvmaf")
        if mixed.returncode != 0 or "using CPU libvmaf" not in mixed.stderr or "VMAF: cpu libvmaf" not in mixed.stdout or "PSNR: gpu" not in mixed.stdout:
            problems.append("mixed VMAF did not stay on the CPU beside GPU PSNR")
        if problems:
            return Result("FAIL", "; ".join(problems), "gpu-metrics")
        return Result("PASS", "GPU metrics match the CPU within the existing tolerance and do not silently substitute", "gpu-metrics")

    def gpu_repeat(world: World) -> Result:
        full = world.binary("full")
        if full is None:
            return Result("FAIL", "full binary is absent", "gpu-repeat")
        ref, dist, size, fmt = _pair(world, 176, 176, "420", 10, 2)
        args = [str(full), "--metrics", "psnr", "--metric-device", "gpu", "-v", "0",
                "-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt]
        first = run_cmd(args, timeout=60)
        for _ in range(2):
            nxt = run_cmd(args, timeout=60)
            if nxt.returncode != 0 or _stable(nxt.stdout) != _stable(first.stdout):
                return Result("FAIL", "GPU PSNR stdout differed across repeats", "gpu-repeat")
        return Result("PASS", "3 sequential GPU PSNR runs matched", " ".join(args))

    def av1_vvc_gpu(world: World) -> Result:
        env = {"PIXVQA_FULL": str(world.full), "PIXVQA_RUN_GPU": "1"}
        return _script(world, "test_av1_vvc.py", env, 240, "swdec")

    return [
        _case("gpu-metrics", "hardware", ["hardware"], ["metric-device:gpu", "decode:cpu"], gpu_metrics),
        _case("gpu-repeat", "hardware", ["hardware"], ["stability"], gpu_repeat),
        _case("hardware-av1-hevc-vvc", "hardware", ["hardware"], ["codec:av1", "codec:hevc", "codec:vvc", "decode:gpu"], av1_vvc_gpu),
    ]


def cases() -> list:
    return _combinatorial() + _codec_cases() + _negative_cases() + _hdr_cases() + _stability_cases() + _regression_cases() + _hardware_cases()
