#!/usr/bin/env python3
"""Independent analytical BT.2020 CL/PQ/HLG -> PU21 CLI regression vectors.

Synthesizes genuine constant-luminance Yc'CbcCrc (not retagged NCL) using
forward transfer functions, quantizes full/limited YUV420 (8/10-bit), and
compares real PixVQA output to a separate scalar transfer/display/PU21 oracle.
--gpu-exe optionally enforces CUDA vs CPU numeric parity on the same vectors.
"""
from __future__ import annotations

import argparse
import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

KR, KG, KB = .2627, .6780, .0593
P = (.353487901, .3734658629, 8.277049286e-05, .9062562627,
     .09150303166, .9099517204, 596.3148142)
RGB_RE = re.compile(r"\[ PU21-PSNR\s+r:([0-9.]+) g:([0-9.]+) b:([0-9.]+) avg:([0-9.]+) \]")
LUMA_RE = {name: re.compile(rf"\[ {name}\s+y:([0-9.]+) u:n/a v:n/a avg:([0-9.]+) \]")
           for name in ("PU21-SSIM", "PU21-MS-SSIM")}
# Golden values anchor the existing SSIM implementation for two CL cases.
# The RGB-PSNR oracle below remains independently computed, not golden-only.
LUMA_GOLDEN = {"pq": (0.996211, 0.998917), "hlg": (0.988571, 0.996722)}
SAMPLES = (((.10, .30, .15), (.21, .09, .43)),
           ((.43, .12, .31), (.04, .40, .11)),
           ((.06, .08, .65), (.30, .32, .08)))


def pq_encode(x: float) -> float:
    a, b = 2610/16384, 2523/32
    c1, c2, c3 = 3424/4096, 2413/128, 2392/128
    p = x ** a
    return ((c1 + c2*p) / (1+c3*p)) ** b


def pq_decode(x: float) -> float:
    x = min(1, max(0, x))
    m1, m2 = 2610/16384, 2523/32
    c1, c2, c3 = 3424/4096, 2413/128, 2392/128
    p = x ** (1/m2)
    return 10000 * (max(p-c1, 0)/(c2-c3*p)) ** (1/m1)


def hlg_encode(x: float) -> float:
    a = .17883277
    b = 1-4*a
    c = 1-a*math.log(12-b)
    return math.sqrt(3*x) if x <= 1/12 else a*math.log(12*x-b)+c


def hlg_decode(x: float) -> float:
    a = .17883277
    b = 1-4*a
    c = 1-a*math.log(12-b)
    x = min(1, max(0, x))
    return x*x/3 if x <= .5 else (math.exp((x-c)/a)+b)/12


def encode_cl(rgb: tuple[float, ...], mode: str) -> tuple[float, float, float]:
    r, g, b = rgb
    l = KR*r + KG*g + KB*b
    tf = pq_encode if mode == "pq" else hlg_encode
    yc, rp, bp = tf(l), tf(r), tf(b)
    rd, bd = rp-yc, bp-yc
    return yc, bd/(1.9404 if bd <= 0 else 1.5816), rd/(1.7184 if rd <= 0 else .9936)


def quantize(ycc: tuple[float, ...], bits: int, range_: str) -> tuple[int, int, int]:
    y, cb, cr = ycc
    if range_ == "limited":
        sh = 1 << (bits-8)
        codes = (round(16*sh + 219*sh*y),
                 round(128*sh + 224*sh*cb), round(128*sh + 224*sh*cr))
    else:
        maxv = (1 << bits)-1
        codes = (round(y*maxv), round((cb+.5)*maxv), round((cr+.5)*maxv))
    assert all(0 <= c < 1 << bits for c in codes), codes
    return codes


def nominal(code: tuple[int, ...], bits: int, range_: str) -> tuple[float, float, float]:
    y, cb, cr = code
    if range_ == "limited":
        sh = 1 << (bits-8)
        return ((y-16*sh)/(219*sh), (cb-128*sh)/(224*sh), (cr-128*sh)/(224*sh))
    maxv = (1 << bits)-1
    return (y/maxv, cb/maxv-.5, cr/maxv-.5)


def display_cl(code: tuple[int, ...], bits: int, range_: str, mode: str,
               peak: float) -> tuple[float, float, float]:
    yc, cb, cr = nominal(code, bits, range_)
    r_nl = yc + (1.7184 if cr <= 0 else .9936)*cr
    b_nl = yc + (1.9404 if cb <= 0 else 1.5816)*cb
    decode = pq_decode if mode == "pq" else hlg_decode
    r, b, lum = decode(r_nl), decode(b_nl), decode(yc)
    g = max(0.0, (lum - KR*r - KB*b)/KG)
    if mode == "pq":
        return r,g,b
    scene_l = KR*r + KG*g + KB*b
    gamma = 1.2+.42*math.log10(peak/1000)
    gain = 0 if scene_l <= 0 else peak * scene_l**(gamma-1)
    return r*gain,g*gain,b*gain


def pu(x: float) -> float:
    x = max(.005, min(10000, x))
    p1,p2,p3,p4,p5,p6,p7 = P
    power = x**p4
    return max(0, p7*(((p1+p2*power)/(1+p3*power))**p5-p6))


def expected(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    delta = [(pu(v)-pu(w))**2 for v,w in zip(a,b)]
    return tuple(10*math.log10(256*256/mse) if mse else 999.99
                 for mse in (*delta, sum(delta)/3))


def write_yuv(path: Path, code: tuple[int, ...], bits: int) -> None:
    w = h = 32
    counts = (w*h, w*h//4, w*h//4)
    payload = b"".join((bytes([v])*n if bits == 8 else struct.pack("<H",v)*n)
                       for v,n in zip(code, counts))
    path.write_bytes(payload)


def run(exe: Path, ref: Path, dist: Path, bits: int, range_: str, mode: str,
        device: str, metrics: str = "pu21-psnr") -> tuple[subprocess.CompletedProcess[str], tuple[float,...] | None]:
    cmd = [str(exe), "--metrics", metrics, "--metric-device", device,
           "--hdr-transfer", mode, "--hdr-primaries", "bt2020",
           "--hdr-matrix", "bt2020c", "--hdr-range", range_,
           "-r", str(ref), "-s", "32x32", "-f", f"420p:{bits}",
           "-t", str(dist), "-s", "32x32", "-f", f"420p:{bits}",
           "--frames", "1", "-v", "0"]
    process = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
    found = RGB_RE.search(process.stdout)
    return process, tuple(map(float, found.groups())) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-exe", type=Path, default=Path("build/cpu/pixvqa"))
    parser.add_argument("--gpu-exe", type=Path, default=None,
                        help="Full CUDA binary; when supplied, GPU parity is mandatory")
    args = parser.parse_args()
    cpu = args.cpu_exe.resolve()
    gpu = args.gpu_exe.resolve() if args.gpu_exe else None
    if not cpu.is_file() or (gpu and not gpu.is_file()):
        parser.error("requested PixVQA binary is absent")
    checks = 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_cl_analytical_") as d:
        directory = Path(d)
        for mode in ("pq", "hlg"):
            for bits in (8, 10):
                for range_ in ("limited", "full"):
                    for i, (a, b) in enumerate(SAMPLES):
                        a_code = quantize(encode_cl(a,mode), bits, range_)
                        b_code = quantize(encode_cl(b,mode), bits, range_)
                        ra,rb = directory/"ref.yuv",directory/"dist.yuv"
                        write_yuv(ra,a_code,bits)
                        write_yuv(rb,b_code,bits)
                        target = expected(display_cl(a_code,bits,range_,mode,1000),
                                          display_cl(b_code,bits,range_,mode,1000))
                        result, metrics = run(cpu,ra,rb,bits,range_,mode,"cpu")
                        label = f"{mode} {bits}bit {range_} vector={i}"
                        if result.returncode or metrics is None:
                            raise AssertionError(f"{label} CPU run failed: rc={result.returncode} "
                                                 f"{result.stderr[-1200:]} {result.stdout[-400:]}")
                        for channel,got,want in zip(("R","G","B","pooled"),metrics,target):
                            if abs(got-want) > 1.5e-4:
                                raise AssertionError(f"{label} {channel} got={got} expected={want}")
                        if gpu:
                            gr, gm = run(gpu,ra,rb,bits,range_,mode,"gpu")
                            if gr.returncode or gm is None:
                                raise AssertionError(f"{label} GPU failed: {gr.stderr[-1200:]}")
                            for channel,gv,cv in zip(("R","G","B","pooled"),gm,metrics):
                                if abs(gv-cv) > max(1e-4, abs(cv)*1e-6):
                                    raise AssertionError(f"{label} GPU {channel} {gv} != CPU {cv}")
                        checks += 1
                        print(f"PASS {label}")
        # Regression anchors for display-luma PU21 SSIM/MS-SSIM on real CL
        # input; the independent RGB reconstruction/PSNR oracle runs above.
        all_metrics = "pu21-psnr,pu21-ssim,pu21-msssim"
        for mode in ("pq", "hlg"):
            a_code = quantize(encode_cl(SAMPLES[0][0],mode), 10, "limited")
            b_code = quantize(encode_cl(SAMPLES[0][1],mode), 10, "limited")
            ra, rb = directory/"ref.yuv", directory/"dist.yuv"
            write_yuv(ra,a_code,10)
            write_yuv(rb,b_code,10)
            scored, _ = run(cpu,ra,rb,10,"limited",mode,"cpu",all_metrics)
            if scored.returncode:
                raise AssertionError(f"{mode} CL PU21 SSIM launch failed: {scored.stderr}")
            for name, target in zip(LUMA_RE, LUMA_GOLDEN[mode]):
                found = LUMA_RE[name].search(scored.stdout)
                if not found or abs(float(found.group(1))-target) > 1e-5:
                    raise AssertionError(f"{mode} CL {name} regression: {found and found.group(1)} expected {target}")
            if gpu:
                scored_gpu, _ = run(gpu,ra,rb,10,"limited",mode,"gpu",all_metrics)
                if scored_gpu.returncode:
                    raise AssertionError(f"{mode} GPU PU21 SSIM failed: {scored_gpu.stderr}")
                for name in LUMA_RE:
                    cv = float(LUMA_RE[name].search(scored.stdout).group(1))
                    gm = LUMA_RE[name].search(scored_gpu.stdout)
                    if not gm or abs(float(gm.group(1))-cv) > 1e-4:
                        raise AssertionError(f"{mode} GPU {name} parity mismatch")
            checks += 1
            write_yuv(rb,a_code,10)
            identical, _ = run(cpu,ra,rb,10,"limited",mode,"cpu",all_metrics)
            if identical.returncode:
                raise AssertionError(f"{mode} identical CL test failed: {identical.stderr}")
            for name in LUMA_RE:
                found = LUMA_RE[name].search(identical.stdout)
                if not found or abs(float(found.group(1))-1.0) > 1e-6:
                    raise AssertionError(f"{mode} identical CL {name} not 1.0")
            checks += 1
            print(f"PASS {mode} CL PU21-SSIM/MS-SSIM scored and identical baselines")
        # Code-domain comparisons remain CL/CL-only and must reject CL/NCL.
        common = [str(cpu), "--metrics", "psnr", "--hdr-transfer", "pq",
                  "--hdr-primaries", "bt2020", "--hdr-range", "limited",
                  "-r", str(ra), "-s", "32x32", "-f", "420p:10",
                  "-t", str(rb), "-s", "32x32", "-f", "420p:10", "-v", "0"]
        # Recreate known genuine 10-bit PQ samples for the gating test.
        write_yuv(ra, quantize(encode_cl(SAMPLES[0][0],"pq"),10,"limited"),10)
        write_yuv(rb, quantize(encode_cl(SAMPLES[0][1],"pq"),10,"limited"),10)
        reject = subprocess.run(common+["--ref-hdr-matrix", "bt2020c",
                                        "--dist-hdr-matrix", "bt2020nc"],
                                text=True,capture_output=True,timeout=60)
        if reject.returncode != 2 or "incompatible BT.2020 matrices" not in reject.stderr:
            raise AssertionError(f"CL/NCL direct PSNR was not rejected: {reject.stderr}")
        checks += 1
        matched = subprocess.run(common+["--hdr-matrix","bt2020c"],
                                 text=True,capture_output=True,timeout=60)
        if matched.returncode or not re.search(r"Total frames:\s*1",matched.stdout):
            raise AssertionError(f"CL/CL sample PSNR failed: {matched.stderr}")
        checks += 1
        # G clipping is signaled; the metric remains computable with clamped G.
        write_yuv(ra,(400,1000,1000),10)
        write_yuv(rb,(400,1000,1000),10)
        clamp, _ = run(cpu,ra,rb,10,"limited","pq","cpu")
        if clamp.returncode or "clipped out-of-gamut CL" not in clamp.stderr:
            raise AssertionError(f"out-of-gamut CL not reported: {clamp.stderr}")
        checks += 1
    print(f"PASS {checks} analytic/CLI checks (GPU parity={'yes' if gpu else 'not executed'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
