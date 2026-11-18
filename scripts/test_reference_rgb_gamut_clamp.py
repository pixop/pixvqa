#!/usr/bin/env python3
"""Reference RGB gamut clamp: delta-only identity, subsampling, CLI, and regression checks."""

from __future__ import annotations

import atexit
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BIN = os.path.join(ROOT, "build", "cpu", "pixvqa")
SELFTEST = os.path.join(ROOT, "build", "cpu", "selftest")
BIN = os.environ.get("PIXVQA_BIN", DEFAULT_BIN if os.path.isfile(DEFAULT_BIN) else os.path.join(ROOT, "bin", "pixvqa"))

_TMP: tempfile.TemporaryDirectory[str] | None = None
_TMP_SEQ = 0


def _tmp_dir() -> str:
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory(prefix="pixvqa-ref-gamut-")
        atexit.register(_TMP.cleanup)
    return _TMP.name


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([BIN, *args], capture_output=True, text=True, cwd=ROOT)


def write_tmp(data: bytes, suffix: str) -> str:
    global _TMP_SEQ
    _TMP_SEQ += 1
    path = os.path.join(_tmp_dir(), f"case{_TMP_SEQ}{suffix}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def pack_plane(vals: list[int]) -> bytes:
    return struct.pack(f"<{len(vals)}H", *vals)


def yuv_frame(fmt: str, w: int, h: int, y_fill: int, u_fill: int | list[int], v_fill: int) -> bytes:
    """fmt like 420,422,444 (no p suffix)."""
    y_n = w * h
    y = pack_plane([y_fill] * y_n)
    if fmt == "444":
        cw, ch = w, h
    elif fmt == "422":
        cw, ch = w // 2, h
    else:
        cw, ch = w // 2, h // 2
    if isinstance(u_fill, list):
        u = pack_plane(u_fill)
    else:
        u = pack_plane([u_fill] * (cw * ch))
    v = pack_plane([v_fill] * (cw * ch))
    return y + u + v


def checker_u(w: int, h: int, fmt: str, a: int, b: int) -> list[int]:
    if fmt == "444p":
        cw, ch = w, h
    elif fmt == "422p":
        cw, ch = w // 2, h
    else:
        cw, ch = w // 2, h // 2
    return [a if (ci + cj) % 2 == 0 else b for cj in range(ch) for ci in range(cw)]


def psnr_line(stdout: str) -> dict[str, float]:
    for line in stdout.splitlines():
        if line.strip().startswith("[ PSNR"):
            out: dict[str, float] = {}
            for token in line.replace("[", " ").replace("]", " ").split():
                if ":" in token:
                    k, v = token.split(":", 1)
                    out[k] = float("inf") if v.lower() in ("inf", "999.99") else float(v)
            return out
    raise AssertionError(f"PSNR summary missing:\n{stdout}")


def expect_inf(psnr: dict[str, float], planes: tuple[str, ...] = ("y", "u", "v")) -> bool:
    return all(psnr.get(p, 0) > 1e6 for p in planes)


def main() -> int:
    if not os.path.isfile(BIN):
        if os.environ.get("PIXVQA_TEST_SKIP_NO_BIN") == "1":
            print(f"SKIP: pixvqa binary not found at {BIN}")
            return 0
        print(f"FAIL: pixvqa binary not found at {BIN}", file=sys.stderr)
        return 1

    ok = fail = 0

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal ok, fail
        if cond:
            print(f"OK   {name}")
            ok += 1
        else:
            print(f"FAIL {name} {detail}")
            fail += 1

    base_psnr = ["-v", "0", "--metrics", "psnr", "-n", "1"]
    clamp = [
        "--reference-rgb-gamut-clamp",
        "--reference-matrix",
        "bt709",
        "--reference-range",
        "limited",
    ]

    if os.path.isfile(SELFTEST):
        st = subprocess.run([SELFTEST], capture_output=True, text=True, cwd=ROOT)
        check("selftest-exit", st.returncode == 0, st.stderr[-400:] if st.returncode else "")
        for matrix in ("bt2020nc", "bt709", "bt601", "smpte240m"):
            check(
                f"analytic-oracle-{matrix}",
                st.returncode == 0 and f"OK   ref_gamut_analytic_{matrix}" in st.stdout,
                st.stderr[-300:] if st.returncode else "",
            )
    else:
        check("selftest-present", False, f"missing {SELFTEST}")

    def pack44410(w: int, h: int, y: int, u: int, v: int) -> bytes:
        p = pack_plane
        return p([y] * (w * h)) + p([u] * (w * h)) + p([v] * (w * h))

    extreme = pack44410(8, 8, 64, 64, 512)
    for matrix, yc, uc, vc in (
        ("bt2020nc", 113, 485, 478),
        ("bt709", 123, 480, 474),
        ("bt601", 152, 461, 447),
        ("smpte240m", 134, 473, 467),
    ):
        src = write_tmp(extreme, ".yuv")
        golden = write_tmp(pack44410(8, 8, yc, uc, vc), ".yuv")
        proc = run(
            [
                "-r",
                src,
                "-t",
                golden,
                "-s",
                "8x8",
                "-f",
                "444p:10",
                *base_psnr,
                "--reference-rgb-gamut-clamp",
                "--reference-matrix",
                matrix,
                "--reference-range",
                "limited",
            ]
        )
        check(
            f"analytic-cli-{matrix}-psnr-inf",
            proc.returncode == 0 and expect_inf(psnr_line(proc.stdout)),
            proc.stdout[-200:] if proc.returncode == 0 else proc.stderr,
        )

    w = h = 8
    matrix_up_loc = [
        ("420p", "left", None),
        ("420p", "center", ["--hdr-chroma-loc", "center"]),
        ("420p", "topleft", ["--hdr-chroma-loc", "topleft"]),
        ("422p", "left", None),
        ("422p", "center", ["--hdr-chroma-loc", "center"]),
        ("422p", "topleft", ["--hdr-chroma-loc", "topleft"]),
    ]
    for fmt, label, hdr_extra in matrix_up_loc:
        ref = write_tmp(
            yuv_frame(fmt.replace("p", ""), w, h, 502, checker_u(w, h, fmt, 470, 554), 512),
            ".yuv",
        )
        f_arg = f"{fmt}:10"
        for up, up_arg in (("bilinear", []), ("nearest", ["--hdr-chroma-up", "nearest"])):
            args = [
                "-r",
                ref,
                "-t",
                ref,
                "-s",
                f"{w}x{h}",
                "-f",
                f_arg,
                *base_psnr,
                *clamp,
                *up_arg,
            ]
            if hdr_extra:
                args.extend(hdr_extra)
            proc = run(args)
            check(
                f"identity-{fmt}-{label}-{up}",
                proc.returncode == 0 and expect_inf(psnr_line(proc.stdout)),
                proc.stdout[-400:] if proc.returncode == 0 else proc.stderr,
            )

    ref420 = write_tmp(
        yuv_frame("420", w, h, 502, checker_u(w, h, "420p", 470, 554), 512),
        ".yuv",
    )
    check(
        "raw-missing-matrix",
        run(["-r", ref420, "-t", ref420, "-s", "8x8", "-f", "420p:10", *base_psnr, *clamp[:1]]).returncode
        == 2,
    )

    check(
        "reject-ictcp-matrix",
        run(
            [
                "-r",
                ref420,
                "-t",
                ref420,
                "-s",
                "8x8",
                "-f",
                "420p:10",
                *base_psnr,
                "--reference-rgb-gamut-clamp",
                "--reference-matrix",
                "ictcp",
                "--reference-range",
                "limited",
            ]
        ).returncode
        == 2,
    )

    check(
        "reject-nr-only-clamp",
        run(
            [
                "-t",
                ref420,
                "-s",
                "8x8",
                "-f",
                "420p:10",
                "--metrics",
                "cvqa-hdr-nr",
                *clamp,
            ]
        ).returncode
        == 2,
    )

    partial = write_tmp(
        b"".join(
            [
                pack_plane([64, 502, 502, 502]),
                pack_plane([300]),
                pack_plane([512]),
            ]
        ),
        ".yuv",
    )
    proc_partial = run(
        [
            "-r",
            partial,
            "-t",
            partial,
            "-s",
            "2x2",
            "-f",
            "420p:10",
            *base_psnr,
            *clamp,
        ]
    )
    partial_golden = write_tmp(
        b"".join(
            [
                pack_plane([64, 502, 502, 502]),
                pack_plane([349]),
                pack_plane([512]),
            ]
        ),
        ".yuv",
    )
    proc_oracle = run(
        [
            "-r",
            partial,
            "-t",
            partial_golden,
            "-s",
            "2x2",
            "-f",
            "420p:10",
            *base_psnr,
            *clamp,
        ]
    )
    check(
        "partial-420-oracle-u-exact",
        proc_oracle.returncode == 0 and expect_inf(psnr_line(proc_oracle.stdout), ("u",)),
        proc_oracle.stdout[-300:] if proc_oracle.returncode == 0 else proc_oracle.stderr,
    )
    check(
        "partial-420-clamped-u-not-buggy",
        proc_partial.returncode == 0
        and 20.0 < psnr_line(proc_partial.stdout)["u"] < 35.0
        and psnr_line(proc_partial.stdout)["y"] > 30.0,
        proc_partial.stderr if proc_partial.returncode else "",
    )

    for matrix in ("bt2020nc", "bt601", "smpte240m"):
        proc_m = run(
            [
                "-r",
                ref420,
                "-t",
                ref420,
                "-s",
                "8x8",
                "-f",
                "420p:10",
                *base_psnr,
                "--reference-rgb-gamut-clamp",
                "--reference-matrix",
                matrix,
                "--reference-range",
                "limited",
            ]
        )
        check(
            f"identity-matrix-{matrix}",
            proc_m.returncode == 0 and expect_inf(psnr_line(proc_m.stdout)),
            proc_m.stderr if proc_m.returncode else "",
        )

    def yuv_frame_8(fmt: str, w: int, h: int, y: int, u: int, v: int) -> bytes:
        yb = bytes([y] * (w * h))
        if fmt == "444":
            cw, ch = w, h
        elif fmt == "422":
            cw, ch = w // 2, h
        else:
            cw, ch = w // 2, h // 2
        return yb + bytes([u] * (cw * ch)) + bytes([v] * (cw * ch))

    grey8 = write_tmp(yuv_frame_8("444", 4, 4, 235, 128, 128), ".yuv")
    proc_8_lim = run(
        [
            "-r",
            grey8,
            "-t",
            grey8,
            "-s",
            "4x4",
            "-f",
            "444p:8",
            *base_psnr,
            "--reference-rgb-gamut-clamp",
            "--reference-matrix",
            "bt601",
            "--reference-range",
            "limited",
        ]
    )
    check(
        "identity-8bit-limited-bt601",
        proc_8_lim.returncode == 0 and expect_inf(psnr_line(proc_8_lim.stdout)),
    )

    cw8, ch8 = 8 // 2, 8
    chroma_u8 = [128 if i % 2 == 0 else 140 for i in range(cw8 * ch8)]
    chroma8 = write_tmp(
        bytes([160] * (8 * 8)) + bytes(chroma_u8) + bytes([128] * (cw8 * ch8)),
        ".yuv",
    )
    proc_8_full = run(
        [
            "-r",
            chroma8,
            "-t",
            chroma8,
            "-s",
            "8x8",
            "-f",
            "422p:8",
            *base_psnr,
            "--reference-rgb-gamut-clamp",
            "--reference-matrix",
            "bt709",
            "--reference-range",
            "full",
        ]
    )
    check(
        "identity-8bit-full-422-chroma",
        proc_8_full.returncode == 0 and expect_inf(psnr_line(proc_8_full.stdout)),
    )

    chroma10 = write_tmp(
        yuv_frame("420", 8, 8, 502, checker_u(8, 8, "420p", 420, 600), 512),
        ".yuv",
    )
    proc_10_full = run(
        [
            "-r",
            chroma10,
            "-t",
            chroma10,
            "-s",
            "8x8",
            "-f",
            "420p:10",
            *base_psnr,
            "--reference-rgb-gamut-clamp",
            "--reference-matrix",
            "bt709",
            "--reference-range",
            "full",
        ]
    )
    check(
        "identity-10bit-full-420-chroma",
        proc_10_full.returncode == 0 and expect_inf(psnr_line(proc_10_full.stdout)),
    )

    proc_pq = run(
        [
            "-r",
            ref420,
            "-t",
            ref420,
            "-s",
            "8x8",
            "-f",
            "420p:10",
            *base_psnr,
            "--hdr-transfer",
            "pq",
            "--reference-rgb-gamut-clamp",
            "--reference-matrix",
            "bt2020nc",
            "--reference-range",
            "limited",
        ]
    )
    check("identity-pq-code-domain", proc_pq.returncode == 0 and expect_inf(psnr_line(proc_pq.stdout)))

    proc_hlg = run(
        [
            "-r",
            ref420,
            "-t",
            ref420,
            "-s",
            "8x8",
            "-f",
            "420p:10",
            *base_psnr,
            "--hdr-transfer",
            "hlg",
            "--reference-rgb-gamut-clamp",
            "--reference-matrix",
            "bt2020nc",
            "--reference-range",
            "limited",
        ]
    )
    check("identity-hlg-code-domain", proc_hlg.returncode == 0 and expect_inf(psnr_line(proc_hlg.stdout)))

    check(
        "reject-bt2020cl",
        run(
            [
                "-r",
                ref420,
                "-t",
                ref420,
                "-s",
                "8x8",
                "-f",
                "420p:10",
                *base_psnr,
                "--reference-rgb-gamut-clamp",
                "--reference-matrix",
                "bt2020c",
                "--reference-range",
                "limited",
            ]
        ).returncode
        == 2,
    )

    check(
        "reject-missing-range",
        run(
            [
                "-r",
                ref420,
                "-t",
                ref420,
                "-s",
                "8x8",
                "-f",
                "420p:10",
                *base_psnr,
                "--reference-rgb-gamut-clamp",
                "--reference-matrix",
                "bt709",
            ]
        ).returncode
        == 2,
    )

    def reject_geometry(
        name: str, data: bytes, size: str, fmt: str, *needles: str, rc_ok: tuple[int, ...] = (2,)
    ) -> None:
        path = write_tmp(data, ".yuv")
        proc = run(
            [
                "-r",
                path,
                "-t",
                path,
                "-s",
                size,
                "-f",
                fmt,
                *base_psnr,
                *clamp,
            ]
        )
        err_text = proc.stderr + proc.stdout
        ok_msg = proc.returncode in rc_ok and any(n in err_text for n in needles)
        check(
            name,
            ok_msg,
            err_text[-240:] if not ok_msg else "",
        )

    reject_geometry(
        "reject-odd-422-1x2",
        struct.pack("<HH", 502, 502),
        "1x2",
        "422p:10",
        "422p requires even width",
        "4:2:2 requires even width",
        rc_ok=(1, 2),
    )
    reject_geometry(
        "reject-odd-422-3x2",
        struct.pack("<6H", *([502] * 6)) + struct.pack("<2H", 512, 512) + struct.pack("<2H", 512, 512),
        "3x2",
        "422p:10",
        "422p requires even width",
        "4:2:2 requires even width",
        rc_ok=(1, 2),
    )
    reject_geometry(
        "reject-odd-420-3x3",
        struct.pack("<9H", *([502] * 9))
        + struct.pack("<4H", *([512] * 4))
        + struct.pack("<4H", *([512] * 4)),
        "3x3",
        "420p:10",
        "420p requires even width and height",
        "4:2:0 requires even width and height",
        rc_ok=(1, 2),
    )

    ok422_2x1 = write_tmp(
        struct.pack("<2H", 502, 502) + struct.pack("<1H", 512) + struct.pack("<1H", 512),
        ".yuv",
    )
    proc_422_ht = run(
        [
            "-r",
            ok422_2x1,
            "-t",
            ok422_2x1,
            "-s",
            "2x1",
            "-f",
            "422p:10",
            *base_psnr,
            *clamp,
        ]
    )
    check(
        "accept-422-2x1-odd-height",
        proc_422_ht.returncode == 0 and expect_inf(psnr_line(proc_422_ht.stdout)),
        proc_422_ht.stderr if proc_422_ht.returncode else "",
    )

    ok444_1x1 = write_tmp(struct.pack("<3H", 502, 512, 512), ".yuv")
    proc_444 = run(
        [
            "-r",
            ok444_1x1,
            "-t",
            ok444_1x1,
            "-s",
            "1x1",
            "-f",
            "444p:10",
            *base_psnr,
            *clamp,
        ]
    )
    check(
        "accept-444-1x1",
        proc_444.returncode == 0 and expect_inf(psnr_line(proc_444.stdout)),
        proc_444.stderr if proc_444.returncode else "",
    )

    hot = write_tmp(yuv_frame("444", 4, 4, 512, 512, 1023), ".yuv")
    neutral = write_tmp(yuv_frame("444", 4, 4, 512, 512, 512), ".yuv")
    proc_plain = run(["-r", hot, "-t", neutral, "-s", "4x4", "-f", "444p:10", *base_psnr])
    clamp_flags = [
        *clamp[:1],
        "--reference-matrix",
        "bt709",
        "--reference-range",
        "full",
    ]
    proc_clamp = run(
        ["-r", hot, "-t", neutral, "-s", "4x4", "-f", "444p:10", *base_psnr, *clamp_flags]
    )
    check("clamp-plain-runs", proc_plain.returncode == 0, proc_plain.stderr)
    check("clamp-runs", proc_clamp.returncode == 0, proc_clamp.stderr)
    if proc_plain.returncode == 0 and proc_clamp.returncode == 0:
        plain = psnr_line(proc_plain.stdout)
        clamped = psnr_line(proc_clamp.stdout)
        check(
            "clamp-improves-avg",
            clamped["avg"] > plain["avg"] + 1.0,
            f"plain={plain['avg']} clamp={clamped['avg']}",
        )
    else:
        check("clamp-improves-avg", False, "pixvqa run failed")

    proc_cv = run(
        [
            "-r",
            ref420,
            "-t",
            ref420,
            "-s",
            "8x8",
            "-f",
            "420p:10",
            *base_psnr,
            "--metrics",
            "psnr,cvvdp",
            *clamp,
        ]
    )
    check(
        "cvvdp-reject",
        proc_cv.returncode == 2 and "not supported with cvvdp" in proc_cv.stderr,
        proc_cv.stderr[-300:] if proc_cv.returncode != 2 else "",
    )

    print(f"{ok} passed, {fail} failed (binary={BIN})")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())