#!/usr/bin/env python3
"""Focused production safety tests; no GPU/model assets required."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def test_csv_writer() -> None:
    from pixvqa_cvvdp.python_api import write_official_csv
    from pixvqa_cvvdp.adapter import parse_result_csv
    pairs = [
        ("distorted.mkv", "reference.mkv"),
        ('distorted, "special".mkv', 'reference, "another".mkv'),
        ("test\nwith-newline.mkv", "ref\nwith-newline.mkv"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "result.csv"
        for test_path, ref_path in pairs:
            write_official_csv(p, test_path, ref_path, 9.311432838439941)
            with p.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            assert rows == [
                ["test", "reference", "cvvdp"],
                [test_path, ref_path, "9.311432838439941"],
            ], rows
            assert parse_result_csv(p) == 9.311432838439941
    print("OK   hardening-csv-roundtrip-special-paths")


def test_raw_sample_policy() -> None:
    from pixvqa_cvvdp.adapter import (
        AdapterError, _raw_sample_frame_indices, validate_raw_yuv,
    )
    assert _raw_sample_frame_indices(100, 0) == list(range(8)) + [99]
    assert _raw_sample_frame_indices(100, 1) == [0]
    assert _raw_sample_frame_indices(100, 4) == [0, 1, 2, 3]
    assert _raw_sample_frame_indices(100, 12) == list(range(8)) + [11]
    assert _raw_sample_frame_indices(5, 0) == list(range(5))
    assert _raw_sample_frame_indices(0, 0) == []
    with tempfile.TemporaryDirectory() as tmp:
        frame_size = 16 * 16 * 3
        good = bytes(frame_size)
        bad = bytearray(frame_size)
        bad[1] = 4  # little endian sample 1024, beyond 10-bit range
        path = Path(tmp) / "raw.yuv"
        # A cap of 3 ignores a malformed frame outside the processed prefix.
        path.write_bytes(good * 3 + bytes(bad) + good)
        assert validate_raw_yuv(str(path), 16, 16, 3) == 5
        for cap in (0, 4, 5):
            try:
                validate_raw_yuv(str(path), 16, 16, cap)
            except AdapterError as exc:
                assert "10-bit" in str(exc)
            else:
                raise AssertionError(f"invalid inspected frame accepted (cap={cap})")
        # Bounded full-file preflight does NOT inspect every middle frame.
        path.write_bytes(good * 40 + bytes(bad) + good * 59)
        assert validate_raw_yuv(str(path), 16, 16, 0) == 100
        # Positive prefix: last inspected frame must be frame N-1, not file end.
        path.write_bytes(good * 11 + bytes(bad) + good * 88)
        for cap in (12, 0):
            if cap == 0:  # middle frame is deliberately outside sampled indices
                assert validate_raw_yuv(str(path), 16, 16, cap) == 100
                continue
            try:
                validate_raw_yuv(str(path), 16, 16, cap)
            except AdapterError as exc:
                assert "10-bit" in str(exc)
            else:
                raise AssertionError("bad final processed frame accepted")
    print("OK   hardening-bounded-raw-preflight")


def test_cpp_resolution_lifetime() -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        print("SKIP hardening-cpp-resolution (no C++ compiler)")
        return
    source = r'''
#include "pixvqa/metric_resolve.hpp"
#include <iostream>
#include <string>
int main() {
    for (const auto& group : {"sdr", "pq", "hlg"}) {
        pixvqa::MetricTransferCli cli;
        cli.metric_group = group;
        pixvqa::CvvdpResolveState cvvdp;
        cvvdp.selected = true;
        std::string err;
        if (!pixvqa::resolve_cvvdp_parameters(cli, {}, {}, cvvdp, err)) {
            std::cerr << err << "\n";
            return 1;
        }
        if (cvvdp.interpretation != group || !cvvdp.config_resolved) return 2;
    }
    pixvqa::MetricTransferCli cli;
    cli.preset_hdr = true;
    pixvqa::CvvdpResolveState cvvdp;
    cvvdp.selected = true;
    std::string err;
    if (!pixvqa::resolve_cvvdp_parameters(cli, {}, {}, cvvdp, err)) return 3;
    if (cvvdp.interpretation != "pq") return 4;
    return 0;
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "resolver.cpp"
        exe = Path(tmp) / "resolver"
        cpp.write_text(source, encoding="utf-8")
        cmd = [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-I", str(ROOT / "include"),
               str(cpp), str(ROOT / "src" / "metric_resolve.cpp"), "-o", str(exe)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        proc = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
        for token in ("--metrics sdr", "--metrics pq", "--metrics hlg", "--preset hdr"):
            assert token in proc.stderr, (token, proc.stderr)
    print("OK   hardening-cpp-cvvdp-group-log-source")


if __name__ == "__main__":
    test_csv_writer()
    test_raw_sample_policy()
    test_cpp_resolution_lifetime()
