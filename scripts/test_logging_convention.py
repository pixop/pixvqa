#!/usr/bin/env python3
"""Dependency-free regression for PixVQA child-stderr normalization.

This compiles only the header-only relay helper. A full CUDA/learned-metric
runtime test is intentionally left to the workstation.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include "backend_log.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
  using pixvqa::backend_log::Relay;
  Relay cvvdp("CVVDP");
  std::vector<std::string> lines;
  auto emit = [&](const std::string& s) { lines.push_back(s); };
  const std::string cvvdp_input =
      "[INFO] Running on device: cuda:0\n"
      "WARNING:root: The mean color value is less than 1 - the image may not be scaled in absolute photometric units!\n"
      "pixvqa-cvvdp: warning: selected CPU\n"
      "first fragmented upstream message\n";
  for (size_t pos = 0; pos < cvvdp_input.size(); pos += 7) {
    cvvdp.consume(cvvdp_input.data() + pos, std::min(size_t(7), cvvdp_input.size() - pos), emit);
  }
  cvvdp.finish(emit);
  assert(lines.size() == 4);
  assert(lines[0].find("pixvqa-cvvdp: info: upstream: [INFO]") == 0);
  assert(lines[1].find("pixvqa-cvvdp: warning: upstream: WARNING:root:") == 0);
  assert(lines[2] == "pixvqa-cvvdp: warning: selected CPU\n");
  assert(lines[3] == "pixvqa-cvvdp: info: upstream: first fragmented upstream message\n");

  Relay nr("CVQA-HDR-NR");
  lines.clear();
  const std::string nr_input =
      "No pretrained weights loaded for model 'ViT-B-16-SigLIP2-384'\n"
      "pixvqa-cvqa-hdr-nr: error: checkpoint not found\n"
      "pixvqa-cvqahdr: invalid checkpoint\n"
      "Traceback (most recent call last):";
  nr.consume(nr_input.data(), nr_input.size(), emit);
  nr.finish(emit);
  assert(lines.size() == 4);
  assert(lines[0].find("pixvqa-cvqa-hdr-nr: info: upstream open_clip: No pretrained weights") == 0);
  assert(lines[1] == "pixvqa-cvqa-hdr-nr: error: checkpoint not found\n");
  assert(lines[2] == "pixvqa-cvqa-hdr-nr: error: invalid checkpoint\n");
  assert(lines[3].find("pixvqa-cvqa-hdr-nr: error: upstream: Traceback") == 0);
  std::cout << "PASS backend stderr normalization, fragmentation, final flush, severity and identity\n";
}
'''


def main() -> int:
    compiler = shutil.which("g++") or shutil.which("c++")
    if not compiler:
        print("SKIP no C++ compiler")
        return 77
    with tempfile.TemporaryDirectory(prefix="pixvqa_log_test_") as temp:
        src = Path(temp) / "test.cpp"
        exe = Path(temp) / "test"
        src.write_text(HARNESS, encoding="utf-8")
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "src"), str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    # Exercise the actual PixVQA subprocess relay with a synthetic backend.
    # No third-party Python packages, video codecs or GPU are needed.
    pixvqa = ROOT / "build" / "cpu" / "pixvqa"
    if pixvqa.is_file():
        with tempfile.TemporaryDirectory(prefix="pixvqa_log_smoke_") as temp:
            base = Path(temp)
            raw = base / "ref.yuv"
            raw.write_bytes((512).to_bytes(2, "little") * (64 * 64 + 2 * 32 * 32))
            fake = base / "fake_python"
            fake.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' 'WARNING:root: The mean color value is less than 1' >&2\n"
                "printf '%s\\n' '[INFO] Running on device: cpu' >&2\n"
                "printf '%s\\n' '{\"metric\":\"cvvdp\",\"jod\":7.0}'\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            command = [str(pixvqa), "-r", str(raw), "-t", str(raw),
                       "-s", "64x64", "-f", "420p:10", "--fps", "24",
                       "--metrics", "cvvdp", "--cvvdp-interpretation", "pq",
                       "--cvvdp-display", "standard_hdr_pq",
                       "--cvvdp-python", str(fake), "-n", "1", "-v", "0"]
            done = subprocess.run(command, capture_output=True, text=True, timeout=20)
            assert done.returncode == 0, (done.returncode, done.stderr)
            assert done.stdout.strip() == '{"metric":"cvvdp","jod":7.0}', done.stdout
            assert "pixvqa-cvvdp: info: display=standard_hdr_pq" in done.stderr
            assert "pixvqa-cvvdp: warning: upstream: WARNING:root:" in done.stderr
            assert "pixvqa-cvvdp: info: upstream: [INFO]" in done.stderr
            print("PASS PixVQA-to-mock-backend stderr relay; stdout JSON unchanged")
    else:
        print("SKIP PixVQA binary integration (make cpu first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
