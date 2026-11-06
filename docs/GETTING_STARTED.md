# Installation & Builds

PixVQA supports Linux x86-64 (Ubuntu 22.04/24.04, including WSL2). Native Windows builds are not validated.

## Guided installation (recommended)

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

The installer offers a dav1d/FFmpeg software-decoding build and optional CUDA/NVDEC support. The NVIDIA build requires a compatible driver and CUDA 12.8 installed beforehand. To isolate PixVQA-managed dependencies under this checkout, instead run:

```bash
PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh
```

For unattended installation, use `PIXVQA_ENABLE_DAV1D=0|1` and `PIXVQA_ENABLE_NVIDIA=0|1` to answer the corresponding prompts. **The installer does not install Python-backed metrics**; set them up separately in [Optional Metrics](LEARNED_METRICS.md).

## Build with GNU Make

With dependencies already installed, choose a target (Make does not download dependencies):

| Command | Inputs / acceleration | Executable |
|---|---|---|
| `make cpu` | Raw YUV; CPU metrics | `build/cpu/pixvqa` |
| `make swdec` | FFmpeg compressed decoding; CPU metrics | `build/swdec/pixvqa` |
| `make full` | FFmpeg, CUDA metrics and NVDEC | `build/full/pixvqa` |
| `make test-cpu` | CPU self-tests | `build/cpu/selftest` |

```bash
make -j"$(nproc)" swdec
./build/swdec/pixvqa --version
make install VARIANT=swdec PREFIX="$PWD"
./bin/pixvqa --version
```

`make` does not replace `bin/pixvqa`; run `make install VARIANT=cpu|swdec|full PREFIX="$PWD"` after building the desired variant. If using checkout-local dependencies, export `PIXVQA_DEPS_ROOT="$PWD/deps"` before `make`.

## Try it

For FFmpeg-enabled builds:

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

For raw YUV (supported by CPU builds), see [CLI Usage](CLI_USAGE.md#raw-yuv). For a basic installation check:

```bash
python3 scripts/run_qa.py --profile smoke
```

If automatic installation fails, see the detailed [Manual Installation](INSTALL.md). The smoke test is a user-facing check, not a numerical-conformance certification.
