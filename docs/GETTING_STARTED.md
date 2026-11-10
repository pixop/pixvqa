# Installation & Builds

PixVQA supports Ubuntu 22.04/24.04 x86-64 (including WSL2). Native Windows builds are not validated.

## Guided Installation (Recommended)

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

The installer offers FFmpeg software decoding and optional CUDA/NVDEC support. NVIDIA requires a compatible driver and CUDA 12.8 already installed. To isolate downloaded dependencies inside the checkout, use `PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh`.

For noninteractive runs, set `PIXVQA_ENABLE_DAV1D=0|1`, `PIXVQA_ENABLE_NVIDIA=0|1`, and optionally `PIXVQA_ENABLE_LEARNED=1` (the guided prompt for learned metrics defaults to **no**). When learned metrics are selected, the installer provisions a private Python 3.12 interpreter if needed, creates the checkout-local `.venv`, and installs pinned Python dependencies (PyTorch **2.13.0+cu130** / torchvision **0.28.0+cu130** from the CUDA 13.0 wheel index — not the native CUDA 12.8 toolkit used for `make full`). **Reused environments must match the lockfile exactly**; mismatches fail closed — set `PIXVQA_LEARNED_ENV_DIR` to a new directory rather than upgrading in place. Upstream model checkpoints are supplied separately; see [Learned Metrics](LEARNED_METRICS.md).

The **native guided install still requires CUDA toolkit 12.8** when you enable the NVIDIA build; that requirement is independent of the learned Python wheel stack.

## Build Variants

If dependencies are **already installed**, use GNU Make (no downloads):

| Command | Input / acceleration | Output |
|---|---|---|
| `make cpu` | Raw YUV, CPU metrics | `build/cpu/pixvqa` |
| `make swdec` | FFmpeg compressed decoding, CPU metrics | `build/swdec/pixvqa` |
| `make full` | FFmpeg, CUDA metrics and NVDEC | `build/full/pixvqa` |

```bash
make -j"$(nproc)" swdec
make install VARIANT=swdec PREFIX="$PWD"
./bin/pixvqa --version
```

`make install` is required to update `bin/pixvqa`; building a variant alone does not. See [Manual Installation](INSTALL.md) for prerequisites and environment variables.

## Rebuilding after Source Changes

The guided installer **already compiles and installs** PixVQA; there is no need
to run Make after the first installation. When modifying the source code later,
rebuild the desired variant and reinstall its binary into this checkout:

```bash
# Full CUDA/NVDEC build (use `cpu` or `swdec` instead if installed):
make -j"$(nproc)" full
make install VARIANT=full PREFIX="$PWD"
./bin/pixvqa --version
```

Make recompiles changed source files; `make install` updates `bin/pixvqa`.
A successful **guided installation** saves its dependency paths to the ignored
`build/local-config.mk`, so the commands above work in a new shell without
re-exporting FFmpeg, GCC or CUDA paths. Command-line Make assignments and
exported environment variables take precedence over saved values.

For older installations without this file, see [Manual Installation](INSTALL.md)
for environment variables. You do **not** need to rerun the guided installer
or rebuild third-party dependencies solely to recompile PixVQA.

## First Evaluation

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

For raw YUV, color/HDR options, and metric groups, see [CLI Usage](CLI_USAGE.md). Check installed capabilities with `./bin/pixvqa --version`.

## Release qualification (maintainers)

Full release checks rebuild binaries, run the complete QA profile, and (for `full`) GPU qualification. See [Manual Installation](INSTALL.md#release-qualification) for prerequisites (`PIXVQA_TEST_FFMPEG`, VVC fixtures, Valgrind, ASan, interpreter choice). **`reports/release/` is overwritten** on each run.
