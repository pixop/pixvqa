# Installation and Building

PixVQA supports **Linux x86-64**, including Ubuntu 22.04 and 24.04 (native Linux and WSL2). Native Windows builds have not been validated.

There are **two ways to build PixVQA**, both available to all users:

- **Guided installation (recommended for first-time users):** `install_pixvqa.sh` sets up the required dependencies, compiles PixVQA, and installs the executable as `bin/pixvqa`.
- **Incremental builds (for anyone with dependencies already installed):** GNU Make compiles a selected variant without reinstalling dependencies. This is useful after updating the source code or when choosing CPU-only, software-decoding, or CUDA/NVDEC support. You do **not** need to be a developer to use it.

### Option 1: Guided Installation (Recommended for First-Time Users)

Clone the repository and run the interactive installer:

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
```

The installer downloads and builds the required dependencies and asks which capabilities to enable. **FFmpeg software decoding** supports compressed video; **NVIDIA CUDA/NVDEC** acceleration is optional and requires a compatible NVIDIA driver and CUDA toolkit 12.8 already installed.

By default, the installer installs the chosen build into **`bin/pixvqa`** inside the checkout.

**To keep PixVQA-managed dependencies inside this repository** rather than in shared directories under `$HOME`, use the following command *instead* of the plain installer command above:

```bash
PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh
```

Check which capabilities were installed:

```bash
./bin/pixvqa --version
python3 scripts/run_qa.py --profile smoke
```

The version output reports the compiled build and available decoders (for example, `build: cuda` and `decode: software nvdec` for a full build).

For a first quality comparison, use the bundled HEVC sample clips (requires FFmpeg software decoding or the full build):

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```

CPU-only builds accept raw YUV instead; see the [raw YUV example](CLI_USAGE.md#uncompressed-reference-and-distorted-video).

### Option 2: Building from Source (Incremental Builds)

**Anyone can use this workflow**, provided the required dependencies are already installed (for example, by running the guided installer once). The `make` commands do **not** download or install missing dependencies.

Choose the build that matches your needs:

| Command | Capabilities | Compiled executable |
|---|---|---|
| `make cpu` | CPU metrics, raw YUV inputs only | `build/cpu/pixvqa` |
| `make swdec` | CPU metrics and FFmpeg software decoding | `build/swdec/pixvqa` |
| `make full` | CUDA metrics, FFmpeg software decoding, and NVDEC | `build/full/pixvqa` |
| `make test-cpu` | Build and run CPU self-tests | `build/cpu/selftest` |

For example, if you previously installed dependencies into `./deps` and want the full CUDA/NVDEC build:

```bash
export PIXVQA_DEPS_ROOT="$PWD/deps"
make -j"$(nproc)" full
./build/full/pixvqa --version
```

Replace `full` with `swdec` or `cpu` to compile another variant. If your dependencies were installed elsewhere, use the appropriate dependency paths instead; see the [Make Build Guide](BUILD_MAKE.md).

**Building is not the same as installing.** Each `make` target writes its own executable under `build/`. Running `make full`, for example, does **not** replace `bin/pixvqa`. The executable at **`bin/pixvqa` is whichever variant you last installed**, either through the guided installer or through `make install`.

To install your chosen compiled variant into `bin/pixvqa`, run:

```bash
make install VARIANT=full PREFIX="$PWD"
./bin/pixvqa --version
```

Use `VARIANT=cpu` or `VARIANT=swdec` if that is the variant you built. The `PREFIX="$PWD"` argument explicitly installs into the current checkout; without it, an existing `PREFIX` environment variable may redirect the installation. Installing a different variant **replaces** the previous `bin/pixvqa` executable.

After changing or updating source files, run the same `make` target again. GNU Make recompiles only affected components; if you want to use the updated build via `./bin/pixvqa`, run the matching `make install` command again.

For dependency configuration, compiler options, and troubleshooting, see the [Make Build Guide](BUILD_MAKE.md).

### Troubleshooting and Manual Installation

If the guided installer fails, consult the [Manual Installation Guide](INSTALL.md). Follow the numbered sections in order, verifying that each stage completes successfully before continuing.

For clean-install testing, full release QA prerequisites, and known validation limitations, see [Installation Validation](INSTALL_VALIDATION.md). The `smoke` profile above is the recommended post-installation check.

**Note:** VMAF always runs on CPU libvmaf. Other supported metrics can use CUDA when NVIDIA acceleration is enabled.
