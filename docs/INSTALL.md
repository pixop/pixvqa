# Installing PixVQA

Build PixVQA from this checkout on Linux x86-64. WSL2 is a tested environment. Native Windows is not. The SSIM and MS-SSIM objects are compiled with `-mavx2`. The other objects use `-march=native`. The scripts read prefixes from the environment and do not embed a machine path.

This tree does not ship FFmpeg, libdav1d, CUDA, or a prebuilt `pixvqa`. Notices are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Shell scripts use LF. `.gitattributes` sets `*.sh text eol=lf`.

## 1. System tools

The PixVQA scripts call these commands by name: `g++`, `python3`, `git`, `meson`, `ninja`, `nasm`, and `pkg-config`. `bash build.sh` and `scripts/run_qa.py` need `python3`. `scripts/build_libvmaf.sh` exits if `git`, `meson`, `ninja`, `nasm`, or `pkg-config` is missing. dav1d also needs `nasm`.

On Ubuntu 24.04 those commands come from:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3 git meson ninja-build nasm pkg-config
```

`GCC_ENV` is a GCC prefix whose `bin/g++` is the C++17 compiler. The decode script refuses to run unless that file exists, and it puts `$GCC_ENV/bin` on `PATH`. A distribution `g++` is enough when you point `GCC_ENV` at its prefix, for example `/usr` if `/usr/bin/g++` exists.

`scripts/build_libvmaf.sh`, `scripts/build_libvmaf_3.2.1.sh`, and `scripts/build_zimg.sh` prepend `$GCC_ENV/bin` only when `GCC_ENV` is set. They do not search a user-specific compiler directory. `gcc`, `g++`, and GNU `make` must be on `PATH` for the zimg build. Ubuntu's `build-essential` package supplies `make`. If autoconf, automake, libtool, or m4 are missing, `scripts/build_zimg.sh` downloads the Ubuntu 24.04 packages named in that script and checks their SHA-256 values. The libvmaf and zimg commits stay pinned.

## 2. libdav1d 1.5.1

AV1 software decoding uses libdav1d. The validated library is 1.5.1.

```bash
git clone --depth 1 --branch 1.5.1 https://code.videolan.org/videolan/dav1d.git
cd dav1d
meson setup build --prefix="$HOME/dav1d-1.5.1" --default-library=shared --buildtype=release
ninja -C build
ninja -C build install
```

Meson may install the shared library in `lib` or `lib64`. `DAV1D_LIB` is the directory that contains `libdav1d.so`, not the prefix root.

## 3. FFmpeg n8.1.3

The validated FFmpeg is tag `n8.1.3`, commit `1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7`, libavcodec 62.28.103. It is a shared LGPL 2.1-or-later build. `build_decode.sh` requires both of these files:

- `$FFMPEG_PREFIX/lib/libavcodec.so`
- `$FFMPEG_PREFIX/include/libavcodec/avcodec.h`

Configure from that tag. Point the dav1d flags at the prefix from the previous section. Leave the native VVC decoder enabled. Do not pass `--enable-gpl` or `--enable-version3`.

```bash
git clone --depth 1 --branch n8.1.3 https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
git rev-parse HEAD   # 1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7
export DAV1D_LIB="$HOME/dav1d-1.5.1/lib64"   # or .../lib, whichever holds libdav1d.so
./configure \
  --prefix="$HOME/ffmpeg-n8.1.3" \
  --enable-shared --disable-static \
  --disable-gpl --disable-version3 \
  --disable-doc --disable-programs \
  --disable-avdevice --disable-avfilter \
  --disable-swscale --disable-swresample \
  --disable-encoders --disable-muxers \
  --disable-autodetect \
  --enable-libdav1d --enable-protocol=file \
  --extra-cflags="-I$HOME/dav1d-1.5.1/include" \
  --extra-ldflags="-L$DAV1D_LIB -Wl,-rpath,$DAV1D_LIB"
make -j"$(nproc)"
make install
export FFMPEG_PREFIX="$HOME/ffmpeg-n8.1.3"
```

PixVQA opens libdav1d for AV1, and libaom-av1 only when libdav1d is not in that FFmpeg build. It does not use FFmpeg's native `av1` decoder. VVC uses the native `vvc` decoder. `--disable-encoders` does not remove decoders.

NVDEC needs the NVIDIA Video Codec SDK headers (`ffnvcodec`) on the `pkg-config` path, plus these three flags on the same `./configure` line:

```text
--enable-ffnvcodec --enable-nvdec --enable-cuvid
```

The validated library was built with those three flags. A configure that omits them has not been given its own test. A CPU decode of AV1 and VVC does not require the CUDA toolkit. `--enable-nvdec` does not grant redistribution of the NVIDIA driver.

## 4. libvmaf commit 74f930d and zimg 3.0.6

From the PixVQA checkout. GNU `make` must be on `PATH`. `scripts/build_libvmaf.sh` builds zimg first.

```bash
export GCC_ENV=/path/to/gcc
bash scripts/build_libvmaf.sh
```

The script checks out Netflix `vmaf` commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef`. This is not a tagged release. The last tag used by the previous recipe is v3.2.1, commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`, still built by `scripts/build_libvmaf_3.2.1.sh`. Meson still reports project version 3.2.0. `vmaf --version` prints `74f930d`.

zimg is tag `release-3.0.6`, commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd`, static, prefix `deps/zimg-3.0.6`. libvmaf is static, prefix `deps/libvmaf-74f930d`, with `-Denable_zimg=true` and CUDA off. The install stops unless `pkg-config --libs --static` contains `-lzimg` and `picture.h` declares `vmaf_picture_convert`. `scripts/libvmaf_flags.sh` refuses a prefix that fails those checks, so a leftover `deps/libvmaf-3.2.1` is not linked.

The model files in [model/](../model/README.md) are what `--model` loads. libvmaf is linked statically.

Without the 74f930d prefix the PixVQA binary still builds, and selecting `vmaf` is an error.

## 5. CPU builds

Raw YUV only. No FFmpeg, libdav1d, or CUDA. The executable does not open compressed video.

```bash
export GCC_ENV=/path/to/gcc
export LIBVMAF_PREFIX="$PWD/deps/libvmaf-74f930d"
bash build.sh
```

The executable is `build/cpu/pixvqa`.

Software decoding, including AV1 and VVC. Leave `GCC13` and `CUDA_HOME` unset. If both are set and `nvcc` is on that prefix, this script also tries to link the GPU binary.

```bash
export FFMPEG_PREFIX="$HOME/ffmpeg-n8.1.3"
export DAV1D_LIB="$HOME/dav1d-1.5.1/lib64"
export GCC_ENV=/path/to/gcc
export LIBVMAF_PREFIX="$PWD/deps/libvmaf-74f930d"
bash build_decode.sh
```

The executable is `build/swdec/pixvqa`. `--decode gpu` on this binary is an error.

## 6. Optional NVIDIA build

`build_cuda.sh` compiles `src/cuda_metrics.cu`. `build_decode.sh` then links `build/full/pixvqa`.

`GCC13` must contain `bin/x86_64-conda-linux-gnu-g++`. The script's own example is conda-forge GCC 13.4:

```bash
micromamba create -y -p "$GCC13" -c conda-forge gxx_linux-64=13.4
```

`CUDA_HOME` must contain `bin/nvcc`. The validated toolkit is CUDA 12.8. `PIXVQA_LIBCUDA_DIR` is the directory containing `libcuda.so.1`. On WSL2 that is often `/usr/lib/wsl/lib`. On a native Linux driver install it is often `/usr/lib/x86_64-linux-gnu`. If the variable is unset, the script uses one of those when the directory is present.

```bash
export GCC13=/path/to/gcc-13
export CUDA_HOME=/path/to/cuda
bash build_cuda.sh
export FFMPEG_PREFIX="$HOME/ffmpeg-n8.1.3"
export DAV1D_LIB="$HOME/dav1d-1.5.1/lib64"
export GCC_ENV=/path/to/gcc
export LIBVMAF_PREFIX="$PWD/deps/libvmaf-74f930d"
export PIXVQA_LIBCUDA_DIR=/path/to/libcuda
bash build_decode.sh
```

`build/full/pixvqa` accepts `--decode gpu` for HEVC 10-bit 4:2:0 and AV1 8-bit or 10-bit 4:2:0. It does not fall back to software. GPU PSNR and GPU SSIM require 10-bit samples. VMAF and the three PU21 metrics stay on the CPU.

## 7. Install and runtime libraries

`PREFIX` is required. `PIXVQA_BIN` defaults to `build/cpu/pixvqa`.

```bash
PREFIX="$HOME/pixvqa" PIXVQA_BIN=build/swdec/pixvqa bash scripts/install.sh
"$PREFIX/bin/pixvqa" --version
```

The script copies the executable, the model JSON files, the README, the third-party notices, and the PU21 notice. It does not copy FFmpeg, libdav1d, or CUDA. SDR models are in `$PREFIX/share/pixvqa/model`. HDR v1.1.8 models are in `$PREFIX/share/pixvqa/model/hdr_vmaf_v1.1.8` and `$PREFIX/share/pixvqa/model/hdr_vmaf_v1.1.8_hfr`. Pass one file to `--model`.

Use `PIXVQA_BIN=build/full/pixvqa` for the GPU binary, or `build/cpu/pixvqa` for raw YUV only.

A decode build loads `libavformat`, `libavcodec`, `libavutil`, and `libdav1d` from the directories recorded at link time. A GPU build also loads `libcudart` and `libcuda`. libvmaf does not appear in `ldd`.

```bash
ldd build/swdec/pixvqa
ldd build/full/pixvqa
```

A missing `.so` at startup is a loader-path mismatch. The install script does not rewrite those paths.

## 8. Smoke check

From the checkout, after `bash build.sh`:

```bash
python3 scripts/run_qa.py --profile smoke
```

`smoke` needs the CPU binary and Python. It does not need FFmpeg. `full` also needs `FFMPEG`, `VVDEC`, and `VMAF_CLI`. `hardware` runs only where `build/full/pixvqa` and `nvidia-smi` exist. A skipped hardware run is not a pass. The hosted workflow is [.github/workflows/qa.yml](../.github/workflows/qa.yml).

## What is pinned

| Piece | Pin |
|---|---|
| libdav1d | 1.5.1, shared |
| FFmpeg | tag `n8.1.3`, commit `1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7` |
| libvmaf | commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` (not a release tag; previous tag v3.2.1), static, zimg 3.0.6, CUDA off |
| CUDA toolkit, when used | 12.8, with GCC 13.4 as the nvcc host compiler |
| VMAF models | eight SDR v1.0.16 JSON files and eight HDR v1.1.8 JSON files in `model/` |

## Verification status

On `main` before this branch, the CPU workflow in sections 4, 5, 7, and 8 was executed against libvmaf tag v3.2.1, commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`. That recipe is still `scripts/build_libvmaf_3.2.1.sh`. `bash build.sh` produced `build/cpu/pixvqa` (`pixvqa version 0.3.0-dev`, `decode: none`). `scripts/install.sh` installed that binary. `python3 scripts/run_qa.py --profile smoke` reported `PASS 16  FAIL 0  SKIP 0`. A later pass on that base ran `build_decode.sh` and `build_cuda.sh` against the existing FFmpeg n8.1.3, dav1d 1.5.1, GCC 13, and CUDA 12.8 prefixes. That produced `build/swdec/pixvqa` (`decode: software`) and `build/full/pixvqa` (`decode: software nvdec`). Those decode and CUDA prefixes were not rebuilt on this branch.

This branch builds libvmaf from commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` with zimg 3.0.6. That checkout is not a release tag. Meson still reports 3.2.0, and `vmaf --version` prints `74f930d`. The results of the CPU rebuild, model checksums, HDR parity, and SDR regression are in [reports/HDR_VMAF_PUBLIC.md](../reports/HDR_VMAF_PUBLIC.md).

These steps match the libraries used for the earlier AV1, VVC, and NVDEC checks, and were not repeated from an empty machine for this document:

- The Ubuntu package install in section 1.
- The libdav1d clone and Meson build in section 2.
- The FFmpeg clone, `./configure`, and `make` in section 3, including a configure that omits `--enable-ffnvcodec`, `--enable-nvdec`, and `--enable-cuvid`.
- The GCC 13 and CUDA steps in section 6.

## Continuous integration (GitHub Actions)

Workflow file: `.github/workflows/qa.yml` on the [pixop/pixvqa](https://github.com/pixop/pixvqa) repository.

**When CI runs**

- Every **push** to `main`.
- Every **pull request** targeting `main`.
- **Manual** runs from the Actions tab (`workflow_dispatch`). Optional input **hardware** runs the GPU hardware profile only when set to true and a matching self-hosted NVIDIA runner exists.

CI does **not** install PixVQA on end-user machines. It only validates this source tree on GitHub’s Ubuntu runners (or your self-hosted GPU runner when explicitly requested).

**What hosted CI executes (CPU, `ubuntu-24.04`)**

1. `bash build.sh` — compiles `build/cpu/pixvqa` and `build/cpu/selftest`, then runs selftest and `scripts/test_io_regressions.py` inside the build script.
2. `build/cpu/selftest` — repeated as an explicit CI step.
3. `scripts/test_io_regressions.py` — CPU-only CLI and I/O checks (no FFmpeg).
4. `scripts/test_qcs.py` — QCS regression (Ubuntu package `python3-numpy`).
5. `scripts/run_qa.py --profile smoke` — 16-case smoke profile on the CPU binary.

**What hosted CI does not run**

- **Full QA** (~233 cases): needs FFmpeg, libdav1d, libvmaf, VMAF CLI, `build/swdec` or `build/full`, and other fixtures documented in the QA harness. A separate workflow job records **NOT RUN** with an artifact; it is not a passing full validation.
- **CUDA / NVDEC / hardware profile**: no GPU on GitHub-hosted runners. Use manual dispatch with **hardware** enabled and a self-hosted runner labeled `self-hosted`, `linux`, `nvidia`.

**Local validation (same as release QA)**

```bash
python3 scripts/run_qa.py --profile smoke
python3 scripts/run_qa.py --profile full    # after deps in sections 2–6
python3 scripts/run_qa.py --profile hardware  # GPU build; export PIXVQA_CHECK_GPU=1
```

## Troubleshooting

- `Set FFMPEG_PREFIX` or `Set GCC_ENV`: `build_decode.sh` did not find `lib/libavcodec.so` or `bin/g++`.
- `this build has no CPU decoder`: the binary is `build/cpu/pixvqa`. Use `build/swdec/pixvqa` or `build/full/pixvqa` for compressed video.
- `GPU decoding is not compiled into this binary`: `--decode gpu` was asked of a binary built without NVDEC.
- `Set GCC13` or `Set CUDA_HOME`: the CUDA script did not see `bin/x86_64-conda-linux-gnu-g++` or `bin/nvcc`.
- `Set PIXVQA_LIBCUDA_DIR`: `libcuda.so.1` was not in the usual WSL or Linux driver directory.
- `no video stream`: `--input compressed` was used on raw YUV.
- `vmaf requires --model`: pass a JSON file. There is no default.
- `vmaf rejects HDR transfer`: the effective transfer is PQ or HLG.
- A missing `libdav1d.so` or `libavcodec.so` means the loader path does not match the link. `ldd` names the files.
