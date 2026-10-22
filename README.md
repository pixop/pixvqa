# PixVQA

Proposed repository: `pixop/pixvqa`. The project license is not selected. See `docs/LICENSING.md`.

PixVQA compares a reference picture sequence with a distorted one and prints quality scores. The executable is `pixvqa`.

It scores raw planar YUV, and, when the binary is built with FFmpeg, compressed video. Decoding uses the codec FFmpeg reports. There is no codec or container allowlist. A decoded picture is scored only when it is already planar 8-bit or 10-bit 4:2:0, 4:2:2, or 4:4:4. The two inputs are decoded independently and paired in presentation order. There is no resize, crop, chroma conversion, or tone map.

CPU decoding does not need an NVIDIA driver. GPU decoding is NVDEC for HEVC 10-bit 4:2:0 and AV1 8-bit or 10-bit 4:2:0. Other codecs, including VVC, are rejected on `--decode gpu` and are not replaced by a software decoder.

The tested environment is Linux, including WSL2. A native Windows build is not part of this tree.

The intended logo path is `docs/brand/pixop-vqa-logo.png`. The image is not in this tree. See `docs/brand/README.md`.

## Supported metrics

Select them with `--metrics`. These eight names are supported:

| Name | What it scores |
|---|---|
| `psnr` | PSNR and MSE on stored sample codes |
| `xpsnr` | XPSNR. Requires `--fps` |
| `ssim` | Classical single-scale SSIM |
| `msssim` | Five-scale product MS-SSIM. Not implied by `ssim` |
| `vmaf` | Netflix VMAF v1 through libvmaf 3.2.1, on the CPU. Requires `--model` |
| `pu21-psnr` | Color PSNR on PU21-encoded linear RGB |
| `pu21-ssim` | SSIM on PU21-encoded luminance |
| `pu21-msssim` | MS-SSIM on PU21-encoded luminance |

`ssim` uses an 11×11 Gaussian, sigma 1.5, K1 = 0.01, K2 = 0.03. The dynamic range is 255 for 8-bit and 1023 for 10-bit. Combined YUV `avg` is sample-count weighted: 4:2:0 `(4Y+U+V)/6`, 4:2:2 `(2Y+U+V)/4`, 4:4:4 `(Y+U+V)/3`.

`--qcs` prints one sequence percentage per selected PSNR, XPSNR, SSIM, or MS-SSIM score. VMAF is not a QCS metric. Scores are text on stdout. There is no JSON output mode.

PU21-XPSNR and PU21-VMAF were investigated and are not supported. XPSNR's weighting was built for coded samples, and the shipped VMAF model was not validated on PU21 values. The names `pu21-xpsnr` and `pu21-vmaf` are rejected.

A numerical match validates the implementation. It does not establish that a metric predicts subjective quality better than another.

## HDR

PU21 is the `banding_glare` encoding from Mantiuk and Azimi, Picture Coding Symposium 2021. DOI: [10.1109/PCS50896.2021.9477471](https://doi.org/10.1109/PCS50896.2021.9477471). Reference code: [gfxdisp/pu21](https://github.com/gfxdisp/pu21). The BSD-3-Clause notice is `licenses/PU21_BSD.txt`.

PQ and HLG are not inferred from bit depth, a filename, or BT.2020 primaries. The order for each field is per-input option, shared option, stream metadata, documented default, then an error. PSNR, SSIM, MS-SSIM, and XPSNR still compare stored sample codes. VMAF uses the SDR-trained v1 model and is refused when the effective transfer is PQ or HLG.

`--help-hdr` lists every HDR option. `--hdr-peak-nits` is an evaluation-display value. It is not the mastering-display peak, MaxCLL, or MaxFALL.

## Installation

Dependencies are selected with environment variables. The build scripts do not embed a developer home directory.

### CPU metrics, raw YUV

No CUDA, no NVIDIA driver, and no FFmpeg. VMAF is included when `deps/libvmaf-3.2.1` exists or `LIBVMAF_PREFIX` is set. AV1 and VVC are not decoded by this binary.

```bash
bash build.sh
```

The executable is `build/cpu/pixvqa`.

### CPU decoding, including AV1 and VVC

This binary does not need the CUDA toolkit or an NVIDIA driver. It needs a C++17 compiler, a shared LGPL FFmpeg built with `--disable-gpl`, `--enable-libdav1d`, and the native VVC decoder, plus libdav1d at runtime.

```bash
export FFMPEG_PREFIX=/path/to/ffmpeg
export DAV1D_LIB=/path/to/dav1d/lib
export GCC_ENV=/path/to/gcc
export LIBVMAF_PREFIX=/path/to/libvmaf
bash build_decode.sh
```

Leave `GCC13` and `CUDA_HOME` unset. The executable is `build/swdec/pixvqa`. AV1 software decoding uses libdav1d. VVC uses FFmpeg's native decoder. A compressed input with `--decode gpu` is an error on this binary.

Use a shared FFmpeg built with `--disable-gpl`, `--disable-version3`, `--enable-shared`, and `--enable-libdav1d`, and with the native VVC decoder left enabled. libdav1d 1.5.1 is BSD-2-Clause. Notices are in `THIRD_PARTY_NOTICES.md`. This tree does not ship FFmpeg, libdav1d, CUDA, or a prebuilt `pixvqa`.

### GPU metrics and NVDEC

Build the CUDA objects, then the full binary. `PIXVQA_LIBCUDA_DIR` is the directory containing `libcuda.so.1`. On WSL2 that is often `/usr/lib/wsl/lib`. On a native Linux driver install it is often `/usr/lib/x86_64-linux-gnu`.

```bash
export GCC13=/path/to/gcc-for-nvcc
export CUDA_HOME=/path/to/cuda
bash build_cuda.sh
export FFMPEG_PREFIX=/path/to/ffmpeg
export DAV1D_LIB=/path/to/dav1d/lib
export GCC_ENV=/path/to/gcc
export LIBVMAF_PREFIX=/path/to/libvmaf
export PIXVQA_LIBCUDA_DIR=/path/to/libcuda
bash build_decode.sh
```

The executable is `build/full/pixvqa`. `--decode gpu` supports HEVC 10-bit 4:2:0 and AV1 8-bit or 10-bit 4:2:0. It does not silently fall back to software. GPU PSNR and GPU SSIM require 10-bit input. VMAF and PU21 stay on the CPU.

### Install and models

```bash
PREFIX=/opt/pixvqa PIXVQA_BIN=build/swdec/pixvqa bash scripts/install.sh
```

Models are copied to `$PREFIX/share/pixvqa/model`. Pass that directory in `--model`. Nothing selects a model for you.

Runtime libraries are not copied. A decode build needs `libavformat`, `libavcodec`, `libavutil`, and `libdav1d` on the loader path. When VMAF is enabled, `libvmaf` is linked statically from `LIBVMAF_PREFIX` and is not a separate runtime library. A GPU build also needs the NVIDIA driver libraries. `ldd build/swdec/pixvqa` shows which files the binary was linked against.

`packaging/SHA256SUMS` lists the source files in this tree. It is not a release upload.

## Usage

The examples below use the names the checker creates. `pixvqa` is the binary on `PATH`.

Basic SDR comparison:

```bash
pixvqa --metrics psnr \
  -r reference.yuv -s 64x64 -f 420p:8 \
  -t distorted.yuv -s 64x64 -f 420p:8
```

Several metrics. XPSNR needs a frame rate:

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim --fps 24 --threads 1 \
  -r reference.yuv -s 64x64 -f 420p:8 \
  -t distorted.yuv -s 64x64 -f 420p:8
```

Raw reference against a compressed file:

```bash
pixvqa --metrics psnr --decode cpu \
  -r reference.yuv --input raw -s 64x64 -f 420p:8 \
  -t clip.h264.mp4 --input compressed -s 64x64 -f 420p:8
```

AV1, software decoder:

```bash
pixvqa --metrics psnr --decode cpu \
  -r clip.av1.yuv --input raw -s 64x64 -f 420p:8 \
  -t clip.av1.mp4 --input compressed -s 64x64 -f 420p:8
```

VVC, software decoder. `--decode gpu` rejects VVC:

```bash
pixvqa --metrics psnr --decode cpu \
  -r clip.vvc.yuv --input raw -s 64x64 -f 420p:10 \
  -t clip.vvc.mp4 --input compressed -s 64x64 -f 420p:10
```

HEVC on NVDEC. This needs `build/full/pixvqa`:

```bash
pixvqa --metrics psnr --decode gpu \
  -r clip.hevc.yuv --input raw -s 256x256 -f 420p:10 \
  -t clip.hevc.mp4 --input compressed -s 256x256 -f 420p:10
```

PQ. The PSNR text is still a sample-code score. PU21 uses the transfer:

```bash
pixvqa --metrics psnr,pu21-psnr --hdr-transfer pq \
  -r reference10.yuv -s 64x64 -f 420p:10 \
  -t distorted10.yuv -s 64x64 -f 420p:10
```

HLG with an explicit evaluation peak:

```bash
pixvqa --metrics pu21-psnr --hdr-transfer hlg --hdr-peak-nits 1000 \
  -r reference10.yuv -s 64x64 -f 420p:10 \
  -t distorted10.yuv -s 64x64 -f 420p:10
```

One input PQ and the other HLG:

```bash
pixvqa --metrics pu21-psnr --ref-hdr-transfer pq --dist-hdr-transfer hlg \
  --hdr-peak-nits 1000 \
  -r reference10.yuv -s 64x64 -f 420p:10 \
  -t distorted10.yuv -s 64x64 -f 420p:10
```

VMAF on SDR samples. The model path is required:

```bash
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
  -r vmaf_ref.yuv -s 320x240 -f 420p:8 \
  -t vmaf_dist.yuv -s 320x240 -f 420p:8
```

## Accuracy and limits

SSIM and MS-SSIM follow the classical definitions above. XPSNR follows the integer-rate high-pass formulation in the source. VMAF is Netflix VMAF v1, model JSON supplied by the user, minimum frame size enforced for the SpEED features. PU21 uses binary64, the `banding_glare` curve, domain `[0.005, 10000]` cd/m², and peak 256.

Identical samples print PSNR `999.99`. VMAF is compared with libvmaf 3.2.1. Model checksums are in `reports/VMAF_MODEL_CHECKSUMS.md`. A numerical match checks the implementation. It does not rank subjective quality.

Decoder limits that were tested, and are not a claim of full codec conformance:

- Software pictures must already be planar 8-bit or 10-bit 4:2:0, 4:2:2, or 4:4:4. `nv12` as a software layout, RGB, and 12-bit are rejected.
- AV1 GPU pictures are NV12 at 8-bit and P010 at 10-bit. The 8-bit path is deinterleaved on the host.
- The AV1 fixtures have no film-grain header. libdav1d is opened with the library defaults `filmgrain=-1` and `oppoint=-1`.
- The VVC checks are short 10-bit 4:2:0 clips, including a byte compare with VVdeC. They are not a conformance suite.
- A mid-stream resolution change is not supported as a scored input.
- `--decode auto` prefers a working software decoder. It uses AV1 NVDEC only when no software AV1 decoder is linked.

## Quality checks

```bash
python3 scripts/run_qa.py --profile smoke
python3 scripts/run_qa.py --profile full
python3 scripts/run_qa.py --profile hardware
```

`full` needs FFmpeg, libdav1d, and the VMAF CLI (`FFMPEG`, `VVDEC`, `VMAF_CLI`). `hardware` runs only where `build/full/pixvqa` and `nvidia-smi` exist. A skipped hardware test is not a pass.

GitHub-hosted runners build the CPU binary and run smoke. They record SKIP for the full profile. The hardware job runs only when requested on a self-hosted NVIDIA runner. See `.github/workflows/qa.yml`.

## Troubleshooting

- `this build has no CPU decoder`: the binary is `build/cpu/pixvqa`. Use `build/swdec/pixvqa` or `build/full/pixvqa`.
- `GPU decoding is not compiled`: `--decode gpu` was asked of a binary built without NVDEC.
- `no video stream`: `--input compressed` was used on raw YUV.
- `vmaf requires --model`: pass a model JSON. There is no default.
- `vmaf rejects HDR transfer`: the effective transfer is PQ or HLG.
- A missing `libdav1d.so` or `libavcodec.so` at startup means the loader path does not match the build. `ldd` on the executable names the files.
