# Third-party notices

This file records the dependencies PixVQA is built and validated against.
It is an engineering inventory for legal review. It is not a certification
that a particular distribution complies with those licenses.

PixVQA's own source license has not been selected. Do not publish the
repository, and do not ship a binary, until that license is chosen and a
`LICENSE` file is added.

## FFmpeg

Validated build: FFmpeg n8.1.3, libavcodec 62.28.103, configured with
`--disable-gpl` and installed to a private prefix. The license reported by
that configure is LGPL 2.1 or later.

The CPU decode binary and the GPU binary link libavformat, libavcodec, and
libavutil dynamically. LGPL obligations for a distributed binary include
providing the corresponding source, the exact configure flags, and a way for
a recipient to relink. This repository does not vendor the FFmpeg source.
The license-relevant flags are `--enable-shared`, `--disable-static`,
`--disable-gpl`, `--disable-version3`, `--enable-libdav1d`,
`--enable-ffnvcodec`, `--enable-nvdec`, and `--enable-cuvid`,
with swscale, encoders, muxers, and libx264 left disabled.
Pass those prefixes with `FFMPEG_PREFIX` and `DAV1D_LIB`.
This source tree does not include the FFmpeg or libdav1d binaries.

`--enable-nvdec` and `--enable-cuvid` use the NVIDIA Video Codec SDK headers
through ffnvcodec. Those headers are not an LGPL grant to redistribute the
NVIDIA driver.

## libdav1d

Version 1.5.1. BSD-2-Clause. libavcodec links it when FFmpeg is configured
with `--enable-libdav1d`. A CPU decode build needs that shared library at
runtime. It does not require CUDA.

## libvmaf and the VMAF models

libvmaf 3.2.1, Netflix commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`.
The license text redistributed in this tree is BSD-2-Clause-Patent:
`third_party/netflix-vmaf/LICENSE` and `third_party/netflix-vmaf/NOTICE`.

The eight JSON files under `model/` are byte copies of the VMAF v1.0.16
models from that commit. See `model/README.md`. PixVQA does not select a
model automatically. The user passes `--model`.

## PU21

The banding_glare formula is from Mantiuk and Azimi, PCS 2021.
DOI: https://doi.org/10.1109/PCS50896.2021.9477471
Reference code: https://github.com/gfxdisp/pu21
The reference implementation is BSD-3-Clause. The notice is
`licenses/PU21_BSD.txt`. The IEEE paper itself is not redistributed.

## CUDA and NVDEC

The GPU binary links libcudart and libcuda. Those are NVIDIA components.
This package does not copy them. A recipient needs a driver and, to rebuild,
the CUDA toolkit. `--decode gpu` without a GPU or without the full binary
is an error. It does not switch to a software decoder.

## What this tree does not add

The validated FFmpeg build uses `--disable-gpl`. GPL-only FFmpeg components
are not enabled. No proprietary video dataset is included.

## Trademark

The Pixop name and logo are not covered by the dependency licenses above.
The logo file is not in this tree. See `docs/brand/README.md`.
A project license has not been selected. See `docs/LICENSING.md`.
