# Command-line Usage

PixVQA supports raw YUV and compressed video. For compressed inputs, resolution and pixel format are detected automatically; raw YUV requires `-s` (resolution) and `-f` (pixel format). Reference and distorted inputs must have matching properties.

XPSNR requires `--fps` unless a reliable constant frame rate can be determined from the compressed inputs. GPU acceleration requires an NVIDIA-enabled build.

The examples use `pixvqa` to refer to `./bin/pixvqa` or an executable on `PATH`. The `samples/` directory contains short Big Buck Bunny excerpts.

**Compressed Reference and Distorted Video**

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_av1.mp4 \
  -t samples/dist_av1.mp4
```

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_vvc.mp4 \
  -t samples/dist_vvc.mp4
```

**Uncompressed Reference and Compressed Distorted Video**

```bash
pixvqa --metrics psnr,ssim \
  -r reference.yuv --input raw -s 1920x1080 -f 420p:8 \
  -t distorted.mp4
```

**Uncompressed Reference and Distorted Video**

XPSNR requires `--fps` because raw YUV does not contain frame-rate metadata.

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim --fps 24 --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

**VMAF**

VMAF requires an explicit model selection using `--model`. Bundled VMAF models are available in the `model/` directory.

```bash
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

The bundled HDR VMAF v1.1.8 models assume limited-range 10-bit 4:2:0 PQ with
BT.2020 primaries and the BT.2020 non-constant-luminance matrix. PixVQA applies
that model configuration automatically; ensure both inputs really have those
characteristics. Contradictory metadata or HDR CLI options produce a warning and
do not convert the samples.

```bash
pixvqa --metrics vmaf \
  --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

**ColorVideoVDP (optional external backend)**

CVVDP is installed separately from PixVQA and only runs as `--metrics cvvdp`.
It requires explicit sample interpretation and an official display model; PixVQA
does not infer HDR parameters for it. It does not support raw YUV in this first
integration; inputs must be formats accepted and validated by the official
CVVDP CLI.

```bash
pixvqa --metrics cvvdp \
  -r reference.mkv \
  -t distorted.mkv \
  --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq \
  --cvvdp-output-json result.cvvdp.json
```

See [COLORVIDEOVDP.md](COLORVIDEOVDP.md) for isolated installation, supported
displays, the JSON result record, and incompatible native options.

**PU21 Metrics for HDR Video (PQ and HLG)**

PU21 metrics account for HDR transfer functions, unlike conventional metrics that operate on encoded sample values. For PQ and HLG, PixVQA defaults to BT.2020 primaries, BT.2020nc matrix, and limited range when these are unspecified. HLG defaults to a 1000 cd/m² display peak; PQ requires no display peak. See the [PU21 guide](PU21.md) for details.

```bash
pixvqa --metrics pu21-psnr,pu21-ssim --hdr-transfer pq \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

```bash
pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer hlg \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

**GPU Decoding (NVDEC)**

Decode the compressed HEVC 10-bit 4:2:0 video using NVIDIA NVDEC while reading the reference as raw YUV.

```bash
pixvqa --metrics psnr --decode gpu \
  -r reference10.yuv --input raw -s 1920x1080 -f 420p:10 \
  -t reference10.hevc.mp4
```

## Command-line Options

`pixvqa --help` lists the options. `pixvqa --help-hdr` lists the HDR options and defaults. QCS is documented in [QCS.md](QCS.md). PQ and HLG setup is documented in [PU21.md](PU21.md).

| Option | Purpose |
|---|---|
| `-r`, `-t` | Reference path and distorted path |
| `--metrics` | Comma-separated metric names |
| `--decode` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--metric-device` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--threads` | `auto` or a positive integer. Default `auto` |
| `-s WxH` | Luma size. Required for raw YUV. Checked when set on compressed video |
| `-f` | `420p`, `422p`, or `444p`, with `:8` or `:10`. Required for raw YUV |
| `--input` | `auto`, `raw`, or `compressed`. Default `auto`. `raw` does not probe |
| `--fps` | `N` or `N/D`. Required for `xpsnr` when no constant compressed rate is available. Decimals truncate toward zero |
| `--model` | VMAF model JSON. No default |
| `--cvvdp-interpretation` | Required for CVVDP: `sdr` or `pq`; never inferred |
| `--cvvdp-display` | Required official CVVDP display model |
| `--cvvdp-python` | Optional interpreter path for the isolated adapter |
| `--cvvdp-bin` | Optional official CVVDP executable path |
| `--cvvdp-output-json` | Persistent CVVDP machine-readable run record |
| `--hdr-transfer` | `pq`, `hlg`, `sdr`, `linear`, or `auto` |
| `-v` | `1` prints each frame and the summary. `0` prints the summary. Default `1` |
| `--qcs` | Enable QCS for selected supported per-frame metrics |
| `--qcs-range` | `metric=lower:upper`, comma-separated and repeatable, with lower < upper |
