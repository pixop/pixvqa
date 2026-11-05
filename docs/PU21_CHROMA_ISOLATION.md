# PU21 chrominance-isolation experiment

`scripts/test_pu21_chroma_isolation.py` is a reproducible HDR experiment that
separates chromatic differences from display-linear luminance differences. It
does not alter any quality metric. The fixture uses one 256×256 frame of
limited-range BT.2020 non-constant-luminance, PQ, planar YUV444P10.

Run it from a learned-metrics environment that has ColorVideoVDP, CUDA, FFmpeg
with `libx265`, and a FULL PixVQA build:

```bash
PATH=/home/sergio/bin:$PATH \
FFMPEG=/home/sergio/bin/ffmpeg \
/home/sergio/pixvqa-learned-env/bin/python \
  scripts/test_pu21_chroma_isolation.py
```

The script keeps all generated inputs in a temporary directory and emits a JSON
record to stdout. Set `PIXVQA_BIN` to select a different FULL binary.

## Input generation

The neutral reference contains eight 32×32 tiled regions. Its Y′ codes are
410, 450, 490, 530, 570, 610, 650, and 690; Cb and Cr are 512. The distorted
tile uses the corresponding `(Y′, Cb, Cr)` triplets:

```text
(341,664,448) (289,664,688) (461,376,448) (507,352,472)
(495,424,640) (513,688,400) (617,376,592) (677,352,520)
```

For each candidate the generator expands limited-range Y′CbCr with BT.2020
NCL, applies the ST 2084 EOTF to R′G′B′, and calculates display luminance as
`0.2627 R + 0.6780 G + 0.0593 B`. The triplets were selected on the 10-bit
limited-range code lattice to retain the neutral tile's display luminance after
quantization while changing chromaticity. Every distorted RGB signal remains
legal: its PQ-code components are in `[0.116704, 0.882118]`, with an RGB-code
spread as large as `0.553886`.

The exact values cannot remain perfectly equal after quantization. The
validated maximum mismatch is `0.0251534 cd/m²`, or `1.31642e-4` relative at
the darkest tile. The test limits those figures to `0.026 cd/m²` and
`1.4e-4`, respectively. This quantization residual is why the structural
unity threshold is numerical rather than exact.

Native metrics score the original raw YUV444P10 files:

```bash
pixvqa --metrics psnr,ssim,msssim,pu21-psnr,pu21-ssim,pu21-msssim \
  --device cpu --hdr-transfer pq --hdr-primaries bt2020 \
  --hdr-matrix bt2020nc --hdr-range limited \
  -r reference.yuv -s 256x256 -f 444p:10 \
  -t distorted.yuv -s 256x256 -f 444p:10
```

ColorVideoVDP does not accept raw YUV444P10 in PixVQA's adapter; its raw
interface is intentionally YUV420P10 only. The script therefore losslessly
encodes the same samples to HEVC 4:4:4 10-bit, sets HEVC PQ/BT.2020 metadata,
checks the tags with `ffprobe`, and byte-compares FFmpeg's decoded output with
the source YUV before running:

```bash
pixvqa --metrics cvvdp --device gpu \
  --cvvdp-interpretation pq --cvvdp-display standard_hdr_pq \
  -r reference.mp4 -t distorted.mp4 -n 1
```

## Observed result

On the validated environment (FULL build, CPU native run, CUDA ColorVideoVDP),
the conventional code-domain scores were PSNR `y=22.38`, `u=16.84`, `v=20.41`,
and `avg=19.26`; SSIM `y=0.915790`, `u=0.769574`, `v=0.769510`, and
`avg=0.818291`; and MS-SSIM `y=0.835784`, `u=0.218982`, `v=0.274743`, and
`avg=0.443170`.

PU21-PSNR detected the display-light chromatic difference strongly:
`r=8.348018`, `g=9.974417`, `b=2.700113`, and pooled `avg=5.828644`.
The high-precision QCS trace recorded PU21-SSIM as
`0.9999999978855553` and PU21-MS-SSIM as `0.9999999964714604`. Both are
within the explicit `5e-9` unity tolerance and print as `1.000000`:

```text
[ PU21-PSNR     r:8.348018 g:9.974417 b:2.700113 avg:5.828644 ]
[ PU21-SSIM     y:1.000000 u:n/a v:n/a avg:1.000000 ]
[ PU21-MS-SSIM  y:1.000000 u:n/a v:n/a avg:1.000000 ]
```

ColorVideoVDP reported `0.9223061` JOD on CUDA, so it also detects the
chromatic difference.

The conventional metrics differ because they compare the stored Y′, Cb, and Cr
sample planes. Y′ is not display-linear luminance under PQ, and matching
display luminance requires different Y′ values when chromaticity changes.
Their chroma scores consequently differ even more. PU21-SSIM and
PU21-MS-SSIM instead compare only the PU21 encoding of reconstructed
display-linear luminance, so the small score departures from unity are fully
explained by the measured 10-bit quantization residual. PU21-PSNR compares the
three encoded display RGB components and therefore responds to the intended
chromatic difference.
