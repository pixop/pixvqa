# Reference RGB gamut clamp

Opt-in normalization for **full-reference** metrics only. When enabled, PixVQA may adjust the
decoded reference in place before native FR metrics run. The test stream and all NR metrics are
unchanged.

This is a **code-domain reference transform**, not a reproduction of any particular production
processing pipeline. It applies **matrix-domain R′G′B′ clamping in the nonlinear code-value
domain** (no PQ/HLG EOTF) on planar YUV only.

## Algorithm (delta-only)

Per luma pixel:

1. Reconstruct input Y′CbCr (nominal) using the chosen chroma location and upsampling filter.
2. Convert to nonlinear R′G′B′ with the selected matrix.
3. If **all** channels lie in [0, 1] (within tolerance), **do nothing** for that pixel.
4. Otherwise clamp R′G′B′ to [0, 1], invert to Y′CbCr, and record **nominal deltas** ΔY, ΔCb,
   ΔCr vs. the reconstructed input.

Apply:

- **Y**: each luma code += quantize(yn₀ + ΔY) − yn₀ only when ΔY ≠ 0.
- **Cb/Cr**: each luma pixel **always** contributes chroma upsampling weights to the
  denominator; only clipped pixels add weighted ΔCb/ΔCr to the numerator. Final chroma correction
  is `sum(w·Δ) / sum(w)` per site (same bilinear/nearest weights as the read path).

If **no** pixel required clamping, the frame is bypassed and remains **bit-identical**.

In-gamut frames must stay **bit-identical** for any subsampling, chroma phase, and upsampler.

## CLI

```bash
pixvqa --metrics psnr -r ref.yuv -t test.yuv -s 1920x1080 -f 420p:10 \
  --reference-rgb-gamut-clamp \
  --reference-matrix bt709 \
  --reference-range limited
```

| Flag | Meaning |
|------|---------|
| `--reference-rgb-gamut-clamp` | Enable reference normalization (requires an FR metric in `--metrics`). |
| `--reference-matrix` | `bt709`, `bt2020nc`, `bt601`, `smpte170m`, `bt470bg`, `smpte240m`. Rejects `bt2020c` / `ictcp`. |
| `--reference-range` | `limited` or `full`. |

**Raw** YUV: matrix and range are **required** (no silent BT.709 default).

**Compressed** reference: matrix and range default from stream metadata when unambiguous; chroma
location defaults from stream metadata when present, else **left**; override with `--hdr-chroma-loc`
/ `--ref-hdr-chroma-loc` and `--hdr-chroma-up`.

Rejected combinations:

- BT.2020 **constant luminance** matrix
- **ICtCp** / unknown matrix without an explicit override
- **CVVDP** in the same run (CVVDP reads `-r` from disk)
- **NR-only** `--metrics` (flag must pair with at least one FR metric; mixed FR+NR runs clamp only
  on the native FR path)

## Supported scope

- Matrices: BT.709, BT.2020 NCL, BT.601 family, SMPTE 240M.
- 8/10-bit planar 420/422/444.
- PQ/HLG content uses the **same** code-domain transform (matrix + range only; transfer is not
  applied).

## Tests

From the repo root after `make cpu`:

```bash
make cpu && make swdec
build/cpu/selftest
PIXVQA_BIN=build/cpu/pixvqa python3 scripts/test_reference_rgb_gamut_clamp.py
bash scripts/test_reference_rgb_gamut_compressed.sh   # optional, needs ffmpeg + swdec
```

BT.601 matrix coefficients apply **only** inside this feature (`ref_parse_gamut_matrix_token`);
global `--hdr-matrix` / `hdr_display_cdm2` behavior is unchanged.

The integration script includes subsampling identity fixtures that fail on the pre-delta anchor
resynthesis bug (420/422, left/center/topleft, bilinear/nearest).

## Limitations

- CVVDP and other learned FR backends that bypass the native frame loop are not integrated.
- Chroma delta spreading assumes the PU21-compatible upsampling model; exotic MPEG chroma siting
  may need explicit `--ref-hdr-chroma-loc`.
- For subsampled 4:2:0 and 4:2:2, chroma back-projection is an approximate resampling of per-pixel
  clipping deltas; because chroma is shared across pixels, the resulting YUV may not map to
  perfectly in-gamut RGB at every pixel and a second clamp pass is not guaranteed to be
  idempotent.