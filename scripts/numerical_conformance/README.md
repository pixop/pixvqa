# Numerical conformance (SDR/PQ/HLG UHD fixtures and baselines)

Synthetic **3840×2160**, **5 frames**, **60 fps**, **YUV420 10-bit limited** fixtures live under
`.cache/pixvqa-numerical-fixtures/` (gitignored). They are generated automatically; do not commit
`*.yuv` / `*.mp4` / `*.mkv`.

## Scene and color science

Module `scene.py` builds five frames of deterministic scene-linear RGB (gradient, shadow patch,
highlight, stripes, moving edge). Module `color_science.py` maps linear RGB to:

| Family | Display / encoding path |
|--------|-------------------------|
| SDR | BT.709 gamma (1/2.4) on linear, then BT.709 YCbCr limited 10-bit |
| PQ | ST 2084 PQ OETF on linear 0–10000 nits, BT.2020 YCbCr limited |
| HLG | BT.2100 HLG OETF, BT.2020 YCbCr limited |

PQ/HLG are **not** relabelled SDR samples; each family uses its own non-linear RGB before matrixing.

## Containers

| Asset | Container | Notes |
|-------|-----------|--------|
| `reference_lossless.*` | **MKV** (HEVC lossless Main10) | MP4 lossless HEVC is unreliable; MKV documented substitute |
| `distorted_lossy.mp4` | **MP4** (HEVC lossy) | Exercises MP4 + VUI |

Verified: decode(lossless MKV) == `reference.yuv`; decode(`distorted_lossy.mp4`) == `distorted.yuv`.

## Metric registry

See `registry.py` (mirrors `kMetricCaps` in `src/main.cpp` and `docs/METRICS_AND_FORMATS.md`).

## Commands (RTX 4070 WSL)

```bash
pip install numpy   # if missing
make -j8 full cpu swdec

export PIXVQA_BIN=build/full/pixvqa
export PIXVQA_CLI_CONFORMANCE_STRICT=1
export PIXVQA_NUMERICAL_STRICT=1

# Phase 1 only (42 CLI cases)
python3 scripts/test_metric_parameters.py

# Phase 2 smoke (fast numerical)
python3 scripts/test_metric_numerical_conformance.py --profile smoke --strict

# Phase 2 comprehensive
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict

# Both phases
python3 scripts/run_pixvqa_validation.py --profile comprehensive --strict
```

## Baselines

Candidate results: `baselines/numerical_conformance/<profile>.candidate.json`  
Approve after review:

```bash
python3 scripts/test_metric_numerical_conformance.py --profile smoke --approve-baseline
```

Approved baselines are **never** overwritten by normal runs.

Optional real-content CVVDP check (not part of portable fixtures): BerlinSeqs JOD
5f=`8.788558959960938`, 60f=`9.17333984375`.
