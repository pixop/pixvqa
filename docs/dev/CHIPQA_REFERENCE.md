# HDR-ChipQA unified reference integration

Standalone ChipQA (`deps/ChipQA`) is **not** integrated. PixVQA uses upstream **HDR-ChipQA** (`deps/hdr_chipqa`) for SDR and PQ HDR, matching Ebenezer et al. 2024 §4.7 (BT.709 gamma-coded YUV + `color_space=BT709` for SDR; BT.2020 PQ-coded YUV for HDR). **HLG is supported only as an experimental extrapolation** (same upstream `color_space=BT2020` feature path, no EOTF); it is **not** relabeled as PQ and scores are **not** paper-validated.

| Metric | Repo path | Commit | Models (SHA-256 prefix) |
|--------|-----------|--------|-------------------------|
| `hdr-chipqa` | `deps/hdr_chipqa` | `916157d…` | scaler `ea5815fe…`, SVR `825f4700…` |

## Pipelines

| Input | Upstream flags | Notes |
|-------|----------------|-------|
| Compressed SDR (MP4/MKV…) | `--color_space BT709 --bit_depth 8` | ffprobe rejects PQ; HLG tags accepted and decoded to yuv420p10le |
| Compressed HLG | `--color_space BT2020 --bit_depth 10` | Requires arib-std-b67, bt2020/bt2020nc, 10-bit; **experimental** |
| Raw SDR YUV | `--color_space BT709 --bit_depth 8` (or 10) | `-f 420p:8` or `420p:10`; no PQ EOTF |
| Raw PQ HDR YUV | `--color_space BT2020 --bit_depth 10` | `-f 420p:10`, `--fps`; PQ-coded luma as-is |
| Raw HLG YUV | `--color_space BT2020 --bit_depth 10` | `-f 420p:10`, `--transfer hlg` (or `arib-std-b67`); **experimental** |

Upstream `hdr_chipqa.py` exposes only `--color_space {BT709,BT2020}` and `--bit_depth`; there is **no per-transfer switch**. PQ and HLG both use `color_space=BT2020` with 10-bit YUV codes fed directly to the reader (no EOTF). This is a **technical extrapolation** for HLG, not published validation.

### Input metadata (transfer / matrix)

| Mode | ffprobe transfer | ffprobe matrix | CLI override | Adapter `color_space` |
|------|------------------|----------------|--------------|----------------------|
| SDR | bt709, gamma22, unspecified 8-bit | bt709 | `--transfer sdr` (optional) | BT709 |
| PQ | smpte2084, pq | bt2020 / bt2020nc | `--transfer pq` | BT2020 |
| HLG | arib-std-b67, hlg | bt2020 / bt2020nc | `--transfer hlg` | BT2020 (experimental) |

Unknown HDR transfers (e.g. smpte428) are rejected at setup.

## HLG experimental warning

Once per evaluation, stderr logs:

> WARNING: HDR-ChipQA on HLG is experimental: HLG performance was not experimentally validated in the HDR-ChipQA paper; scores are uncalibrated and should not be treated as established subjective-quality predictions.

## Python environments

Two interpreters: feature extraction (CPython 3.12+, `hdr_chipqa.py`) and SVR inference (CPython 3.10, sklearn 1.0.2).

| Role | Path (known working) | Environment variable | CLI flag |
|------|---------------------|---------------------|----------|
| Feature extraction | `/home/sergio/.venvs/pixvqa-chipqa/bin/python` | `PIXVQA_HDR_CHIPQA_FEATURE_PYTHON` | `--hdr-chipqa-python` |
| SVR inference | `/home/sergio/.venvs/pixvqa-chipqa-ref/bin/python` | `PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON` | `--hdr-chipqa-inference-python` |

TOML `[hdr-chipqa]` section supported when running via `pixvqa`.

Legacy env names `PIXVQA_CHIPQA_FEATURE_PYTHON` / `PIXVQA_CHIPQA_INFERENCE_PYTHON` are still honored as fallbacks.

### Provision (do not install into system Python)

```bash
# Feature env — Python 3.12 + numpy shim for upstream feature code
python3.12 -m venv ~/.venvs/pixvqa-chipqa
~/.venvs/pixvqa-chipqa/bin/pip install numpy scipy scikit-image joblib opencv-python-headless

# Inference env — pinned sklearn 1.0.2 (GridSearchCV pickle incompatible with 1.5+)
python3.10 -m venv ~/.venvs/pixvqa-chipqa-ref
~/.venvs/pixvqa-chipqa-ref/bin/pip install 'numpy==1.22.4' 'scikit-learn==1.0.2' 'joblib==1.1.1'
```

## Contract

- **`hdr-chipqa`**: unified SDR + PQ HDR + **experimental HLG**; included automatically in `sdr`/`pq`/`hlg` groups; also selectable alone (`--metrics hdr-chipqa`) or appended redundantly (e.g. `--metrics pq,hdr-chipqa`).
- HLG must not be silently inferred from 10-bit input; explicit stream tags or `--transfer hlg` / `arib-std-b67` required.
- Unknown HDR transfers (e.g. smpte428) are rejected.
- **QCS:** no `HDR-ChipQA-QCS` line (pooled scalar only); `--qcs` with **only** `hdr-chipqa` exits setup. Groups and mixes with eligible metrics keep QCS on the other selections.
- Pooled video scalar only.
- **Group selection:** `sdr`/`pq`/`hlg` set `hdr_chipqa.selected`; repository resolves from `[hdr-chipqa].repository`, `--hdr-chipqa-repo`, or `deps/hdr_chipqa` beside the executable (setup exits if still empty). Transfer resolves from explicit HDR options, inferred group transfer, or `metric_group`.
- **PU21:** no blanket exclusion; PU21 and HDR-ChipQA may run together (e.g. `--metrics hlg` or `--metrics pq`).

## Goldens

| Clip | Pipeline | Score | Feature SHA (prefix) |
|------|----------|-------|----------------------|
| `samples/dist_hevc.mp4` | SDR BT709 | `100.80971697` | `f359dcb0…` |
| `reports/chipqa-reference/derived_pq_distorted_960x540_60f.yuv` | PQ HDR | `-77.42408692` | `61a10beb…` |

HLG raw scoring on the same PQ YUV fixture yields **identical features** (same YUV domain) but is labeled `experimental` in adapter output.

### Reference parity — exact commands

```bash
cd /mnt/d/projects/pixvqa-public
export PIXVQA_HDR_CHIPQA_FEATURE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa/bin/python
export PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa-ref/bin/python

# All goldens (SDR BT709, PQ BT2020, HLG raw + compressed contract)
$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON scripts/test_chipqa_reference.py
```

**Skip vs pass:** tests print `SKIP …` when fixtures or env vars are missing; `OK …` when a check passes; `FAIL …` + exit 1 on regression. With both env vars set (paths above), expect **14× OK, 0× SKIP, 0× FAIL**.

### pixvqa CLI examples

```bash
# SDR compressed
build/cpu/pixvqa --metrics hdr-chipqa -t samples/dist_hevc.mp4 \
  --hdr-chipqa-python "$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON" \
  --hdr-chipqa-inference-python "$PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON"

# PQ raw YUV (10-bit, fps required)
build/cpu/pixvqa --metrics hdr-chipqa -t reports/chipqa-reference/derived_pq_distorted_960x540_60f.yuv \
  -s 960x540 -f 420p:10 --fps 60 --transfer pq \
  --hdr-chipqa-python "$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON" \
  --hdr-chipqa-inference-python "$PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON"

# HLG raw YUV (experimental)
build/cpu/pixvqa --metrics hdr-chipqa -t reports/chipqa-reference/derived_pq_distorted_960x540_60f.yuv \
  -s 960x540 -f 420p:10 --fps 60 --transfer hlg \
  --hdr-chipqa-python "$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON" \
  --hdr-chipqa-inference-python "$PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON"
```

## Known limitations

- HDR-ChipQA training dark-MOS range (310 clips): ≈4.7–89.8; BerlinSeqs pilot scores exceeded 100 (extrapolation).
- PQ golden fixture is **960×540 lanczos downscale in PQ-coded YUV domain** — not native 4K upstream parity.
- HLG has **no subjective validation** in Ebenezer et al. 2024; scores are uncalibrated.
- Standalone ChipQA SVR pickles are **not** interchangeable with HDR-ChipQA unified models.
- Feature extraction needs `numpy_legacy_shim.py` on CPython 3.12+ (upstream uses deprecated `np.int` aliases).

## Validation summary

Full primary-pass methodology, pristine PQ/SDR ladders, MOS-scale upstream review, and **excluded** double-compression experiments: [`HDR_CHIPQA_VALIDATION_SUMMARY.md`](HDR_CHIPQA_VALIDATION_SUMMARY.md).

**Primary pristine PQ evidence (not exhaustive):** BerlinSeqs CQP+CRF (`reports/hdr_chipqa_fixed_qp_20261107`, `reports/hdr_chipqa_crf_vs_qp_pristine_20261107`), BerlinSeqs corrected 30 s segment (`reports/hdr_chipqa_source_audit_20261107`), DinnerScene CQP+CRF (`reports/hdr_chipqa_pq_final_single_validation_20261107`). **SDR:** CityAlley (`reports/hdr_chipqa_monotonicity_20261107`). **HLG:** experimental only.

Scores are unbounded SVR outputs in MOS **training** units (≈4.7–89.8 on dark scenes); values outside that range are extrapolation, not calibrated subjective quality.

## Operational testing

```bash
make cpu
export PIXVQA_HDR_CHIPQA_FEATURE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa/bin/python
export PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa-ref/bin/python
$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON scripts/test_chipqa_reference.py
python3 scripts/test_learned_metrics_cli.py   # mocked backends, no GPU
```

See also [`NR_METRIC_INTEGRATION_PLAN.md`](NR_METRIC_INTEGRATION_PLAN.md).
