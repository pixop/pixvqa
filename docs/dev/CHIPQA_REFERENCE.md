# ChipQA / HDR-ChipQA reference integration

Upstream checkouts (git-ignored `deps/`, no LICENSE in upstream roots):

| Metric | Repo path | Commit | Models (SHA-256 prefix) |
|--------|-----------|--------|-------------------------|
| `chipqa` | `deps/ChipQA` | `53af39c…` | scaler `761c2b17…`, SVR `f13b6f97…` |
| `hdr-chipqa` | `deps/hdr_chipqa` | `916157d…` | scaler `ea5815fe…`, SVR `825f4700…` |

## Python environments

Two interpreters are required: one for feature extraction (CPython 3.12+, upstream `chipqa.py` / `hdr_chipqa.py`) and one for SVR inference (CPython 3.10, sklearn 1.0.2, bundled `.z` models only). Adapters do **not** embed machine-specific defaults; set env vars or pass CLI flags.

| Role | Environment variable | CLI flag |
|------|---------------------|----------|
| SDR feature extraction | `PIXVQA_CHIPQA_FEATURE_PYTHON` | `--chipqa-python` |
| SDR SVR inference | `PIXVQA_CHIPQA_INFERENCE_PYTHON` | `--chipqa-inference-python` |
| HDR feature extraction | `PIXVQA_HDR_CHIPQA_FEATURE_PYTHON` | `--hdr-chipqa-python` |
| HDR SVR inference | `PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON` | `--hdr-chipqa-inference-python` |

Values may be a venv directory or a `bin/python` path. TOML `[chipqa]` / `[hdr-chipqa]` sections are also supported when running via `pixvqa`.

Example (paths are deployment-specific):

```bash
export PIXVQA_CHIPQA_FEATURE_PYTHON=/path/to/pixvqa-chipqa
export PIXVQA_CHIPQA_INFERENCE_PYTHON=/path/to/pixvqa-chipqa-ref
export PIXVQA_HDR_CHIPQA_FEATURE_PYTHON=/path/to/pixvqa-chipqa
export PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON=/path/to/pixvqa-chipqa-ref
python3 scripts/test_chipqa_reference.py
```

Joblib feature dumps from py3.12 are bridged to `.npy` before inference (no user pickle paths).

## Contract

- **`chipqa`**: SDR compressed video only (`-t`); rejects PQ/HLG metadata.
- **`hdr-chipqa`**: PQ-coded **raw** YUV420p10le BT.2020 only; requires `-s`, `-f 420p:10`, `--fps`; rejects HLG and compressed inputs.
- **Incompatible** with all `pu21-*` metrics (both orders).
- **QCS unsupported** (`--qcs` exits setup).
- Pooled video scalar only; no per-frame temporal scores.

## Goldens (`reports/chipqa-reference/REFERENCE_BASELINE.json`)

| Clip | Score | Feature SHA (prefix) |
|------|-------|-------------------|
| `samples/dist_hevc.mp4` 960×540 60f | `46.09665088` | `268cedf8…` |
| `derived_pq_distorted_960x540_60f.yuv` (DERIVED_REFERENCE_FIXTURE) | `-77.42408692` | `61a10beb…` |

Parity: `python3 scripts/test_chipqa_reference.py`
