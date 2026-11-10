# CUDA 13 learned-metrics validation — consolidated release gates

Date: 2026-11-10. Candidate branch: `security/resolve-learned-env-20261110`. **Do not merge automatically**.

## RTX 4070 (manually operated i9-9900K workstation)

Candidate Python 3.12.16, torch `2.13.0+cu130`, torchvision `0.28.0+cu130`; baseline repository `.venv` torch `2.11.0+cu128`; driver 617.42; both environments `pip check` passed. Full numerical values were supplied from the i9 terminal.

| Metric | Baseline | Candidate | Verdict |
|---|---:|---:|---|
| SimpleVQA Aerial 1199f | 3.524660348892212 | 3.524660348892212 | exact score, 3 repetitions each |
| CVQA-HDR-NR Aerial raw | 0.7270195484161377 | 0.7270038723945618 | small cross-stack delta, repeated scores stable |
| CVQA-HDR-NR calibrated | 9.147192303602765 | 9.147164601130157 | small delta |
| CVVDP SDR 8f | 9.195816993713379 | 9.195816993713379 | exact score |
| CVVDP PQ 8f | 10.0 | 10.0 | compatible, **saturated** (insufficient for strong numerical conclusion) |
| HDR-ChipQA SDR | 100.77857570271938 | 100.77857570271938 | exact score and feature SHA `4a89afe2...` |
| HDR-ChipQA PQ 32f | -73.23462450735413 | -73.23462450735413 | exact score and feature SHA `52d8c261...` |

HDR-ChipQA PQ fixture SHA-256 `f7c7337be7da801a3f25b26a654200f4e490c973eab453e8697acbcc898f1989` (320x180, 32f, YUV420p10le). The negative score is a raw SVR output and not subjective quality calibration. Its inferencer was Python 3.10 / scikit-learn 1.0.2, held constant across both runs.

SimpleVQA import-path defect fixed in commit `5d9858a`: isolated import test and real 1199-frame CUDA 13 Aerial run succeed with `PYTHONPATH` unset and exact previous candidate score. Checkpoint SHA-256 `8a9546ec164953ff74698f413e859260e5f396b9c1ea646a26c309b926c0421d`.

## RTX 5080: CVVDP PQ nonsaturated controlled A/B test

Source script: `scripts/make_cvvdp_pq_nonsaturated_fixture.py`. Generated with Python 3.12, NumPy 2.5.2, `--frames 16 --width 320 --height 180 --shift 64`. Limited-range, synthetic BT.2020 PQ code values (not perceptually calibrated HDR material). Raw YUV420p10le at 60 fps. CVVDP 0.5.7, display `standard_hdr_pq`, device `cuda`, 16 frames. Exact fixture hashes:

- Reference SHA-256: `97abefbe6ca921c68b698489443633f51f9fe593ed6d9f0854e2d6f048506eef`
- Distorted SHA-256: `0e0627c55b93302392bc661d32b4cdee1ddea355719dec7b4d6227aef0cc9ebb`

Both torch 2.11.0+cu128 (original repository `.venv`) and torch 2.13.0+cu130 (isolated candidate env) evaluated the **same bytes on the same RTX 5080**. Both returned **6.9395551681518555 JOD**, delta exactly 0.0. This nonsaturated test resolves the score-clipping caveat for the tested stack on RTX 5080.

Example independent RTX 4070 reproduction after fetching this branch (does not modify either virtualenv):

```bash
ROOT="$HOME/codeccomp/metrics/pixvqa"
TEST="$HOME/pixvqa-cuda13-validation"
SRC="$TEST/source"
OUT="$TEST/results/cvvdp_pq_nonsaturated"
mkdir -p "$OUT"
"$TEST/.venv/bin/python" "$SRC/scripts/make_cvvdp_pq_nonsaturated_fixture.py" \
  --out "$OUT" --frames 16 --width 320 --height 180 --shift 64
sha256sum "$OUT"/*.yuv
for stack in 12 13; do
  if [ "$stack" = 12 ]; then py="$ROOT/.venv/bin/python"; else py="$TEST/.venv/bin/python"; fi
  PYTHONPATH="$SRC/python" "$py" -m pixvqa_cvvdp \
    --ref "$OUT/pq_reference_420p10.yuv" --test "$OUT/pq_distorted_420p10.yuv" \
    --interpretation pq --display standard_hdr_pq --device cuda --nframes 16 \
    --raw-width 320 --raw-height 180 --raw-fps-num 60 --raw-fps-den 1 \
    --output-json "$OUT/cuda${stack}_details.json" > "$OUT/cuda${stack}.json" \
    2> "$OUT/cuda${stack}.stderr" || exit 1
done
"$ROOT/.venv/bin/python" - "$OUT" <<'PYCODE'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); a=json.loads((p/'cuda12.json').read_text()); b=json.loads((p/'cuda13.json').read_text())
print('CUDA 12.8 JOD',a['jod']); print('CUDA 13.0 JOD',b['jod']); print('delta',b['jod']-a['jod'])
PYCODE
```

## RTX 4070: nonsaturated CVVDP PQ dual-stack confirmation

The i9-9900K / RTX 4070 workstation independently regenerated the 16-frame synthetic PQ fixture from `scripts/make_cvvdp_pq_nonsaturated_fixture.py` at branch commit `4901661`. Fixture SHA-256 verification passed (the terminal explicitly reported `pq_distorted_420p10.yuv: OK`; no generator mismatch was reported). Both runs exited 0:

| Runtime | CVVDP JOD |
|---|---:|
| torch 2.11.0+cu128 | 6.9395551681518555 |
| torch 2.13.0+cu130 | 6.9395551681518555 |
| Delta | **0.0** |

The RTX 4070 scores equal the independently recorded RTX 5080 scores for this fixture. This closes the nonsaturated CVVDP PQ cross-stack validation on both GPUs. The quality score remains a synthetic regression value, not a subjective MOS calibration.

## Packaging and provenance

- Lower-level `scripts/setup_learned_env.sh` intentionally installs the shared learned runtime only; the full `scripts/install_learned_metrics.sh` calls `scripts/setup_hdr_chipqa_envs.sh` for missing HDR-ChipQA feature dependencies and the separate pinned scikit-learn inferencer. Do not describe lower-level setup alone as full HDR-ChipQA provisioning.
- The original repository main branch still expects cu128 stack in its CVVDP provenance metadata; its candidate-run `version_matches_expected=false` was not a score failure. The experimental branch matches torch2.13/cu130. CVVDP package `direct_url.json` did not expose the pinned Git commit, so verify the preserved source/export stamp `COLORVIDEOVDP_COMMIT` and the pinned exported source tree rather than claiming the wheel metadata proves the commit.
- The CVQA-HDR-NR RTX 4070 delta is small but not zero; acceptance requires the agreed numerical tolerance and repeatability policy.

## Security and merge gates

An earlier GitHub push reported four default-branch alerts (one high, one moderate, two low); the latest push response at commit `6117399` reports **one low-severity alert**, linking to `https://github.com/pixop/pixvqa/security/dependabot/8`. The intervening alert-state changes were not independently investigated, and push notices are not a substitute for an alert inventory. Attempted unauthenticated GitHub Dependabot alerts API access from RTX 5080 returned **HTTP 401 Requires authentication**; affected manifests, vulnerable dependency ranges, fixes, exploitability, and alert #8 details **have not been inspected**. An authenticated reviewer must inspect the remaining alert and confirm the present alert inventory before security approval.

Remaining: confirm approved CVQA tolerance; authenticate and triage alerts; review the complete branch diff/CI; only then consider merging. Keep `main` unchanged until authorized.
