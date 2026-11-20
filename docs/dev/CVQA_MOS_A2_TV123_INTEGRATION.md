# CVQA-to-MOS — experimental A2-TV123 integration (feature branch)

**Updated:** 20 November 2026  
**Branch:** `feature/cvqa-mos-a2-tv123-ai25-20261120`  
**Status:** Experimental opt-in calibration. **Not** merged, **not** production-certified.

## Selection summary

A2-TV123 jointly calibrates a unified sigmoid plus smoothstep amplitude to the **individual** subjective scores of TV1, TV2 and TV3 (435 human ratings on 145 clips, 15 source families). Amplitude tuning used **75% human / 25% AI** source-balanced MAE (40 AI-rated clips from contact sheets). **No AI inference runs at scoring time.**

Scientific decision (external, read-only):

```text
perceptual_optimization/calibration/production_readiness_20261113/
AI_EXTENSION_TV2_20261119/analysis/AI_INFORMED_POSTCALIBRATION_20261120/
A2_TV123_INDIVIDUAL_RATINGS_20261120/
SELECTED_A2_TV123_IMPLEMENTATION_DECISION_20261120.md
```

Experiment summary: sibling `results/experiment_summary.json` in that folder.

## Frozen formula (do not retune)

For canonical CVQA score `x`, `I_2160` (1 at 2160p), `I_PQ` (1 for PQ HDR):

```text
z = 0.09446591142061583
  + 0.6046767170474217 * ((x - 8.5) / 2)
  + 0.6167771933889601 * I_2160
  + 0.11819874714947097 * I_PQ

baseline = 1 + 4 / (1 + exp(-z))
t = clamp((x - 7.75) / 2.25, 0, 1)
MOS = clamp(baseline + 0.240 * t*t*(3 - 2*t), 1, 5)
```

**Mutually exclusive with v7:** no M7 metadata correction, no SDR +0.5107733 native offset, no C2 low-tail adapter, no per-transfer score normalization.

## Runtime default vs opt-in

| Setting | Algorithm | Behavior |
|---|---|---|
| *(unset)* | `v7_convex_c2` | Frozen v7+M7+C2 — **unchanged default** |
| explicit / env | `a2_tv123_ai25` | Experimental A2-TV123 |

### CLI

```bash
PYTHONPATH=python python3 -m pixvqa_cvqahdr \
  --test clip.mp4 --repo … --checkpoint … --profile … --output-json out.json \
  --mos-calibration-algorithm a2_tv123_ai25
```

### Environment

```bash
export PIXVQA_MOS_CALIBRATION_ALGORITHM=a2_tv123_ai25
```

Aliases accepted: `a2`, `a2_tv123`, `a2_tv123_ai25`. Invalid values **fail closed** with `invalid_mos_calibration_algorithm:…`.

The legacy env `PIXVQA_CVQA_MOS_POLICY` remains v7-only and is **not** used to select A2.

## Support gating (unchanged domain rules)

MOS is attempted only when ffprobe reports supported **SDR or PQ** transfer at **1080p or 2160p**. HLG, ambiguous transfer, unsupported heights and upstream pin mismatch suppress MOS without changing canonical CVQA.

A2 score OOD flags use nominal research span **[0, 10]** on the canonical 0–10 scale (extrapolation flagged above 10; input never clamped). Runtime does **not** implicitly support 720p despite research LOW curves including 720p content.

## Provenance in JSON / protocol

- `mos_calibration_algorithm`: `v7_convex_c2` or `a2_tv123_ai25`
- A2 records include `unified_sigmoid`, `smoothstep_amplitude`, `ai_loss_fraction`, dedicated digests — **not** v7 fitted-model hashes, SDR selection locks or C2 low-tail metadata
- Protocol lines add `pixvqa-value mos_calibration_algorithm …` when present

## Verification

```bash
# v7 regression (must pass unchanged)
PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py

# A2 experimental suite
PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration_a2.py
```

## Caveats

- Human-only control C1-TV123 had slightly lower pooled human MAE than selected A2; do not claim AI improved subjective accuracy.
- 435 TV ratings are triplicated labels on 145 clips — not 435 independent videos.
- Retrospective study only; prospective validation and production qualification are separate gates.