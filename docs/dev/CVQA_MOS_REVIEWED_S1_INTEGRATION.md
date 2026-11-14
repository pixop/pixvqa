# CVQA-HDR-NR reviewed S1 integration (PixVQA staging)

Date: 2026-11-14
Branch: `feature/cvqa-hdr-nr-mos-calibration`
Status: **integrated in working tree; not production-subjective-quality certified**

## Policy model

| Policy ID | Default? | HDR PQ | SDR/VD | Coefficient source |
|---|---|---|---|---|
| `legacy_target_lambda1_both` | **yes** | Original_U target λ=1.0 | U_v2 target λ=1.0 | `mos_target_reweight_20261113/artifacts/endpoints.csv` |
| `reviewed_s1_hdr0_sdr1` | opt-in | Original_U baseline λ=0.0 | U_v2 target λ=1.0 | `s1_operational_20261114/manifest/coefficients_frozen_v1.json` |
| `selected_s1_sdr_p1_hdr_c0` | opt-in | Reviewed S1 C0 (no post-offset) | Reviewed S1 + native MOS offset | Same frozen S1 + composite offset locks |

The third policy (`selected_s1_sdr_p1_hdr_c0`) is documented in [CVQA_MOS_FINAL_SELECTED_IMPLEMENTATION.md](CVQA_MOS_FINAL_SELECTED_IMPLEMENTATION.md). Default remains legacy; selected is **shadow-not-prod**. **v2 (2026-11-14):** selected removes bilateral input support clamp (`input_support_clamp=false`); legacy and reviewed unchanged.

**SDR route coefficients are identical** under both policies. Only the HDR PQ backbone/gamma differ.

### Opt-in

```bash
export PIXVQA_CVQA_MOS_POLICY=reviewed_s1_hdr0_sdr1
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4
```

Unset or `legacy_target_lambda1_both` preserves prior provisional behavior. Unknown policy values **fail closed** (`mos_calibration.status=unavailable`, reason `invalid_policy:…`); canonical `quality_score` is never altered.

Direct Python:

```python
from pixvqa_cvqahdr.mos_calibration import predict_mos, REVIEWED_MOS_POLICY
estimate = predict_mos(8.5, transfer="sdr", encoded_height=1080, policy=REVIEWED_MOS_POLICY)
```

## Output semantics

- Field `mos_calibrated_cvqa` retained for protocol/C++ compatibility.
- Record also exposes `estimated_perceptual_quality` and `output_scale: "estimated perceptual quality 1-5"`.
- This is **not** measured subjective MOS; labels are synthetic training targets from development cohorts.

## CVQA upstream pins and reviewed output gate

Runtime pins (from `python/pixvqa_cvqahdr/__init__.py`):

| Pin | Value |
|---|---|
| `PINNED_COMMIT` | `76bff6d5c078d51be06eddf1ed11591f3678145a` |
| `PINNED_CHECKPOINT_SHA256` | `b37659902ce96e2f49de22e5e0c865c694921e3b8e3e55b770adc3be7cac8b88` |
| `PINNED_PROFILE_SHA256` | `601a7314e802a6867c33784355587bd892ad49990d2074cc64fe1bd7cdf6d49f` |

**Preflight vs reviewed gate (important):**

| Stage | What it checks |
|---|---|
| `verify_upstream()` at preflight | Git `HEAD == PINNED_COMMIT` only; returns `commit_verified` bool |
| `gate_reviewed_mos_outcome()` in adapter | **Reviewed and selected policies:** explicit `actual_commit`, checkpoint SHA-256, profile SHA-256 all match pins |

For `reviewed_s1_hdr0_sdr1` and `selected_s1_sdr_p1_hdr_c0`, estimated perceptual quality is **not published** (`mos_calibrated_cvqa` omitted, `status=unavailable`, `reason=cvqa_upstream_incompatible`) when any pin is missing or mismatched. `commit_verified=True` is **ignored** if `actual_commit != PINNED_COMMIT`. Legacy default policy is **not** gated.

Canonical `quality_score` is always unchanged. `mos_calibration.cvqa_upstream_compatibility` is still emitted for debugging.

**Coefficient digests (reviewed policy):**

| Field | Meaning |
|---|---|
| `coefficients_source_bundle_sha256` | Sealed research manifest bundle (`67d39be4…`) — provenance only |
| `coefficients_digest_sha256` | Same as source bundle (legacy API compat) — **not** runtime validation of embedded coeffs |
| `embedded_coefficients_sha256` | Recomputed from coefficients embedded in `mos_calibration.py` (`037d22ba…`) |

Reviewed policy manifest marks CVQA build ID as `UNRESOLVED_PLACEHOLDER` — coefficients assume the pinned PixVQA stack above.

## Verified checks (this integration pass)

See `scripts/test_cvqahdr_mos_calibration.py` output for the live count. Coverage includes:

- All legacy fixture tests (68 rows, λ=1 both routes) unchanged at ≤1e−8.
- Reviewed S1 golden 68 rows from sealed `golden_avt_68.json` at ≤1e−6 per row.
- HDR PQ legacy ≠ reviewed on shared scores; SDR bit-identical across policies.
- Invalid policy, HLG rejection, OOD support flags, protocol `mos_policy` token.
- Self-contained fixtures under `tests/fixtures/mos_calibration/` (no runtime dependency on `perceptual_optimization` repo).

## Production readiness blockers

| Blocker | Notes |
|---|---|
| Default remains legacy | Reviewed S1 requires explicit `PIXVQA_CVQA_MOS_POLICY` |
| Reviewed upstream gate | Fail-closed pin gate implemented; end-to-end runtime shadow not executed |
| No traffic shadow | ≥14d gate not executed |
| No prospective subjective study | Customer-facing SDR claims need independent trial |
| HDR evidence thin | n=8, two source families; PES outlier sensitivity |
| Display vs scored geometry | 30/54 SDR ratings presented upscaled |
| Temporal aggregation | `E[f(score)] ≠ f(E[score])`; clip-level only |
| Preflight pin scope | Checkpoint/profile SHA not hard-failed at preflight (reviewed MOS output gated only) |

## Changed paths

- `python/pixvqa_cvqahdr/mos_calibration.py` — dual policies, OOD flags, upstream gate
- `python/pixvqa_cvqahdr/adapter.py` — upstream metadata into MOS record
- `python/pixvqa_cvqahdr/__main__.py` — optional `mos_policy` protocol token
- `scripts/test_cvqahdr_mos_calibration.py` — reviewed golden + policy regression
- `tests/fixtures/mos_calibration/avt_reviewed_s1_hdr0_sdr1.json` — project-owned golden 68
- `docs/LEARNED_METRICS.md` — opt-in documentation

No commits, pushes, or deployment performed in this pass.
## Independently repeated integration checks — 2026-11-14

These are **software-integration observations**, not new native-subjective-MOS validation.

- Frozen research oracle: compared the PixVQA reviewed policy directly with the original 392-cohort `controlled_392_vs_437/avt_predictions.csv`; **68/68 observation keys, CVQA inputs, native MOS labels and model predictions matched exactly** (HDR8=8, SDR_HEVC54=54, VD_HEVC6_1.6H=6; maximum numerical difference 0). The golden fixture was also checked against its original upstream CSV.
- Dedicated tests: `PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py` — **65/65 PASS**. Python module compilation and `git diff --check` passed. Managed-venv CVQA chunking tests: **4/4 PASS**. Native `make cpu` and `make test-cpu` self-tests passed.
- Standalone CVQA CUDA adapter: the Aerial HEVC 1920x1080 10-bit SDR BT.709 video, `--max-frames 120`, produced canonical CVQA **9.718151437854885** and reviewed S1 estimate **4.006829798905894**, with 2 samples. `mos_calibration.cvqa_upstream_compatibility.integration_gate=verified`: upstream commit, checkpoint SHA256 and profile SHA256 all equalled the pins. Detailed JSON: `/tmp/cvqa_reviewed_s1_smoke_20261114.json` (local WSL test artifact).
- Native PixVQA C++ CLI: a 3.05-second, 183-frame, stream-copied Aerial 1920x1080 SDR BT.709 HEVC fixture at `/tmp/cvqa_s1_native_short_20261114.mp4` produced canonical CVQA **9.75393933094847**, raw model score **1.0907790660858154**, 3 sampled frames, and reviewed estimated quality **4.030906844066543**. Exit code 0; policy `reviewed_s1_hdr0_sdr1`.
- Repeating the **same native clip** with the environment policy unset (legacy default) produced *exactly the same* canonical CVQA, raw model score and SDR calibrated estimate (**4.030906844066543**), now labeled `legacy_target_lambda1_both`. This verifies real-video SDR route equivalence and preserves canonical output on this sample. HDR policy differentiation remains covered by golden tests, not a real HDR-video runtime smoke.
- Model-construction warning about SigLIP `pretrained=False` is expected: `NR_model.SigLIP2_384_multi_dataset()` first initializes its backbone and then `adapter.run()` loads the pinned checkpoint with `model.load_state_dict(state_dict, strict=True)` before evaluation. A random-weight inference was not performed.

**Still pending for production release:** representative longer-duration shadow operation, dedicated HDR10 PQ end-to-end smoke, prospective domain-representative ratings for customer claims, product signoff for which policy becomes default and for aggregation cadence, and formal changeset review/commit. The present work remains uncommitted and **not deployed**.
