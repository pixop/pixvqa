# CVQA MOS independent audit remediation (2026-11-14)

Remediation for findings M01–M04 and L01–L02. Frozen coefficients, offsets, and digests are unchanged.

| Artifact | SHA256 |
|----------|--------|
| Frozen S1 coefficient bundle | `67d39be487fccee6b0a08ef33a4dd7e8adf1745578eff9060ab48e394069a02c` |
| Selected composite digest | `463b641347d68585ea9f99fc6d8c58b3ea274896458f6b1e58cbf58592b43cd3` |
| Selected embedded digest | `b855164620e66b208826cc68d516fb89297f240ff3c4f019288790d3b1d39916` |

Post-remediation verification (`PYTHONPATH=python python3 -c …`): all three digests match.

## M01 — Adapter MOS pipeline boundary

**Issue:** `apply_mos_calibration` alone was guarded; `gate_reviewed_mos_outcome` and `mos_calibration_record` could raise after canonical CVQA completed, discarding the run.

**Action:** `python/pixvqa_cvqahdr/adapter.py` (~636–665). The full optional MOS stage (apply → upstream gate → record) is inside one `try`/`except`. On failure, `minimal_unavailable_mos_record()` emits a fail-closed record with `calibration_error:<ExceptionType>`; `quality_score` and core record fields are written with `json.dumps(..., allow_nan=False)`.

**Tests:** `MosAdapterPipelineSafetyTests` (gate/record failure → minimal record).

## M02 — Integrity enforcement under `python -O`

**Issue:** `assert` digest checks are disabled under optimized Python (`-O`).

**Action:** `python/pixvqa_cvqahdr/mos_calibration.py`: `_enforce_digest_match()` replaces asserts in `coefficients_digest()` and `embedded_coefficients_sha256()`. `apply_mos_calibration()` verifies embedded (+ selected/reviewed composite) digest before publishing an available outcome. `minimal_unavailable_mos_record()` skips nested digest verification.

**Tests:** `MosIntegrityEnforcementTests` (mismatch raises; subprocess normal + `-O`; in-memory `m1_params` tamper → unavailable, canonical preserved; minimal record has no digest fields).

## M03 — Fixture skips treated as success

**Issue:** `setUpClass` used `unittest.SkipTest` when fixtures were absent; runner counted skips as success.

**Action:** `scripts/test_cvqahdr_mos_calibration.py`: `_require_fixture()` fails on missing path or ≠68 rows. `MosFixtureAvailabilityTests` + fault-injection patch. `__main__` exits non-zero on any skip.

**CI:** Registered in `.github/workflows/qa.yml` as stdlib-only step (`PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py`).

**Test output:**

```
Ran 99 tests in 0.408s
OK
```

## M04 — Native executable rebuilt and tested with real HDR10/PQ

**Original issue:** The old `bin/pixvqa` binary omitted the MOS-calibrated combined-summary field although `src/main.cpp` contained the feature. The audit found this to be an artifact/source mismatch, not a calibration arithmetic defect.

**Resolution:** Rebuilt two identified, local-only binaries from the post-sync source (base Git commit `1b2d5a716dc35152fc3b745490da1ce4ffa4b290` plus the uncommitted selected-calibration implementation):

- `build/cpu/pixvqa`: SHA-256 `98d5723d891ef64ed7a9cb1c97612d3375e0fd177ea3926f043c98864084021b`; new MOS summary symbols present.
- `build/swdec/pixvqa`: SHA-256 `24d74fd0aad1d301e00bd11e29b9b2ca71decf6fdb1980efdaadb425b1b1ee93`; software FFmpeg decode and MOS summary present.

The **old installed `bin/pixvqa` is intentionally untouched**. Do not promote it as the reviewed native artifact. Neither new executable is staged or intended for Git.

**Environment:** Existing FFmpeg n8.1.3 executables at `/home/sergio/bin/ffmpeg` and `/home/sergio/bin/ffprobe` were selected via the documented process-local overrides `PIXVQA_FFMPEG_BIN`, `PIXVQA_FFPROBE_BIN` and `PIXVQA_CVQAHDR_FFPROBE`. Both SDR and genuine HDR preflights passed. No installation, tagging, transcoding or repository environment edits were necessary.

**Real-media evidence (two midpoint frames from the first two seconds, CUDA inference):**

| Route / video | Canonical CVQA | Selected MOS | Runtime result |
|---|---:|---:|---|
| SDR HEVC Main10/BT.709, BasketballDrive 1080p | 8.382970297550287 | 3.3051935956878817 | Standalone Python and native combined PASSED |
| HDR10 PQ/BT.2020, LIVE CenterPanorama 1080p/6M | 9.261606105959345 | 3.3137226691011237 | Standalone Python and native standalone PASSED |
| HDR10 PQ/BT.2020, LIVE CenterPanorama 2160p/15M | 9.932609594080645 | 3.9458550479642307 | Standalone Python and native combined PASSED |

**Native combined cases:** In `psnr,cvqa-hdr-nr` mode, the rebuilt software-decoder executable printed `MOS-calibrated CVQA:3.3051935956878817` for available SDR and `MOS-calibrated CVQA:3.9458550479642307` for available genuine 2160p HDR10. An invalid-MOS-policy negative control printed `MOS-calibrated CVQA:unavailable` while preserving canonical CVQA `8.382970297550287`. The HDR 2160p pair produced PSNR average approximately 48.71 dB. An SDR reference identical to the distorted clip was deliberately used for the combined-mode integration check; its PSNR of 999.99 is not a meaningful coding-quality measurement.

**Exact source video identities (SHA-256):**

- BasketballDrive SDR HEVC: `9a92653369f09212c34221d9e2ae728f14671e795697c4ad42e24682fd138e3f`
- LIVE HDR CenterPanorama 1080p/6M: `3266d6beccd490c0285712d2b2731c2eb1fba1bd7398128a67f8ff82fc042f7a`
- LIVE HDR CenterPanorama 2160p/15M: `51ebb034a301015966232bffc3338b02cc75bbb09689997ceb300d5947f47a24`

These media tests establish native routing and report serialization for genuine PQ material. They do **not** certify full-duration production throughput, complete codec/profile coverage, native Windows behavior or subjective MOS accuracy.

## L01 — Upstream gate loses diagnostic flags

**Action:** `gate_reviewed_mos_outcome()` now copies `input_support_clamp` and `support_extrapolation` into the fail-closed outcome.

**Tests:** `MosGateDiagnosticPreservationTests.test_upstream_gate_preserves_support_extrapolation_flags` (score 11.0 PQ OOD + unpinned upstream).

## L02 — Non-finite scores in JSON audit fields

**Action:** `_json_safe_number()` sanitizes MOS audit numerics to `null`; adapter uses `allow_nan=False` for JSON writes. Nonfinite canonical input still fails MOS closed with explicit reason.

**Tests:** `MosJsonSanitizationTests`.

## Modified paths

| Path | Change |
|------|--------|
| `python/pixvqa_cvqahdr/adapter.py` | Unified MOS try boundary; `allow_nan=False` |
| `python/pixvqa_cvqahdr/mos_calibration.py` | Digest enforcement, sanitization, minimal record, gate flags, pre-publish integrity |
| `scripts/test_cvqahdr_mos_calibration.py` | Fixture hard-fail, +11 regression tests, skip rejection |
| `.github/workflows/qa.yml` | MOS suite in CPU hosted QA |
| `docs/dev/CVQA_MOS_INDEPENDENT_AUDIT_REMEDIATION_20261114.md` | This document |

## Independent post-fix verification

- `PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py`: **99/99 passed**, no skips.
- `PYTHONPATH=python .venv/bin/python scripts/test_cvqahdr_mos_calibration.py`: **99/99 passed**, no skips.
- `PYTHONPATH=python .venv/bin/python -O scripts/test_cvqahdr_mos_calibration.py`: **99/99 passed**, no skips.
- `PYTHONPATH=python .venv/bin/python tests/test_cvqa_chunking.py`: **4/4 passed**.
- Selected composite, embedded and original frozen-manifest SHA-256 digests unchanged. Selected SDR/HDR MOS is numerically identical to the auditor's independent values at CVQA=10/11.
- The required calibration test suite is registered in `.github/workflows/qa.yml` CPU hosted QA and succeeds under system Python.

## Remaining release qualifications

1. **Build provenance:** The tested native candidates were built from the latest synchronized upstream base and the then-uncommitted feature work. After a reviewed source commit, the release process should rebuild and identify the native binary against that committed SHA. The prior installed `bin/pixvqa` remains stale and must not be silently deployed.
2. **Upstream source-state approval:** The CVQA upstream compatibility gate verifies declared Git commit and checkpoint/profile SHA identities; it does not hash every imported source byte. Independently approve source patches or require clean upstream deployment builds.
3. **Clean installation/packaging:** A standalone wheel and all optional learned-metric dependencies were not newly qualified by this focused remediation. CI's fixture-backed numeric suite is working, but a clean packaging test is still a separate release gate.
4. **Scientific validity:** The retrospective AVT data and the three short genuine-media integrations do not establish prospective subjective MOS accuracy, HDR MOS 4.5–5 performance, long-duration temporal behavior or out-of-support reliability.
5. **Deployment:** This work is suitable for a reviewed *feature-branch source commit and push*, not automatic production deployment. Default remains legacy; selected SDR/HDR calibration stays opt-in.

**Audit closure status:** M01, M02, M03, M04, L01 and L02 have implementation changes and direct local regression evidence. External auditor re-review of the patches and a final native release-build/packaging verification remain advisable before production rollout.
