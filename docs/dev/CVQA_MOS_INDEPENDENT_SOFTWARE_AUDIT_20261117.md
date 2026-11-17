# Independent Software Review — CVQA-to-MOS v7 and CVQA CLI Regression

**Review date:** 17 November 2026  
**Repository:** `pixop/pixvqa`  
**Branch:** `feature/cvqa-hdr-nr-mos-calibration`  
**Baseline revision examined:** `8fa2790698eedcb4f8d56127895b0235cf5c1a84`  
**Scope:** transfer-separated calibration software, coefficient provenance and integrity, fail-closed routing, regression tests, and previously failing CompressedVQA CLI sampling-preflight suite.

## Executive assessment

The selected SDR and HDR PQ calibration functions reproduce the frozen v7 research coefficients. Their mathematical form, per-transfer normalization, output clipping, routing from validated ffprobe metadata, and retained canonical CVQA score were consistent with the scientific deliverable. Independent native-case and four-tuple numerical oracles already cover the principal prediction paths.

The previously reported CVQA CLI failure was **not an error in the production sampling algorithm**. The test expected obsolete diagnostic wording; after correcting that assertion, the same test exposed two additional stale expectations concerning ignored compressed-video options and missing Python interpreter execution. A broader learned-metric CLI suite also contained two native-PQ cases that incorrectly included options belonging to unselected learned metrics. These test defects have been corrected without changing the production decoder or sampling algorithm.

Two further reliability concerns were found in the calibration module itself: public metadata getters exposed mutable internal data, and direct prediction bypassed coefficient-fingerprint enforcement. Both are corrected and covered by new regression tests.

**Software disposition:** the focused calibration and CLI suites pass after these corrections. This does **not** constitute a production or scientific subjective-calibration approval. An independent prospective native-MOS experiment remains necessary, particularly for high-MOS HDR.

## Reviewed contract

The new models are fitted with transfer-isolated development data: **SDR U-v2 on 200 SDR-only observations** and **HDR PQ Original U on 192 PQ-only observations**. Both retain the original M1 sigmoid plus M7 metadata adjustment. The selected SDR route has a +0.5107732905216783 native-MOS post-offset; the PQ route has no post-offset. The runtime uses independent mean, standard deviation and training-score support for each transfer. Canonical CVQA is not input-clamped; output estimated MOS is confined to [1,5].

Reference values at canonical CompressedVQA 10.0:

| Scored height | SDR predicted MOS | HDR PQ predicted MOS |
|---|---:|---:|
| 1080p | 4.6328337676073605 | 3.9182233198793637 |
| 2160p | 4.71826893567017 | 4.147123703872427 |

The model output remains an **estimated perceptual quality rating**, not a MOS from viewers. The canonical CompressedVQA value is not modified by this calibration.

## Findings and actions

### F1 — Stale sampling-preflight CLI assertion (resolved; test defect)

`scripts/test_cvqahdr_nr_cli.py` expected a hyphenated `sampling-buffer` fragment at 100 frames/60 fps. The adapter produces `NR sampling buffer`, rejecting the clip correctly because it has fewer than **two** complete 60-frame intervals. We updated the expected phrase and added cases for 50, 60, 100, 119 and the **minimum valid 120 frames**. There was no justification for modifying the actual sampling policy.

### F2 — Other obsolete focused CLI assertions (resolved; test defects)

The focused CLI test expected the raw geometry option `-s` to be rejected as incompatible with compressed NR inputs. The current C++ behavior intentionally **ignores** unused raw-video geometry flags with an informational message. The test now verifies this documented behavior while preserving the failure on missing model assets.

The same test expected a previous missing-interpreter diagnostic. The actual backend reports **exit 127** for a nonexistent Python executable. The assertion now verifies the current process-execution failure code. Actual successful mocked execution, temporary output retention and cleanup tests were also run.

### F3 — Native PQ cases passed unrelated learned-metric options (resolved; test defect)

The wider `scripts/test_learned_metrics_cli.py` suite failed two native-PQ display/verbosity cases (`pq-v1` and `pq-v0`). Both appended CVVDP and CompressedVQA-only flags even though `--metrics pq` selects neither learned metric. The production C++ argument parser appropriately rejects this combination. We removed the extraneous learned-only flags from those two native cases. The broader suite then passed.

### F4 — Writable references exposed via public calibration metadata (resolved; moderate)

`policy_spec()` and `calibrators_for_policy()` previously returned their internal dictionaries directly. A caller changing the returned nested structures could silently corrupt future calibration provenance or routing metadata. The getters now return **deep copies**. A new test mutates returned nested fields and confirms the internal model, calibration version, selected route and digest remain unchanged.

### F5 — Standalone prediction bypassed coefficient integrity (resolved; moderate)

`apply_mos_calibration()` checked selected and embedded coefficient hashes, but `predict_mos_detailed()` (and thus `predict_mos()`) did not. In an isolated test, an in-memory change to the SDR sigmoid slope caused direct prediction to shift from approximately 3.8903 to 4.2560 at CVQA 9 despite the mismatch being detectable by the digest function. We added integrity checks before the direct prediction path can publish a value. A regression test now expects a `ValueError` on tampered coefficients and verifies the normal prediction remains unchanged after restoration.

These checks detect accidental or unexpected in-process modifications; they are not a security guarantee against an adversary who can modify both source code and stored digests.

## Verification evidence

| Check | Result |
|---|---|
| Selected SDR/PQ numerical parity with frozen four-tuple M1+M7 oracle | Passed |
| 4,800 randomized route predictions against separate frozen four-tuple evaluator | Passed; maximum absolute difference 1.78×10⁻¹⁵ MOS |
| Extreme finite-score monotonicity and boundedness | Passed |
| Existing 68-clip native AVT golden fixture comparison | Passed |
| SDR54 / HDR35 selected-core records and SDR VD6 transfer records | Passed |
| Correct transfer-specific score support and non-clamped inputs | Passed |
| Monotonicity, bounded output and unsupported-metadata rejection | Passed |
| Source SHA-256 and embedded-model fingerprint checks | Passed |
| Public metadata mutation resistance | Passed |
| Direct API fails closed on altered coefficient values | Passed |
| MOS regression suite (normal Python) | **48 tests passed** |
| MOS regression suite (optimized Python) | **48 tests passed** |
| Focused CVQA-HDR-NR CLI dispatch and sampling test | **All 16 named scenarios passed** |
| Broader learned-metrics CLI suite | **133 checks passed** |
| Production FFmpeg-only sampling/decoder contract | Passed, 19 sample midpoints |

The 133-case suite includes mocked learned-metric behavior, so its result must not be represented as full real-checkpoint inference or production video-quality validation.

## Remaining limitations and recommendations

1. **Subjective calibration:** HDR35 retrospective MAE remains 0.6042, worse than the historical mixed-training curve's 0.5552. Native-rated HDR with MOS ≥4.5 is absent. Neither correction above changes these scientific limitations.
2. **Metadata fidelity:** transfer selection relies on the encoded stream's ffprobe transfer label. Unknown, HLG and unsupported labels are rejected, but correct upstream metadata tagging is still an input requirement.
3. **Runtime coverage:** the successful suites establish the tested paths on the available WSL workstation, not comprehensive real-checkpoint GPU inference, all supported codecs or every device/OS configuration.
4. **Release management:** the current Git feature branch can be reviewed or tested remotely; do not merge into `main` or label the calibration subjectively production-certified without independent approval.

## File-level changes

- `python/pixvqa_cvqahdr/mos_calibration.py` — return defensive metadata snapshots and enforce both coefficient digests in standalone prediction.
- `scripts/test_cvqahdr_mos_calibration.py` — regression tests for metadata mutation and direct API integrity.
- `scripts/test_cvqahdr_nr_cli.py` — correct current CLI diagnostics and strengthen sampling-boundary checks.
- `scripts/test_learned_metrics_cli.py` — remove invalid learned-only flags from native PQ tests.
- This report — independent audit scope, findings, test evidence and residual risks.

The selected M1/M7 coefficients, normalization data, post-offsets, upstream pins, canonical CVQA algorithm, FFmpeg sampling algorithm and C++ runtime are **unchanged** by these fixes.
