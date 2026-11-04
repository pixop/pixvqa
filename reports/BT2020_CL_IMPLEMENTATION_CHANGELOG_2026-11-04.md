# PixVQA 0.3.0-dev — BT.2020 CL/PU21 implementation and release notes

**Date:** 2026-11-04
**Scope:** source-only changes on top of the supplied `pixvqa_source(2).tar`.
**Version:** project version is intentionally unchanged at `0.3.0-dev`.

## Source edits

- `src/hdr.cpp`, `include/pixvqa/hdr.hpp`: preserve the existing BT.2020
  CL piecewise inverse introduced in Patch 3A; make the scalar display-luma
  helper use the CL RGB reconstruction instead of a non-constant-luminance
  inverse; expose optional out-of-gamut reporting. CL component G is derived
  from linearized Yc, R and B. Bound reconstructed G to the display or HLG
  scene domain and flag substantive out-of-gamut values rather than silently
  hiding them. Ignore tiny round-off below a documented tolerance. All NCL
  code paths retain their prior inverse formula.
- `src/cuda_pu21.cu`: mirror the CL clipping/checking policy in the device
  inverse and propagate failed display transforms into the kernel's `clamped`
  indicator. The `constant_luminance` packing is retained. **Not built or
  executed with NVCC in the local environment.**
- `src/pu21.cpp`: propagate the inverse gamut flag to the existing PU21
  clipping warning. The 0.005..10,000 cd/m2 PU21 domain remains unchanged.
- `src/main.cpp`: allow BT.2020 CL/CL CPU/GPU PU21 paths when both sides have
  matching PQ or HLG transfer, BT.2020 primaries, and matching explicit range.
  CL/NCL sample comparisons, all CL VMAF paths and unvalidated CL learned
  metrics remain rejected. Fix diagnostics that printed literal `\\n`; emit
  display-transform notes only for selected display-domain metrics.
- `python/pixvqa_cvvdp/adapter.py`: reject tagged BT.2020 CL; require the
  validated NCL and limited-range metadata for PQ and HLG interpretation.
- `python/pixvqa_cvqahdr/adapter.py`: run an ffprobe color-matrix preflight
  before upstream NR inference, rejecting CL and failing closed when the
  matrix probe itself fails. No upstream model code is changed.

## Regression and integration harness

- `src/selftest.cpp`: independent forward-encoded RGB->BT.2020 CL numerical
  vectors for both chroma sign branches, primary and neutral colors,
  PQ/HLG inverse and display-luma, out-of-gamut indications and full/limited
  code normalization at 8/10/12/16 bits.
- `scripts/test_bt2020cl_numerical.py`: genuine CL-encoded 32x32 YUV420
  vectors with an independent analytical PU21 RGB PSNR oracle, PQ/HLG,
  8/10-bit full/limited, negative/positive chroma. Includes PU21 SSIM and
  MS-SSIM CLI regression/identity checks, CL/NCL rejection, CL/CL acceptance,
  and explicit gamut-clipping notification. Optional `--gpu-exe` makes GPU
  parity mandatory for the eligible synthetic vectors.
- `scripts/test_bt2020cl_learned_guards.py`: mocked Python adapter metadata
  preflight tests. `scripts/test_cvvdp_edge_policies.py` extended for CL
  and PQ range; stale help-line counts in `scripts/test_pu21_numeric.py`
  adjusted for the added help text.
- `scripts/run_real_video_integration.py`: explicit, ordered **146-case**
  phases A–G manifest, multi-metric output parsing, single-worker execution,
  file/command-stamped resumability, GPU parity phase, atomic JSON reports,
  exact frame-count checks, and truthful PASS/FAIL/BLOCKED/NOT_RUN reporting.
  Filtered runs use separate report filenames and cannot overwrite the
  `full_146_results.json` full-campaign report. Completion for a filtered
  selection is distinguished from completion of all 146 cases.
- `scripts/test_integration_campaign_manifest.py`: offline fake-executable
  tests of 146 unique cases, original Phase A order, required GPU selection,
  result parsing, resume invalidation, and missing-fixture blocking.
- `reports/real_video_146_manifest.json`: machine-readable frozen list of
  all proposed test IDs and requested metrics. `docs/BT2020_CL_VALIDATION.md`
  provides formulas, compatibility policy, commands, provenance, and
  workstation execution instructions. `docs/PU21.md` documents CL usage.

## Executed here — PASS

| Operation | Observed result | Limits |
| --- | --- | --- |
| `make -j2 cpu` | Exit 0 | CPU-only build; no FFmpeg/CUDA/libvmaf integration |
| `make -j2 test-cpu` | Exit 0, `All selftests passed` | Includes new analytical CL reconstruction cases |
| `scripts/test_bt2020cl_numerical.py` | **31/31** analytical + CLI checks PASS | CPU binary; GPU option not supplied |
| `scripts/test_pu21_numeric.py` | Exit 0, all original numerical tests PASS | Uses CPU binary |
| `scripts/test_cvvdp_edge_policies.py` | All 4 cases PASS | Adapter unit-level policy tests |
| `scripts/test_bt2020cl_learned_guards.py` | Exit 0 | Mocked ffprobe / adapter preflights, not inference |
| `scripts/test_cvqahdr_nr_cli.py` | All 16 CLI dispatch/sampling test cases PASS | Mocked upstream model; no real checkpoint |
| `scripts/test_integration_campaign_manifest.py` | Exit 0 | Mock executable; tests runner not real video metrics |
| Python `compileall` | Exit 0 | Syntax compilation only |
| `--list` integration | **146 uniquely identified cases** | Manifest generation, no real cases run |

## Not executed here — workstation required

- Full FFmpeg 8.1.3 / libvmaf / GCC 13 / CUDA 12.8 compilation via `make -j2 full`.
- NVCC compilation of the edited `.cu` file and numeric CPU-vs-CUDA parity,
  including `scripts/test_pu21_cuda.py` and `--gpu-exe` analytical vectors.
- Actual FFmpeg HEVC/AV1/VVC decoding and HDR metadata roundtrip tests.
- Genuine 4K video fixtures, Phase A rerun on changed code, and the 146-case
  live integration execution. The original 48/48 Phase A and 42/42 CLI
  conformance counts were **historical handoff results** from an earlier
  workstation version, not new passes in this work.
- Heavy CVVDP and CVQA-HDR-NR model execution.

## Case-count provenance and acceptance status

The handoff supplies counts **A=48, B=12, C=24, D=18, E=18, F=12** (132), but
contains actual case definitions only for A. The generated B–F definitions
are **new proposals matching those counts**, not recovered historical test
specifications. The extra **G=14** are *explicit new proposals* to reconcile
132 with 146, testing PQ=1000-nit clipping and HLG=4000-nit display peaks.
The full 146-case campaign is **prepared but NOT RUN**; no integration PASS
counts have been fabricated. Review B–G against any original specification
before accepting the coverage matrix as the project's intended campaign.

## Limits and residual risks

1. CUDA kernel compiled previously (Patch 3A) on the user's WSL, but this
   iteration's CUDA edits are **source-reviewed, not NVCC-compiled**.
2. Out-of-gamut CL components are bounded and warned about; per-pixel counts,
   color-science policy for out-of-gamut pixels, and CPU/GPU flag exactness
   are not yet validated on the GPU.
3. CL/NCL mixed-input PU21 could in principle use a separate validated
   display-domain conversion, but remains conservatively rejected.
4. A `bt2020c` metadata tag is not proof that the underlying samples were
   CL encoded. The new mathematical vectors are genuinely CL encoded.
5. Real-video campaign is single-worker to protect WSL memory. A 3-frame
   run is an integration smoke test, not a statistical video-quality study.

See `docs/BT2020_CL_VALIDATION.md` for the workstation command sequence.
