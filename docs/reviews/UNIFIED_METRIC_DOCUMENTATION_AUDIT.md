# Unified metric parameters — documentation and integration audit

**Branch:** `feature/unified-metric-parameters`
**Audit HEAD:** `74fdb451de7fa3d6ad94b73afaf804d28eab1708` (`74fdb45`)
**Audit date:** 2026-11-03
**Mode:** Historical read-only inspection of the development branch at `74fdb45`

**RTX 4070 status (reported by maintainers, not re-run in this audit environment):** CLI conformance 42/42; comprehensive numerical 101 PASS / 0 FAIL / 0 SKIP / 7 NA; numerical regression, internal consistency, and candidate reproducibility PASS.

**Historical status:** This report records issues found before the follow-up
documentation and orchestrator fixes. Its original NO-GO findings are not the
current status of `main`. For current instructions, see
[Validation](../VALIDATION.md) and the repository [README](../../README.md).

---

## Executive summary

The **implementation and harness** for unified CLI parameters, numerical conformance, baselines, and validation levels appear aligned with RTX results on `74fdb45`, including the approved 95-case comprehensive baseline in `baselines/numerical_conformance/comprehensive.json`.

**User-facing documentation integration is not merge-ready.** Critical gaps include: (1) the root documentation tree never points to the new validation workflow; (2) Phase 1 CLI conformance is **documented and wired to the wrong script** (`test_metric_parameters.py` vs the 42-case `metric_cli_conformance` entry point); (3) `reports/NUMERICAL_VALIDATION_PREFLIGHT.md` is stale relative to approved baseline and the five-level JSON schema; (4) validation-level terminology is inconsistent across files (three levels vs five, internal checks conflated with independent reference parity).

**Recommendation:** Merge `feature/unified-metric-parameters` to `main` only after a follow-up documentation pass addressing Critical and Major findings below. RTX execution evidence supports **functional** readiness; **documentation and orchestration** need correction.

---

## Documentation inventory

| Path | Role | Linked from root `README.md`? |
|------|------|-------------------------------|
| `README.md` | Product overview, quick start, doc index | — |
| `docs/GETTING_STARTED.md` | Install / builds | Yes |
| `docs/INSTALL.md` | Manual install | Yes |
| `docs/BUILD_MAKE.md` | Make targets | Yes |
| `docs/INSTALL_VALIDATION.md` | Clean-room install gate | Yes |
| `docs/CLI_USAGE.md` | CLI examples, metric groups, unified HDR/CVVDP options | Yes |
| `docs/METRICS_AND_FORMATS.md` | Metrics, formats, CPU/GPU | Yes |
| `docs/COLORVIDEOVDP.md` | CVVDP backend | Yes |
| `docs/COMPRESSEDVQA_HDR.md` | CVQA-HDR-NR | Yes |
| `docs/LEARNED_*.md` | Learned env, config, devices | Partial |
| `docs/PU21.md`, `docs/QCS.md` | HDR / QCS | Partial |
| `docs/RELEASE_VALIDATION.md` | `make release-check*` | No (not in index) |
| `scripts/numerical_conformance/README.md` | Numerical harness (authoritative for validation) | **No** |
| `scripts/numerical_conformance/LEARNED_DEPS.md` | NA/SKIP for learned metrics in suite | No |
| `reports/NUMERICAL_VALIDATION_PREFLIGHT.md` | RTX preflight notes | No |
| `baselines/numerical_conformance/README.json` | Baseline folder metadata | No |
| `docs/reviews/UNIFIED_METRIC_DOCUMENTATION_AUDIT.md` | This report | No |

Additional markdown under `reports/` (VMAF, GPU, HDR, release readiness) predates this workstream and was not exhaustively re-audited; they remain relevant for release qualification but do not describe the new numerical suite.

---

## Verified findings (evidence-backed)

Findings below were checked against source on `74fdb45`. Line numbers refer to that commit.

### Critical

#### C1 — Phase 1 “42 CLI cases” points at the wrong script

**Evidence**

- `scripts/numerical_conformance/README.md` lines 70–71 instruct Phase 1 as `python3 scripts/test_metric_parameters.py` with comment “42 CLI cases”.
- `reports/NUMERICAL_VALIDATION_PREFLIGHT.md` line 7 labels CLI conformance as `test_metric_parameters.py` with count 42.
- `scripts/run_pixvqa_validation.py` lines 24–28 invoke `test_metric_parameters.py` for phase 1.

**Actual implementation**

- 42 declarative cases: `scripts/metric_cli_cases.py` → `conformance_matrix()` (verified: `len(conformance_matrix()) == 42` on audit machine).
- Runner: `scripts/metric_cli_conformance.py`; entry point: `scripts/test_metric_cli_conformance.py` (see `scripts/test_metric_cli_conformance.py` lines 1–14).
- `scripts/test_metric_parameters.py` lines 198–202: `main()` runs only `test_resolver_unit`, `test_cli_inference`, and `test_learned_native_guard` (~10 focused tests, many SKIP without rebuilt `swdec` binary)—**not** the 42-case matrix.

**Impact:** Developers and `run_pixvqa_validation.py` can believe Phase 1 passed when only resolver/inference tests ran. RTX “42/42” likely used `test_metric_cli_conformance.py` or `metric_cli_conformance.py` directly, not what the written workflow describes.

**Proposed correction (recommendation):** Document and wire Phase 1 to `python3 scripts/test_metric_cli_conformance.py` (or `python3 scripts/metric_cli_conformance.py`); clarify `test_metric_parameters.py` as unified-parameter **unit/integration** tests; update `run_pixvqa_validation.py` accordingly.

---

#### C2 — Root documentation omits the numerical validation workstream

**Evidence**

- `README.md` lines 110–118 list getting started, CLI, metrics, learned guides—**no** link to `scripts/numerical_conformance/README.md`, `scripts/run_pixvqa_validation.py`, baselines, or CLI conformance reports.
- `docs/GETTING_STARTED.md` ends at smoke QA via `scripts/run_qa.py` (lines 33–35)—no Phase 1/2 validation.
- Grep over `docs/**` for `numerical`, `test_metric_parameters`, `run_pixvqa_validation`: **no matches**.

**Impact:** First-time contributors will not discover mandatory RTX/CI-style validation without reading the feature branch diff or `scripts/`.

**Proposed correction:** Add a “Validation & regression” section to the root index linking numerical README, preflight (or successor), baseline approval policy, and optional BerlinSeqs.

---

#### C3 — `NUMERICAL_VALIDATION_PREFLIGHT.md` contradicts current baseline and verdicts

**Evidence**

- Title and framing: `reports/NUMERICAL_VALIDATION_PREFLIGHT.md` line 1 references `feature/unified-metric-parameters` as if still pre-merge campaign material.
- Lines 25–27: numerical regression **NOT_EVALUATED**, baseline **not approved**.
- Lines 71–78: GO/NO-GO table “Approved baseline on disk: **NO-GO**”.
- `baselines/numerical_conformance/comprehensive.json` lines 1–3: `"approved": true` at `74fdb45`.

**Impact:** Misleading for merge decisions and for anyone running post-approval regression.

**Proposed correction:** Refresh or archive preflight doc; state approved baseline path, expected `numerical_regression: PASS`, and RTX command block aligned with C1 fix.

---

### Major

#### M1 — Validation level count and naming inconsistent

**Evidence**

- JSON report defines **five** keys under `validation_levels`: `execution_conformance`, `numerical_regression`, `internal_numerical_consistency`, `independent_reference_parity`, `candidate_reproducibility` (`scripts/numerical_conformance/runner.py` lines 499–516).
- `scripts/numerical_conformance/README.md` lines 110–119 table lists **four** rows and line 119 says “JSON reports expose all four under `validation_levels`.”
- `reports/NUMERICAL_VALIDATION_PREFLIGHT.md` lines 12–34 describes **three** levels and merges internal cross-path checks with Berlin under “Independent reference parity.”

**Implementation truth**

- Internal consistency: same-build cross-path checks (`validation_levels.py`, synthetic IDs documented README lines 121–122).
- Independent reference parity: optional Berlin CVVDP JOD vs pinned constants only (`berlin.py`, README lines 123–125); `NOT_EVALUATED` when env unset (`validation_levels.py` `summarize_independent_reference_parity`).

**Proposed correction:** Single glossary table (five levels + optional Berlin scope) reused in numerical README and any preflight/release doc.

---

#### M2 — Seven N/A cases documented imprecisely in preflight

**Evidence**

- RTX: 7 NA (maintainer report). Registry produces exactly seven comprehensive case IDs (`scripts/numerical_conformance/registry.py` `not_applicable_reason`):
  1. `sdr.raw_raw.cvvdp`
  2. `sdr.raw_comp.cvvdp`
  3. `sdr.raw_raw.cvqa-hdr-nr`
  4. `pq.raw_raw.cvqa-hdr-nr`
  5. `sdr.raw_raw.group_sdr`
  6. `sdr.raw_comp.group_sdr` — **Note:** audit script listed `sdr.raw_comp.group_sdr`; verify: python listed `sdr.raw_comp.group_sdr`? Actually list was:
     - pq.raw_raw.cvqa-hdr-nr
     - pq.raw_raw.group_pq
     - sdr.raw_comp.cvvdp
     - sdr.raw_comp.group_sdr
     - sdr.raw_raw.cvqa-hdr-nr
     - sdr.raw_raw.cvvdp
     - sdr.raw_raw.group_sdr
- `LEARNED_DEPS.md` lines 18–20 explains SDR raw CVVDP NA but does not enumerate all seven IDs.
- Preflight section “N/A vs SKIP (seven comprehensive skips)” (`reports/NUMERICAL_VALIDATION_PREFLIGHT.md` line 36) uses **“skips”** for NA; line 8 table puts NA count in wrong column (`7 | —` under SKIP column header).

**Proposed correction:** Publish explicit table of seven case IDs with one-line rationale each; fix preflight table columns.

---

#### M3 — Baseline folder metadata stale

**Evidence**

- `baselines/numerical_conformance/README.json` lines 3–4: `"approved": false` with note to `--approve-baseline`.
- `comprehensive.json` approved true at same directory.

**Proposed correction:** Update README.json to reflect approved comprehensive baseline and pointer to numerical README (recommendation only—do not change in this audit commit per user boundary except audit file).

---

#### M4 — `RELEASE_VALIDATION.md` silent on new harness

**Evidence**

- `docs/RELEASE_VALIDATION.md` lists eleven `make release-check` stages (lines 26–38); no mention of `run_pixvqa_validation.py`, numerical conformance, or CLI conformance matrix added on this branch (`git diff origin/main...HEAD` adds scripts but not release doc updates).

**Proposed correction:** Clarify relationship: release-check vs optional/full GPU numerical campaign.

---

#### M5 — Executable path conventions split without cross-link

**Evidence**

- Product docs use `./bin/pixvqa` (`README.md` lines 18–24, `GETTING_STARTED.md` line 22).
- Numerical harness defaults `PIXVQA_BIN=build/full/pixvqa` (`scripts/numerical_conformance/README.md` lines 66–67; `scripts/numerical_conformance/config.py` `FULL_BIN` default).

Both are valid after `make full` vs installer layout; not explained in one place.

**Proposed correction:** One paragraph in numerical README: set `PIXVQA_BIN` to installed `bin/pixvqa` or `build/full/pixvqa`.

---

### Minor

#### m1 — Hard-coded personal paths in numerical README

`scripts/numerical_conformance/README.md` line 54: `export PIXVQA_LEARNED_PYTHON=/home/sergio/...` — replace with placeholder in docs (recommendation).

#### m2 — `test_metric_parameters.py` module docstring vs usage

Line 2: “Tests for unified CVVDP / HDR transfer parameter resolution” — accurate for file content, confusing next to “42 CLI cases” label elsewhere.

#### m3 — Numerical README says “all four” validation levels

Line 119; should be five (see M1).

#### m4 — Approved baseline provenance commit vs branch HEAD

`comprehensive.json` meta `pixvqa_commit` is `b0906e5…` while branch HEAD is `74fdb45` (baseline file commit). Candidate reproducibility policy documents infrastructure allowlist (`candidate_commit_compatibility.py`); user-facing docs do not explain that **regression baseline values** were captured at an earlier metric commit while **harness** may be newer—only relevant when comparing candidates, not approved regression (recommendation: one sentence in numerical README).

#### m5 — BerlinSeqs optional env vars not mentioned in root docs

Only in numerical README lines 123–125; acceptable if index links to numerical doc.

---

## Technical correctness spot-check (CLI / metrics)

| Topic | Doc source | Implementation | Verdict |
|--------|------------|----------------|---------|
| Metric groups `sdr` / `pq` / `hlg` | `docs/CLI_USAGE.md` lines 267–279 | `scripts/numerical_conformance/metric_caps.py` `GROUP_METRICS`; `src/main.cpp` group handling | **Consistent** |
| `--device` / `--metric-device` alias | `CLI_USAGE.md` lines 301–302 | `src/main.cpp` lines 2948, help text ~197–199 | **Consistent** |
| `--hdr-transfer`, `--tf`, per-side transfer | `CLI_USAGE.md` lines 320–321 | `src/main.cpp` `metric_resolve` / conflict checks; exercised in `test_metric_parameters.py` | **Consistent** (when swdec built) |
| CVVDP raw YUV420p10 PQ/HLG only | `CLI_USAGE.md`, `COLORVIDEOVDP.md` | `src/main.cpp` ~833–839; `registry.py` `cvvdp_skip_reason` SDR raw → NA | **Consistent** |
| CVQA compressed-only, no `--frames` | `CLI_USAGE.md` lines 104–106 | `metric_caps.py` `cvqa-hdr-nr` `honors_frames: False` | **Consistent** |
| HDR not inferred from bit depth alone | `METRICS_AND_FORMATS.md` lines 103–104 | `src/main.cpp` messaging ~2346 | **Consistent** |
| 95 baseline vs 6 internal synthetics | numerical README 121–122 | `validation_levels.py` `INTERNAL_SYNTHETIC_CASE_SUFFIXES`; `baselines.py` `requires_approved_baseline_regression` | **Consistent** |
| Default numerical JSON path | — | `scripts/test_metric_numerical_conformance.py` line 20 → `reports/numerical_conformance.json` | Documented in preflight line 63; not in root README |

**CLI help:** No `pixvqa` binary present in audit workspace; help strings verified via `src/main.cpp` only.

---

## Readability and recommended documentation hierarchy

**Verified gap:** Validation content is siloed under `scripts/numerical_conformance/` with no root navigation.

**Recommended structure (recommendation, not implemented):**

1. **README.md** — Add “Quality validation” bullet linking Phase 1 CLI matrix + Phase 2 numerical suite.
2. **docs/GETTING_STARTED.md** — After smoke QA, optional “Full metric validation” pointer.
3. **docs/VALIDATION.md** (new, optional) — Thin hub: prerequisites (`numpy`, `make full`, learned TOML), env vars (`PIXVQA_BIN`, `PIXVQA_LEARNED_PYTHON`, strict flags), two phases, five validation levels, baseline vs mathematical correctness, Berlin optional.
4. **scripts/numerical_conformance/README.md** — Deep reference (fixtures, tolerances, approval, `--compare-candidate`, allowlist policy).
5. **reports/** — Either update preflight or move to `docs/VALIDATION_RTX.md` with date and commit pin.

---

## Integration readiness (Git diff vs `main`)

**Verified on diff stat (`67 files`, +8805 lines):** New validation scripts, baselines, `main.cpp` / `metric_resolve.cpp`, learned adapter changes, and `docs/CLI_USAGE.md` updates are present.

**Documentation-related omissions in diff:**

- No updates to root `README.md` for validation.
- `RELEASE_VALIDATION.md`, `INSTALL_VALIDATION.md` unchanged.
- Preflight report added but now stale relative to `74fdb45`.
- Orchestration bug: `run_pixvqa_validation.py` phase 1 script name (C1).

**Functional readiness (from maintainer RTX + local unit tests):**

| Check | Audit environment result |
|-------|---------------------------|
| `scripts/test_numerical_baselines.py` | 19/19 OK |
| `scripts/test_numerical_validation_levels.py` | 6/6 OK |
| `scripts/test_metric_parameters.py` | 1 OK, 2 SKIP (no swdec binary) |
| Comprehensive baseline case count | 95 when `comprehensive.json` approved (test skips if missing) |

---

## Merge-readiness assessment

| Dimension | Assessment |
|-----------|------------|
| RTX execution / regression (reported) | **Ready** — PASS on stated matrix with approved baseline |
| Harness self-tests (partial local) | **Ready** — numerical unit tests pass |
| CLI conformance orchestration vs docs | **Not ready** — C1 |
| User documentation discoverability | **Not ready** — C2 |
| Validation campaign docs accuracy | **Not ready** — C3, M1, M2 |
| Metric / CLI user guides (`CLI_USAGE`, `METRICS_AND_FORMATS`) | **Largely ready** — unified parameters documented; spot-check pass |

**Overall documentation merge recommendation:** **Conditional NO-GO** until Critical items C1–C3 are addressed. Code merge may be acceptable on a separate policy decision, but **documentation and `run_pixvqa_validation.py` should not be considered complete.**

---

## Distinction: verified findings vs recommendations

| Verified findings | Recommendations (not executed in this audit) |
|-------------------|---------------------------------------------|
| C1–C3, M1–M5, m1–m5 with file/line evidence | Edit docs, fix `run_pixvqa_validation.py`, refresh preflight, update README.json |
| Seven NA case IDs from `registry.py` | Publish NA table in numerical README |
| Five `validation_levels` keys in `runner.py` | Unify glossary across docs |
| RTX results as reported by user | Re-run RTX after doc fixes using corrected Phase 1 command |

---

## Suggested RTX verification commands (post-doc-fix)

Use after correcting Phase 1 entry point (recommended):

```bash
git checkout feature/unified-metric-parameters
git pull
# HEAD at or after documentation fix commits

export PIXVQA_BIN=build/full/pixvqa
export PIXVQA_CLI_CONFORMANCE_STRICT=1
export PIXVQA_NUMERICAL_STRICT=1
export PIXVQA_LEARNED_PYTHON=/path/to/learned-env/bin/python

python3 scripts/test_metric_cli_conformance.py
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict \
  --json reports/numerical_conformance_comprehensive.json
python3 scripts/run_pixvqa_validation.py --profile comprehensive --strict
```

Expect: CLI matrix 42/42 PASS; comprehensive 101 PASS / 7 NA; `numerical_regression: PASS`; five validation levels as in JSON schema.

---

*End of audit report.*
