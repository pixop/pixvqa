# CVQA-to-MOS — historical reviewed/legacy policies (superseded)

> **Historical record:** Superseded for runtime on 27 November 2026 by [canonical B08](CVQA_MOS_A2_TV123_INTEGRATION.md). Earlier commands, selectors and module paths below describe the archived implementation, not current PixVQA.

**Superseded 2026-11-14.** PixVQA runtime now uses only the scientifically selected formula (`selected_s1_sdr_p1_hdr_c0`). The prior triple-policy model (`legacy_target_lambda1_both`, `reviewed_s1_hdr0_sdr1`, env-based selection) has been removed from active code paths.

## Obsolete environment variable

`PIXVQA_CVQA_MOS_POLICY` is retained in documentation only. Values other than unset or `selected_s1_sdr_p1_hdr_c0` fail closed with `obsolete_policy_env:…` so legacy formulas cannot be activated.

## Current documentation

See [LEARNED_METRICS.md](../LEARNED_METRICS.md) and [CVQA_MOS_FINAL_SELECTED_IMPLEMENTATION.md](CVQA_MOS_FINAL_SELECTED_IMPLEMENTATION.md).

## Archived fixtures

`tests/fixtures/mos_calibration/avt_reviewed_s1_hdr0_sdr1.json` and `avt_target_lambda1_selected_route.json` remain in the tree for historical reference but are no longer exercised by CI regression tests.