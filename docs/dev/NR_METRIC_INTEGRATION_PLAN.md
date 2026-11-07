# NR metric integration plan (development)

Status: proposed implementation contracts; **standalone SimpleVQA upstream reference runner**
(`python -m pixvqa_simplevqa`, [`SIMPLEVQA_REFERENCE.md`](SIMPLEVQA_REFERENCE.md)) exists for external parity runs.
**ChipQA** is not integrated. **`pixvqa --metrics simplevqa`** (SDR-only NR, Python backend) is implemented in `src/main.cpp` with tests in `scripts/test_learned_metrics_cli.py`.

## Existing integration points

- The public CLI and metric capability table are in `src/main.cpp` (`kMetricCaps`, group selection, FR vs NR validation).
- Existing learned backends are `python/pixvqa_cvqahdr` and `python/pixvqa_cvvdp`.
- CLI and mixed-backend tests are in `scripts/test_learned_metrics_cli.py`, `scripts/test_metric_cli_conformance.py`, and `scripts/run_qa.py`.
- The synthetic UHD numerical suite is in `scripts/numerical_conformance`; generated data are cached and not committed.
- The tracked schema is `config/nr_validation_corpus.example.json` (portable placeholders). Keep real paths in Git-ignored `config/nr_validation_corpus.local.json`; validate read-only with `python3 scripts/check_nr_validation_corpus.py --check-files`. Without a local file, pass `--manifest` explicitly or copy the example and fill its paths.

## Proposed user contract

- `simplevqa`: NR, SDR only, initially Python/PyTorch reference implementation.
- `chipqa`: NR, SDR -> original ChipQA + its scaler/SVR; PQ -> HDR-ChipQA + its scaler/SVR; HLG -> unsupported.
- `cvqa-hdr-nr`: retain existing SDR/PQ support and behavior.
- Preserve `--metrics sdr|pq|hlg` grouping; do not make implicit HLG-to-PQ conversions.
- Preserve the current mixed FR+NR command shape: `-r REF -t DIST`, while NR backends consume only DIST.
- Record the resolved transfer, source revision, preprocessing variant, model/checkpoint identity, device, sampling plan, and number of evaluated samples.
- A pooled prediction must not be labeled per-frame or passed to per-frame QCS. Windowed/temporal outputs require explicit documented semantics.

## Model reference before optimization

1. Verify upstream license/weights and pin revision and artifact hashes.
2. Establish upstream golden predictions on fixed SDR clips; for ChipQA also pin feature vectors and SVR input/output.
3. Test raw YUV -> deterministic, metadata-tagged compressed derivatives with exact frame windows; do not encode during score-parity runs.
4. Add a Python reference backend, without changing native metric calculations.
5. Optimize SimpleVQA preprocessing, memory and neural inference only after golden parity (ONNX Runtime/TensorRT if feasible).
6. Port shared ChipQA feature extraction to C++/SIMD after SDR/HDR reference parity; GPU kernels only if profiling warrants.

## Test gates

- `smoke`: no model dependencies; missing models and unsupported transfers are explicit failures before scoring.
- `reference-parity`: same frames/crops/resize/transfer as upstream; compare scores and intermediate features.
- `compatibility`: raw/compressed input, fps, frame caps, wrong transfer, short/truncated sequences, failed setup, FR+NR mixed runs.
- `performance`: 1080p and 2160p, 8/10-bit, single and concurrent jobs, peak RSS and VRAM; benchmark outside public CI.
- `perceptual`: independent MOS/DMOS evaluation, title-disjoint dataset splits, correlations and confidence intervals.
- Do not claim integration success for mocked backend scores; do not interpret synthetic fixtures as perceptual validation.

## Known unrelated issue

Superposition 1080p/60, 3600 frames: native PixVQA returned `read error: Cannot allocate memory` and `[ NATIVE status:failed ]` while CVQA-HDR-NR completed. This is an open reliability issue (no root cause confirmed); do not use that job as an NR parity oracle or change the active benchmarking computer.
