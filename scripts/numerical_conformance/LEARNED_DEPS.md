## CVQA-HDR-NR assets (optional learned metric)

Configure in `config/learned_metrics.local.toml`:

```toml
[cvqa-hdr-nr]
repository = "/path/to/CompressedVQA-HDR"
checkpoint = "/path/to/NR_HDR_VQA.pth"
profile = "/path/to/NR_HDR_VQA.npy"
```

Or environment overrides: `PIXVQA_CVQAHDR_PYTHON`, plus repo/checkpoint/profile via TOML.

When checkpoint paths are wrong or files are absent, comprehensive cases for `cvqa-hdr-nr` and SDR/PQ groups are **SKIP** before pixvqa runs (preflight reads this TOML). Strict mode still exits nonzero until assets are installed.

CVQA-only invocations use `-t distorted.mp4` only (no `--fps`, `-s`, `-f`, or `-n`). Combined metric groups keep native geometry and `--fps`.

## CVVDP SDR raw inputs

PixVQA does not support CVVDP with SDR raw YUV (`raw_raw` / `raw_comp` with `--metrics sdr` transfer). The suite records **NA** (not applicable) for standalone CVVDP and affected SDR groups on those scenarios (product contract, not a harness defect). Missing checkpoints remain **SKIP** (dependency) and mandatory **SKIP** still fails `--strict`.
