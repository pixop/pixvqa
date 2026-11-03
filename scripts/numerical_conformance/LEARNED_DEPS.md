## CVQA-HDR-NR assets (optional learned metric)

Configure in `config/learned_metrics.local.toml`:

```toml
[cvqa-hdr-nr]
repository = "/path/to/CompressedVQA-HDR"
checkpoint = "/path/to/NR_HDR_VQA.pth"
profile = "/path/to/NR_HDR_VQA.npy"
```

Or environment overrides: `PIXVQA_CVQAHDR_PYTHON`, plus repo/checkpoint/profile via TOML.

When checkpoint paths are wrong or files are absent, comprehensive cases for `cvqa-hdr-nr` and SDR/PQ groups report **SKIP** (not PASS). Strict mode still exits nonzero until assets are installed or cases are marked optional.

## CVVDP SDR raw inputs

PixVQA does not support CVVDP with SDR raw YUV (`raw_raw` / `raw_comp` with `--metrics sdr` transfer). The suite records **SKIP** for standalone CVVDP and SDR groups on those scenarios (product contract, not a harness defect).
