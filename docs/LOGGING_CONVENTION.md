# Diagnostic logging convention (2026-11-04)

This change affects **human-readable stderr messages** only. PixVQA's
per-frame output, final numerical metric summaries, `VMAF:`/`VMAF-EXT:` stdout
metadata, adapter JSON, `pixvqa-protocol`/`pixvqa-value`/`pixvqa-sample`
records, and exit codes remain unchanged.

The canonical stderr prefix is `pixvqa[-component]: severity: message`.
`severity` is `info`, `warning`, or `error`, and components are `vmaf`,
`cvvdp`, and `cvqa-hdr-nr`. The core `pixvqa:` prefix is retained for
miscellaneous CLI diagnostics for compatibility with existing scripts.

Examples (values are illustrative):

```text
pixvqa-vmaf: info: model selection=auto model=model/hdr_vmaf_v1.1.8_hfr/...
pixvqa-vmaf: info: backend=cpu conversion=... display=2160 hfr_temporal=five_frame
pixvqa-cvvdp: info: interpretation=hlg source=--metrics hlg
pixvqa-cvvdp: info: display=standard_hdr_hlg source=default for transfer
pixvqa-cvvdp: warning: upstream: WARNING:root: The mean color value is less than 1 - the image may not be scaled in absolute photometric units!
pixvqa-cvqa-hdr-nr: info: upstream open_clip: No pretrained weights loaded for model 'ViT-B-16-SigLIP2-384'
```

The OpenCLIP `No pretrained weights loaded` notice is **informational**, not
a failed model load: PixVQA intentionally constructs with `pretrained=False`
and loads the selected checkpoint separately via `load_state_dict(strict=True)`.
An actual missing or incompatible checkpoint remains a runtime error.

The upstream CVVDP warning about mean color below 1 is **not suppressed**:
it may indicate a genuine interpretation/photometric scaling issue. The
message is retained with its originating library text and a warning severity.
Unknown third-party diagnostic text is passed through with `upstream:` so the
source of the message is clear. The relay handles fragmented stderr writes
while stdout stays dedicated to the numeric report protocol.

## Validation

Local dependency-free test:

```bash
python3 scripts/test_logging_convention.py
```

On the configured workstation, rebuild `make -j2 full` and rerun representative
VMAF, CVVDP, and CVQA-HDR-NR cases; the regression above cannot execute CUDA,
FFmpeg, or upstream neural models. The CLI tests referencing explicit stderr
message text were updated with the new component prefix, not weakened.
