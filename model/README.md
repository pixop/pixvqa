# VMAF v1.0.16 model JSON (vendored)

These eight files are exact byte copies of the official Netflix VMAF models at
commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c` (tag `v3.2.1`).

| Local file | Upstream path |
|---|---|
| `vmaf_v1.0.16_3d0h.json` | `model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json` |
| `vmaf_v1.0.16_5d0h.json` | `model/vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json` |
| `vmaf_v1.0.16_1d5h_2160.json` | `model/vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json` |
| `vmaf_v1.0.16_3d0h_2160.json` | `model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json` |
| `vmaf_v1.0.16_hfr_3d0h.json` | `model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json` |
| `vmaf_v1.0.16_hfr_5d0h.json` | `model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_5d0h.json` |
| `vmaf_v1.0.16_hfr_1d5h_2160.json` | `model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json` |
| `vmaf_v1.0.16_hfr_3d0h_2160.json` | `model/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h_2160.json` |

License and copyright: Netflix, Inc. See `third_party/netflix-vmaf/LICENSE`
(BSD-2-Clause-Patent) and `third_party/netflix-vmaf/NOTICE`.

Do not edit these JSON files in place. PixVQA identifies them by SHA-256 of the
loaded file contents.

# HDR VMAF v1.1.8 model JSON (vendored)

These eight files are exact byte copies of the official Netflix HDR VMAF models
at commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef`. That commit is not a
release tag. The SDR files above stay the v3.2.1 bytes.

| Local file | Upstream path |
|---|---|
| `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json` | `model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json` |
| `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json` | `model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json` |
| `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json` | `model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json` |
| `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json` | `model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json` |
| `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json` | `model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json` |
| `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json` | `model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json` |
| `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json` | `model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json` |
| `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json` | `model/hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json` |

`3d0h_2160` and `hfr_3d0h_2160` clip scores to `[0,110]`. The other six clip to
`[0,100]`. PixVQA does not rescale either range. Same license as the SDR models.
