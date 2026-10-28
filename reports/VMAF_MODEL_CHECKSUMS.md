# VMAF model SHA-256 (exact byte identity)

SDR upstream: Netflix/vmaf `f85a853692a8c730d0270cd733c8bb30b5b93b7c` (tag `v3.2.1`).
HDR upstream: Netflix/vmaf `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` (not a release tag).

Verified by `scripts/verify_vmaf_model_checksums.py` downloading each
upstream file and comparing SHA-256 / bytes to `model/`.

Result: **exact match for all sixteen files**.

| Pin | File | SHA-256 |
|---|---|---|
| `f85a853` | `vmaf_v1.0.16_3d0h.json` | `e4cf8c147e1368b35497d772920bc92f98c1ad7853c1033d8a836947f427140e` |
| `f85a853` | `vmaf_v1.0.16_5d0h.json` | `64e12d9c1e4ed5d79da7fa515b82fac7247a9e2ccc91e0ebce31d05489500385` |
| `f85a853` | `vmaf_v1.0.16_1d5h_2160.json` | `634d36e0502cb797384726cdd1a106a2c207827595a9e4958d3dae1bd517b139` |
| `f85a853` | `vmaf_v1.0.16_3d0h_2160.json` | `77f552fd8d5b6c6a2166bfc4470c9368ec68d2425ed40f1cc235b0cc500f33c2` |
| `f85a853` | `vmaf_v1.0.16_hfr_3d0h.json` | `dcfdfab1c00685c1e5f75b6f1078bb4a44c722654e4de090fa5c66784c75241b` |
| `f85a853` | `vmaf_v1.0.16_hfr_5d0h.json` | `590f612c87df94badca58e563b318fa9958eac631ab8aadd73bff8cc1adfc515` |
| `f85a853` | `vmaf_v1.0.16_hfr_1d5h_2160.json` | `0371db55d3de81a760125f6d65eb4386f1d38af6cf1d00c2f48158e3596246d4` |
| `f85a853` | `vmaf_v1.0.16_hfr_3d0h_2160.json` | `1bc502f00aeee11433e2d09468186bbf1bbce56ee650c9bade06c7435ebdd4db` |
| `74f930d` | `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json` | `10c9ad02b10ebf77507cf2ba5825c9b60d0a1d95c811e5f382085bb9f26103ad` |
| `74f930d` | `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json` | `f2ca19c167c31c7e6b9d567770ee151a727757e601bcb954c2228c33768d6af8` |
| `74f930d` | `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json` | `53d36d3173012efa6eb65ada92aa3967d3caceee68d088275b2a51a320948510` |
| `74f930d` | `hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json` | `6e1bb29ade13e1cad05e1d13fed2ad76fc8e3d3621bd5e02ab23e5c98c0afffd` |
| `74f930d` | `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json` | `e881b71d283bad7886afff0e5f6bd35fd59c8e9d58f7b8823bac9435e226d34f` |
| `74f930d` | `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json` | `0c33ee5c588307b3790703b958cf5e9dc5f7907f98cdb4e58e3b91648d8c48c2` |
| `74f930d` | `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json` | `c8a979574c0d79bb22198323e6b4024b47bd305e5aa09575c4735fef43476bd8` |
| `74f930d` | `hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json` | `f32904ac748319a8dc584aeffa776df30e5002e95270414a57a9802a9c2b31d1` |

PixVQA prints `sha256=` on the `VMAF:` metadata line using the digest of the
**actual loaded file** (resolved path), not a preset name.

Reproduce:

```bash
python3 scripts/verify_vmaf_model_checksums.py
```
