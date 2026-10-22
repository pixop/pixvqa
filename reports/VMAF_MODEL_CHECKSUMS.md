# VMAF v1.0.16 model SHA-256 (exact byte identity)

Upstream pin: Netflix/vmaf `f85a853692a8c730d0270cd733c8bb30b5b93b7c` (tag `v3.2.1`).

Verified by `scripts/verify_vmaf_model_checksums.py` downloading each
upstream file and comparing SHA-256 / bytes to `model/`.

Result: **exact match for all eight files**.

| File | SHA-256 |
|---|---|
| `vmaf_v1.0.16_3d0h.json` | `e4cf8c147e1368b35497d772920bc92f98c1ad7853c1033d8a836947f427140e` |
| `vmaf_v1.0.16_5d0h.json` | `64e12d9c1e4ed5d79da7fa515b82fac7247a9e2ccc91e0ebce31d05489500385` |
| `vmaf_v1.0.16_1d5h_2160.json` | `634d36e0502cb797384726cdd1a106a2c207827595a9e4958d3dae1bd517b139` |
| `vmaf_v1.0.16_3d0h_2160.json` | `77f552fd8d5b6c6a2166bfc4470c9368ec68d2425ed40f1cc235b0cc500f33c2` |
| `vmaf_v1.0.16_hfr_3d0h.json` | `dcfdfab1c00685c1e5f75b6f1078bb4a44c722654e4de090fa5c66784c75241b` |
| `vmaf_v1.0.16_hfr_5d0h.json` | `590f612c87df94badca58e563b318fa9958eac631ab8aadd73bff8cc1adfc515` |
| `vmaf_v1.0.16_hfr_1d5h_2160.json` | `0371db55d3de81a760125f6d65eb4386f1d38af6cf1d00c2f48158e3596246d4` |
| `vmaf_v1.0.16_hfr_3d0h_2160.json` | `1bc502f00aeee11433e2d09468186bbf1bbce56ee650c9bade06c7435ebdd4db` |

PixVQA prints `sha256=` on the `VMAF:` metadata line using the digest of the
**actual loaded file** (resolved path), not a preset name.

Reproduce:

```bash
python3 scripts/verify_vmaf_model_checksums.py
```
