# Project license

PixVQA is licensed under the **MIT License** (SPDX identifier `MIT`). The full text is in the repository root [`LICENSE`](../LICENSE).

Copyright (c) 2026 Pixop ApS.

The MIT license applies to the PixVQA source code and documentation that Pixop ApS contributes in this repository. It does not replace the licenses on third-party components, models, or notices listed below. It does not grant permission to use Pixop trademarks, trade names, or logos except as required to comply with the MIT notice on the Software itself. See [docs/brand/README.md](brand/README.md).

## Third-party and bundled material (not relicensed as MIT)

These stay under their own terms. Keep their notices when you redistribute or ship binaries:

| Component | License / terms | Where documented |
|---|---|---|
| Netflix libvmaf (linked at build time; not vendored as a full tree here) | BSD-2-Clause-Patent | `third_party/netflix-vmaf/`, `scripts/build_libvmaf.sh` |
| VMAF model JSON files in `model/` | BSD-2-Clause-Patent (Netflix VMAF) | `model/README.md`, `third_party/netflix-vmaf/NOTICE` |
| FFmpeg (user-built dependency) | LGPL 2.1 or later when built without `--enable-gpl` and without `--enable-version3` | `docs/INSTALL.md`, `THIRD_PARTY_NOTICES.md` |
| libdav1d (user-built dependency) | BSD-2-Clause | `THIRD_PARTY_NOTICES.md` |
| PU21 reference attribution | BSD-3-Clause | `licenses/PU21_BSD.txt` |
| XPSNR | Implementation in this tree; academic reference in README | ICASSP 2020 paper cited in README |
| QCS descriptor | Formula implemented in this tree; external reference implementation cited in QA docs | Not a ninth metric; see README |

Do not treat dependency notices or model files as if they were MIT-licensed by the root `LICENSE` alone.

## Historical note

Before this release candidate, `docs/LICENSING.md` compared MIT and BSD-3-Clause without selecting either. The user selected MIT for the PixVQA project contribution.
