# BT.2020 constant-luminance and PU21 — engineering notes

PixVQA 0.3.0-dev source update, 2026-11-04. These notes document source
semantics; they do **not** assert CUDA or real 4K campaign completion.

## Signal definition and inverse

The recognized matrix is `bt2020c` (FFmpeg `AVCOL_SPC_BT2020_CL`), distinct
from `bt2020nc`. BT.2020 constant-luminance encodes
`Yc' = T(0.2627 R + 0.6780 G + 0.0593 B)`; `T` is the transfer function.
`Cbc` and `Crc` contain *differences of non-linear* component values:

```
R' = Yc' + (Crc <= 0 ? 1.7184 : 0.9936) * Crc
B' = Yc' + (Cbc <= 0 ? 1.9404 : 1.5816) * Cbc
R = T^-1(R'), B = T^-1(B'), Yc = T^-1(Yc')
G = (Yc - 0.2627 R - 0.0593 B) / 0.6780
```

For PQ, the inverses return nits on the 0..10,000 cd/m2 ST 2084 scale. For
HLG, they return relative scene linear values, then the existing BT.2100 OOTF
is applied with the configured peak, black and system gamma. The inverse must
not pass Yc'CbcCrc through the NCL Y'CbCr->R'G'B' matrix.

References (equations and normalization):

- Khronos Data Format Specification 1.4, section 15.2,
  <https://registry.khronos.org/DataFormat/specs/1.4/dataformat.1.4.pdf>
- Linux V4L2 detailed colorspace definitions, BT.2020 constant luminance,
  <https://docs.kernel.org/6.6/userspace-api/media/v4l/colorspaces-details.html>

Code values use limited-range `Y=(code-16*2^(b-8))/(219*2^(b-8))` and
`C=(code-128*2^(b-8))/(224*2^(b-8))`, or full-range `Y=code/(2^b-1)` and
`C=code/(2^b-1)-0.5`, for b=8..16 in CPU helpers. The GPU PU21 backend is
currently limited to b=8 or 10.

When CL reconstruction produces an out-of-range non-linear component or an
out-of-gamut G component, it is bounded to the supported display domain. This
is **not silent**: PU21 records the event in its `clamped` flag and prints a
warning once per run. Tiny floating-point G round-off within 1e-8 nits (PQ)
or 1e-10 (HLG scene units) is not counted as an out-of-gamut event. As before,
PU21 clamps values outside its supported 0.005..10,000 cd/m2 encoder range.
This is a practical clipping policy for quantization and subsampled chroma,
not a claim that an out-of-gamut CL signal is physically valid. Neither CPU
nor GPU currently provides per-pixel clipping counts.

## Compatibility contract

| Selected path | CL/CL | CL/NCL |
| --- | --- | --- |
| PSNR, SSIM, MS-SSIM, XPSNR (stored samples) | Supported only with matching PQ/HLG transfer, BT.2020 primaries, and explicit matching range | Rejected: decoded sample codes are not comparable |
| PU21-PSNR, PU21-SSIM, PU21-MS-SSIM | Supported with the validated CL inverse (CPU); CUDA parity still requires workstation run | Conservatively rejected by CLI, pending an explicit mixed-domain policy |
| Bundled HDR VMAF or other VMAF | Rejected, no CL-to-NCL conversion | Rejected |
| CVVDP and CVQA-HDR-NR | Rejected at native combined-metric guard and/or Python adapter preflight | Rejected if either input uses CL; CVQA is no-reference |

PU21 uses display-domain RGB and is conceptually capable of comparing CL and
NCL after independently converting them. PixVQA retains the existing strict
CL/NCL rejection until a separately designed test and policy addresses
full-frame mixed matrices, metadata and raw input assumptions. The downloaded
`pq_hevc_qp32.mp4` was reported to be **CL tagged, not proven CL encoded**;
it must **not** be used as an analytical CL inverse oracle.

## Local numerical tests (no GPU)

```bash
make -j2 cpu
make -j2 test-cpu
python3 scripts/test_bt2020cl_numerical.py --cpu-exe build/cpu/pixvqa
python3 scripts/test_bt2020cl_learned_guards.py
python3 scripts/test_cvvdp_edge_policies.py
python3 scripts/test_integration_campaign_manifest.py
```

The C++ selftest uses independently *forward-encoded* linear RGB vectors,
including black, white, neutrals, saturated primaries, both chroma sign
branches, PQ and HLG (1000 / 4000 nit peaks), direct RGB and scalar luminance
checks, and code conversion at 8/10/12/16-bit full/limited range. The Python
numerical CLI oracle independently encodes genuine CL vectors, quantizes
8/10-bit YUV420 in both ranges, decodes nominal values, reconstructs linear
light, and compares PU21-PSNR components and pooled scores to PixVQA.

## Mandatory WSL/CUDA verification

The downloaded archive has no `deps/`, `build/` or 4K source fixtures.
Run from the patched source checkout on the workstation:

```bash
GCC13="$PWD/deps/gcc13" FFMPEG_PREFIX="$PWD/deps/ffmpeg-n8.1.3" \
CUDA_HOME="/usr/local/cuda-12.8" make -j2 full
make -j2 test-cpu
python3 scripts/test_bt2020cl_numerical.py --cpu-exe build/cpu/pixvqa \
  --gpu-exe build/full/pixvqa
python3 scripts/test_pu21_cuda.py
python3 scripts/run_real_video_integration.py --list > reports/real_video_146_manifest.json
python3 scripts/run_real_video_integration.py --check-fixtures
python3 scripts/run_real_video_integration.py --resume --output artifacts/real_video_integration
```

`--check-fixtures` prints missing inputs/models and exits 2 on missing. An
integration case passes only if all requested metrics are parsed, exactly the
requested frames are processed, and the process returns zero. GPU parity cases
add a second mandatory GPU run and numerical comparison. Missing files are
`BLOCKED`, never `PASS`/`SKIP`, and the complete campaign report is at
`artifacts/real_video_integration/full_146_results.json`.

Use `--phase A` through `--phase G`, `--mode pq`, or repeated `--case-id ID`
for incremental runs. **Filtered runs write separate scope-named JSON files** and
cannot overwrite `full_146_results.json`; a `complete` status for a filtered
report refers only to its `selected` cases, while `complete_full_campaign`
requires all 146. `--resume` reuses a prior PASS only for the same scope and if
executable, fixture/model timestamps and sizes, case definition, selected
frames, and commands match. Re-running an unfiltered full campaign does not
currently import PASS records from separate filtered reports. Run sequentially
(one worker) on the 6 GB WSL memory cap.

### Proposed 146-case manifest

| Phase | Count | Content | Provenance |
| --- | ---: | --- | --- |
| A | 48 | 3 modes x 4 x 4 ordered raw/HEVC/AV1/VVC PSNR comparisons | Historical Phase A; the handoff reports 48 past PASS on older code |
| B | 12 | 3 modes x 4 formats, identical-file PSNR/SSIM/MS-SSIM/XPSNR | New proposed definition |
| C | 24 | 3 modes x 8 directed raw/codec cross comparisons, PSNR/SSIM/MS-SSIM | New proposed definition |
| D | 18 | PQ/HLG x 9 pairs, all PU21 variants on CPU | New proposed definition |
| E | 18 | 6 compressed pairs x SDR VMAF, PQ HDR VMAF or HLG XPSNR+SSIM | New proposed definition |
| F | 12 | PQ/HLG x 6 compressed pairs, mandatory CPU-vs-GPU PU21 parity | New proposed definition |
| G | 14 | PQ/HLG x 7 selected pairs, peak-conditioned PU21 (PQ=1000, HLG=4000) | **New proposed additions to reconcile 132->146** |
| **Total** | **146** |  | **No full local integration run** |

B--F are also **not recovered original definitions**: their counts were in
the handoff, but their exact inputs/metric configurations were not. Treat them
as a proposed, inspectable campaign specification until approved. If an
original B--F specification surfaces, review and revise the manifest; do not
interpret matching totals as historical equivalence.
