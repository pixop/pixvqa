# GPU and NVDEC release qualification

Validated product commit: `81d7108c8daf0aedbc569116021caebd4622bb9a` on `feature/hdr-vmaf-remote-reconcile`. PR #3 was not merged. The installed executable was `bin/pixvqa` from that tree:

```
pixvqa version 0.3.0-dev
build: cuda
decode: software nvdec
```

Recommendation: **GO** for GPU functionality in the initial public release. No CUDA or NVDEC defect blocked the tests below.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, compute capability 12.0, 16303 MiB |
| Driver | 610.88 |
| CUDA | Toolkit 12.8, V12.8.93 |
| Host compiler | GCC 13.4.0 for CUDA; GCC 16.2.0 for the CPU objects |
| Final link | GCC 13.4. `/usr/bin/g++` is not installed in this WSL distro, so the documented `/usr/bin/g++` link was not used |
| FFmpeg libraries | n8.1.3 LGPL prefix, `CONFIG_NVDEC`, `CONFIG_CUVID`, and `CONFIG_FFNVCODEC` enabled |
| FFmpeg CLI | n8.1.3, statically linked, including libdav1d and `av1_cuvid` |
| Architectures | `sm_89` and `sm_120` |

The shared `libavcodec` used by PixVQA does not export libdav1d. Software AV1 therefore fails in this local binary. The static CLI still decodes AV1, and that decode was the AV1 oracle. GitHub Actions, which links a dav1d-enabled FFmpeg, already passed software AV1.

## Tolerances already used by the project

- Printed PSNR and XPSNR text must match. CUDA XPSNR also requires exact per-frame WSSE.
- SSIM and MS-SSIM absolute error at most `1e-6` on the printed values. The hardware profile allows `1e-4`.
- PU21 fails only when the absolute error exceeds `1e-4` and the relative error exceeds `1e-6`.
- QCS text must match.

No tolerance was widened. Observed PSNR, SSIM, MS-SSIM, XPSNR, and PU21 differences on the raw and lossless cases were 0 at the printed precision. `scripts/test_cuda_numeric.py` reported worst non-PSNR absolute error `0` for its fp64, fp32, and mixed labels. Those labels do not select a different public kernel: the public GPU path is FP64.

## Validation matrix

| Component | Result | Evidence |
|---|---|---|
| CUDA PSNR | PASS | `scripts/qualify_gpu_release.py`; `scripts/test_cuda_psnr_ssim_8bit.py`; `scripts/test_cuda_numeric.py`. 8-bit and 10-bit, 4:2:0, 4:2:2, and 4:4:4. Printed text matched. |
| CUDA XPSNR | PASS | `scripts/test_xpsnr_cuda_wsse.py` (exact WSSE, including 1080p and 4K) and QCS equality in `scripts/test_qcs.py`. |
| CUDA SSIM | PASS | Same PSNR/SSIM suite. Maximum absolute difference 0 in `test_cuda_numeric.py`. |
| CUDA MS-SSIM | PASS | Raw 8-bit and 10-bit qualification, plus `test_cuda_psnr_ssim_8bit.py` and `test_qcs.py`. |
| CUDA PU21-PSNR | PASS | `scripts/test_pu21_cuda.py` and the qualification matrix: 8-bit and 10-bit, PQ and HLG, 4:2:0, 4:2:2, and 4:4:4, including `--hdr-peak-nits`. |
| CUDA PU21-SSIM | PASS | Same PU21 runs. |
| CUDA PU21-MS-SSIM | PASS | Same PU21 runs. |
| NVDEC HEVC | PASS | Lossless 256×144 Main 10 4:2:0 matched software-decode CPU scores for PSNR, SSIM, MS-SSIM, XPSNR, and PU21. Kimono HEVC, first 4 frames, matched as well. `--decode gpu` stayed on `hevc_cuvid`. |
| NVDEC AV1 | PASS | 8-bit 256×144 and Kimono 10-bit, first 4 frames. NVDEC PSNR matched the dav1d CLI decode. CUDA scores on those NVDEC frames matched CPU scores. |
| Automatic metadata detection | PASS | On NVDEC: compressed+compressed, raw+compressed, compressed+raw, and raw+raw. Wrong `-s` and `-f` were rejected. PQ and HLG came from the stream. XPSNR used `24/1` and matched `--fps 24`; `--fps 30` was honored. `scripts/test_detect_format.py` also passed. |
| QCS with CUDA metrics | PASS | `scripts/test_qcs.py` for PSNR, SSIM, MS-SSIM, XPSNR, and VMAF. Qualification QCS for those metrics and for PU21 matched the CPU text. |
| Mixed CPU/GPU metrics | PASS | `--metric-device gpu` keeps PSNR and XPSNR on CUDA and prints `VMAF: cpu libvmaf` for both the SDR and HDR models. `scripts/test_vmaf_device_fallback.py` passed. |

## Commands

```bash
# Local prefixes: conda-forge GCC 16.2.0, GCC 13.4 for the CUDA link,
# CUDA toolkit 12.8, FFmpeg n8.1.3, dav1d 1.5.1, and the WSL libcuda directory.
bash build.sh && bash build_cuda.sh && bash build_decode.sh
PREFIX="$PWD" PIXVQA_BIN="$PWD/build/full/pixvqa" bash scripts/install.sh
env -u LD_LIBRARY_PATH ./bin/pixvqa --version

python3 scripts/qualify_gpu_release.py
python3 scripts/run_qa.py --profile hardware
python3 scripts/test_pu21_cuda.py
python3 scripts/test_detect_format.py
python3 scripts/test_p3_gpu_decode.py
python3 scripts/test_vmaf_device_fallback.py
python3 scripts/test_cuda_numeric.py
python3 scripts/test_xpsnr_cuda_wsse.py
# numpy is not installed for /usr/bin/python3; these two used a NumPy-enabled Python:
python3 scripts/test_cuda_psnr_ssim_8bit.py
python3 scripts/test_qcs.py
```

## Counts

| Run | Passed | Failed | Skipped |
|---|---:|---:|---:|
| `scripts/qualify_gpu_release.py` on `bin/pixvqa` | 50 | 0 | 0 |
| Hardware QA profile | 2 | 1 | 0 |
| `test_cuda_psnr_ssim_8bit.py` | all | 0 | 0 |
| `test_pu21_cuda.py` | all | 0 | 0 |
| `test_qcs.py` | all | 0 | 0 |
| `test_detect_format.py` | all | 0 | 0 |
| `test_p3_gpu_decode.py` | all | 0 | 0 |
| `test_vmaf_device_fallback.py` | all | 0 | 0 |
| `test_cuda_numeric.py` | all | 0 | 0 |
| `test_xpsnr_cuda_wsse.py` | all | 0 | 0 |
| `test_av1_vvc.py` with `PIXVQA_RUN_GPU=1` | NVDEC rows passed | script failed | VVC fixture absent |

The hardware profile is **not** a pass: 2 passed and 1 failed. `gpu-metrics` and `gpu-repeat` passed. `hardware-av1-hevc-vvc` failed. The software FFmpeg linked into `build/swdec/pixvqa` has no libdav1d or libaom-av1, so AV1 software decode reports `AV1 software decoding requires libdav1d or libaom-av1`. `tests/fixtures/vvc/bbb_360p.vvc` is absent, so the VVC row fails instead of being skipped. Inside that same failed script, the NVDEC checks passed for 8-bit AV1, 10-bit AV1, and 10-bit HEVC. The failure is not suppressed, and it is not a CUDA score mismatch. The repeated-run check was three sequential CUDA runs plus two concurrent jobs. It is not a stress test. No CUDA runtime or GPU memory error appeared.

Kimono files `ref_hevc.mp4`, `dist_hevc.mp4`, `ref_av1.mp4`, `dist_av1.mp4`, `ref_vvc.mp4`, and `dist_vvc.mp4` were present locally at 960×540. Scoring used the first 4 frames. The files were not committed.

## Known limitations

- `--decode gpu` accepts HEVC Main 10 4:2:0 and AV1 8-bit or 10-bit 4:2:0. Widths below 144 are rejected by NVDEC. H.264, 8-bit HEVC, 4:2:2, and VVC are rejected and are not silently decoded in software.
- VMAF, including HDR VMAF, stays on CPU libvmaf when a GPU is requested.
- This WSL environment has no `/usr/bin/g++`. The combined binary was linked with GCC 13.4 instead of the linker named in `docs/INSTALL.md`.
- The pinned VVC fixture `tests/fixtures/vvc/bbb_360p.vvc` is not in this checkout. Local Kimono VVC was software-decoded and was rejected on NVDEC.

## GitHub Actions on `81d7108`

| Check | Result |
|---|---|
| Install on ubuntu-22.04 | pass |
| Install on ubuntu-24.04 | pass |
| CPU build and hosted tests | pass |
| Full CPU suite | pass |
| Hardware / GPU (manual, self-hosted only) | skipping |

There is no release-blocking GPU issue in the local qualification.
