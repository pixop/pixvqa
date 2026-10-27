# Public HDR VMAF and CPU fallback — integration

Recommendation: **superseded**. Do not publish `feature/hdr-vmaf-public-integration` as the release branch. The current release candidate is `feature/hdr-vmaf-remote-reconcile`. See [HDR_VMAF_RELEASE_QUALIFICATION.md](HDR_VMAF_RELEASE_QUALIFICATION.md).

The sections below record the earlier HDR-plus-fallback integration. They were not rewritten.

## 1. Feature commits

| Branch | Worktree | Commit |
|---|---|---|
| `feature/hdr-vmaf-v1-models` | `D:\projects\pixvqa-public-hdr-vmaf` | `30664c6941104695937c4f45e483e3e52c45af6c` |
| `feature/vmaf-device-fallback` | `D:\projects\pixvqa-public-vmaf-fallback` | `fdeb12bfce2f48a3a7dfa0a4e4a386335570f774` |

Both start from public `main` at `10574f6181021a34c3b698460c408ace9e547dd2`. The eight SDR model JSON files are not in either commit. Their git blobs match `HEAD`. The fallback worktree still shows those files as modified because the working-tree line endings differ from the index stat; `git diff` on them is empty and they were not staged.

## 2. Integration branch

| Item | Value |
|---|---|
| Worktree | `D:\projects\pixvqa-public-hdr-integration` |
| Branch | `feature/hdr-vmaf-public-integration` |
| First parent | `30664c6` |
| Merged | `fdeb12b` |

The original feature worktrees were not edited after their commits. Build prefixes under the HDR worktree's gitignored `deps/` were read through directory junctions. Those junctions are not part of the commit.

## 3. Conflict resolution

Git reported a content conflict only in `.gitattributes`. The resolution keeps `model/*.json text eol=lf` and the two HDR model directory rules.

`src/main.cpp`, `include/pixvqa/types.hpp`, `README.md`, and `scripts/qa/inventory.py` merged without conflict markers. The merged `resolve_execution` is the fallback policy. The HDR color checks, model table, libvmaf `74f930d`, and zimg 3.0.6 stayed. One help sentence still said a VMAF-only GPU request was rejected. That sentence now says HLG and full-range PQ stay rejected, and a GPU request runs HDR VMAF on CPU libvmaf and warns. It does not say CUDA HDR.

`VmafBackend` remains CPU. The summary prints `requested device` and `VMAF: cpu libvmaf`. The warning is:

```text
pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf.
```

There is no `--vmaf-backend`. CUDA metric kernels were not edited. HDR VMAF is not implemented on CUDA.

## 4. Combined CPU validation

Binary: `build/cpu/pixvqa`, `pixvqa version 0.3.0-dev`, `decode: none`. Linked to libvmaf `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` and zimg 3.0.6.

`bash build.sh`: selftest, IO regressions, golden-log, and production-parser checks passed.

| Check | Result |
|---|---|
| `scripts/test_vmaf_hdr.py` | 27 scored cases, worst abs `5.000e-7`, plus the rejection cases. GPU request on the 1080 `3d0h` pair matched the CPU pooled line, warned, and reported `backend=cpu` |
| Real HEVC | BasketballGame frames 1800–1859, x265 CRF 28. `hfr_3d0h_2160` mean `94.852236`, `3d0h_2160` mean `91.299559`, Lanczos 1080 `hfr_3d0h` mean `95.002617`. Worst abs about `5.0e-7` |
| `scripts/test_vmaf_device_fallback.py` | SDR and HDR CPU / auto / GPU requests. GPU warns. Scores match. CPU-only build still rejects XPSNR or PSNR on `--metric-device gpu` |
| `scripts/test_vmaf_errors.py` | 13 checks passed, including the GPU warning before the 64×64 size error, and SDR PQ/HLG rejection |
| `scripts/test_vmaf_smoke.py` | Score abs `0`, feature abs `3.789e-7` |
| `scripts/test_vmaf_numeric.py` | Eight SDR models. Score abs `0`. Largest feature abs `3.505e-7`. `3d0h_2160` pooled `108.879197`, `hfr_3d0h_2160` pooled `110` |
| `scripts/test_vmaf_8bit.py` | CLI abs `3.526e-7`, C-API oracle abs `0` |
| `scripts/test_vmaf_numeric_suite.py` | 36 cases. Pool abs max about `5.0e-7` |
| `scripts/verify_vmaf_model_checksums.py` | 16/16 exact SHA-256 matches |
| `scripts/run_qa.py --profile smoke` | `PASS 16 FAIL 0 SKIP 0` |
| `scripts/install.sh` | Temp prefix received 16 model JSON files |

`scripts/test_vmaf_hdr.py` resolves FFmpeg from `FFMPEG` or `PATH`. It no longer defaults to `/home/sergio/bin/ffmpeg`. The real-content run set `FFMPEG` in the environment. The BasketballGame master was available. Other masters from the internal 48-case set were not run.

## 5. CUDA validation

Existing local tools, not a newly installed toolkit:

- `nvcc` CUDA 12.8 at `/home/sergio/cuda-toolkit`
- GCC 13.4 at `/home/sergio/gcc13-nvcc`
- FFmpeg n8.1.3 at `/home/sergio/ffmpeg-n8.1.3-lgpl`
- libdav1d 1.5.1 at `/home/sergio/dav1d-1.5.1/lib64`
- GPU: NVIDIA GeForce RTX 5080

`build_cuda.sh` produced `build/cuda/pixvqa` (`decode: none`). `build_decode.sh` produced `build/swdec/pixvqa` and `build/full/pixvqa` (`decode: software nvdec`).

| Request | Result |
|---|---|
| HDR VMAF, `--metric-device gpu`, raw 1920×1080, 3 frames | Same pooled line as CPU libvmaf: mean/harmonic/min/max `100.000000`. Warning present. `backend=cpu`. `VMAF: cpu libvmaf` |
| HDR VMAF + XPSNR, `--metric-device gpu` | `XPSNR: gpu` and `VMAF: cpu libvmaf`. Same VMAF line |
| SDR VMAF + PSNR, `--metric-device gpu` | `PSNR: gpu` and `VMAF: cpu libvmaf`. Same VMAF line as the CPU binary, mean `100.000000` |
| PSNR + SSIM + MS-SSIM, no VMAF, `--metric-device gpu` | All three `gpu`. No VMAF warning |
| HDR HEVC, `--decode gpu` vs `--decode cpu` | Identical VMAF line: mean `94.337889`, harmonic `94.337747`, min `94.173012`, max `94.425630`, 3 frames. Summary: `requested device: gpu`, `distorted: gpu`, `VMAF: cpu libvmaf`, `backend=cpu` |
| SDR HEVC, `--decode gpu` vs `--decode cpu` | Identical VMAF line: mean `94.588688`, harmonic `94.587903`, min `94.313557`, max `94.962672`, 3 frames. Same device lines |

The 320×240 device-fallback fixture still prints VMAF mean `0.000000` for both SDR and HDR. That is the model clip floor on that pair, and the three device selections print the same line.

`scripts/test_vmaf_device_fallback.py` also passed its full-binary section, including HDR VMAF alone, HDR VMAF + XPSNR, and SDR VMAF + PSNR.

## 6. Installation

`scripts/install.sh` copies the eight SDR JSON files and both HDR directories. A temp install contained 16 JSON files.

`scripts/build_zimg.sh`, `scripts/build_libvmaf.sh`, and `scripts/build_libvmaf_3.2.1.sh` prepend `$GCC_ENV/bin` only when `GCC_ENV` is set. They no longer look for `$HOME/gcc-env`. `gcc`, `g++`, and GNU `make` must be on `PATH`. Ubuntu `build-essential` supplies `make`. If the autotools are missing, `build_zimg.sh` still downloads the pinned Ubuntu 24.04 debs and checks the SHA-256 values in the script. The libvmaf commit and the zimg 3.0.6 commit are unchanged. That automatic deb fallback was later restricted so Ubuntu 22.04 cannot extract Ubuntu 24.04 binaries; the restriction is recorded in the release-qualification report.

The CPU build on this machine used `GCC_ENV=/home/sergio/gcc-env` (conda-forge g++ 16.2.0). That path is not a default in the scripts.

## 7. Device matrix

| Request | Behavior |
|---|---|
| SDR or HDR VMAF, CPU or auto | CPU libvmaf, no warning |
| SDR or HDR VMAF, GPU | CPU libvmaf, warning, `backend=cpu` |
| HDR VMAF + XPSNR, GPU | CPU HDR VMAF + CUDA XPSNR |
| SDR VMAF + PSNR, GPU | CPU SDR VMAF + CUDA PSNR |
| CUDA metrics without VMAF | Those metrics stay on CUDA |
| CPU-only binary, VMAF-only GPU | CPU libvmaf and the warning |
| CPU-only binary, XPSNR or PSNR with GPU | Still rejected. No CPU substitute |

## 8. Recommendation

**Superseded.** Do not merge `feature/hdr-vmaf-public-integration` into GitHub `main` on this report alone. Remote `main` moved after this integration. The release candidate and its qualification result are in [HDR_VMAF_RELEASE_QUALIFICATION.md](HDR_VMAF_RELEASE_QUALIFICATION.md).

The integrated tree keeps HDR colorimetry, the eight HDR models, the SDR models and their clips, libvmaf `74f930d`, zimg 3.0.6, CUDA metric behavior, and GPU decoding. VMAF's reported backend is CPU. The warning does not describe a CUDA HDR score.

Coverage bounds, already true of the public HDR validation: the real-content CLI parity is one 60-frame BasketballGame CRF 28 pair plus a Lanczos 1080 derivative, and the NVDEC comparison is 3 frames. The internal 48-case set was not repeated here.
