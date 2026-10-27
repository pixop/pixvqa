# HDR VMAF release qualification

Recommendation for merging `feature/hdr-vmaf-remote-reconcile` into GitHub `main`: **CONDITIONAL GO**.

This addendum qualifies the branch after CPU review of `ce21c7b14fdac00e5f0158b4aa1860b709598175`. The historical measurements in [HDR_VMAF_INTEGRATION.md](HDR_VMAF_INTEGRATION.md) stay as recorded. Do not publish `feature/hdr-vmaf-public-integration` from that earlier report.

The qualification fixes sit on top of the reviewed commit. Their commit is the child of `ce21c7b14fdac00e5f0158b4aa1860b709598175` on `feature/hdr-vmaf-remote-reconcile`. No remote was updated.

## 1. Workspace

| Item | Value |
|---|---|
| Worktree | `D:\projects\pixvqa-public-remote-reconcile` |
| Branch | `feature/hdr-vmaf-remote-reconcile` |
| Reviewed HEAD | `ce21c7b14fdac00e5f0158b4aa1860b709598175` |
| Parents | `9eeb674` (remote main) and `4e030bf` (HDR integration) |
| Local `main` | `4e030bf` at `D:\projects\pixvqa-public`, unchanged |
| Other worktrees | unchanged |

The reviewed commit was still HEAD, and the worktree was clean, before the qualification edits. No branch switch, rebase, or history rewrite was done.

## 2. Installation fixes

`install_pixvqa.sh` and `docs/INSTALL.md` now install and require `perl`, `autoconf`, `automake`, `libtool`/`libtoolize`, and `m4` from the same Ubuntu release. `/usr/bin/g++` remains mandatory for `GCC_ENV=/usr` and for the final NVDEC link. The script does not substitute another compiler.

`scripts/build_zimg.sh` uses autoconf, automake, libtoolize, and m4 from `PATH` when they are present. If they are absent and `/etc/os-release` is not Ubuntu 24.04, the script exits. On a reported Ubuntu 22.04 it also prints that it is refusing Ubuntu 24.04 binary packages. The SHA-256-pinned Ubuntu 24.04 packages remain only for a machine that is already Ubuntu 24.04 and has none of those tools. zimg stays release-3.0.6, commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd`.

`scripts/libvmaf_flags.sh` links a prefix only when `pixvqa-libvmaf-stamp.txt` records libvmaf `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` and zimg `3.0.6`. A missing stamp, the previous tag commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`, or any other zimg version clears `PIXVQA_HAS_VMAF` and the flags. The default prefix remains `deps/libvmaf-74f930d`. Multiarch `lib`, `lib64`, and `lib/x86_64-linux-gnu` search, the `pcfiledir` check, `vmaf_picture_convert`, and `-lzimg` are unchanged. `scripts/build_libvmaf_3.2.1.sh` does not write this stamp, so that prefix cannot pass the check.

No machine-specific absolute path was added to the scripts or install documents.

## 3. Rebuild

Host: WSL Ubuntu 24.04. CPU compiler: conda-forge g++ 16.2.0 (`GCC_ENV=/home/sergio/gcc-env`). CUDA: nvcc 12.8, V12.8.93, `CUDA_HOME=/home/sergio/cuda-toolkit`, gencode `sm_89` and `sm_120`. FFmpeg: n8.1.3 at `/home/sergio/ffmpeg-n8.1.3-lgpl`. GPU: NVIDIA GeForce RTX 5080. Python for the suites: Miniconda Python with NumPy.

The previous executables were deleted first. Commands, from the worktree:

```bash
export GCC_ENV=/home/sergio/gcc-env
export CUDA_HOME=/home/sergio/cuda-toolkit
export FFMPEG_PREFIX=/home/sergio/ffmpeg-n8.1.3-lgpl
export DAV1D_LIB=/home/sergio/dav1d-1.5.1/lib64
export PIXVQA_LIBCUDA_DIR=/usr/lib/wsl/lib
export CUDART_LIBDIR=/home/sergio/cuda-toolkit/targets/x86_64-linux/lib
bash build.sh
bash build_cuda.sh
PIXVQA_FULL_LINK_CXX=/home/sergio/gcc13-nvcc/bin/x86_64-conda-linux-gnu-g++ bash build_decode.sh
```

`/usr/bin/g++` is absent here, so the full link used the existing `PIXVQA_FULL_LINK_CXX` override: conda-forge g++ 13.4.0. That override is evidence for this machine only. The committed default remains `/usr/bin/g++`. A system-GCC final link was **not** validated.

`build.sh` reported `All selftests passed`. Versions from the new binaries: CPU `build: cpu`, `decode: none`; CUDA `build: cuda`, `decode: none`; full `build: cuda`, `decode: software nvdec`. All three contain `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` and `3.0.6`, and none contain `f85a853692a8c730d0270cd733c8bb30b5b93b7c`. The linked stamp records zimg commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd` and pkg-config version 3.0.6. libvmaf and zimg were not rebuilt; the existing pinned prefixes were checked and reused.

## 4. CPU and CUDA results

Suites were run on the new CPU binary unless a row names another binary. HDR parity worst absolute difference is `5.000e-7` across 27 cases. Real BasketballGame means: `94.852236`, `91.299559`, and `95.002617`. SDR numeric suite pool absolute max is about `4.888e-7` (mean) through `4.997e-7` (harmonic). The 8-bit CLI absolute max is `3.526e-7`. Error checks passed. `scripts/test_vmaf_device_fallback.py` passed against `build/full/pixvqa`, including the three-frame floor case `0.000000`. Smoke QA: `PASS 16 FAIL 0 SKIP 0`.

CUDA comparison used a new 320×240 10-bit pair, 8 frames. CPU and CUDA scores matched. GPU-only runs did not emit the VMAF warning. Requesting VMAF with `--metric-device gpu` printed `pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf.` and `VMAF: cpu libvmaf`. The four CUDA metrics stayed on CUDA in the same run.

| Request | Backend | Score |
|---|---|---|
| PSNR | `PSNR: gpu fp64` | y `31.22`, avg `32.98` |
| SSIM | `SSIM: gpu fp64` | avg `0.993616` |
| MS-SSIM | `MS-SSIM: gpu fp64` | avg `0.999207` |
| XPSNR | `XPSNR: gpu` | avg `22.1175` |
| All four | each line stays `gpu` | same four scores |
| SDR VMAF, GPU requested | CPU libvmaf, warning | mean `100.000000`, 8 frames |
| SDR VMAF + all four | CUDA metrics `gpu`, VMAF CPU | same metric scores, same VMAF line |
| HDR VMAF, GPU requested | CPU libvmaf, warning | mean `100.000000`, 8 frames |
| HDR VMAF + all four | CUDA metrics `gpu`, VMAF CPU | same metric scores, same VMAF line |

The VMAF mean of `100.000000` is the ceiling on that mild pair. It matches the CPU binary. It is not a CUDA VMAF score.

## 5. Software decode and NVDEC

The full binary is the one linked in section 3. NVDEC accepts HEVC at 420p:10 only, so the SDR bitstream is 10-bit limited BT.709 rather than 8-bit. The raw reference has no stream tags. HDR tests pass `--ref-hdr-transfer pq`, `--ref-hdr-primaries bt2020`, `--ref-hdr-matrix bt2020nc`, and `--ref-hdr-range limited` for that raw side only. The distorted HEVC color comes from its stream tags.

| Case | Frames | Color | VMAF | Decode |
|---|---|---|---|---|
| HDR HEVC, software vs NVDEC | 8 | `color_ref=limited:bt2020:pq:bt2020nc`, `color_dist=limited:bt2020:pq:bt2020nc` | mean `68.178175`, harmonic `68.151843`, min `66.337679`, max `70.668921` | identical scores; NVDEC run has `backend=nvdec`, `distorted: gpu`, `VMAF: cpu libvmaf`, `PSNR: gpu` |
| SDR BT.709 HEVC, software vs NVDEC | 8 | `color_ref=unset`, `color_dist=unset` | mean `95.408410`, harmonic `95.407068`, min `94.877161`, max `95.997443` | identical scores; NVDEC run has `backend=nvdec`, `distorted: gpu`, `VMAF: cpu libvmaf`, `PSNR: gpu` |
| HDR 1080 extract from BasketballGame, software vs NVDEC | 16 | distorted stream `limited:bt2020:pq:bt2020nc` | mean `80.047936`, harmonic `79.990552`, min `73.942982`, max `83.065113` | identical VMAF; NVDEC run also has `XPSNR: gpu` and `VMAF: cpu libvmaf` |

SDR `unset` is the SDR model path: the BT.709 tags were accepted and were not turned into an HDR color spec. PSNR on the decode rows is the compressed-versus-raw score, y `39.62`, avg `41.38`, and it stayed on CUDA.

`scripts/test_av1_vvc.py` with `PIXVQA_RUN_GPU=1` on these binaries:

| Check | Result |
|---|---|
| AV1 8-bit NVDEC | pass |
| AV1 10-bit NVDEC | pass |
| HEVC 10-bit NVDEC | pass |
| Truncated AV1 is not scored | pass |
| AV1 software decode | fail: `AV1 software decoding requires libdav1d or libaom-av1` |
| VVC fixture | not run: `tests/fixtures/vvc/bbb_360p.vvc` is absent |

The FFmpeg CLI can decode the generated AV1. The FFmpeg libraries this PixVQA build links do not provide libdav1d or libaom, so software AV1 is an environment limit of this prefix. The VVC fixture was not downloaded.

## 6. Guided installation

End-to-end guided installation: **NOT RUN**.

No disposable Ubuntu 22.04 or 24.04 environment was available. Package installation, system changes, and an FFmpeg rebuild were not run on this workstation. This WSL image is Ubuntu 24.04, but `/usr/bin/g++`, autoconf, automake, libtoolize, and m4 are not installed, and the existing home prefixes were reused. That is not a clean install.

Static checks that did run:

- `bash -n` passed for `install_pixvqa.sh`, `scripts/build_zimg.sh`, `scripts/build_libvmaf.sh`, `scripts/libvmaf_flags.sh`, `build.sh`, `build_cuda.sh`, and `build_decode.sh`.
- The installer apt line contains `perl`, `autoconf`, `automake`, `libtool`, and `m4`, and the script still sets `LIBVMAF_PREFIX` to `deps/libvmaf-74f930d` and `PIXVQA_FULL_LINK_CXX=/usr/bin/g++`.
- With autotools hidden, the zimg decision exits 1 when the version id is 22.04 and prints `refusing Ubuntu 24.04 binary autotools on Ubuntu 22.04`. A real Ubuntu 22.04 system was not booted.
- On this Ubuntu 24.04 image the same hidden-tool decision would reach the pinned package download. That download was not executed, and zimg was not rebuilt. The existing prefix was already zimg 3.0.6.
- A prefix with no stamp is refused. A stamp carrying commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c` is refused (`PIXVQA_HAS_VMAF=0`). A stamp carrying zimg `3.2.1` is refused. The real prefix, with the HDR stamp, sets `PIXVQA_HAS_VMAF=1` and keeps `-lzimg`.
- `PREFIX` set to a temp directory, `scripts/install.sh` copied 16 model JSON files.
- `scripts/verify_vmaf_model_checksums.py` reported `MATCH` for all 16 models against Netflix `f85a853` (SDR) and `74f930d` (HDR). The script rewrote `reports/VMAF_MODEL_CHECKSUMS.md` with no content change; that file was not committed.

The clean-install procedure is [INSTALL_VALIDATION.md](../docs/INSTALL_VALIDATION.md). Supported OS claims remain Ubuntu 22.04 and 24.04. Neither was given a fresh installer PASS in this qualification.

## 7. Remaining risks

- The documented final linker, `/usr/bin/g++`, was not the linker that produced the NVDEC binary.
- Guided installation on a fresh Ubuntu 22.04 or 24.04 system was not run. The Ubuntu 24.04 autotools package fallback was not downloaded or checksum-verified in this run.
- Software AV1 needs an FFmpeg build with libdav1d or libaom. This prefix does not have it.
- The VVC fixture is intentionally absent. GPU rejection of VVC was not rechecked.
- HDR CLI parity remains the existing 27-case set plus one 60-frame BasketballGame pair. The internal 48-case set was not repeated. Decode comparisons are 8 frames at 320×240 and 16 frames at 1080p.
- HDR feature extraction, VMAF models, model clip ranges, and CUDA kernels were not changed.

## 8. Recommendation

**CONDITIONAL GO** for merging this branch into GitHub `main`.

The reviewed scoring behavior was rebuilt and rechecked. PSNR, SSIM, MS-SSIM, and XPSNR stay on CUDA. VMAF stays on CPU libvmaf and warns when a GPU device is requested. HDR and SDR HEVC NVDEC match software decoding on the frames above. The installer now asks for Perl and distro autotools, refuses another Ubuntu release's autotools packages, and will not link an unstamped or stale libvmaf prefix.

It is not a GO while the clean guided install and the `/usr/bin/g++` final link are still untested. It is not a NO-GO: no scoring, model, or CUDA-metric failure was found on this candidate.
