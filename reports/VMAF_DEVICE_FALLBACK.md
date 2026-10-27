# VMAF CPU fallback — public pixvqa

Status: **implemented, uncommitted**, on `feature/vmaf-device-fallback`. CPU-build tests passed. CUDA mixed execution was not run because this worktree has no full binary.

## Workspace

| Item | Value |
|---|---|
| Worktree | `D:\projects\pixvqa-public-vmaf-fallback` |
| Branch | `feature/vmaf-device-fallback` |
| Base | `main` at `10574f6181021a34c3b698460c408ace9e547dd2` |
| Binary | `build/cpu/pixvqa`, libvmaf 3.2.1 (`f85a853`), prefix taken from the internal checkout's existing `deps/libvmaf-3.2.1` for this build only |
| Compiler | g++ 16.2.0 |

No commit, push, or merge. `--vmaf-backend` was not added. GPU decoding was not changed. CUDA kernels were not changed.

## Behavior

`--metric-device gpu` with VMAF, alone or with other metrics, selects CPU libvmaf and prints:

```text
pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf.
```

`--metric-device cpu` and `auto` do not print that warning. `auto` still keeps VMAF on the CPU, and it still does not enable the 4K NVDEC GPU profile when VMAF is selected.

PSNR, SSIM, MS-SSIM, and XPSNR still have no CPU substitute. A CPU-only build rejects `--metric-device gpu` for those metrics with `no GPU metrics`. PU21-only on the GPU is still an error.

There is no JSON result document. The README states that. The execution summary is the device report. It now starts with the request and names VMAF's actual backend:

```text
Execution:
  requested device: gpu
  reference: raw
  distorted: raw
  VMAF: cpu libvmaf
```

`VMAF: ... backend=cpu` on the model line is unchanged. A GPU request is not labeled as a CUDA VMAF score. Existing `PSNR: gpu` / `PSNR: cpu` lines are unchanged when those metrics run.

## Tests

`bash build.sh` passed selftest, IO regressions, golden logs, and the public parser checks.

`scripts/test_vmaf_errors.py` passed, including the 64×64 GPU request: the warning is printed, then the known model still rejects the frame with `needs at least 320x240` (exit 2).

`scripts/test_vmaf_device_fallback.py` on the CPU binary, 320×240 10-bit, 3 frames, ramp versus a flat field:

| Request | Result |
|---|---|
| VMAF, `cpu` | CPU libvmaf, no warning, `requested device: cpu` |
| VMAF, `auto` | CPU libvmaf, no warning, `requested device: auto` |
| VMAF, `gpu` | CPU libvmaf, warning, `requested device: gpu` |
| Same three | `[ VMAF mean:0.000000 harmonic:0.000000 min:0.000000 max:0.000000 frames:3 ]` |
| VMAF + XPSNR, `gpu` | Rejected. `no GPU metrics`. XPSNR was not moved to the CPU |
| PSNR + MS-SSIM, `gpu` | Rejected. `no GPU metrics` |
| `--decode gpu --input compressed` on a raw file | Rejected before a VMAF score. `no video stream` |

The mean of 0 is this fixture hitting the clip floor. The three device selections printed the same line.

Not run, because `build/full/pixvqa` is absent:

- VMAF + XPSNR on a CUDA device
- VMAF + PSNR/MS-SSIM on a CUDA device
- GPU metrics without VMAF on a CUDA device
- NVDEC plus CPU VMAF on a real bitstream
- A CUDA build with no visible GPU

The code path for the mixed case is the existing GPU assignment plus `vmaf_cpu_notice`. It was not executed.

Machine-readable log: `reports/vmaf_device_fallback_results.json`.

## Files

| Path | Change |
|---|---|
| `src/main.cpp` | VMAF-only GPU is a warning and CPU libvmaf. Summary prints `requested device` and `VMAF: cpu libvmaf` |
| `include/pixvqa/types.hpp` | Comment only. `VmafBackend` stays CPU |
| `README.md` | The GPU-only VMAF error is now the warning and the CPU backend |
| `.gitattributes` | `model/*.json text eol=lf` added beside the existing `*.sh` rule, so the model SHA-256 used for the size check survives a Windows checkout |
| `scripts/test_vmaf_errors.py` | GPU VMAF expects the warning, then the size error |
| `scripts/test_vmaf_gate_b.py` | `vmaf_gpu_uses_cpu` expects exit 0, the warning, and `backend=cpu` |
| `scripts/qa/cases_media.py` | Hardware case expects the fallback beside `PSNR: gpu` |
| `scripts/qa/inventory.py` | Rule text: VMAF falls back; the other GPU metrics do not |
| `scripts/test_vmaf_device_fallback.py` | New CPU-build coverage |
| `reports/vmaf_device_fallback_results.json` | This run |

Model JSON bytes were restored to LF in the worktree. `git diff` on those files is empty.

## HDR note

Do not copy this onto `D:\projects\pixop-vqa-hdr-vmaf`. That tree has no `--metric-device`. The internal dependency and the CUDA VMAF inventory are in `D:\projects\pixop-vqa-vmaf-fallback\reports\VMAF_DEVICE_FALLBACK.md`.
