# Continuous integration

Pull requests run two automatic workflows. The full CPU suite and the GPU suite are started by hand. A missing tool, fixture, or GPU is a failure or an explicit gap. It is not a successful run.

GitHub-hosted runners check that the guided installer can reproduce a CPU build. They are not minimal Ubuntu installations.

## Automatic on every pull request

**QA** (`.github/workflows/qa.yml`) runs on Ubuntu 24.04:

- `bash build.sh` (CPU binary only)
- `build/cpu/selftest`
- `scripts/test_io_regressions.py`
- `scripts/test_qcs.py`
- `python3 scripts/run_qa.py --profile smoke`

This job does not build dav1d, FFmpeg, libvmaf, or CUDA. VMAF QCS is skipped there when libvmaf is absent, and that smoke case accepts the no-libvmaf refusal. The job does not claim full functional coverage.

The same workflow still has a manual hardware input. It runs only `--profile hardware`, and only on a self-hosted runner labeled `self-hosted`, `linux`, and `nvidia`. The GPU workflow below is the one used for release GPU validation.

**Clean installation** (`.github/workflows/install.yml`) runs the guided installer on `ubuntu-22.04` and `ubuntu-24.04`. Each job starts from the checkout, with no prebuilt PixVQA binary, and runs:

```bash
PIXVQA_DEPS_ROOT="$PWD/deps" \
PIXVQA_ENABLE_DAV1D=1 \
PIXVQA_ENABLE_NVIDIA=0 \
bash install_pixvqa.sh
```

`PIXVQA_ENABLE_DAV1D=1` selects software decoding, including dav1d. `PIXVQA_ENABLE_NVIDIA=0` selects the CPU build. When those variables are unset, the installer still asks on a terminal. It then checks `./bin/pixvqa --version` with and without `LD_LIBRARY_PATH`, checks that `ldd` resolves every library, and runs smoke QA against `bin/pixvqa`.

## Manual full CPU suite

**Full CPU QA** (`.github/workflows/full-cpu.yml`) is `workflow_dispatch` only. It installs the same CPU and software-decoding configuration on Ubuntu 24.04, then runs `python3 scripts/run_qa.py --profile full`.

That profile covers PSNR, XPSNR, SSIM, MS-SSIM, SDR and HDR VMAF, PU21-PSNR, PU21-SSIM, PU21-MS-SSIM, QCS, 8-bit and 10-bit, SDR, PQ, and HLG, raw YUV, software HEVC, AV1, and VVC, and invalid-input cases. The installer builds FFmpeg with libdav1d. Apt FFmpeg supplies the independent HEVC and AV1 encodes. Ubuntu's FFmpeg does not decode VVC.

Start it from the branch under test:

```bash
gh workflow run "Full CPU QA" --ref feature/hdr-vmaf-remote-reconcile
```

The job fails unless the profile verdict is PASS. Optional skips such as valgrind do not fail it. A mandatory skip does.

### VVC

`tests/fixtures/vvc/bbb_360p.vvc` is not in the repository. Its redistribution rights are not established, so CI does not download it. A missing fixture is a failure, not a skip.

The full CPU job tries to build VVenC 1.14.0 and VVdeC 3.2.1 and generate a small synthetic 176x144 10-bit stream. PixVQA's software decoder is compared with the VVdeC reconstruction at PSNR 999.99. If that generation fails, the VVC case still runs and fails. The six local Kimono MP4s are not used.

## Manual GPU suite

**GPU QA** (`.github/workflows/gpu.yml`) is `workflow_dispatch` only:

```bash
gh workflow run "GPU QA" --ref feature/hdr-vmaf-remote-reconcile
```

It requires a self-hosted runner labeled `self-hosted`, `linux`, and `nvidia`, plus CUDA toolkit 12.8 already installed (`nvcc` on `PATH` or `/usr/local/cuda-12.8`). The workflow does not install the driver or the toolkit. It fails immediately when `nvidia-smi` or CUDA 12.8 is missing. A queued run with no runner is not a pass, and this repository does not register one.

When that runner exists, the job installs with dav1d and NVIDIA enabled, then runs:

- `--profile hardware` (CUDA PSNR, SSIM, MS-SSIM, 8-bit QCS agreement, HEVC and AV1 NVDEC)
- `scripts/test_cuda_psnr_ssim_8bit.py` (PSNR, SSIM, MS-SSIM, XPSNR, 8-bit and 10-bit, QCS)
- `scripts/test_pu21_cuda.py` (all three PU21 metrics, 8-bit and 10-bit, QCS)

Both sides of each CPU/GPU comparison use the installed CUDA binary's CPU and GPU metric devices. VMAF stays on CPU libvmaf. Numerical tolerances are unchanged.

## What remains outside these jobs

- A minimal cloud image that is not a GitHub-hosted runner.
- GPU metrics and NVDEC, until an authorized self-hosted runner exists.
- The external Big Buck Bunny VVC bitstream.
- Film-grain AV1, a mid-stream resolution change, concurrent GPU processes, and a long stress run.
- Valgrind and ASan, unless those binaries and tools are present. Their absence is an optional skip.
