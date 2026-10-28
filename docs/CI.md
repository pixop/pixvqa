# Continuous integration

Every pull request and every push to `main` runs the short CPU workflow. The Ubuntu installation matrix runs when the installer or its dependencies change, or when someone requests it. The full CPU suite and the GPU suite are started by hand. A missing tool, fixture, or GPU is a failure or an explicit gap. It is not a successful run.

`main` has no required status checks. If protection is added later, require **CPU build and hosted tests**. Do not require **Install on ubuntu-22.04** or **Install on ubuntu-24.04**: those jobs are skipped when the installer is unchanged, and a required skipped check can block the pull request.

GitHub-hosted runners check that the guided installer can reproduce a CPU build. They are not minimal Ubuntu installations. Installation jobs compile dav1d, FFmpeg, and libvmaf in the job. Those builds are not cached, so a requested installation stays clean.

## Automatic on every pull request and on `main`

**QA** (`.github/workflows/qa.yml`) runs on Ubuntu 24.04 for every pull request targeting `main` and every push to `main`:

- `bash build.sh` (CPU binary only)
- `build/cpu/selftest`
- `scripts/test_io_regressions.py`
- `scripts/test_qcs.py`
- `python3 scripts/run_qa.py --profile smoke`

This job does not build dav1d, FFmpeg, libvmaf, or CUDA. VMAF QCS is skipped there when libvmaf is absent, and that smoke case accepts the no-libvmaf refusal. The job does not claim full functional coverage.

The same workflow still has a manual hardware input. It runs only `--profile hardware`, and only on a self-hosted runner labeled `self-hosted`, `linux`, and `nvidia`. The GPU workflow below is the one used for release GPU validation.

**Clean installation** (`.github/workflows/install.yml`) runs the guided installer on `ubuntu-22.04` and `ubuntu-24.04` when a pull request changes one of these files:

- `install_pixvqa.sh`, `build.sh`, `build_decode.sh`, or `build_cuda.sh`
- `scripts/build_*.sh`, `scripts/install*.sh`, `scripts/deps_layout.sh`, `scripts/finish_install.sh`, `scripts/git_retry.sh`, or `scripts/libvmaf_flags.sh`
- `scripts/verify_installed_pixvqa.sh` or `scripts/ci/verify_install.sh`
- `.github/workflows/install.yml`

A documentation change or a CLI change outside that list does not start those two jobs. A small decision job still runs and records the skip. Pushing to `main` does not start this workflow.

Request both Ubuntu versions for any pull request by adding the `install` label. That label stays in effect for later pushes. `workflow_dispatch` also runs both versions; GitHub offers that button for this file because it is already on `main`. Either request compiles dependencies from source and does not restore a cache.

Each installation job starts from the checkout, with no prebuilt PixVQA binary, and runs:

```bash
PIXVQA_DEPS_ROOT="$PWD/deps" \
PIXVQA_ENABLE_DAV1D=1 \
PIXVQA_ENABLE_NVIDIA=0 \
bash install_pixvqa.sh
```

`PIXVQA_ENABLE_DAV1D=1` selects software decoding, including dav1d. `PIXVQA_ENABLE_NVIDIA=0` selects the CPU build. When those variables are unset, the installer still asks on a terminal. It then checks `./bin/pixvqa --version` with and without `LD_LIBRARY_PATH`, checks that `ldd` resolves every library, and runs smoke QA against `bin/pixvqa`.

## Manual full CPU suite

**Full CPU QA** (`.github/workflows/full-cpu.yml`) does not run on an ordinary pull request or on a documentation change. It installs the same CPU and software-decoding configuration on Ubuntu 24.04, then runs `python3 scripts/run_qa.py --profile full`. The profile fails unless its verdict is PASS. Optional skips such as valgrind do not fail it. A mandatory skip does.

That profile covers PSNR, XPSNR, SSIM, MS-SSIM, SDR and HDR VMAF, PU21-PSNR, PU21-SSIM, PU21-MS-SSIM, QCS, 8-bit and 10-bit, SDR, PQ, and HLG, raw YUV, software HEVC, AV1, and VVC, and invalid-input cases. The installer builds FFmpeg with libdav1d. Apt FFmpeg supplies the encoders. A second `ffmpeg` program is built from that same n8.1.3 source and used only as the decode oracle. It is not installed over the libraries PixVQA links. Ubuntu's FFmpeg does not decode VVC.

The workflow file is on `main`, so manual dispatch is available. It checks out the ref you name:

```bash
gh workflow run "Full CPU QA" --ref <branch>
```

On a pull request, add the `full-cpu` label. That runs this same job on the pull request head as merged with `main`. A later push does not start it again; remove the label and add it once more to test the new commit. Documentation-only changes do not start it.

### VVC

`tests/fixtures/vvc/bbb_360p.vvc` is not in the repository. Its redistribution rights are not established, so CI does not download it. A missing fixture is a failure, not a skip.

The full CPU job tries to build VVenC 1.14.0 and VVdeC 3.2.1 and generate a small synthetic 176x144 10-bit stream. PixVQA's software decoder is compared with the VVdeC reconstruction at PSNR 999.99. If that generation fails, the VVC case still runs and fails. The six local Kimono MP4s are not used.

## Manual GPU suite

**GPU QA** (`.github/workflows/gpu.yml`) is `workflow_dispatch` only:

```bash
gh workflow run "GPU QA" --ref feature/hdr-vmaf-remote-reconcile
```

`workflow_dispatch` for this file is available once it is on the default branch. Do not start it unless a runner with those labels exists. A queued job with no runner is not a successful GPU run.

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
