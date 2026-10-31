# Learned metrics on CPU and CUDA

`cvvdp` and `cvqa-hdr-nr` run on either the CPU or an NVIDIA GPU through
PyTorch, from the one [shared environment](LEARNED_ENVIRONMENT.md). This page
covers device selection, CPU-vs-CUDA agreement, speed, and platform support.

## Selecting the device

`--device` is an alias of `--metric-device`. Both take `auto`, `cpu`, or
`gpu`. Giving both with different values is an error.

| Option | Native metrics | `cvvdp`, `cvqa-hdr-nr` |
|---|---|---|
| `cpu` | CPU | CPU. CUDA is never initialized for the model |
| `gpu` | CUDA build required; no CPU substitute (VMAF stays on CPU) | CUDA required. Without a CUDA device visible to PyTorch, preflight fails and no metric runs |
| `auto` (default) | CPU, except the resident UHD profile ([CLI_USAGE.md](CLI_USAGE.md)) | CUDA when `torch.cuda.is_available()`, else CPU with a `no CUDA device is visible to PyTorch; auto selected cpu` notice on stderr |

So `auto` is decided per metric, and in a mixed run native metrics can run on
the CPU while learned metrics run on CUDA. Use `--device cpu` for an all-CPU
run. MPS is never selected.

Each learned backend reports the device it used:

- in the summary line, for example `[ CVVDP jod:7.415657997131348 ... device:cuda ]`;
- in the protocol, as `pixvqa-value device` and `pixvqa-value device_requested`;
- in its JSON record, as `device`, `device_requested`, and `device_info`. `device_info` records the PyTorch version, its CUDA build, CUDA availability, CPU threads, Python, the platform, and on CUDA the GPU name, compute capability, memory, and cuDNN version.

The `auto` fallback notice is emitted once per backend execution, not during
the preflight process.

### How each backend enforces the device

- **CVVDP.**
  - The official `cvvdp` CLI's `get_best_device`, in `pycvvdp/utils.py`, turns an unavailable `cuda` request into a logged warning and a CPU run. PixVQA therefore never passes upstream `auto`, and never passes `cuda` without checking `torch.cuda.is_available()` first.
  - After the run, the adapter reads the CLI's `[INFO] Running on device: X` line. If that line is missing, or names a different device, the run is an error.
  - The QCS frame sequence is re-pooled in the adapter process on the same device.
- **CVQA-HDR-NR.**
  - The official model is built with `.to(device)`, and the checkpoint is loaded with `map_location=device`.
  - Before inference, the adapter checks that every parameter and the input tensor are on the selected device.
  - Upstream preprocessing (`VQA_NR.video_processing`, OpenCV) always runs on the CPU.

The upstream code has no hard-coded `.cuda()` calls and no CUDA extensions on
these paths, so both models run on the CPU without modification. Upstream
`cvvdp_metric.py` already loads its calibration checkpoint with
`map_location='cpu'` when CUDA is absent.

## CPU vs CUDA agreement

Validation ran `scripts/validate_learned_devices.py`, which scores the same pair
with `--device cpu` and `--device gpu`. It runs `--qcs` and keeps the JSON
records. It checks the reported devices, then compares the pooled scores,
temporal predictions, and QCS against the tolerances in the script.

Validation environment:

- Platform: WSL2 Ubuntu 24.04 (kernel 6.6.87.2) on Windows 11 build 26300.
- Hardware: Intel Core Ultra 7 265K (20 threads) and an RTX 5080 (compute capability 12.0).
- Software: PyTorch 2.11.0+cu128, cuDNN 9.19, and CVVDP 0.5.7 from commit `166f5ae`.
- CompressedVQA-HDR commit `76bff6d`, with checkpoint SHA-256 `b3765990…` and profile SHA-256 `601a7314…`.

Largest absolute CPU-to-CUDA difference:

| Content | CVVDP JOD | CVVDP per-frame JOD-eq. | CVVDP QCS | NR score | NR raw | NR per-sample score | NR QCS |
|---|---|---|---|---|---|---|---|
| BasketballGame 1080p PQ, crf20 vs crf40, 120 frames | 0 | 2.4e-6 | 0 | 7.8e-7 | 2.4e-7 | 2.4e-6 | 1.2e-5 |
| testsrc2 256x144 SDR BT.709, qp18 vs qp36, 48 frames | 0 | 9.5e-7 | 0 | 1.0e-6 | 3.6e-7 | 1.6e-6 | 1.1e-5 |
| testsrc2 256x144 HLG, qp18 vs qp36, 48 frames (CVVDP only) | 0 | 1.9e-6 | 2e-6 | - | - | - | - |
| testsrc2 256x144 PQ, `auto` with CUDA visible vs hidden | 0 | - | 2e-6 | 2.4e-6 | 8.3e-7 | - | 2.2e-5 |

The BasketballGame scores were CVVDP JOD `7.415657997131348` on both devices.
NR was `4.444974268667124` on CUDA, the same value as earlier releases, and
`4.444973489574522` on the CPU.

Tolerances, as absolute differences, are set in `scripts/validate_learned_devices.py`:

| Quantity | Tolerance | Largest observed |
|---|---|---|
| CVVDP pooled JOD | 1e-3 | 0 |
| CVVDP per-frame JOD-equivalent | 5e-3 | 2.4e-6 |
| CVVDP per-frame pooled distortion | 5e-3 | 7.6e-5 |
| CVVDP QCS (percentage points) | 0.5 | 2e-6 |
| NR quality score | 1e-3 | 2.4e-6 |
| NR raw model score | 1e-3 | 8.3e-7 |
| NR per-sample quality score | 2e-3 | 2.4e-6 |
| NR QCS (percentage points) | 0.5 | 2.2e-5 |

The results are not bit-identical because the float32 reductions run in a different order on each device:

- **CVVDP.** On CUDA, the official metric processes blocks of frames sized from free GPU memory. On the CPU it processes one frame at a time.
- **NR.** The SigLIP2 forward pass uses cuBLAS on CUDA and oneDNN/MKL on the CPU.

The differences observed are at float32 rounding level, below 1e-5 on
0-10 scales. The tolerances leave about 100x headroom for other GPUs, CPUs, and
thread counts. They are still far below any meaningful quality difference: 1e-3
JOD, and 0.5 QCS points (one frame in 200 crossing a band). CVVDP's pooled JOD
was bit-identical in every run, despite per-frame differences of up to 2.4e-6.

Reproduce:

```bash
python3 scripts/validate_learned_devices.py --pixvqa build/swdec/pixvqa \
  -r BasketballGame_1080p_crf20.mp4 -t BasketballGame_1080p_crf40.mp4 --out /tmp/pixvqa-devices
```

Use `--interpretation sdr --display standard_fhd` for SDR. The script writes
`report.json`, which holds the scores, differences, provenance, and
performance, and exits 1 when a tolerance or device check fails.

## Errors

| Case | Result |
|---|---|
| `--device gpu`, no CUDA device visible to PyTorch (`CUDA_VISIBLE_DEVICES=`) | Exit 2 in preflight, before any metric runs: `CUDA was requested but PyTorch 2.11.0+cu128 (CUDA build 12.8) reports no CUDA device; not falling back to CPU` |
| `--device gpu` with native metrics on a build without CUDA | Exit 2: `this build has no GPU metrics; refusing --metric-device gpu` |
| `--device metal` | Exit 2: `unsupported --device metal (use auto, cpu, or gpu)` |
| `--device cpu --metric-device gpu` | Exit 2: `conflicting --device cpu and --metric-device gpu` |
| Adapter `--device mps` or another value | Exit 2: argparse rejects it (choices `auto`, `cpu`, `cuda`) |
| Official CLI log reports a device other than the one requested | Exit 1: `cvvdp reported device ..., not ...` |

## Performance

All timings come from the same machine and BasketballGame 1080p PQ pair:

| | CPU | CUDA |
|---|---|---|
| pixvqa wall time, CVVDP + NR + QCS | 284.9 s | 23.3 s |
| Official CVVDP CLI, 120 frames | 272.4 s (2.27 s/frame) | 10.4 s (0.087 s/frame) |
| CVVDP QCS re-pooling | 0.67 s | 0.98 s |
| NR model build and checkpoint load | 2.3 s | 2.5 s |
| NR preprocessing, OpenCV decode of 120 frames and resize, on CPU for both | 0.61 s | 0.61 s |
| NR inference, 2 samples | 0.52 s | 0.20 s |
| Peak RSS, largest process | 2.56 GB (CVVDP CLI) | 2.30 GB (NR adapter) |
| Peak GPU memory | - | NR: 0.81 GB allocated by PyTorch. CVVDP: up to 13.4 GB device-wide, including the CUDA context |

Notes on the measurements:

- **The CPU bottleneck is CVVDP.** It processes one frame at a time through the pyramid and temporal filters, about 26 times slower than CUDA. NR is cheap on either device: it samples one frame per second, and its OpenCV preprocessing runs on the CPU in both cases.
- **Fixed startup costs.** The CUDA wall time includes about 9 s of interpreter startup, PyTorch imports, and CUDA initialization across the four backend processes (two preflights, two runs).
- **CVVDP's CUDA memory use scales with free memory.** It sizes its frame blocks from free GPU memory (NVML free memory minus 1 GB), so it uses most of an idle GPU. `--cvvdp-gpu-mem GB` forwards upstream's decimal-GB cap and limits the block budget to the lower of the cap and the normal free-memory estimate. Omit it to preserve the upstream default.
- **CVVDP memory-cap validation.** On the 120-frame BasketballGame 1080p PQ pair, the upstream default peaked at 14,062 MiB as sampled by `nvidia-smi`; `--cvvdp-gpu-mem 2` peaked at 2,208 MiB. The pooled JOD was bit-identical (`7.415657997131348`); maximum per-frame differences were `1.53e-5` in distortion and `9.54e-7` in JOD-equivalent. The default took 13.69 s and the 2 GB cap took 14.86 s (8.5% slower in this run). The JSON record stores `gpu_mem_gb` as `null` or `2.0`, respectively.
- **Small clips.** For the 256x144 48-frame clips, CVVDP took 47 s on the CPU and 3.1 s on CUDA. On the CPU, per-frame fixed costs dominate small inputs.
- **CUDA GPU memory figures.** For CVVDP, `nvidia-smi` sampled device-wide memory every 0.2 s, because the official CLI runs as a child process. For NR, the figure is `torch.cuda.max_memory_allocated`.

## Platforms

| Platform | Status |
|---|---|
| WSL2 Ubuntu 24.04 on Windows 11 | Verified: everything above, CPU and CUDA (driver 610.88 through `/usr/lib/wsl/lib`) |
| Native Linux | Not run on bare metal. The PixVQA binary, adapters, and PyTorch wheels are the same as under WSL2; only the NVIDIA driver path differs |
| Native Windows 11 | Not verified end to end. See below |

On native Windows:

- **The PixVQA binary.** It builds and runs on Linux only: it uses `/proc/self/exe`, `fork`/`exec`, `mkdtemp`, and `/tmp` backend directories. The learned metrics therefore run through PixVQA only under Linux or WSL2.
- **The Python adapters on their own.**
  - `python -m pixvqa_cvvdp` and `python -m pixvqa_cvqahdr` use `pathlib`, `tempfile`, and argument-list `subprocess` calls. They do not invoke a shell or hard-code Linux paths.
  - The CVVDP adapter finds `Scripts\cvvdp.exe` beside the interpreter.
  - For raw YUV, it falls back to a hard link when Windows refuses a symlink (no Developer Mode). The temporary directory must then be on the same volume as the YUV files.
  - Peak RSS is reported as `null` on Windows (no `resource` module).
- **What was checked on Windows 11 with Python 3.11.9 and torch 2.11.0+cpu.**
  - The adapters import.
  - Device resolution works: `auto` gives `cpu`, and `cuda` is rejected.
  - `cvvdp.exe` is found, and the symlink-to-hard-link fallback and its cleanup work.
  - The NR sampler gives the same schedule as on Linux.
- **What was not run.** No end-to-end metric run: that Python has no pycvvdp, open_clip, timm, OpenCV, or ffmpeg/ffprobe, and the shared environment is Linux-only.
- **Requirements for an end-to-end run.** A Windows environment built from `python/requirements/learned-env.txt`, with the CUDA wheel index for GPU use, and `ffmpeg`/`ffprobe` on `PATH`. The official CVVDP reads video through `ffmpeg-python`, so ffmpeg is required.
