# Optional Learned Metrics

Up to **four** Python-backed stacks can share configuration via [`config/learned_metrics.local.toml`](../config/learned_metrics.example.toml) and [`python/requirements/learned-env.txt`](../python/requirements/learned-env.txt):

| Backend | Role in PixVQA | Supported inputs |
|---|---|---|
| `cvvdp` ([ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)) | Full-reference JOD (`pixvqa --metrics cvvdp`) | Compressed or raw YUV420p10le, including mixed raw/compressed |
| `cvqa-hdr-nr` ([CompressedVQA-HDR](https://github.com/sunwei925/CompressedVQA-HDR)) | No-reference (`pixvqa --metrics cvqa-hdr-nr`) | Compressed SDR and HDR10; not raw YUV or HLG |
| SimpleVQA ([upstream](https://github.com/sunwei925/SimpleVQA)) | No-reference (`pixvqa --metrics simplevqa`); same module also runs standalone ([SIMPLEVQA_REFERENCE.md](dev/SIMPLEVQA_REFERENCE.md)); **CPU supported** via staged overlay; **CUDA batch opt-in** `--use-worker` on `python -m pixvqa_simplevqa` only (not forwarded by `pixvqa` yet) | Compressed **SDR** only (explicit PQ/HLG rejected; unknown/missing stream transfer is allowed at the C++ gate—Python still ffprobes); raw YUV rejected |
| `hdr-chipqa` ([HDR-ChipQA](https://github.com/Netflix/hdr_chipqa)) | No-reference pooled MOS-like score | SDR BT.709, PQ HDR, and **experimental** HLG; raw YUV420 8/10-bit or compressed |

For SDR/PQ/HLG format support and input caveats (YUV layout, transfer tags, QCS), see [Video Format Support by Metric](METRICS_AND_FORMATS.md#video-format-support-by-metric).

What you must provide falls into **two** buckets:

1. **Python environment and packages** — a CPython 3.12 virtual environment with the locked dependencies in [`python/requirements/learned-env.txt`](../python/requirements/learned-env.txt), plus ColorVideoVDP at the pinned commit (installed from a local Git checkout; [`scripts/setup_learned_env.sh`](../scripts/setup_learned_env.sh) never downloads model weights). That same environment includes PyTorchVideo and related pins so you can run the SimpleVQA adapter against a Git checkout under `deps/SimpleVQA` (or elsewhere).
2. **Upstream repositories and model assets** — Git checkouts under `deps/` (ColorVideoVDP, CompressedVQA-HDR, SimpleVQA, hdr_chipqa) plus NR/SimpleVQA checkpoints where each backend expects them. HDR-ChipQA SVR/scaler pickles ship **in** the hdr_chipqa repository (SHA-256 pinned). CompressedVQA-HDR NR weights are fetched from upstream Dropbox links when the installer can verify the pinned digest. SimpleVQA UGC weights are on **Google Drive**; the installer provisions `gdown==5.2.0` inside its managed learned environment, handles large-file confirmation, and accepts the checkpoint only when its SHA-256 matches the known artifact. If the host blocks automated downloads, the installer exits with an actionable error. Review each upstream license before redistributing or mirroring weights.

The **guided installer** (`install_pixvqa.sh`) does **not** install learned metrics unless you opt in: the prompt defaults to **no**, or set `PIXVQA_ENABLE_LEARNED=1` for non-interactive installs. If Python 3.12 is not already available, PixVQA provisions it in a private dependency prefix (`$HOME/pixvqa-deps/python312` by default, or `$PIXVQA_DEPS_ROOT/python312` for an isolated installation) using micromamba/Conda, without replacing Ubuntu system Python. This interpreter is distinct from the checkout-local `.venv` it creates. Provide `PYTHON=/path/to/python3.12` to override automatic discovery/provisioning. When opted in, it runs [`scripts/install_learned_metrics.sh`](../scripts/install_learned_metrics.sh), which:

- creates or reuses the **repository-local** `.venv` (CVVDP + NR + SimpleVQA deps) and clones ColorVideoVDP under `deps/ColorVideoVDP` by default (Git-ignored);
- provisions HDR-ChipQA **inference** on private **CPython 3.10** (`$PIXVQA_DEPS_ROOT/python310` via micromamba when needed) with **sklearn 1.0.2** in `.venv-chipqa-inference` by default;
- reuses the shared `.venv` for HDR-ChipQA **feature** extraction (adds `scikit-image` when missing);
- clones or reuses pinned upstream repos, downloads verifiable checkpoints, merges `config/learned_metrics.local.toml` without overwriting your paths, and runs readiness checks (import/load probes).
- prefetches SimpleVQA's auxiliary Kinetics SlowFast R50 pretrained checkpoint (`SLOWFAST_8x8_R50.pyth`, SHA-256 pinned) into `torch.hub.get_dir()/checkpoints`, the exact location used by PyTorchVideo at inference; reuses and verifies an existing checkpoint, and fails if it is missing in offline mode. Concurrent installers synchronize via a cache lock and the download is atomic. This does not change the SlowFast model weights or inference math.

Set `PIXVQA_LEARNED_INSTALL_OFFLINE=1` to skip network fetch (verify-only). The installer **never** overwrites an existing file that matches the pin, **never** accepts HTML as a checkpoint, and **never** claims full installation when Google Drive or Dropbox fetches fail verification.

System-installed Python, CUDA, the NVIDIA driver, and package-manager or pip **caches** outside the repo are still normal prerequisites and are not removed when you move or delete the checkout.

For **manual setup** (full control, CI, or an existing environment), create the venv and config yourself:

```bash
scripts/setup_learned_env.sh /absolute/path/to/.venv /absolute/path/to/ColorVideoVDP
cp config/learned_metrics.example.toml config/learned_metrics.local.toml
# Edit learned_metrics.local.toml — paths to checkpoints, NR repo, and [learned].python
```

Copy [`config/learned_metrics.example.toml`](../config/learned_metrics.example.toml) to `config/learned_metrics.local.toml` (Git-ignored). The [example](../config/learned_metrics.example.toml) uses a **relative** `python` path resolved next to the config file (`../.venv/bin/python`). After running the installer with an empty `learned_metrics.local.toml`, the generated file uses an **absolute** path to the selected interpreter so the template line does not have to match your filesystem layout.

### Defaults and overrides

| Variable | Default | Role |
|----------|---------|------|
| `PIXVQA_LEARNED_ENV_DIR` | `<repo>/.venv` | Virtual environment directory; must be an absolute path when set |
| `PIXVQA_CVVDP_SOURCE` | `<repo>/deps/ColorVideoVDP` | ColorVideoVDP Git checkout used by `setup_learned_env.sh` |
| `PIXVQA_LEARNED_VENV_USE_SYMLINKS` | unset | If set (non-empty), `setup_learned_env.sh` creates the venv **without** `--copies` even on WSL 9p/drvfs mounts (Linux ext4 or intentional symlink venv) |
| `PIXVQA_HDR_CHIPQA_INFERENCE_ENV_DIR` | `<repo>/.venv-chipqa-inference` | sklearn 1.0.2 SVR environment (CPython 3.10) |
| `PIXVQA_HDR_CHIPQA_FEATURE_PYTHON` | shared `.venv` | Override HDR-ChipQA feature extractor interpreter |
| `PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON` | unset | Override host 3.10 used to create the inference venv |
| `PIXVQA_LEARNED_INSTALL_OFFLINE` | unset | If set, `install_learned_metrics.sh` verifies local assets only (no clone/download) |

Older documentation referred to `$HOME/pixvqa-learned-env`; that remains valid if you set `PIXVQA_LEARNED_ENV_DIR` accordingly. The installer preserves custom paths in an existing `learned_metrics.local.toml`, but repairs known erroneous base-Python paths written by an earlier installer. If the file already lists a different `[learned].python` than the environment selected for this run, the script prints a **warning**; to opt into a repo-local venv, create a new venv, set `PIXVQA_LEARNED_ENV_DIR` to its path (or use the default `.venv`), update `[learned].python` and any checkpoint paths, and retire the old venv yourself.

**WSL2 and Windows-backed mounts:** Repositories on drives exposed under `/mnt/...` (e.g. `D:` → `/mnt/d/...`) are often served to Linux as **9p** (WSL) or similar. Default `venv` layout uses symlinks to the host Python; those links can fail on some cross-filesystem layouts. For those mounts, [`scripts/setup_learned_env.sh`](../scripts/setup_learned_env.sh) uses `python -m venv --copies`, which duplicates the interpreter into `.venv` so shebangs and imports resolve (validated in practice on 9p). Expect **higher latency** and possible tooling edge cases compared to ext4; prefer a clone under the Linux filesystem (e.g. `$HOME`) when you need maximum I/O performance or want a symlink-based venv (`PIXVQA_LEARNED_VENV_USE_SYMLINKS=1` on ext4). This is **not** a guarantee for every mount or future WSL change — after creation, confirm with `"$PIXVQA_LEARNED_ENV_DIR/bin/python" -c 'import sys; print(sys.executable)'`.

`.venv` is **deployment-specific**: do not commit it; after relocating the repository, recreate the environment or point `PIXVQA_LEARNED_ENV_DIR` at your existing venv.

## Setup

> ⚠️ **CVVDP GPU resource usage:** High-resolution video and concurrent CVVDP jobs can require substantial GPU memory and compute. Monitor GPU utilization and available VRAM (for example, with `nvidia-smi`) and verify that the workload fits your GPU before running large batches. CVVDP has been verified on NVIDIA GeForce RTX 4070 and RTX 5080; this does not guarantee sufficient resources for every resolution, configuration or degree of parallelism. The optional `--cvvdp-gpu-mem GB` setting provides a memory budget, but is not a substitute for monitoring actual usage.

> **HDR-ChipQA host-RAM / concurrency limitation (observed on WSL2):** During a 1920x1080, 60 fps, 1,199-frame (~20 s) benchmark with five concurrent PixVQA jobs, Linux OOM-killed several Python feature-extraction processes (roughly 2.9-3.5 GiB anonymous RSS at termination) under a WSL2 limit of **10 GB RAM + 4 GB swap**. The same multi-job benchmarking workflow ran after excluding `hdr-chipqa`. This supports treating HDR-ChipQA as **unsuitable for this constrained parallel workload**, not a universal per-job RAM estimate or proof that all implementations/4K jobs fail. For parallel campaigns on comparable hosts, omit `hdr-chipqa` (e.g. `--metrics psnr,ssim,msssim,xpsnr,vmaf,cvqa-hdr-nr,simplevqa`), or run HDR-ChipQA in a separately controlled/serial pass after checking available host RAM and peak RSS. `--threads` does not impose a RAM cap across independent PixVQA processes. Do not confuse this **system RAM** limitation with CVVDP's **GPU VRAM** warning above.

## Examples

```bash
# Full-reference ColorVideoVDP (PQ):
./bin/pixvqa --metrics cvvdp --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq -r reference.mp4 -t distorted.mp4

# CompressedVQA-HDR NR:
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# Combine with native metrics:
./bin/pixvqa --metrics psnr,cvvdp,cvqa-hdr-nr --hdr-transfer pq \
  -r reference.mp4 -t distorted.mp4

# HDR-ChipQA (NR; also selected automatically by --metrics sdr|pq|hlg):
./bin/pixvqa --metrics pq -r reference.mp4 -t distorted.mp4
```

### Metric groups, CVVDP, and HDR-ChipQA

- **`sdr`**, **`pq`**, and **`hlg`** groups include **HDR-ChipQA** automatically (HLG scores are experimental). Configure `[hdr-chipqa]` in `learned_metrics.local.toml` or pass `--hdr-chipqa-*` overrides.
- **CVVDP** is **not** in group baselines (GPU cost). Append it explicitly, e.g. `--metrics pq,cvvdp`, `--metrics sdr,cvvdp`, or `--metrics hlg,cvvdp`; the group then fills `--cvvdp-interpretation` and `--cvvdp-display` when possible.
- **`all`** still selects CVVDP (and native metrics) but does **not** imply a transfer group; it does not auto-select HDR-ChipQA.

Compressed video metadata may supply transfer; when it cannot, set `--hdr-transfer` or `--cvvdp-interpretation`. Specify a suitable SDR display explicitly (e.g. `--cvvdp-display standard_fhd`). Raw CVVDP input requires geometry, format, rate and correct transfer/color interpretation.

`--frames N` caps both learned metrics. In a mixed native/learned run, all metrics use the common successfully compared frame prefix. The NR model samples at approximately 1 Hz by default; adjust using `--cvqa-nr-sample-interval-sec` and `--cvqa-nr-chunk-size-sec`.

HDR-ChipQA implementation details and validation evidence: [CHIPQA_REFERENCE.md](dev/CHIPQA_REFERENCE.md).

See [CLI Usage](CLI_USAGE.md) for options and [Metrics & Formats](METRICS_AND_FORMATS.md) for compatibility.

### ColorVideoVDP's conflicting FFmpeg package metadata

The pinned upstream ColorVideoVDP v0.5.7 declares both `ffmpeg-python>=0.2.0`
and `ffmpeg>=1.4`, although its video reader uses `ffmpeg-python`'s
`ffmpeg.input` / `ffmpeg.probe` / `ffmpeg.run_async` API. The separate PyPI
`ffmpeg` distribution exposes the same `ffmpeg` module and can break this API.
PixVQA therefore builds the **same pinned upstream source** with only the
conflicting `ffmpeg>=1.4` packaging requirement removed. It does not edit the
upstream Git checkout or change scoring code. The learned installer repairs
existing venv installations, installs missing actual requirements
(`matplotlib`, `PyEXR`), validates the `ffmpeg-python` API and requires
`python -m pip check` to pass. This metadata correction does not install or
replace the FFmpeg command-line executable.
