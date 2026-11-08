# Learned Metrics: Advanced Setup Reference

This page is for custom deployments and troubleshooting. Most users only need [Optional Learned Metrics](../LEARNED_METRICS.md).

## Python environments and model assets

The guided installer is `install_pixvqa.sh` (choose learned metrics when prompted; the default is no). For non-interactive setup, set `PIXVQA_ENABLE_LEARNED=1`. Alternatively, run `bash scripts/install_learned_metrics.sh` after building PixVQA.

The installer creates `.venv` using CPython 3.12 for CVVDP, CompressedVQA-HDR NR, SimpleVQA, and HDR-ChipQA feature extraction. HDR-ChipQA additionally uses `.venv-chipqa-inference` with CPython 3.10 and scikit-learn 1.0.2 for its pretrained SVR. A private CPython 3.12 may be provisioned using micromamba/Conda when necessary; provide `PYTHON=/path/to/python3.12` to choose an existing interpreter.

It clones or reuses pinned upstream repositories under `deps/`, verifies existing weights, downloads missing weights (CompressedVQA-HDR NR from Dropbox, SimpleVQA UGC from Google Drive, SlowFast R50 backbone from its upstream distribution), and merges `config/learned_metrics.local.toml` without overwriting custom entries. HDR-ChipQA SVR and scaler files are supplied by its upstream repository. Downloads are hash-verified, and the installer fails when required assets cannot be retrieved. Consult upstream licenses before redistributing model assets. The installer does not replace system Python/CUDA/driver or remove package caches.

The complete version-pinned Python requirements are in [`python/requirements/learned-env.txt`](../../python/requirements/learned-env.txt), and installer-specific dependency repair is in `scripts/setup_hdr_chipqa_envs.sh` and related scripts. These are the authoritative package lists; do not maintain a duplicate pinned list in documentation.

## Custom paths and overrides

| Variable | Default | Purpose |
|---|---|---|
| `PIXVQA_LEARNED_ENV_DIR` | `<repo>/.venv` | Shared Python environment (absolute path) |
| `PIXVQA_CVVDP_SOURCE` | `<repo>/deps/ColorVideoVDP` | Pinned ColorVideoVDP checkout |
| `PIXVQA_LEARNED_VENV_USE_SYMLINKS` | unset | Allow symlink-based venv instead of copies on WSL mounts |
| `PIXVQA_HDR_CHIPQA_INFERENCE_ENV_DIR` | `<repo>/.venv-chipqa-inference` | HDR-ChipQA SVR environment |
| `PIXVQA_HDR_CHIPQA_FEATURE_PYTHON` | shared `.venv` | Alternate feature-extraction interpreter |
| `PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON` | unset | Alternate Python 3.10 for inference venv |
| `PIXVQA_LEARNED_INSTALL_OFFLINE` | unset | Verify existing local assets without network |

`config/learned_metrics.local.toml` is local and Git-ignored. The installer preserves custom Python and repository paths. If `[learned].python` differs from the chosen venv, it warns: align the config with the interpreter actually used. Older `$HOME/pixvqa-learned-env` locations remain usable via `PIXVQA_LEARNED_ENV_DIR`.

### WSL2 and mounted Windows drives

On `/mnt/c` or `/mnt/d`, Windows-backed (9p/drvfs) mounts may not handle venv symlinks normally, so the installer creates the environment with `python -m venv --copies` by default. Linux ext4 checkouts generally offer better I/O performance. Use `PIXVQA_LEARNED_VENV_USE_SYMLINKS=1` only when symlinks are supported. Environments should be recreated when moving a repository; do not commit `.venv`.

### Manual installation

```bash
scripts/setup_learned_env.sh /absolute/path/to/.venv /absolute/path/to/ColorVideoVDP
cp config/learned_metrics.example.toml config/learned_metrics.local.toml
# Set the repository, checkpoint and Python paths in the local config.
```

### ColorVideoVDP upstream FFmpeg packaging issue

Pinned CVVDP v0.5.7 declares both `ffmpeg-python>=0.2.0` and the conflicting `ffmpeg>=1.4` Python distribution, although it needs the `ffmpeg-python` API. PixVQA rebuilds the identical pinned source with only the conflicting dependency declaration removed, retains `ffmpeg-python`, installs the genuine dependencies (`matplotlib`, `PyEXR`) and validates the environment with `pip check`. This does not change the scoring code or the FFmpeg executable.
