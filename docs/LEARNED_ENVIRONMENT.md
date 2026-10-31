# Learned-metric environment

`cvvdp` and `cvqa-hdr-nr` run in one Python environment with one CUDA build of
PyTorch. The pinned specification is
[`python/requirements/learned-env.txt`](../python/requirements/learned-env.txt);
[`scripts/setup_learned_env.sh`](../scripts/setup_learned_env.sh) creates it.

```bash
scripts/setup_learned_env.sh ~/pixvqa-learned-env ~/src/ColorVideoVDP
cp config/learned_metrics.example.toml config/learned_metrics.local.toml
```

Then point `[learned] python` at `~/pixvqa-learned-env/bin/python`; see
[learned-metric config](LEARNED_METRICS_CONFIG.md).

The same environment runs both metrics on the CPU (`--device cpu`) and on
CUDA. Device selection, CPU-vs-CUDA tolerances, timings, and platform status
(WSL2 verified, native Windows not verified end to end) are in
[LEARNED_DEVICES.md](LEARNED_DEVICES.md).

## What it contains

| Component | Pin |
|---|---|
| Python | CPython 3.12, `venv` (validated with 3.12.3 from `/usr/bin/python3.12`) |
| PyTorch | `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `triton==3.6.0`, CUDA 12.8 runtime wheels |
| Shared | `numpy==2.5.2` |
| CompressedVQA-HDR NR | `open_clip_torch==3.1.0`, `timm==1.0.19`, `opencv-python==4.11.0.86`, `pillow==11.3.0`, and their runtime dependencies |
| ColorVideoVDP | `cvvdp==0.5.7` from commit `166f5ae95ecbd841ae727984cd237de213875399`, plus `scipy`, `ffmpeg-python`, `future`, `imageio`, `einops`, `nvidia-ml-py` |

The PyTorch wheels come from the `cu128` index, which includes `sm_120` kernels
for the RTX 50 series. Everything else comes from PyPI. The lock pins all 57 installed
distributions other than `cvvdp` (pinned by commit) and `pip`. There is no second
PyTorch, and nothing for notebooks, training, plotting, or building.

The setup script:

1. creates a `venv` (refuses an existing directory);
2. installs `torch` and `torchvision` from the `cu128` index, constrained by the lock;
3. installs the rest of the lock from PyPI;
4. exports commit `166f5ae` of the given ColorVideoVDP checkout with `git archive`
   and installs it with `--no-deps`. The checkout is only read. pip's own
   `git+file://` install cannot fetch a pinned commit from a shallow checkout;
5. writes the commit to `<env>/COLORVIDEOVDP_COMMIT`;
6. runs `pip check`.

`pip check` reports three requirements that `cvvdp`'s `setup.py` declares but
that its video scoring path does not use: `ffmpeg` (a PyPI placeholder, not the
FFmpeg program), `matplotlib` (plots), and `pyexr` (EXR images). The script
accepts exactly those three and fails on any other unmet requirement. The
previous CVVDP environment did not have `pyexr` either.

The adapters are not installed. The config's `adapter_path` puts the checkout's
`python/` on `PYTHONPATH`, so adapter edits need no reinstall. CVVDP decodes
compressed video with the `ffmpeg`/`ffprobe` executables on `PATH`; the config's
`[learned] path` supplies them.

The run record's `runtime.cvvdp.commit_verified` is `false`, as it was for the
previous editable install, because the package metadata does not carry a VCS
commit. The commit is pinned by the setup script and recorded in
`COLORVIDEOVDP_COMMIT`.

## Differences from the previous two environments

The two environments it replaces were a conda environment for CVVDP and a
`venv` for NR. Both had the same `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`,
`numpy==2.5.2`, and CUDA wheels.

| Package | CVVDP env | NR env | Shared env | Reason |
|---|---|---|---|---|
| Python | 3.12.15 (conda) | 3.12.3 | 3.12.3 | System interpreter, as NR |
| `pillow` | 12.3.0 | 11.3.0 | 11.3.0 | NR preprocessing was validated with 11.3.0. CVVDP decodes video through FFmpeg, not Pillow |
| `huggingface_hub`, `hf-xet`, `filelock`, `fsspec`, `networkx` | older | newer | NR versions | No numerical role. Weights are loaded from the local checkpoint |
| `matplotlib` and its dependencies | present | absent | absent | Plotting only |
| `wheel` / `meson`, `ninja` | `wheel` | `meson`, `ninja` | absent | Build tools |
| `cvvdp` | editable checkout | absent | `git archive` of the same commit, not editable | Does not depend on the working tree |

## Numerical parity

Validated on an RTX 5080 (WSL2) with BasketballGame 1080p, PQ/BT.2020 10-bit,
120 frames at 59.94 fps: reference `crf20`, test `crf40`.

| Value | Previous environments | Shared environment |
|---|---|---|
| CVVDP JOD | `7.415657997131348` | `7.415657997131348` |
| CVVDP per-frame distortion and JOD-equivalent (120 frames) | | identical, every frame |
| CVVDP QCS [0, 10] | `74.247569` | `74.247569` |
| NR `quality_score` (crf40) | `4.444974268667124` | `4.444974268667124` |
| NR `raw_model_score` | `-1.1983554363250732` | `-1.1983554363250732` |
| NR calibrated samples | `4.2378421984481065`, `4.6480186687712148` | identical |
| NR QCS [0, 10] | `44.429304` | `44.429304` |

The comparisons are exact equality of the full-precision JSON values. They hold
for each metric alone, for `--metrics cvvdp,cvqa-hdr-nr --qcs`, and for
`--preset hdr --metrics psnr,pu21-psnr,cvvdp,cvqa-hdr-nr --qcs`. CVVDP with
`--features` (used for QCS) gives the same JOD as without it. No numerical
difference was observed.
