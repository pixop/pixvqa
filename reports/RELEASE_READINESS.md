# Release readiness

Recommendation: **GO** for the release candidate on `feature/hdr-vmaf-remote-reconcile`. PR #3 stays open, unmerged, and private. The default branch is `main`.

Parent of this review: `6f6db50949b23c724f0cc65cf51456c7a8f61a20`. No metric, QCS formula, or HDR path was changed.

## README

Compressed examples now use caller-supplied names and no `-s` or `-f`:

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim -r ref_hevc.mp4 -t dist_hevc.mp4
pixvqa --metrics psnr,xpsnr,ssim,msssim -r ref_av1.mp4 -t dist_av1.mp4
pixvqa --metrics psnr,xpsnr,ssim,msssim -r ref_vvc.mp4 -t dist_vvc.mp4
```

The README states that those names belong to the caller. The files are gitignored and were not added. QCS text, thresholds, and references are unchanged. NVDEC is stated as HEVC Main 10 4:2:0 and AV1 8-bit or 10-bit 4:2:0. `--decode gpu` rejects other codecs, including VVC.

## Commands run on `bin/pixvqa`

The installed binary reports `build: cuda` and `decode: software nvdec`. Each command used `-v 0` and the metrics `psnr,xpsnr,ssim,msssim`. All three exited 0. Each stream is 960×540, 10-bit 4:2:0, 24 frames, 24 fps. XPSNR selected `24/1`.

| Command | Decode | Summary |
|---|---|---|
| `-r ref_hevc.mp4 -t dist_hevc.mp4` | software `hevc` | PSNR 35.79, SSIM 0.917018, MS-SSIM 0.964386, XPSNR 31.3503 |
| same HEVC pair with `-s 960x540 -f 420p:10 --fps 24` | software `hevc` | same four summaries |
| `-r ref_av1.mp4 -t dist_av1.mp4` | `av1_cuvid` | PSNR 40.67, SSIM 0.968935, MS-SSIM 0.989121, XPSNR 36.1826 |
| same AV1 pair with `--decode gpu` | `av1_cuvid` | same four summaries |
| `-r ref_vvc.mp4 -t dist_vvc.mp4` | software `vvc` | PSNR 35.06, SSIM 0.916946, MS-SSIM 0.963445, XPSNR 31.2858 |

AV1 used NVDEC because this `libavcodec` has no libdav1d. `scripts/test_cli_help.py` and `scripts/check_readme_examples.py` then passed on generated fixtures, including the three new commands. The GPU README example passed in `test_cli_help.py`.

## Hardware QA

`python3 scripts/run_qa.py --profile hardware` on this machine: **2 pass, 1 fail**.

| Case | Result |
|---|---|
| `gpu-metrics` | PASS. CUDA PSNR, SSIM, and MS-SSIM matched the CPU text within the existing tolerance. VMAF stayed on CPU libvmaf. |
| `gpu-repeat` | PASS. Three sequential GPU PSNR runs matched. |
| `hardware-av1-hevc-vvc` | FAIL. Not a CUDA score mismatch. |

That case runs `scripts/test_av1_vvc.py` with `PIXVQA_BIN=build/swdec/pixvqa` and `PIXVQA_RUN_GPU=1`. The software binary's FFmpeg has no libdav1d, so AV1 software decode stops with `AV1 software decoding requires libdav1d or libaom-av1`. VVC is `FAIL` because `tests/fixtures/vvc/bbb_360p.vvc` is absent on purpose. The same run's NVDEC checks passed: AV1 8-bit, AV1 10-bit, and HEVC 10-bit.

This is not a release-blocking product defect. The installer build used by GitHub Actions links dav1d, and full CPU QA on `81d7108` passed AV1. The failure is not suppressed.

GPU Developer reproduction, from the repository root, with `bin/pixvqa`, `build/swdec/pixvqa`, `build/full/pixvqa`, ffmpeg, and an NVIDIA GPU:

```bash
python3 scripts/run_qa.py --profile hardware
```

## Licensing and hygiene

PixVQA is MIT. `docs/LICENSING.md` and `THIRD_PARTY_NOTICES.md` keep libvmaf, zimg, the VMAF models, FFmpeg, dav1d, and PU21 on their own terms. Model JSON files are vendored Netflix bytes under BSD-2-Clause-Patent, as stated in `model/README.md`. No credentials, `.env` files, or sample videos are tracked. `*.mp4` is gitignored.

Tracked reports and scripts do not name a home directory. Gitignored measurement JSON under `reports/` can still contain a local absolute path and is not source. This review does not change repository visibility.

## CI

Hosted results already on this branch:

| Check | Commit | Result |
|---|---|---|
| [QA](https://github.com/pixop/pixvqa/actions/runs/40687395442) | `6f6db50` | pass |
| [Clean installation](https://github.com/pixop/pixvqa/actions/runs/40687395439) | `6f6db50` | pass, Ubuntu 22.04 and 24.04 |
| [Full CPU QA](https://github.com/pixop/pixvqa/actions/runs/40660016002) | `81d7108` | pass, 234 / 0 / 2 optional skips |

Full CPU QA was not repeated for the documentation-only commit `6f6db50`. GPU CI remains manual and has no self-hosted runner. Local GPU qualification is the separate record in `reports/GPU_NVDEC_RELEASE.md`: 50/50, plus the hardware-profile failure classified above.

## Limits that remain

- Software AV1 requires libdav1d or libaom-av1 in the linked FFmpeg. This local shared library has neither.
- `--decode gpu` does not decode VVC, 8-bit HEVC, 4:2:2, or codecs other than HEVC Main 10 4:2:0 and AV1 8-bit or 10-bit 4:2:0.
- VMAF stays on CPU libvmaf.
- The public VVC check still depends on a fixture that is not stored in the repository.
