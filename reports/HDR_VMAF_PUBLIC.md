# Public HDR VMAF v1.1.8 — implementation report

Worktree: `D:\projects\pixvqa-public-hdr-vmaf`
Branch: `feature/hdr-vmaf-v1-models`
Base: public `main` at `10574f6` (`Merge pull request #2 from pixop/docs/readme-badges`)
Status: implemented and validated. Not committed, not pushed, not merged.

Recommendation: **CONDITIONAL GO** for merging this branch into public `main` by itself.

## 1. What landed

The validated CPU HDR VMAF path from the internal worktree `D:\projects\pixop-vqa-hdr-vmaf` was adapted onto public PixVQA. The public CLI stays `--metric-device cpu|gpu|auto`. There is no `--vmaf-backend`.

HDR models require explicit complete colorimetry on both the reference and the distorted input. Explicit means a stream tag or a CLI override (`HdrSource::Stream` or `HdrSource::Override`) on all four fields. Fields filled by `hdr_resolve` are not accepted. Ten-bit input alone does not imply PQ. No tone mapping is applied.

The only accepted tuple is:

- transfer PQ (SMPTE ST 2084)
- primaries BT.2020
- matrix BT.2020 nonconstant luminance (`bt2020nc`)
- range limited
- pixel format YUV 4:2:0, 10-bit
- CPU libvmaf

That tuple becomes the libvmaf picture color string `limited:bt2020:pq:bt2020nc` on each side. libvmaf and zimg perform the model-directed conversion to 16-bit 4:2:0 ICtCp. A conversion failure is fatal.

Rejected, with an error:

- missing or partial colorimetry
- HLG
- full-range PQ
- an SDR model combined with an explicit PQ or HLG transfer
- `--metric-device gpu` when VMAF is the only selected metric

The last rejection is the behavior already on public `main`. This branch does not implement the CPU fallback from `feature/vmaf-device-fallback`. When VMAF is selected together with a CUDA-capable metric, the existing notice remains: VMAF stays on the CPU. That notice is not a new fallback and it does not score HDR on CUDA.

SDR models still leave picture color unset. Their JSON bytes are the v3.2.1 files. Score clipping stays the model's own `[0,100]` or `[0,110]`. Scores are not rescaled. Per-frame and pooled mean, harmonic mean, min, and max are unchanged.

No new CLI flag was added. Public PixVQA already has `--hdr-transfer`, `--hdr-primaries`, `--hdr-matrix`, and `--hdr-range`, including the reference and distorted variants. A second colorimetry mechanism would have duplicated that parser.

## 2. Modified-file inventory

26 tracked files changed (`git diff --stat`: 719 insertions, 152 deletions) plus the new files below. No commit was created.

Tracked:

- `.gitattributes` — `model/**/*.json` and the HDR model directories are `eol=lf`, so a checkout does not rewrite the SHA-256 bytes
- `README.md` — libvmaf badge `74f930d`, HDR CLI example
- `THIRD_PARTY_NOTICES.md`, `third_party/netflix-vmaf/NOTICE`, `docs/LICENSING.md` — pin, previous tag v3.2.1, zimg 3.0.6 (WTFPL)
- `build.sh` — comment for the new pin
- `docs/INSTALL.md` — build pin, nested model install, verification split between the old v3.2.1 run and this branch
- `include/pixvqa/types.hpp`, `include/pixvqa/vmaf.hpp`
- `src/main.cpp`, `src/vmaf.cpp`
- `model/README.md`
- `reports/VMAF_BUILD.md`, `reports/VMAF_MODEL_CHECKSUMS.md`
- `scripts/build_libvmaf.sh`, `scripts/libvmaf_flags.sh`, `scripts/build_vmaf_ref_oracle.sh`, `scripts/install.sh`
- `scripts/verify_vmaf_model_checksums.py`
- `scripts/qa/harness.py`, `scripts/qa/inventory.py`, `scripts/test_cli_help.py`
- `scripts/test_vmaf_8bit.py`, `scripts/test_vmaf_numeric.py`, `scripts/test_vmaf_numeric_suite.py`, `scripts/test_vmaf_smoke.py` — default prefix `deps/libvmaf-74f930d`

New, untracked:

- `model/hdr_vmaf_v1.1.8/` — four official JSON files
- `model/hdr_vmaf_v1.1.8_hfr/` — four official JSON files
- `scripts/build_zimg.sh`
- `scripts/build_libvmaf_3.2.1.sh` — previous public recipe, prefix `deps/libvmaf-3.2.1`
- `scripts/fetch_hdr_vmaf_models.py`
- `scripts/test_vmaf_hdr.py`
- `reports/HDR_VMAF_PUBLIC.md` — this file

`scripts/install.sh` still copies the eight top-level SDR JSON files. It also copies the two HDR directories. A temp install to `/tmp/pixvqa-hdr-inst` contained all 16 JSON files.

SDR `model/vmaf_v1.0.16*.json` content is unchanged. The Windows checkout had CRLF working copies, which changed the on-disk SHA-256 while `git diff` stayed empty under `core.autocrlf`. Those eight files were rewritten to LF. `git hash-object` of `model/vmaf_v1.0.16_3d0h.json` is `9bdebbfdee470e6db1563811baa87151e0b60534`, equal to `HEAD`. They are not in `git status`.

Untouched on purpose:

- `D:\projects\pixvqa-public` (`main`, clean, `10574f6`)
- `D:\projects\pixvqa-public-vmaf-fallback` (`feature/vmaf-device-fallback`, its existing uncommitted work is still there)
- `D:\projects\pixop-vqa-hdr-vmaf` (read only)
- CUDA metric kernels

## 3. Dependency build

| Piece | Pin |
|---|---|
| libvmaf source | `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` |
| Previous tag, still buildable | v3.2.1, `f85a853692a8c730d0270cd733c8bb30b5b93b7c`, via `scripts/build_libvmaf_3.2.1.sh` |
| zimg | 3.0.6, tag `release-3.0.6`, commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd` |
| Meson | `-Denable_zimg=true -Denable_cuda=false -Denable_float=false -Ddefault_library=static -Denable_tools=true` |
| Prefixes | `deps/libvmaf-74f930d`, `deps/zimg-3.0.6` (gitignored) |
| Source checkouts | `deps/src/vmaf-74f930d`, `deps/src/zimg` |

This commit is not a tagged release newer than v3.2.1. Meson reports project version 3.2.0. `deps/libvmaf-74f930d/bin/vmaf --version` prints `74f930d`. The stamp file `pixvqa-libvmaf-stamp.txt` records the commit, the previous tag, zimg 3.0.6, and the Meson flags.

`scripts/libvmaf_flags.sh` refuses a prefix whose `picture.h` has no `vmaf_picture_convert`, and it refuses a pkg-config link line without `-lzimg`. A leftover `deps/libvmaf-3.2.1` is not linked.

Observed link line for `build/cpu/pixvqa`:

```
-pthread -I.../deps/libvmaf-74f930d/include -I.../include/libvmaf -I.../deps/zimg-3.0.6/include
-DPIXVQA_LIBVMAF_COMMIT=74f930d17de8c58e7c2ce55e2b897b0030ebd8ef -DPIXVQA_ZIMG_VERSION=3.0.6
-lvmaf -pthread -lm -lzimg -lstdc++
```

`nm` on `build/cpu/pixvqa` shows `vmaf_picture_convert`, `vmaf_picture_convert_context_init`, and `vmaf_picture_convert_context_close` as defined symbols. The same binary contains the commit string `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` and `3.0.6`. Static library and the installed headers come from that one prefix. `pkg-config --modversion libvmaf` is `3.2.0`. `pkg-config --libs --static libvmaf` includes `-lzimg`.

Host tools on this machine: conda-forge gcc/g++ 16.2.0 from `/home/sergio/gcc-env/bin`. GNU make is required on `PATH` by `scripts/build_zimg.sh`. The default WSL `PATH` did not have it; the library build used `/home/sergio/ffmpeg-build-tools/bin/make`. Autotools were not installed. `build_zimg.sh` downloaded the Ubuntu 24.04 autoconf, automake, libtool, m4, and autotools-dev debs into gitignored `deps/src/host-debs` and checked the SHA-256 values stored in the script.

CPU binary: `build/cpu/pixvqa`, `pixvqa version 0.3.0-dev`, `decode: none`.

CUDA rebuild was not run. `nvcc` is not installed in this WSL. `nvidia-smi` is present at `/usr/lib/wsl/lib/nvidia-smi`. No CUDA kernel file is in the diff.

## 4. HDR numerical parity

Oracle: `deps/libvmaf-74f930d/bin/vmaf` from the same static build. Tolerance `1e-4`. Both sides used `--threads 8` and the same color strings (`limited` / `bt2020` / `pq` / `bt2020nc`). Feature dump uses `PIXVQA_VMAF_FEATURES=1` (`pix_getenv` always prefixes `PIXVQA_`).

`scripts/test_vmaf_hdr.py` result: **27 scored cases, 0 failures, worst absolute difference `4.99989e-7`**. Pooled VMAF mean, harmonic mean, min, and max matched the CLI's printed values (`score_abs` 0 on every case). The worst difference is in CAMBI, SpEED chroma, ADM3, or Motion3, which is the six-decimal CLI quantization.

Synthetic, 8 frames, identical / blur / resample:

| Geometry | Models | Clip observed |
|---|---|---|
| 1920×1080 | `3d0h`, `5d0h`, `hfr_3d0h`, `hfr_5d0h` | `[0,100]` |
| 3840×2160 | `1d5h_2160`, `3d0h_2160`, `hfr_1d5h_2160`, `hfr_3d0h_2160` | `[0,100]` except `3d0h_2160` and `hfr_3d0h_2160`, which are `[0,110]` |

Identical-frame `3d0h_2160` mean was `107.440841`. Identical-frame `hfr_3d0h_2160` mean was `109.531811`. Both sit above 100 and inside the model clip.

Real HEVC, one source, 60 frames (source frames 1800–1859):

- Master: `BasketballGame_3840x2160_5994_420_10b_pq_bt2020l_29m25s_3600fr.yuv` (filename declares PQ, BT.2020, limited, 59.94 fps)
- Encode: libx265, preset fast, CRF 28, `colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited`, rate `60000/1001`, then decoded to raw `yuv420p10le`
- `hfr_3d0h_2160` mean `94.852236`, worst feature abs `4.998e-7`, clip 110
- `3d0h_2160` mean `91.299559`, worst feature abs `4.998e-7`, clip 110
- 1080p is a Lanczos downscale of those raw frames, not a native 1080 master. `hfr_3d0h` mean `95.002617`, worst feature abs `4.920e-7`, clip 100

The internal 48-case real-content set was not re-run here. Sources other than this BasketballGame slice, other CRFs, native 1080, and 50 fps / 60 fps masters were not executed in public PixVQA. HFR behavior in this run comes from the HFR model JSON (five-frame motion window and moving average). Neither pixvqa nor the official CLI was given a frame-rate flag; raw YUV scoring walks frames in order.

Rejections, all exit code 2, all matched the expected text:

| Case | Needle |
|---|---|
| no color flags | explicit PQ transfer |
| `--hdr-transfer pq` only | explicit complete colorimetry |
| HLG | unsupported and unvalidated |
| `--hdr-range full` | full-range PQ |
| SDR `vmaf_v1.0.16_3d0h` plus PQ | rejects explicit HDR |
| `--metrics vmaf --metric-device gpu` | no GPU |

Machine detail JSON is gitignored at `reports/hdr_vmaf_public_parity.json`.

## 5. SDR regression

All of these passed on `build/cpu/pixvqa` linked to the `74f930d` library.

| Script | Result |
|---|---|
| `scripts/test_vmaf_errors.py` | 13 checks, including VMAF-only `--metric-device gpu` still rejected on this CPU build (`no GPU metrics`) |
| `scripts/test_vmaf_smoke.py` | 3 frames, 320×240, 10-bit, `3d0h`. Score abs `0`. Feature abs `3.789e-7` |
| `scripts/test_vmaf_numeric.py` | 8 SDR models, identical frames. Score abs `0`. Largest feature abs `3.505e-7`. Pooled `100`, except `3d0h_2160` `108.879197` and `hfr_3d0h_2160` `110` |
| `scripts/test_vmaf_8bit.py` | 5 frames, 320×240, 8-bit. CLI abs `3.526e-7`. C-API oracle abs `0` |
| `scripts/test_vmaf_numeric_suite.py` | 36 cases (temporal, unclipped, pools, near-zero). Pool abs max: mean `4.888e-7`, harmonic mean `4.997e-7`, min `4.748e-7`, max `4.960e-7` |

`scripts/build_vmaf_ref_oracle.sh` rebuilt `build/cpu/vmaf_ref_oracle` against the same prefix before the 8-bit and suite runs. libvmaf printed `est_params: covariance matrix is singular` during the 8-bit oracle; the scores still matched.

`bash build.sh` on this tree: all selftests passed, all IO regressions passed, golden-log compatibility passed, production-parser compatibility passed.

`python3 scripts/run_qa.py --profile smoke`: `PASS 16  FAIL 0  SKIP 0`.

Model checksums: `scripts/verify_vmaf_model_checksums.py` downloaded each upstream file. **16 of 16 exact SHA-256 matches.** SDR files against `f85a853` (v3.2.1). HDR files against `74f930d`. Table: `reports/VMAF_MODEL_CHECKSUMS.md`.

## 6. Public CLI examples

SDR, unchanged. An SDR model is refused when the effective transfer is PQ or HLG.

```bash
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

HDR, validated limited-range PQ BT.2020. The same four flags apply to both inputs. Per-side flags exist when the two inputs differ. The score keeps the model clip.

```bash
pixvqa --metrics vmaf --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json --threads 8 \
  --hdr-transfer pq --hdr-primaries bt2020 --hdr-matrix bt2020nc --hdr-range limited \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

4K model with the `[0,110]` clip:

```bash
pixvqa --metrics vmaf --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json --threads 8 \
  --hdr-transfer pq --hdr-primaries bt2020 --hdr-matrix bt2020nc --hdr-range limited \
  -r reference10.yuv -s 3840x2160 -f 420p:10 \
  -t distorted10.yuv -s 3840x2160 -f 420p:10
```

Metadata adds one line. `backend` is `cpu`. `conversion` comes from the model JSON.

```
VMAF-EXT: commit=74f930d17de8c58e7c2ce55e2b897b0030ebd8ef zimg=3.0.6 backend=cpu conversion=... display=... hfr_temporal=... color_ref=limited:bt2020:pq:bt2020nc color_dist=limited:bt2020:pq:bt2020nc
```

`--metric-device` is unchanged: `cpu`, `gpu`, or `auto`. VMAF alone with `gpu`, including an HDR model, runs CPU libvmaf and prints `pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf.` The summary reports `VMAF: cpu libvmaf`.

## 7. Installation and compatibility

| Check | Result |
|---|---|
| CPU-only `bash build.sh` | `build/cpu/pixvqa`, selftest, IO, golden-log, production parser passed |
| Library load | static `libvmaf.a` plus `libzimg.a` from the `74f930d` prefix; `vmaf_picture_convert` is inside the binary |
| Headers | `deps/libvmaf-74f930d/include/libvmaf/picture.h` declares `vmaf_picture_convert` |
| Previous recipe | `scripts/build_libvmaf_3.2.1.sh` remains; current flags refuse that prefix |
| Model SHA | 16/16 match |
| `scripts/install.sh` | installed the CPU binary and all 16 model JSON files under a temp prefix |
| Smoke QA | 16 pass, 0 fail, 0 skip |
| Public CLI | `--metrics`, `--model`, `--metric-device`, `--decode`, existing HDR flags. No `--vmaf-backend` |
| CUDA binary | not rebuilt; `nvcc` is absent in this WSL; kernels were not edited |
| Python / PyTorch | no new dependency |

`docs/INSTALL.md` keeps the earlier v3.2.1 smoke result labeled as the `main` run. This branch's results are the tables above.

## 8. Performance baseline

CPU: Intel Core Ultra 7 265K, 20 processors visible to WSL. The benchmark passed `--threads 8`. Two repeats. `PIXVQA_PROFILE=1` times input reading and libvmaf separately. Wall time is process time around that run and includes model load.

The 4K model is `hdr_vmaf_v1.1.8_hfr_3d0h_2160`. Its conversion target is limited BT.2020 PQ to 16-bit 4:2:0 ICtCp at display height 2160. The 1080 model is `hdr_vmaf_v1.1.8_hfr_3d0h`, display height 1080. Both sides used `limited:bt2020:pq:bt2020nc`. Inputs were the raw YUV pair from section 4, on `/mnt/d` (a Windows filesystem through WSL).

| Run | Frames | Wall | End-to-end fps | Input read | libvmaf compute | Compute fps |
|---|---:|---:|---:|---:|---:|---:|
| 3840×2160, repeat 1 | 60 | 5.098 s | 11.77 | 1.276 s | 3.660 s | 16.39 |
| 3840×2160, repeat 2 | 60 | 4.821 s | 12.45 | 1.022 s | 3.644 s | 16.47 |
| 1920×1080, repeat 1 | 60 | 1.190 s | 50.44 | 0.253 s | 0.880 s | 68.19 |
| 1920×1080, repeat 2 | 60 | 1.213 s | 49.45 | 0.211 s | 0.934 s | 64.24 |

The frame loop reads the reference frame and the distorted frame, then calls `vmaf.push`. Those stages do not overlap. That is the same serial input-reading structure as the internal adapter. On this run the 4K read is about 1.0–1.3 s of a 4.8–5.1 s wall, a larger share than the internal benchmark, consistent with reading the pair through the WSL mount. No threaded reader was added.

The internal benchmark (120 frames, about 13.4–13.8 fps for PixVQA and 18.3–18.8 fps for the CLI at 4K) was not repeated on this public binary. This table is the public baseline only.

## 9. Compatibility with `feature/vmaf-device-fallback`

The fallback branch is `D:\projects\pixvqa-public-vmaf-fallback`, still at `10574f6`, with its own uncommitted edits. This HDR work did not read-write that tree except to inspect `src/main.cpp`.

Intended fallback behavior, already implemented there and not reimplemented here:

- VMAF plus `--metric-device gpu` runs CPU libvmaf
- stderr warning: `VMAF CUDA computation is unavailable; using CPU libvmaf`
- other CUDA-capable metrics stay on CUDA
- GPU decode stays independent of the VMAF compute backend

The integration applies that fallback to SDR and HDR VMAF. A CPU-only binary accepts a VMAF-only GPU request and uses CPU libvmaf. The warning names CPU libvmaf and does not describe a CUDA HDR score. HLG, full-range PQ, and incomplete colorimetry stay errors.

The two branches both edit `src/main.cpp` (help text and device resolution), `include/pixvqa/types.hpp`, `README.md`, `.gitattributes`, and `scripts/qa/inventory.py`. The fallback branch also edits `scripts/test_vmaf_errors.py`. It does not edit `src/vmaf.cpp`. HDR color checks can sit beside the fallback device policy.

Recommended order:

1. Land `feature/vmaf-device-fallback` onto `main` first, including its warning and the CPU-only-build exception for a VMAF-only GPU request.
2. Rebase `feature/hdr-vmaf-v1-models` onto that result.
3. In `src/main.cpp`, keep the fallback device policy and keep the HDR color policy. VMAF, including an HDR model, stays on CPU libvmaf. The warning must keep saying CPU libvmaf. HLG, full-range PQ, and incomplete colorimetry stay errors. There is still no CUDA HDR path.
4. Keep SDR and HDR model files as LF. The fallback worktree currently shows the eight SDR JSON files as modified because of CRLF. The git blob is unchanged. A combined tree should keep the LF bytes so the known-model SHA-256 table matches the file PixVQA reads.

Landing HDR first is possible. The fallback rebase would then have to replace the VMAF-only GPU error without deleting the HDR color checks. Doing both in one merge mixes those two `main.cpp` edits and is the riskier order.

## 10. Recommendation

**CONDITIONAL GO** to merge `feature/hdr-vmaf-v1-models` into public `main` as its own change.

The CPU transfer meets the authorized bar:

- 27 HDR scored cases against the official CLI from the same libvmaf/zimg build, worst abs `5.0e-7`, plus 6 rejection cases
- SDR error, smoke, eight-model numeric, 8-bit C-API, and 36-case suite all passed
- 16 model files match upstream SHA-256
- the public CLI, SDR defaults, and the VMAF-only GPU rejection are intact
- the binary loads libvmaf `74f930d` with zimg 3.0.6, and v3.2.1 remains a separate recipe

Conditions, all already true of this worktree:

- Merge this branch alone. Integrate `feature/vmaf-device-fallback` afterwards, in the order in section 9. Do not treat CPU libvmaf as a GPU HDR score.
- The public real-content retest is 3 cases on one 60-frame BasketballGame CRF 28 pair, plus a Lanczos 1080p derivative. It does not repeat the internal 48-case set.
- The CUDA pixvqa binary was not rebuilt here. CUDA kernels were not modified. A later CUDA build will pick up the new static libvmaf through `scripts/libvmaf_flags.sh` and still has no HDR VMAF on CUDA.
- The serial input read remains. The numbers in section 8 are a baseline. A threaded reader needs a separate authorization.

No merge, rebase, commit, or push was done.
