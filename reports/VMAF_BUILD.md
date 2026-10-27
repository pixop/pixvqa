# libvmaf commit 74f930d build and link strategy

## Pin

| Item | Value |
|---|---|
| Repository | https://github.com/Netflix/vmaf |
| Commit | `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` |
| Release tag | none. This is not a newer tagged libvmaf. The previous tag is v3.2.1 |
| Previous tag | `v3.2.1`, commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`, script `scripts/build_libvmaf_3.2.1.sh` |
| `vmaf --version` | `74f930d` |
| Meson project version | `3.2.0` |
| zimg | 3.0.6, commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd`, prefix `deps/zimg-3.0.6` |

`scripts/build_libvmaf.sh` verifies `git rev-parse HEAD` equals the pinned commit before configure, builds zimg, and refuses the install unless `-lzimg` and `vmaf_picture_convert` are present.

## Obtain / build

```bash
# Requires git, meson, ninja, nasm, pkg-config, gcc, g++, GNU make, and perl.
bash scripts/build_libvmaf.sh
```

Install prefix: `deps/libvmaf-74f930d` (gitignored). Source checkout: `deps/src/vmaf-74f930d`.

Meson options:

```text
--buildtype=release
-Denable_tests=false
-Denable_docs=false
-Denable_tools=true
-Denable_cuda=false
-Denable_zimg=true
-Dbuilt_in_models=true
-Denable_float=false
-Ddefault_library=static
```

CUDA is never enabled. The official `vmaf` CLI from the same build is the numerical oracle.

## Link into PixVQA (optional)

`scripts/libvmaf_flags.sh` detects the prefix (`lib` or `lib64`) and exports static `pkg-config` flags.

**libvmaf is optional.** `build.sh`, `build_cuda.sh`, and `build_decode.sh` enable
`-DPIXVQA_HAS_VMAF=1`, compile `src/vmaf.cpp`, and link `libvmaf.a` only when
`deps/libvmaf-74f930d` is present and passes the zimg checks. A v3.2.1 prefix is refused. Without the 74f930d prefix, PixVQA builds the
dependency-free CPU configuration; `vmaf` in `--metrics` is then rejected at runtime.

Requiring libvmaf for every build would regress the previously supported
dependency-free CPU tree. The minimal fix (applied) is optional detection rather
than a hard build failure.

CUDA builds still link only the CPU VMAF adapter when libvmaf is present; no
libvmaf CUDA, no CUDA runtime requirement for VMAF itself.

## Models

Vendored JSON under `model/` (exact byte copies of the pinned Netflix commit).
See `reports/VMAF_MODEL_CHECKSUMS.md`, `model/README.md`, and
`third_party/netflix-vmaf/` for license/attribution.

PixVQA loads models via `--model <path.json>` (`vmaf_model_load_from_path`).
Metadata prints the SHA-256 of the actual loaded file.
