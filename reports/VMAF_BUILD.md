# libvmaf 3.2.1 build and link strategy (Phase 1)

## Pin

| Item | Value |
|---|---|
| Repository | https://github.com/Netflix/vmaf |
| Tag | `v3.2.1` |
| Commit | `f85a853692a8c730d0270cd733c8bb30b5b93b7c` |
| `vmaf_version()` / CLI | `f85a853` |
| Meson project version | `3.2.0` (API patch level; tag is 3.2.1) |

`scripts/build_libvmaf.sh` verifies `git rev-parse HEAD` equals the pinned commit before configure.

## Obtain / build

```bash
# Requires meson, ninja, nasm, pkg-config (e.g. micromamba install in GCC_ENV)
bash scripts/build_libvmaf.sh
```

Install prefix: `deps/libvmaf-3.2.1` (gitignored). Source checkout: `deps/src/vmaf`.

Meson options:

```text
--buildtype=release
-Denable_tests=false
-Denable_docs=false
-Denable_tools=true
-Denable_cuda=false
-Dbuilt_in_models=true
-Denable_float=false
-Ddefault_library=static
```

CPU feature extractors and built-in `vmaf_v1.0.16*` models are compiled in. CUDA is never enabled. The official `vmaf` CLI is installed beside the library for numerical oracles.

## Link into PixVQA (optional)

`scripts/libvmaf_flags.sh` detects the prefix (`lib` or `lib64`) and exports static `pkg-config` flags.

**libvmaf is optional.** `build.sh`, `build_cuda.sh`, and `build_decode.sh` enable
`-DPIXVQA_HAS_VMAF=1`, compile `src/vmaf.cpp`, and link `libvmaf.a` only when
`deps/libvmaf-3.2.1` is present. Without that prefix, PixVQA builds the
dependency-free CPU configuration; `--vmaf` is then rejected at runtime.

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
