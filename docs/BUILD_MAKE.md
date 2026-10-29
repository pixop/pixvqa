# Incremental GNU Make builds

For day-to-day development after dependencies are installed, `make` rebuilds only
changed translation units. The guided installer and `docs/INSTALL.md` still use
`build.sh`, `build_decode.sh`, and `build_cuda.sh` (full clean compiles).

## Targets

| Target | Output | Notes |
|--------|--------|--------|
| `make cpu` | `build/cpu/pixvqa` | Raw YUV; optional libvmaf from `deps/` |
| `make swdec` | `build/swdec/pixvqa` | FFmpeg n8.1.3 software decode |
| `make full` | `build/full/pixvqa` | CUDA metrics + NVDEC + FFmpeg |
| `make test-cpu` | runs `build/cpu/selftest` | |
| `make install VARIANT=… PREFIX=…` | wraps `scripts/install.sh` | `VARIANT` = `cpu`, `swdec`, or `full` |
| `make clean` | removes `build/cpu`, `build/swdec`, `build/cuda`, `build/full` | |

Object files and generated `config.mk` files live under separate `build/<variant>/`
trees. Header dependencies use `-MMD -MP`. Each `config.mk` carries the selected toolchain, flags and paths, and is refreshed
only when those settings change. Objects depending on it then rebuild.

## Environment

`scripts/make_config.sh` sources `scripts/deps_layout.sh` (same defaults as the
installer) and `scripts/libvmaf_flags.sh`. Override paths when needed:

```bash
# Legacy layout (see deps_layout.sh): $HOME/ffmpeg-n8.1.3, $HOME/pixvqa-deps/gcc13, …
export GCC13="$HOME/pixvqa-deps/gcc13"
export CUDA_HOME=/usr/local/cuda-12.8
export GCC_ENV=/usr          # swdec host compiler (default /usr when /usr/bin/g++ exists)
# Optional: FFMPEG_PREFIX, DAV1D_LIB, PIXVQA_DEPS_ROOT, LIBVMAF_PREFIX, ZIMG_PREFIX
make full
```

`make full` links with `/usr/bin/g++` by default (`PIXVQA_FULL_LINK_CXX`) so FFmpeg
uses the system glibc. CUDA `.cu` files compile with `nvcc` for `sm_89` and `sm_120`.

## Quick checks

```bash
make clean && make full          # first build compiles
make full                        # second build should be a no-op
touch src/main.cpp && make full  # only main.o + link; no nvcc
make swdec && ldd build/swdec/pixvqa | grep libavcodec
```

Do not commit `build/` or generated `config.mk` files.
