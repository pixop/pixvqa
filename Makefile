# Incremental PixVQA builds: make cpu | swdec | full
# Paths: scripts/deps_layout.sh + environment (docs/BUILD_MAKE.md).
# install_pixvqa.sh still uses build.sh / build_decode.sh / build_cuda.sh.

ROOT := $(abspath .)
SHELL := /bin/bash

DEPFLAGS := -MMD -MP

.PHONY: all cpu swdec cuda full clean help install test-cpu test config-force
.DEFAULT_GOAL := all

# Only initialize the configurations needed by the selected target(s).
# CPU-only development and `make help` must not require FFmpeg or CUDA.
GOALS := $(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)

all: cpu
test: test-cpu

config-force:

help:
	@echo "PixVQA incremental Make targets"
	@echo "  make cpu              CPU-only -> build/cpu/pixvqa"
	@echo "  make swdec            FFmpeg software decode -> build/swdec/pixvqa"
	@echo "  make full             CUDA + NVDEC + FFmpeg -> build/full/pixvqa"
	@echo "  make test-cpu         build/cpu/selftest and run it"
	@echo "  make install VARIANT=full PREFIX=/path/to/prefix"
	@echo "  make clean            remove build/cpu build/swdec build/cuda build/full"
	@echo ""
	@echo "Optional environment (see scripts/deps_layout.sh defaults):"
	@echo "  FFMPEG_PREFIX  GCC_ENV  GCC13  CUDA_HOME  DAV1D_LIB  PIXVQA_DEPS_ROOT"

clean:
	rm -rf build/cpu build/swdec build/cuda build/full

install:
	@test -n "$(PREFIX)" || (echo "Set PREFIX= install directory" >&2; exit 1)
	VARIANT="$${VARIANT:-cpu}"; \
	case "$$VARIANT" in cpu|swdec|full) ;; *) echo "VARIANT must be cpu, swdec, or full" >&2; exit 1;; esac; \
	test -x "$(ROOT)/build/$$VARIANT/pixvqa" || (echo "Missing build/$$VARIANT/pixvqa; run make $$VARIANT first" >&2; exit 1); \
	PREFIX="$(PREFIX)" PIXVQA_BIN="$(ROOT)/build/$$VARIANT/pixvqa" bash scripts/install.sh

# --- cpu ---

build/cpu/config.mk: config-force scripts/make_config.sh scripts/deps_layout.sh scripts/libvmaf_flags.sh Makefile
	@mkdir -p $(@D)
	@scripts/make_config.sh cpu > $@.tmp || { rm -f $@.tmp; exit 1; }
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

ifneq ($(filter all cpu test test-cpu,$(GOALS)),)
include build/cpu/config.mk
endif
CPU_LIB_OBJS := $(patsubst $(ROOT)/src/%.cpp,build/cpu/%.o,$(CPU_LIB_SRCS))

CPU_SIMD_OBJ := build/cpu/ssim_simd.o build/cpu/msssim_avx2.o

build/cpu/%.o: $(ROOT)/src/%.cpp build/cpu/config.mk | build/cpu
	$(CPU_CXX) $(CPU_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

build/cpu/ssim_simd.o: $(ROOT)/src/ssim_simd.cpp build/cpu/config.mk | build/cpu
	$(CPU_CXX) $(CPU_CXXFLAGS) $(DEPFLAGS) -mavx2 -mfma -c $< -o $@

build/cpu/msssim_avx2.o: $(ROOT)/src/msssim_avx2.cpp build/cpu/config.mk | build/cpu
	$(CPU_CXX) $(CPU_CXXFLAGS) $(DEPFLAGS) -mavx2 -ffp-contract=off -fno-fast-math -c $< -o $@

build/cpu/main.o: $(ROOT)/src/main.cpp build/cpu/config.mk | build/cpu
	$(CPU_CXX) $(CPU_CXXFLAGS) $(DEPFLAGS) -DPIXVQA_BUILD_KIND=\"$(CPU_BUILD_KIND)\" -c $< -o $@

build/cpu/selftest_main.o: $(ROOT)/src/selftest.cpp build/cpu/config.mk | build/cpu
	$(CPU_CXX) $(CPU_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

-include $(CPU_LIB_OBJS:.o=.d)
-include build/cpu/ssim_simd.d build/cpu/msssim_avx2.d build/cpu/main.d build/cpu/selftest_main.d

cpu: build/cpu/pixvqa build/cpu/selftest

build/cpu/pixvqa: build/cpu/config.mk build/cpu/main.o $(CPU_LIB_OBJS) $(CPU_SIMD_OBJ)
	$(CPU_CXX) $(CPU_CXXFLAGS) build/cpu/main.o $(CPU_LIB_OBJS) $(CPU_SIMD_OBJ) -o $@ $(CPU_LINK_LIBS)

build/cpu/selftest: build/cpu/config.mk build/cpu/selftest_main.o $(CPU_LIB_OBJS) $(CPU_SIMD_OBJ)
	$(CPU_CXX) $(CPU_CXXFLAGS) build/cpu/selftest_main.o $(CPU_LIB_OBJS) $(CPU_SIMD_OBJ) -o $@ $(CPU_LINK_LIBS)

test-cpu: build/cpu/selftest
	build/cpu/selftest

# --- swdec ---

build/swdec/config.mk: config-force scripts/make_config.sh scripts/deps_layout.sh scripts/libvmaf_flags.sh Makefile
	@mkdir -p $(@D)
	@scripts/make_config.sh swdec > $@.tmp || { rm -f $@.tmp; exit 1; }
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

ifneq ($(filter swdec,$(GOALS)),)
include build/swdec/config.mk
endif
SWDEC_LIB_OBJS := $(patsubst $(ROOT)/src/%.cpp,build/swdec/%.o,$(SWDEC_LIB_SRCS))

SWDEC_SIMD_OBJ := build/swdec/ssim_simd.o build/swdec/msssim_avx2.o

build/swdec/%.o: $(ROOT)/src/%.cpp build/swdec/config.mk | build/swdec
	$(SWDEC_CXX) $(SWDEC_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

build/swdec/ssim_simd.o: $(ROOT)/src/ssim_simd.cpp build/swdec/config.mk | build/swdec
	$(SWDEC_CXX) $(SWDEC_CXXFLAGS) $(DEPFLAGS) -mavx2 -mfma -c $< -o $@

build/swdec/msssim_avx2.o: $(ROOT)/src/msssim_avx2.cpp build/swdec/config.mk | build/swdec
	$(SWDEC_CXX) $(SWDEC_CXXFLAGS) $(DEPFLAGS) -mavx2 -ffp-contract=off -fno-fast-math -c $< -o $@

build/swdec/main.o: $(ROOT)/src/main.cpp build/swdec/config.mk | build/swdec
	$(SWDEC_CXX) $(SWDEC_CXXFLAGS) $(DEPFLAGS) -DPIXVQA_BUILD_KIND=\"$(SWDEC_BUILD_KIND)\" -c $< -o $@

-include $(SWDEC_LIB_OBJS:.o=.d)
-include build/swdec/ssim_simd.d build/swdec/msssim_avx2.d build/swdec/main.d

swdec: build/swdec/pixvqa

build/swdec/pixvqa: build/swdec/config.mk build/swdec/main.o $(SWDEC_LIB_OBJS) $(SWDEC_SIMD_OBJ)
	$(SWDEC_CXX) $(SWDEC_CXXFLAGS) build/swdec/main.o $(SWDEC_LIB_OBJS) $(SWDEC_SIMD_OBJ) -o $@ $(SWDEC_LDFLAGS)

# --- cuda (objects reused by full) ---

build/cuda/config.mk: config-force scripts/make_config.sh scripts/deps_layout.sh scripts/libvmaf_flags.sh Makefile
	@mkdir -p $(@D)
	@scripts/make_config.sh cuda > $@.tmp || { rm -f $@.tmp; exit 1; }
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

ifneq ($(filter cuda full,$(GOALS)),)
include build/cuda/config.mk
endif
CUDA_LIB_OBJS := $(patsubst $(ROOT)/src/%.cpp,build/cuda/%.o,$(CUDA_LIB_SRCS))

build/cuda/gcc13_host.stamp: build/cuda/config.mk
	@GCC13="$(CUDA_GCC13)"; \
	test -n "$$GCC13"; \
	ln -sfn x86_64-conda-linux-gnu-gcc "$$GCC13/bin/gcc"; \
	ln -sfn x86_64-conda-linux-gnu-g++ "$$GCC13/bin/g++"; \
	ln -sfn x86_64-conda-linux-gnu-gcc "$$GCC13/bin/cc"; \
	ln -sfn x86_64-conda-linux-gnu-g++ "$$GCC13/bin/c++"; \
	touch $@

CUDA_SIMD_OBJ := build/cuda/ssim_simd.o build/cuda/msssim_avx2.o
CUDA_NV_OBJ := build/cuda/cuda_metrics.o build/cuda/cuda_pu21.o
FULL_CUDA_EXTRA_OBJS := build/cuda/hdr.o build/cuda/pu21.o build/cuda/ssim_ref.o build/cuda/msssim.o

build/cuda/%.o: $(ROOT)/src/%.cpp build/cuda/config.mk | build/cuda
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

build/cuda/ssim_simd.o: $(ROOT)/src/ssim_simd.cpp build/cuda/config.mk | build/cuda
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(DEPFLAGS) -mavx2 -mfma -c $< -o $@

build/cuda/msssim_avx2.o: $(ROOT)/src/msssim_avx2.cpp build/cuda/config.mk | build/cuda
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(DEPFLAGS) -mavx2 -ffp-contract=off -fno-fast-math -c $< -o $@

build/cuda/cuda_metrics.o: $(ROOT)/src/cuda_metrics.cu build/cuda/config.mk build/cuda/gcc13_host.stamp | build/cuda
	$(CUDA_NVCC) -O3 -std=c++17 -Iinclude --compiler-bindir $(CUDA_GCC13)/bin \
	  -gencode arch=compute_89,code=sm_89 \
	  -gencode arch=compute_120,code=sm_120 \
	  -gencode arch=compute_120,code=compute_120 \
	  -MMD -MP -c $< -o $@

build/cuda/cuda_pu21.o: $(ROOT)/src/cuda_pu21.cu build/cuda/config.mk build/cuda/gcc13_host.stamp | build/cuda
	$(CUDA_NVCC) -O3 -std=c++17 -Iinclude --fmad=false --compiler-bindir $(CUDA_GCC13)/bin \
	  -gencode arch=compute_89,code=sm_89 \
	  -gencode arch=compute_120,code=sm_120 \
	  -gencode arch=compute_120,code=compute_120 \
	  -MMD -MP -c $< -o $@

-include $(CUDA_LIB_OBJS:.o=.d)
-include build/cuda/ssim_simd.d build/cuda/msssim_avx2.d
-include build/cuda/cuda_metrics.d build/cuda/cuda_pu21.d

build/cuda/main.o: $(ROOT)/src/main.cpp build/cuda/config.mk | build/cuda
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(DEPFLAGS) -DPIXVQA_BUILD_KIND=\"cuda\" -c $< -o $@

cuda: build/cuda/pixvqa

build/cuda/pixvqa: build/cuda/config.mk build/cuda/main.o $(CUDA_LIB_OBJS) $(CUDA_SIMD_OBJ) $(CUDA_NV_OBJ)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) build/cuda/main.o $(CUDA_LIB_OBJS) $(CUDA_SIMD_OBJ) $(CUDA_NV_OBJ) -o $@ $(CUDA_LDFLAGS)

# --- full ---

build/full/config.mk: config-force scripts/make_config.sh scripts/deps_layout.sh scripts/libvmaf_flags.sh Makefile
	@mkdir -p $(@D)
	@scripts/make_config.sh full > $@.tmp || { rm -f $@.tmp; exit 1; }
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

ifneq ($(filter full,$(GOALS)),)
include build/full/config.mk
endif

build/full/main.o: $(ROOT)/src/main.cpp build/full/config.mk | build/full
	$(FULL_CXX) $(FULL_CXXFLAGS) $(DEPFLAGS) -DPIXVQA_BUILD_KIND=\"cuda\" -c $< -o $@

build/full/decode.o: $(ROOT)/src/decode.cpp build/full/config.mk | build/full
	$(FULL_CXX) $(FULL_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

-include build/full/main.d build/full/decode.d

FULL_LINK_OBJS = build/full/main.o build/full/decode.o \
	build/cuda/yuv_io.o build/cuda/psnr.o build/cuda/thread_pool.o build/cuda/xpsnr.o \
	$(FULL_CUDA_EXTRA_OBJS) $(CUDA_SIMD_OBJ) $(CUDA_NV_OBJ) \
	$(filter build/cuda/vmaf.o,$(CUDA_LIB_OBJS))

full: build/full/pixvqa

build/full/pixvqa: build/cuda/config.mk build/cuda/gcc13_host.stamp build/full/config.mk \
		$(FULL_LINK_OBJS)
	$(FULL_FULL_LINK_CXX) $(FULL_CXXFLAGS) -DPIXVQA_BUILD_KIND=\"cuda\" $(FULL_LINK_OBJS) -o $@ $(FULL_LDFLAGS)
