CXX ?= g++
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 -Iinclude -pthread
NATIVE ?= -march=native
SIMDFLAGS := -mavx2 -mfma

BUILD := build/cpu
BIN := $(BUILD)/pixvqa
TEST := $(BUILD)/selftest

# CPU objects only. The release entrypoints are build.sh and build_decode.sh.
LIB_SRCS := src/yuv_io.cpp src/psnr.cpp src/ssim_ref.cpp src/msssim.cpp src/thread_pool.cpp src/xpsnr.cpp src/hdr.cpp src/pu21.cpp
LIB_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRCS))
SIMD_OBJ := $(BUILD)/ssim_simd.o $(BUILD)/msssim_avx2.o

.PHONY: all clean test

all: $(BIN) $(TEST)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(NATIVE) -c $< -o $@

$(BUILD)/ssim_simd.o: src/ssim_simd.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(NATIVE) $(SIMDFLAGS) -c $< -o $@

$(BUILD)/msssim_avx2.o: src/msssim_avx2.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(NATIVE) -mavx2 -ffp-contract=off -fno-fast-math -c $< -o $@

$(BIN): src/main.cpp $(LIB_OBJS) $(SIMD_OBJ)
	$(CXX) $(CXXFLAGS) $(NATIVE) -DPIXVQA_BUILD_KIND=\"cpu\" $^ -o $@

$(TEST): src/selftest.cpp $(LIB_OBJS) $(SIMD_OBJ)
	$(CXX) $(CXXFLAGS) $(NATIVE) $^ -o $@

test: $(TEST)
	$(TEST)

clean:
	rm -rf build
