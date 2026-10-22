cmake_minimum_required(VERSION 3.16)
project(pixvqa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PIXVQA_NATIVE "Enable -march=native" ON)

# CPU-only library. CUDA and FFmpeg builds use build_cuda.sh and build_decode.sh.
add_library(pixvqa_core
  src/yuv_io.cpp
  src/psnr.cpp
  src/ssim_ref.cpp
  src/ssim_simd.cpp
  src/msssim.cpp
  src/msssim_avx2.cpp
  src/thread_pool.cpp
  src/xpsnr.cpp
  src/hdr.cpp
  src/pu21.cpp
)

target_include_directories(pixvqa_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(pixvqa_core PRIVATE -O3 -Wall -Wextra)
if(PIXVQA_NATIVE)
  target_compile_options(pixvqa_core PRIVATE -march=native)
endif()
set_source_files_properties(src/ssim_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(src/msssim_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off;-fno-fast-math")

find_package(Threads REQUIRED)
target_link_libraries(pixvqa_core PUBLIC Threads::Threads)

add_executable(pixvqa src/main.cpp)
target_link_libraries(pixvqa PRIVATE pixvqa_core)
target_compile_definitions(pixvqa PRIVATE PIXVQA_BUILD_KIND="cpu")
target_compile_options(pixvqa PRIVATE -O3 -Wall -Wextra)

add_executable(selftest src/selftest.cpp)
target_link_libraries(selftest PRIVATE pixvqa_core)
target_compile_options(selftest PRIVATE -O3 -Wall -Wextra)

enable_testing()
add_test(NAME selftest COMMAND selftest)
