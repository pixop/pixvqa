// CUDA PSNR, SSIM, and five-scale MS-SSIM.
//
// Architecture (one stream, reusable allocations, no per-frame cudaMalloc):
//   1. Copy Frame planes into Ctx-owned pinned staging, then H2D tightly packed uint16.
//      Frame memory is never cudaHostRegister'd (avoids dangling pins on realloc).
//   2. PSNR: exact int64 squared-difference reduction per plane.
//   3. MS-SSIM / SSIM: separable 11-tap Gaussian, sigma 1.5, clamp-to-edge,
//      matching the CPU scalar border rule. Horizontal pass writes five moment
//      planes. Vertical pass keeps the five filtered values in registers and
//      reduces SSIM/CS. No per-pixel SSIM map is stored.
//   4. Even 2x2 pyramid: mean of four samples, nearest-even, clamp to [0, peak].
//      Odd dimensions that still need a coarser scale fail instead of changing
//      the metric. Scale count follows luma min(H,W), shared with chroma.
//   5. fp64 filters in double. fp32 filters in float and evaluates SSIM in float.
//      mixed filters in float and evaluates SSIM in double.
//      All three reduce the spatial sum in double.
//
// This file does not decode. It can convert a decoder-owned P010 surface into
// the metric planes. Those kernels only read the surface.

#include "pixvqa/cuda_backend.hpp"
#include <cuda.h>
#include "pixvqa/msssim.hpp"
#include "pixvqa/xpsnr.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace pixvqa {

namespace {

constexpr int kRadius = 5;
constexpr int kTaps = 11;
constexpr int kThreads = 256;

__constant__ float c_k32[kTaps];
__constant__ double c_k64[kTaps];

#define CUDA_CHECK(expr)                                                                    \
  do {                                                                                      \
    cudaError_t _e = (expr);                                                                \
    if (_e != cudaSuccess) {                                                                \
      throw std::runtime_error(std::string("CUDA: ") + #expr + ": " + cudaGetErrorString(_e)); \
    }                                                                                       \
  } while (0)

void check_launch(const char* what) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel ") + what + ": " + cudaGetErrorString(e));
  }
}

double mse_to_psnr(double mse, double peak) {
  if (!(mse > 0.0)) return std::numeric_limits<double>::infinity();
  return 10.0 * std::log10((peak * peak) / mse);
}

template <typename T>
__device__ T kern_at(int i);

template <>
__device__ float kern_at<float>(int i) {
  return c_k32[i];
}

template <>
__device__ double kern_at<double>(int i) {
  return c_k64[i];
}

__device__ double block_sum_256(double v) {
  __shared__ double sm[kThreads];
  const int tx = threadIdx.x;
  sm[tx] = v;
  __syncthreads();
  for (int offset = kThreads / 2; offset >= 32; offset >>= 1) {
    if (tx < offset) sm[tx] += sm[tx + offset];
    __syncthreads();
  }
  double x = (tx < 32) ? sm[tx] : 0.0;
  if (tx < 32) {
    x += __shfl_down_sync(0xffffffff, x, 16);
    x += __shfl_down_sync(0xffffffff, x, 8);
    x += __shfl_down_sync(0xffffffff, x, 4);
    x += __shfl_down_sync(0xffffffff, x, 2);
    x += __shfl_down_sync(0xffffffff, x, 1);
  }
  return x;
}

__device__ unsigned long long block_sum_u64(unsigned long long v) {
  __shared__ unsigned long long sm[kThreads];
  const int tx = threadIdx.x;
  sm[tx] = v;
  __syncthreads();
  for (int offset = kThreads / 2; offset >= 32; offset >>= 1) {
    if (tx < offset) sm[tx] += sm[tx + offset];
    __syncthreads();
  }
  unsigned long long x = (tx < 32) ? sm[tx] : 0ull;
  if (tx < 32) {
    x += __shfl_down_sync(0xffffffff, x, 16);
    x += __shfl_down_sync(0xffffffff, x, 8);
    x += __shfl_down_sync(0xffffffff, x, 4);
    x += __shfl_down_sync(0xffffffff, x, 2);
    x += __shfl_down_sync(0xffffffff, x, 1);
  }
  return x;
}

__global__ void sse_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int w,
                           int h, unsigned long long* acc) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  unsigned long long local = 0ull;
  if (y < h && x < w) {
    const long long d =
        static_cast<long long>(a[static_cast<size_t>(y) * w + x]) -
        static_cast<long long>(b[static_cast<size_t>(y) * w + x]);
    local = static_cast<unsigned long long>(d * d);
  }
  const unsigned long long sum = block_sum_u64(local);
  if (threadIdx.x == 0 && y < h) atomicAdd(acc, sum);
}

// Horizontal Gaussian of x, y, x^2, y^2, xy. Clamp-to-edge. Output stride is w.
template <typename T>
__global__ void filter_h(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int w,
                         int h, T* __restrict__ o0, T* __restrict__ o1, T* __restrict__ o2,
                         T* __restrict__ o3, T* __restrict__ o4) {
  constexpr int R = kRadius;
  __shared__ uint16_t sa[kThreads + 2 * R];
  __shared__ uint16_t sb[kThreads + 2 * R];
  const int y = static_cast<int>(blockIdx.y);
  const int x0 = static_cast<int>(blockIdx.x) * kThreads;
  const int tx = threadIdx.x;
  if (y >= h) return;
  for (int i = tx; i < kThreads + 2 * R; i += kThreads) {
    int gx = x0 + i - R;
    if (gx < 0) gx = 0;
    if (gx >= w) gx = w - 1;
    sa[i] = a[static_cast<size_t>(y) * w + gx];
    sb[i] = b[static_cast<size_t>(y) * w + gx];
  }
  __syncthreads();
  const int x = x0 + tx;
  if (x >= w) return;
  T sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
#pragma unroll
  for (int t = 0; t < kTaps; ++t) {
    const T k = kern_at<T>(t);
    const T av = static_cast<T>(sa[tx + t]);
    const T bv = static_cast<T>(sb[tx + t]);
    sx += k * av;
    sy += k * bv;
    sxx += k * av * av;
    syy += k * bv * bv;
    sxy += k * av * bv;
  }
  const size_t idx = static_cast<size_t>(y) * w + x;
  o0[idx] = sx;
  o1[idx] = sy;
  o2[idx] = sxx;
  o3[idx] = syy;
  o4[idx] = sxy;
}

// Vertical Gaussian + SSIM/CS. Valid crop is radius 5. Spatial sum is double.
template <typename T, bool kFormula64>
__global__ void filter_v_ssim(const T* __restrict__ m1s, const T* __restrict__ m2s,
                              const T* __restrict__ s1s, const T* __restrict__ s2s,
                              const T* __restrict__ s12s, int w, int h, double c1, double c2,
                              int full, double* acc) {
  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = static_cast<int>(blockIdx.y);
  double score = 0.0;
  if (x < ow && y < oh) {
    const int cx = x + kRadius;
    const int cy = y + kRadius;
    T m1 = 0, m2 = 0, s1 = 0, s2 = 0, s12 = 0;
#pragma unroll
    for (int t = 0; t < kTaps; ++t) {
      int yy = cy + (t - kRadius);
      if (yy < 0) yy = 0;
      if (yy >= h) yy = h - 1;
      const T k = kern_at<T>(t);
      const size_t idx = static_cast<size_t>(yy) * w + cx;
      m1 += k * m1s[idx];
      m2 += k * m2s[idx];
      s1 += k * s1s[idx];
      s2 += k * s2s[idx];
      s12 += k * s12s[idx];
    }
    if (kFormula64) {
      const double M1 = static_cast<double>(m1);
      const double M2 = static_cast<double>(m2);
      const double V1 = static_cast<double>(s1) - M1 * M1;
      const double V2 = static_cast<double>(s2) - M2 * M2;
      const double V12 = static_cast<double>(s12) - M1 * M2;
      const double cs = (2.0 * V12 + c2) / (V1 + V2 + c2);
      if (full) {
        const double lum = (2.0 * M1 * M2 + c1) / (M1 * M1 + M2 * M2 + c1);
        score = lum * cs;
      } else {
        score = cs;
      }
    } else {
      const T C1 = static_cast<T>(c1);
      const T C2 = static_cast<T>(c2);
      const T V1 = s1 - m1 * m1;
      const T V2 = s2 - m2 * m2;
      const T V12 = s12 - m1 * m2;
      const T cs = (static_cast<T>(2) * V12 + C2) / (V1 + V2 + C2);
      T val = cs;
      if (full) {
        const T lum = (static_cast<T>(2) * m1 * m2 + C1) / (m1 * m1 + m2 * m2 + C1);
        val = lum * cs;
      }
      score = static_cast<double>(val);
    }
  }
  const double sum = block_sum_256(score);
  if (threadIdx.x == 0 && y < oh) atomicAdd(acc, sum);
}

__global__ void p010_y_kernel(const uint8_t* src, int pitch_bytes, uint16_t* dst, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const uint16_t* row =
      reinterpret_cast<const uint16_t*>(src + static_cast<size_t>(y) * static_cast<size_t>(pitch_bytes));
  dst[static_cast<size_t>(y) * static_cast<size_t>(w) + x] = static_cast<uint16_t>(row[x] >> 6);
}

__global__ void p010_uv_kernel(const uint8_t* src, int pitch_bytes, uint16_t* dst_u, uint16_t* dst_v,
                               int cw, int ch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cw || y >= ch) return;
  const uint16_t* row =
      reinterpret_cast<const uint16_t*>(src + static_cast<size_t>(y) * static_cast<size_t>(pitch_bytes));
  dst_u[static_cast<size_t>(y) * static_cast<size_t>(cw) + x] = static_cast<uint16_t>(row[x * 2] >> 6);
  dst_v[static_cast<size_t>(y) * static_cast<size_t>(cw) + x] = static_cast<uint16_t>(row[x * 2 + 1] >> 6);
}

struct Ctx {
  bool ready = false;
  bool shutting_down = false;
  cudaStream_t stream = nullptr;
  cudaEvent_t ev_a = nullptr, ev_b = nullptr, ev_c = nullptr, ev_d = nullptr, ev_x = nullptr, ev_y = nullptr;
  uint16_t* plane[3][2] = {};
  size_t plane_cap[3][2] = {};
  uint16_t* alt[2] = {};
  size_t alt_cap[2] = {};
  void* work = nullptr;
  size_t work_samples = 0;
  unsigned long long* d_sse = nullptr;
  double* d_acc = nullptr;
  // MS-SSIM pyramid images (4 slots). Not aliased to plane[][], alt, or work.
  void* msssim_pyr = nullptr;
  size_t msssim_pyr_samples = 0;
  double* d_msssim = nullptr;
  // Owned pinned staging. Frame planes are never host-registered: they are
  // copied into this buffer so reallocations and Frame destruction cannot leave
  // dangling cudaHostRegister entries.
  uint16_t* pin_stage = nullptr;
  size_t pin_samples = 0;
  // XPSNR history is not aliased to plane[][] or alt/work. MS-SSIM overwrites those.
  int16_t* xpsnr_m1 = nullptr;
  int16_t* xpsnr_m2 = nullptr;
  size_t xpsnr_hist_cap = 0;
  int xpsnr_w = 0, xpsnr_h = 0, xpsnr_depth = 0, xpsnr_rate = 0;
  int xpsnr_chroma = 0;
  bool xpsnr_reset_done = false;
  struct XpsnrDesc {
    int x, y, bw, bh;
  };
  XpsnrDesc* d_desc = nullptr;
  int desc_cap = 0;
  int desc_w = 0, desc_h = 0;
  unsigned long long* d_blk_sse = nullptr;
  unsigned long long* d_blk_sa = nullptr;
  unsigned long long* d_blk_ta = nullptr;
  uint8_t* d_blk_valid = nullptr;
  unsigned long long* d_chr_u = nullptr;
  unsigned long long* d_chr_v = nullptr;
  int blk_cap = 0;
  unsigned long long* d_raw_sse = nullptr;
  bool xpsnr_raw = false;
  int xpsnr_nblk = 0;
  std::vector<unsigned long long> h_blk_sse, h_blk_sa, h_blk_ta, h_chr_u, h_chr_v;
  std::vector<uint8_t> h_blk_valid;
  std::vector<XpsnrDesc> h_desc;
  double ms_h2d = 0, ms_kernel = 0, ms_d2h = 0, ms_xpsnr = 0;
  int h2d_test_copies = 0;
  CUcontext primary = nullptr;
  int frames = 0;
  size_t alloc_bytes = 0;
  std::string name;
  int cc_major = 0, cc_minor = 0;
  size_t vram = 0;
  int fp64_ratio = 0;
  int driver = 0, runtime = 0;
  int sms = 0;
  // Null unless a VMAF frame snapshot is attached. Not used by the CLI.
  CudaFrameSnapshotHook snapshot_hook = nullptr;
};

Ctx& ctx() {
  static Ctx c;
  return c;
}

bool profile_on() {
  const char* e = pix_getenv("CUDA_PROFILE");
  return e && e[0] == '1';
}

void free_device_u16(uint16_t*& ptr, size_t& cap, size_t& alloc_bytes) {
  if (!ptr) return;
  alloc_bytes -= cap * sizeof(uint16_t);
  cudaFree(ptr);
  ptr = nullptr;
  cap = 0;
}

void cuda_shutdown_impl() {
  auto& c = ctx();
  if (!c.ready || c.shutting_down) return;
  c.shutting_down = true;
  if (c.stream) {
    cudaStreamSynchronize(c.stream);
  }
  for (int p = 0; p < 3; ++p) {
    for (int s = 0; s < 2; ++s) free_device_u16(c.plane[p][s], c.plane_cap[p][s], c.alloc_bytes);
  }
  for (int s = 0; s < 2; ++s) free_device_u16(c.alt[s], c.alt_cap[s], c.alloc_bytes);
  if (c.work) {
    cudaFree(c.work);
    c.work = nullptr;
    c.work_samples = 0;
  }
  if (c.d_sse) {
    cudaFree(c.d_sse);
    c.d_sse = nullptr;
  }
  if (c.d_acc) {
    cudaFree(c.d_acc);
    c.d_acc = nullptr;
  }
  if (c.msssim_pyr) {
    cudaFree(c.msssim_pyr);
    c.msssim_pyr = nullptr;
    c.msssim_pyr_samples = 0;
  }
  if (c.d_msssim) {
    cudaFree(c.d_msssim);
    c.d_msssim = nullptr;
  }
  if (c.pin_stage) {
    cudaFreeHost(c.pin_stage);
    c.pin_stage = nullptr;
    c.pin_samples = 0;
  }
  auto free_counted = [&](void* ptr, size_t bytes) {
    if (!ptr) return;
    cudaFree(ptr);
    c.alloc_bytes -= bytes;
  };
  free_counted(c.xpsnr_m1, c.xpsnr_hist_cap * sizeof(int16_t));
  free_counted(c.xpsnr_m2, c.xpsnr_hist_cap * sizeof(int16_t));
  c.xpsnr_m1 = c.xpsnr_m2 = nullptr;
  c.xpsnr_hist_cap = 0;
  free_counted(c.d_desc, static_cast<size_t>(c.desc_cap) * sizeof(Ctx::XpsnrDesc));
  c.d_desc = nullptr;
  c.desc_cap = 0;
  free_counted(c.d_blk_sse, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  free_counted(c.d_blk_sa, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  free_counted(c.d_blk_ta, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  free_counted(c.d_blk_valid, static_cast<size_t>(c.blk_cap) * sizeof(uint8_t));
  free_counted(c.d_chr_u, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  free_counted(c.d_chr_v, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  c.d_blk_sse = c.d_blk_sa = c.d_blk_ta = c.d_chr_u = c.d_chr_v = nullptr;
  c.d_blk_valid = nullptr;
  c.blk_cap = 0;
  if (c.d_raw_sse) {
    cudaFree(c.d_raw_sse);
    c.d_raw_sse = nullptr;
    c.alloc_bytes -= 3 * sizeof(unsigned long long);
  }
  c.xpsnr_reset_done = false;
  if (c.ev_a) cudaEventDestroy(c.ev_a);
  if (c.ev_b) cudaEventDestroy(c.ev_b);
  if (c.ev_c) cudaEventDestroy(c.ev_c);
  if (c.ev_d) cudaEventDestroy(c.ev_d);
  if (c.ev_x) cudaEventDestroy(c.ev_x);
  if (c.ev_y) cudaEventDestroy(c.ev_y);
  c.ev_a = c.ev_b = c.ev_c = c.ev_d = c.ev_x = c.ev_y = nullptr;
  if (c.stream) cudaStreamDestroy(c.stream);
  c.stream = nullptr;
  c.ready = false;
  c.shutting_down = false;
  c.alloc_bytes = 0;
}

void ensure_pin_stage(size_t samples) {
  auto& c = ctx();
  if (c.pin_samples >= samples) return;
  if (c.pin_stage) {
    CUDA_CHECK(cudaFreeHost(c.pin_stage));
    c.pin_stage = nullptr;
    c.pin_samples = 0;
  }
  uint16_t* p = nullptr;
  cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&p), samples * sizeof(uint16_t),
                                cudaHostAllocDefault);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: cudaHostAlloc pinned staging failed: ") +
                             cudaGetErrorString(e));
  }
  c.pin_stage = p;
  c.pin_samples = samples;
}

void upload_kernel_weights() {
  double k[kTaps];
  double sum = 0.0;
  for (int i = 0; i < kTaps; ++i) {
    const double x = static_cast<double>(i - kRadius);
    k[i] = std::exp(-(x * x) / (2.0 * 1.5 * 1.5));
    sum += k[i];
  }
  for (int i = 0; i < kTaps; ++i) k[i] /= sum;
  float kf[kTaps];
  for (int i = 0; i < kTaps; ++i) kf[i] = static_cast<float>(k[i]);
  CUDA_CHECK(cudaMemcpyToSymbol(c_k64, k, sizeof(k)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_k32, kf, sizeof(kf)));
}

void grow_u16(uint16_t*& ptr, size_t& cap, size_t need, size_t& alloc_bytes) {
  if (cap >= need) return;
  if (ptr) {
    alloc_bytes -= cap * sizeof(uint16_t);
    CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
    cap = 0;
  }
  CUDA_CHECK(cudaMalloc(&ptr, need * sizeof(uint16_t)));
  cap = need;
  alloc_bytes += need * sizeof(uint16_t);
}

void ensure_storage(const VideoFormat& fmt) {
  auto& c = ctx();
  const size_t ny = static_cast<size_t>(fmt.width) * fmt.height;
  const size_t nc = static_cast<size_t>(fmt.chroma_width()) * fmt.chroma_height();
  const size_t need[3] = {ny, nc, nc};
  for (int p = 0; p < 3; ++p) {
    for (int s = 0; s < 2; ++s) {
      grow_u16(c.plane[p][s], c.plane_cap[p][s], need[p], c.alloc_bytes);
    }
  }
  // Full-frame scratch so ping-pong downsample never writes past the allocation.
  for (int s = 0; s < 2; ++s) grow_u16(c.alt[s], c.alt_cap[s], ny, c.alloc_bytes);
  if (c.work_samples < ny) {
    if (c.work) {
      c.alloc_bytes -= c.work_samples * 5 * sizeof(double);
      CUDA_CHECK(cudaFree(c.work));
      c.work = nullptr;
      c.work_samples = 0;
    }
    const size_t bytes = ny * 5 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&c.work, bytes));
    c.work_samples = ny;
    c.alloc_bytes += bytes;
  }
  if (!c.d_sse) {
    CUDA_CHECK(cudaMalloc(&c.d_sse, 3 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&c.d_acc, 32 * sizeof(double)));
    c.alloc_bytes += 3 * sizeof(unsigned long long) + 32 * sizeof(double);
  }
  // Two full frames of tightly packed samples for staged H2D.
  ensure_pin_stage(2 * (ny + nc + nc));
}

void init_once() {
  auto& c = ctx();
  if (c.ready) return;
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev <= 0) throw std::runtime_error("no CUDA device");
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  c.name = prop.name;
  c.cc_major = prop.major;
  c.cc_minor = prop.minor;
  c.vram = prop.totalGlobalMem;
  c.sms = prop.multiProcessorCount;
  if (prop.major < 8 || (prop.major == 8 && prop.minor < 9)) {
    throw std::runtime_error("unsupported compute capability " + std::to_string(prop.major) + "." +
                             std::to_string(prop.minor) + " (this CUDA build requires >= 8.9)");
  }
  cudaDeviceGetAttribute(&c.fp64_ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, 0);
  cudaDriverGetVersion(&c.driver);
  cudaRuntimeGetVersion(&c.runtime);
  CUDA_CHECK(cudaStreamCreate(&c.stream));
  CUDA_CHECK(cudaEventCreate(&c.ev_a));
  CUDA_CHECK(cudaEventCreate(&c.ev_b));
  CUDA_CHECK(cudaEventCreate(&c.ev_c));
  CUDA_CHECK(cudaEventCreate(&c.ev_d));
  CUDA_CHECK(cudaEventCreate(&c.ev_x));
  CUDA_CHECK(cudaEventCreate(&c.ev_y));
  upload_kernel_weights();
  c.ready = true;
  static bool atexit_set = false;
  if (!atexit_set) {
    std::atexit(cuda_shutdown_impl);
    atexit_set = true;
  }
  const char* info = pix_getenv("CUDA_INFO");
  if (info && info[0] == '1') {
    cuda_print_device_info();
  }
}

// Pack one plane into the owned pinned staging buffer and schedule H2D.
size_t upload_plane(const uint16_t* src, int stride, int w, int h, uint16_t* dst,
                    size_t pin_offset) {
  auto& c = ctx();
  uint16_t* pin = c.pin_stage + pin_offset;
  if (stride == w) {
    std::memcpy(pin, src, static_cast<size_t>(w) * h * sizeof(uint16_t));
  } else {
    for (int y = 0; y < h; ++y) {
      std::memcpy(pin + static_cast<size_t>(y) * w, src + static_cast<size_t>(y) * stride,
                  static_cast<size_t>(w) * sizeof(uint16_t));
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(dst, pin, static_cast<size_t>(w) * h * sizeof(uint16_t),
                             cudaMemcpyHostToDevice, c.stream));
  return pin_offset + static_cast<size_t>(w) * h;
}

void launch_sse(const uint16_t* a, const uint16_t* b, int w, int h, unsigned long long* acc) {
  sse_kernel<<<dim3((w + kThreads - 1) / kThreads, h), kThreads, 0, ctx().stream>>>(a, b, w, h, acc);
  check_launch("sse");
}

// Conventional MS-SSIM. Valid 11-tap Gaussian (no edge clamp), float
// 2x2 average pooling with pytorch odd-size zero padding, and a host-side
// product of the published weights. Scratch is work (horizontal moments) and
// msssim_pyr (downsampled images). plane[][] is only read.
template <typename Src, typename T>
__global__ void msssim_horiz(const Src* __restrict__ a, const Src* __restrict__ b, int w, int h,
                           T* __restrict__ ox, T* __restrict__ oy, T* __restrict__ oxx,
                           T* __restrict__ oyy, T* __restrict__ oxy) {
  const int ow = w - (kTaps - 1);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= ow || y >= h) return;
  const Src* ra = a + static_cast<size_t>(y) * w + x;
  const Src* rb = b + static_cast<size_t>(y) * w + x;
  T sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
#pragma unroll
  for (int t = 0; t < kTaps; ++t) {
    const T k = kern_at<T>(t);
    const T av = static_cast<T>(ra[t]);
    const T bv = static_cast<T>(rb[t]);
    sx += k * av;
    sy += k * bv;
    sxx += k * av * av;
    syy += k * bv * bv;
    sxy += k * av * bv;
  }
  const size_t idx = static_cast<size_t>(y) * ow + x;
  ox[idx] = sx;
  oy[idx] = sy;
  oxx[idx] = sxx;
  oyy[idx] = syy;
  oxy[idx] = sxy;
}

template <typename T, bool kFormula64>
__global__ void msssim_vert(const T* __restrict__ hx, const T* __restrict__ hy, const T* __restrict__ hxx,
                          const T* __restrict__ hyy, const T* __restrict__ hxy, int stride, int h,
                          double c1, double c2, double* acc_cs, double* acc_ssim) {
  const int ow = stride;
  const int oh = h - (kTaps - 1);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y);
  double cs_v = 0.0;
  double ss_v = 0.0;
  if (x < ow && y < oh) {
    T mx = 0, my = 0, xx = 0, yy = 0, xy = 0;
#pragma unroll
    for (int t = 0; t < kTaps; ++t) {
      const T k = kern_at<T>(t);
      const size_t idx = static_cast<size_t>(y + t) * stride + x;
      mx += k * hx[idx];
      my += k * hy[idx];
      xx += k * hxx[idx];
      yy += k * hyy[idx];
      xy += k * hxy[idx];
    }
    if (kFormula64) {
      const double M1 = static_cast<double>(mx);
      const double M2 = static_cast<double>(my);
      const double V1 = static_cast<double>(xx) - M1 * M1;
      const double V2 = static_cast<double>(yy) - M2 * M2;
      const double V12 = static_cast<double>(xy) - M1 * M2;
      const double cs = (2.0 * V12 + c2) / (V1 + V2 + c2);
      const double lum = (2.0 * M1 * M2 + c1) / (M1 * M1 + M2 * M2 + c1);
      cs_v = cs;
      ss_v = lum * cs;
    } else {
      const T C1 = static_cast<T>(c1);
      const T C2 = static_cast<T>(c2);
      const T V1 = xx - mx * mx;
      const T V2 = yy - my * my;
      const T V12 = xy - mx * my;
      const T cs = (static_cast<T>(2) * V12 + C2) / (V1 + V2 + C2);
      const T lum = (static_cast<T>(2) * mx * my + C1) / (mx * mx + my * my + C1);
      cs_v = static_cast<double>(cs);
      ss_v = static_cast<double>(lum * cs);
    }
  }
  const double sum_cs = block_sum_256(cs_v);
  const double sum_ss = block_sum_256(ss_v);
  if (threadIdx.x == 0 && y < oh) {
    atomicAdd(acc_cs, sum_cs);
    atomicAdd(acc_ssim, sum_ss);
  }
}

template <typename Src, typename T>
__global__ void msssim_pool2(const Src* __restrict__ src, int w, int h, T* __restrict__ dst, int ow,
                           int oh) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= ow || y >= oh) return;
  const int pad_h = h & 1;
  const int pad_w = w & 1;
  T sum = 0;
#pragma unroll
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int sy = y * 2 + dy - pad_h;
      const int sx = x * 2 + dx - pad_w;
      if (sy >= 0 && sy < h && sx >= 0 && sx < w) {
        sum += static_cast<T>(src[static_cast<size_t>(sy) * w + sx]);
      }
    }
  }
  dst[static_cast<size_t>(y) * ow + x] = sum * static_cast<T>(0.25);
}

int msssim_pool_dim(int n) {
  const int pad = n & 1;
  return (n + 2 * pad - 2) / 2 + 1;
}

void ensure_msssim_scratch(int height, int width) {
  auto& c = ctx();
  const size_t samples =
      static_cast<size_t>(msssim_pool_dim(height)) * static_cast<size_t>(msssim_pool_dim(width));
  if (c.msssim_pyr_samples < samples) {
    if (c.msssim_pyr) {
      c.alloc_bytes -= c.msssim_pyr_samples * 4 * sizeof(double);
      CUDA_CHECK(cudaFree(c.msssim_pyr));
      c.msssim_pyr = nullptr;
      c.msssim_pyr_samples = 0;
    }
    CUDA_CHECK(cudaMalloc(&c.msssim_pyr, samples * 4 * sizeof(double)));
    c.msssim_pyr_samples = samples;
    c.alloc_bytes += samples * 4 * sizeof(double);
  }
  if (!c.d_msssim) {
    CUDA_CHECK(cudaMalloc(&c.d_msssim, 30 * sizeof(double)));
    c.alloc_bytes += 30 * sizeof(double);
  }
}

template <typename T, bool kFormula64>
void launch_msssim_scale(const void* src_a, const void* src_b, bool src_u16, int w, int h, int scale,
                       int plane, double c1, double c2, int pair, bool pool) {
  auto& c = ctx();
  const int ow = w - (kTaps - 1);
  const int oh = h - (kTaps - 1);
  if (ow <= 0 || oh <= 0) throw std::runtime_error("plane too small for MS-SSIM");
  T* maps[5];
  auto* base = static_cast<T*>(c.work);
  for (int i = 0; i < 5; ++i) maps[i] = base + static_cast<size_t>(i) * c.work_samples;
  dim3 block(16, 16);
  dim3 grid((ow + 15) / 16, (h + 15) / 16);
  if (src_u16) {
    msssim_horiz<uint16_t, T><<<grid, block, 0, c.stream>>>(
        static_cast<const uint16_t*>(src_a), static_cast<const uint16_t*>(src_b), w, h, maps[0],
        maps[1], maps[2], maps[3], maps[4]);
  } else {
    msssim_horiz<T, T><<<grid, block, 0, c.stream>>>(static_cast<const T*>(src_a),
                                                    static_cast<const T*>(src_b), w, h, maps[0],
                                                    maps[1], maps[2], maps[3], maps[4]);
  }
  check_launch("msssim_horiz");
  double* acc_cs = c.d_msssim + plane * 10 + scale * 2;
  double* acc_ss = acc_cs + 1;
  CUDA_CHECK(cudaMemsetAsync(acc_cs, 0, 2 * sizeof(double), c.stream));
  msssim_vert<T, kFormula64><<<dim3((ow + kThreads - 1) / kThreads, oh), kThreads, 0, c.stream>>>(
      maps[0], maps[1], maps[2], maps[3], maps[4], ow, h, c1, c2, acc_cs, acc_ss);
  check_launch("msssim_vert");
  if (!pool) return;
  const int nw = msssim_pool_dim(w);
  const int nh = msssim_pool_dim(h);
  auto* bytes = static_cast<char*>(c.msssim_pyr);
  const size_t slot_bytes = c.msssim_pyr_samples * sizeof(double);
  T* dst_a = reinterpret_cast<T*>(bytes + static_cast<size_t>(pair) * 2 * slot_bytes);
  T* dst_b = reinterpret_cast<T*>(bytes + (static_cast<size_t>(pair) * 2 + 1) * slot_bytes);
  dim3 pblock(16, 16);
  dim3 pgrid((nw + 15) / 16, (nh + 15) / 16);
  if (src_u16) {
    msssim_pool2<uint16_t, T><<<pgrid, pblock, 0, c.stream>>>(static_cast<const uint16_t*>(src_a), w, h,
                                                           dst_a, nw, nh);
    msssim_pool2<uint16_t, T><<<pgrid, pblock, 0, c.stream>>>(static_cast<const uint16_t*>(src_b), w, h,
                                                           dst_b, nw, nh);
  } else {
    msssim_pool2<T, T><<<pgrid, pblock, 0, c.stream>>>(static_cast<const T*>(src_a), w, h, dst_a, nw, nh);
    msssim_pool2<T, T><<<pgrid, pblock, 0, c.stream>>>(static_cast<const T*>(src_b), w, h, dst_b, nw, nh);
  }
  check_launch("msssim_pool");
}

void launch_ssim(const uint16_t* a, const uint16_t* b, int w, int h, double* acc, CudaPrecision prec,
                 bool full, double c1, double c2) {
  auto& c = ctx();
  if (w < kTaps || h < kTaps) throw std::runtime_error("plane too small for 11x11 SSIM window");
  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  CUDA_CHECK(cudaMemsetAsync(acc, 0, sizeof(double), c.stream));
  if (prec == CudaPrecision::Fp64) {
    auto* base = static_cast<double*>(c.work);
    double* o[5];
    for (int i = 0; i < 5; ++i) o[i] = base + static_cast<size_t>(i) * c.work_samples;
    filter_h<double><<<dim3((w + kThreads - 1) / kThreads, h), kThreads, 0, c.stream>>>(
        a, b, w, h, o[0], o[1], o[2], o[3], o[4]);
    check_launch("filter_h_fp64");
    filter_v_ssim<double, true><<<dim3((ow + kThreads - 1) / kThreads, oh), kThreads, 0, c.stream>>>(
        o[0], o[1], o[2], o[3], o[4], w, h, c1, c2, full ? 1 : 0, acc);
    check_launch("filter_v_fp64");
  } else if (prec == CudaPrecision::Mixed) {
    auto* base = static_cast<float*>(c.work);
    float* o[5];
    for (int i = 0; i < 5; ++i) o[i] = base + static_cast<size_t>(i) * c.work_samples;
    filter_h<float><<<dim3((w + kThreads - 1) / kThreads, h), kThreads, 0, c.stream>>>(
        a, b, w, h, o[0], o[1], o[2], o[3], o[4]);
    check_launch("filter_h_mixed");
    filter_v_ssim<float, true><<<dim3((ow + kThreads - 1) / kThreads, oh), kThreads, 0, c.stream>>>(
        o[0], o[1], o[2], o[3], o[4], w, h, c1, c2, full ? 1 : 0, acc);
    check_launch("filter_v_mixed");
  } else {
    auto* base = static_cast<float*>(c.work);
    float* o[5];
    for (int i = 0; i < 5; ++i) o[i] = base + static_cast<size_t>(i) * c.work_samples;
    filter_h<float><<<dim3((w + kThreads - 1) / kThreads, h), kThreads, 0, c.stream>>>(
        a, b, w, h, o[0], o[1], o[2], o[3], o[4]);
    check_launch("filter_h_fp32");
    filter_v_ssim<float, false><<<dim3((ow + kThreads - 1) / kThreads, oh), kThreads, 0, c.stream>>>(
        o[0], o[1], o[2], o[3], o[4], w, h, c1, c2, full ? 1 : 0, acc);
    check_launch("filter_v_fp32");
  }
}

__device__ int xpsnr_u16(const uint16_t* p, int pitch, int x, int y) {
  return static_cast<int>(p[static_cast<size_t>(y) * pitch + x]);
}

__device__ int xpsnr_i16(const int16_t* p, int pitch, int x, int y) {
  return static_cast<int>(p[static_cast<size_t>(y) * pitch + x]);
}

__device__ void xpsnr_store_i16(int16_t* p, int pitch, int x, int y, int value) {
  p[static_cast<size_t>(y) * pitch + x] = static_cast<int16_t>(value);
}

__device__ unsigned long long xpsnr_uabs(int v) {
  return static_cast<unsigned long long>(v < 0 ? -v : v);
}

// High-resolution 6-tap. Coordinates are relative to the block origin.
// The caller only visits sites whose taps stay inside the plane.
__device__ int xpsnr_highds_tap(const uint16_t* org, int pitch, int x0, int y0, int y, int x) {
  auto at = [&](int yy, int xx) { return xpsnr_u16(org, pitch, x0 + xx, y0 + yy); };
  return 12 * (at(y, x) + at(y, x + 1) + at(y + 1, x) + at(y + 1, x + 1)) -
         3 * (at(y - 1, x) + at(y - 1, x + 1) + at(y + 2, x) + at(y + 2, x + 1)) -
         3 * (at(y, x - 1) + at(y, x + 2) + at(y + 1, x - 1) + at(y + 1, x + 2)) -
         2 * (at(y - 1, x - 1) + at(y - 1, x + 2) + at(y + 2, x - 1) + at(y + 2, x + 2)) -
         (at(y - 2, x - 1) + at(y - 2, x) + at(y - 2, x + 1) + at(y - 2, x + 2) +
          at(y + 3, x - 1) + at(y + 3, x) + at(y + 3, x + 1) + at(y + 3, x + 2) +
          at(y - 1, x - 2) + at(y, x - 2) + at(y + 1, x - 2) + at(y + 2, x - 2) +
          at(y - 1, x + 3) + at(y, x + 3) + at(y + 1, x + 3) + at(y + 2, x + 3));
}

__device__ int xpsnr_pixel_tap(const uint16_t* org, int pitch, int x0, int y0, int yy, int xx) {
  auto at = [&](int dy, int dx) { return xpsnr_u16(org, pitch, x0 + dx, y0 + dy); };
  return 12 * at(yy, xx) -
         2 * (at(yy, xx - 1) + at(yy, xx + 1) + at(yy - 1, xx) + at(yy + 1, xx)) -
         (at(yy - 1, xx - 1) + at(yy - 1, xx + 1) + at(yy + 1, xx - 1) + at(yy + 1, xx + 1));
}

// One CUDA block per luma block. Integer SSE, spatial activity and temporal
// activity. History is read before it is written. Pitch is in samples.
// Layout is planar uint16 with the code in the low bits.
__global__ void xpsnr_luma_kernel(const uint16_t* __restrict__ org, const uint16_t* __restrict__ rec,
                                  int16_t* __restrict__ m1, int16_t* __restrict__ m2,
                                  const Ctx::XpsnrDesc* __restrict__ desc, unsigned long long* sse,
                                  unsigned long long* sa, unsigned long long* ta, uint8_t* valid,
                                  int pitch, int pic_w, int pic_h, int high_res, int order2) {
  const Ctx::XpsnrDesc d = desc[blockIdx.x];
  const int x0 = d.x;
  const int y0 = d.y;
  const int bw = d.bw;
  const int bh = d.bh;
  const int b_val = high_res ? 2 : 1;
  const int x_act = x0 > 0 ? 0 : b_val;
  const int y_act = y0 > 0 ? 0 : b_val;
  const int w_act = (x0 + bw < pic_w) ? bw : (bw - b_val);
  const int h_act = (y0 + bh < pic_h) ? bh : (bh - b_val);
  const bool active = !(w_act <= x_act || h_act <= y_act);

  unsigned long long local_sse = 0ull;
  unsigned long long local_sa = 0ull;
  unsigned long long local_ta = 0ull;
  const int n = bw * bh;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const int lx = i % bw;
    const int ly = i / bw;
    const int ax = x0 + lx;
    const int ay = y0 + ly;
    const long long diff = static_cast<long long>(xpsnr_u16(org, pitch, ax, ay)) -
                           static_cast<long long>(xpsnr_u16(rec, pitch, ax, ay));
    local_sse += static_cast<unsigned long long>(diff * diff);
  }

  if (active && high_res && w_act > 12) {
    const int nx = (w_act - x_act) / 2;
    const int ny = (h_act - y_act) / 2;
    const int ns = nx * ny;
    for (int i = static_cast<int>(threadIdx.x); i < ns; i += static_cast<int>(blockDim.x)) {
      const int x = x_act + (i % nx) * 2;
      const int y = y_act + (i / nx) * 2;
      local_sa += xpsnr_uabs(xpsnr_highds_tap(org, pitch, x0, y0, y, x));
    }
  } else if (active && !high_res) {
    const int ww = w_act - x_act;
    const int hh = h_act - y_act;
    const int ns = ww * hh;
    for (int i = static_cast<int>(threadIdx.x); i < ns; i += static_cast<int>(blockDim.x)) {
      const int xx = x_act + (i % ww);
      const int yy = y_act + (i / ww);
      local_sa += xpsnr_uabs(xpsnr_pixel_tap(org, pitch, x0, y0, yy, xx));
    }
  }

  if (active && high_res) {
    const int nx = bw / 2;
    const int ny = bh / 2;
    const int ns = nx * ny;
    for (int i = static_cast<int>(threadIdx.x); i < ns; i += static_cast<int>(blockDim.x)) {
      const int lx = (i % nx) * 2;
      const int ly = (i / nx) * 2;
      const int ax = x0 + lx;
      const int ay = y0 + ly;
      const int cur = xpsnr_u16(org, pitch, ax, ay) + xpsnr_u16(org, pitch, ax + 1, ay) +
                      xpsnr_u16(org, pitch, ax, ay + 1) + xpsnr_u16(org, pitch, ax + 1, ay + 1);
      const int p1 = xpsnr_i16(m1, pitch, ax, ay) + xpsnr_i16(m1, pitch, ax + 1, ay) +
                     xpsnr_i16(m1, pitch, ax, ay + 1) + xpsnr_i16(m1, pitch, ax + 1, ay + 1);
      int t = cur - p1;
      int p2s[4];
      if (order2) {
        p2s[0] = xpsnr_i16(m2, pitch, ax, ay);
        p2s[1] = xpsnr_i16(m2, pitch, ax + 1, ay);
        p2s[2] = xpsnr_i16(m2, pitch, ax, ay + 1);
        p2s[3] = xpsnr_i16(m2, pitch, ax + 1, ay + 1);
        const int p2 = p2s[0] + p2s[1] + p2s[2] + p2s[3];
        t = cur - 2 * p1 + p2;
      }
      const int old0 = xpsnr_i16(m1, pitch, ax, ay);
      const int old1 = xpsnr_i16(m1, pitch, ax + 1, ay);
      const int old2 = xpsnr_i16(m1, pitch, ax, ay + 1);
      const int old3 = xpsnr_i16(m1, pitch, ax + 1, ay + 1);
      local_ta += 2ull * xpsnr_uabs(t);
      if (order2) {
        xpsnr_store_i16(m2, pitch, ax, ay, old0);
        xpsnr_store_i16(m2, pitch, ax + 1, ay, old1);
        xpsnr_store_i16(m2, pitch, ax, ay + 1, old2);
        xpsnr_store_i16(m2, pitch, ax + 1, ay + 1, old3);
      }
      xpsnr_store_i16(m1, pitch, ax, ay, xpsnr_u16(org, pitch, ax, ay));
      xpsnr_store_i16(m1, pitch, ax + 1, ay, xpsnr_u16(org, pitch, ax + 1, ay));
      xpsnr_store_i16(m1, pitch, ax, ay + 1, xpsnr_u16(org, pitch, ax, ay + 1));
      xpsnr_store_i16(m1, pitch, ax + 1, ay + 1, xpsnr_u16(org, pitch, ax + 1, ay + 1));
    }
  } else if (active && !high_res) {
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
      const int lx = i % bw;
      const int ly = i / bw;
      const int ax = x0 + lx;
      const int ay = y0 + ly;
      const int cur = xpsnr_u16(org, pitch, ax, ay);
      const int p1 = xpsnr_i16(m1, pitch, ax, ay);
      const int p2 = xpsnr_i16(m2, pitch, ax, ay);
      const int t = order2 ? (cur - 2 * p1 + p2) : (cur - p1);
      local_ta += 2ull * xpsnr_uabs(t);
      if (order2) xpsnr_store_i16(m2, pitch, ax, ay, p1);
      xpsnr_store_i16(m1, pitch, ax, ay, cur);
    }
  }

  const unsigned long long sse_sum = block_sum_u64(local_sse);
  const unsigned long long sa_sum = block_sum_u64(local_sa);
  const unsigned long long ta_sum = block_sum_u64(local_ta);
  if (threadIdx.x == 0) {
    sse[blockIdx.x] = sse_sum;
    sa[blockIdx.x] = sa_sum;
    ta[blockIdx.x] = ta_sum;
    valid[blockIdx.x] = active ? 1 : 0;
  }
}

// One CUDA block per chroma block, raster order matching the luma weight table.
__global__ void xpsnr_chroma_kernel(const uint16_t* __restrict__ org_u,
                                    const uint16_t* __restrict__ rec_u,
                                    const uint16_t* __restrict__ org_v,
                                    const uint16_t* __restrict__ rec_v, int pitch, int plane_w,
                                    int plane_h, int bx, int by, int n_x, unsigned long long* sse_u,
                                    unsigned long long* sse_v) {
  const int col = static_cast<int>(blockIdx.x) % n_x;
  const int row = static_cast<int>(blockIdx.x) / n_x;
  const int x0 = col * bx;
  const int y0 = row * by;
  const int bw = bx < (plane_w - x0) ? bx : (plane_w - x0);
  const int bh = by < (plane_h - y0) ? by : (plane_h - y0);
  unsigned long long local_u = 0ull;
  unsigned long long local_v = 0ull;
  const int n = bw * bh;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const int lx = i % bw;
    const int ly = i / bw;
    const int ax = x0 + lx;
    const int ay = y0 + ly;
    const long long du = static_cast<long long>(xpsnr_u16(org_u, pitch, ax, ay)) -
                         static_cast<long long>(xpsnr_u16(rec_u, pitch, ax, ay));
    const long long dv = static_cast<long long>(xpsnr_u16(org_v, pitch, ax, ay)) -
                         static_cast<long long>(xpsnr_u16(rec_v, pitch, ax, ay));
    local_u += static_cast<unsigned long long>(du * du);
    local_v += static_cast<unsigned long long>(dv * dv);
  }
  const unsigned long long sum_u = block_sum_u64(local_u);
  const unsigned long long sum_v = block_sum_u64(local_v);
  if (threadIdx.x == 0) {
    sse_u[blockIdx.x] = sum_u;
    sse_v[blockIdx.x] = sum_v;
  }
}

void ensure_xpsnr_history(size_t samples) {
  auto& c = ctx();
  if (c.xpsnr_m1 && c.xpsnr_m2 && c.xpsnr_hist_cap >= samples) return;
  if (c.xpsnr_m1) {
    c.alloc_bytes -= c.xpsnr_hist_cap * sizeof(int16_t);
    CUDA_CHECK(cudaFree(c.xpsnr_m1));
    c.xpsnr_m1 = nullptr;
  }
  if (c.xpsnr_m2) {
    c.alloc_bytes -= c.xpsnr_hist_cap * sizeof(int16_t);
    CUDA_CHECK(cudaFree(c.xpsnr_m2));
    c.xpsnr_m2 = nullptr;
  }
  c.xpsnr_hist_cap = 0;
  CUDA_CHECK(cudaMalloc(&c.xpsnr_m1, samples * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc(&c.xpsnr_m2, samples * sizeof(int16_t)));
  c.xpsnr_hist_cap = samples;
  c.alloc_bytes += samples * sizeof(int16_t) * 2;
}

void ensure_xpsnr_blocks(int n) {
  auto& c = ctx();
  if (c.blk_cap >= n && c.d_blk_sse) return;
  auto drop = [&](void* ptr, size_t bytes) {
    if (!ptr) return;
    c.alloc_bytes -= bytes;
    CUDA_CHECK(cudaFree(ptr));
  };
  drop(c.d_blk_sse, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  drop(c.d_blk_sa, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  drop(c.d_blk_ta, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  drop(c.d_blk_valid, static_cast<size_t>(c.blk_cap) * sizeof(uint8_t));
  drop(c.d_chr_u, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  drop(c.d_chr_v, static_cast<size_t>(c.blk_cap) * sizeof(unsigned long long));
  c.d_blk_sse = c.d_blk_sa = c.d_blk_ta = c.d_chr_u = c.d_chr_v = nullptr;
  c.d_blk_valid = nullptr;
  c.blk_cap = 0;
  CUDA_CHECK(cudaMalloc(&c.d_blk_sse, static_cast<size_t>(n) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&c.d_blk_sa, static_cast<size_t>(n) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&c.d_blk_ta, static_cast<size_t>(n) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&c.d_blk_valid, static_cast<size_t>(n) * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&c.d_chr_u, static_cast<size_t>(n) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&c.d_chr_v, static_cast<size_t>(n) * sizeof(unsigned long long)));
  c.blk_cap = n;
  c.alloc_bytes += static_cast<size_t>(n) *
                   (5 * sizeof(unsigned long long) + sizeof(uint8_t));
}

void xpsnr_launch_on_uploaded(const VideoFormat& fmt, int integer_rate) {
  auto& c = ctx();
  if (!c.xpsnr_reset_done) {
    throw std::runtime_error("CUDA XPSNR: history was not reset");
  }
  if (fmt.width != c.xpsnr_w || fmt.height != c.xpsnr_h || fmt.bit_depth != c.xpsnr_depth ||
      static_cast<int>(fmt.chroma) != c.xpsnr_chroma || integer_rate != c.xpsnr_rate) {
    throw std::runtime_error("CUDA XPSNR: frame format does not match cuda_xpsnr_reset");
  }
  if (!xpsnr_geometry_supported(fmt.width, fmt.height)) {
    throw std::runtime_error(
        "xpsnr: odd luma width or height is not supported when width*height is above "
        "2048*1152; the downsampled high-pass would read past the plane");
  }
  const int b = xpsnr_block_size(fmt.width, fmt.height);
  c.xpsnr_raw = b < 4;
  c.xpsnr_nblk = 0;
  if (c.xpsnr_raw) {
    if (!c.d_raw_sse) {
      CUDA_CHECK(cudaMalloc(&c.d_raw_sse, 3 * sizeof(unsigned long long)));
      c.alloc_bytes += 3 * sizeof(unsigned long long);
    }
    CUDA_CHECK(cudaMemsetAsync(c.d_raw_sse, 0, 3 * sizeof(unsigned long long), c.stream));
    const int pw[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
    const int ph[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
    for (int p = 0; p < 3; ++p) {
      launch_sse(c.plane[p][0], c.plane[p][1], pw[p], ph[p], c.d_raw_sse + p);
    }
    return;
  }
  const int w_blk = (fmt.width + b - 1) / b;
  const int h_blk = (fmt.height + b - 1) / b;
  const int n = w_blk * h_blk;
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const int bx = b * cw / fmt.width;
  const int by = b * ch / fmt.height;
  if (bx <= 0 || by <= 0) {
    throw std::runtime_error("xpsnr: chroma block size is zero for this geometry");
  }
  if ((cw + bx - 1) / bx != w_blk || (ch + by - 1) / by != h_blk) {
    throw std::runtime_error("xpsnr: chroma block grid does not match luma weights");
  }
  if (c.desc_w != fmt.width || c.desc_h != fmt.height) {
    c.h_desc.resize(static_cast<size_t>(n));
    int i = 0;
    for (int y = 0; y < fmt.height; y += b) {
      const int bh = (y + b > fmt.height) ? (fmt.height - y) : b;
      for (int x = 0; x < fmt.width; x += b, ++i) {
        const int bw = (x + b > fmt.width) ? (fmt.width - x) : b;
        c.h_desc[static_cast<size_t>(i)] = Ctx::XpsnrDesc{x, y, bw, bh};
      }
    }
    if (c.desc_cap < n) {
      if (c.d_desc) {
        c.alloc_bytes -= static_cast<size_t>(c.desc_cap) * sizeof(Ctx::XpsnrDesc);
        CUDA_CHECK(cudaFree(c.d_desc));
        c.d_desc = nullptr;
        c.desc_cap = 0;
      }
      CUDA_CHECK(cudaMalloc(&c.d_desc, static_cast<size_t>(n) * sizeof(Ctx::XpsnrDesc)));
      c.desc_cap = n;
      c.alloc_bytes += static_cast<size_t>(n) * sizeof(Ctx::XpsnrDesc);
    }
    CUDA_CHECK(cudaMemcpyAsync(c.d_desc, c.h_desc.data(),
                               static_cast<size_t>(n) * sizeof(Ctx::XpsnrDesc),
                               cudaMemcpyHostToDevice, c.stream));
    c.desc_w = fmt.width;
    c.desc_h = fmt.height;
  }
  ensure_xpsnr_blocks(n);
  c.xpsnr_nblk = n;
  const int high = xpsnr_high_resolution(fmt.width, fmt.height) ? 1 : 0;
  const int order2 = integer_rate >= 32 ? 1 : 0;
  xpsnr_luma_kernel<<<n, kThreads, 0, c.stream>>>(
      c.plane[0][0], c.plane[0][1], c.xpsnr_m1, c.xpsnr_m2, c.d_desc, c.d_blk_sse, c.d_blk_sa,
      c.d_blk_ta, c.d_blk_valid, fmt.width, fmt.width, fmt.height, high, order2);
  check_launch("xpsnr_luma");
  xpsnr_chroma_kernel<<<n, kThreads, 0, c.stream>>>(
      c.plane[1][0], c.plane[1][1], c.plane[2][0], c.plane[2][1], cw, cw, ch, bx, by, w_blk,
      c.d_chr_u, c.d_chr_v);
  check_launch("xpsnr_chroma");
}

void xpsnr_schedule_d2h() {
  auto& c = ctx();
  if (c.xpsnr_raw) return;
  const int n = c.xpsnr_nblk;
  c.h_blk_sse.resize(static_cast<size_t>(n));
  c.h_blk_sa.resize(static_cast<size_t>(n));
  c.h_blk_ta.resize(static_cast<size_t>(n));
  c.h_blk_valid.resize(static_cast<size_t>(n));
  c.h_chr_u.resize(static_cast<size_t>(n));
  c.h_chr_v.resize(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpyAsync(c.h_blk_sse.data(), c.d_blk_sse,
                             static_cast<size_t>(n) * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, c.stream));
  CUDA_CHECK(cudaMemcpyAsync(c.h_blk_sa.data(), c.d_blk_sa,
                             static_cast<size_t>(n) * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, c.stream));
  CUDA_CHECK(cudaMemcpyAsync(c.h_blk_ta.data(), c.d_blk_ta,
                             static_cast<size_t>(n) * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, c.stream));
  CUDA_CHECK(cudaMemcpyAsync(c.h_blk_valid.data(), c.d_blk_valid,
                             static_cast<size_t>(n) * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                             c.stream));
  CUDA_CHECK(cudaMemcpyAsync(c.h_chr_u.data(), c.d_chr_u,
                             static_cast<size_t>(n) * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, c.stream));
  CUDA_CHECK(cudaMemcpyAsync(c.h_chr_v.data(), c.d_chr_v,
                             static_cast<size_t>(n) * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, c.stream));
}

void xpsnr_finish_wsse(const VideoFormat& fmt, CudaXpsnrRequest* req, const unsigned long long* raw) {
  auto& c = ctx();
  uint64_t wsse[3] = {};
  if (c.xpsnr_raw) {
    wsse[0] = raw[0];
    wsse[1] = raw[1];
    wsse[2] = raw[2];
  } else {
    std::vector<XpsnrBlockStats> luma(static_cast<size_t>(c.xpsnr_nblk));
    std::vector<uint64_t> su(static_cast<size_t>(c.xpsnr_nblk));
    std::vector<uint64_t> sv(static_cast<size_t>(c.xpsnr_nblk));
    for (int i = 0; i < c.xpsnr_nblk; ++i) {
      luma[static_cast<size_t>(i)].sse = c.h_blk_sse[static_cast<size_t>(i)];
      luma[static_cast<size_t>(i)].sa = c.h_blk_sa[static_cast<size_t>(i)];
      luma[static_cast<size_t>(i)].ta = c.h_blk_ta[static_cast<size_t>(i)];
      luma[static_cast<size_t>(i)].activity_valid = c.h_blk_valid[static_cast<size_t>(i)];
      su[static_cast<size_t>(i)] = c.h_chr_u[static_cast<size_t>(i)];
      sv[static_cast<size_t>(i)] = c.h_chr_v[static_cast<size_t>(i)];
    }
    xpsnr_wsse_from_block_stats(fmt, luma.data(), su.data(), sv.data(), wsse);
  }
  req->wsse_y = wsse[0];
  req->wsse_u = wsse[1];
  req->wsse_v = wsse[2];
}

struct PlaneIn {
  const uint16_t* ref;
  const uint16_t* dist;
  int w, h, sref, sdist;
};

}  // namespace

void cuda_shutdown() { cuda_shutdown_impl(); }

void cuda_print_device_info() {
  init_once();
  const auto& c = ctx();
  std::fprintf(stderr,
               "pixvqa cuda: device=%s cc=%d.%d vram_mib=%zu sms=%d fp32_per_fp64=%d "
               "driver=%d.%d runtime=%d.%d\n",
               c.name.c_str(), c.cc_major, c.cc_minor, c.vram / (1024 * 1024), c.sms, c.fp64_ratio,
               c.driver / 1000, (c.driver % 1000) / 10, c.runtime / 1000, (c.runtime % 1000) / 10);
}

void cuda_dump_profile() {
  if (!profile_on()) return;
  const auto& c = ctx();
  std::fprintf(stderr,
               "[PIXVQA_CUDA_PROFILE] h2d=%.1fms kernel=%.1fms d2h=%.1fms frames=%d "
               "alloc_mib=%.1f xpsnr=%.1fms device=%s h2d_test=%d\n",
               c.ms_h2d, c.ms_kernel, c.ms_d2h, c.frames, c.alloc_bytes / (1024.0 * 1024.0), c.ms_xpsnr,
               c.name.c_str(), c.h2d_test_copies);
}

CudaProfileSnapshot cuda_profile_snapshot() {
  CudaProfileSnapshot s;
  if (!ctx().ready) return s;
  const auto& c = ctx();
  s.h2d_ms = c.ms_h2d;
  s.kernel_ms = c.ms_kernel;
  s.d2h_ms = c.ms_d2h;
  s.xpsnr_ms = c.ms_xpsnr;
  s.frames = c.frames;
  s.h2d_test_copies = c.h2d_test_copies;
  return s;
}

void cuda_profile_reset() {
  if (!ctx().ready) return;
  auto& c = ctx();
  c.ms_h2d = 0;
  c.ms_kernel = 0;
  c.ms_d2h = 0;
  c.ms_xpsnr = 0;
  c.frames = 0;
  c.h2d_test_copies = 0;
}

void cuda_xpsnr_reset(const VideoFormat& fmt, int integer_rate) {
  if (fmt.bit_depth != 8 && fmt.bit_depth != 10) {
    throw std::runtime_error("CUDA XPSNR supports 8-bit and 10-bit only");
  }
  if (integer_rate <= 0) {
    throw std::runtime_error("xpsnr: frame rate must be a positive integer or rational");
  }
  if (!xpsnr_geometry_supported(fmt.width, fmt.height)) {
    throw std::runtime_error(
        "xpsnr: odd luma width or height is not supported when width*height is above "
        "2048*1152; the downsampled high-pass would read past the plane");
  }
  init_once();
  auto& c = ctx();
  const size_t n = static_cast<size_t>(fmt.width) * static_cast<size_t>(fmt.height);
  ensure_xpsnr_history(n);
  CUDA_CHECK(cudaMemsetAsync(c.xpsnr_m1, 0, n * sizeof(int16_t), c.stream));
  CUDA_CHECK(cudaMemsetAsync(c.xpsnr_m2, 0, n * sizeof(int16_t), c.stream));
  c.xpsnr_w = fmt.width;
  c.xpsnr_h = fmt.height;
  c.xpsnr_depth = fmt.bit_depth;
  c.xpsnr_chroma = static_cast<int>(fmt.chroma);
  c.xpsnr_rate = integer_rate;
  c.xpsnr_reset_done = true;
  c.desc_w = 0;
  c.desc_h = 0;
}

void cuda_score_device_frame(const Frame* host_ref, const DevicePlanarFrame* device_ref,
                             const Frame* host_test, const DevicePlanarFrame* device_test,
                             const VideoFormat& fmt, double peak, CudaPrecision prec, bool do_psnr,
                             bool do_ssim, bool do_msssim, PsnrFrame* psnr, PlaneScore* ssim,
                             PlaneScore* msssim, CudaXpsnrRequest* xpsnr) {
  if (device_ref) {
    throw std::runtime_error("CUDA device reference frames are not supported");
  }
  if (!host_ref) throw std::runtime_error("CUDA host reference frame is required");
  if (!host_test && !device_test) {
    throw std::runtime_error("CUDA test frame is missing");
  }
  const bool sample_metrics = do_psnr || do_ssim;
  if (!sample_metrics && !do_msssim && !xpsnr) {
    throw std::runtime_error("CUDA: no metric requested");
  }
  // Host frames store 8-bit codes in uint16. The SSE and SSIM kernels read
  // those codes and use the caller-supplied peak, so 8-bit and 10-bit share them.
  if (sample_metrics && fmt.bit_depth != 8 && fmt.bit_depth != 10) {
    throw std::runtime_error("CUDA PSNR and SSIM support 8-bit and 10-bit only");
  }
  if (do_msssim && fmt.bit_depth != 8 && fmt.bit_depth != 10) {
    throw std::runtime_error("CUDA MS-SSIM supports 8-bit and 10-bit only");
  }
  if (xpsnr && fmt.bit_depth != 8 && fmt.bit_depth != 10) {
    throw std::runtime_error("CUDA XPSNR supports 8-bit and 10-bit only");
  }
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const bool prof = profile_on();
  if (prof) CUDA_CHECK(cudaEventRecord(c.ev_a, c.stream));

  const DevicePlane* dev_test[3] = {
      device_test ? &device_test->y : nullptr,
      device_test ? &device_test->u : nullptr,
      device_test ? &device_test->v : nullptr,
  };
  const PlaneIn planes[3] = {
      {host_ref->y.ptr(), host_test ? host_test->y.ptr() : nullptr, fmt.width, fmt.height,
       host_ref->y.stride, host_test ? host_test->y.stride : 0},
      {host_ref->u.ptr(), host_test ? host_test->u.ptr() : nullptr, fmt.chroma_width(),
       fmt.chroma_height(), host_ref->u.stride, host_test ? host_test->u.stride : 0},
      {host_ref->v.ptr(), host_test ? host_test->v.ptr() : nullptr, fmt.chroma_width(),
       fmt.chroma_height(), host_ref->v.stride, host_test ? host_test->v.stride : 0},
  };
  size_t pin_off = 0;
  int test_copies = 0;
  for (int p = 0; p < 3; ++p) {
    pin_off = upload_plane(planes[p].ref, planes[p].sref, planes[p].w, planes[p].h, c.plane[p][0],
                           pin_off);
    if (host_test) {
      pin_off = upload_plane(planes[p].dist, planes[p].sdist, planes[p].w, planes[p].h,
                             c.plane[p][1], pin_off);
      ++test_copies;
    } else {
      const DevicePlane* dp = dev_test[p];
      if (!dp || dp->ptr != c.plane[p][1] || dp->pitch_samples != dp->width ||
          dp->width != planes[p].w || dp->height != planes[p].h) {
        throw std::runtime_error(
            "CUDA device test plane must be the metric buffer with pitch_samples equal to width");
      }
    }
  }
  c.h2d_test_copies += test_copies;
  if (prof) CUDA_CHECK(cudaEventRecord(c.ev_b, c.stream));
  // Same stream as the uploads above, so the copy observes those pixels.
  // Later kernels are queued after the hook and may overwrite plane[][].
  // A null hook does not record events or launch work.
  if (c.snapshot_hook) c.snapshot_hook(fmt, c.stream);

  CUDA_CHECK(cudaMemsetAsync(c.d_sse, 0, 3 * sizeof(unsigned long long), c.stream));
  CUDA_CHECK(cudaMemsetAsync(c.d_acc, 0, 32 * sizeof(double), c.stream));

  const double c1 = (0.01 * peak) * (0.01 * peak);
  const double c2 = (0.03 * peak) * (0.03 * peak);

  if (do_psnr) {
    for (int p = 0; p < 3; ++p) {
      launch_sse(c.plane[p][0], c.plane[p][1], planes[p].w, planes[p].h, c.d_sse + p);
    }
  }
  // XPSNR reads the uploaded full-resolution frames. PSNR, SSIM, and
  // five-scale MS-SSIM only read those planes.
  if (xpsnr) {
    if (prof) CUDA_CHECK(cudaEventRecord(c.ev_x, c.stream));
    xpsnr_launch_on_uploaded(fmt, xpsnr->integer_rate);
    if (prof) CUDA_CHECK(cudaEventRecord(c.ev_y, c.stream));
  }
  if (do_ssim) {
    for (int p = 0; p < 3; ++p) {
      launch_ssim(c.plane[p][0], c.plane[p][1], planes[p].w, planes[p].h, c.d_acc + p, prec,
                  /*full=*/true, c1, c2);
    }
  }
  if (do_msssim) {
    ensure_msssim_scratch(fmt.height, fmt.width);
    const bool f64 = prec == CudaPrecision::Fp64;
    const bool formula64 = prec != CudaPrecision::Fp32;
    for (int p = 0; p < 3; ++p) {
      const int scales = msssim_scales(planes[p].h, planes[p].w);
      if (scales < 1) {
        throw std::runtime_error("plane too small for MS-SSIM (need at least 11x11)");
      }
      const void* src_a = c.plane[p][0];
      const void* src_b = c.plane[p][1];
      bool src_u16 = true;
      int cw = planes[p].w;
      int ch = planes[p].h;
      int pair = 0;
      for (int scale = 0; scale < scales; ++scale) {
        const bool pool = scale + 1 < scales;
        const int next_pair = src_u16 ? 0 : (pair ^ 1);
        if (f64) {
          launch_msssim_scale<double, true>(src_a, src_b, src_u16, cw, ch, scale, p, c1, c2,
                                          next_pair, pool);
        } else if (formula64) {
          launch_msssim_scale<float, true>(src_a, src_b, src_u16, cw, ch, scale, p, c1, c2, next_pair,
                                         pool);
        } else {
          launch_msssim_scale<float, false>(src_a, src_b, src_u16, cw, ch, scale, p, c1, c2,
                                          next_pair, pool);
        }
        if (pool) {
          auto* bytes = static_cast<char*>(c.msssim_pyr);
          const size_t slot_bytes = c.msssim_pyr_samples * sizeof(double);
          if (f64) {
            src_a = bytes + static_cast<size_t>(next_pair) * 2 * slot_bytes;
            src_b = bytes + (static_cast<size_t>(next_pair) * 2 + 1) * slot_bytes;
          } else {
            src_a = bytes + static_cast<size_t>(next_pair) * 2 * slot_bytes;
            src_b = bytes + (static_cast<size_t>(next_pair) * 2 + 1) * slot_bytes;
          }
          src_u16 = false;
          pair = next_pair;
          cw = msssim_pool_dim(cw);
          ch = msssim_pool_dim(ch);
        }
      }
    }
  }
  if (prof) CUDA_CHECK(cudaEventRecord(c.ev_c, c.stream));

  unsigned long long h_sse[3] = {};
  unsigned long long h_raw[3] = {};
  double h_acc[32] = {};
  double h_msssim[30] = {};
  if (do_psnr) {
    CUDA_CHECK(cudaMemcpyAsync(h_sse, c.d_sse, 3 * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, c.stream));
  }
  if (do_ssim) {
    CUDA_CHECK(cudaMemcpyAsync(h_acc, c.d_acc, 32 * sizeof(double), cudaMemcpyDeviceToHost, c.stream));
  }
  if (do_msssim) {
    CUDA_CHECK(cudaMemcpyAsync(h_msssim, c.d_msssim, 30 * sizeof(double), cudaMemcpyDeviceToHost, c.stream));
  }
  if (xpsnr) {
    if (c.xpsnr_raw) {
      CUDA_CHECK(cudaMemcpyAsync(h_raw, c.d_raw_sse, 3 * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, c.stream));
    } else {
      xpsnr_schedule_d2h();
    }
  }
  if (prof) CUDA_CHECK(cudaEventRecord(c.ev_d, c.stream));
  CUDA_CHECK(cudaStreamSynchronize(c.stream));

  if (prof) {
    float t = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t, c.ev_a, c.ev_b));
    c.ms_h2d += t;
    CUDA_CHECK(cudaEventElapsedTime(&t, c.ev_b, c.ev_c));
    c.ms_kernel += t;
    if (xpsnr) {
      CUDA_CHECK(cudaEventElapsedTime(&t, c.ev_x, c.ev_y));
      c.ms_xpsnr += t;
    }
    CUDA_CHECK(cudaEventElapsedTime(&t, c.ev_c, c.ev_d));
    c.ms_d2h += t;
  }
  if (xpsnr) xpsnr_finish_wsse(fmt, xpsnr, h_raw);
  c.frames += 1;

  if (do_psnr && psnr) {
    const double inv_y = 1.0 / static_cast<double>(planes[0].w * planes[0].h);
    const double inv_c = 1.0 / static_cast<double>(planes[1].w * planes[1].h);
    psnr->mse.y = static_cast<double>(h_sse[0]) * inv_y;
    psnr->mse.u = static_cast<double>(h_sse[1]) * inv_c;
    psnr->mse.v = static_cast<double>(h_sse[2]) * inv_c;
    psnr->mse.avg = yuv_sample_weight_avg(psnr->mse.y, psnr->mse.u, psnr->mse.v, fmt.chroma);
    psnr->psnr.y = mse_to_psnr(psnr->mse.y, peak);
    psnr->psnr.u = mse_to_psnr(psnr->mse.u, peak);
    psnr->psnr.v = mse_to_psnr(psnr->mse.v, peak);
    psnr->psnr.avg = mse_to_psnr(psnr->mse.avg, peak);
  }
  auto finish_plane = [&](int plane, int base_acc) {
    const int cw = planes[plane].w;
    const int ch = planes[plane].h;
    const int ow = cw - 2 * kRadius;
    const int oh = ch - 2 * kRadius;
    if (ow <= 0 || oh <= 0) throw std::runtime_error("plane too small for SSIM at scale");
    const double n = static_cast<double>(ow) * static_cast<double>(oh);
    return h_acc[base_acc] / n;
  };
  if (do_ssim && ssim) {
    ssim->y = finish_plane(0, 0);
    ssim->u = finish_plane(1, 1);
    ssim->v = finish_plane(2, 2);
    ssim->avg = yuv_sample_weight_avg(ssim->y, ssim->u, ssim->v, fmt.chroma);
  }
  if (do_msssim && msssim) {
    const char* dbg = pix_getenv("MSSSIM_DEBUG");
    const bool debug = dbg && dbg[0] == '1';
    const char* names[3] = {"y", "u", "v"};
    constexpr double kProd[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
    double scores[3] = {};
    for (int p = 0; p < 3; ++p) {
      const int scales = msssim_scales(planes[p].h, planes[p].w);
      int cw = planes[p].w;
      int ch = planes[p].h;
      double prod = 1.0;
      for (int s = 0; s < scales; ++s) {
        const double n = static_cast<double>(cw - (kTaps - 1)) * static_cast<double>(ch - (kTaps - 1));
        const double cs = h_msssim[p * 10 + s * 2] / n;
        const double ssim = h_msssim[p * 10 + s * 2 + 1] / n;
        if (debug) {
          std::fprintf(stderr, "msssim_comp %s %d %d %d %.12g %.12g\n", names[p], s, ch, cw, cs, ssim);
        }
        const double term = std::max(0.0, (s + 1 == scales) ? ssim : cs);
        prod *= std::pow(term, kProd[s]);
        if (s + 1 < scales) {
          cw = msssim_pool_dim(cw);
          ch = msssim_pool_dim(ch);
        }
      }
      scores[p] = prod;
    }
    msssim->y = scores[0];
    msssim->u = scores[1];
    msssim->v = scores[2];
    msssim->avg = yuv_sample_weight_avg(msssim->y, msssim->u, msssim->v, fmt.chroma);
  }
}

void cuda_ensure_primary_context() {
  init_once();
  auto& c = ctx();
  if (c.primary) return;
  const CUresult er = cuCtxGetCurrent(&c.primary);
  if (er != CUDA_SUCCESS || !c.primary) {
    throw std::runtime_error("CUDA primary context is not current");
  }
}

uintptr_t cuda_primary_context_handle() {
  cuda_ensure_primary_context();
  return reinterpret_cast<uintptr_t>(ctx().primary);
}

void cuda_convert_p010_to_test_planes(const P010DeviceSurface& src, const VideoFormat& fmt) {
  if (fmt.chroma != ChromaFormat::YUV420 || fmt.bit_depth != 10) {
    throw std::runtime_error("P010 conversion supports 420p:10 only");
  }
  if (!src.y || !src.uv || src.y_pitch_bytes < fmt.width * 2 || src.uv_pitch_bytes < fmt.width * 2) {
    throw std::runtime_error("P010 surface pitch is smaller than the picture");
  }
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const dim3 block(32, 8);
  const dim3 grid_y((fmt.width + 31) / 32, (fmt.height + 7) / 8);
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const dim3 grid_c((cw + 31) / 32, (ch + 7) / 8);
  p010_y_kernel<<<grid_y, block, 0, c.stream>>>(static_cast<const uint8_t*>(src.y), src.y_pitch_bytes,
                                                c.plane[0][1], fmt.width, fmt.height);
  p010_uv_kernel<<<grid_c, block, 0, c.stream>>>(static_cast<const uint8_t*>(src.uv), src.uv_pitch_bytes,
                                                 c.plane[1][1], c.plane[2][1], cw, ch);
  check_launch("p010 conversion");
  CUDA_CHECK(cudaStreamSynchronize(c.stream));
}

void cuda_download_test_planes(Frame& out, const VideoFormat& fmt) {
  init_once();
  ensure_storage(fmt);
  out.allocate(fmt);
  auto& c = ctx();
  const size_t ny = static_cast<size_t>(fmt.width) * fmt.height;
  const size_t nc = static_cast<size_t>(fmt.chroma_width()) * fmt.chroma_height();
  CUDA_CHECK(cudaMemcpy(out.y.ptr(), c.plane[0][1], ny * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.u.ptr(), c.plane[1][1], nc * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.v.ptr(), c.plane[2][1], nc * sizeof(uint16_t), cudaMemcpyDeviceToHost));
}

DevicePlanarFrame pack_frame(uint16_t* y, uint16_t* u, uint16_t* v, int w, int h, int cw, int ch) {
  DevicePlanarFrame out;
  out.y = DevicePlane{y, w, h, w};
  out.u = DevicePlane{u, cw, ch, cw};
  out.v = DevicePlane{v, cw, ch, cw};
  return out;
}

CudaWritableFrame pack_writable(uint16_t* y, uint16_t* u, uint16_t* v, int w, int h, int cw, int ch) {
  CudaWritableFrame out;
  out.y = CudaWritablePlane{y, w, h, w};
  out.u = CudaWritablePlane{u, cw, ch, cw};
  out.v = CudaWritablePlane{v, cw, ch, cw};
  return out;
}

DevicePlanarFrame cuda_metric_test_frame(const VideoFormat& fmt) {
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  return pack_frame(c.plane[0][1], c.plane[1][1], c.plane[2][1], fmt.width, fmt.height, cw, ch);
}

DevicePlanarFrame cuda_metric_ref_frame(const VideoFormat& fmt) {
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  return pack_frame(c.plane[0][0], c.plane[1][0], c.plane[2][0], fmt.width, fmt.height, cw, ch);
}

CudaWritableFrame cuda_metric_writable_ref(const VideoFormat& fmt) {
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  return pack_writable(c.plane[0][0], c.plane[1][0], c.plane[2][0], fmt.width, fmt.height,
                       fmt.chroma_width(), fmt.chroma_height());
}

CudaWritableFrame cuda_metric_writable_test(const VideoFormat& fmt) {
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  return pack_writable(c.plane[0][1], c.plane[1][1], c.plane[2][1], fmt.width, fmt.height,
                       fmt.chroma_width(), fmt.chroma_height());
}

void* cuda_metric_stream_handle() {
  init_once();
  return ctx().stream;
}

std::size_t cuda_metric_tracked_bytes() {
  if (!ctx().ready) return 0;
  return ctx().alloc_bytes;
}

void cuda_set_frame_snapshot_hook(CudaFrameSnapshotHook hook) { ctx().snapshot_hook = hook; }

void require_device_ptr(const void* ptr, const char* what) {
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string(what) + " is not a CUDA device pointer");
  }
  if (attr.type != cudaMemoryTypeDevice) {
    throw std::runtime_error(std::string(what) + " is not device memory");
  }
}

void pack_to_pin(const uint16_t* src, int stride, int w, int h, uint16_t* pin) {
  if (stride == w) {
    std::memcpy(pin, src, static_cast<size_t>(w) * h * sizeof(uint16_t));
    return;
  }
  for (int y = 0; y < h; ++y) {
    std::memcpy(pin + static_cast<size_t>(y) * w, src + static_cast<size_t>(y) * stride,
                static_cast<size_t>(w) * sizeof(uint16_t));
  }
}

void record_event(void* event, cudaStream_t stream) {
  if (!event) return;
  CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(event), stream));
}

void cuda_stage_host_yuv420p10(const CudaHostPlanar& ref, const CudaHostPlanar& dist,
                               const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                               void* ev_snapshot_end, double* pack_ms) {
  if (fmt.bit_depth != 10 || fmt.chroma != ChromaFormat::YUV420) {
    throw std::runtime_error("CUDA stage supports 10-bit 4:2:0 only");
  }
  if (fmt.width < 2 || fmt.height < 2 || (fmt.width & 1) || (fmt.height & 1)) {
    throw std::runtime_error("CUDA stage requires positive even dimensions");
  }
  const CudaHostPlanar pics[2] = {ref, dist};
  const int ws[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
  const int hs[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
  for (const CudaHostPlanar& pic : pics) {
    const CudaHostPlane planes[3] = {pic.y, pic.u, pic.v};
    for (int p = 0; p < 3; ++p) {
      if (!planes[p].ptr || planes[p].width != ws[p] || planes[p].height != hs[p] ||
          planes[p].pitch_samples < ws[p]) {
        throw std::runtime_error("CUDA stage host plane is missing or has a short pitch");
      }
    }
  }
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const size_t ny = static_cast<size_t>(fmt.width) * fmt.height;
  const size_t nc = static_cast<size_t>(fmt.chroma_width()) * fmt.chroma_height();
  const size_t plane_samples[3] = {ny, nc, nc};
  const auto pack0 = std::chrono::steady_clock::now();
  size_t pin_off = 0;
  uint16_t* packed[2][3];
  for (int side = 0; side < 2; ++side) {
    const CudaHostPlane planes[3] = {pics[side].y, pics[side].u, pics[side].v};
    for (int p = 0; p < 3; ++p) {
      packed[side][p] = c.pin_stage + pin_off;
      pack_to_pin(planes[p].ptr, planes[p].pitch_samples, ws[p], hs[p], packed[side][p]);
      pin_off += plane_samples[p];
    }
  }
  if (pack_ms) {
    *pack_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pack0).count();
  }
  record_event(ev_copy_begin, c.stream);
  for (int side = 0; side < 2; ++side) {
    for (int p = 0; p < 3; ++p) {
      CUDA_CHECK(cudaMemcpyAsync(c.plane[p][side], packed[side][p],
                                 plane_samples[p] * sizeof(uint16_t), cudaMemcpyHostToDevice, c.stream));
    }
  }
  record_event(ev_copy_end, c.stream);
  if (c.snapshot_hook) c.snapshot_hook(fmt, c.stream);
  record_event(ev_snapshot_end, c.stream);
}

void cuda_stage_host_direct_yuv420p10(const CudaHostPlanar& ref, const CudaHostPlanar& dist,
                                      const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                                      void* ev_snapshot_end, double* pack_ms) {
  if (fmt.bit_depth != 10 || fmt.chroma != ChromaFormat::YUV420) {
    throw std::runtime_error("CUDA direct stage supports 10-bit 4:2:0 only");
  }
  if (fmt.width < 2 || fmt.height < 2 || (fmt.width & 1) || (fmt.height & 1)) {
    throw std::runtime_error("CUDA direct stage requires positive even dimensions");
  }
  const CudaHostPlanar pics[2] = {ref, dist};
  const int ws[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
  const int hs[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
  for (const CudaHostPlanar& pic : pics) {
    const CudaHostPlane planes[3] = {pic.y, pic.u, pic.v};
    for (int p = 0; p < 3; ++p) {
      if (!planes[p].ptr || planes[p].width != ws[p] || planes[p].height != hs[p] ||
          planes[p].pitch_samples < ws[p]) {
        throw std::runtime_error("CUDA direct stage host plane is missing or has a short pitch");
      }
    }
  }
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  if (pack_ms) *pack_ms = 0;
  record_event(ev_copy_begin, c.stream);
  for (int side = 0; side < 2; ++side) {
    const CudaHostPlane planes[3] = {pics[side].y, pics[side].u, pics[side].v};
    for (int p = 0; p < 3; ++p) {
      CUDA_CHECK(cudaMemcpy2DAsync(c.plane[p][side], static_cast<size_t>(ws[p]) * sizeof(uint16_t),
                                   planes[p].ptr, static_cast<size_t>(planes[p].pitch_samples) * sizeof(uint16_t),
                                   static_cast<size_t>(ws[p]) * sizeof(uint16_t), static_cast<size_t>(hs[p]),
                                   cudaMemcpyHostToDevice, c.stream));
    }
  }
  record_event(ev_copy_end, c.stream);
  if (c.snapshot_hook) c.snapshot_hook(fmt, c.stream);
  record_event(ev_snapshot_end, c.stream);
}

void cuda_stage_p010_pair(const P010DeviceSurface& ref, const P010DeviceSurface& dist,
                          const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                          void* ev_snapshot_end) {
  if (fmt.bit_depth != 10 || fmt.chroma != ChromaFormat::YUV420) {
    throw std::runtime_error("P010 stage supports 420p:10 only");
  }
  if (fmt.width < 2 || fmt.height < 2 || (fmt.width & 1) || (fmt.height & 1)) {
    throw std::runtime_error("P010 stage requires positive even dimensions");
  }
  const P010DeviceSurface surfaces[2] = {ref, dist};
  for (const P010DeviceSurface& src : surfaces) {
    if (!src.y || !src.uv) throw std::runtime_error("P010 stage is missing a plane");
    if ((src.y_pitch_bytes & 1) || (src.uv_pitch_bytes & 1)) {
      throw std::runtime_error("P010 stage pitch is not a multiple of two bytes");
    }
    if (src.y_pitch_bytes < fmt.width * 2 || src.uv_pitch_bytes < fmt.width * 2) {
      throw std::runtime_error("P010 stage pitch is smaller than the picture");
    }
    require_device_ptr(src.y, "P010 luma");
    require_device_ptr(src.uv, "P010 chroma");
  }
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const dim3 block(32, 8);
  const dim3 grid_y((fmt.width + 31) / 32, (fmt.height + 7) / 8);
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const dim3 grid_c((cw + 31) / 32, (ch + 7) / 8);
  record_event(ev_copy_begin, c.stream);
  for (int side = 0; side < 2; ++side) {
    p010_y_kernel<<<grid_y, block, 0, c.stream>>>(static_cast<const uint8_t*>(surfaces[side].y),
                                                  surfaces[side].y_pitch_bytes, c.plane[0][side], fmt.width,
                                                  fmt.height);
    p010_uv_kernel<<<grid_c, block, 0, c.stream>>>(static_cast<const uint8_t*>(surfaces[side].uv),
                                                   surfaces[side].uv_pitch_bytes, c.plane[1][side],
                                                   c.plane[2][side], cw, ch);
  }
  check_launch("p010 stage");
  record_event(ev_copy_end, c.stream);
  if (c.snapshot_hook) c.snapshot_hook(fmt, c.stream);
  record_event(ev_snapshot_end, c.stream);
}

void cuda_stage_host_ref_p010_dist(const CudaHostPlanar& ref, const P010DeviceSurface& dist,
                                   const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                                   void* ev_snapshot_end) {
  if (fmt.bit_depth != 10 || fmt.chroma != ChromaFormat::YUV420) {
    throw std::runtime_error("mixed stage supports 10-bit 4:2:0 only");
  }
  if (fmt.width < 2 || fmt.height < 2 || (fmt.width & 1) || (fmt.height & 1)) {
    throw std::runtime_error("mixed stage requires positive even dimensions");
  }
  const CudaHostPlane host[3] = {ref.y, ref.u, ref.v};
  const int ws[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
  const int hs[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
  for (int p = 0; p < 3; ++p) {
    if (!host[p].ptr || host[p].width != ws[p] || host[p].height != hs[p] || host[p].pitch_samples < ws[p]) {
      throw std::runtime_error("mixed stage host reference is missing or has a short pitch");
    }
  }
  if (!dist.y || !dist.uv || (dist.y_pitch_bytes & 1) || (dist.uv_pitch_bytes & 1) ||
      dist.y_pitch_bytes < fmt.width * 2 || dist.uv_pitch_bytes < fmt.width * 2) {
    throw std::runtime_error("mixed stage P010 distorted surface is missing or has a short pitch");
  }
  require_device_ptr(dist.y, "mixed P010 luma");
  require_device_ptr(dist.uv, "mixed P010 chroma");
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const dim3 block(32, 8);
  const dim3 grid_y((fmt.width + 31) / 32, (fmt.height + 7) / 8);
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const dim3 grid_c((cw + 31) / 32, (ch + 7) / 8);
  record_event(ev_copy_begin, c.stream);
  for (int p = 0; p < 3; ++p) {
    CUDA_CHECK(cudaMemcpy2DAsync(c.plane[p][0], static_cast<size_t>(ws[p]) * sizeof(uint16_t), host[p].ptr,
                                 static_cast<size_t>(host[p].pitch_samples) * sizeof(uint16_t),
                                 static_cast<size_t>(ws[p]) * sizeof(uint16_t), static_cast<size_t>(hs[p]),
                                 cudaMemcpyHostToDevice, c.stream));
  }
  p010_y_kernel<<<grid_y, block, 0, c.stream>>>(static_cast<const uint8_t*>(dist.y), dist.y_pitch_bytes, c.plane[0][1],
                                                fmt.width, fmt.height);
  p010_uv_kernel<<<grid_c, block, 0, c.stream>>>(static_cast<const uint8_t*>(dist.uv), dist.uv_pitch_bytes,
                                                 c.plane[1][1], c.plane[2][1], cw, ch);
  check_launch("mixed stage");
  record_event(ev_copy_end, c.stream);
  if (c.snapshot_hook) c.snapshot_hook(fmt, c.stream);
  record_event(ev_snapshot_end, c.stream);
}

struct BenchSlot {
  uint16_t* pin = nullptr;
  cudaEvent_t h0 = nullptr;
  cudaEvent_t h1 = nullptr;
  cudaEvent_t k1 = nullptr;
  cudaEvent_t done = nullptr;
  std::vector<unsigned long long> sse, sa, ta, cu, cv;
  std::vector<uint8_t> valid;
  unsigned long long psnr_sse[3] = {};
  int frame = -1;
  bool pending = false;
  size_t pin_bytes = 0;
};

struct BenchPipe {
  bool open = false;
  bool do_psnr = false;
  bool do_xpsnr = false;
  VideoFormat fmt{};
  int rate = 0;
  double peak = 1023.0;
  int submitted = 0;
  int reaped = 0;
  BenchSlot slot[2];
  std::vector<CudaBenchResult> results;
  CudaBenchTiming timing{};
};

BenchPipe& bench_pipe() {
  static BenchPipe pipe;
  return pipe;
}

void bench_h2d(const uint16_t* ry, const uint16_t* ru, const uint16_t* rv, const uint16_t* dy, const uint16_t* du,
               const uint16_t* dv, const VideoFormat& fmt) {
  auto& c = ctx();
  const uint16_t* src[2][3] = {{ry, ru, rv}, {dy, du, dv}};
  const int w[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
  const int h[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
  for (int side = 0; side < 2; ++side) {
    for (int p = 0; p < 3; ++p) {
      CUDA_CHECK(cudaMemcpyAsync(c.plane[p][side], src[side][p],
                                 static_cast<size_t>(w[p]) * static_cast<size_t>(h[p]) * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, c.stream));
    }
  }
}

void bench_launch_kernels(const VideoFormat& fmt, bool do_psnr, bool do_xpsnr, int rate) {
  auto& c = ctx();
  if (do_psnr) {
    CUDA_CHECK(cudaMemsetAsync(c.d_sse, 0, 3 * sizeof(unsigned long long), c.stream));
    const int pw[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
    const int ph[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
    for (int p = 0; p < 3; ++p) launch_sse(c.plane[p][0], c.plane[p][1], pw[p], ph[p], c.d_sse + p);
  }
  if (do_xpsnr) {
    xpsnr_launch_on_uploaded(fmt, rate);
    if (c.xpsnr_raw) throw std::runtime_error("benchmark XPSNR raw path is not used at these sizes");
  }
}

void bench_queue_d2h(bool do_psnr, bool do_xpsnr, unsigned long long* psnr_dst, std::vector<unsigned long long>* blk_sse,
                     std::vector<unsigned long long>* blk_sa, std::vector<unsigned long long>* blk_ta,
                     std::vector<uint8_t>* blk_valid, std::vector<unsigned long long>* chr_u,
                     std::vector<unsigned long long>* chr_v) {
  auto& c = ctx();
  if (do_psnr) {
    CUDA_CHECK(cudaMemcpyAsync(psnr_dst, c.d_sse, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, c.stream));
  }
  if (do_xpsnr) {
    const int n = c.xpsnr_nblk;
    blk_sse->resize(static_cast<size_t>(n));
    blk_sa->resize(static_cast<size_t>(n));
    blk_ta->resize(static_cast<size_t>(n));
    blk_valid->resize(static_cast<size_t>(n));
    chr_u->resize(static_cast<size_t>(n));
    chr_v->resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpyAsync(blk_sse->data(), c.d_blk_sse, static_cast<size_t>(n) * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(blk_sa->data(), c.d_blk_sa, static_cast<size_t>(n) * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(blk_ta->data(), c.d_blk_ta, static_cast<size_t>(n) * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(blk_valid->data(), c.d_blk_valid, static_cast<size_t>(n) * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(chr_u->data(), c.d_chr_u, static_cast<size_t>(n) * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(chr_v->data(), c.d_chr_v, static_cast<size_t>(n) * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, c.stream));
  }
}

void bench_store_psnr(CudaBenchResult& out, const VideoFormat& fmt, double peak, const unsigned long long* sse) {
  const double inv_y = 1.0 / static_cast<double>(fmt.width * fmt.height);
  const double inv_c = 1.0 / static_cast<double>(fmt.chroma_width() * fmt.chroma_height());
  const double mse_y = static_cast<double>(sse[0]) * inv_y;
  const double mse_u = static_cast<double>(sse[1]) * inv_c;
  const double mse_v = static_cast<double>(sse[2]) * inv_c;
  const double mse_avg = yuv_sample_weight_avg(mse_y, mse_u, mse_v, fmt.chroma);
  out.psnr_valid = 1;
  out.psnr_y = mse_to_psnr(mse_y, peak);
  out.psnr_u = mse_to_psnr(mse_u, peak);
  out.psnr_v = mse_to_psnr(mse_v, peak);
  out.psnr_avg = mse_to_psnr(mse_avg, peak);
}

void bench_store_xpsnr(CudaBenchResult& out, const VideoFormat& fmt, BenchSlot& slot) {
  auto& c = ctx();
  std::vector<XpsnrBlockStats> luma(static_cast<size_t>(c.xpsnr_nblk));
  std::vector<uint64_t> su(static_cast<size_t>(c.xpsnr_nblk));
  std::vector<uint64_t> sv(static_cast<size_t>(c.xpsnr_nblk));
  for (int i = 0; i < c.xpsnr_nblk; ++i) {
    luma[static_cast<size_t>(i)].sse = slot.sse[static_cast<size_t>(i)];
    luma[static_cast<size_t>(i)].sa = slot.sa[static_cast<size_t>(i)];
    luma[static_cast<size_t>(i)].ta = slot.ta[static_cast<size_t>(i)];
    luma[static_cast<size_t>(i)].activity_valid = slot.valid[static_cast<size_t>(i)];
    su[static_cast<size_t>(i)] = slot.cu[static_cast<size_t>(i)];
    sv[static_cast<size_t>(i)] = slot.cv[static_cast<size_t>(i)];
  }
  uint64_t wsse[3] = {};
  xpsnr_wsse_from_block_stats(fmt, luma.data(), su.data(), sv.data(), wsse);
  out.xpsnr_valid = 1;
  out.wsse_y = wsse[0];
  out.wsse_u = wsse[1];
  out.wsse_v = wsse[2];
}

float bench_elapsed(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

void cuda_bench_score_sync(const uint16_t* ref_y, const uint16_t* ref_u, const uint16_t* ref_v, const uint16_t* dist_y,
                           const uint16_t* dist_u, const uint16_t* dist_v, const VideoFormat& fmt, double peak,
                           bool direct_pinned, bool do_psnr, CudaXpsnrRequest* xpsnr, CudaBenchResult* result,
                           CudaBenchTiming* timing) {
  if (!do_psnr && !xpsnr) throw std::runtime_error("benchmark score requested no metric");
  init_once();
  ensure_storage(fmt);
  auto& c = ctx();
  const int pw[3] = {fmt.width, fmt.chroma_width(), fmt.chroma_width()};
  const int ph[3] = {fmt.height, fmt.chroma_height(), fmt.chroma_height()};
  const uint16_t* src[2][3] = {{ref_y, ref_u, ref_v}, {dist_y, dist_u, dist_v}};
  CudaBenchTiming local{};
  if (!direct_pinned) {
    const auto t0 = std::chrono::steady_clock::now();
    size_t pin_off = 0;
    for (int side = 0; side < 2; ++side) {
      for (int p = 0; p < 3; ++p) {
        const size_t n = static_cast<size_t>(pw[p]) * static_cast<size_t>(ph[p]);
        std::memcpy(c.pin_stage + pin_off, src[side][p], n * sizeof(uint16_t));
        pin_off += n;
      }
    }
    local.memcpy_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    size_t back = 0;
    for (int side = 0; side < 2; ++side) {
      for (int p = 0; p < 3; ++p) {
        src[side][p] = c.pin_stage + back;
        back += static_cast<size_t>(pw[p]) * static_cast<size_t>(ph[p]);
      }
    }
  }
  struct SyncEvents {
    cudaEvent_t h0 = nullptr;
    cudaEvent_t h1 = nullptr;
    cudaEvent_t k1 = nullptr;
    cudaEvent_t d1 = nullptr;
    bool ready = false;
  };
  static SyncEvents ev;
  if (!ev.ready) {
    CUDA_CHECK(cudaEventCreate(&ev.h0));
    CUDA_CHECK(cudaEventCreate(&ev.h1));
    CUDA_CHECK(cudaEventCreate(&ev.k1));
    CUDA_CHECK(cudaEventCreate(&ev.d1));
    ev.ready = true;
  }
  cudaEvent_t h0 = ev.h0, h1 = ev.h1, k1 = ev.k1, d1 = ev.d1;
  CUDA_CHECK(cudaEventRecord(h0, c.stream));
  bench_h2d(src[0][0], src[0][1], src[0][2], src[1][0], src[1][1], src[1][2], fmt);
  CUDA_CHECK(cudaEventRecord(h1, c.stream));
  bench_launch_kernels(fmt, do_psnr, xpsnr != nullptr, xpsnr ? xpsnr->integer_rate : 0);
  CUDA_CHECK(cudaEventRecord(k1, c.stream));
  unsigned long long psnr_sse[3] = {};
  BenchSlot host;
  bench_queue_d2h(do_psnr, xpsnr != nullptr, psnr_sse, &host.sse, &host.sa, &host.ta, &host.valid, &host.cu, &host.cv);
  CUDA_CHECK(cudaEventRecord(d1, c.stream));
  const auto sync0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaStreamSynchronize(c.stream));
  local.sync_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0).count();
  local.h2d_ms = bench_elapsed(h0, h1);
  local.kernel_ms = bench_elapsed(h1, k1);
  local.d2h_ms = bench_elapsed(k1, d1);
  CudaBenchResult out{};
  if (do_psnr) bench_store_psnr(out, fmt, peak, psnr_sse);
  if (xpsnr) {
    bench_store_xpsnr(out, fmt, host);
    xpsnr->wsse_y = out.wsse_y;
    xpsnr->wsse_u = out.wsse_u;
    xpsnr->wsse_v = out.wsse_v;
  }
  if (result) *result = out;
  if (timing) *timing = local;
}

void bench_reap_slot(BenchPipe& pipe, int slot_index) {
  BenchSlot& slot = pipe.slot[slot_index];
  if (!slot.pending) return;
  const auto t0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventSynchronize(slot.done));
  pipe.timing.sync_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  pipe.timing.h2d_ms += bench_elapsed(slot.h0, slot.h1);
  pipe.timing.kernel_ms += bench_elapsed(slot.h1, slot.k1);
  pipe.timing.d2h_ms += bench_elapsed(slot.k1, slot.done);
  if (pipe.results.size() != static_cast<size_t>(slot.frame)) {
    throw std::runtime_error("benchmark pipe reaped a frame out of order");
  }
  CudaBenchResult out{};
  if (pipe.do_psnr) bench_store_psnr(out, pipe.fmt, pipe.peak, slot.psnr_sse);
  if (pipe.do_xpsnr) bench_store_xpsnr(out, pipe.fmt, slot);
  pipe.results.push_back(out);
  slot.pending = false;
  pipe.reaped += 1;
}

void bench_queue_slot(BenchPipe& pipe, BenchSlot& slot, const uint16_t* ry, const uint16_t* ru, const uint16_t* rv,
                      const uint16_t* dy, const uint16_t* du, const uint16_t* dv) {
  auto& c = ctx();
  CUDA_CHECK(cudaEventRecord(slot.h0, c.stream));
  bench_h2d(ry, ru, rv, dy, du, dv, pipe.fmt);
  CUDA_CHECK(cudaEventRecord(slot.h1, c.stream));
  bench_launch_kernels(pipe.fmt, pipe.do_psnr, pipe.do_xpsnr, pipe.rate);
  CUDA_CHECK(cudaEventRecord(slot.k1, c.stream));
  bench_queue_d2h(pipe.do_psnr, pipe.do_xpsnr, slot.psnr_sse, &slot.sse, &slot.sa, &slot.ta, &slot.valid, &slot.cu,
                  &slot.cv);
  CUDA_CHECK(cudaEventRecord(slot.done, c.stream));
  slot.frame = pipe.submitted;
  slot.pending = true;
  pipe.submitted += 1;
}

void bench_ensure_pipe_slot(BenchPipe& pipe, int index) {
  auto& c = ctx();
  BenchSlot& slot = pipe.slot[index];
  const size_t bytes = pipe.fmt.frame_bytes() * 2;
  if (slot.pin_bytes < bytes) {
    if (slot.pin) CUDA_CHECK(cudaFreeHost(slot.pin));
    slot.pin = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&slot.pin), bytes, cudaHostAllocDefault));
    slot.pin_bytes = bytes;
  }
  if (!slot.h0) {
    CUDA_CHECK(cudaEventCreate(&slot.h0));
    CUDA_CHECK(cudaEventCreate(&slot.h1));
    CUDA_CHECK(cudaEventCreate(&slot.k1));
    CUDA_CHECK(cudaEventCreate(&slot.done));
  }
  (void)c;
}

void cuda_bench_pipe_open(const VideoFormat& fmt, int integer_rate, bool do_psnr, bool do_xpsnr) {
  if (!do_psnr && !do_xpsnr) throw std::runtime_error("benchmark pipe requested no metric");
  init_once();
  ensure_storage(fmt);
  auto& pipe = bench_pipe();
  if (pipe.open) cudaStreamSynchronize(ctx().stream);
  pipe.open = true;
  pipe.do_psnr = do_psnr;
  pipe.do_xpsnr = do_xpsnr;
  pipe.fmt = fmt;
  pipe.rate = integer_rate;
  pipe.peak = fmt.bit_depth == 8 ? 255.0 : 1023.0;
  pipe.submitted = 0;
  pipe.reaped = 0;
  pipe.results.clear();
  pipe.timing = {};
  for (int i = 0; i < 2; ++i) {
    pipe.slot[i].pending = false;
    pipe.slot[i].frame = -1;
    bench_ensure_pipe_slot(pipe, i);
  }
}

void cuda_bench_pipe_submit_copied(const uint16_t* ref_y, const uint16_t* ref_u, const uint16_t* ref_v,
                                   const uint16_t* dist_y, const uint16_t* dist_u, const uint16_t* dist_v) {
  auto& pipe = bench_pipe();
  if (!pipe.open) throw std::runtime_error("benchmark pipe is not open");
  const int index = pipe.submitted % 2;
  if (pipe.submitted >= 2) bench_reap_slot(pipe, index);
  BenchSlot& slot = pipe.slot[index];
  const int pw[3] = {pipe.fmt.width, pipe.fmt.chroma_width(), pipe.fmt.chroma_width()};
  const int ph[3] = {pipe.fmt.height, pipe.fmt.chroma_height(), pipe.fmt.chroma_height()};
  const uint16_t* src[6] = {ref_y, ref_u, ref_v, dist_y, dist_u, dist_v};
  const auto t0 = std::chrono::steady_clock::now();
  size_t off = 0;
  for (int i = 0; i < 6; ++i) {
    const int p = i % 3;
    const size_t n = static_cast<size_t>(pw[p]) * static_cast<size_t>(ph[p]);
    std::memcpy(slot.pin + off, src[i], n * sizeof(uint16_t));
    off += n;
  }
  pipe.timing.memcpy_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const size_t ny = static_cast<size_t>(pw[0]) * ph[0];
  const size_t nc = static_cast<size_t>(pw[1]) * ph[1];
  bench_queue_slot(pipe, slot, slot.pin, slot.pin + ny, slot.pin + ny + nc, slot.pin + ny + 2 * nc,
                   slot.pin + 2 * ny + 2 * nc, slot.pin + 2 * ny + 3 * nc);
}

void cuda_bench_pipe_acquire(uint16_t** ref_y, uint16_t** ref_u, uint16_t** ref_v, uint16_t** dist_y, uint16_t** dist_u,
                             uint16_t** dist_v) {
  auto& pipe = bench_pipe();
  if (!pipe.open) throw std::runtime_error("benchmark pipe is not open");
  const int index = pipe.submitted % 2;
  if (pipe.submitted >= 2) bench_reap_slot(pipe, index);
  const size_t ny = static_cast<size_t>(pipe.fmt.width) * pipe.fmt.height;
  const size_t nc = static_cast<size_t>(pipe.fmt.chroma_width()) * pipe.fmt.chroma_height();
  uint16_t* pin = pipe.slot[index].pin;
  *ref_y = pin;
  *ref_u = pin + ny;
  *ref_v = pin + ny + nc;
  *dist_y = pin + ny + 2 * nc;
  *dist_u = pin + 2 * ny + 2 * nc;
  *dist_v = pin + 2 * ny + 3 * nc;
}

void cuda_bench_pipe_submit_filled() {
  auto& pipe = bench_pipe();
  if (!pipe.open) throw std::runtime_error("benchmark pipe is not open");
  const int index = pipe.submitted % 2;
  BenchSlot& slot = pipe.slot[index];
  const size_t ny = static_cast<size_t>(pipe.fmt.width) * pipe.fmt.height;
  const size_t nc = static_cast<size_t>(pipe.fmt.chroma_width()) * pipe.fmt.chroma_height();
  uint16_t* pin = slot.pin;
  bench_queue_slot(pipe, slot, pin, pin + ny, pin + ny + nc, pin + ny + 2 * nc, pin + 2 * ny + 2 * nc,
                   pin + 2 * ny + 3 * nc);
}

void cuda_bench_pipe_finish(CudaBenchResult* out, int* count, int capacity, CudaBenchTiming* timing) {
  auto& pipe = bench_pipe();
  if (!pipe.open) throw std::runtime_error("benchmark pipe is not open");
  while (pipe.reaped < pipe.submitted) bench_reap_slot(pipe, pipe.reaped % 2);
  CUDA_CHECK(cudaStreamSynchronize(ctx().stream));
  if (pipe.reaped != pipe.submitted || static_cast<int>(pipe.results.size()) != pipe.submitted) {
    throw std::runtime_error("benchmark pipe did not complete every frame");
  }
  if (capacity < pipe.submitted) throw std::runtime_error("benchmark pipe result buffer is short");
  for (int i = 0; i < pipe.submitted; ++i) out[i] = pipe.results[static_cast<size_t>(i)];
  if (count) *count = pipe.submitted;
  if (timing) *timing = pipe.timing;
  pipe.open = false;
}

__global__ void bench_keepawake(unsigned loops, unsigned long long* sink) {
  unsigned long long x = static_cast<unsigned long long>(threadIdx.x) + 1ull;
  for (unsigned i = 0; i < loops; ++i) x = x * 1664525ull + 1013904223ull;
  if (x == 0) *sink = x;
}

struct BenchSustain {
  std::atomic<bool> run{false};
  std::thread thread;
  cudaStream_t stream = nullptr;
  unsigned long long* sink = nullptr;
  ~BenchSustain() {
    run.store(false);
    if (thread.joinable()) thread.join();
  }
};

BenchSustain& bench_sustain() {
  static BenchSustain sustain;
  return sustain;
}

void cuda_bench_sustain_start() {
  init_once();
  auto& s = bench_sustain();
  if (s.run.load()) return;
  if (!s.stream) {
    CUDA_CHECK(cudaStreamCreate(&s.stream));
    CUDA_CHECK(cudaMalloc(&s.sink, sizeof(unsigned long long)));
  }
  s.run = true;
  s.thread = std::thread([] {
    cudaSetDevice(0);
    auto& s = bench_sustain();
    while (s.run.load()) {
      bench_keepawake<<<48, 256, 0, s.stream>>>(8000000u, s.sink);
      cudaStreamSynchronize(s.stream);
    }
  });
}

void cuda_bench_sustain_stop() {
  auto& s = bench_sustain();
  s.run = false;
  if (s.thread.joinable()) s.thread.join();
  if (s.stream) CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void cuda_score_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt, double peak,
                      CudaPrecision prec, bool do_psnr, bool do_ssim, bool do_msssim,
                      PsnrFrame* psnr, PlaneScore* ssim, PlaneScore* msssim,
                      CudaXpsnrRequest* xpsnr) {
  cuda_score_device_frame(&ref, nullptr, &test, nullptr, fmt, peak, prec, do_psnr, do_ssim,
                          do_msssim, psnr, ssim, msssim, xpsnr);
}

}  // namespace pixvqa
