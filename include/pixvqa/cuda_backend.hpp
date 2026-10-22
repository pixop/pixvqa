#pragma once

// Optional CUDA metric backend. Only the CUDA build provides these symbols.
// The CPU-only build does not include this header and does not link it.

#include "pixvqa/types.hpp"

namespace pixvqa {

void cuda_print_device_info();

// Optional XPSNR request on the same upload as PSNR / MS-SSIM.
// On return the three WSSE integers are filled. Scoring stays on the host.
// Null leaves XPSNR off. There is no CPU fallback if the device call fails.
// --cuda-prec does not apply to this path.
struct CudaXpsnrRequest {
  int integer_rate = 0;
  std::uint64_t wsse_y = 0;
  std::uint64_t wsse_u = 0;
  std::uint64_t wsse_v = 0;
};

// Planar uint16, sample in the low bits, pitch in samples. This is the only
// layout the metric kernels accept. A future NVDEC adapter converts P010 into
// these planes before any metric runs. XPSNR does not see decoder objects.
enum class DeviceLayout { PlanarU16 };

struct DevicePlane {
  const void* ptr = nullptr;
  int width = 0;
  int height = 0;
  int pitch_samples = 0;  // kernels index y * pitch_samples + x; never a byte pitch
};

struct DevicePlanarFrame {
  DevicePlane y, u, v;
};

// Zero both luma history planes. Call once per video, before the first frame.
// Throws if the geometry is unsupported or the device cannot allocate.
void cuda_xpsnr_reset(const VideoFormat& fmt, int integer_rate);

// One upload of both frames. Optional CUDA XPSNR reads those planes after
// PSNR and SSIM and before five-scale MS-SSIM. MS-SSIM reads the uploaded
// planes and writes only its own scratch.
// Throws std::runtime_error on device, allocation, or unsupported-geometry errors.
// No CPU fallback.
void cuda_score_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt, double peak,
                      CudaPrecision prec, bool do_psnr, bool do_ssim, bool do_msssim,
                      PsnrFrame* psnr, PlaneScore* ssim, PlaneScore* msssim,
                      CudaXpsnrRequest* xpsnr = nullptr);

// Tightly packed or pitched host 4:2:0 planes. pitch_samples is in samples.
struct CudaHostPlane {
  const std::uint16_t* ptr = nullptr;
  int width = 0;
  int height = 0;
  int pitch_samples = 0;
};

struct CudaHostPlanar {
  CudaHostPlane y, u, v;
};

// Pack both pictures into the metric-owned planes and queue the snapshot hook.
// No PSNR, SSIM, MS-SSIM, or XPSNR work is launched. Does not synchronize.
// ev_copy_begin / ev_copy_end / ev_snapshot_end are cudaEvent_t values, or null.
// The copy interval excludes the host pack. The snapshot interval is the
// device-to-device copy into the active VMAF snapshot.
void cuda_stage_host_yuv420p10(const CudaHostPlanar& ref, const CudaHostPlanar& dist,
                               const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                               void* ev_snapshot_end, double* pack_ms);

// Pitched host-to-device copy of the same six planes. No pinned pack and no
// metric kernel. Source pitch is in samples and may exceed width. Does not
// synchronize. pack_ms is set to 0: this entry point performs no host copy.
void cuda_stage_host_direct_yuv420p10(const CudaHostPlanar& ref, const CudaHostPlanar& dist,
                                      const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                                      void* ev_snapshot_end, double* pack_ms);

// Host sides are uploaded. A null host test uses device_test, which must already
// point at the metric-owned planes with pitch_samples equal to each plane width.
// device_ref is not used. cuda_score_frame is the host/host wrapper.
void cuda_score_device_frame(const Frame* host_ref, const DevicePlanarFrame* device_ref,
                             const Frame* host_test, const DevicePlanarFrame* device_test,
                             const VideoFormat& fmt, double peak, CudaPrecision prec, bool do_psnr,
                             bool do_ssim, bool do_msssim, PsnrFrame* psnr, PlaneScore* ssim,
                             PlaneScore* msssim, CudaXpsnrRequest* xpsnr = nullptr);

// Make the NVIDIA primary context current. Used by NVDEC before FFmpeg creates
// its CUDA device, including when metrics stay on the CPU.
void cuda_ensure_primary_context();
uintptr_t cuda_primary_context_handle();

struct P010DeviceSurface {
  const void* y = nullptr;
  int y_pitch_bytes = 0;
  const void* uv = nullptr;
  int uv_pitch_bytes = 0;
};

// Convert two G1 P010 surfaces into the metric reference and distorted planes
// (sample = stored word >> 6, interleaved UV split) and queue the snapshot hook.
// Does not synchronize and does not download. Events are cudaEvent_t or null.
// The copy interval is the conversion kernels. The snapshot interval follows it.
void cuda_stage_p010_pair(const P010DeviceSurface& ref, const P010DeviceSurface& dist,
                          const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                          void* ev_snapshot_end);

// Reference is pitched host YUV420P10 copied into metric plane[p][0]. Distorted
// is one G1 P010 surface converted into plane[p][1] with word >> 6. Queues both
// on the metric stream and then the snapshot hook. Does not synchronize.
// The host reference and the P010 surface must stay alive until that stream completes.
void cuda_stage_host_ref_p010_dist(const CudaHostPlanar& ref, const P010DeviceSurface& dist,
                                   const VideoFormat& fmt, void* ev_copy_begin, void* ev_copy_end,
                                   void* ev_snapshot_end);

// Read-only on the P010 surface. Writes metric-owned plane[p][1] with
// pitch_samples equal to the plane width. Synchronizes the metric stream
// before returning. sample = word >> 6.
void cuda_convert_p010_to_test_planes(const P010DeviceSurface& src, const VideoFormat& fmt);

// Device-to-host copy of those metric test planes into a tightly packed Frame.
void cuda_download_test_planes(Frame& out, const VideoFormat& fmt);

// The metric-owned test planes just written by P010 conversion.
// pitch_samples equals each plane width.
DevicePlanarFrame cuda_metric_test_frame(const VideoFormat& fmt);

// Metric-owned reference planes (index 0). Same layout as the test planes.
// pitch_samples equals each plane width. A later host upload overwrites them.
DevicePlanarFrame cuda_metric_ref_frame(const VideoFormat& fmt);

// Writable view of the same metric planes. The G1 isolation test fills them
// after a snapshot. Production scoring does not call these.
struct CudaWritablePlane {
  void* ptr = nullptr;
  int width = 0;
  int height = 0;
  int pitch_samples = 0;
};

struct CudaWritableFrame {
  CudaWritablePlane y, u, v;
};

CudaWritableFrame cuda_metric_writable_ref(const VideoFormat& fmt);
CudaWritableFrame cuda_metric_writable_test(const VideoFormat& fmt);

// cudaStream_t of the single metric stream, after the context is initialized.
// Null is not returned once initialization has succeeded.
void* cuda_metric_stream_handle();

// Bytes currently held by the metric context's own cudaMalloc tracking.
std::size_t cuda_metric_tracked_bytes();

// Invoked after uploads are queued and before metric kernels, on the metric
// stream (the void* is a cudaStream_t). May only enqueue work on that stream.
// Must not synchronize and must not write the metric planes. A null hook
// leaves the scoring path unchanged.
using CudaFrameSnapshotHook = void (*)(const VideoFormat& fmt, void* stream);
void cuda_set_frame_snapshot_hook(CudaFrameSnapshotHook hook);

// Stderr only, and only when PIXVQA_CUDA_PROFILE=1. Never touches metric stdout.
void cuda_dump_profile();

// Diagnostic counters. They advance only while PIXVQA_CUDA_PROFILE=1 and stay
// zero otherwise. Reading or resetting them does not change a score.
struct CudaProfileSnapshot {
  double h2d_ms = 0;
  double kernel_ms = 0;
  double d2h_ms = 0;
  double xpsnr_ms = 0;
  int frames = 0;
  int h2d_test_copies = 0;
};
CudaProfileSnapshot cuda_profile_snapshot();
void cuda_profile_reset();

// Benchmark-only timing. These entry points are not used by production dispatch.
// Kernels, XPSNR history order, and the host WSSE reduction match cuda_score_frame.
struct CudaBenchTiming {
  double memcpy_ms = 0;
  double h2d_ms = 0;
  double kernel_ms = 0;
  double d2h_ms = 0;
  double sync_ms = 0;
};

struct CudaBenchResult {
  int psnr_valid = 0;
  double psnr_y = 0;
  double psnr_u = 0;
  double psnr_v = 0;
  double psnr_avg = 0;
  int xpsnr_valid = 0;
  std::uint64_t wsse_y = 0;
  std::uint64_t wsse_u = 0;
  std::uint64_t wsse_v = 0;
};

// Synchronizes before return. direct_pinned skips the staging memcpy; the six
// pointers are then tightly packed pinned memory that stays alive until return.
// Otherwise the planes are copied into the metric pinned stage and then uploaded,
// which is the production upload. The memcpy is finished before H2D is queued so
// the two buckets do not overlap.
void cuda_bench_score_sync(const std::uint16_t* ref_y, const std::uint16_t* ref_u, const std::uint16_t* ref_v,
                           const std::uint16_t* dist_y, const std::uint16_t* dist_u, const std::uint16_t* dist_v,
                           const VideoFormat& fmt, double peak, bool direct_pinned, bool do_psnr,
                           CudaXpsnrRequest* xpsnr, CudaBenchResult* result, CudaBenchTiming* timing);

// Two-slot pipeline on the metric stream. submit_copied stages a pageable frame
// and returns without waiting for the GPU. submit_filled queues a frame already
// written into the pinned slot from acquire. finish waits until every frame's
// download is complete and returns results in frame order.
void cuda_bench_pipe_open(const VideoFormat& fmt, int integer_rate, bool do_psnr, bool do_xpsnr);
void cuda_bench_pipe_submit_copied(const std::uint16_t* ref_y, const std::uint16_t* ref_u, const std::uint16_t* ref_v,
                                   const std::uint16_t* dist_y, const std::uint16_t* dist_u, const std::uint16_t* dist_v);
void cuda_bench_pipe_acquire(std::uint16_t** ref_y, std::uint16_t** ref_u, std::uint16_t** ref_v,
                             std::uint16_t** dist_y, std::uint16_t** dist_u, std::uint16_t** dist_v);
void cuda_bench_pipe_submit_filled();
void cuda_bench_pipe_finish(CudaBenchResult* out, int* count, int capacity, CudaBenchTiming* timing);

// Temporary side-stream compute so a session can be measured while the GPU is
// busy. It does not lock clocks and does not touch the metric stream. stop
// joins the worker and synchronizes that stream.
void cuda_bench_sustain_start();
void cuda_bench_sustain_stop();

// Release device allocations, pinned staging, streams and events. Safe to call
// more than once. Also registered with atexit on first successful init.
void cuda_shutdown();

}  // namespace pixvqa
