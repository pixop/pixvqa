#pragma once

#include "pixvqa/types.hpp"

#include <cstdint>
#include <vector>

namespace pixvqa {

// Integer frame rate used by FFmpeg's xpsnr filter: trunc(num/den) toward zero.
// Order is 1 when that integer is < 32, otherwise 2.
inline int xpsnr_integer_rate(int num, int den) {
  if (den == 0) return 0;
  return num / den;
}

inline int xpsnr_temporal_order(int integer_rate) { return integer_rate < 32 ? 1 : 2; }

// Block size in luma samples. Matches FFmpeg n8.1.3:
//   b = max(1, 4 * (int)(32 * sqrt(w*h / (3840*2160)) + 0.5))
int xpsnr_block_size(int width, int height);

// True when luma sample count is strictly above 2048*1152 (FFmpeg high-pass path).
inline bool xpsnr_high_resolution(int width, int height) {
  return static_cast<long long>(width) * height > 2048LL * 1152LL;
}

// High-resolution activity visits every second sample and reads three samples
// past that site, while the picture-edge inset is only two samples. With a
// step of two, the last tap stays inside the plane when the edge block is
// even in both axes. Block size on that path is a multiple of four, so even
// luma width and height are required. Odd luma edges are rejected instead of
// clamping the filter. At or below 2048*1152 the per-pixel filter uses a
// one-sample halo and odd dimensions stay supported.
inline bool xpsnr_geometry_supported(int width, int height) {
  if (width <= 0 || height <= 0) return false;
  if (!xpsnr_high_resolution(width, height)) return true;
  return (width % 2 == 0) && (height % 2 == 0);
}

struct XpsnrFrame {
  double y = 0;
  double u = 0;
  double v = 0;
  double avg = 0;
  double min_score = 0;
  // Rounded perceptually weighted SSE in FFmpeg's plane units (after avg_act).
  std::uint64_t wsse_y = 0;
  std::uint64_t wsse_u = 0;
  std::uint64_t wsse_v = 0;
};

struct XpsnrSummary {
  XpsnrFrame score;
  int frames = 0;
};

// Integer block products from the CPU loop or the CUDA kernels.
// activity_valid == 0 means the activity window is empty: weight is 1 and
// history is not updated. sa/ta are unused in that case.
struct XpsnrBlockStats {
  std::uint64_t sse = 0;
  std::uint64_t sa = 0;
  std::uint64_t ta = 0;
  std::uint8_t activity_valid = 0;
};

// Host weighting, low-resolution smoothing, and integer WSSE rounding.
// Raster order matches wsse_planes. b < 4 is not handled here.
void xpsnr_wsse_from_block_stats(const VideoFormat& fmt, const XpsnrBlockStats* luma,
                                 const std::uint64_t* sse_u, const std::uint64_t* sse_v,
                                 std::uint64_t out_wsse[3]);

// Activity is measured on the reference (FFmpeg input 0 / "main", variable org).
// Distortion is reference versus reconstructed (FFmpeg input 1).
// Plane scores and `min` follow FFmpeg n8.1.3. `avg` is the pixvqa extension:
// sample-count-weighted distortion, then the SMR over frames.
class XpsnrMeter {
 public:
  // threads > 1 splits independent luma block-rows. Pictures at or below
  // 640×480 stay serial so FFmpeg's in-loop weight smoothing keeps its order.
  void start(const VideoFormat& fmt, int integer_frame_rate, int threads = 1);
  XpsnrFrame push(const Frame& reference, const Frame& reconstructed);
  // Scoring half of push. Does not read pixels or history.
  XpsnrFrame adopt_wsse(std::uint64_t wsse_y, std::uint64_t wsse_u, std::uint64_t wsse_v);
  XpsnrSummary summary() const;

  const VideoFormat& format() const { return fmt_; }
  int integer_frame_rate() const { return frame_rate_; }

 private:
  void ensure_history();
  void wsse_planes(const Frame& reference, const Frame& reconstructed, std::uint64_t out[3]);

  VideoFormat fmt_{};
  int frame_rate_ = 0;
  int threads_ = 1;
  int frames_ = 0;
  double sum_sqrt_wsse_[3] = {};
  double sum_xpsnr_[3] = {};
  double sum_sqrt_d_ = 0;  // sum of sqrt(d_YUV)
  bool started_ = false;
  bool hist_ready_ = false;
  std::vector<int16_t> m1_;
  std::vector<int16_t> m2_;
  std::vector<double> sse_luma_;
  std::vector<double> weights_;
};

}  // namespace pixvqa
