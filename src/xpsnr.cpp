#include "pixvqa/thread_pool.hpp"
#include "pixvqa/xpsnr.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pixvqa {
namespace {

constexpr int kGamma = 2;

int sample_u16(const uint16_t* p, int stride, int x, int y) {
  return static_cast<int>(p[static_cast<size_t>(y) * stride + x]);
}

int sample_i16(const int16_t* p, int stride, int x, int y) {
  return static_cast<int>(p[static_cast<size_t>(y) * stride + x]);
}

void store_i16(int16_t* p, int stride, int x, int y, int value) {
  p[static_cast<size_t>(y) * stride + x] = static_cast<int16_t>(value);
}

// Downsampled 6-tap high-pass used above HD. Integer, so the sum is exact.
uint64_t highds(const uint16_t* org, int stride, int x0, int y0, int x_act, int y_act, int w_act,
                int h_act) {
  uint64_t sa = 0;
  for (int y = y_act; y < h_act; y += 2) {
    for (int x = x_act; x < w_act; x += 2) {
      auto at = [&](int yy, int xx) {
        return sample_u16(org, stride, x0 + xx, y0 + yy);
      };
      const int f =
          12 * (at(y, x) + at(y, x + 1) + at(y + 1, x) + at(y + 1, x + 1)) -
          3 * (at(y - 1, x) + at(y - 1, x + 1) + at(y + 2, x) + at(y + 2, x + 1)) -
          3 * (at(y, x - 1) + at(y, x + 2) + at(y + 1, x - 1) + at(y + 1, x + 2)) -
          2 * (at(y - 1, x - 1) + at(y - 1, x + 2) + at(y + 2, x - 1) + at(y + 2, x + 2)) -
          (at(y - 2, x - 1) + at(y - 2, x) + at(y - 2, x + 1) + at(y - 2, x + 2) +
           at(y + 3, x - 1) + at(y + 3, x) + at(y + 3, x + 1) + at(y + 3, x + 2) +
           at(y - 1, x - 2) + at(y, x - 2) + at(y + 1, x - 2) + at(y + 2, x - 2) +
           at(y - 1, x + 3) + at(y, x + 3) + at(y + 1, x + 3) + at(y + 2, x + 3));
      sa += static_cast<uint64_t>(std::abs(f));
    }
  }
  return sa;
}

uint64_t diff1st_ds(const uint16_t* org, int16_t* m1, int stride, int x0, int y0, uint32_t block_w,
                    uint32_t block_h) {
  uint64_t ta = 0;
  for (uint32_t y = 0; y < block_h; y += 2) {
    for (uint32_t x = 0; x < block_w; x += 2) {
      const int ax = x0 + static_cast<int>(x);
      const int ay = y0 + static_cast<int>(y);
      const int t = sample_u16(org, stride, ax, ay) + sample_u16(org, stride, ax + 1, ay) +
                    sample_u16(org, stride, ax, ay + 1) + sample_u16(org, stride, ax + 1, ay + 1) -
                    (sample_i16(m1, stride, ax, ay) + sample_i16(m1, stride, ax + 1, ay) +
                     sample_i16(m1, stride, ax, ay + 1) + sample_i16(m1, stride, ax + 1, ay + 1));
      ta += static_cast<uint64_t>(std::abs(t));
      store_i16(m1, stride, ax, ay, sample_u16(org, stride, ax, ay));
      store_i16(m1, stride, ax + 1, ay, sample_u16(org, stride, ax + 1, ay));
      store_i16(m1, stride, ax, ay + 1, sample_u16(org, stride, ax, ay + 1));
      store_i16(m1, stride, ax + 1, ay + 1, sample_u16(org, stride, ax + 1, ay + 1));
    }
  }
  return ta * static_cast<uint64_t>(kGamma);
}

uint64_t diff2nd_ds(const uint16_t* org, int16_t* m1, int16_t* m2, int stride, int x0, int y0,
                    uint32_t block_w, uint32_t block_h) {
  uint64_t ta = 0;
  for (uint32_t y = 0; y < block_h; y += 2) {
    for (uint32_t x = 0; x < block_w; x += 2) {
      const int ax = x0 + static_cast<int>(x);
      const int ay = y0 + static_cast<int>(y);
      const int cur = sample_u16(org, stride, ax, ay) + sample_u16(org, stride, ax + 1, ay) +
                      sample_u16(org, stride, ax, ay + 1) + sample_u16(org, stride, ax + 1, ay + 1);
      const int p1 = sample_i16(m1, stride, ax, ay) + sample_i16(m1, stride, ax + 1, ay) +
                     sample_i16(m1, stride, ax, ay + 1) + sample_i16(m1, stride, ax + 1, ay + 1);
      const int p2 = sample_i16(m2, stride, ax, ay) + sample_i16(m2, stride, ax + 1, ay) +
                     sample_i16(m2, stride, ax, ay + 1) + sample_i16(m2, stride, ax + 1, ay + 1);
      const int t = cur - 2 * p1 + p2;
      ta += static_cast<uint64_t>(std::abs(t));
      store_i16(m2, stride, ax, ay, sample_i16(m1, stride, ax, ay));
      store_i16(m2, stride, ax + 1, ay, sample_i16(m1, stride, ax + 1, ay));
      store_i16(m2, stride, ax, ay + 1, sample_i16(m1, stride, ax, ay + 1));
      store_i16(m2, stride, ax + 1, ay + 1, sample_i16(m1, stride, ax + 1, ay + 1));
      store_i16(m1, stride, ax, ay, sample_u16(org, stride, ax, ay));
      store_i16(m1, stride, ax + 1, ay, sample_u16(org, stride, ax + 1, ay));
      store_i16(m1, stride, ax, ay + 1, sample_u16(org, stride, ax, ay + 1));
      store_i16(m1, stride, ax + 1, ay + 1, sample_u16(org, stride, ax + 1, ay + 1));
    }
  }
  return ta * static_cast<uint64_t>(kGamma);
}

uint64_t sse_block(const uint16_t* org, int org_stride, const uint16_t* rec, int rec_stride, int x,
                   int y, uint32_t bw, uint32_t bh) {
  uint64_t sse = 0;
  for (uint32_t row = 0; row < bh; ++row) {
    const uint16_t* o = org + (static_cast<size_t>(y) + row) * org_stride + x;
    const uint16_t* r = rec + (static_cast<size_t>(y) + row) * rec_stride + x;
    for (uint32_t col = 0; col < bw; ++col) {
      const int64_t d = static_cast<int64_t>(o[col]) - static_cast<int64_t>(r[col]);
      sse += static_cast<uint64_t>(d * d);
    }
  }
  return sse;
}

// Returns unweighted SSE. Writes ms_act (already squared) and updates luma history,
// matching calc_squared_error_and_weight() in FFmpeg n8.1.3 vf_xpsnr.c.
double block_sse_and_weight(const uint16_t* org, int stride, int16_t* m1, int16_t* m2,
                            const uint16_t* rec, int rec_stride, int pic_w, int pic_h, int x, int y,
                            uint32_t block_w, uint32_t block_h, int bit_depth, int frame_rate,
                            double* ms_act) {
  const int b_val = (static_cast<long long>(pic_w) * pic_h > 2048LL * 1152LL) ? 2 : 1;
  const int x_act = (x > 0 ? 0 : b_val);
  const int y_act = (y > 0 ? 0 : b_val);
  const int w_act =
      (static_cast<uint32_t>(x) + block_w < static_cast<uint32_t>(pic_w) ? static_cast<int>(block_w)
                                                                         : static_cast<int>(block_w) - b_val);
  const int h_act = (static_cast<uint32_t>(y) + block_h < static_cast<uint32_t>(pic_h)
                         ? static_cast<int>(block_h)
                         : static_cast<int>(block_h) - b_val);

  const double sse = static_cast<double>(sse_block(org, stride, rec, rec_stride, x, y, block_w, block_h));
  if (w_act <= x_act || h_act <= y_act) return sse;

  uint64_t sa = 0;
  if (b_val > 1) {
    // FFmpeg discards the high-pass result when the activity window is <= 12
    // samples wide. That branch is preserved so the weight matches the oracle.
    if (w_act > 12) {
      sa = highds(org, stride, x, y, x_act, y_act, w_act, h_act);
    } else {
      (void)highds(org, stride, x, y, x_act, y_act, w_act, h_act);
    }
  } else {
    for (int yy = y_act; yy < h_act; ++yy) {
      for (int xx = x_act; xx < w_act; ++xx) {
        auto at = [&](int dy, int dx) { return sample_u16(org, stride, x + dx, y + dy); };
        const int f = 12 * at(yy, xx) - 2 * (at(yy, xx - 1) + at(yy, xx + 1) + at(yy - 1, xx) +
                                             at(yy + 1, xx)) -
                      (at(yy - 1, xx - 1) + at(yy - 1, xx + 1) + at(yy + 1, xx - 1) +
                       at(yy + 1, xx + 1));
        sa += static_cast<uint64_t>(std::abs(f));
      }
    }
  }

  *ms_act = static_cast<double>(sa) / (static_cast<double>(w_act - x_act) * static_cast<double>(h_act - y_act));

  uint64_t ta = 0;
  if (b_val > 1) {
    if (frame_rate < 32) {
      ta = diff1st_ds(org, m1, stride, x, y, block_w, block_h);
    } else {
      ta = diff2nd_ds(org, m1, m2, stride, x, y, block_w, block_h);
    }
  } else if (frame_rate < 32) {
    for (uint32_t row = 0; row < block_h; ++row) {
      for (uint32_t col = 0; col < block_w; ++col) {
        const int ax = x + static_cast<int>(col);
        const int ay = y + static_cast<int>(row);
        const int t = sample_u16(org, stride, ax, ay) - sample_i16(m1, stride, ax, ay);
        ta += static_cast<uint64_t>(kGamma) * static_cast<uint64_t>(std::abs(t));
        store_i16(m1, stride, ax, ay, sample_u16(org, stride, ax, ay));
      }
    }
  } else {
    for (uint32_t row = 0; row < block_h; ++row) {
      for (uint32_t col = 0; col < block_w; ++col) {
        const int ax = x + static_cast<int>(col);
        const int ay = y + static_cast<int>(row);
        const int t = sample_u16(org, stride, ax, ay) - 2 * sample_i16(m1, stride, ax, ay) +
                      sample_i16(m2, stride, ax, ay);
        ta += static_cast<uint64_t>(kGamma) * static_cast<uint64_t>(std::abs(t));
        store_i16(m2, stride, ax, ay, sample_i16(m1, stride, ax, ay));
        store_i16(m1, stride, ax, ay, sample_u16(org, stride, ax, ay));
      }
    }
  }

  *ms_act += static_cast<double>(ta) / (static_cast<double>(block_w) * static_cast<double>(block_h));
  const double floor_act = static_cast<double>(1 << (bit_depth - 6));
  if (*ms_act < floor_act) *ms_act = floor_act;
  *ms_act *= *ms_act;
  return sse;
}

double plane_xpsnr_from_sum(double sum_sqrt_wsse, double sum_xpsnr, uint32_t width, uint32_t height,
                            uint64_t max_error_sq, uint64_t num_frames) {
  if (num_frames == 0) return INFINITY;
  // FFmpeg: SMR when the summed sqrt(WSSE) is at least one per frame.
  // Otherwise the arithmetic mean of the per-frame scores (infinity if any
  // frame had WSSE 0). With integer WSSE that else-branch is +inf.
  if (sum_sqrt_wsse >= static_cast<double>(num_frames)) {
    const double avg_dist = sum_sqrt_wsse / static_cast<double>(num_frames);
    const double num = static_cast<double>(static_cast<uint64_t>(width) * height * max_error_sq);
    return 10.0 * std::log10(num / (avg_dist * avg_dist));
  }
  return sum_xpsnr / static_cast<double>(num_frames);
}

double combined_frame_xpsnr(uint64_t wsse_y, uint64_t wsse_u, uint64_t wsse_v, double n_total,
                            double max_error_sq) {
  const double distortion_sum =
      static_cast<double>(wsse_y) + static_cast<double>(wsse_u) + static_cast<double>(wsse_v);
  if (!(distortion_sum > 0.0)) return INFINITY;
  const double d = distortion_sum / n_total;
  return 10.0 * std::log10(max_error_sq / d);
}

}  // namespace

int xpsnr_block_size(int width, int height) {
  const double r =
      (static_cast<double>(width) * static_cast<double>(height)) / (3840.0 * 2160.0);
  const auto rounded = static_cast<int32_t>(32.0 * std::sqrt(r) + 0.5);
  return std::max(1, 4 * rounded);
}

void xpsnr_wsse_from_block_stats(const VideoFormat& fmt, const XpsnrBlockStats* luma,
                                 const uint64_t* sse_u, const uint64_t* sse_v, uint64_t out_wsse[3]) {
  if (!luma || !sse_u || !sse_v || !out_wsse) {
    throw std::runtime_error("xpsnr: missing block statistics");
  }
  const int w = fmt.width;
  const int h = fmt.height;
  const int depth = fmt.bit_depth;
  const int b = xpsnr_block_size(w, h);
  if (b < 4) {
    throw std::runtime_error("xpsnr: block statistics are not used when block size is below 4");
  }
  const int w_blk = (w + b - 1) / b;
  const int h_blk = (h + b - 1) / b;
  const int n_blk = w_blk * h_blk;
  const double r = (static_cast<double>(w) * static_cast<double>(h)) / (3840.0 * 2160.0);
  const double avg_act = std::sqrt(16.0 * static_cast<double>(1 << (2 * depth - 9)) /
                                   std::sqrt(std::max(0.00001, r)));
  const bool high = xpsnr_high_resolution(w, h);
  const int b_val = high ? 2 : 1;
  const bool smooth = static_cast<long long>(w) * h <= 640LL * 480LL;
  const double floor_act = static_cast<double>(1 << (depth - 6));

  std::vector<double> weights(static_cast<size_t>(n_blk), 0.0);
  std::vector<double> sse(static_cast<size_t>(n_blk), 0.0);
  uint32_t idx = 0;
  for (int y = 0; y < h; y += b) {
    const int bh = (y + b > h) ? (h - y) : b;
    for (int x = 0; x < w; x += b, ++idx) {
      const int bw = (x + b > w) ? (w - x) : b;
      const XpsnrBlockStats& st = luma[idx];
      sse[idx] = static_cast<double>(st.sse);
      const int x_act = (x > 0) ? 0 : b_val;
      const int y_act = (y > 0) ? 0 : b_val;
      const int w_act = (x + bw < w) ? bw : (bw - b_val);
      const int h_act = (y + bh < h) ? bh : (bh - b_val);
      const bool host_valid = !(w_act <= x_act || h_act <= y_act);
      if ((st.activity_valid != 0) != host_valid) {
        throw std::runtime_error("xpsnr: activity window flag does not match the block geometry");
      }
      if (!host_valid) {
        weights[idx] = 1.0;
        continue;
      }
      const double window =
          static_cast<double>(w_act - x_act) * static_cast<double>(h_act - y_act);
      double ms = static_cast<double>(st.sa) / window;
      ms += static_cast<double>(st.ta) / (static_cast<double>(bw) * static_cast<double>(bh));
      if (ms < floor_act) ms = floor_act;
      ms *= ms;
      weights[idx] = 1.0 / std::sqrt(ms);
    }
  }
  if (static_cast<int>(idx) != n_blk) {
    throw std::runtime_error("xpsnr: luma block count mismatch");
  }
  if (smooth) {
    idx = 0;
    for (int y = 0; y < h; y += b) {
      for (int x = 0; x < w; x += b, ++idx) {
        double ms_act_prev = 0.0;
        if (x == 0) {
          ms_act_prev = (idx > 1 ? weights[idx - 2] : 0);
        } else {
          ms_act_prev = (x > b ? std::max(weights[idx - 2], weights[idx]) : weights[idx]);
        }
        if (idx > static_cast<uint32_t>(w_blk)) {
          ms_act_prev = std::max(ms_act_prev, weights[idx - 1 - w_blk]);
        }
        if (idx > 0 && weights[idx - 1] > ms_act_prev) weights[idx - 1] = ms_act_prev;
        if ((x + b >= w) && (y + b >= h) && (idx > static_cast<uint32_t>(w_blk))) {
          ms_act_prev = std::max(weights[idx - 1], weights[idx - w_blk]);
          if (weights[idx] > ms_act_prev) weights[idx] = ms_act_prev;
        }
      }
    }
  }

  double wsse_luma = 0.0;
  for (int i = 0; i < n_blk; ++i) wsse_luma += sse[static_cast<size_t>(i)] * weights[static_cast<size_t>(i)];
  out_wsse[0] = (wsse_luma <= 0.0 ? 0 : static_cast<uint64_t>(wsse_luma * avg_act + 0.5));

  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const int bx = b * cw / w;
  const int by = b * ch / h;
  if (bx <= 0 || by <= 0) {
    throw std::runtime_error("xpsnr: chroma block size is zero for this geometry");
  }
  const int c_wblk = (cw + bx - 1) / bx;
  const int c_hblk = (ch + by - 1) / by;
  if (c_wblk != w_blk || c_hblk != h_blk) {
    throw std::runtime_error("xpsnr: chroma block grid does not match luma weights");
  }
  const uint64_t* chroma_sse[2] = {sse_u, sse_v};
  for (int c = 0; c < 2; ++c) {
    double wsse_chroma = 0.0;
    idx = 0;
    for (int y = 0; y < ch; y += by) {
      for (int x = 0; x < cw; x += bx, ++idx) {
        wsse_chroma += static_cast<double>(chroma_sse[c][idx]) * weights[idx];
      }
    }
    out_wsse[c + 1] =
        (wsse_chroma <= 0.0 ? 0 : static_cast<uint64_t>(wsse_chroma * avg_act + 0.5));
  }
}

void XpsnrMeter::start(const VideoFormat& fmt, int integer_frame_rate, int threads) {
  if (fmt.width <= 0 || fmt.height <= 0) {
    throw std::runtime_error("xpsnr: invalid image dimensions");
  }
  if (fmt.bit_depth < 6 || fmt.bit_depth > 16) {
    throw std::runtime_error("xpsnr: bit depth must be from 6 to 16");
  }
  if (integer_frame_rate <= 0) {
    throw std::runtime_error("xpsnr: frame rate must be a positive integer or rational");
  }
  if ((fmt.chroma == ChromaFormat::YUV420 && (fmt.width % 2 || fmt.height % 2)) ||
      (fmt.chroma == ChromaFormat::YUV422 && (fmt.width % 2))) {
    throw std::runtime_error("xpsnr: chroma subsampling requires even dimensions");
  }
  if (!xpsnr_geometry_supported(fmt.width, fmt.height)) {
    throw std::runtime_error(
        "xpsnr: odd luma width or height is not supported when width*height is above "
        "2048*1152; the downsampled high-pass would read past the plane");
  }
  fmt_ = fmt;
  frame_rate_ = integer_frame_rate;
  threads_ = std::max(1, threads);
  frames_ = 0;
  sum_sqrt_wsse_[0] = sum_sqrt_wsse_[1] = sum_sqrt_wsse_[2] = 0;
  sum_xpsnr_[0] = sum_xpsnr_[1] = sum_xpsnr_[2] = 0;
  sum_sqrt_d_ = 0;
  started_ = true;
  hist_ready_ = false;
  m1_.clear();
  m2_.clear();
}

void XpsnrMeter::ensure_history() {
  if (hist_ready_) return;
  const size_t n = static_cast<size_t>(fmt_.width) * static_cast<size_t>(fmt_.height);
  m1_.assign(n, 0);
  m2_.assign(n, 0);
  const int b = xpsnr_block_size(fmt_.width, fmt_.height);
  const int w_blk = (fmt_.width + b - 1) / b;
  const int h_blk = (fmt_.height + b - 1) / b;
  sse_luma_.assign(static_cast<size_t>(w_blk) * h_blk, 0.0);
  weights_.assign(static_cast<size_t>(w_blk) * h_blk, 0.0);
  hist_ready_ = true;
}

void XpsnrMeter::wsse_planes(const Frame& reference, const Frame& reconstructed, uint64_t out[3]) {
  const int w = fmt_.width;
  const int h = fmt_.height;
  const int depth = fmt_.bit_depth;
  const double r = (static_cast<double>(w) * static_cast<double>(h)) / (3840.0 * 2160.0);
  const int b = xpsnr_block_size(w, h);
  const int w_blk = (w + b - 1) / b;
  const double avg_act = std::sqrt(16.0 * static_cast<double>(1 << (2 * depth - 9)) /
                                   std::sqrt(std::max(0.00001, r)));

  out[0] = out[1] = out[2] = 0;
  if (b >= 4) {
    const uint16_t* org = reference.y.ptr();
    const uint16_t* rec = reconstructed.y.ptr();
    const int stride = reference.y.stride;
    const int h_blk = (h + b - 1) / b;
    const bool smooth = static_cast<long long>(w) * h <= 640LL * 480LL;
    uint32_t idx = 0;
    auto score_block = [&](int x, int y, uint32_t block_w, uint32_t block_h, uint32_t block_index) {
      double ms_act = 1.0;
      sse_luma_[block_index] =
          block_sse_and_weight(org, stride, m1_.data(), m2_.data(), rec, stride, w, h, x, y, block_w,
                               block_h, depth, frame_rate_, &ms_act);
      weights_[block_index] = 1.0 / std::sqrt(ms_act);
    };
    if (!smooth && threads_ > 1 && h_blk > 1) {
      // Each block row writes a disjoint history band. The weighted sum below
      // stays in raster order, so the rounded WSSE matches the serial loop.
      PersistentFor::instance().run(threads_, 0, h_blk, [&](int row0, int row1) {
        for (int row = row0; row < row1; ++row) {
          const int y = row * b;
          const uint32_t block_h = static_cast<uint32_t>(y + b > h ? h - y : b);
          uint32_t block_index = static_cast<uint32_t>(row * w_blk);
          for (int x = 0; x < w; x += b, ++block_index) {
            const uint32_t block_w = static_cast<uint32_t>(x + b > w ? w - x : b);
            score_block(x, y, block_w, block_h, block_index);
          }
        }
      });
    }
    for (int y = 0; y < h && (smooth || threads_ <= 1 || h_blk <= 1); y += b) {
      const uint32_t block_h = static_cast<uint32_t>(y + b > h ? h - y : b);
      for (int x = 0; x < w; x += b, ++idx) {
        const uint32_t block_w = static_cast<uint32_t>(x + b > w ? w - x : b);
        double ms_act_prev = 0.0;
        score_block(x, y, block_w, block_h, idx);
        if (smooth) {
          if (x == 0) {
            ms_act_prev = (idx > 1 ? weights_[idx - 2] : 0);
          } else {
            ms_act_prev = (x > b ? std::max(weights_[idx - 2], weights_[idx]) : weights_[idx]);
          }
          if (idx > static_cast<uint32_t>(w_blk)) {
            ms_act_prev = std::max(ms_act_prev, weights_[idx - 1 - w_blk]);
          }
          if (idx > 0 && weights_[idx - 1] > ms_act_prev) weights_[idx - 1] = ms_act_prev;
          if ((x + b >= w) && (y + b >= h) && (idx > static_cast<uint32_t>(w_blk))) {
            ms_act_prev = std::max(weights_[idx - 1], weights_[idx - w_blk]);
            if (weights_[idx] > ms_act_prev) weights_[idx] = ms_act_prev;
          }
        }
      }
    }

    double wsse_luma = 0.0;
    idx = 0;
    for (int y = 0; y < h; y += b) {
      for (int x = 0; x < w; x += b, ++idx) wsse_luma += sse_luma_[idx] * weights_[idx];
    }
    out[0] = (wsse_luma <= 0.0 ? 0 : static_cast<uint64_t>(wsse_luma * avg_act + 0.5));

    const int plane_w[3] = {w, fmt_.chroma_width(), fmt_.chroma_width()};
    const int plane_h[3] = {h, fmt_.chroma_height(), fmt_.chroma_height()};
    const uint16_t* org_p[3] = {reference.y.ptr(), reference.u.ptr(), reference.v.ptr()};
    const uint16_t* rec_p[3] = {reconstructed.y.ptr(), reconstructed.u.ptr(), reconstructed.v.ptr()};
    const int stride_p[3] = {reference.y.stride, reference.u.stride, reference.v.stride};
    for (int c = 1; c < 3; ++c) {
      const uint32_t bx = static_cast<uint32_t>(b * plane_w[c] / w);
      const uint32_t by = static_cast<uint32_t>(b * plane_h[c] / h);
      if (bx == 0 || by == 0) {
        throw std::runtime_error("xpsnr: chroma block size is zero for this geometry");
      }
      double wsse_chroma = 0.0;
      idx = 0;
      for (int y = 0; y < plane_h[c]; y += static_cast<int>(by)) {
        const uint32_t block_h =
            static_cast<uint32_t>(y + static_cast<int>(by) > plane_h[c] ? plane_h[c] - y : by);
        for (int x = 0; x < plane_w[c]; x += static_cast<int>(bx), ++idx) {
          const uint32_t block_w =
              static_cast<uint32_t>(x + static_cast<int>(bx) > plane_w[c] ? plane_w[c] - x : bx);
          wsse_chroma += static_cast<double>(sse_block(org_p[c], stride_p[c], rec_p[c], stride_p[c], x,
                                                       y, block_w, block_h)) *
                         weights_[idx];
        }
      }
      out[c] = (wsse_chroma <= 0.0 ? 0 : static_cast<uint64_t>(wsse_chroma * avg_act + 0.5));
    }
  } else {
    const uint16_t* org_p[3] = {reference.y.ptr(), reference.u.ptr(), reference.v.ptr()};
    const uint16_t* rec_p[3] = {reconstructed.y.ptr(), reconstructed.u.ptr(), reconstructed.v.ptr()};
    const int stride_p[3] = {reference.y.stride, reference.u.stride, reference.v.stride};
    const int plane_w[3] = {w, fmt_.chroma_width(), fmt_.chroma_width()};
    const int plane_h[3] = {h, fmt_.chroma_height(), fmt_.chroma_height()};
    for (int c = 0; c < 3; ++c) {
      out[c] = sse_block(org_p[c], stride_p[c], rec_p[c], stride_p[c], 0, 0,
                         static_cast<uint32_t>(plane_w[c]), static_cast<uint32_t>(plane_h[c]));
    }
  }
}

XpsnrFrame XpsnrMeter::adopt_wsse(uint64_t wsse_y, uint64_t wsse_u, uint64_t wsse_v) {
  if (!started_) throw std::runtime_error("xpsnr: start() was not called");
  const uint64_t wsse[3] = {wsse_y, wsse_u, wsse_v};
  const uint64_t max_sq = static_cast<uint64_t>((1 << fmt_.bit_depth) - 1);
  const uint64_t max_error_sq = max_sq * max_sq;
  const uint32_t dims_w[3] = {static_cast<uint32_t>(fmt_.width),
                              static_cast<uint32_t>(fmt_.chroma_width()),
                              static_cast<uint32_t>(fmt_.chroma_width())};
  const uint32_t dims_h[3] = {static_cast<uint32_t>(fmt_.height),
                              static_cast<uint32_t>(fmt_.chroma_height()),
                              static_cast<uint32_t>(fmt_.chroma_height())};

  XpsnrFrame frame;
  frame.wsse_y = wsse[0];
  frame.wsse_u = wsse[1];
  frame.wsse_v = wsse[2];
  double cur[3];
  for (int c = 0; c < 3; ++c) {
    const double sqrt_wsse = std::sqrt(static_cast<double>(wsse[c]));
    cur[c] = plane_xpsnr_from_sum(sqrt_wsse, INFINITY, dims_w[c], dims_h[c], max_error_sq, 1);
    sum_sqrt_wsse_[c] += sqrt_wsse;
    sum_xpsnr_[c] += cur[c];
  }
  frame.y = cur[0];
  frame.u = cur[1];
  frame.v = cur[2];
  frame.min_score = std::min(frame.y, std::min(frame.u, frame.v));

  const double n_total = static_cast<double>(fmt_.plane_samples('y') + fmt_.plane_samples('u') +
                                             fmt_.plane_samples('v'));
  const double distortion_sum =
      static_cast<double>(wsse[0]) + static_cast<double>(wsse[1]) + static_cast<double>(wsse[2]);
  const double d = distortion_sum / n_total;
  frame.avg = combined_frame_xpsnr(wsse[0], wsse[1], wsse[2], n_total, static_cast<double>(max_error_sq));
  sum_sqrt_d_ += std::sqrt(std::max(0.0, d));
  ++frames_;
  return frame;
}

XpsnrFrame XpsnrMeter::push(const Frame& reference, const Frame& reconstructed) {
  if (!started_) throw std::runtime_error("xpsnr: start() was not called");
  ensure_history();
  uint64_t wsse[3];
  wsse_planes(reference, reconstructed, wsse);
  return adopt_wsse(wsse[0], wsse[1], wsse[2]);
}

XpsnrSummary XpsnrMeter::summary() const {
  XpsnrSummary out;
  out.frames = frames_;
  if (frames_ <= 0) {
    out.score.y = out.score.u = out.score.v = out.score.avg = out.score.min_score = INFINITY;
    return out;
  }
  const uint64_t max_sq = static_cast<uint64_t>((1 << fmt_.bit_depth) - 1);
  const uint64_t max_error_sq = max_sq * max_sq;
  const uint32_t dims_w[3] = {static_cast<uint32_t>(fmt_.width),
                              static_cast<uint32_t>(fmt_.chroma_width()),
                              static_cast<uint32_t>(fmt_.chroma_width())};
  const uint32_t dims_h[3] = {static_cast<uint32_t>(fmt_.height),
                              static_cast<uint32_t>(fmt_.chroma_height()),
                              static_cast<uint32_t>(fmt_.chroma_height())};
  const uint64_t n = static_cast<uint64_t>(frames_);
  out.score.y = plane_xpsnr_from_sum(sum_sqrt_wsse_[0], sum_xpsnr_[0], dims_w[0], dims_h[0], max_error_sq, n);
  out.score.u = plane_xpsnr_from_sum(sum_sqrt_wsse_[1], sum_xpsnr_[1], dims_w[1], dims_h[1], max_error_sq, n);
  out.score.v = plane_xpsnr_from_sum(sum_sqrt_wsse_[2], sum_xpsnr_[2], dims_w[2], dims_h[2], max_error_sq, n);
  out.score.min_score = std::min(out.score.y, std::min(out.score.u, out.score.v));

  // pixvqa avg: SMR of the per-frame combined distortion. R == 0 (every frame
  // identical) is +infinity. There is no log-domain fallback on this extension.
  const double r = sum_sqrt_d_ / static_cast<double>(frames_);
  if (!(r > 0.0)) {
    out.score.avg = INFINITY;
  } else {
    const double max_value = static_cast<double>((1 << fmt_.bit_depth) - 1);
    out.score.avg = 20.0 * std::log10(max_value / r);
  }
  return out;
}

}  // namespace pixvqa
