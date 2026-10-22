// PU21 banding_glare encoder.
//
// Coefficients and the encoding formula are taken from the reference
// implementation matlab/pu21_encoder.m (parameters updated 2020-06-02):
//   https://github.com/gfxdisp/pu21
// Copyright (c) 2021, Graphics and Displays group - University of Cambridge.
// That implementation is BSD-3-Clause. The license text is licenses/PU21_BSD.txt.
// This file reimplements the formula. It does not contain the MATLAB sources.
//
// Mantiuk, R. K., and Azimi, M. (2021). "PU21: A Novel Perceptually Uniform
// Encoding for Adapting Existing Quality Metrics for HDR." Picture Coding
// Symposium (PCS). https://doi.org/10.1109/PCS50896.2021.9477471
//
// Intermediate values use IEEE-754 binary64. The quadratic approximation is
// not used. Encoded values are not rescaled to 0-1. The nominal metric peak
// remains 256, the range at which 100 cd/m2 maps to about 256.

#include "pixvqa/pu21.hpp"

#include "pixvqa/msssim.hpp"
#include "pixvqa/ssim.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pixvqa {
namespace {

constexpr double kLMin = 0.005;
constexpr double kLMax = 10000.0;
constexpr double kPuPeak = 256.0;

double sample_clamp(const uint16_t* data, int stride, int w, int h, int x, int y) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;
  return static_cast<double>(data[static_cast<size_t>(y) * stride + x]);
}

double bilinear_code(const uint16_t* data, int stride, int w, int h, double x, double y) {
  if (w <= 0 || h <= 0) return 0.0;
  if (x < 0.0) x = 0.0;
  if (y < 0.0) y = 0.0;
  const double x_max = static_cast<double>(w - 1);
  const double y_max = static_cast<double>(h - 1);
  if (x > x_max) x = x_max;
  if (y > y_max) y = y_max;
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);
  const double a = sample_clamp(data, stride, w, h, x0, y0);
  const double b = sample_clamp(data, stride, w, h, x1, y0);
  const double c = sample_clamp(data, stride, w, h, x0, y1);
  const double d = sample_clamp(data, stride, w, h, x1, y1);
  return (1.0 - tx) * (1.0 - ty) * a + tx * (1.0 - ty) * b + (1.0 - tx) * ty * c + tx * ty * d;
}

double nearest_code(const uint16_t* data, int stride, int w, int h, double x, double y) {
  const int ix = static_cast<int>(std::floor(x + 0.5));
  const int iy = static_cast<int>(std::floor(y + 0.5));
  return sample_clamp(data, stride, w, h, ix, iy);
}

void chroma_coord(HdrChromaLoc loc, ChromaFormat fmt, int x, int y, double& cx, double& cy) {
  if (fmt == ChromaFormat::YUV444) {
    cx = static_cast<double>(x);
    cy = static_cast<double>(y);
    return;
  }
  if (fmt == ChromaFormat::YUV422) {
    cy = static_cast<double>(y);
    cx = loc == HdrChromaLoc::Center ? (static_cast<double>(x) - 0.5) / 2.0
                                     : static_cast<double>(x) / 2.0;
    return;
  }
  if (loc == HdrChromaLoc::Center) {
    cx = (static_cast<double>(x) - 0.5) / 2.0;
    cy = (static_cast<double>(y) - 0.5) / 2.0;
  } else if (loc == HdrChromaLoc::TopLeft) {
    cx = static_cast<double>(x) / 2.0;
    cy = static_cast<double>(y) / 2.0;
  } else {
    cx = static_cast<double>(x) / 2.0;
    cy = (static_cast<double>(y) - 0.5) / 2.0;
  }
}

double reconstruct_code(const PlaneBuffer& plane, HdrChromaUp up, double x, double y) {
  if (up == HdrChromaUp::Nearest) {
    return nearest_code(plane.ptr(), plane.stride, plane.width, plane.height, x, y);
  }
  return bilinear_code(plane.ptr(), plane.stride, plane.width, plane.height, x, y);
}

struct EncodedPlane {
  int width = 0;
  int height = 0;
  std::vector<double> r;
  std::vector<double> g;
  std::vector<double> b;
  std::vector<double> y;
  bool clamped = false;
};

bool encode_frame(const Frame& frame, const VideoFormat& fmt, const HdrEffective& hdr, EncodedPlane& out,
                  std::string& err) {
  if (fmt.width < 1 || fmt.height < 1) {
    err = "empty frame";
    return false;
  }
  out.width = fmt.width;
  out.height = fmt.height;
  const size_t n = static_cast<size_t>(fmt.width) * fmt.height;
  out.r.assign(n, 0.0);
  out.g.assign(n, 0.0);
  out.b.assign(n, 0.0);
  out.y.assign(n, 0.0);
  out.clamped = false;
  const double kr = hdr.side.matrix == HdrMatrix::Bt709 ? 0.2126 : 0.2627;
  const double kg = hdr.side.matrix == HdrMatrix::Bt709 ? 0.7152 : 0.6780;
  const double kb = hdr.side.matrix == HdrMatrix::Bt709 ? 0.0722 : 0.0593;
  for (int y = 0; y < fmt.height; ++y) {
    const uint16_t* yrow = frame.y.ptr() + static_cast<size_t>(y) * frame.y.stride;
    for (int x = 0; x < fmt.width; ++x) {
      double cx = 0, cy = 0;
      chroma_coord(hdr.chroma_loc, fmt.chroma, x, y, cx, cy);
      const double cb_code = reconstruct_code(frame.u, hdr.chroma_up, cx, cy);
      const double cr_code = reconstruct_code(frame.v, hdr.chroma_up, cx, cy);
      double yn = 0, cbn = 0, crn = 0;
      if (!hdr_codes_to_nominal(fmt.bit_depth, hdr.side.range, static_cast<double>(yrow[x]), cb_code,
                                cr_code, yn, cbn, crn)) {
        err = "unsupported bit depth or range";
        return false;
      }
      double rn = 0, gn = 0, bn = 0;
      if (!hdr_display_rgb(hdr, yn, cbn, crn, rn, gn, bn)) {
        err = "no display transform for this transfer";
        return false;
      }
      const double lum = kr * rn + kg * gn + kb * bn;
      auto encode_one = [&](double nits) {
        if (nits < kLMin || nits > kLMax) out.clamped = true;
        return pu21_encode(nits);
      };
      const size_t i = static_cast<size_t>(y) * fmt.width + x;
      out.r[i] = encode_one(rn);
      out.g[i] = encode_one(gn);
      out.b[i] = encode_one(bn);
      out.y[i] = encode_one(lum);
    }
  }
  return true;
}

}  // namespace

double pu21_encode(double luminance_cdm2) {
  double y = luminance_cdm2;
  if (y < kLMin) y = kLMin;
  if (y > kLMax) y = kLMax;
  // V = max(p7 * (((p1 + p2*Y^p4) / (1 + p3*Y^p4))^p5 - p6), 0)
  constexpr double p1 = 0.353487901;
  constexpr double p2 = 0.3734658629;
  constexpr double p3 = 8.277049286e-05;
  constexpr double p4 = 0.9062562627;
  constexpr double p5 = 0.09150303166;
  constexpr double p6 = 0.9099517204;
  constexpr double p7 = 596.3148142;
  const double yp = std::pow(y, p4);
  const double inner = (p1 + p2 * yp) / (1.0 + p3 * yp);
  const double value = p7 * (std::pow(inner, p5) - p6);
  return value > 0.0 ? value : 0.0;
}

void pu21_reconstructed_chroma(const Frame& frame, const VideoFormat& fmt, HdrChromaLoc loc,
                               HdrChromaUp up, int x, int y, double& cb_code, double& cr_code) {
  double cx = 0, cy = 0;
  chroma_coord(loc, fmt.chroma, x, y, cx, cy);
  cb_code = reconstruct_code(frame.u, up, cx, cy);
  cr_code = reconstruct_code(frame.v, up, cx, cy);
}

std::string pu21_score_frame(const Frame& ref, const Frame& dist, const VideoFormat& fmt,
                             const HdrEffective& ref_hdr, const HdrEffective& dist_hdr, bool want_psnr,
                             bool want_ssim, bool want_msssim, int threads, Pu21FrameScore& out) {
  out = Pu21FrameScore{};
  if (threads < 1) threads = 1;
  if ((want_ssim || want_msssim) && (fmt.width < 11 || fmt.height < 11)) {
    return "PU21-SSIM and PU21-MS-SSIM need at least 11x11 luma";
  }
  EncodedPlane a, b;
  std::string err;
  if (!encode_frame(ref, fmt, ref_hdr, a, err)) return "reference: " + err;
  if (!encode_frame(dist, fmt, dist_hdr, b, err)) return "distorted: " + err;
  out.clamped = a.clamped || b.clamped;
  const size_t n = a.r.size();
  if (want_psnr) {
    long double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double dr = a.r[i] - b.r[i];
      const double dg = a.g[i] - b.g[i];
      const double db = a.b[i] - b.b[i];
      acc += static_cast<long double>(dr * dr + dg * dg + db * db);
    }
    out.mse = static_cast<double>(acc / static_cast<long double>(n * 3));
    if (out.mse == 0.0) {
      out.psnr = std::numeric_limits<double>::infinity();
    } else {
      out.psnr = 10.0 * std::log10((kPuPeak * kPuPeak) / out.mse);
    }
  }
  if (want_ssim) {
    out.ssim = ssim_mean_doubles(a.y.data(), fmt.width, b.y.data(), fmt.width, fmt.width, fmt.height,
                                 kPuPeak, true);
  }
  if (want_msssim) {
    out.msssim = msssim_plane_doubles(a.y.data(), b.y.data(), fmt.width, fmt.height, kPuPeak, threads);
  }
  return {};
}

}  // namespace pixvqa
