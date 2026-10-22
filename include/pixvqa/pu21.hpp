#pragma once

#include "pixvqa/hdr.hpp"
#include "pixvqa/types.hpp"

#include <string>

namespace pixvqa {

// banding_glare PU21. IEEE-754 binary64. Input is clamped to [0.005, 10000].
// Coefficients: gfxdisp/pu21 matlab/pu21_encoder.m. See licenses/PU21_BSD.txt.
double pu21_encode(double luminance_cdm2);

struct Pu21FrameScore {
  double psnr = 0;  // +inf when the RGB mean square error is 0
  double mse = 0;
  double ssim = 0;
  double msssim = 0;
  bool clamped = false;
};

// One reconstruction is shared by every selected PU21 metric.
// SSIM and MS-SSIM run only when requested. A frame smaller than 11x11 cannot
// host those two metrics.
std::string pu21_score_frame(const Frame& ref, const Frame& dist, const VideoFormat& fmt,
                             const HdrEffective& ref_hdr, const HdrEffective& dist_hdr, bool want_psnr,
                             bool want_ssim, bool want_msssim, int threads, Pu21FrameScore& out);

// Reconstructed chroma sample in code units, after siting and upsampling.
void pu21_reconstructed_chroma(const Frame& frame, const VideoFormat& fmt, HdrChromaLoc loc,
                               HdrChromaUp up, int x, int y, double& cb_code, double& cr_code);

}  // namespace pixvqa
