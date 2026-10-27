#include "pixvqa/cpu_dispatch.hpp"
#include "pixvqa/qcs.hpp"
#include "pixvqa/hdr.hpp"
#include "pixvqa/psnr.hpp"
#include "pixvqa/pu21.hpp"
#include "pixvqa/ssim.hpp"
#include "pixvqa/thread_pool.hpp"
#include "pixvqa/msssim.hpp"
#include "pixvqa/xpsnr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace pixvqa;

static int g_fail = 0;

static void expect_near(const char* name, double got, double exp, double tol) {
  if (std::fabs(got - exp) > tol) {
    std::fprintf(stderr, "FAIL %s: got=%.10f exp=%.10f tol=%.2e\n", name, got, exp, tol);
    ++g_fail;
  } else {
    std::printf("OK   %s: %.10f (err=%.3e)\n", name, got, got - exp);
  }
}

static PlaneBuffer fill(int w, int h, uint16_t v) {
  PlaneBuffer p;
  p.allocate(w, h);
  std::fill(p.storage.begin(), p.storage.end(), v);
  return p;
}

static PlaneBuffer checker(int w, int h, int cell, uint16_t lo, uint16_t hi) {
  PlaneBuffer p;
  p.allocate(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const bool on = (((x / cell) + (y / cell)) % 2) != 0;
      p.ptr()[static_cast<size_t>(y) * w + x] = on ? hi : lo;
    }
  }
  return p;
}

static double constant_ssim(double a, double b, double peak) {
  const double c1 = (0.01 * peak) * (0.01 * peak);
  return (2.0 * a * b + c1) / (a * a + b * b + c1);
}

static Frame solid(const VideoFormat& fmt, uint16_t y, uint16_t u, uint16_t v) {
  Frame frame;
  frame.allocate(fmt);
  std::fill(frame.y.storage.begin(), frame.y.storage.end(), y);
  std::fill(frame.u.storage.begin(), frame.u.storage.end(), u);
  std::fill(frame.v.storage.begin(), frame.v.storage.end(), v);
  return frame;
}

int main() {
  // PSNR identical → inf / printed 999.99 path
  {
    auto a = fill(64, 64, 512);
    auto b = fill(64, 64, 512);
    const double mse = plane_mse(a.view(), b.view());
    expect_near("mse_identical", mse, 0.0, 0);
  }
  {
    auto a = fill(64, 64, 512);
    auto b = fill(64, 64, 520);
    const double mse = plane_mse(a.view(), b.view());
    expect_near("mse_delta8", mse, 64.0, 1e-12);
  }

  // Classical SSIM. Constant fields have a closed form: variances are zero.
  {
    auto a = fill(32, 32, 512);
    auto b = fill(32, 32, 512);
    expect_near("ssim_identical_scalar", ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Scalar), 1.0, 1e-12);
    expect_near("ssim_identical_avx2", ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Avx2), 1.0, 1e-9);
    const double dark = constant_ssim(0.0, 1023.0, 1023.0);
    auto z = fill(32, 32, 0);
    auto peak = fill(32, 32, 1023);
    expect_near("ssim_const_10bit", ssim_mean(z.view(), peak.view(), 1023.0, true, SsimKernelMode::Scalar), dark, 1e-12);
    expect_near("ssim_const_10bit_avx2", ssim_mean(z.view(), peak.view(), 1023.0, true, SsimKernelMode::Avx2), dark, 1e-9);
    auto z8 = fill(32, 32, 0);
    auto p8 = fill(32, 32, 255);
    expect_near("ssim_const_8bit", ssim_mean(z8.view(), p8.view(), 255.0, true, SsimKernelMode::Scalar), constant_ssim(0.0, 255.0, 255.0), 1e-12);
  }
  {
    auto a = checker(64, 64, 4, 100, 900);
    auto b = checker(64, 64, 4, 900, 100);
    const double scalar = ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Scalar, 1);
    const double tiled = ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Avx2, 4);
    const double full = ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Avx2Full, 2);
    expect_near("ssim_distorted_below_one", scalar < 1.0 ? 1.0 : 0.0, 1.0, 0);
    expect_near("ssim_scalar_vs_avx2", tiled, scalar, 1e-9);
    expect_near("ssim_tiled_vs_full", full, scalar, 1e-9);
    const double t1 = ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Avx2, 1);
    expect_near("ssim_threads_1_vs_4", t1, tiled, 1e-12);
  }
  {
    VideoFormat fmt;
    fmt.width = 32;
    fmt.height = 32;
    fmt.bit_depth = 10;
    fmt.chroma = ChromaFormat::YUV420;
    Frame ref = solid(fmt, 400, 512, 512);
    Frame dist = solid(fmt, 400, 0, 512);
    const PlaneScore s = compute_ssim_frame(ref, dist, fmt, 1023.0, SsimKernelMode::Scalar, 1);
    expect_near("ssim_y_identical", s.y, 1.0, 1e-12);
    expect_near("ssim_u_constant", s.u, constant_ssim(512.0, 0.0, 1023.0), 1e-12);
    expect_near("ssim_v_identical", s.v, 1.0, 1e-12);
    const double avg = (4.0 * s.y + s.u + s.v) / 6.0;
    expect_near("ssim_yuv420_weight", s.avg, avg, 1e-12);
    fmt.chroma = ChromaFormat::YUV444;
    Frame ref4 = solid(fmt, 400, 200, 800);
    Frame dist4 = solid(fmt, 100, 200, 800);
    const PlaneScore s4 = compute_ssim_frame(ref4, dist4, fmt, 1023.0, SsimKernelMode::Scalar, 1);
    expect_near("ssim_yuv444_weight", s4.avg, (s4.y + s4.u + s4.v) / 3.0, 1e-12);
  }
  {
    VideoFormat fmt;
    fmt.width = 64;
    fmt.height = 64;
    fmt.bit_depth = 10;
    fmt.chroma = ChromaFormat::YUV420;
    Frame a = solid(fmt, 400, 512, 512);
    const PlaneScore msssim = compute_msssim_frame(a, a, fmt, 1023.0, 1);
    expect_near("msssim_identical", msssim.y, 1.0, 1e-12);
    expect_near("msssim_identical_avg", msssim.avg, 1.0, 1e-12);
    if (cpu_has_avx2()) {
      PlaneBuffer xa = checker(64, 64, 5, 80, 400);
      PlaneBuffer xb = checker(64, 64, 7, 90, 500);
      const double scalar = msssim_plane_scalar(xa.view(), xb.view(), 1023.0, 1, nullptr);
      const double avx2 = msssim_plane_avx2(xa.view(), xb.view(), 1023.0, 1, nullptr);
      expect_near("msssim_scalar_vs_avx2", avx2, scalar, 1e-9);
    }
    bool too_small = false;
    try {
      auto tiny = fill(8, 8, 10);
      (void)ssim_mean(tiny.view(), tiny.view(), 1023.0, true, SsimKernelMode::Scalar);
    } catch (const std::runtime_error&) {
      too_small = true;
    }
    if (!too_small) {
      std::fprintf(stderr, "FAIL ssim_rejects_small_plane\n");
      ++g_fail;
    } else {
      std::printf("OK   ssim_rejects_small_plane\n");
    }
  }

  // A throwing task must surface and leave the pool usable.
  {
    bool threw = false;
    try {
      parallel_for_ranges(4, 0, 32, [](int lo, int) {
        if (lo == 8) throw std::runtime_error("boom");
      });
    } catch (const std::runtime_error&) {
      threw = true;
    }
    if (!threw) {
      std::fprintf(stderr, "FAIL exception_propagates\n");
      ++g_fail;
    } else {
      std::printf("OK   exception_propagates\n");
    }
    auto a = checker(64, 64, 2, 200, 800);
    auto b = checker(64, 64, 2, 800, 200);
    const double after = ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Avx2, 2);
    const double full = ssim_mean(a.view(), b.view(), 1023.0, true, SsimKernelMode::Avx2Full, 1);
    expect_near("after_exception", after, full, 1e-12);
  }

  expect_near("xpsnr_block_2160", xpsnr_block_size(3840, 2160), 128, 0);
  expect_near("xpsnr_block_1080", xpsnr_block_size(1920, 1080), 64, 0);
  expect_near("xpsnr_order_30", xpsnr_temporal_order(30), 1, 0);
  expect_near("xpsnr_order_32", xpsnr_temporal_order(32), 2, 0);
  expect_near("xpsnr_rate_ntsc", xpsnr_integer_rate(60000, 1001), 59, 0);
  if (!xpsnr_geometry_supported(3840, 2160) || !xpsnr_geometry_supported(1920, 1080) ||
      !xpsnr_geometry_supported(1921, 1080) || !xpsnr_geometry_supported(2048, 1152) ||
      !xpsnr_geometry_supported(2048, 1154) || !xpsnr_geometry_supported(1600, 1600) ||
      xpsnr_geometry_supported(1600, 1601) || xpsnr_geometry_supported(1601, 1600) ||
      xpsnr_geometry_supported(1601, 1601) || xpsnr_geometry_supported(2048, 1153)) {
    std::fprintf(stderr, "FAIL xpsnr_geometry_supported\n");
    ++g_fail;
  } else {
    std::printf("OK   xpsnr_geometry_supported\n");
  }
  if (!xpsnr_high_resolution(3840, 2160) || xpsnr_high_resolution(1920, 1080)) {
    std::fprintf(stderr, "FAIL xpsnr_high_resolution gate\n");
    ++g_fail;
  } else {
    std::printf("OK   xpsnr_high_resolution\n");
  }

  {
    VideoFormat fmt;
    fmt.width = 64;
    fmt.height = 64;
    fmt.bit_depth = 10;
    fmt.chroma = ChromaFormat::YUV420;
    Frame a, b;
    a.allocate(fmt);
    b.allocate(fmt);
    std::fill(a.y.storage.begin(), a.y.storage.end(), 400);
    std::fill(a.u.storage.begin(), a.u.storage.end(), 512);
    std::fill(a.v.storage.begin(), a.v.storage.end(), 512);
    b = a;
    b.y.ptr()[0] = 408;
    XpsnrMeter meter;
    meter.start(fmt, 60);
    const XpsnrFrame f0 = meter.push(a, b);
    if (!std::isfinite(f0.y) || std::isfinite(f0.u) || std::isfinite(f0.v) || !std::isfinite(f0.avg)) {
      std::fprintf(stderr, "FAIL xpsnr_one_pixel_domains y=%g u=%g v=%g avg=%g\n", f0.y, f0.u, f0.v,
                   f0.avg);
      ++g_fail;
    } else {
      std::printf("OK   xpsnr_one_pixel_domains\n");
    }
    const double n_total =
        static_cast<double>(fmt.plane_samples('y') + fmt.plane_samples('u') + fmt.plane_samples('v'));
    const double d = (static_cast<double>(f0.wsse_y) + f0.wsse_u + f0.wsse_v) / n_total;
    const double expect = 10.0 * std::log10((1023.0 * 1023.0) / d);
    expect_near("xpsnr_avg_from_wsse", f0.avg, expect, 1e-9);
    const double db_mean = (f0.y + f0.u + f0.v) / 3.0;
    if (std::isfinite(db_mean) || std::fabs(f0.avg - f0.y) < 1e-6) {
      std::fprintf(stderr, "FAIL xpsnr_avg_is_not_db_mean avg=%g y=%g\n", f0.avg, f0.y);
      ++g_fail;
    } else {
      std::printf("OK   xpsnr_avg_is_not_db_mean\n");
    }

    XpsnrMeter temporal;
    temporal.start(fmt, 30);
    (void)temporal.push(a, b);
    const XpsnrFrame slow = temporal.push(a, b);
    XpsnrMeter temporal60;
    temporal60.start(fmt, 60);
    (void)temporal60.push(a, b);
    const XpsnrFrame fast = temporal60.push(a, b);
    if (!(std::fabs(slow.y - fast.y) > 1e-6)) {
      std::fprintf(stderr, "FAIL xpsnr_fps_order slow=%g fast=%g\n", slow.y, fast.y);
      ++g_fail;
    } else {
      std::printf("OK   xpsnr_fps_order\n");
    }

    XpsnrMeter again;
    again.start(fmt, 60);
    const XpsnrFrame replay = again.push(a, b);
    expect_near("xpsnr_reset", replay.y, f0.y, 0.0);

    XpsnrMeter adopted;
    adopted.start(fmt, 60);
    const XpsnrFrame via = adopted.adopt_wsse(f0.wsse_y, f0.wsse_u, f0.wsse_v);
    if (via.wsse_y != f0.wsse_y || via.wsse_u != f0.wsse_u || via.wsse_v != f0.wsse_v ||
        std::fabs(via.y - f0.y) > 0.0 || std::fabs(via.avg - f0.avg) > 0.0 ||
        std::isinf(via.u) != std::isinf(f0.u) || std::isinf(via.v) != std::isinf(f0.v)) {
      std::fprintf(stderr, "FAIL xpsnr_adopt_wsse\n");
      ++g_fail;
    } else {
      std::printf("OK   xpsnr_adopt_wsse\n");
    }

    VideoFormat odd;
    odd.width = 1600;
    odd.height = 1601;
    odd.bit_depth = 10;
    odd.chroma = ChromaFormat::YUV444;
    bool rejected = false;
    try {
      XpsnrMeter bad;
      bad.start(odd, 60);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected) {
      std::fprintf(stderr, "FAIL xpsnr_odd_highres_rejected\n");
      ++g_fail;
    } else {
      std::printf("OK   xpsnr_odd_highres_rejected\n");
    }
  }

  {
    auto fail = [](const char* name) {
      std::fprintf(stderr, "FAIL %s\n", name);
      ++g_fail;
    };
    HdrSide absent;
    if (!hdr_reject_reason(absent, absent, false, 0).empty()) fail("hdr_missing_pair");
    else std::printf("OK   hdr_missing_pair\n");

    HdrSide pq;
    pq.transfer = HdrTransfer::Pq;
    pq.transfer_source = HdrSource::Stream;
    pq.primaries = HdrPrimaries::Bt2020;
    pq.primaries_source = HdrSource::Stream;
    pq.matrix = HdrMatrix::Bt2020Ncl;
    pq.matrix_source = HdrSource::Stream;
    pq.range = HdrRange::Limited;
    pq.range_source = HdrSource::Stream;
    const std::string missing = hdr_reject_reason(pq, absent, false, 0);
    if (!missing.empty()) fail("hdr_pq_vs_missing");
    else std::printf("OK   hdr_pq_vs_missing\n");

    HdrSide hlg = pq;
    hlg.transfer = HdrTransfer::Hlg;
    const std::string clash = hdr_reject_reason(pq, hlg, false, 0);
    if (!clash.empty()) fail("hdr_pq_vs_hlg");
    else std::printf("OK   hdr_pq_vs_hlg\n");

    if (!hdr_reject_reason(pq, pq, false, 0).empty()) fail("hdr_matching_pq");
    else std::printf("OK   hdr_matching_pq\n");

    HdrSide bare = pq;
    bare.range = HdrRange::Unspecified;
    bare.range_source = HdrSource::Absent;
    if (!hdr_reject_reason(bare, bare, false, 0).empty()) {
      fail("hdr_pq_range_missing");
    } else {
      std::printf("OK   hdr_pq_range_missing\n");
    }

    HdrSide full = pq;
    full.range = HdrRange::Full;
    if (!hdr_reject_reason(pq, full, false, 0).empty()) fail("hdr_range_independent");
    else std::printf("OK   hdr_range_independent\n");

    if (!hdr_reject_reason(hlg, hlg, false, 0).empty()) fail("hdr_hlg_peak_optional");
    else std::printf("OK   hdr_hlg_peak_optional\n");
    if (!hdr_reject_reason(hlg, hlg, true, 1000.0).empty()) fail("hdr_hlg_peak_1000");
    else std::printf("OK   hdr_hlg_peak_1000\n");
    if (hdr_reject_reason(pq, pq, true, 1000.0).find("hlg-peak") == std::string::npos) fail("hdr_pq_rejects_hlg_peak");
    else std::printf("OK   hdr_pq_rejects_hlg_peak\n");

    HdrSide bt601 = pq;
    bt601.primaries = HdrPrimaries::Bt601;
    bt601.matrix = HdrMatrix::Bt601;
    if (hdr_reject_reason(bt601, bt601, false, 0).find("not a supported") == std::string::npos) {
      fail("hdr_bt601_pq_rejected");
    } else {
      std::printf("OK   hdr_bt601_pq_rejected\n");
    }

    HdrSide overridden = hlg;
    overridden.transfer_conflict = true;
    const std::string applied = hdr_apply_override(overridden, true, "st2084", true, "bt2020l");
    if (!applied.empty() || overridden.transfer != HdrTransfer::Pq ||
        overridden.transfer_source != HdrSource::Override || overridden.transfer_conflict ||
        overridden.range != HdrRange::Limited || !hdr_reject_reason(overridden, pq, false, 0).empty()) {
      fail("hdr_override_replaces_conflict");
    } else {
      std::printf("OK   hdr_override_replaces_conflict\n");
    }

    HdrSide observed;
    hdr_observe(observed, HdrTransfer::Pq, HdrPrimaries::Bt2020, HdrMatrix::Bt2020Ncl, HdrRange::Limited);
    hdr_observe(observed, HdrTransfer::Hlg, HdrPrimaries::Bt2020, HdrMatrix::Bt2020Ncl, HdrRange::Limited);
    if (!observed.transfer_conflict || observed.transfer != HdrTransfer::Pq) fail("hdr_stream_conflict_keeps_first");
    else std::printf("OK   hdr_stream_conflict_keeps_first\n");
    hdr_observe(observed, HdrTransfer::Unspecified, HdrPrimaries::Unspecified, HdrMatrix::Unspecified,
                HdrRange::Unspecified);
    if (!observed.transfer_conflict) fail("hdr_unspecified_does_not_clear_conflict");
    else std::printf("OK   hdr_unspecified_does_not_clear_conflict\n");

    double yn = 0, cbn = 0, crn = 0, nits = -1;
    if (!hdr_codes_to_nominal(10, HdrRange::Limited, 64, 512, 512, yn, cbn, crn) || yn != 0.0 || cbn != 0.0 ||
        crn != 0.0 || !hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt2020Ncl, yn, cbn, crn, 0, nits) ||
        nits != 0.0) {
      fail("hdr_pq_limited_black");
    } else {
      std::printf("OK   hdr_pq_limited_black\n");
    }
    if (!hdr_codes_to_nominal(10, HdrRange::Limited, 940, 512, 512, yn, cbn, crn) || yn != 1.0 ||
        !hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt2020Ncl, yn, cbn, crn, 0, nits) ||
        std::fabs(nits - 10000.0) > 1e-4) {
      fail("hdr_pq_limited_peak");
    } else {
      std::printf("OK   hdr_pq_limited_peak: %.8f\n", nits);
    }
    if (!hdr_codes_to_nominal(8, HdrRange::Limited, 235, 128, 128, yn, cbn, crn) ||
        !hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt709, yn, cbn, crn, 0, nits) ||
        std::fabs(nits - 10000.0) > 1e-4) {
      fail("hdr_pq_bt709_8bit_peak");
    } else {
      std::printf("OK   hdr_pq_bt709_8bit_peak\n");
    }

    constexpr double m1 = 2610.0 / 16384.0;
    constexpr double m2 = 2523.0 / 32.0;
    constexpr double c1 = 3424.0 / 4096.0;
    constexpr double c2 = 2413.0 / 128.0;
    constexpr double c3 = 2392.0 / 128.0;
    const double y100 = 100.0 / 10000.0;
    const double e100 = std::pow((c1 + c2 * std::pow(y100, m1)) / (1.0 + c3 * std::pow(y100, m1)), m2);
    double grey100 = 0;
    if (!hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt2020Ncl, e100, 0, 0, 0, grey100) ||
        std::fabs(grey100 - 100.0) > 1e-6) {
      std::fprintf(stderr, "FAIL hdr_pq_100nit_roundtrip got=%.10f e=%.10f\n", grey100, e100);
      ++g_fail;
    } else {
      std::printf("OK   hdr_pq_100nit_roundtrip: %.8f\n", grey100);
    }

    double grey = 0, chroma = 0;
    if (!hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt2020Ncl, 0.5, 0, 0, 0, grey) ||
        !hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt2020Ncl, 0.5, 0.08, -0.04, 0, chroma) ||
        !(std::fabs(chroma - grey) > 1.0)) {
      fail("hdr_pq_chroma_uses_rgb_eotf");
    } else {
      std::printf("OK   hdr_pq_chroma_uses_rgb_eotf grey=%.6f chroma=%.6f\n", grey, chroma);
    }

    double hlg_peak = 0, hlg_black = 0, hlg_mid = 0;
    if (!hdr_display_cdm2(HdrTransfer::Hlg, HdrMatrix::Bt2020Ncl, 1, 0, 0, 1000, hlg_peak) ||
        std::fabs(hlg_peak - 1000.0) > 1e-6 ||
        !hdr_display_cdm2(HdrTransfer::Hlg, HdrMatrix::Bt2020Ncl, 0, 0, 0, 1000, hlg_black) ||
        hlg_black != 0.0) {
      fail("hdr_hlg_grey_ends");
    } else {
      std::printf("OK   hdr_hlg_grey_ends\n");
    }
    const double scene_mid = (0.5 * 0.5) / 3.0;
    const double expect_mid = 1000.0 * std::pow(scene_mid, hdr_hlg_system_gamma(1000.0));
    if (!hdr_display_cdm2(HdrTransfer::Hlg, HdrMatrix::Bt2020Ncl, 0.5, 0, 0, 1000, hlg_mid) ||
        std::fabs(hlg_mid - expect_mid) > 1e-8) {
      std::fprintf(stderr, "FAIL hdr_hlg_mid got=%.10f exp=%.10f\n", hlg_mid, expect_mid);
      ++g_fail;
    } else {
      std::printf("OK   hdr_hlg_mid: %.8f\n", hlg_mid);
    }
    double hlg_400 = 0;
    if (!hdr_display_cdm2(HdrTransfer::Hlg, HdrMatrix::Bt2020Ncl, 1, 0, 0, 400, hlg_400) ||
        std::fabs(hlg_400 - 400.0) > 1e-6 ||
        std::fabs(hdr_hlg_system_gamma(400.0) - (1.2 + 0.42 * std::log10(0.4))) > 1e-12) {
      fail("hdr_hlg_peak_400");
    } else {
      std::printf("OK   hdr_hlg_peak_400 gamma=%.9f\n", hdr_hlg_system_gamma(400.0));
    }

    const double kr = 0.2627, kb = 0.0593, kg = 0.6780;
    const double y = 0.75, cb = 0.05, cr = -0.02;
    const double r = y + 2.0 * (1.0 - kr) * cr;
    const double b = y + 2.0 * (1.0 - kb) * cb;
    const double g = (y - kr * r - kb * b) / kg;
    auto hlg_e = [](double e) {
      constexpr double a = 0.17883277;
      const double bb = 1.0 - 4.0 * a;
      const double c = 1.0 - a * std::log(12.0 - bb);
      if (e <= 0.5) return (e * e) / 3.0;
      return (std::exp((e - c) / a) + bb) / 12.0;
    };
    const double ys = kr * hlg_e(r) + kg * hlg_e(g) + kb * hlg_e(b);
    const double rgb_nits = 1000.0 * std::pow(ys, hdr_hlg_system_gamma(1000.0));
    const double luma_only = 1000.0 * std::pow(hlg_e(y), hdr_hlg_system_gamma(1000.0));
    double got_nits = 0;
    if (!hdr_display_cdm2(HdrTransfer::Hlg, HdrMatrix::Bt2020Ncl, y, cb, cr, 1000, got_nits) ||
        std::fabs(got_nits - rgb_nits) > 1e-7 || std::fabs(got_nits - luma_only) < 1e-3) {
      std::fprintf(stderr, "FAIL hdr_hlg_chroma got=%.8f rgb=%.8f luma=%.8f\n", got_nits, rgb_nits, luma_only);
      ++g_fail;
    } else {
      std::printf("OK   hdr_hlg_chroma rgb=%.6f luma_only=%.6f\n", got_nits, luma_only);
    }

    double ignored = 0;
    if (hdr_display_cdm2(HdrTransfer::Bt709, HdrMatrix::Bt2020Ncl, 1, 0, 0, 0, ignored) ||
        hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt601, 1, 0, 0, 0, ignored)) {
      fail("hdr_unsupported_transform");
    } else {
      std::printf("OK   hdr_unsupported_transform\n");
    }

    const std::string shown = hdr_display_line(pq, 0);
    if (shown.find("peak=10000") == std::string::npos || shown.find("st2084") == std::string::npos ||
        hdr_display_line(hlg, 1000).find("gamma=1.200000000") == std::string::npos) {
      fail("hdr_display_line");
    } else {
      std::printf("OK   hdr_display_line\n");
    }
    if (hdr_side_line("reference", absent).find("transfer=unspecified (absent)") == std::string::npos) {
      fail("hdr_side_line_absent");
    } else {
      std::printf("OK   hdr_side_line_absent\n");
    }

    HdrKnobs knobs;
    HdrEffective resolved;
    HdrSide bare_pq;
    bare_pq.transfer = HdrTransfer::Pq;
    bare_pq.transfer_source = HdrSource::Override;
    if (!hdr_resolve(bare_pq, knobs, resolved).empty() || resolved.side.primaries != HdrPrimaries::Bt2020 ||
        resolved.side.primaries_source != HdrSource::Assumption ||
        resolved.side.matrix != HdrMatrix::Bt2020Ncl || resolved.side.range != HdrRange::Limited ||
        resolved.pq_clip) {
      fail("hdr_pq_defaults");
    } else {
      std::printf("OK   hdr_pq_defaults\n");
    }
    knobs.require_hdr_transfer = true;
    if (hdr_resolve(absent, knobs, resolved).find("unspecified") == std::string::npos) {
      fail("hdr_pu21_requires_transfer");
    } else {
      std::printf("OK   hdr_pu21_requires_transfer\n");
    }
    knobs = HdrKnobs{};
    if (!hdr_resolve(hlg, knobs, resolved).empty() || std::fabs(resolved.peak_nits - 1000.0) > 1e-12 ||
        resolved.peak_source != HdrSource::Assumption || std::fabs(resolved.hlg_gamma - 1.2) > 1e-12) {
      fail("hdr_hlg_default_peak");
    } else {
      std::printf("OK   hdr_hlg_default_peak\n");
    }
    knobs.peak_set = true;
    knobs.peak_nits = 400.0;
    const double gamma_400 = 1.2 + 0.42 * std::log10(0.4);
    if (!hdr_resolve(hlg, knobs, resolved).empty() || std::fabs(resolved.hlg_gamma - gamma_400) > 1e-12 ||
        resolved.peak_source != HdrSource::Override || resolved.gamma_source != HdrSource::Assumption) {
      fail("hdr_hlg_gamma_follows_peak");
    } else {
      std::printf("OK   hdr_hlg_gamma_follows_peak\n");
    }
    knobs.gamma_set = true;
    knobs.gamma = 1.0;
    if (!hdr_resolve(hlg, knobs, resolved).empty() || resolved.hlg_gamma != 1.0 ||
        resolved.gamma_source != HdrSource::Override) {
      fail("hdr_hlg_explicit_gamma");
    } else {
      std::printf("OK   hdr_hlg_explicit_gamma\n");
    }

    expect_near("pu21_encode_0.005", pu21_encode(0.005), 5.4704566540382245e-10, 1e-18);
    expect_near("pu21_encode_clamp_low", pu21_encode(0.001), pu21_encode(0.005), 0.0);
    expect_near("pu21_encode_0.1", pu21_encode(0.1), 5.7170738396694469, 1e-9);
    expect_near("pu21_encode_1", pu21_encode(1.0), 36.543911139419201, 1e-9);
    expect_near("pu21_encode_10", pu21_encode(10.0), 123.64748355384738, 1e-9);
    expect_near("pu21_encode_100", pu21_encode(100.0), 256.38389731270388, 1e-9);
    expect_near("pu21_encode_1000", pu21_encode(1000.0), 420.09692134924433, 1e-9);
    expect_near("pu21_encode_10000", pu21_encode(10000.0), 595.39392002009504, 1e-8);
    expect_near("pu21_encode_clamp_high", pu21_encode(20000.0), pu21_encode(10000.0), 0.0);
    if (!(pu21_encode(0.1) < pu21_encode(1.0) && pu21_encode(1.0) < pu21_encode(100.0) &&
          pu21_encode(100.0) < pu21_encode(1000.0) && pu21_encode(1000.0) < pu21_encode(10000.0))) {
      fail("pu21_encode_monotone");
    } else {
      std::printf("OK   pu21_encode_monotone\n");
    }

    HdrEffective pq_eff;
    knobs = HdrKnobs{};
    if (!hdr_resolve(pq, knobs, pq_eff).empty()) fail("hdr_pq_resolve_for_rgb");
    double rr = 0, gg = 0, bb = 0;
    const double y_c = 0.75, cb_c = 0.05, cr_c = -0.02;
    if (!hdr_display_rgb(pq_eff, y_c, cb_c, cr_c, rr, gg, bb)) fail("hdr_pq_rgb");
    else {
      const double lum = 0.2627 * rr + 0.6780 * gg + 0.0593 * bb;
      double scalar = 0;
      if (!hdr_display_cdm2(HdrTransfer::Pq, HdrMatrix::Bt2020Ncl, y_c, cb_c, cr_c, 0, scalar) ||
          std::fabs(lum - scalar) > 1e-8) {
        fail("hdr_pq_rgb_luminance");
      } else {
        std::printf("OK   hdr_pq_rgb_luminance\n");
      }
    }
    HdrEffective hlg_eff;
    knobs.legacy_hlg_peak = true;
    knobs.legacy_hlg_peak_nits = 1000.0;
    if (!hdr_resolve(hlg, knobs, hlg_eff).empty() ||
        !hdr_display_rgb(hlg_eff, y_c, cb_c, cr_c, rr, gg, bb)) {
      fail("hdr_hlg_rgb");
    } else {
      const double lum = 0.2627 * rr + 0.6780 * gg + 0.0593 * bb;
      double scalar = 0;
      if (!hdr_display_cdm2(HdrTransfer::Hlg, HdrMatrix::Bt2020Ncl, y_c, cb_c, cr_c, 1000, scalar) ||
          std::fabs(lum - scalar) > 1e-7) {
        std::fprintf(stderr, "FAIL hdr_hlg_rgb_luminance lum=%.10f scalar=%.10f\n", lum, scalar);
        ++g_fail;
      } else {
        std::printf("OK   hdr_hlg_rgb_luminance\n");
      }
    }

    VideoFormat fmt444;
    fmt444.width = 32;
    fmt444.height = 32;
    fmt444.bit_depth = 10;
    fmt444.chroma = ChromaFormat::YUV444;
    Frame grey_a;
    Frame grey_b;
    grey_a.allocate(fmt444);
    grey_b.allocate(fmt444);
    auto paint = [](Frame& frame, uint16_t ycode, uint16_t chroma) {
      std::fill(frame.y.storage.begin(), frame.y.storage.end(), ycode);
      std::fill(frame.u.storage.begin(), frame.u.storage.end(), chroma);
      std::fill(frame.v.storage.begin(), frame.v.storage.end(), chroma);
    };
    paint(grey_a, 512, 512);
    paint(grey_b, 512, 512);
    Pu21FrameScore same;
    const std::string same_err =
        pu21_score_frame(grey_a, grey_b, fmt444, pq_eff, pq_eff, true, true, true, 1, same);
    if (!same_err.empty() || !std::isinf(same.psnr) || same.mse != 0.0 || std::fabs(same.ssim - 1.0) > 1e-12 ||
        std::fabs(same.msssim - 1.0) > 1e-12) {
      std::fprintf(stderr, "FAIL pu21_identical err=%s psnr=%.6f ssim=%.8f msssim=%.8f\n", same_err.c_str(),
                   same.psnr, same.ssim, same.msssim);
      ++g_fail;
    } else {
      std::printf("OK   pu21_identical\n");
    }
    paint(grey_b, 600, 512);
    double grey_y = 0, grey_cb = 0, grey_cr = 0;
    hdr_codes_to_nominal(10, HdrRange::Limited, 512, 512, 512, grey_y, grey_cb, grey_cr);
    double r0 = 0, g0 = 0, b0 = 0, r1 = 0, g1 = 0, b1 = 0;
    hdr_display_rgb(pq_eff, grey_y, grey_cb, grey_cr, r0, g0, b0);
    hdr_codes_to_nominal(10, HdrRange::Limited, 600, 512, 512, grey_y, grey_cb, grey_cr);
    hdr_display_rgb(pq_eff, grey_y, grey_cb, grey_cr, r1, g1, b1);
    const double pu0 = pu21_encode(0.2627 * r0 + 0.6780 * g0 + 0.0593 * b0);
    const double pu1 = pu21_encode(0.2627 * r1 + 0.6780 * g1 + 0.0593 * b1);
    const double pu_r0 = pu21_encode(r0);
    const double pu_r1 = pu21_encode(r1);
    const double mse = ((pu_r0 - pu_r1) * (pu_r0 - pu_r1)) ;
    const double expect_psnr = 10.0 * std::log10((256.0 * 256.0) / mse);
    const double ssim_c1 = (0.01 * 256.0) * (0.01 * 256.0);
    const double expect_ssim = (2.0 * pu0 * pu1 + ssim_c1) / (pu0 * pu0 + pu1 * pu1 + ssim_c1);
    const int scales = msssim_scales(32, 32);
    double expect_msssim = std::pow(expect_ssim, kMsssimWeight[scales - 1]);
    Pu21FrameScore diff;
    const std::string diff_err =
        pu21_score_frame(grey_a, grey_b, fmt444, pq_eff, pq_eff, true, true, true, 1, diff);
    if (!diff_err.empty()) {
      std::fprintf(stderr, "FAIL pu21_grey_diff %s\n", diff_err.c_str());
      ++g_fail;
    } else {
      expect_near("pu21_psnr_grey", diff.psnr, expect_psnr, 1e-8);
      expect_near("pu21_ssim_grey", diff.ssim, expect_ssim, 1e-10);
      expect_near("pu21_msssim_grey", diff.msssim, expect_msssim, 1e-10);
    }
    HdrEffective hlg_score;
    knobs = HdrKnobs{};
    knobs.peak_set = true;
    knobs.peak_nits = 1000.0;
    hdr_resolve(hlg, knobs, hlg_score);
    Pu21FrameScore hlg_diff;
    if (!pu21_score_frame(grey_a, grey_b, fmt444, hlg_score, hlg_score, true, true, true, 1, hlg_diff).empty() ||
        !(hlg_diff.psnr > 0.0) || std::fabs(hlg_diff.psnr - diff.psnr) < 1e-3) {
      fail("pu21_hlg_differs_from_pq");
    } else {
      std::printf("OK   pu21_hlg_differs_from_pq: %.6f vs %.6f\n", hlg_diff.psnr, diff.psnr);
    }
    Pu21FrameScore crossed;
    if (!pu21_score_frame(grey_a, grey_b, fmt444, pq_eff, hlg_score, true, false, false, 1, crossed).empty() ||
        !(crossed.psnr > 0.0)) {
      fail("pu21_independent_transfers");
    } else {
      std::printf("OK   pu21_independent_transfers: %.6f\n", crossed.psnr);
    }

    VideoFormat fmt420 = fmt444;
    fmt420.chroma = ChromaFormat::YUV420;
    Frame ramp;
    ramp.allocate(fmt420);
    std::fill(ramp.y.storage.begin(), ramp.y.storage.end(), 400);
    std::fill(ramp.v.storage.begin(), ramp.v.storage.end(), 512);
    ramp.u.storage[0] = 100;
    if (ramp.u.storage.size() > 1) ramp.u.storage[1] = 300;
    double cb_left = 0, cr_left = 0, cb_mid = 0, cr_mid = 0;
    pu21_reconstructed_chroma(ramp, fmt420, HdrChromaLoc::Left, HdrChromaUp::Bilinear, 0, 0, cb_left, cr_left);
    pu21_reconstructed_chroma(ramp, fmt420, HdrChromaLoc::Center, HdrChromaUp::Bilinear, 1, 0, cb_mid, cr_mid);
    if (std::fabs(cb_left - 100.0) > 1e-9 || std::fabs(cb_mid - 150.0) > 1e-6) {
      std::fprintf(stderr, "FAIL pu21_chroma_siting left=%.6f mid=%.6f\n", cb_left, cb_mid);
      ++g_fail;
    } else {
      std::printf("OK   pu21_chroma_siting\n");
    }

    PlaneBuffer ssim_a = fill(16, 16, 40);
    PlaneBuffer ssim_b = checker(16, 16, 4, 40, 90);
    std::vector<double> ssim_da(static_cast<size_t>(16) * 16);
    std::vector<double> ssim_db(static_cast<size_t>(16) * 16);
    for (int i = 0; i < 16 * 16; ++i) {
      ssim_da[static_cast<size_t>(i)] = static_cast<double>(ssim_a.storage[static_cast<size_t>(i)]);
      ssim_db[static_cast<size_t>(i)] = static_cast<double>(ssim_b.storage[static_cast<size_t>(i)]);
    }
    const double ssim_int = ssim_mean_scalar(ssim_a.view(), ssim_b.view(), 1023.0, true, 1);
    const double ssim_f = ssim_mean_doubles(ssim_da.data(), 16, ssim_db.data(), 16, 16, 16, 1023.0, true);
    expect_near("pu21_ssim_kernel_matches_scalar", ssim_f, ssim_int, 1e-12);
    PlaneBuffer ms_a = fill(32, 32, 80);
    PlaneBuffer ms_b = checker(32, 32, 3, 80, 140);
    std::vector<double> ms_da(ms_a.storage.begin(), ms_a.storage.end());
    std::vector<double> ms_db(ms_b.storage.begin(), ms_b.storage.end());
    const double ms_int = msssim_plane_scalar(ms_a.view(), ms_b.view(), 1023.0, 1, nullptr);
    const double ms_f = msssim_plane_doubles(ms_da.data(), ms_db.data(), 32, 32, 1023.0, 1);
    expect_near("pu21_msssim_kernel_matches_scalar", ms_f, ms_int, 1e-12);

    Frame psnr_a;
    Frame psnr_b;
    psnr_a.allocate(fmt444);
    psnr_b.allocate(fmt444);
    paint(psnr_a, 400, 512);
    paint(psnr_b, 420, 512);
    const PsnrFrame ordinary = compute_psnr_frame(psnr_a, psnr_b, fmt444, 1023.0);
    Pu21FrameScore alongside;
    pu21_score_frame(psnr_a, psnr_b, fmt444, pq_eff, pq_eff, true, false, false, 1, alongside);
    const PsnrFrame again = compute_psnr_frame(psnr_a, psnr_b, fmt444, 1023.0);
    if (ordinary.psnr.avg != again.psnr.avg || ordinary.mse.avg != again.mse.avg) {
      fail("pu21_does_not_change_psnr");
    } else {
      std::printf("OK   pu21_does_not_change_psnr\n");
    }
  }

  // QCS on [30, 50]. Expected values are the closed form, not a second call
  // into QcsAccum. +inf contributes 100%, -inf contributes 0%, NaN is omitted.
  {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto check = [&](const char* name, std::initializer_list<double> qs, bool want, double exp) {
      QcsAccum acc{30.0, 50.0, 0.0, 0};
      for (double q : qs) acc.add(q);
      if (!want) {
        if (acc.available()) {
          std::fprintf(stderr, "FAIL %s: expected unavailable\n", name);
          ++g_fail;
        } else {
          std::printf("OK   %s: unavailable\n", name);
        }
        return;
      }
      expect_near(name, acc.percent(), exp, 1e-12);
    };
    check("qcs_finite_25", {25.0}, true, 0.0);
    check("qcs_finite_30", {30.0}, true, 0.0);
    check("qcs_finite_35", {35.0}, true, 25.0);
    check("qcs_finite_40", {40.0}, true, 50.0);
    check("qcs_finite_45", {45.0}, true, 75.0);
    check("qcs_finite_50", {50.0}, true, 100.0);
    check("qcs_finite_55", {55.0}, true, 100.0);
    check("qcs_finite_70", {70.0}, true, 100.0);
    check("qcs_A_35_posinf", {35.0, inf}, true, 62.5);
    check("qcs_B_all_posinf", {inf, inf, inf}, true, 100.0);
    check("qcs_C_35_55", {35.0, 55.0}, true, 62.5);
    check("qcs_D_mixed_posinf", {25.0, 40.0, 70.0, inf}, true, 62.5);
    check("qcs_E_40_neginf", {40.0, -inf}, true, 25.0);
    check("qcs_F_perm_D", {inf, 70.0, 25.0, 40.0}, true, 62.5);
    check("qcs_G_all_nan", {nan, nan}, false, 0.0);
    check("qcs_nan_mixed", {40.0, nan}, true, 50.0);
  }

  // VMAF default band [70, 100]. A score above 100 is kept and contributes
  // 100% at this upper bound. The same closed form, not a second formula.
  {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto check = [&](const char* name, std::initializer_list<double> qs, bool want, double exp) {
      QcsAccum acc{70.0, 100.0, 0.0, 0};
      for (double q : qs) acc.add(q);
      if (!want) {
        if (acc.available()) {
          std::fprintf(stderr, "FAIL %s: expected unavailable\n", name);
          ++g_fail;
        } else {
          std::printf("OK   %s: unavailable\n", name);
        }
        return;
      }
      expect_near(name, acc.percent(), exp, 1e-12);
    };
    check("qcs_vmaf_60", {60.0}, true, 0.0);
    check("qcs_vmaf_70", {70.0}, true, 0.0);
    check("qcs_vmaf_85", {85.0}, true, 50.0);
    check("qcs_vmaf_100", {100.0}, true, 100.0);
    check("qcs_vmaf_105", {105.0}, true, 100.0);
    check("qcs_vmaf_110", {110.0}, true, 100.0);
    check("qcs_vmaf_above_posinf", {110.0, inf}, true, 100.0);
    check("qcs_vmaf_mixed_above", {60.0, 85.0, 110.0}, true, 50.0);
    check("qcs_vmaf_nan_above", {110.0, nan}, true, 100.0);
    // Upper bound past the score: 110 is not clipped to 100 before the formula.
    QcsAccum wide{70.0, 120.0, 0.0, 0};
    wide.add(110.0);
    expect_near("qcs_vmaf_110_on_70_120", wide.percent(), 80.0, 1e-12);
  }

  if (g_fail) {
    std::fprintf(stderr, "%d selftests failed\n", g_fail);
    return 1;
  }
  std::printf("All selftests passed\n");
  return 0;
}
