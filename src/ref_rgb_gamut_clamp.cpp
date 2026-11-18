#include "pixvqa/ref_rgb_gamut_clamp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pixvqa {
namespace {

constexpr double kRgbGamutEps = 1e-9;

bool matrix_coefs(const RefRgbGamutClampConfig& cfg, double& kr, double& kg, double& kb) {
  if (cfg.smpte240m) {
    kr = 0.212;
    kg = 0.701;
    kb = 0.087;
    return true;
  }
  if (cfg.matrix == HdrMatrix::Bt709) {
    kr = 0.2126;
    kg = 0.7152;
    kb = 0.0722;
    return true;
  }
  if (cfg.matrix == HdrMatrix::Bt2020Ncl) {
    kr = 0.2627;
    kg = 0.6780;
    kb = 0.0593;
    return true;
  }
  if (cfg.matrix == HdrMatrix::Bt601) {
    kr = 0.299;
    kg = 0.587;
    kb = 0.114;
    return true;
  }
  return false;
}

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

bool rgb_needs_clamp(double r, double g, double b) {
  return r < -kRgbGamutEps || r > 1.0 + kRgbGamutEps || g < -kRgbGamutEps || g > 1.0 + kRgbGamutEps ||
         b < -kRgbGamutEps || b > 1.0 + kRgbGamutEps;
}

template <typename Fn>
void chroma_touch_sites(const VideoFormat& fmt, HdrChromaUp up, int cw, int ch, double cx, double cy,
                        Fn&& each) {
  if (fmt.chroma == ChromaFormat::YUV444) {
    const int ix = static_cast<int>(std::round(cx));
    const int iy = static_cast<int>(std::round(cy));
    if (ix < 0 || iy < 0 || ix >= cw || iy >= ch) return;
    each(ix, iy, 1.0);
    return;
  }
  cx = std::clamp(cx, 0.0, static_cast<double>(cw - 1));
  cy = std::clamp(cy, 0.0, static_cast<double>(ch - 1));
  if (up == HdrChromaUp::Nearest) {
    const int ix = static_cast<int>(std::floor(cx + 0.5));
    const int iy = static_cast<int>(std::floor(cy + 0.5));
    each(ix, iy, 1.0);
    return;
  }
  const int x0 = static_cast<int>(std::floor(cx));
  const int y0 = static_cast<int>(std::floor(cy));
  const int x1 = std::min(x0 + 1, cw - 1);
  const int y1 = std::min(y0 + 1, ch - 1);
  const double tx = cx - static_cast<double>(x0);
  const double ty = cy - static_cast<double>(y0);
  each(x0, y0, (1.0 - tx) * (1.0 - ty));
  each(x1, y0, tx * (1.0 - ty));
  each(x0, y1, (1.0 - tx) * ty);
  each(x1, y1, tx * ty);
}

void accumulate_chroma_weights(const VideoFormat& fmt, HdrChromaUp up, int cw, int ch, double cx,
                               double cy, std::vector<double>& weight) {
  chroma_touch_sites(fmt, up, cw, ch, cx, cy, [&](int ix, int iy, double w) {
    const size_t i = static_cast<size_t>(iy) * cw + ix;
    weight[i] += w;
  });
}

void accumulate_chroma_deltas(const VideoFormat& fmt, HdrChromaUp up, int cw, int ch, double cx,
                              double cy, double dcb, double dcr, std::vector<double>& acc_cb,
                              std::vector<double>& acc_cr) {
  if (dcb == 0.0 && dcr == 0.0) return;
  chroma_touch_sites(fmt, up, cw, ch, cx, cy, [&](int ix, int iy, double w) {
    const size_t i = static_cast<size_t>(iy) * cw + ix;
    acc_cb[i] += w * dcb;
    acc_cr[i] += w * dcr;
  });
}

HdrMatrix stream_matrix_ref(const HdrSide& side) {
  if (side.matrix_conflict) return HdrMatrix::Unspecified;
  return side.matrix;
}

HdrRange stream_range_ref(const HdrSide& side) {
  if (side.range_conflict) return HdrRange::Unspecified;
  return side.range;
}

bool plane_buffer_adequate(const PlaneBuffer& plane) {
  if (plane.width < 1 || plane.height < 1) return false;
  if (plane.stride < plane.width) return false;
  const size_t need =
      static_cast<size_t>(plane.stride) * static_cast<size_t>(plane.height);
  return plane.storage.size() >= need;
}

std::string ref_gamut_clamp_validate_geometry(const VideoFormat& fmt, const Frame& frame) {
  if (fmt.width < 1 || fmt.height < 1) return "empty frame";
  if (fmt.bit_depth != 8 && fmt.bit_depth != 10) return "unsupported bit depth for gamut clamp";

  switch (fmt.chroma) {
    case ChromaFormat::YUV420:
      if ((fmt.width & 1) != 0 || (fmt.height & 1) != 0) {
        return "420p requires even width and height";
      }
      break;
    case ChromaFormat::YUV422:
      if ((fmt.width & 1) != 0) return "422p requires even width";
      break;
    case ChromaFormat::YUV444:
      break;
  }

  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  if (cw < 1 || ch < 1) return "invalid chroma plane dimensions";

  if (frame.y.width != fmt.width || frame.y.height != fmt.height) {
    return "luma plane geometry mismatch";
  }
  if (frame.u.width != cw || frame.u.height != ch) {
    return "cb plane geometry mismatch";
  }
  if (frame.v.width != cw || frame.v.height != ch) {
    return "cr plane geometry mismatch";
  }

  if (!plane_buffer_adequate(frame.y)) return "luma plane buffer too small";
  if (!plane_buffer_adequate(frame.u)) return "cb plane buffer too small";
  if (!plane_buffer_adequate(frame.v)) return "cr plane buffer too small";

  return {};
}

}  // namespace

bool ref_parse_gamut_matrix_token(const std::string& token, RefRgbGamutClampConfig& out) {
  if (token == "bt2020c" || token == "bt2020cl" || token == "ictcp") {
    return false;
  }
  if (token == "smpte240m") {
    out.smpte240m = true;
    out.matrix = HdrMatrix::Other;
    return true;
  }
  if (token == "bt601" || token == "smpte170m" || token == "bt470bg") {
    out.smpte240m = false;
    out.matrix = HdrMatrix::Bt601;
    return true;
  }
  out.smpte240m = false;
  return hdr_parse_matrix_token(token, out.matrix);
}

std::string ref_gamut_clamp_resolve(const Options& opt, bool ref_raw, const HdrSide& ref_stream,
                                    RefRgbGamutClampConfig& cfg) {
  cfg = RefRgbGamutClampConfig{};
  if (!opt.reference_rgb_gamut_clamp) return {};

  if (opt.ref_gamut_matrix_set) {
    if (opt.ref_gamut_smpte240m) {
      cfg.smpte240m = true;
    } else {
      cfg.matrix = opt.ref_gamut_matrix;
      cfg.smpte240m = false;
    }
  } else if (ref_raw) {
    return "raw reference requires --reference-matrix (bt709, bt2020nc, bt601, smpte240m, ...)";
  } else {
    const HdrMatrix stream = stream_matrix_ref(ref_stream);
    if (stream == HdrMatrix::Bt2020Cl) {
      return "reference RGB gamut clamp does not support BT.2020 constant luminance; use bt2020nc or omit the transform";
    }
    if (stream == HdrMatrix::Unspecified || stream == HdrMatrix::Other) {
      return "reference stream matrix is unknown or ambiguous; set --reference-matrix explicitly";
    }
    cfg.matrix = stream;
  }

  if (cfg.smpte240m) {
    // coefficient table only
  } else if (cfg.matrix == HdrMatrix::Bt2020Cl) {
    return "reference RGB gamut clamp does not support BT.2020 constant luminance";
  } else if (cfg.matrix == HdrMatrix::Unspecified || cfg.matrix == HdrMatrix::Other) {
    return "unsupported --reference-matrix";
  }

  if (opt.ref_gamut_range_set) {
    cfg.range = opt.ref_gamut_range;
  } else if (ref_raw) {
    return "raw reference requires --reference-range limited or full";
  } else {
    const HdrRange stream = stream_range_ref(ref_stream);
    if (stream == HdrRange::Unspecified) {
      return "reference stream range is unknown; set --reference-range explicitly";
    }
    cfg.range = stream;
  }

  if (opt.ref_hdr.chroma_loc_set) {
    if (!hdr_parse_chroma_loc(opt.ref_hdr.chroma_loc, cfg.chroma_loc)) {
      return "invalid --ref-hdr-chroma-loc";
    }
  } else if (opt.hdr.chroma_loc_set) {
    if (!hdr_parse_chroma_loc(opt.hdr.chroma_loc, cfg.chroma_loc)) {
      return "invalid --hdr-chroma-loc";
    }
  } else if (!ref_raw && ref_stream.chroma_loc != HdrChromaLoc::Unspecified &&
             ref_stream.chroma_loc_source == HdrSource::Stream && !ref_stream.chroma_loc_conflict &&
             !ref_stream.chroma_loc_unsupported) {
    cfg.chroma_loc = ref_stream.chroma_loc;
  }

  if (opt.ref_hdr.chroma_up_set) {
    if (!hdr_parse_chroma_up(opt.ref_hdr.chroma_up, cfg.chroma_up)) {
      return "invalid --ref-hdr-chroma-up";
    }
  } else if (opt.hdr.chroma_up_set) {
    if (!hdr_parse_chroma_up(opt.hdr.chroma_up, cfg.chroma_up)) {
      return "invalid --hdr-chroma-up";
    }
  }

  double kr = 0, kg = 0, kb = 0;
  if (!matrix_coefs(cfg, kr, kg, kb)) return "unsupported reference matrix coefficients";
  (void)kr;
  (void)kg;
  (void)kb;
  return {};
}

std::string ref_gamut_clamp_summary(const RefRgbGamutClampConfig& cfg) {
  std::string matrix;
  if (cfg.smpte240m) {
    matrix = "smpte240m";
  } else {
    matrix = hdr_matrix_name(cfg.matrix);
  }
  return std::string("reference RGB gamut clamp: matrix=") + matrix + " range=" +
         hdr_range_name(cfg.range) + " chroma-loc=" + hdr_chroma_loc_name(cfg.chroma_loc) +
         " chroma-up=" + hdr_chroma_up_name(cfg.chroma_up) +
         " (delta-only; nonlinear code domain; see docs/REFERENCE_RGB_GAMUT_CLAMP.md)";
}

std::string ref_gamut_clamp_frame(Frame& frame, const VideoFormat& fmt,
                                  const RefRgbGamutClampConfig& cfg) {
  if (const std::string geom = ref_gamut_clamp_validate_geometry(fmt, frame); !geom.empty()) {
    return geom;
  }
  double kr = 0, kg = 0, kb = 0;
  if (!matrix_coefs(cfg, kr, kg, kb)) return "unsupported reference matrix";

  const size_t n = static_cast<size_t>(fmt.width) * fmt.height;
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const size_t cn = static_cast<size_t>(cw) * ch;

  std::vector<double> delta_y;
  std::vector<double> acc_cb;
  std::vector<double> acc_cr;
  bool any_clamp = false;

  for (int y = 0; y < fmt.height; ++y) {
    const uint16_t* yrow = frame.y.ptr() + static_cast<size_t>(y) * frame.y.stride;
    for (int x = 0; x < fmt.width; ++x) {
      double cx = 0, cy = 0;
      chroma_coord(cfg.chroma_loc, fmt.chroma, x, y, cx, cy);
      const double cb_code = reconstruct_code(frame.u, cfg.chroma_up, cx, cy);
      const double cr_code = reconstruct_code(frame.v, cfg.chroma_up, cx, cy);
      double yn0 = 0, cbn0 = 0, crn0 = 0;
      if (!hdr_codes_to_nominal(fmt.bit_depth, cfg.range, static_cast<double>(yrow[x]), cb_code,
                                cr_code, yn0, cbn0, crn0)) {
        return "unsupported bit depth or range";
      }

      double rp = 0, gp = 0, bp = 0;
      hdr_nominal_ycbcr_to_rgb(kr, kg, kb, yn0, cbn0, crn0, rp, gp, bp);
      if (!rgb_needs_clamp(rp, gp, bp)) continue;

      if (!any_clamp) {
        any_clamp = true;
        delta_y.assign(n, 0.0);
        acc_cb.assign(cn, 0.0);
        acc_cr.assign(cn, 0.0);
      }

      rp = std::clamp(rp, 0.0, 1.0);
      gp = std::clamp(gp, 0.0, 1.0);
      bp = std::clamp(bp, 0.0, 1.0);
      double yn1 = 0, cbn1 = 0, crn1 = 0;
      hdr_nominal_rgb_to_ycbcr(kr, kg, kb, rp, gp, bp, yn1, cbn1, crn1);

      const double dy = yn1 - yn0;
      const double dcb = cbn1 - cbn0;
      const double dcr = crn1 - crn0;

      const size_t i = static_cast<size_t>(y) * fmt.width + x;
      delta_y[i] = dy;
      accumulate_chroma_deltas(fmt, cfg.chroma_up, cw, ch, cx, cy, dcb, dcr, acc_cb, acc_cr);
    }
  }

  if (!any_clamp) return {};

  std::vector<double> chroma_weight(cn, 0.0);
  for (int y = 0; y < fmt.height; ++y) {
    for (int x = 0; x < fmt.width; ++x) {
      double cx = 0, cy = 0;
      chroma_coord(cfg.chroma_loc, fmt.chroma, x, y, cx, cy);
      accumulate_chroma_weights(fmt, cfg.chroma_up, cw, ch, cx, cy, chroma_weight);
    }
  }

  for (int y = 0; y < fmt.height; ++y) {
    uint16_t* yrow = frame.y.ptr() + static_cast<size_t>(y) * frame.y.stride;
    for (int x = 0; x < fmt.width; ++x) {
      const size_t i = static_cast<size_t>(y) * fmt.width + x;
      if (delta_y[i] == 0.0) continue;
      double yn = 0, cbn = 0, crn = 0;
      if (!hdr_codes_to_nominal(fmt.bit_depth, cfg.range, static_cast<double>(yrow[x]), 0.0, 0.0,
                                yn, cbn, crn)) {
        return "y nominal failed";
      }
      int yc = 0, cb = 0, cr = 0;
      if (!hdr_nominal_to_codes(fmt.bit_depth, cfg.range, yn + delta_y[i], cbn, crn, yc, cb, cr)) {
        return "y quantize failed";
      }
      yrow[x] = static_cast<uint16_t>(yc);
    }
  }

  for (int cj = 0; cj < ch; ++cj) {
    uint16_t* urow = frame.u.ptr() + static_cast<size_t>(cj) * frame.u.stride;
    uint16_t* vrow = frame.v.ptr() + static_cast<size_t>(cj) * frame.v.stride;
    for (int ci = 0; ci < cw; ++ci) {
      const size_t i = static_cast<size_t>(cj) * cw + ci;
      if (chroma_weight[i] <= 0.0) continue;
      const double dcb = acc_cb[i] / chroma_weight[i];
      const double dcr = acc_cr[i] / chroma_weight[i];
      double yn = 0, cbn0 = 0, crn0 = 0;
      if (!hdr_codes_to_nominal(fmt.bit_depth, cfg.range, 0.0, static_cast<double>(urow[ci]),
                                static_cast<double>(vrow[ci]), yn, cbn0, crn0)) {
        return "chroma nominal failed";
      }
      int yd = 0, ucode = 0, vcode = 0;
      if (!hdr_nominal_to_codes(fmt.bit_depth, cfg.range, yn, cbn0 + dcb, crn0 + dcr, yd, ucode,
                                vcode)) {
        return "chroma quantize failed";
      }
      urow[ci] = static_cast<uint16_t>(ucode);
      vrow[ci] = static_cast<uint16_t>(vcode);
    }
  }

  return {};
}

}  // namespace pixvqa