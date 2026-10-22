#include "pixvqa/hdr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace pixvqa {
namespace {

struct Coef {
  double kr;
  double kg;
  double kb;
};

bool coef_for(HdrMatrix matrix, Coef& c) {
  if (matrix == HdrMatrix::Bt709) {
    c = {0.2126, 0.7152, 0.0722};
    return true;
  }
  if (matrix == HdrMatrix::Bt2020Ncl) {
    c = {0.2627, 0.6780, 0.0593};
    return true;
  }
  return false;
}

// SMPTE ST 2084 EOTF. E is the nonlinear color component in [0, 1].
double pq_eotf(double e) {
  if (e <= 0.0) return 0.0;
  if (e > 1.0) e = 1.0;
  constexpr double m1 = 2610.0 / 16384.0;
  constexpr double m2 = 2523.0 / 32.0;
  constexpr double c1 = 3424.0 / 4096.0;
  constexpr double c2 = 2413.0 / 128.0;
  constexpr double c3 = 2392.0 / 128.0;
  const double np = std::pow(e, 1.0 / m2);
  const double num = std::max(np - c1, 0.0);
  const double den = c2 - c3 * np;
  if (!(den > 0.0)) return 0.0;
  return 10000.0 * std::pow(num / den, 1.0 / m1);
}

// BT.2100 inverse HLG OETF. a is the published constant. b = 1-4a and
// c = 1 - a ln(12-b) are the exact relations; they round to 0.28466892 and
// 0.55991073. c is taken from the peak so a signal of 1 maps to scene value 1.
double hlg_inverse_oetf(double e) {
  if (e <= 0.0) return 0.0;
  if (e > 1.0) e = 1.0;
  constexpr double a = 0.17883277;
  const double b = 1.0 - 4.0 * a;
  const double c = 1.0 - a * std::log(12.0 - b);
  if (e <= 0.5) return (e * e) / 3.0;
  return (std::exp((e - c) / a) + b) / 12.0;
}

void ycbcr_to_rgb(const Coef& c, double y, double cb, double cr, double& r, double& g, double& b) {
  r = y + 2.0 * (1.0 - c.kr) * cr;
  b = y + 2.0 * (1.0 - c.kb) * cb;
  g = (y - c.kr * r - c.kb * b) / c.kg;
}

bool is_hdr(HdrTransfer t) { return t == HdrTransfer::Pq || t == HdrTransfer::Hlg; }

bool supported_color(HdrPrimaries primaries, HdrMatrix matrix) {
  return (primaries == HdrPrimaries::Bt2020 && matrix == HdrMatrix::Bt2020Ncl) ||
         (primaries == HdrPrimaries::Bt709 && matrix == HdrMatrix::Bt709);
}

template <typename T>
void observe_field(T& current, HdrSource& source, bool& conflict, T incoming) {
  if (incoming == T::Unspecified) return;
  if (current == T::Unspecified) {
    current = incoming;
    source = HdrSource::Stream;
    return;
  }
  if (current != incoming) conflict = true;
}

std::string peak_text(double peak) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", peak);
  return buf;
}

}  // namespace

const char* hdr_transfer_name(HdrTransfer v) {
  switch (v) {
    case HdrTransfer::Unspecified:
      return "unspecified";
    case HdrTransfer::Sdr:
      return "sdr";
    case HdrTransfer::Bt709:
      return "bt709";
    case HdrTransfer::Bt2020_10:
      return "bt2020-10";
    case HdrTransfer::Smpte170m:
      return "smpte170m";
    case HdrTransfer::Linear:
      return "linear";
    case HdrTransfer::Pq:
      return "pq";
    case HdrTransfer::Hlg:
      return "hlg";
    case HdrTransfer::Other:
      return "other";
  }
  return "other";
}

const char* hdr_primaries_name(HdrPrimaries v) {
  switch (v) {
    case HdrPrimaries::Unspecified:
      return "unspecified";
    case HdrPrimaries::Bt709:
      return "bt709";
    case HdrPrimaries::Bt2020:
      return "bt2020";
    case HdrPrimaries::Bt601:
      return "bt601";
    case HdrPrimaries::Other:
      return "other";
  }
  return "other";
}

const char* hdr_matrix_name(HdrMatrix v) {
  switch (v) {
    case HdrMatrix::Unspecified:
      return "unspecified";
    case HdrMatrix::Bt709:
      return "bt709";
    case HdrMatrix::Bt2020Ncl:
      return "bt2020nc";
    case HdrMatrix::Bt601:
      return "bt601";
    case HdrMatrix::Other:
      return "other";
  }
  return "other";
}

const char* hdr_range_name(HdrRange v) {
  switch (v) {
    case HdrRange::Unspecified:
      return "unspecified";
    case HdrRange::Limited:
      return "limited";
    case HdrRange::Full:
      return "full";
  }
  return "other";
}

const char* hdr_source_name(HdrSource v) {
  switch (v) {
    case HdrSource::Absent:
      return "absent";
    case HdrSource::Stream:
      return "stream";
    case HdrSource::Override:
      return "override";
    case HdrSource::Assumption:
      return "assumption";
  }
  return "absent";
}

const char* hdr_chroma_loc_name(HdrChromaLoc v) {
  switch (v) {
    case HdrChromaLoc::Left:
      return "left";
    case HdrChromaLoc::Center:
      return "center";
    case HdrChromaLoc::TopLeft:
      return "topleft";
    case HdrChromaLoc::Unspecified:
      return "unspecified";
  }
  return "unspecified";
}

const char* hdr_chroma_up_name(HdrChromaUp v) {
  return v == HdrChromaUp::Nearest ? "nearest" : "bilinear";
}

bool hdr_parse_chroma_loc(const std::string& token, HdrChromaLoc& out) {
  if (token == "left") {
    out = HdrChromaLoc::Left;
    return true;
  }
  if (token == "center") {
    out = HdrChromaLoc::Center;
    return true;
  }
  if (token == "topleft") {
    out = HdrChromaLoc::TopLeft;
    return true;
  }
  return false;
}

bool hdr_parse_chroma_up(const std::string& token, HdrChromaUp& out) {
  if (token == "bilinear") {
    out = HdrChromaUp::Bilinear;
    return true;
  }
  if (token == "nearest") {
    out = HdrChromaUp::Nearest;
    return true;
  }
  return false;
}

bool hdr_parse_primaries_token(const std::string& token, HdrPrimaries& out) {
  if (token == "bt2020") {
    out = HdrPrimaries::Bt2020;
    return true;
  }
  if (token == "bt709") {
    out = HdrPrimaries::Bt709;
    return true;
  }
  return false;
}

bool hdr_parse_matrix_token(const std::string& token, HdrMatrix& out) {
  if (token == "bt2020nc") {
    out = HdrMatrix::Bt2020Ncl;
    return true;
  }
  if (token == "bt709") {
    out = HdrMatrix::Bt709;
    return true;
  }
  return false;
}

bool hdr_parse_range_token(const std::string& token, HdrRange& out) {
  if (token == "limited") {
    out = HdrRange::Limited;
    return true;
  }
  if (token == "full") {
    out = HdrRange::Full;
    return true;
  }
  return false;
}

bool hdr_parse_transfer(const std::string& token, HdrTransfer& out) {
  if (token == "sdr") {
    out = HdrTransfer::Sdr;
    return true;
  }
  if (token == "linear") {
    out = HdrTransfer::Linear;
    return true;
  }
  if (token == "st2084" || token == "pq") {
    out = HdrTransfer::Pq;
    return true;
  }
  if (token == "stdb67" || token == "hlg") {
    out = HdrTransfer::Hlg;
    return true;
  }
  return false;
}

bool hdr_parse_colorspace(const std::string& token, HdrPrimaries& primaries, HdrMatrix& matrix,
                          HdrRange& range) {
  if (token == "bt709l") {
    primaries = HdrPrimaries::Bt709;
    matrix = HdrMatrix::Bt709;
    range = HdrRange::Limited;
    return true;
  }
  if (token == "bt709f") {
    primaries = HdrPrimaries::Bt709;
    matrix = HdrMatrix::Bt709;
    range = HdrRange::Full;
    return true;
  }
  if (token == "bt2020l") {
    primaries = HdrPrimaries::Bt2020;
    matrix = HdrMatrix::Bt2020Ncl;
    range = HdrRange::Limited;
    return true;
  }
  if (token == "bt2020f") {
    primaries = HdrPrimaries::Bt2020;
    matrix = HdrMatrix::Bt2020Ncl;
    range = HdrRange::Full;
    return true;
  }
  if (token == "bt601l") {
    primaries = HdrPrimaries::Bt601;
    matrix = HdrMatrix::Bt601;
    range = HdrRange::Limited;
    return true;
  }
  if (token == "bt601f") {
    primaries = HdrPrimaries::Bt601;
    matrix = HdrMatrix::Bt601;
    range = HdrRange::Full;
    return true;
  }
  return false;
}

void hdr_observe(HdrSide& side, HdrTransfer transfer, HdrPrimaries primaries, HdrMatrix matrix,
                 HdrRange range) {
  observe_field(side.transfer, side.transfer_source, side.transfer_conflict, transfer);
  observe_field(side.primaries, side.primaries_source, side.primaries_conflict, primaries);
  observe_field(side.matrix, side.matrix_source, side.matrix_conflict, matrix);
  observe_field(side.range, side.range_source, side.range_conflict, range);
}

void hdr_observe_chroma(HdrSide& side, HdrChromaLoc loc, bool unsupported) {
  if (unsupported) {
    side.chroma_loc_unsupported = true;
    return;
  }
  if (loc == HdrChromaLoc::Unspecified) return;
  if (side.chroma_loc_source == HdrSource::Absent) {
    side.chroma_loc = loc;
    side.chroma_loc_source = HdrSource::Stream;
    return;
  }
  if (side.chroma_loc != loc) side.chroma_loc_conflict = true;
}

std::string hdr_apply_override(HdrSide& side, bool transfer_set, const std::string& transfer,
                               bool color_set, const std::string& color_space) {
  if (transfer_set) {
    HdrTransfer parsed;
    if (!hdr_parse_transfer(transfer, parsed)) return "unsupported transfer " + transfer;
    side.transfer = parsed;
    side.transfer_source = HdrSource::Override;
    side.transfer_conflict = false;
  }
  if (color_set) {
    HdrPrimaries primaries;
    HdrMatrix matrix;
    HdrRange range;
    if (!hdr_parse_colorspace(color_space, primaries, matrix, range)) {
      return "unsupported color space " + color_space;
    }
    side.primaries = primaries;
    side.matrix = matrix;
    side.range = range;
    side.primaries_source = HdrSource::Override;
    side.matrix_source = HdrSource::Override;
    side.range_source = HdrSource::Override;
    side.primaries_conflict = false;
    side.matrix_conflict = false;
    side.range_conflict = false;
  }
  return {};
}

std::string hdr_reject_reason(const HdrSide& reference, const HdrSide& distorted, bool hlg_peak_set,
                              double hlg_peak_nits) {
  auto conflict = [](const char* role, const HdrSide& side) -> std::string {
    std::string fields;
    if (side.transfer_conflict) fields += " transfer";
    if (side.primaries_conflict) fields += " primaries";
    if (side.matrix_conflict) fields += " matrix";
    if (side.range_conflict) fields += " range";
    if (side.chroma_loc_conflict) fields += " chroma-location";
    if (side.chroma_loc_unsupported) fields += " chroma-location-unsupported";
    if (fields.empty()) return {};
    return std::string(role) + " metadata disagrees inside the stream:" + fields;
  };
  const std::string ref_conflict = conflict("reference", reference);
  if (!ref_conflict.empty()) return ref_conflict;
  const std::string dist_conflict = conflict("distorted", distorted);
  if (!dist_conflict.empty()) return dist_conflict;

  if (hlg_peak_set) {
    if (!(hlg_peak_nits > 0.0) || hlg_peak_nits > 10000.0 || !std::isfinite(hlg_peak_nits)) {
      return "HLG display peak is outside (0, 10000] cd/m2";
    }
    if (reference.transfer != HdrTransfer::Hlg && distorted.transfer != HdrTransfer::Hlg) {
      return "--hlg-peak applies only when the effective transfer is HLG";
    }
  }
  auto unsupported = [](const char* role, const HdrSide& side) -> std::string {
    if (!is_hdr(side.transfer)) return {};
    if (side.primaries == HdrPrimaries::Unspecified || side.matrix == HdrMatrix::Unspecified) return {};
    if (supported_color(side.primaries, side.matrix)) return {};
    std::ostringstream os;
    os << role << " primaries " << hdr_primaries_name(side.primaries) << " and matrix "
       << hdr_matrix_name(side.matrix) << " are not a supported PQ/HLG color system";
    return os.str();
  };
  const std::string ref_color = unsupported("reference", reference);
  if (!ref_color.empty()) return ref_color;
  return unsupported("distorted", distorted);
}

std::string hdr_resolve(const HdrSide& side_in, const HdrKnobs& knobs, HdrEffective& out) {
  out = HdrEffective{};
  out.side = side_in;
  const bool hdr = is_hdr(out.side.transfer);
  if (knobs.require_hdr_transfer) {
    if (out.side.transfer == HdrTransfer::Unspecified) {
      return "transfer is unspecified; PQ and HLG are not inferred from bit depth, filename, or primaries";
    }
    if (!hdr) {
      std::ostringstream os;
      os << "PU21 requires transfer pq or hlg; got " << hdr_transfer_name(out.side.transfer);
      return os.str();
    }
  }
  if (hdr) {
    if (out.side.primaries == HdrPrimaries::Unspecified) {
      out.side.primaries = HdrPrimaries::Bt2020;
      out.side.primaries_source = HdrSource::Assumption;
      out.notices.push_back("primaries assumed bt2020");
    }
    if (out.side.matrix == HdrMatrix::Unspecified) {
      out.side.matrix = HdrMatrix::Bt2020Ncl;
      out.side.matrix_source = HdrSource::Assumption;
      out.notices.push_back("matrix assumed bt2020nc");
    }
    if (out.side.range == HdrRange::Unspecified) {
      out.side.range = HdrRange::Limited;
      out.side.range_source = HdrSource::Assumption;
      out.notices.push_back("range assumed limited");
    }
    if (!supported_color(out.side.primaries, out.side.matrix)) {
      std::ostringstream os;
      os << "primaries " << hdr_primaries_name(out.side.primaries) << " and matrix "
         << hdr_matrix_name(out.side.matrix) << " are not a supported PQ/HLG color system";
      return os.str();
    }
  }

  const bool peak_explicit = knobs.peak_set;
  const bool legacy_peak = knobs.legacy_hlg_peak && out.side.transfer == HdrTransfer::Hlg;
  if (peak_explicit || legacy_peak || knobs.black_set || knobs.gamma_set) {
    if (!hdr) {
      return "display peak, black, and HLG gamma apply only when the transfer is PQ or HLG";
    }
  }
  if (out.side.transfer == HdrTransfer::Pq) {
    if (knobs.black_set) return "display black applies only when the effective transfer is HLG";
    if (knobs.gamma_set) return "HLG system gamma applies only when the effective transfer is HLG";
    out.peak_nits = 10000.0;
    out.peak_source = HdrSource::Absent;
    out.pq_clip = false;
    if (peak_explicit) {
      if (!(knobs.peak_nits > 0.0) || knobs.peak_nits > 10000.0 || !std::isfinite(knobs.peak_nits)) {
        return "PQ evaluation clip is outside (0, 10000] cd/m2";
      }
      out.pq_clip = true;
      out.peak_nits = knobs.peak_nits;
      out.peak_source = HdrSource::Override;
      out.notices.push_back("PQ linear light above the evaluation peak is clipped");
    }
  } else if (out.side.transfer == HdrTransfer::Hlg) {
    double peak = 1000.0;
    out.peak_source = HdrSource::Assumption;
    if (peak_explicit) {
      peak = knobs.peak_nits;
      out.peak_source = HdrSource::Override;
    } else if (legacy_peak) {
      peak = knobs.legacy_hlg_peak_nits;
      out.peak_source = HdrSource::Override;
    } else {
      out.notices.push_back("HLG evaluation peak assumed 1000 cd/m2");
    }
    if (!(peak > 0.0) || peak > 10000.0 || !std::isfinite(peak)) {
      return "HLG display peak is outside (0, 10000] cd/m2";
    }
    out.peak_nits = peak;
    out.black_nits = 0.0;
    out.black_source = HdrSource::Assumption;
    if (knobs.black_set) {
      if (!(knobs.black_nits >= 0.0) || knobs.black_nits >= peak || !std::isfinite(knobs.black_nits)) {
        return "display black must be in [0, peak) cd/m2";
      }
      out.black_nits = knobs.black_nits;
      out.black_source = HdrSource::Override;
    }
    if (knobs.gamma_set) {
      if (!(knobs.gamma > 0.0) || knobs.gamma > 10.0 || !std::isfinite(knobs.gamma)) {
        return "HLG system gamma must be in (0, 10]";
      }
      out.hlg_gamma = knobs.gamma;
      out.gamma_source = HdrSource::Override;
      out.notices.push_back("HLG system gamma is the explicit value, not the value derived from the peak");
    } else {
      out.hlg_gamma = hdr_hlg_system_gamma(peak);
      out.gamma_source = HdrSource::Assumption;
      out.notices.push_back("HLG system gamma derived from the evaluation peak");
    }
  }

  out.chroma_loc = HdrChromaLoc::Left;
  out.chroma_loc_source = HdrSource::Assumption;
  out.chroma_up = HdrChromaUp::Bilinear;
  out.chroma_up_source = HdrSource::Assumption;
  if (knobs.chroma_loc_set) {
    out.chroma_loc = knobs.chroma_loc;
    out.chroma_loc_source = HdrSource::Override;
  } else if (out.side.chroma_loc_source == HdrSource::Stream &&
             out.side.chroma_loc != HdrChromaLoc::Unspecified) {
    out.chroma_loc = out.side.chroma_loc;
    out.chroma_loc_source = HdrSource::Stream;
  } else if (hdr) {
    out.notices.push_back("chroma location assumed left");
  }
  if (knobs.chroma_up_set) {
    out.chroma_up = knobs.chroma_up;
    out.chroma_up_source = HdrSource::Override;
  } else if (hdr) {
    out.notices.push_back("chroma reconstruction assumed bilinear");
  }
  return {};
}

std::string hdr_side_line(const char* role, const HdrSide& side) {
  std::ostringstream os;
  os << role << ": transfer=" << hdr_transfer_name(side.transfer) << " ("
     << hdr_source_name(side.transfer_source) << ") primaries=" << hdr_primaries_name(side.primaries)
     << " (" << hdr_source_name(side.primaries_source) << ") matrix=" << hdr_matrix_name(side.matrix)
     << " (" << hdr_source_name(side.matrix_source) << ") range=" << hdr_range_name(side.range)
     << " (" << hdr_source_name(side.range_source) << ")";
  return os.str();
}

double hdr_hlg_system_gamma(double peak_nits) {
  return 1.2 + 0.42 * std::log10(peak_nits / 1000.0);
}

std::string hdr_display_line(const HdrSide& side, double hlg_peak_nits) {
  std::ostringstream os;
  if (side.transfer == HdrTransfer::Pq) {
    os << "display: pq st2084 eotf peak=10000 cd/m2 primaries=" << hdr_primaries_name(side.primaries)
       << " matrix=" << hdr_matrix_name(side.matrix) << " range=" << hdr_range_name(side.range)
       << " luminance=linear-rgb";
    return os.str();
  }
  if (side.transfer == HdrTransfer::Hlg) {
    char gamma[64];
    std::snprintf(gamma, sizeof(gamma), "%.9f", hdr_hlg_system_gamma(hlg_peak_nits));
    os << "display: hlg bt2100 inverse-oetf+ootf peak=" << peak_text(hlg_peak_nits)
       << " cd/m2 gamma=" << gamma << " primaries=" << hdr_primaries_name(side.primaries)
       << " matrix=" << hdr_matrix_name(side.matrix) << " range=" << hdr_range_name(side.range)
       << " luminance=linear-rgb";
    return os.str();
  }
  return {};
}

bool hdr_display_cdm2(HdrTransfer transfer, HdrMatrix matrix, double y, double cb, double cr,
                      double hlg_peak_nits, double& nits) {
  Coef c;
  if (!coef_for(matrix, c)) return false;
  double r = 0, g = 0, b = 0;
  ycbcr_to_rgb(c, y, cb, cr, r, g, b);
  if (transfer == HdrTransfer::Pq) {
    nits = c.kr * pq_eotf(r) + c.kg * pq_eotf(g) + c.kb * pq_eotf(b);
    return std::isfinite(nits);
  }
  if (transfer == HdrTransfer::Hlg) {
    if (!(hlg_peak_nits > 0.0) || !std::isfinite(hlg_peak_nits)) return false;
    const double rs = hlg_inverse_oetf(r);
    const double gs = hlg_inverse_oetf(g);
    const double bs = hlg_inverse_oetf(b);
    const double ys = c.kr * rs + c.kg * gs + c.kb * bs;
    const double gamma = hdr_hlg_system_gamma(hlg_peak_nits);
    nits = ys <= 0.0 ? 0.0 : hlg_peak_nits * std::pow(ys, gamma);
    return std::isfinite(nits);
  }
  return false;
}

bool hdr_codes_to_nominal(int bit_depth, HdrRange range, double y, double cb, double cr, double& yn,
                          double& cb_n, double& cr_n) {
  if (bit_depth < 8 || bit_depth > 16) return false;
  if (range == HdrRange::Full) {
    const double maxv = static_cast<double>((1 << bit_depth) - 1);
    yn = y / maxv;
    cb_n = cb / maxv - 0.5;
    cr_n = cr / maxv - 0.5;
    return true;
  }
  if (range == HdrRange::Limited) {
    const double scale = static_cast<double>(1 << (bit_depth - 8));
    yn = (y - 16.0 * scale) / (219.0 * scale);
    cb_n = (cb - 128.0 * scale) / (224.0 * scale);
    cr_n = (cr - 128.0 * scale) / (224.0 * scale);
    return true;
  }
  return false;
}

bool hdr_codes_to_nominal(int bit_depth, HdrRange range, int y, int cb, int cr, double& yn, double& cb_n,
                          double& cr_n) {
  return hdr_codes_to_nominal(bit_depth, range, static_cast<double>(y), static_cast<double>(cb),
                              static_cast<double>(cr), yn, cb_n, cr_n);
}

bool hdr_display_rgb(const HdrEffective& cfg, double y, double cb, double cr, double& r_nits,
                     double& g_nits, double& b_nits) {
  Coef c;
  if (!coef_for(cfg.side.matrix, c)) return false;
  double r = 0, g = 0, b = 0;
  ycbcr_to_rgb(c, y, cb, cr, r, g, b);
  if (cfg.side.transfer == HdrTransfer::Pq) {
    r_nits = pq_eotf(r);
    g_nits = pq_eotf(g);
    b_nits = pq_eotf(b);
    if (cfg.pq_clip) {
      r_nits = std::min(r_nits, cfg.peak_nits);
      g_nits = std::min(g_nits, cfg.peak_nits);
      b_nits = std::min(b_nits, cfg.peak_nits);
    }
    return std::isfinite(r_nits) && std::isfinite(g_nits) && std::isfinite(b_nits);
  }
  if (cfg.side.transfer == HdrTransfer::Hlg) {
    if (!(cfg.peak_nits > cfg.black_nits) || !std::isfinite(cfg.hlg_gamma)) return false;
    const double rs = hlg_inverse_oetf(r);
    const double gs = hlg_inverse_oetf(g);
    const double bs = hlg_inverse_oetf(b);
    const double ys = c.kr * rs + c.kg * gs + c.kb * bs;
    if (!(ys > 0.0)) {
      r_nits = g_nits = b_nits = cfg.black_nits;
      return true;
    }
    const double yd =
        cfg.black_nits + (cfg.peak_nits - cfg.black_nits) * std::pow(ys, cfg.hlg_gamma);
    const double scale = yd / ys;
    r_nits = rs * scale;
    g_nits = gs * scale;
    b_nits = bs * scale;
    return std::isfinite(r_nits) && std::isfinite(g_nits) && std::isfinite(b_nits);
  }
  return false;
}

std::string hdr_display_text(const char* role, const HdrEffective& side) {
  std::ostringstream os;
  os << "display " << role << ": ";
  if (side.side.transfer == HdrTransfer::Pq) {
    os << "pq st2084 eotf peak=10000 cd/m2 (constant) clip=";
    if (side.pq_clip) {
      os << peak_text(side.peak_nits) << " cd/m2 (" << hdr_source_name(side.peak_source) << ")";
    } else {
      os << "none";
    }
  } else if (side.side.transfer == HdrTransfer::Hlg) {
    char gamma[64];
    std::snprintf(gamma, sizeof(gamma), "%.9f", side.hlg_gamma);
    const char* gamma_how = side.gamma_source == HdrSource::Override ? "override" : "derived";
    os << "hlg bt2100 inverse-oetf+ootf peak=" << peak_text(side.peak_nits) << " cd/m2 ("
       << hdr_source_name(side.peak_source) << ") gamma=" << gamma << " (" << gamma_how
       << ") black=" << peak_text(side.black_nits) << " cd/m2 (" << hdr_source_name(side.black_source)
       << ")";
  } else {
    return {};
  }
  os << " primaries=" << hdr_primaries_name(side.side.primaries)
     << " matrix=" << hdr_matrix_name(side.side.matrix) << " range=" << hdr_range_name(side.side.range)
     << " luminance=linear-rgb chroma-loc=" << hdr_chroma_loc_name(side.chroma_loc) << " ("
     << hdr_source_name(side.chroma_loc_source) << ") chroma-up=" << hdr_chroma_up_name(side.chroma_up)
     << " (" << hdr_source_name(side.chroma_up_source) << ")";
  return os.str();
}

}  // namespace pixvqa
