#include "pixvqa/cpu_dispatch.hpp"
#include "pixvqa/hdr.hpp"
#include "pixvqa/psnr.hpp"
#include "pixvqa/pu21.hpp"
#include "pixvqa/qcs.hpp"
#include "pixvqa/msssim.hpp"
#include "pixvqa/ssim.hpp"
#include "pixvqa/types.hpp"
#include "pixvqa/version.hpp"
#include "pixvqa/xpsnr.hpp"
#include "pixvqa/yuv_io.hpp"
#ifdef PIXVQA_HAS_VMAF
#include "pixvqa/vmaf.hpp"
#endif
#ifdef PIXVQA_HAS_CUDA
#include "pixvqa/cuda_backend.hpp"
#endif
#ifdef PIXVQA_HAS_FFMPEG
#include "pixvqa/decode.hpp"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <cctype>
#include <limits>
#include <memory>
#include <poll.h>
#include <string>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace pixvqa;

namespace {

bool pu21_selected(const Options& opt);

struct SideSpec {
  std::string path;
  VideoFormat format;
  bool has_size = false;
  bool has_fmt = false;
  bool compressed = false;
  FrameRateKind rate = FrameRateKind::Missing;
  int fps_num = 0;
  int fps_den = 1;
  int nominal_num = 0;
  int nominal_den = 1;
  InputKind input = InputKind::Auto;
};

struct CvvdpOptions {
  bool selected = false;
  bool options_seen = false;
  std::string interpretation;
  std::string display;
  std::string python;
  std::string cvvdp_bin;
  double gpu_mem_gb = 0.0;
  bool gpu_mem_set = false;
  std::string output_json;
  bool output_json_set = false;
};

struct CvqaHdrNrOptions {
  bool selected = false;
  bool options_seen = false;
  // First raw-YUV option (-s, -f, --fps); NR reads compressed video only.
  std::string raw_option;
  std::string python;
  std::string repository;
  std::string checkpoint;
  std::string profile;
  std::string output_json;
  bool output_json_set = false;
};

struct TemporaryCvvdpDirectory {
  std::string path;

  ~TemporaryCvvdpDirectory() {
    if (path.empty()) return;
    const std::string json = path + "/result.json";
    const std::string csv = path + "/result.csv";
    unlink(json.c_str());
    unlink(csv.c_str());
    rmdir(path.c_str());
  }
};

struct TemporaryCvqaHdrDirectory {
  std::string path;

  ~TemporaryCvqaHdrDirectory() {
    if (path.empty()) return;
    const std::string json = path + "/result.json";
    unlink(json.c_str());
    rmdir(path.c_str());
  }
};

const char* chroma_token(ChromaFormat c) {
  switch (c) {
    case ChromaFormat::YUV420:
      return "yuv420";
    case ChromaFormat::YUV422:
      return "yuv422";
    case ChromaFormat::YUV444:
      return "yuv444";
  }
  return "yuv";
}

// Historical display token for the PQ transfer. The stored name stays st2084.
const char* transfer_display(const std::string& tf) {
  if (tf == "st2084") return "st2048";
  return tf.c_str();
}

bool known_cs(const std::string& v) {
  static const char* k[] = {"bt601l", "bt601f", "bt709l", "bt709f", "bt2020l", "bt2020f"};
  for (const char* s : k) {
    if (v == s) return true;
  }
  return false;
}

bool known_tf(const std::string& v) {
  static const char* k[] = {"sdr", "linear", "st2084", "stdb67"};
  for (const char* s : k) {
    if (v == s) return true;
  }
  return false;
}

void print_banner() {
  std::printf("\nPixop VQA\n");
  std::printf("Version %s %s for Linux 64-bit \n", PIXVQA_PRODUCT_NAME, PIXVQA_VERSION);
  std::printf("\n\n");
}

void print_version_and_exit(const char* /*argv0*/) {
  std::printf("%s version %s\n", PIXVQA_PRODUCT_NAME, PIXVQA_VERSION);
  std::printf("build: %s\n", PIXVQA_BUILD_KIND);
#ifdef PIXVQA_HAS_NVDEC
  std::printf("decode: software nvdec\n");
#elif defined(PIXVQA_HAS_FFMPEG)
  std::printf("decode: software\n");
#else
  std::printf("decode: none\n");
#endif
  std::exit(0);
}

void usage(const char* argv0) {
  std::printf("Usage: %s [options] -r REF -t TEST\n", argv0);
  std::fputs(
      "\n"
      "Compare a reference with a distorted sequence. --metrics is required.\n"
      "--help and --version do not read inputs. See docs/QCS.md and docs/PU21.md.\n"
      "\n"
      "Input\n"
      "  -r, -t PATH         Reference and distorted. -t always; -r unless only cvqa-hdr-nr.\n"
      "  -s WxH              Luma size. Required for raw YUV. Both inputs must match.\n"
      "  -f 420p|422p|444p[:8|:10]\n"
      "                      Required for raw YUV. Omitted bits default to 10.\n"
      "  --input auto|raw|compressed\n"
      "                      Default auto. raw does not probe. compressed needs a stream.\n"
      "  --decode auto|cpu|gpu\n"
      "                      Default auto. gpu is NVDEC and is not replaced by software.\n"
      "  --fps N|N/D         Positive integer or rational. xpsnr needs it without one\n"
      "                      constant compressed rate.\n"
      "  -n, --frames N      Stop after N frames. N is a positive integer.\n"
      "  -v 0|1              Default 1 (each frame). 0 prints the summary only.\n"
      "\n"
      "Metrics\n"
      "  --metrics LIST      psnr, xpsnr, ssim, msssim, vmaf, pu21-psnr, pu21-ssim,\n"
      "                      pu21-msssim, cvvdp, cvqa-hdr-nr, or all. Groups pq, hlg, sdr\n"
      "                      select that transfer's metrics (docs/CLI_USAGE.md).\n"
      "  --preset hdr        HDR10/PQ master run: the same metrics as --metrics pq.\n"
      "  --model PATH        VMAF model JSON. Required for vmaf unless a group sets it.\n"
      "  --vmaf-model PATH   Alias for --model. The two must agree.\n"
      "\n"
      "Execution\n"
      "  --metric-device, --device auto|cpu|gpu\n"
      "                      Default auto. gpu has no CPU substitute. VMAF stays on CPU.\n"
      "  --threads auto|N    Default auto. N is a positive integer for metric workers.\n"
      "  --cvvdp-python, --cvvdp-bin, --cvvdp-output-json PATH; --cvvdp-gpu-mem GB\n"
      "                      CVVDP Python (or PIXVQA_CVVDP_PYTHON), cvvdp, JSON, CUDA cap (GB).\n"
      "  --cvqahdr-python PATH  NR interpreter (or PIXVQA_CVQAHDR_PYTHON).\n"
      "  --cvqahdr-repo, --cvqahdr-checkpoint, --cvqahdr-profile PATH\n"
      "                      Upstream checkout, NR checkpoint (.pth), calibration (.npy).\n"
      "  --cvqahdr-output-json PATH  Persistent NR JSON run record.\n"
      "  --learned-config PATH  TOML defaults for both (config/learned_metrics.local.toml).\n"
      "\n"
      "QCS\n"
      "  --qcs               One value per selected metric. See docs/QCS.md.\n"
      "  --qcs-range LIST    metric=lower:upper. Repeatable. Requires --qcs, lower < upper.\n"
      "                      Defaults: psnr, xpsnr, pu21-psnr [30,50]; ssim, msssim,\n"
      "                      pu21-ssim, pu21-msssim [0.8,1.0]; vmaf [70,100]; cvvdp\n"
      "                      and cvqa-hdr-nr [0,10] (per-frame and sampled-frame).\n"
      "\n"
      "HDR\n"
      "  --hdr-transfer auto|pq|hlg|sdr|linear\n"
      "  --hdr-primaries, --hdr-matrix, --hdr-range, --hdr-chroma-loc,\n"
      "  --hdr-chroma-up, --hdr-peak-nits, --hdr-black-nits, --hdr-hlg-gamma\n"
      "                      Values and defaults: --help-hdr. Setup: docs/PU21.md.\n"
      "  --cs, --tf          Legacy color and transfer overrides. See --help-hdr.\n"
      "\n"
      "General\n"
      "  -h, --help          This text. Exit 0.\n"
      "  --help-hdr          HDR options, values, and defaults. Exit 0.\n"
      "  --version           Name, version, build, and decode support. Exit 0.\n"
      "  Run order: native, cvvdp, cvqa-hdr-nr. Exit 0 ok, 2 setup/preflight, 1 runtime.\n"
      "  Unknown options are an error. --wsssim, --wspsnr, --rfilter, --tfilter,\n"
      "  --geom, and nonzero --seek are unsupported.\n",
      stdout);
}

void print_help_hdr() {
  std::fputs(
      "HDR parameters\n"
      "\n"
      "PU21 metrics: pu21-psnr, pu21-ssim, pu21-msssim. See docs/PU21.md.\n"
      "Order for every field: per-input option, shared option, stream metadata,\n"
      "then the documented default.\n"
      "\n"
      "When the transfer is pq or hlg and a field is missing:\n"
      "  primaries bt2020    matrix bt2020nc    range limited\n"
      "  chroma-loc left     chroma-up bilinear\n"
      "  HLG peak 1000 cd/m2    HLG black 0 cd/m2    gamma 1.2 at that peak\n"
      "PQ stays absolute until --hdr-peak-nits sets an evaluation clip.\n"
      "\n"
      "  --hdr-transfer auto|pq|hlg|sdr|linear\n"
      "  --hdr-primaries auto|bt2020|bt709\n"
      "  --hdr-matrix auto|bt2020nc|bt709\n"
      "  --hdr-range auto|limited|full\n"
      "  --hdr-chroma-loc auto|left|center|topleft\n"
      "  --hdr-chroma-up bilinear|nearest\n"
      "  --hdr-peak-nits auto|N        (0, 10000] cd/m2\n"
      "  --hdr-black-nits auto|N       HLG only, [0, peak)\n"
      "  --hdr-hlg-gamma auto|N        HLG only, (0, 10]\n"
      "Prefix ref- or dist- to set one input, for example --ref-hdr-transfer pq.\n"
      "auto leaves the field unset so metadata or the default applies.\n"
      "\n"
      "Legacy, still accepted:\n"
      "  --tf sdr|linear|st2084|stdb67     st2084 is pq, stdb67 is hlg\n"
      "  --cs bt709l|bt709f|bt2020l|bt2020f|bt601l|bt601f\n"
      "  --hlg-peak N    --ref-tf --dist-tf --ref-cs --dist-cs\n"
      "bt601 is rejected for PQ and HLG. Same-scope pairs are an error:\n"
      "--tf with --hdr-transfer, including ref- and dist-, and --hlg-peak with --hdr-peak-nits.\n",
      stdout);
}

std::string fmt_psnr(double v) {
  if (!std::isfinite(v)) return "999.99";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f", v);
  return buf;
}

// Short MSE print: 0.00 for zero, otherwise one decimal.
std::string fmt_mse(double v) {
  char buf[32];
  if (v == 0.0) {
    std::snprintf(buf, sizeof(buf), "0.00");
  } else {
    std::snprintf(buf, sizeof(buf), "%.1f", v);
  }
  return buf;
}

void print_psnr(const PlaneScore& psnr, const PlaneScore& mse) {
  std::printf("[ PSNR  y:%s u:%s v:%s avg:%s ] [ MSE  y:%s u:%s v:%s avg:%s ] ",
              fmt_psnr(psnr.y).c_str(), fmt_psnr(psnr.u).c_str(), fmt_psnr(psnr.v).c_str(),
              fmt_psnr(psnr.avg).c_str(), fmt_mse(mse.y).c_str(), fmt_mse(mse.u).c_str(),
              fmt_mse(mse.v).c_str(), fmt_mse(mse.avg).c_str());
}

void print_pu21_value(const char* name, double value) {
  if (!std::isfinite(value)) {
    std::printf("[ %s:999.99 ] ", name);
    return;
  }
  std::printf("[ %s:%.6f ] ", name, value);
}

void print_pu21_scores(const Options& opt, const Pu21FrameScore& score) {
  if (opt.enable_pu21_psnr) print_pu21_value("PU21-PSNR", score.psnr);
  if (opt.enable_pu21_ssim) print_pu21_value("PU21-SSIM", score.ssim);
  if (opt.enable_pu21_msssim) print_pu21_value("PU21-MS-SSIM", score.msssim);
}

void print_metric(const char* name, const PlaneScore& s) {
  std::printf("[ %s  y:%.6f u:%.6f v:%.6f avg:%.6f ] ", name, s.y, s.u, s.v, s.avg);
}

void print_msssim(const PlaneScore& s) { print_metric("MS-SSIM", s); }

PlaneScore score_msssim(const Frame& ref, const Frame& test, const VideoFormat& fmt, double peak,
                      int threads) {
  const char* dbg = pix_getenv("MSSSIM_DEBUG");
  if (!(dbg && dbg[0] == '1')) return compute_msssim_frame(ref, test, fmt, peak, threads);
  PlaneScore out;
  const PlaneView refs[3] = {ref.y.view(), ref.u.view(), ref.v.view()};
  const PlaneView tests[3] = {test.y.view(), test.u.view(), test.v.view()};
  const char* name[3] = {"y", "u", "v"};
  double* slot[3] = {&out.y, &out.u, &out.v};
  for (int p = 0; p < 3; ++p) {
    MsssimPlaneInfo info;
    *slot[p] = msssim_plane(refs[p], tests[p], peak, threads, &info);
    for (int s = 0; s < info.scales; ++s) {
      std::fprintf(stderr, "msssim_comp %s %d %d %d %.12g %.12g\n", name[p], s, info.scale[s].height,
                   info.scale[s].width, info.scale[s].cs, info.scale[s].ssim);
    }
  }
  out.avg = yuv_sample_weight_avg(out.y, out.u, out.v, fmt.chroma);
  return out;
}

void trim_ascii_space(std::string& s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
}

constexpr unsigned kQcsCvqaHdrNrBit = 1u << 8;
constexpr unsigned kQcsCvvdpBit = 1u << 9;
constexpr const char* kHdrVmafModel = "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json";
constexpr const char* kSdrVmafModel = "model/vmaf_v1.0.16_3d0h.json";

bool assign_qcs_range(Options& opt, const std::string& spec) {
  const auto eq = spec.find('=');
  const auto colon = spec.find(':');
  if (eq == std::string::npos || colon == std::string::npos || colon < eq + 2 ||
      eq == 0 || colon + 1 >= spec.size()) {
    std::fprintf(stderr, "pixvqa: invalid --qcs-range %s (use metric=lower:upper)\n", spec.c_str());
    return false;
  }
  const std::string metric = spec.substr(0, eq);
  const std::string lo_s = spec.substr(eq + 1, colon - eq - 1);
  const std::string hi_s = spec.substr(colon + 1);
  char* end = nullptr;
  const double lo = std::strtod(lo_s.c_str(), &end);
  if (end == lo_s.c_str() || *end != '\0' || !std::isfinite(lo)) {
    std::fprintf(stderr, "pixvqa: invalid --qcs-range lower bound in %s\n", spec.c_str());
    return false;
  }
  const double hi = std::strtod(hi_s.c_str(), &end);
  if (end == hi_s.c_str() || *end != '\0' || !std::isfinite(hi)) {
    std::fprintf(stderr, "pixvqa: invalid --qcs-range upper bound in %s\n", spec.c_str());
    return false;
  }
  if (!(hi > lo)) {
    std::fprintf(stderr, "pixvqa: --qcs-range requires lower < upper (%s)\n", spec.c_str());
    return false;
  }
  double* slot_lo = nullptr;
  double* slot_hi = nullptr;
  unsigned bit = 0;
  if (metric == "psnr") {
    slot_lo = &opt.qcs_psnr_lo;
    slot_hi = &opt.qcs_psnr_hi;
    bit = 1u << 0;
  } else if (metric == "xpsnr") {
    slot_lo = &opt.qcs_xpsnr_lo;
    slot_hi = &opt.qcs_xpsnr_hi;
    bit = 1u << 1;
  } else if (metric == "ssim") {
    slot_lo = &opt.qcs_ssim_lo;
    slot_hi = &opt.qcs_ssim_hi;
    bit = 1u << 2;
  } else if (metric == "msssim") {
    slot_lo = &opt.qcs_msssim_lo;
    slot_hi = &opt.qcs_msssim_hi;
    bit = 1u << 3;
  } else if (metric == "vmaf") {
    slot_lo = &opt.qcs_vmaf_lo;
    slot_hi = &opt.qcs_vmaf_hi;
    bit = 1u << 4;
  } else if (metric == "pu21-psnr") {
    slot_lo = &opt.qcs_pu21_psnr_lo;
    slot_hi = &opt.qcs_pu21_psnr_hi;
    bit = 1u << 5;
  } else if (metric == "pu21-ssim") {
    slot_lo = &opt.qcs_pu21_ssim_lo;
    slot_hi = &opt.qcs_pu21_ssim_hi;
    bit = 1u << 6;
  } else if (metric == "pu21-msssim") {
    slot_lo = &opt.qcs_pu21_msssim_lo;
    slot_hi = &opt.qcs_pu21_msssim_hi;
    bit = 1u << 7;
  } else if (metric == "cvqa-hdr-nr") {
    slot_lo = &opt.qcs_cvqahdr_nr_lo;
    slot_hi = &opt.qcs_cvqahdr_nr_hi;
    bit = kQcsCvqaHdrNrBit;
  } else if (metric == "cvvdp") {
    slot_lo = &opt.qcs_cvvdp_lo;
    slot_hi = &opt.qcs_cvvdp_hi;
    bit = kQcsCvvdpBit;
  } else {
    std::fprintf(stderr, "pixvqa: unknown --qcs-range metric %s\n", metric.c_str());
    return false;
  }
  if ((opt.qcs_range_seen & bit) != 0) {
    if (*slot_lo == lo && *slot_hi == hi) return true;
    std::fprintf(stderr, "pixvqa: conflicting --qcs-range for %s\n", metric.c_str());
    return false;
  }
  *slot_lo = lo;
  *slot_hi = hi;
  opt.qcs_range_seen |= bit;
  opt.qcs_range_set = true;
  return true;
}

bool assign_qcs_range_list(Options& opt, const std::string& spec) {
  if (spec.empty()) {
    std::fprintf(stderr, "pixvqa: empty --qcs-range entry\n");
    return false;
  }
  std::size_t begin = 0;
  for (;;) {
    const auto comma = spec.find(',', begin);
    const auto end = comma == std::string::npos ? spec.size() : comma;
    std::string item = spec.substr(begin, end - begin);
    trim_ascii_space(item);
    if (item.empty()) {
      std::fprintf(stderr, "pixvqa: empty --qcs-range entry\n");
      return false;
    }
    if (!assign_qcs_range(opt, item)) return false;
    if (comma == std::string::npos) return true;
    begin = comma + 1;
  }
}

void format_qcs_bound(double x, char* buf, size_t n) {
  const double tenth = std::round(x * 10.0) / 10.0;
  if ((x == 0.0 || std::fabs(x) >= 2.0) && std::fabs(x - std::round(x)) < 1e-9) {
    std::snprintf(buf, n, "%.0f", x);
  } else if (std::fabs(x - tenth) < 1e-9) {
    std::snprintf(buf, n, "%.1f", x);
  } else {
    std::snprintf(buf, n, "%.6g", x);
  }
}

void print_qcs(const char* name, const QcsAccum& acc) {
  char lo[64], hi[64];
  format_qcs_bound(acc.lower, lo, sizeof(lo));
  format_qcs_bound(acc.upper, hi, sizeof(hi));
  if (!acc.available()) {
    std::printf("[ %s value:unavailable range:[%s,%s] ]\n", name, lo, hi);
    return;
  }
  std::printf("[ %s value:%.6f range:[%s,%s] ]\n", name, acc.percent(), lo, hi);
}

void accumulate(PlaneScore& sum, const PlaneScore& s) {
  sum.y += s.y;
  sum.u += s.u;
  sum.v += s.v;
  sum.avg += s.avg;
}

std::string fmt_xpsnr(double v) {
  if (!std::isfinite(v)) return "inf";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.4f", v);
  return buf;
}

void print_xpsnr(const XpsnrFrame& s) {
  std::printf("[ XPSNR y:%s u:%s v:%s avg:%s min:%s ] ", fmt_xpsnr(s.y).c_str(),
              fmt_xpsnr(s.u).c_str(), fmt_xpsnr(s.v).c_str(), fmt_xpsnr(s.avg).c_str(),
              fmt_xpsnr(s.min_score).c_str());
}

#ifdef PIXVQA_HAS_VMAF
void print_vmaf(const VmafFrame& s) {
  // Per-frame: score only. Temporal pools belong in the summary line.
  std::printf("[ VMAF:%.6f ] ", s.score);
}

void print_vmaf_meta(const VmafMeta& m) {
  if (m.score_clip_known) {
    std::printf(
        "VMAF: model=%s path=%s resolved=%s libvmaf=%s backend=%s "
        "clip:[%.0f,%.0f] sha256=%s\n",
        m.model_id.c_str(), m.model_path_requested.c_str(), m.model_path_resolved.c_str(),
        m.libvmaf_version.c_str(), m.backend.c_str(), m.score_clip_lo, m.score_clip_hi,
        m.model_sha256.c_str());
  } else {
    std::printf(
        "VMAF: model=%s path=%s resolved=%s libvmaf=%s backend=%s "
        "clip:unknown sha256=%s\n",
        m.model_id.c_str(), m.model_path_requested.c_str(), m.model_path_resolved.c_str(),
        m.libvmaf_version.c_str(), m.backend.c_str(), m.model_sha256.c_str());
  }
  const char* temporal = "none";
  if (m.motion_five_frame && m.motion_moving_average) temporal = "five_frame+moving_average";
  else if (m.motion_five_frame) temporal = "five_frame";
  else if (m.motion_moving_average) temporal = "moving_average";
  std::printf(
      "VMAF-EXT: commit=%s zimg=%s backend=%s conversion=%s display=%d "
      "hfr_temporal=%s color_ref=%s color_dist=%s\n",
      m.libvmaf_commit.c_str(), m.zimg_version.c_str(), m.backend.c_str(), m.conversion.c_str(),
      m.display_short_side, temporal, m.color_ref.c_str(), m.color_dist.c_str());
}
#endif

// The whole token must be a base-10 integer. A partial parse is not a value.
bool parse_cli_int(const std::string& text, long& out) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long n = std::strtol(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || (end && *end != '\0')) return false;
  out = n;
  return true;
}

bool parse_fps(const std::string& s, int& num, int& den) {
  if (s.empty()) return false;
  try {
    const auto slash = s.find('/');
    if (slash != std::string::npos) {
      size_t i1 = 0;
      size_t i2 = 0;
      num = std::stoi(s.substr(0, slash), &i1);
      den = std::stoi(s.substr(slash + 1), &i2);
      if (i1 != slash || i2 + slash + 1 != s.size()) return false;
      return num > 0 && den > 0 && (num / den) > 0;
    }
    if (s.find('.') != std::string::npos) {
      size_t idx = 0;
      const double v = std::stod(s, &idx);
      if (idx != s.size() || !(v > 0.0)) return false;
      num = static_cast<int>(v);  // trunc toward zero, matching integer-rate XPSNR
      den = 1;
      return num > 0;
    }
    size_t idx = 0;
    num = std::stoi(s, &idx);
    den = 1;
    return idx == s.size() && num > 0;
  } catch (...) {
    return false;
  }
}

bool formats_equal(const VideoFormat& a, const VideoFormat& b) {
  return a.width == b.width && a.height == b.height && a.bit_depth == b.bit_depth &&
         a.chroma == b.chroma;
}

void apply_size(SideSpec& side, const std::string& s) {
  if (!parse_size(s, side.format.width, side.format.height)) {
    std::fprintf(stderr, "bad -s size\n");
    std::exit(2);
  }
  side.has_size = true;
}

void apply_format(SideSpec& side, const std::string& s) {
  if (!parse_format(s, side.format.chroma, side.format.bit_depth)) {
    std::fprintf(stderr, "unsupported -f format (need 8-bit or 10-bit 420/422/444)\n");
    std::exit(2);
  }
  side.has_fmt = true;
}

void apply_size_both(SideSpec& ref, SideSpec& test, const std::string& s) {
  apply_size(ref, s);
  apply_size(test, s);
}

void apply_format_both(SideSpec& ref, SideSpec& test, const std::string& s) {
  apply_format(ref, s);
  apply_format(test, s);
}

// What the unified CLI needs to know about each metric before it runs any of them.
enum class MetricExec { Native, PythonBackend };
enum class MetricQcs { PerFrame, SampledFrames, Unsupported };

struct MetricCaps {
  const char* name;
  bool full_reference;
  MetricExec exec;
  bool accepts_raw;      // raw YUV test input
  bool honors_frames;    // --frames limits what is scored
  MetricQcs qcs;
};

constexpr MetricCaps kMetricCaps[] = {
    {"psnr", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"xpsnr", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"ssim", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"msssim", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"vmaf", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"pu21-psnr", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"pu21-ssim", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    {"pu21-msssim", true, MetricExec::Native, true, true, MetricQcs::PerFrame},
    // Raw CVVDP is limited to YUV420p10le PQ pairs; see prepare_cvvdp_raw.
    {"cvvdp", true, MetricExec::PythonBackend, true, true, MetricQcs::Unsupported},
    {"cvqa-hdr-nr", false, MetricExec::PythonBackend, false, false, MetricQcs::SampledFrames},
};

const MetricCaps& metric_caps(const char* name) {
  for (const MetricCaps& caps : kMetricCaps) {
    if (std::strcmp(caps.name, name) == 0) return caps;
  }
  std::abort();
}

void select_all_metrics(Options& opt, CvvdpOptions& cvvdp, CvqaHdrNrOptions& cvqahdr) {
  opt.enable_psnr = true;
  opt.enable_xpsnr = true;
  opt.enable_ssim = true;
  opt.enable_msssim = true;
  opt.enable_vmaf = true;
  opt.enable_pu21_psnr = true;
  opt.enable_pu21_ssim = true;
  opt.enable_pu21_msssim = true;
  cvvdp.selected = true;
  cvqahdr.selected = true;
}

bool add_metric_name(Options& opt, CvvdpOptions& cvvdp, CvqaHdrNrOptions& cvqahdr,
                     const std::string& name) {
  if (name.empty()) {
    std::fprintf(stderr, "pixvqa: empty --metrics entry\n");
    return false;
  }
  if (name == "all") {
    select_all_metrics(opt, cvvdp, cvqahdr);
    return true;
  }
  if (name == "pq" || name == "hlg" || name == "sdr") {
    if (!opt.metric_group.empty() && opt.metric_group != name) {
      std::fprintf(stderr, "pixvqa: --metrics %s and %s cannot be combined\n",
                   opt.metric_group.c_str(), name.c_str());
      return false;
    }
    opt.metric_group = name;
    opt.enable_psnr = opt.enable_xpsnr = opt.enable_ssim = opt.enable_msssim = true;
    cvvdp.selected = true;
    if (name == "sdr") {
      opt.enable_vmaf = true;
      cvqahdr.selected = true;
    } else if (name == "pq") {
      opt.enable_vmaf = true;
      opt.enable_pu21_psnr = opt.enable_pu21_ssim = opt.enable_pu21_msssim = true;
      cvqahdr.selected = true;
    } else {
      opt.enable_pu21_psnr = opt.enable_pu21_ssim = opt.enable_pu21_msssim = true;
    }
    return true;
  }
  if (name == "psnr") opt.enable_psnr = true;
  else if (name == "xpsnr") opt.enable_xpsnr = true;
  else if (name == "ssim") opt.enable_ssim = true;
  else if (name == "msssim") opt.enable_msssim = true;
  else if (name == "vmaf") opt.enable_vmaf = true;
  else if (name == "pu21-psnr") opt.enable_pu21_psnr = true;
  else if (name == "pu21-ssim") opt.enable_pu21_ssim = true;
  else if (name == "pu21-msssim") opt.enable_pu21_msssim = true;
  else if (name == "cvvdp") cvvdp.selected = true;
  else if (name == "cvqa-hdr-nr") cvqahdr.selected = true;
  else {
    std::fprintf(stderr,
                 "pixvqa: unsupported metric %s (supported: psnr, xpsnr, ssim, msssim, vmaf, "
                 "pu21-psnr, pu21-ssim, pu21-msssim, cvvdp, cvqa-hdr-nr; groups: all, pq, hlg, "
                 "sdr)\n",
                 name.c_str());
    return false;
  }
  return true;
}

bool parse_metrics_list(Options& opt, CvvdpOptions& cvvdp, CvqaHdrNrOptions& cvqahdr,
                        const std::string& list) {
  std::string cur;
  auto flush = [&](const std::string& name) { return add_metric_name(opt, cvvdp, cvqahdr, name); };
  for (char c : list) {
    if (c == ',') {
      if (!flush(cur)) return false;
      cur.clear();
    } else if (c != ' ' && c != '\t') {
      cur.push_back(static_cast<char>(c));
    }
  }
  return flush(cur);
}

bool native_selected(const Options& opt) {
  return opt.enable_psnr || opt.enable_ssim || opt.enable_msssim || opt.enable_xpsnr ||
         opt.enable_vmaf || pu21_selected(opt);
}

std::vector<const MetricCaps*> selected_metric_caps(const Options& opt, const CvvdpOptions& cvvdp,
                                                    const CvqaHdrNrOptions& cvqahdr) {
  std::vector<const MetricCaps*> out;
  const std::pair<bool, const char*> picks[] = {
      {opt.enable_psnr, "psnr"},           {opt.enable_xpsnr, "xpsnr"},
      {opt.enable_ssim, "ssim"},           {opt.enable_msssim, "msssim"},
      {opt.enable_vmaf, "vmaf"},           {opt.enable_pu21_psnr, "pu21-psnr"},
      {opt.enable_pu21_ssim, "pu21-ssim"}, {opt.enable_pu21_msssim, "pu21-msssim"},
      {cvvdp.selected, "cvvdp"},           {cvqahdr.selected, "cvqa-hdr-nr"},
  };
  for (const auto& pick : picks) {
    if (pick.first) out.push_back(&metric_caps(pick.second));
  }
  return out;
}

std::string join_metric_names(const std::vector<const MetricCaps*>& caps, bool full_reference) {
  std::string names;
  for (const MetricCaps* c : caps) {
    if (c->full_reference != full_reference) continue;
    if (!names.empty()) names += ", ";
    names += c->name;
  }
  return names;
}

void mark_cvqahdr_raw_option(CvqaHdrNrOptions& cvqahdr, const std::string& option) {
  if (cvqahdr.raw_option.empty()) cvqahdr.raw_option = option;
}

bool path_ends_with_yuv(const std::string& path) {
  const char* ext = ".yuv";
  const size_t n = std::strlen(ext);
  if (path.size() < n) return false;
  for (size_t i = 0; i < n; ++i) {
    const unsigned char a = static_cast<unsigned char>(path[path.size() - n + i]);
    const unsigned char b = static_cast<unsigned char>(ext[i]);
    if (std::tolower(a) != std::tolower(b)) return false;
  }
  return true;
}

void inherit_side_geometry(SideSpec& ref, SideSpec& test) {
  if (!ref.has_size && test.has_size) {
    ref.format.width = test.format.width;
    ref.format.height = test.format.height;
    ref.has_size = true;
  }
  if (!test.has_size && ref.has_size) {
    test.format.width = ref.format.width;
    test.format.height = ref.format.height;
    test.has_size = true;
  }
  if (!ref.has_fmt && test.has_fmt) {
    ref.format.chroma = test.format.chroma;
    ref.format.bit_depth = test.format.bit_depth;
    ref.has_fmt = true;
  }
  if (!test.has_fmt && ref.has_fmt) {
    test.format.chroma = ref.format.chroma;
    test.format.bit_depth = ref.format.bit_depth;
    test.has_fmt = true;
  }
}

bool cvvdp_raw_requested(const SideSpec& ref, const SideSpec& test, const Options& opt) {
  return ref.has_size || test.has_size || ref.has_fmt || test.has_fmt || opt.fps_set ||
         path_ends_with_yuv(opt.ref_path) || path_ends_with_yuv(opt.test_path);
}

bool prepare_cvvdp_raw(const SideSpec& ref, const SideSpec& test, const Options& opt,
                       const CvvdpOptions& cvvdp) {
  if (!ref.has_size || !test.has_size || !ref.has_fmt || !test.has_fmt) {
    std::fprintf(stderr, "pixvqa: cvvdp raw YUV requires -s and -f\n");
    return false;
  }
  if (!formats_equal(ref.format, test.format)) {
    std::fprintf(stderr, "pixvqa: cvvdp reference and test -s/-f disagree\n");
    return false;
  }
  if (ref.format.chroma != ChromaFormat::YUV420 || ref.format.bit_depth != 10) {
    std::fprintf(stderr,
                 "pixvqa: cvvdp raw input supports only planar YUV420p10le (-f 420p:10)\n");
    return false;
  }
  if (ref.format.width % 2 != 0 || ref.format.height % 2 != 0) {
    std::fprintf(stderr, "pixvqa: cvvdp raw YUV420 requires positive even width and height\n");
    return false;
  }
  if (!opt.fps_set) {
    std::fprintf(stderr, "pixvqa: cvvdp raw YUV requires --fps\n");
    return false;
  }
  if (cvvdp.interpretation != "pq" && cvvdp.interpretation != "hlg") {
    std::fprintf(stderr,
                 "pixvqa: cvvdp raw YUV supports only --cvvdp-interpretation pq or hlg; "
                 "SDR raw YUV is unsupported\n");
    return false;
  }
  return true;
}

bool assign_cvvdp_value(std::string& field, const char* flag, const std::string& value) {
  if (!field.empty() && field != value) {
    std::fprintf(stderr, "pixvqa: conflicting %s values (%s vs %s)\n", flag, field.c_str(),
                 value.c_str());
    return false;
  }
  field = value;
  return true;
}

bool assign_cvqahdr_value(std::string& field, const char* flag, const std::string& value) {
  if (!field.empty() && field != value) {
    std::fprintf(stderr, "pixvqa: conflicting %s values (%s vs %s)\n", flag, field.c_str(),
                 value.c_str());
    return false;
  }
  field = value;
  return true;
}

bool validate_cvvdp_options(const CvvdpOptions& cvvdp) {
  if (cvvdp.interpretation != "sdr" && cvvdp.interpretation != "pq" &&
      cvvdp.interpretation != "hlg") {
    std::fprintf(stderr,
                 "pixvqa: cvvdp requires --cvvdp-interpretation sdr, pq, or hlg; "
                 "it is never inferred\n");
    return false;
  }
  if (cvvdp.display.empty()) {
    std::fprintf(stderr, "pixvqa: cvvdp requires --cvvdp-display <official-display-model>\n");
    return false;
  }
  return true;
}

bool validate_cvqahdr_options(const CvqaHdrNrOptions& cvqahdr) {
  if (cvqahdr.repository.empty() || cvqahdr.checkpoint.empty() || cvqahdr.profile.empty()) {
    std::fprintf(stderr,
                 "pixvqa: cvqa-hdr-nr requires --cvqahdr-repo, --cvqahdr-checkpoint, and "
                 "--cvqahdr-profile, or the same keys in the learned-metric config\n");
    return false;
  }
  return true;
}

// Transfer family of a CLI token: "pq", "hlg", "sdr", or "" for anything else.
std::string transfer_family(const std::string& token) {
  if (token == "pq" || token == "st2084") return "pq";
  if (token == "hlg" || token == "stdb67") return "hlg";
  if (token == "sdr" || token == "bt709") return "sdr";
  return "";
}

std::string executable_dir() {
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  const std::string exe(buf);
  const size_t slash = exe.rfind('/');
  return slash == std::string::npos ? "" : exe.substr(0, slash);
}

bool readable_file(const std::string& path) {
  return !path.empty() && access(path.c_str(), R_OK) == 0;
}

// A checkout-relative file next to the executable: <exe>/../REL (bin/) or
// <exe>/../../REL (build/<kind>/). Empty when neither exists.
std::string locate_beside_executable(const std::string& rel) {
  const std::string dir = executable_dir();
  if (dir.empty()) return "";
  for (const char* up : {"/../", "/../../"}) {
    const std::string candidate = dir + up + rel;
    if (readable_file(candidate)) return candidate;
  }
  return "";
}

// Metric math is unchanged: --preset hdr and --metrics pq|hlg|sdr only select
// metrics and fill the existing HDR, VMAF, and CVVDP options.
bool apply_processing_defaults(Options& opt, CvvdpOptions& cvvdp, CvqaHdrNrOptions& cvqahdr) {
  if (opt.preset_hdr && !opt.metric_group.empty() && opt.metric_group != "pq") {
    std::fprintf(stderr, "pixvqa: --preset hdr cannot be combined with --metrics %s\n",
                 opt.metric_group.c_str());
    return false;
  }
  const std::string kind = opt.preset_hdr ? "pq" : opt.metric_group;
  if (kind.empty()) return true;
  const std::string who = opt.preset_hdr ? "--preset hdr" : "--metrics " + kind;
  if (opt.preset_hdr && !native_selected(opt) && !cvvdp.selected && !cvqahdr.selected) {
    select_all_metrics(opt, cvvdp, cvqahdr);
  }
  const std::pair<bool, std::string> transfers[] = {
      {opt.transfer_set, opt.transfer},
      {opt.ref_transfer_set, opt.ref_transfer},
      {opt.dist_transfer_set, opt.dist_transfer},
      {opt.hdr.transfer_set, opt.hdr.transfer},
      {opt.ref_hdr.transfer_set, opt.ref_hdr.transfer},
      {opt.dist_hdr.transfer_set, opt.dist_hdr.transfer},
  };
  for (const auto& item : transfers) {
    if (item.first && transfer_family(item.second) != kind) {
      std::fprintf(stderr, "pixvqa: %s uses %s; transfer %s conflicts\n", who.c_str(),
                   kind.c_str(), item.second.c_str());
      return false;
    }
  }
  if (kind == "sdr" && pu21_selected(opt)) {
    std::fprintf(stderr,
                 "pixvqa: %s excludes pu21-psnr, pu21-ssim, and pu21-msssim "
                 "(PU21 is for PQ and HLG)\n",
                 who.c_str());
    return false;
  }
  if (kind == "hlg" && opt.enable_vmaf) {
    std::fprintf(stderr,
                 "pixvqa: %s excludes vmaf (the installed HDR VMAF model is PQ-only; "
                 "an SDR VMAF model rejects HLG)\n",
                 who.c_str());
    return false;
  }
  if (kind == "hlg" && cvqahdr.selected) {
    std::fprintf(stderr,
                 "pixvqa: %s excludes cvqa-hdr-nr (the installed checkpoint and OpenCV "
                 "preprocessing are specified for HDR10 and SDR, not HLG)\n",
                 who.c_str());
    return false;
  }
  if (kind != "sdr" && !opt.transfer_set && !opt.hdr.transfer_set) {
    opt.hdr.transfer = kind;
    opt.hdr.transfer_set = true;
  }
  if (kind == "pq" || kind == "hlg") {
    const HdrCli* sides[] = {&opt.hdr, &opt.ref_hdr, &opt.dist_hdr};
    for (const HdrCli* side : sides) {
      if (side->primaries_set && side->primaries != "bt2020") {
        std::fprintf(stderr, "pixvqa: %s uses BT.2020 primaries; primaries %s conflicts\n",
                     who.c_str(), side->primaries.c_str());
        return false;
      }
      if (side->matrix_set && side->matrix != "bt2020nc") {
        std::fprintf(stderr, "pixvqa: %s uses the bt2020nc matrix; matrix %s conflicts\n",
                     who.c_str(), side->matrix.c_str());
        return false;
      }
      if (side->range_set && side->range != "limited") {
        std::fprintf(stderr, "pixvqa: %s uses limited range; range %s conflicts\n", who.c_str(),
                     side->range.c_str());
        return false;
      }
    }
    if (!opt.hdr.primaries_set && !opt.ref_hdr.primaries_set && !opt.dist_hdr.primaries_set) {
      opt.hdr.primaries = "bt2020";
      opt.hdr.primaries_set = true;
    }
    if (!opt.hdr.matrix_set && !opt.ref_hdr.matrix_set && !opt.dist_hdr.matrix_set) {
      opt.hdr.matrix = "bt2020nc";
      opt.hdr.matrix_set = true;
    }
    if (!opt.hdr.range_set && !opt.ref_hdr.range_set && !opt.dist_hdr.range_set) {
      opt.hdr.range = "limited";
      opt.hdr.range_set = true;
    }
  }
  if (opt.enable_vmaf && opt.vmaf_model_path.empty() && kind != "hlg") {
    const std::string rel = kind == "pq" ? kHdrVmafModel : kSdrVmafModel;
    const std::string beside = readable_file(rel) ? "" : locate_beside_executable(rel);
    opt.vmaf_model_path = beside.empty() ? rel : beside;
    opt.vmaf_model_from_model = true;
  }
  if (cvvdp.selected) {
    if (cvvdp.interpretation.empty()) cvvdp.interpretation = kind;
    else if (cvvdp.interpretation != kind) {
      std::fprintf(stderr, "pixvqa: %s uses --cvvdp-interpretation %s\n", who.c_str(),
                   kind.c_str());
      return false;
    }
    const char* display = nullptr;
    if (kind == "pq") display = "standard_hdr_pq";
    else if (kind == "hlg") display = "standard_hdr_hlg";
    if (display) {
      if (cvvdp.display.empty()) cvvdp.display = display;
      else if (cvvdp.display != display) {
        std::fprintf(stderr, "pixvqa: %s uses --cvvdp-display %s\n", who.c_str(), display);
        return false;
      }
    } else if (cvvdp.display.empty()) {
      std::fprintf(stderr,
                   "pixvqa: %s needs --cvvdp-display standard_fhd or standard_4k "
                   "(the SDR display model is not inferred from resolution)\n",
                   who.c_str());
      return false;
    }
  }
  return true;
}

// Values read from config/learned_metrics.local.toml (see docs/LEARNED_METRICS_CONFIG.md).
struct LearnedConfig {
  std::string path;
  std::string python;
  std::string adapter_path;
  std::string extra_path;
  std::string cvvdp_python;
  std::string cvvdp_executable;
  std::string nr_python;
  std::string nr_repository;
  std::string nr_checkpoint;
  std::string nr_profile;
};

LearnedConfig g_learned_config;

bool parse_toml_string(const std::string& line, size_t& i, std::string& out) {
  const char quote = line[i++];
  out.clear();
  while (i < line.size()) {
    const char c = line[i++];
    if (c == quote) return true;
    if (quote == '"' && c == '\\') {
      if (i >= line.size()) return false;
      const char e = line[i++];
      if (e == '"' || e == '\\') out.push_back(e);
      else if (e == 'n') out.push_back('\n');
      else if (e == 't') out.push_back('\t');
      else return false;
    } else {
      out.push_back(c);
    }
  }
  return false;
}

// Values without '/' are commands looked up on PATH. "~/" is $HOME. Other
// relative paths are relative to the configuration file's directory.
std::string resolve_config_path(const std::string& value, const std::string& config_dir) {
  if (value.empty() || value.find('/') == std::string::npos || value[0] == '/') return value;
  if (value.compare(0, 2, "~/") == 0) {
    const char* home = std::getenv("HOME");
    return home ? std::string(home) + value.substr(1) : value;
  }
  std::string rel = value;
  while (rel.compare(0, 2, "./") == 0) rel.erase(0, 2);
  return rel.empty() ? config_dir : config_dir + "/" + rel;
}

// A TOML subset: comments, [section] tables, and key = "string" or 'string'.
bool load_learned_config(const std::string& path, LearnedConfig& config) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "pixvqa: cannot read learned-metric config %s\n", path.c_str());
    return false;
  }
  config = LearnedConfig();
  config.path = path;
  const size_t slash = path.rfind('/');
  const std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
  std::string section;
  std::vector<std::string> seen;
  std::string line;
  int line_no = 0;
  auto bad = [&](const std::string& why) {
    std::fprintf(stderr, "pixvqa: %s:%d: %s\n", path.c_str(), line_no, why.c_str());
    return false;
  };
  while (std::getline(in, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size() || line[i] == '#') continue;
    if (line[i] == '[') {
      const size_t close = line.find(']', i);
      if (close == std::string::npos) return bad("unterminated [section]");
      section = line.substr(i + 1, close - i - 1);
      trim_ascii_space(section);
      std::string rest = line.substr(close + 1);
      trim_ascii_space(rest);
      if (!rest.empty() && rest[0] != '#') return bad("unexpected text after [section]");
      if (section != "learned" && section != "cvvdp" && section != "cvqa-hdr-nr")
        return bad("unknown section [" + section + "] (expected [learned], [cvvdp], or [cvqa-hdr-nr])");
      continue;
    }
    const size_t eq = line.find('=', i);
    if (eq == std::string::npos) return bad("expected key = \"value\"");
    std::string key = line.substr(i, eq - i);
    trim_ascii_space(key);
    size_t v = eq + 1;
    while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
    if (v >= line.size() || (line[v] != '"' && line[v] != '\''))
      return bad("only quoted string values are supported");
    std::string value;
    if (!parse_toml_string(line, v, value)) return bad("malformed string value");
    while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
    if (v < line.size() && line[v] != '#') return bad("unexpected text after value");
    if (section.empty()) return bad("keys must be inside a section");
    const std::string full = section + "." + key;
    if (std::find(seen.begin(), seen.end(), full) != seen.end()) return bad("duplicate key " + full);
    seen.push_back(full);
    std::string* slot = nullptr;
    if (full == "learned.python") slot = &config.python;
    else if (full == "learned.adapter_path") slot = &config.adapter_path;
    else if (full == "learned.path") slot = &config.extra_path;
    else if (full == "cvvdp.python") slot = &config.cvvdp_python;
    else if (full == "cvvdp.executable") slot = &config.cvvdp_executable;
    else if (full == "cvqa-hdr-nr.python") slot = &config.nr_python;
    else if (full == "cvqa-hdr-nr.repository") slot = &config.nr_repository;
    else if (full == "cvqa-hdr-nr.checkpoint") slot = &config.nr_checkpoint;
    else if (full == "cvqa-hdr-nr.profile") slot = &config.nr_profile;
    else return bad("unknown key " + full);
    *slot = resolve_config_path(value, dir);
  }
  return true;
}

// --learned-config, then PIXVQA_LEARNED_CONFIG ("none" disables), then
// config/learned_metrics.local.toml beside the executable's checkout.
bool resolve_learned_config(const std::string& cli_path) {
  std::string path = cli_path;
  bool required = !path.empty();
  if (path.empty()) {
    const char* env = pix_getenv("LEARNED_CONFIG");
    if (env && std::strcmp(env, "none") == 0) return true;
    if (env && env[0]) {
      path = env;
      required = true;
    }
  }
  if (path.empty()) path = locate_beside_executable("config/learned_metrics.local.toml");
  if (path.empty()) return true;
  if (!required && !readable_file(path)) return true;
  if (!load_learned_config(path, g_learned_config)) return false;
  std::fprintf(stderr, "pixvqa: info: learned-metric config %s\n", path.c_str());
  const LearnedConfig& c = g_learned_config;
  if (!c.adapter_path.empty()) {
    const char* existing = std::getenv("PYTHONPATH");
    const std::string value = existing && existing[0]
                                  ? std::string(existing) + ":" + c.adapter_path
                                  : c.adapter_path;
    setenv("PYTHONPATH", value.c_str(), 1);
  }
  if (!c.extra_path.empty()) {
    const char* existing = std::getenv("PATH");
    const std::string value =
        existing && existing[0] ? c.extra_path + ":" + existing : c.extra_path;
    setenv("PATH", value.c_str(), 1);
  }
  return true;
}

// Option, then PIXVQA_<metric>_PYTHON, then the metric's config entry, then
// [learned] python, then python3.
std::string backend_python(const std::string& configured_by_option, const char* env_name,
                           const std::string& config_metric) {
  if (!configured_by_option.empty()) return configured_by_option;
  const char* configured = std::getenv(env_name);
  if (configured && configured[0]) return configured;
  if (!config_metric.empty()) return config_metric;
  if (!g_learned_config.python.empty()) return g_learned_config.python;
  return "python3";
}

void apply_learned_config(CvvdpOptions& cvvdp, CvqaHdrNrOptions& cvqahdr) {
  const LearnedConfig& c = g_learned_config;
  if (cvvdp.cvvdp_bin.empty()) cvvdp.cvvdp_bin = c.cvvdp_executable;
  if (cvqahdr.repository.empty()) cvqahdr.repository = c.nr_repository;
  if (cvqahdr.checkpoint.empty()) cvqahdr.checkpoint = c.nr_checkpoint;
  if (cvqahdr.profile.empty()) cvqahdr.profile = c.nr_profile;
}

void write_all(int fd, const char* data, size_t size) {
  while (size > 0) {
    const ssize_t n = write(fd, data, size);
    if (n < 0) {
      if (errno == EINTR) continue;
      return;
    }
    data += n;
    size -= static_cast<size_t>(n);
  }
}

struct BackendProcess {
  bool started = false;
  bool exited = false;
  int exit_code = -1;
  std::string out;
};

// stdout is captured for the caller; stderr is forwarded as it arrives. Both pipes
// are drained together so a backend filling one cannot block on the other.
BackendProcess run_backend_process(const char* label, std::vector<std::string> args) {
  BackendProcess proc;
  std::fflush(stdout);
  std::fflush(stderr);
  int out_pipe[2];
  int err_pipe[2];
  if (pipe(out_pipe) != 0) {
    std::fprintf(stderr, "pixvqa: could not create %s pipe: %s\n", label, std::strerror(errno));
    return proc;
  }
  if (pipe(err_pipe) != 0) {
    std::fprintf(stderr, "pixvqa: could not create %s pipe: %s\n", label, std::strerror(errno));
    close(out_pipe[0]);
    close(out_pipe[1]);
    return proc;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "pixvqa: could not start %s backend: %s\n", label, std::strerror(errno));
    for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]}) close(fd);
    return proc;
  }
  if (pid == 0) {
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(err_pipe[1], STDERR_FILENO);
    for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]}) close(fd);
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (std::string& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    std::fprintf(stderr, "pixvqa: could not execute %s interpreter '%s': %s\n", label, argv[0],
                 std::strerror(errno));
    _exit(127);
  }
  proc.started = true;
  close(out_pipe[1]);
  close(err_pipe[1]);
  pollfd fds[2] = {{out_pipe[0], POLLIN, 0}, {err_pipe[0], POLLIN, 0}};
  int open_fds = 2;
  char buffer[4096];
  while (open_fds > 0) {
    if (poll(fds, 2, -1) < 0) {
      if (errno == EINTR) continue;
      break;
    }
    for (int i = 0; i < 2; ++i) {
      if (fds[i].fd < 0 || fds[i].revents == 0) continue;
      const ssize_t n = read(fds[i].fd, buffer, sizeof(buffer));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        close(fds[i].fd);
        fds[i].fd = -1;
        --open_fds;
        continue;
      }
      if (i == 0) proc.out.append(buffer, static_cast<size_t>(n));
      else write_all(STDERR_FILENO, buffer, static_cast<size_t>(n));
    }
  }
  for (pollfd& p : fds) {
    if (p.fd >= 0) close(p.fd);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) continue;
    std::fprintf(stderr, "pixvqa: %s backend wait failed: %s\n", label, std::strerror(errno));
    return proc;
  }
  if (WIFEXITED(status)) {
    proc.exited = true;
    proc.exit_code = WEXITSTATUS(status);
  }
  return proc;
}

// 0 completed, 2 configuration or preflight failure, 1 runtime failure.
// Adapters exit 2 for preflight; 127 means the interpreter could not be executed.
int backend_status(const char* label, const BackendProcess& proc) {
  if (!proc.started) return 2;
  if (proc.exited && proc.exit_code == 0) return 0;
  if (proc.exited) {
    std::fprintf(stderr, "pixvqa: %s backend exited %d\n", label, proc.exit_code);
  } else {
    std::fprintf(stderr, "pixvqa: %s backend terminated unexpectedly\n", label);
  }
  if (proc.exited && (proc.exit_code == 2 || proc.exit_code == 127)) return 2;
  return 1;
}

struct BackendSample {
  long index = 0;
  long frame_index = 0;
  double time_seconds = 0.0;
  double raw = 0.0;
  double calibrated = 0.0;
};

struct BackendReport {
  std::vector<std::pair<std::string, std::string>> values;
  std::vector<BackendSample> samples;

  const std::string* find(const char* key) const {
    for (const auto& kv : values) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
};

bool parse_full_double(const std::string& text, double& out) {
  if (text.empty()) return false;
  char* end = nullptr;
  errno = 0;
  out = std::strtod(text.c_str(), &end);
  return end != text.c_str() && *end == '\0' && errno != ERANGE;
}

// Line protocol: "pixvqa-protocol 1", then "pixvqa-value KEY VALUE" and
// "pixvqa-sample INDEX FRAME TIME RAW CALIBRATED" lines, then "pixvqa-end".
// Floats are Python repr, the shortest text that round-trips to the same double.
bool parse_backend_report(const std::string& text, BackendReport& report, std::string& err) {
  std::vector<std::string> lines;
  size_t begin = 0;
  while (begin < text.size()) {
    size_t end = text.find('\n', begin);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(begin, end - begin);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) lines.push_back(line);
    begin = end + 1;
  }
  if (lines.empty() || lines.front() != "pixvqa-protocol 1") {
    err = "backend stdout did not start with pixvqa-protocol 1";
    return false;
  }
  if (lines.back() != "pixvqa-end") {
    err = "backend stdout did not end with pixvqa-end";
    return false;
  }
  for (size_t i = 1; i + 1 < lines.size(); ++i) {
    const std::string& line = lines[i];
    if (line.compare(0, 13, "pixvqa-value ") == 0) {
      const size_t space = line.find(' ', 13);
      if (space == std::string::npos || space == 13 || space + 1 >= line.size()) {
        err = "malformed protocol line: " + line;
        return false;
      }
      report.values.emplace_back(line.substr(13, space - 13), line.substr(space + 1));
    } else if (line.compare(0, 14, "pixvqa-sample ") == 0) {
      std::vector<std::string> fields;
      size_t pos = 14;
      while (pos <= line.size()) {
        size_t next = line.find(' ', pos);
        if (next == std::string::npos) next = line.size();
        fields.push_back(line.substr(pos, next - pos));
        pos = next + 1;
      }
      BackendSample sample;
      long index = 0;
      long frame = 0;
      if (fields.size() != 5 || !parse_cli_int(fields[0], index) || !parse_cli_int(fields[1], frame) ||
          !parse_full_double(fields[2], sample.time_seconds) ||
          !parse_full_double(fields[3], sample.raw) ||
          !parse_full_double(fields[4], sample.calibrated)) {
        err = "malformed protocol line: " + line;
        return false;
      }
      if (index != static_cast<long>(report.samples.size()) ||
          (!report.samples.empty() && frame <= report.samples.back().frame_index)) {
        err = "protocol samples are not in strictly increasing frame order";
        return false;
      }
      sample.index = index;
      sample.frame_index = frame;
      report.samples.push_back(sample);
    } else {
      err = "unexpected protocol line: " + line;
      return false;
    }
  }
  return true;
}

bool report_has_number(const BackendReport& report, const char* key) {
  const std::string* v = report.find(key);
  double d = 0.0;
  return v && parse_full_double(*v, d) && std::isfinite(d);
}

enum class BackendState { NotRun, Completed, Failed };

struct BackendOutcome {
  BackendState state = BackendState::NotRun;
  BackendReport report;
};

struct LearnedPlan {
  Options opt;
  CvvdpOptions cvvdp;
  CvqaHdrNrOptions cvqahdr;
  bool cvvdp_raw = false;
  int raw_width = 0;
  int raw_height = 0;
  TemporaryCvvdpDirectory cvvdp_temporary;
  TemporaryCvqaHdrDirectory cvqahdr_temporary;
  BackendOutcome cvvdp_result;
  BackendOutcome cvqahdr_result;
  bool preflight_passed = false;
  bool executed = false;
};

// Learned backends resolve auto themselves (CUDA when PyTorch sees it, else CPU);
// gpu maps to cuda, which the adapters refuse rather than run on the CPU.
const char* device_token(const Options& opt) {
  if (opt.metric_device == DeviceChoice::Cpu) return "cpu";
  if (opt.metric_device == DeviceChoice::Gpu) return "cuda";
  return "auto";
}

bool make_backend_directory(std::string& path, const char* pattern, const char* label) {
  if (!path.empty()) return true;
  std::vector<char> buffer(pattern, pattern + std::strlen(pattern) + 1);
  if (!mkdtemp(buffer.data())) {
    std::fprintf(stderr, "pixvqa: could not create %s temporary directory: %s\n", label,
                 std::strerror(errno));
    return false;
  }
  path = buffer.data();
  return true;
}

bool cvvdp_command(LearnedPlan& plan, std::vector<std::string>& args) {
  const CvvdpOptions& cvvdp = plan.cvvdp;
  if (!cvvdp.output_json_set &&
      !make_backend_directory(plan.cvvdp_temporary.path, "/tmp/pixvqa-cvvdp-XXXXXX", "CVVDP"))
    return false;
  const std::string output_json =
      cvvdp.output_json_set ? cvvdp.output_json : plan.cvvdp_temporary.path + "/result.json";
  args = {backend_python(cvvdp.python, "PIXVQA_CVVDP_PYTHON", g_learned_config.cvvdp_python),
          "-m", "pixvqa_cvvdp", "--ref",
          plan.opt.ref_path, "--test", plan.opt.test_path, "--interpretation", cvvdp.interpretation,
          "--display", cvvdp.display, "--nframes", std::to_string(plan.opt.max_frames), "--device",
          device_token(plan.opt), "--output-json", output_json};
  if (!cvvdp.output_json_set) args.emplace_back("--transient-output");
  if (plan.opt.enable_qcs) args.emplace_back("--temporal");
  if (!cvvdp.cvvdp_bin.empty()) {
    args.emplace_back("--cvvdp-bin");
    args.push_back(cvvdp.cvvdp_bin);
  }
  if (cvvdp.gpu_mem_set) {
    char memory[64];
    std::snprintf(memory, sizeof(memory), "%.17g", cvvdp.gpu_mem_gb);
    args.emplace_back("--gpu-mem");
    args.emplace_back(memory);
  }
  if (plan.cvvdp_raw) {
    args.emplace_back("--raw-width");
    args.push_back(std::to_string(plan.raw_width));
    args.emplace_back("--raw-height");
    args.push_back(std::to_string(plan.raw_height));
    args.emplace_back("--raw-fps-num");
    args.push_back(std::to_string(plan.opt.fps_num));
    args.emplace_back("--raw-fps-den");
    args.push_back(std::to_string(plan.opt.fps_den));
  }
  return true;
}

bool cvqahdr_command(LearnedPlan& plan, std::vector<std::string>& args) {
  const CvqaHdrNrOptions& cvqahdr = plan.cvqahdr;
  if (!cvqahdr.output_json_set &&
      !make_backend_directory(plan.cvqahdr_temporary.path, "/tmp/pixvqa-cvqahdr-nr-XXXXXX",
                              "CVQA-HDR-NR"))
    return false;
  const std::string output_json =
      cvqahdr.output_json_set ? cvqahdr.output_json : plan.cvqahdr_temporary.path + "/result.json";
  args = {backend_python(cvqahdr.python, "PIXVQA_CVQAHDR_PYTHON", g_learned_config.nr_python),
          "-m", "pixvqa_cvqahdr",
          "--test", plan.opt.test_path, "--repo", cvqahdr.repository, "--checkpoint",
          cvqahdr.checkpoint, "--profile", cvqahdr.profile, "--device", device_token(plan.opt),
          "--output-json", output_json};
  if (!cvqahdr.output_json_set) args.emplace_back("--transient-output");
  return true;
}

// One learned metric, no native metric, no QCS: the adapter's JSON line is the report.
int run_learned_standalone(LearnedPlan& plan) {
  const bool cvvdp = plan.cvvdp.selected;
  const char* label = cvvdp ? "CVVDP" : "CVQA-HDR-NR";
  std::vector<std::string> args;
  if (!(cvvdp ? cvvdp_command(plan, args) : cvqahdr_command(plan, args))) return 1;
  const BackendProcess proc = run_backend_process(label, args);
  const int status = backend_status(label, proc);
  if (status == 0) {
    std::fwrite(proc.out.data(), 1, proc.out.size(), stdout);
    if (proc.out.empty() || proc.out.back() != '\n') std::fputc('\n', stdout);
  } else if (!proc.out.empty()) {
    std::fwrite(proc.out.data(), 1, proc.out.size(), stderr);
  }
  return status;
}

int preflight_one(LearnedPlan& plan, bool cvvdp) {
  const char* label = cvvdp ? "CVVDP" : "CVQA-HDR-NR";
  std::vector<std::string> args;
  if (!(cvvdp ? cvvdp_command(plan, args) : cvqahdr_command(plan, args))) return 2;
  args.emplace_back("--check");
  const BackendProcess proc = run_backend_process(label, args);
  if (backend_status(label, proc) != 0) {
    if (!proc.out.empty()) std::fwrite(proc.out.data(), 1, proc.out.size(), stderr);
    std::fprintf(stderr, "pixvqa: %s preflight failed; no metric was run\n", label);
    return 2;
  }
  return 0;
}

int preflight_learned(LearnedPlan& plan) {
  if (plan.cvvdp.selected && preflight_one(plan, true) != 0) return 2;
  if (plan.cvqahdr.selected && preflight_one(plan, false) != 0) return 2;
  plan.preflight_passed = true;
  return 0;
}

void execute_one(LearnedPlan& plan, bool cvvdp) {
  const char* label = cvvdp ? "CVVDP" : "CVQA-HDR-NR";
  BackendOutcome& outcome = cvvdp ? plan.cvvdp_result : plan.cvqahdr_result;
  outcome.state = BackendState::Failed;
  std::vector<std::string> args;
  if (!(cvvdp ? cvvdp_command(plan, args) : cvqahdr_command(plan, args))) return;
  args.emplace_back("--protocol");
  args.emplace_back("lines");
  if (!cvvdp && plan.opt.enable_qcs) args.emplace_back("--verify-decoded-frames");
  const BackendProcess proc = run_backend_process(label, args);
  if (backend_status(label, proc) != 0) {
    if (!proc.out.empty()) std::fwrite(proc.out.data(), 1, proc.out.size(), stderr);
    return;
  }
  std::string err;
  if (!parse_backend_report(proc.out, outcome.report, err)) {
    std::fprintf(stderr, "pixvqa: %s: %s\n", label, err.c_str());
    return;
  }
  const bool complete =
      cvvdp ? report_has_number(outcome.report, "jod")
            : report_has_number(outcome.report, "quality_score") &&
                  report_has_number(outcome.report, "raw_model_score");
  if (!complete) {
    std::fprintf(stderr, "pixvqa: %s: backend report has no finite score\n", label);
    return;
  }
  outcome.state = BackendState::Completed;
}

// Native metrics, when selected, have already run. CVVDP runs before NR.
void execute_learned(LearnedPlan& plan) {
  plan.executed = true;
  if (plan.cvvdp.selected) execute_one(plan, true);
  if (plan.cvqahdr.selected) execute_one(plan, false);
}

std::string report_value(const BackendReport& report, const char* key) {
  const std::string* v = report.find(key);
  return v ? *v : "unknown";
}

void print_learned_summary(const LearnedPlan& plan) {
  if (plan.cvvdp.selected) {
    const BackendOutcome& r = plan.cvvdp_result;
    if (r.state == BackendState::Completed) {
      std::printf("[ CVVDP jod:%s frames:%s display:%s interpretation:%s device:%s ]\n",
                  report_value(r.report, "jod").c_str(),
                  report_value(r.report, "frames_evaluated").c_str(),
                  report_value(r.report, "display").c_str(),
                  report_value(r.report, "interpretation").c_str(),
                  report_value(r.report, "device").c_str());
    } else {
      std::printf("[ CVVDP status:failed ]\n");
    }
  }
  if (plan.cvqahdr.selected) {
    const BackendOutcome& r = plan.cvqahdr_result;
    if (r.state == BackendState::Completed) {
      double density = 0.0;
      const std::string density_text = report_value(r.report, "sampling_density_hz");
      char density_buf[32] = "unknown";
      if (parse_full_double(density_text, density)) {
        std::snprintf(density_buf, sizeof(density_buf), "%.3f/s", density);
      }
      std::printf(
          "[ CVQA-HDR-NR quality_score:%s raw_model_score:%s samples:%s frames:%s density:%s "
          "device:%s ]\n",
          report_value(r.report, "quality_score").c_str(),
          report_value(r.report, "raw_model_score").c_str(),
          report_value(r.report, "sampled_frames").c_str(),
          report_value(r.report, "opencv_frame_count").c_str(), density_buf,
          report_value(r.report, "device").c_str());
    } else {
      std::printf("[ CVQA-HDR-NR status:failed ]\n");
    }
  }
}

void print_cvvdp_qcs(const LearnedPlan& plan) {
  QcsAccum acc{plan.opt.qcs_cvvdp_lo, plan.opt.qcs_cvvdp_hi, 0, 0};
  char lo[64], hi[64];
  format_qcs_bound(acc.lower, lo, sizeof(lo));
  format_qcs_bound(acc.upper, hi, sizeof(hi));
  const BackendOutcome& r = plan.cvvdp_result;
  const std::string frames = report_value(r.report, "temporal_frames");
  const std::string evaluated = report_value(r.report, "frames_evaluated");
  const char* reason = nullptr;
  if (r.state != BackendState::Completed) reason = "backend-failed";
  else if (frames == "unknown") reason = "no-frame-sequence";
  else if (report_value(r.report, "pooled_parity") != "1") reason = "pooled-parity-mismatch";
  else if (frames == "1") reason = "single-frame";
  else if (r.report.samples.empty() || frames != std::to_string(r.report.samples.size()) ||
           (evaluated != "unknown" && evaluated != frames))
    reason = "frame-count-mismatch";
  if (reason) {
    std::printf("[ CVVDP-QCS value:unavailable range:[%s,%s] reason:%s ]\n", lo, hi, reason);
    return;
  }
  const char* qcs_trace = pix_getenv("QCS_TRACE");
  const bool trace = qcs_trace && qcs_trace[0] == '1';
  for (const BackendSample& sample : r.report.samples) {
    acc.add(sample.calibrated);
    if (trace) std::fprintf(stderr, "qcs_sample cvvdp %.17g\n", sample.calibrated);
  }
  if (!acc.available()) {
    std::printf("[ CVVDP-QCS value:unavailable range:[%s,%s] reason:no-finite-samples ]\n", lo, hi);
    return;
  }
  std::printf("[ CVVDP-QCS value:%.6f range:[%s,%s] frames:%zu basis:per-frame-jod-equivalent ]\n",
              acc.percent(), lo, hi, r.report.samples.size());
  std::fprintf(stderr,
               "pixvqa: CVVDP-QCS integrates met2jod of each frame's pooled distortion from the "
               "official full-video pass; this per-frame JOD-equivalent extends the published "
               "video-level JOD\n");
}

void print_learned_qcs(const LearnedPlan& plan) {
  if (plan.cvvdp.selected) print_cvvdp_qcs(plan);
  if (!plan.cvqahdr.selected) return;
  QcsAccum acc{plan.opt.qcs_cvqahdr_nr_lo, plan.opt.qcs_cvqahdr_nr_hi, 0, 0};
  char lo[64], hi[64];
  format_qcs_bound(acc.lower, lo, sizeof(lo));
  format_qcs_bound(acc.upper, hi, sizeof(hi));
  const BackendOutcome& r = plan.cvqahdr_result;
  const char* reason = nullptr;
  if (r.state != BackendState::Completed) reason = "backend-failed";
  else if (report_value(r.report, "pooled_parity") != "1") reason = "pooled-parity-mismatch";
  else if (report_value(r.report, "padded_samples") != "0") reason = "padded-samples";
  else if (r.report.samples.empty() ||
           report_value(r.report, "sampled_frames") != std::to_string(r.report.samples.size()))
    reason = "sample-count-mismatch";
  if (reason) {
    std::printf("[ CVQA-HDR-NR-QCS value:unavailable range:[%s,%s] reason:%s ]\n", lo, hi, reason);
    return;
  }
  const char* qcs_trace = pix_getenv("QCS_TRACE");
  const bool trace = qcs_trace && qcs_trace[0] == '1';
  for (const BackendSample& sample : r.report.samples) {
    acc.add(sample.calibrated);
    if (trace) std::fprintf(stderr, "qcs_sample cvqa-hdr-nr %.17g\n", sample.calibrated);
  }
  if (!acc.available()) {
    std::printf("[ CVQA-HDR-NR-QCS value:unavailable range:[%s,%s] reason:no-finite-samples ]\n",
                lo, hi);
    return;
  }
  std::printf("[ CVQA-HDR-NR-QCS value:%.6f range:[%s,%s] samples:%zu basis:sampled-frames ]\n",
              acc.percent(), lo, hi, r.report.samples.size());
  std::fprintf(stderr,
               "pixvqa: CVQA-HDR-NR-QCS uses %zu sampled frames (about one per second), not every "
               "frame; per-sample calibration extends the published video-level model\n",
               r.report.samples.size());
}

int learned_exit_status(const LearnedPlan& plan) {
  if (plan.cvvdp.selected && plan.cvvdp_result.state != BackendState::Completed) return 1;
  if (plan.cvqahdr.selected && plan.cvqahdr_result.state != BackendState::Completed) return 1;
  return 0;
}

// Learned metrics without native metrics: a text report like the native summary.
int run_learned_report(LearnedPlan& plan) {
  if (preflight_learned(plan) != 0) return 2;
  print_banner();
  if (!plan.opt.ref_path.empty()) std::printf("%s\n", plan.opt.ref_path.c_str());
  std::printf("%s\n\n", plan.opt.test_path.c_str());
  std::printf("%s learned-metric evaluation - v%s\n\n", PIXVQA_PRODUCT_NAME, PIXVQA_VERSION);
  execute_learned(plan);
  std::printf("\n\nSummary:\n\n");
  print_learned_summary(plan);
  if (plan.opt.enable_qcs) print_learned_qcs(plan);
  std::printf("\n");
  return learned_exit_status(plan);
}

// Set when native metrics run together with learned metrics.
std::unique_ptr<LearnedPlan> g_learned;

bool prepare_learned_with_native(LearnedPlan& plan, const SideSpec& ref, const SideSpec& test) {
  if (plan.cvvdp.selected) {
    if (ref.compressed != test.compressed) {
      std::fprintf(stderr,
                   "pixvqa: cvvdp needs both inputs as raw YUV or both as compressed video\n");
      return false;
    }
    if (!ref.compressed) {
      if (!prepare_cvvdp_raw(ref, test, plan.opt, plan.cvvdp)) return false;
      plan.cvvdp_raw = true;
      plan.raw_width = ref.format.width;
      plan.raw_height = ref.format.height;
    }
  }
  if (plan.cvqahdr.selected && !test.compressed) {
    std::fprintf(stderr,
                 "pixvqa: cvqa-hdr-nr supports compressed video only; the test input is raw YUV\n");
    return false;
  }
  return true;
}

bool parse_device_choice(const std::string& v, DeviceChoice& out) {
  if (v == "auto") out = DeviceChoice::Auto;
  else if (v == "cpu") out = DeviceChoice::Cpu;
  else if (v == "gpu") out = DeviceChoice::Gpu;
  else return false;
  return true;
}

bool gpu_metrics_built() {
#ifdef PIXVQA_HAS_CUDA
  return true;
#else
  return false;
#endif
}

bool cpu_decoder_built() {
#ifdef PIXVQA_HAS_FFMPEG
  return true;
#else
  return false;
#endif
}

bool gpu_decoder_built() {
#ifdef PIXVQA_HAS_NVDEC
  return true;
#else
  return false;
#endif
}

// Software HEVC was frozen at 8 frame threads after the pixel oracle.
// Each software decoder keeps that pool. Do not retune it from --threads.
constexpr int kSoftwareDecoderThreads = 8;

#ifndef PIXVQA_HAS_FFMPEG
bool ends_with_ci(const std::string& path, const char* ext) {
  const size_t n = std::strlen(ext);
  if (path.size() < n) return false;
  for (size_t i = 0; i < n; ++i) {
    const unsigned char a = static_cast<unsigned char>(path[path.size() - n + i]);
    const unsigned char b = static_cast<unsigned char>(ext[i]);
    if (std::tolower(a) != std::tolower(b)) return false;
  }
  return true;
}

bool annexb_hevc_prefix(const unsigned char* b, std::streamsize n) {
  int off = -1;
  if (n >= 4 && b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) off = 4;
  else if (n >= 3 && b[0] == 0 && b[1] == 0 && b[2] == 1) off = 3;
  if (off < 0 || off >= n) return false;
  const unsigned nal = b[static_cast<size_t>(off)];
  if ((nal & 0x80) != 0) return false;
  const unsigned type = (nal >> 1) & 0x3F;
  switch (type) {
    case 0:
    case 1:
    case 19:
    case 20:
    case 21:
    case 32:
    case 33:
    case 34:
    case 35:
    case 39:
      return true;
    default:
      return false;
  }
}

bool file_looks_compressed(const std::string& path, DeviceChoice request) {
  // FIFOs are raw YUV. Opening one here would consume it or block the writer.
  struct stat st;
  if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  unsigned char b[16] = {};
  in.read(reinterpret_cast<char*>(b), sizeof b);
  const std::streamsize n = in.gcount();
  if (n >= 8 && std::memcmp(b + 4, "ftyp", 4) == 0) return true;
  if (n >= 4 && b[0] == 0x1A && b[1] == 0x45 && b[2] == 0xDF && b[3] == 0xA3) return true;
  if (!annexb_hevc_prefix(b, n)) return false;
  // A planar YUV file can start with the same bytes as an HEVC start code.
  // Auto trusts that pattern only for an elementary-stream name. An explicit
  // cpu or gpu request also accepts it, because the user asked for a decoder.
  if (request != DeviceChoice::Auto) return true;
  return ends_with_ci(path, ".hevc") || ends_with_ci(path, ".h265") || ends_with_ci(path, ".265");
}
#endif

bool msssim_cpu_is_avx2() { return cpu_has_avx2(); }

int resolve_execution(Options& opt) {
  std::string ref_codec;
  std::string test_codec;
  auto classify = [&](const std::string& path, InputKind kind, const char* role, bool& compressed,
                      std::string& codec) -> int {
    codec.clear();
    if (kind == InputKind::Raw) {
      compressed = false;
      return 0;
    }
#ifdef PIXVQA_HAS_FFMPEG
    compressed = probe_video_input(path, &codec);
#else
    compressed = file_looks_compressed(path, opt.decode_request);
#endif
    if (kind == InputKind::Compressed && !compressed) {
      std::fprintf(stderr,
                   "pixvqa: %s: no video stream; --input compressed will not read it as raw YUV\n",
                   role);
      return 2;
    }
    return 0;
  };
  bool ref_compressed = false;
  bool test_compressed = false;
  if (classify(opt.ref_path, opt.ref_input, "reference", ref_compressed, ref_codec) != 0) return 2;
  if (classify(opt.test_path, opt.test_input, "test", test_compressed, test_codec) != 0) return 2;
  opt.ref_decode = DecodeMode::Raw;
  opt.decode = DecodeMode::Raw;
  // Software decoding follows the stream codec id. There is no codec or container allowlist.
  auto choose = [&](const std::string& codec, DecodeMode& mode) -> int {
    const bool av1 = codec == "av1";
    const bool vvc = codec == "vvc";
    if (opt.decode_request == DeviceChoice::Cpu) {
      if (!cpu_decoder_built()) {
        std::fprintf(stderr, "pixvqa: this build has no CPU decoder; refusing --decode cpu\n");
        return 2;
      }
      mode = DecodeMode::Software;
      return 0;
    }
    if (opt.decode_request == DeviceChoice::Gpu) {
      if (!gpu_decoder_built()) {
        std::fprintf(stderr, "pixvqa: GPU decoding is not compiled into this binary\n");
        return 2;
      }
      mode = DecodeMode::Nvdec;
      return 0;
    }
    if (vvc) {
      if (!cpu_decoder_built()) {
        std::fprintf(stderr, "pixvqa: VVC software decoding is not compiled into this binary\n");
        return 2;
      }
      mode = DecodeMode::Software;
      return 0;
    }
    if (av1) {
#ifdef PIXVQA_HAS_FFMPEG
      if (cpu_decoder_built() && av1_software_available()) {
        mode = DecodeMode::Software;
        return 0;
      }
#endif
      if (gpu_decoder_built()) {
        mode = DecodeMode::Nvdec;
        return 0;
      }
      std::fprintf(stderr, "pixvqa: AV1 software decoding requires libdav1d or libaom-av1\n");
      return 2;
    }
    if (cpu_decoder_built()) {
      mode = DecodeMode::Software;
      return 0;
    }
    if (gpu_decoder_built()) {
      mode = DecodeMode::Nvdec;
      return 0;
    }
    std::fprintf(stderr,
                 "pixvqa: compressed input requires a decoder; this build has none "
                 "(native metrics need make swdec or make full)\n");
    return 2;
  };
  if (ref_compressed && choose(ref_codec, opt.ref_decode) != 0) return 2;
  if (test_compressed && choose(test_codec, opt.decode) != 0) return 2;
  if ((opt.ref_decode == DecodeMode::Nvdec || opt.decode == DecodeMode::Nvdec) &&
      (opt.format.chroma != ChromaFormat::YUV420 ||
       (opt.format.bit_depth != 8 && opt.format.bit_depth != 10))) {
    std::fprintf(stderr, "pixvqa: --decode gpu supports 8-bit and 10-bit 4:2:0 only\n");
    return 2;
  }

  const bool cuda = gpu_metrics_built();
  const bool gpu_depth_ok = cuda && (opt.format.bit_depth == 8 || opt.format.bit_depth == 10);
  const bool psnr_ok = gpu_depth_ok;
  const bool ssim_ok = gpu_depth_ok;
  const bool msssim_ok = gpu_depth_ok;
  const bool xpsnr_ok = gpu_depth_ok;
  const bool pu21_ok = gpu_depth_ok;
  const bool resident_profile = cuda && opt.decode == DecodeMode::Nvdec && !opt.enable_vmaf &&
                                opt.format.bit_depth == 10 && opt.format.width >= 3840 &&
                                opt.format.height >= 2160;

  auto refuse_gpu = [](const char* metric, const char* why) {
    std::fprintf(stderr, "pixvqa: --metric-device gpu cannot run %s (%s); no CPU substitute\n",
                 metric, why);
  };

  opt.psnr_gpu = false;
  opt.ssim_gpu = false;
  opt.msssim_gpu = false;
  opt.pu21_gpu = false;
  opt.xpsnr_backend = XpsnrBackend::Cpu;
  opt.vmaf_cpu_notice = false;
  opt.vmaf_backend = VmafBackend::Cpu;

  if (opt.metric_device == DeviceChoice::Cpu) {
    // Every selected metric stays on the CPU.
  } else if (opt.metric_device == DeviceChoice::Gpu) {
    const bool other = opt.enable_psnr || opt.enable_ssim || opt.enable_msssim || opt.enable_xpsnr;
    const bool pu21 = pu21_selected(opt);
    // VMAF has no CUDA path. A GPU request still runs libvmaf on the CPU.
    // That does not require a CUDA build and does not substitute for another metric.
    const bool vmaf_only = opt.enable_vmaf && !other && !pu21;
    if (!cuda && !vmaf_only) {
      std::fprintf(stderr, "pixvqa: this build has no GPU metrics; refusing --metric-device gpu\n");
      return 2;
    }
    if (other) {
      if (opt.enable_psnr && !psnr_ok) {
        refuse_gpu("psnr", "GPU PSNR supports 8-bit and 10-bit only");
        return 2;
      }
      if (opt.enable_ssim && !ssim_ok) {
        refuse_gpu("ssim", "GPU SSIM supports 8-bit and 10-bit only");
        return 2;
      }
      if (opt.enable_msssim && !msssim_ok) {
        refuse_gpu("msssim", "GPU MS-SSIM supports 8-bit and 10-bit only");
        return 2;
      }
      if (opt.enable_xpsnr && !xpsnr_ok) {
        refuse_gpu("xpsnr", "GPU XPSNR supports 8-bit and 10-bit only");
        return 2;
      }
    }
    if (pu21) {
      if (!pu21_ok) {
        refuse_gpu("pu21", "GPU PU21 supports 8-bit and 10-bit only");
        return 2;
      }
      opt.pu21_gpu = true;
    }
    if (other) {
      opt.psnr_gpu = opt.enable_psnr;
      opt.ssim_gpu = opt.enable_ssim;
      opt.msssim_gpu = opt.enable_msssim;
      if (opt.enable_xpsnr) opt.xpsnr_backend = XpsnrBackend::Cuda;
    }
    opt.vmaf_cpu_notice = opt.enable_vmaf;
  } else if (resident_profile) {
    opt.psnr_gpu = opt.enable_psnr;
    opt.ssim_gpu = opt.enable_ssim;
    opt.msssim_gpu = opt.enable_msssim;
    if (opt.enable_xpsnr) opt.xpsnr_backend = XpsnrBackend::Cuda;
    if (pu21_selected(opt) && pu21_ok) opt.pu21_gpu = true;
  }

  opt.use_cuda = opt.psnr_gpu || opt.ssim_gpu || opt.msssim_gpu || opt.pu21_gpu ||
                 opt.xpsnr_backend == XpsnrBackend::Cuda;
  // Public GPU PSNR, SSIM, and MS-SSIM use FP64. PIXVQA_CUDA_PREC is not a control.
  opt.cuda_prec = CudaPrecision::Fp64;
  opt.kernel = cpu_has_avx2() ? SsimKernelMode::Avx2 : SsimKernelMode::Scalar;

  opt.logical_visible = logical_processors();
  int budget = 1;
  if (opt.logical_visible > 1) {
    budget = opt.logical_visible / 2;
    if (budget > 8) budget = 8;
    if (budget < 1) budget = 1;
  }
  if (opt.threads_auto) {
    const int software_inputs = (opt.ref_decode == DecodeMode::Software ? 1 : 0) +
                                (opt.decode == DecodeMode::Software ? 1 : 0);
    if (software_inputs > 0) {
      const int room = opt.logical_visible - kSoftwareDecoderThreads * software_inputs;
      opt.threads = room >= 1 ? std::min(budget, room) : 1;
    } else {
      opt.threads = budget;
    }
  } else if (opt.threads < 1) {
    std::fprintf(stderr, "pixvqa: --threads expects auto or a positive integer\n");
    return 2;
  }
  return 0;
}

const char* cpu_impl_label(bool avx2) { return avx2 ? "avx2" : "scalar"; }

const char* input_label(DecodeMode mode) {
  if (mode == DecodeMode::Software) return "cpu";
  if (mode == DecodeMode::Nvdec) return "gpu";
  return "raw";
}

const char* device_choice_name(DeviceChoice choice) {
  if (choice == DeviceChoice::Cpu) return "cpu";
  if (choice == DeviceChoice::Gpu) return "gpu";
  return "auto";
}

void print_execution_summary(const Options& opt) {
  std::printf("Execution:\n");
  std::printf("  requested device: %s\n", device_choice_name(opt.metric_device));
  std::printf("  reference: %s\n", input_label(opt.ref_decode));
  std::printf("  distorted: %s\n", input_label(opt.decode));
  if (opt.threads_auto) {
    std::printf(
        "  threads: %d (auto; %d logical processors visible; this budget is not an optimum)\n",
        opt.threads, opt.logical_visible);
  } else {
    std::printf("  threads: %d\n", opt.threads);
  }
  const bool ref_sw = opt.ref_decode == DecodeMode::Software;
  const bool test_sw = opt.decode == DecodeMode::Software;
  const bool ref_nv = opt.ref_decode == DecodeMode::Nvdec;
  const bool test_nv = opt.decode == DecodeMode::Nvdec;
  if (ref_sw || test_sw) {
    std::printf("  decoder threads: %d%s (fixed)\n", kSoftwareDecoderThreads,
                (ref_sw && test_sw) ? " per input" : "");
  }
  if (ref_nv || test_nv) {
    std::printf("  decoder threads: 1%s (fixed)\n", (ref_nv && test_nv) ? " per input" : "");
  }
  auto line = [](const char* name, bool gpu, const char* extra) {
    if (gpu) {
      std::printf("  %s: gpu%s%s\n", name, extra ? " " : "", extra ? extra : "");
    } else {
      std::printf("  %s: cpu%s%s\n", name, extra ? " " : "", extra ? extra : "");
    }
  };
  if (opt.enable_psnr) line("PSNR", opt.psnr_gpu, opt.psnr_gpu ? "fp64" : nullptr);
  if (opt.enable_xpsnr) line("XPSNR", opt.xpsnr_backend == XpsnrBackend::Cuda, nullptr);
  if (opt.enable_ssim) {
    line("SSIM", opt.ssim_gpu, opt.ssim_gpu ? "fp64" : cpu_impl_label(opt.kernel == SsimKernelMode::Avx2));
  }
  if (opt.enable_msssim) {
    line("MS-SSIM", opt.msssim_gpu,
         opt.msssim_gpu ? "fp64" : cpu_impl_label(msssim_cpu_is_avx2()));
  }
  if (opt.enable_vmaf) line("VMAF", false, "libvmaf");
  if (opt.enable_pu21_psnr) line("PU21-PSNR", opt.pu21_gpu, nullptr);
  if (opt.enable_pu21_ssim) line("PU21-SSIM", opt.pu21_gpu, nullptr);
  if (opt.enable_pu21_msssim) line("PU21-MS-SSIM", opt.pu21_gpu, nullptr);
  std::printf("\n");
}

bool pu21_selected(const Options& opt) {
  return opt.enable_pu21_psnr || opt.enable_pu21_ssim || opt.enable_pu21_msssim;
}

std::string take_string(bool side_set, const std::string& side_value, bool shared_set,
                        const std::string& shared_value, bool& out_set) {
  if (side_set) {
    out_set = true;
    return side_value;
  }
  if (shared_set) {
    out_set = true;
    return shared_value;
  }
  out_set = false;
  return {};
}

int resolve_prepared_side(const HdrSide& side, const Options& opt, bool reference, HdrEffective& out) {
  const HdrCli& one = reference ? opt.ref_hdr : opt.dist_hdr;
  HdrKnobs knobs;
  knobs.require_hdr_transfer = pu21_selected(opt);
  auto pick_double = [](bool side_set, double side_value, bool shared_set, double shared_value, bool& set,
                        double& value) {
    if (side_set) {
      set = true;
      value = side_value;
    } else if (shared_set) {
      set = true;
      value = shared_value;
    }
  };
  pick_double(one.peak_set, one.peak, opt.hdr.peak_set, opt.hdr.peak, knobs.peak_set, knobs.peak_nits);
  pick_double(one.black_set, one.black, opt.hdr.black_set, opt.hdr.black, knobs.black_set, knobs.black_nits);
  pick_double(one.gamma_set, one.gamma, opt.hdr.gamma_set, opt.hdr.gamma, knobs.gamma_set, knobs.gamma);
  if (!knobs.peak_set && opt.hlg_peak_set) {
    knobs.legacy_hlg_peak = true;
    knobs.legacy_hlg_peak_nits = opt.hlg_peak;
  }
  bool loc_set = false;
  const std::string loc =
      take_string(one.chroma_loc_set, one.chroma_loc, opt.hdr.chroma_loc_set, opt.hdr.chroma_loc, loc_set);
  if (loc_set) {
    if (!hdr_parse_chroma_loc(loc, knobs.chroma_loc)) {
      std::fprintf(stderr, "pixvqa: unsupported hdr chroma location %s\n", loc.c_str());
      return 2;
    }
    knobs.chroma_loc_set = true;
  }
  bool up_set = false;
  const std::string up =
      take_string(one.chroma_up_set, one.chroma_up, opt.hdr.chroma_up_set, opt.hdr.chroma_up, up_set);
  if (up_set) {
    if (!hdr_parse_chroma_up(up, knobs.chroma_up)) {
      std::fprintf(stderr, "pixvqa: unsupported hdr chroma reconstruction %s\n", up.c_str());
      return 2;
    }
    knobs.chroma_up_set = true;
  }
  const std::string resolved = hdr_resolve(side, knobs, out);
  if (!resolved.empty()) {
    std::fprintf(stderr, "pixvqa: HDR interpretation rejected: %s\n", resolved.c_str());
    std::fprintf(stderr, "pixvqa: %s\n", hdr_side_line(reference ? "reference" : "distorted", side).c_str());
    return 2;
  }
  return 0;
}

bool vmaf_color_explicit(HdrSource source) {
  return source == HdrSource::Stream || source == HdrSource::Override;
}

// The validated HDR VMAF tuple only. Assumptions filled by hdr_resolve are not accepted.
// spec is limited:bt2020:pq:bt2020nc when the side is explicit and complete.
std::string vmaf_hdr_color_spec(const char* role, const HdrSide& side, std::string& spec) {
  spec.clear();
  if (side.transfer == HdrTransfer::Hlg) {
    return std::string(role) +
           " HLG is unsupported and unvalidated for VMAF "
           "(libvmaf at 74f930d has no HLG transfer characteristic)";
  }
  if (side.transfer != HdrTransfer::Pq || !vmaf_color_explicit(side.transfer_source)) {
    return std::string("HDR VMAF model requires an explicit PQ transfer on the ") + role +
           " (a stream tag or --hdr-transfer pq). PQ is not inferred from 10-bit input";
  }
  if (side.range == HdrRange::Full) {
    return std::string(role) + " full-range PQ is unsupported and unvalidated for VMAF";
  }
  if (side.primaries != HdrPrimaries::Bt2020 || !vmaf_color_explicit(side.primaries_source) ||
      side.matrix != HdrMatrix::Bt2020Ncl || !vmaf_color_explicit(side.matrix_source) ||
      side.range != HdrRange::Limited || !vmaf_color_explicit(side.range_source)) {
    return std::string("HDR VMAF model requires explicit complete colorimetry on the ") + role +
           ": PQ, BT.2020 primaries, bt2020nc matrix, and limited range. "
           "Assumed fields are not used";
  }
  spec = "limited:bt2020:pq:bt2020nc";
  return {};
}

void warn_bundled_hdr_vmaf_conflicts(const char* role, const HdrSide& metadata,
                                     const HdrSide& configured) {
  const auto warn = [&](const char* source, const char* field, const char* value) {
    std::fprintf(
        stderr,
        "pixvqa: warning: HDR VMAF expects PQ / BT.2020 / BT.2020nc / limited. "
        "%s input %s %s is %s. Continuing without conversion.\n",
        role, source, field, value);
  };
  const auto metadata_value = [&](HdrSource source, bool matches, const char* field,
                                  const char* value, bool specified) {
    if (source == HdrSource::Stream && specified && !matches) warn("metadata", field, value);
  };
  metadata_value(metadata.transfer_source, metadata.transfer == HdrTransfer::Pq, "transfer",
                 hdr_transfer_name(metadata.transfer), metadata.transfer != HdrTransfer::Unspecified);
  metadata_value(metadata.primaries_source, metadata.primaries == HdrPrimaries::Bt2020, "primaries",
                 hdr_primaries_name(metadata.primaries),
                 metadata.primaries != HdrPrimaries::Unspecified);
  metadata_value(metadata.matrix_source, metadata.matrix == HdrMatrix::Bt2020Ncl, "matrix",
                 hdr_matrix_name(metadata.matrix), metadata.matrix != HdrMatrix::Unspecified);
  metadata_value(metadata.range_source, metadata.range == HdrRange::Limited, "range",
                 hdr_range_name(metadata.range), metadata.range != HdrRange::Unspecified);

  const auto explicit_value = [&](HdrSource source, bool matches, const char* field, const char* value) {
    if (source == HdrSource::Override && !matches) warn("explicit", field, value);
  };
  explicit_value(configured.transfer_source, configured.transfer == HdrTransfer::Pq, "transfer",
                 hdr_transfer_name(configured.transfer));
  explicit_value(configured.primaries_source, configured.primaries == HdrPrimaries::Bt2020, "primaries",
                 hdr_primaries_name(configured.primaries));
  explicit_value(configured.matrix_source, configured.matrix == HdrMatrix::Bt2020Ncl, "matrix",
                 hdr_matrix_name(configured.matrix));
  explicit_value(configured.range_source, configured.range == HdrRange::Limited, "range",
                 hdr_range_name(configured.range));
}

int report_hdr(const HdrSide& ref_stream, const HdrSide& dist_stream, const Options& opt, bool emit,
               HdrEffective* ref_out, HdrEffective* dist_out, std::string* vmaf_color_ref,
               std::string* vmaf_color_dist) {
  HdrEffective ref_eff;
  HdrEffective dist_eff;
  HdrSide ref = ref_stream;
  HdrSide dist = dist_stream;
  const bool ref_tf = opt.ref_transfer_set || opt.transfer_set || opt.ref_hdr.transfer_set || opt.hdr.transfer_set;
  const bool dist_tf = opt.dist_transfer_set || opt.transfer_set || opt.dist_hdr.transfer_set || opt.hdr.transfer_set;
  const bool ref_cs = opt.ref_color_set || opt.color_space_set;
  const bool dist_cs = opt.dist_color_set || opt.color_space_set;
  const std::string ref_tf_v = opt.ref_hdr.transfer_set
                                   ? opt.ref_hdr.transfer
                                   : (opt.ref_transfer_set ? opt.ref_transfer
                                                          : (opt.hdr.transfer_set ? opt.hdr.transfer : opt.transfer));
  const std::string dist_tf_v =
      opt.dist_hdr.transfer_set
          ? opt.dist_hdr.transfer
          : (opt.dist_transfer_set ? opt.dist_transfer
                                   : (opt.hdr.transfer_set ? opt.hdr.transfer : opt.transfer));
  const std::string ref_cs_v = opt.ref_color_set ? opt.ref_color_space : opt.color_space;
  const std::string dist_cs_v = opt.dist_color_set ? opt.dist_color_space : opt.color_space;
  const std::string ref_applied =
      hdr_apply_override(ref, ref_tf, ref_tf_v, ref_cs, ref_cs_v);
  if (!ref_applied.empty()) {
    std::fprintf(stderr, "pixvqa: %s\n", ref_applied.c_str());
    return 2;
  }
  const std::string dist_applied = hdr_apply_override(dist, dist_tf, dist_tf_v, dist_cs, dist_cs_v);
  if (!dist_applied.empty()) {
    std::fprintf(stderr, "pixvqa: %s\n", dist_applied.c_str());
    return 2;
  }
  auto apply_split = [](HdrSide& side, const HdrCli& cli) -> std::string {
    if (cli.primaries_set) {
      HdrPrimaries parsed;
      if (!hdr_parse_primaries_token(cli.primaries, parsed)) return "unsupported hdr primaries " + cli.primaries;
      side.primaries = parsed;
      side.primaries_source = HdrSource::Override;
      side.primaries_conflict = false;
    }
    if (cli.matrix_set) {
      HdrMatrix parsed;
      if (!hdr_parse_matrix_token(cli.matrix, parsed)) return "unsupported hdr matrix " + cli.matrix;
      side.matrix = parsed;
      side.matrix_source = HdrSource::Override;
      side.matrix_conflict = false;
    }
    if (cli.range_set) {
      HdrRange parsed;
      if (!hdr_parse_range_token(cli.range, parsed)) return "unsupported hdr range " + cli.range;
      side.range = parsed;
      side.range_source = HdrSource::Override;
      side.range_conflict = false;
    }
    return {};
  };
  // A per-input field outranks that side's --cs bundle, which outranks a shared
  // field, which outranks a shared --cs bundle. The bundle was applied above.
  HdrCli ref_split;
  HdrCli dist_split;
  auto take_field = [](bool side_set, const std::string& side_value, bool side_bundle, bool shared_set,
                       const std::string& shared_value, bool& out_set, std::string& out_value) {
    if (side_set) {
      out_set = true;
      out_value = side_value;
    } else if (!side_bundle && shared_set) {
      out_set = true;
      out_value = shared_value;
    }
  };
  take_field(opt.ref_hdr.primaries_set, opt.ref_hdr.primaries, opt.ref_color_set, opt.hdr.primaries_set,
             opt.hdr.primaries, ref_split.primaries_set, ref_split.primaries);
  take_field(opt.ref_hdr.matrix_set, opt.ref_hdr.matrix, opt.ref_color_set, opt.hdr.matrix_set, opt.hdr.matrix,
             ref_split.matrix_set, ref_split.matrix);
  take_field(opt.ref_hdr.range_set, opt.ref_hdr.range, opt.ref_color_set, opt.hdr.range_set, opt.hdr.range,
             ref_split.range_set, ref_split.range);
  take_field(opt.dist_hdr.primaries_set, opt.dist_hdr.primaries, opt.dist_color_set, opt.hdr.primaries_set,
             opt.hdr.primaries, dist_split.primaries_set, dist_split.primaries);
  take_field(opt.dist_hdr.matrix_set, opt.dist_hdr.matrix, opt.dist_color_set, opt.hdr.matrix_set,
             opt.hdr.matrix, dist_split.matrix_set, dist_split.matrix);
  take_field(opt.dist_hdr.range_set, opt.dist_hdr.range, opt.dist_color_set, opt.hdr.range_set, opt.hdr.range,
             dist_split.range_set, dist_split.range);
  const std::string ref_split_err = apply_split(ref, ref_split);
  if (!ref_split_err.empty()) {
    std::fprintf(stderr, "pixvqa: %s\n", ref_split_err.c_str());
    return 2;
  }
  const std::string dist_split_err = apply_split(dist, dist_split);
  if (!dist_split_err.empty()) {
    std::fprintf(stderr, "pixvqa: %s\n", dist_split_err.c_str());
    return 2;
  }
  if (opt.ref_hdr.chroma_loc_set || opt.hdr.chroma_loc_set) {
    ref.chroma_loc_conflict = false;
    ref.chroma_loc_unsupported = false;
  }
  if (opt.dist_hdr.chroma_loc_set || opt.hdr.chroma_loc_set) {
    dist.chroma_loc_conflict = false;
    dist.chroma_loc_unsupported = false;
  }
  const std::string why = hdr_reject_reason(ref, dist, opt.hlg_peak_set, opt.hlg_peak);
  if (!why.empty()) {
    std::fprintf(stderr, "pixvqa: HDR interpretation rejected: %s\n", why.c_str());
    std::fprintf(stderr, "pixvqa: %s\n", hdr_side_line("reference", ref).c_str());
    std::fprintf(stderr, "pixvqa: %s\n", hdr_side_line("distorted", dist).c_str());
    return 2;
  }
  if (resolve_prepared_side(ref, opt, true, ref_eff) != 0) return 2;
  if (resolve_prepared_side(dist, opt, false, dist_eff) != 0) return 2;
#ifdef PIXVQA_HAS_VMAF
  bool hdr_model = false;
  bool bundled_hdr_model = false;
  if (opt.enable_vmaf && !opt.vmaf_model_path.empty()) {
    try {
      const std::string model_path = vmaf_resolve_model_path(opt.vmaf_model_path);
      hdr_model = vmaf_inspect_model_file(model_path).has_conversion;
      bundled_hdr_model = vmaf_is_bundled_hdr_model_file(model_path);
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "pixvqa: %s\n", ex.what());
      return 2;
    }
  }
  if (opt.enable_vmaf && bundled_hdr_model) {
    if (opt.format.chroma != ChromaFormat::YUV420 || opt.format.bit_depth != 10) {
      std::fprintf(stderr, "pixvqa: HDR VMAF supports limited-range YUV420p10 only\n");
      return 2;
    }
    // These model-specific labels are passed only to libvmaf. ref_eff and
    // dist_eff retain the independently resolved configuration used by PU21.
    if (!emit) {
      warn_bundled_hdr_vmaf_conflicts("Reference", ref_stream, ref);
      warn_bundled_hdr_vmaf_conflicts("Distorted", dist_stream, dist);
    }
    if (vmaf_color_ref) *vmaf_color_ref = "limited:bt2020:pq:bt2020nc";
    if (vmaf_color_dist) *vmaf_color_dist = "limited:bt2020:pq:bt2020nc";
  } else if (opt.enable_vmaf && hdr_model) {
    if (opt.format.chroma != ChromaFormat::YUV420 || opt.format.bit_depth != 10) {
      std::fprintf(stderr, "pixvqa: HDR VMAF supports limited-range YUV420p10 only\n");
      return 2;
    }
    std::string ref_spec;
    std::string dist_spec;
    const std::string ref_err = vmaf_hdr_color_spec("reference", ref, ref_spec);
    if (!ref_err.empty()) {
      std::fprintf(stderr, "pixvqa: %s\n", ref_err.c_str());
      return 2;
    }
    const std::string dist_err = vmaf_hdr_color_spec("distorted", dist, dist_spec);
    if (!dist_err.empty()) {
      std::fprintf(stderr, "pixvqa: %s\n", dist_err.c_str());
      return 2;
    }
    if (vmaf_color_ref) *vmaf_color_ref = ref_spec;
    if (vmaf_color_dist) *vmaf_color_dist = dist_spec;
  } else
#endif
  if (opt.enable_vmaf &&
      (ref_eff.side.transfer == HdrTransfer::Pq || ref_eff.side.transfer == HdrTransfer::Hlg ||
       dist_eff.side.transfer == HdrTransfer::Pq || dist_eff.side.transfer == HdrTransfer::Hlg)) {
    const HdrTransfer shown = (ref_eff.side.transfer == HdrTransfer::Pq || ref_eff.side.transfer == HdrTransfer::Hlg)
                                  ? ref_eff.side.transfer
                                  : dist_eff.side.transfer;
    std::fprintf(stderr,
                 "pixvqa: vmaf rejects HDR transfer %s "
                 "(SDR-trained VMAF v1 model; not a validated PQ/HLG score). "
                 "An explicit --tf sdr or --hdr-transfer sdr scores the stored samples as that model.\n",
                 hdr_transfer_name(shown));
    return 2;
  }
  if (ref_out) *ref_out = ref_eff;
  if (dist_out) *dist_out = dist_eff;
  if (!emit) return 0;
  std::printf("HDR:\n  %s\n  %s\n", hdr_side_line("reference", ref_eff.side).c_str(),
              hdr_side_line("distorted", dist_eff.side).c_str());
  for (const std::string& note : ref_eff.notices) std::printf("  notice: reference %s\n", note.c_str());
  for (const std::string& note : dist_eff.notices) std::printf("  notice: distorted %s\n", note.c_str());
  const std::string ref_display = hdr_display_text("reference", ref_eff);
  const std::string dist_display = hdr_display_text("distorted", dist_eff);
  if (!ref_display.empty()) std::printf("  %s\n", ref_display.c_str());
  if (!dist_display.empty()) std::printf("  %s\n", dist_display.c_str());
  if (!ref_display.empty() || !dist_display.empty()) {
    std::printf(
        "  ignored: mastering-display peak, MaxCLL, and MaxFALL are not the evaluation peak\n");
  }
  if (ref_eff.side.transfer != dist_eff.side.transfer &&
      (opt.enable_psnr || opt.enable_ssim || opt.enable_msssim || opt.enable_xpsnr)) {
    std::printf(
        "  conventional: PSNR, SSIM, MS-SSIM, and XPSNR compare stored sample codes and are "
        "not a light-domain comparison of these transfers\n");
  }
  if (pu21_selected(opt)) {
    std::printf(
        "  pu21: variant=banding_glare precision=binary64 domain=[0.005,10000] cd/m2 "
        "peak=256 normalization=100cd/m2->%.6f\n",
        pu21_encode(100.0));
  }
  std::printf("\n");
  return 0;
}

int parse_hdr_option(Options& opt, const std::string& flag, const std::string& value) {
  HdrCli* slot = nullptr;
  std::string body;
  if (flag.compare(0, 10, "--ref-hdr-") == 0) {
    slot = &opt.ref_hdr;
    body = flag.substr(10);
  } else if (flag.compare(0, 11, "--dist-hdr-") == 0) {
    slot = &opt.dist_hdr;
    body = flag.substr(11);
  } else if (flag.compare(0, 6, "--hdr-") == 0) {
    slot = &opt.hdr;
    body = flag.substr(6);
  } else {
    return 1;
  }
  if (value == "auto") return 0;
  auto bad = [&](const char* why) {
    std::fprintf(stderr, "pixvqa: %s %s\n", flag.c_str(), why);
    return 2;
  };
  if (body == "transfer") {
    HdrTransfer parsed;
    if (!hdr_parse_transfer(value, parsed)) return bad("expects auto, pq, hlg, sdr, or linear");
    slot->transfer_set = true;
    slot->transfer = value;
    return 0;
  }
  if (body == "primaries") {
    HdrPrimaries parsed;
    if (!hdr_parse_primaries_token(value, parsed)) return bad("expects auto, bt2020, or bt709");
    slot->primaries_set = true;
    slot->primaries = value;
    return 0;
  }
  if (body == "matrix") {
    HdrMatrix parsed;
    if (!hdr_parse_matrix_token(value, parsed)) return bad("expects auto, bt2020nc, or bt709");
    slot->matrix_set = true;
    slot->matrix = value;
    return 0;
  }
  if (body == "range") {
    HdrRange parsed;
    if (!hdr_parse_range_token(value, parsed)) return bad("expects auto, limited, or full");
    slot->range_set = true;
    slot->range = value;
    return 0;
  }
  if (body == "chroma-loc") {
    HdrChromaLoc parsed;
    if (!hdr_parse_chroma_loc(value, parsed)) return bad("expects auto, left, center, or topleft");
    slot->chroma_loc_set = true;
    slot->chroma_loc = value;
    return 0;
  }
  if (body == "chroma-up") {
    HdrChromaUp parsed;
    if (!hdr_parse_chroma_up(value, parsed)) return bad("expects bilinear or nearest");
    slot->chroma_up_set = true;
    slot->chroma_up = value;
    return 0;
  }
  char* end = nullptr;
  const double number = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || (end && *end) || !std::isfinite(number)) return bad("expects auto or a number");
  if (body == "peak-nits") {
    if (!(number > 0.0) || number > 10000.0) return bad("expects cd/m2 in (0, 10000]");
    slot->peak_set = true;
    slot->peak = number;
    return 0;
  }
  if (body == "black-nits") {
    if (number < 0.0 || number >= 10000.0) return bad("expects cd/m2 in [0, 10000)");
    slot->black_set = true;
    slot->black = number;
    return 0;
  }
  if (body == "hlg-gamma") {
    if (!(number > 0.0) || number > 10.0) return bad("expects a gamma in (0, 10]");
    slot->gamma_set = true;
    slot->gamma = number;
    return 0;
  }
  std::fprintf(stderr, "PARSE ERROR: Argument: %s\n", flag.c_str());
  std::fprintf(stderr, "             Couldn't find match for argument\n\n");
  return 1;
}

#ifdef PIXVQA_HAS_FFMPEG
std::string layout_phrase(const VideoFormat& fmt) {
  const char* chroma = "4:2:0";
  if (fmt.chroma == ChromaFormat::YUV422) chroma = "4:2:2";
  else if (fmt.chroma == ChromaFormat::YUV444) chroma = "4:4:4";
  return std::to_string(fmt.bit_depth) + "-bit " + chroma;
}
#endif

std::string rate_phrase(const SideSpec& side) {
  if (side.rate == FrameRateKind::Variable) {
    return "variable (nominal " + std::to_string(side.nominal_num) + "/" +
           std::to_string(side.nominal_den) + ", average " + std::to_string(side.fps_num) + "/" +
           std::to_string(side.fps_den) + ")";
  }
  if (side.rate == FrameRateKind::Constant) {
    return std::to_string(side.fps_num) + "/" + std::to_string(side.fps_den);
  }
  return "unavailable";
}

bool same_rate(const SideSpec& a, const SideSpec& b) {
  return static_cast<long long>(a.fps_num) * b.fps_den ==
         static_cast<long long>(b.fps_num) * a.fps_den;
}

int classify_side(const Options& opt, SideSpec& side, const char* role) {
  (void)opt;
  struct stat st {};
  if (stat(side.path.c_str(), &st) != 0) {
    std::fprintf(stderr, "pixvqa: cannot open %s file '%s': %s\n",
                 role, side.path.c_str(), std::strerror(errno));
    return 2;
  }
  // FIFOs are valid raw inputs; opening them here could block or consume data.
  if (S_ISREG(st.st_mode)) {
    FILE* file = std::fopen(side.path.c_str(), "rb");
    if (!file) {
      std::fprintf(stderr, "pixvqa: cannot open %s file '%s': %s\n",
                   role, side.path.c_str(), std::strerror(errno));
      return 2;
    }
    std::fclose(file);
  } else if (!S_ISFIFO(st.st_mode)) {
    std::fprintf(stderr, "pixvqa: %s input '%s' must be a regular file or FIFO\n",
                 role, side.path.c_str());
    return 2;
  }
  side.compressed = false;
  if (side.input == InputKind::Raw) return 0;
#ifdef PIXVQA_HAS_FFMPEG
  side.compressed = probe_video_input(side.path);
#else
  side.compressed = file_looks_compressed(side.path, opt.decode_request);
#endif
  if (side.input == InputKind::Compressed && !side.compressed) {
    std::fprintf(stderr,
                 "pixvqa: %s: no video stream; --input compressed will not read it as raw YUV\n",
                 role);
    return 2;
  }
  return 0;
}

int adopt_detected_layout(SideSpec& side, const char* role) {
  if (!side.compressed) {
    if (!side.has_size || !side.has_fmt) {
      std::fprintf(stderr, "pixvqa: %s is raw YUV and requires -s and -f\n", role);
      return 2;
    }
    return 0;
  }
#ifndef PIXVQA_HAS_FFMPEG
  if (!side.has_size || !side.has_fmt) {
    std::fprintf(stderr,
                 "pixvqa: %s '%s' is compressed video, but this build has no FFmpeg decoder for "
                 "native metrics; rebuild with make swdec or make full, or pass raw YUV "
                 "(cvvdp and cvqa-hdr-nr decode in their own Python backends)\n",
                 role, side.path.c_str());
    return 2;
  }
  return 0;
#else
  CompressedInfo info;
  std::string err;
  if (!detect_compressed_info(side.path, info, err)) {
    std::fprintf(stderr, "pixvqa: %s: %s\n", role, err.c_str());
    return 2;
  }
  if (!info.size_known || !info.layout_known) {
    std::fprintf(stderr, "pixvqa: %s: could not detect resolution and pixel format\n", role);
    return 2;
  }
  if (side.has_size &&
      (side.format.width != info.format.width || side.format.height != info.format.height)) {
    std::fprintf(stderr, "pixvqa: %s -s asks for %dx%d, decoded input is %dx%d\n", role,
                 side.format.width, side.format.height, info.format.width, info.format.height);
    return 2;
  }
  if (side.has_fmt && (side.format.bit_depth != info.format.bit_depth ||
                       side.format.chroma != info.format.chroma)) {
    std::fprintf(stderr, "pixvqa: %s -f asks for %s, decoded input is %s\n", role,
                 layout_phrase(side.format).c_str(), layout_phrase(info.format).c_str());
    return 2;
  }
  side.format = info.format;
  side.has_size = true;
  side.has_fmt = true;
  side.rate = info.rate;
  side.fps_num = info.fps_num;
  side.fps_den = info.fps_den;
  side.nominal_num = info.nominal_num;
  side.nominal_den = info.nominal_den;
  return 0;
#endif
}

int prepare_formats(Options& opt, SideSpec& ref, SideSpec& test) {
  if (classify_side(opt, ref, "reference") != 0) return 2;
  if (classify_side(opt, test, "test") != 0) return 2;
  if (adopt_detected_layout(ref, "reference") != 0) return 2;
  if (adopt_detected_layout(test, "test") != 0) return 2;
  return 0;
}

int adopt_frame_rate(Options& opt, const SideSpec& ref, const SideSpec& test) {
  if (!opt.enable_xpsnr || opt.fps_set) return 0;
  if (!ref.compressed && !test.compressed) {
    std::fprintf(stderr,
                 "pixvqa: xpsnr requires --fps (raw YUV has no frame rate; "
                 "example: --fps 60 or --fps 60000/1001)\n");
    return 2;
  }
  const bool bad_ref = ref.compressed && ref.rate != FrameRateKind::Constant;
  const bool bad_test = test.compressed && test.rate != FrameRateKind::Constant;
  if (bad_ref || bad_test || (ref.compressed && test.compressed && !same_rate(ref, test))) {
    std::fprintf(stderr,
                 "pixvqa: xpsnr requires --fps (reference frame rate is %s; "
                 "test frame rate is %s)\n",
                 ref.compressed ? rate_phrase(ref).c_str() : "absent on raw YUV",
                 test.compressed ? rate_phrase(test).c_str() : "absent on raw YUV");
    return 2;
  }
  const SideSpec& source = ref.compressed ? ref : test;
  opt.fps_num = source.fps_num;
  opt.fps_den = source.fps_den;
  opt.fps_set = true;
  std::fprintf(stderr, "pixvqa: xpsnr using frame rate %d/%d from the compressed input\n",
               opt.fps_num, opt.fps_den);
  return 0;
}

}  // namespace

namespace {

int run_cli(int argc, char** argv) {
  Options opt;
  CvvdpOptions cvvdp;
  CvqaHdrNrOptions cvqahdr;
  SideSpec ref_side, test_side;
  SideSpec* bind = nullptr;
  // First option that only native metrics read; an error when no native metric is selected.
  std::string native_only_option;
  std::string learned_config_path;
  std::string metric_device_flag;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };

    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--help-hdr") {
      print_help_hdr();
      return 0;
    }
    if (a == "--version") {
      print_version_and_exit(argv[0]);
    }
    if ((a == "--input" || a == "--decode" || a == "--threads" || a == "-v" || a == "--model" ||
         a == "--vmaf-model" || a == "--cs" || a == "--tf" || a == "--ref-tf" || a == "--dist-tf" ||
         a == "--ref-cs" || a == "--dist-cs" || a == "--hlg-peak" || a == "--seek" ||
         a.compare(0, 6, "--hdr-") == 0 || a.compare(0, 10, "--ref-hdr-") == 0 ||
         a.compare(0, 11, "--dist-hdr-") == 0) &&
        native_only_option.empty()) {
      native_only_option = a;
    }

    if (a == "--metrics") {
      if (!parse_metrics_list(opt, cvvdp, cvqahdr, need("--metrics"))) return 2;
    } else if (a == "--preset") {
      const auto v = need("--preset");
      if (v != "hdr") {
        std::fprintf(stderr, "pixvqa: --preset expects hdr\n");
        return 2;
      }
      opt.preset_hdr = true;
    } else if (a == "--learned-config") {
      learned_config_path = need("--learned-config");
      if (learned_config_path.empty()) {
        std::fprintf(stderr, "pixvqa: --learned-config expects a path\n");
        return 2;
      }
    } else if (a == "--metric-device" || a == "--device") {
      const auto v = need(a.c_str());
      DeviceChoice choice = DeviceChoice::Auto;
      if (!parse_device_choice(v, choice)) {
        std::fprintf(stderr, "pixvqa: unsupported %s %s (use auto, cpu, or gpu)\n", a.c_str(),
                     v.c_str());
        return 2;
      }
      if (!metric_device_flag.empty() && choice != opt.metric_device) {
        std::fprintf(stderr, "pixvqa: conflicting %s %s and %s %s\n", metric_device_flag.c_str(),
                     device_choice_name(opt.metric_device), a.c_str(), v.c_str());
        return 2;
      }
      opt.metric_device = choice;
      metric_device_flag = a;
    } else if (a == "--cvvdp-interpretation") {
      cvvdp.options_seen = true;
      if (!assign_cvvdp_value(cvvdp.interpretation, "--cvvdp-interpretation",
                              need("--cvvdp-interpretation")))
        return 2;
      if (cvvdp.interpretation != "sdr" && cvvdp.interpretation != "pq" &&
          cvvdp.interpretation != "hlg") {
        std::fprintf(stderr, "pixvqa: --cvvdp-interpretation expects sdr, pq, or hlg\n");
        return 2;
      }
    } else if (a == "--cvvdp-display") {
      cvvdp.options_seen = true;
      if (!assign_cvvdp_value(cvvdp.display, "--cvvdp-display", need("--cvvdp-display"))) return 2;
    } else if (a == "--cvvdp-python") {
      cvvdp.options_seen = true;
      if (!assign_cvvdp_value(cvvdp.python, "--cvvdp-python", need("--cvvdp-python"))) return 2;
    } else if (a == "--cvvdp-bin") {
      cvvdp.options_seen = true;
      if (!assign_cvvdp_value(cvvdp.cvvdp_bin, "--cvvdp-bin", need("--cvvdp-bin"))) return 2;
    } else if (a == "--cvvdp-gpu-mem") {
      cvvdp.options_seen = true;
      const std::string v = need("--cvvdp-gpu-mem");
      double value = 0.0;
      if (!parse_full_double(v, value) || !(value > 0.0)) {
        std::fprintf(stderr, "pixvqa: --cvvdp-gpu-mem expects a positive number of GB\n");
        return 2;
      }
      if (cvvdp.gpu_mem_set && cvvdp.gpu_mem_gb != value) {
        std::fprintf(stderr, "pixvqa: conflicting --cvvdp-gpu-mem values\n");
        return 2;
      }
      cvvdp.gpu_mem_gb = value;
      cvvdp.gpu_mem_set = true;
    } else if (a == "--cvvdp-output-json") {
      cvvdp.options_seen = true;
      if (!assign_cvvdp_value(cvvdp.output_json, "--cvvdp-output-json",
                              need("--cvvdp-output-json")))
        return 2;
      cvvdp.output_json_set = true;
    } else if (a == "--cvqahdr-python") {
      cvqahdr.options_seen = true;
      if (!assign_cvqahdr_value(cvqahdr.python, "--cvqahdr-python", need("--cvqahdr-python")))
        return 2;
    } else if (a == "--cvqahdr-repo") {
      cvqahdr.options_seen = true;
      if (!assign_cvqahdr_value(cvqahdr.repository, "--cvqahdr-repo", need("--cvqahdr-repo")))
        return 2;
    } else if (a == "--cvqahdr-checkpoint") {
      cvqahdr.options_seen = true;
      if (!assign_cvqahdr_value(cvqahdr.checkpoint, "--cvqahdr-checkpoint",
                                need("--cvqahdr-checkpoint")))
        return 2;
    } else if (a == "--cvqahdr-profile") {
      cvqahdr.options_seen = true;
      if (!assign_cvqahdr_value(cvqahdr.profile, "--cvqahdr-profile", need("--cvqahdr-profile")))
        return 2;
    } else if (a == "--cvqahdr-output-json") {
      cvqahdr.options_seen = true;
      if (!assign_cvqahdr_value(cvqahdr.output_json, "--cvqahdr-output-json",
                                need("--cvqahdr-output-json")))
        return 2;
      cvqahdr.output_json_set = true;
    } else if (a == "--psnr" || a == "--ssim" || a == "--msssim" || a == "--xpsnr" || a == "--vmaf") {
      std::fprintf(stderr, "pixvqa: %s was removed; use --metrics psnr,xpsnr,ssim,msssim,vmaf\n",
                   a.c_str());
      return 2;
    } else if (a == "--kernel" || a == "--strip" || a == "--cuda-prec" || a == "--simd" ||
               a == "--xpsnr-backend" || a == "--vmaf-backend" || a == "--cuda-info") {
      std::fprintf(stderr,
                   "pixvqa: %s was removed; use --decode, --metric-device, and --threads\n",
                   a.c_str());
      return 2;
    } else if (a == "--qcs") {
      opt.enable_qcs = true;
    } else if (a == "--qcs-range") {
      if (!assign_qcs_range_list(opt, need("--qcs-range"))) return 2;
    } else if (a == "--model") {
      const auto v = need("--model");
      if (opt.vmaf_model_from_vmaf_model && opt.vmaf_model_path != v) {
        std::fprintf(stderr, "pixvqa: --model and --vmaf-model disagree (%s vs %s)\n", v.c_str(),
                     opt.vmaf_model_path.c_str());
        return 2;
      }
      opt.vmaf_model_path = v;
      opt.vmaf_model_from_model = true;
    } else if (a == "--vmaf-model") {
      const auto v = need("--vmaf-model");
      if (opt.vmaf_model_from_model && opt.vmaf_model_path != v) {
        std::fprintf(stderr, "pixvqa: --model and --vmaf-model disagree (%s vs %s)\n",
                     opt.vmaf_model_path.c_str(), v.c_str());
        return 2;
      }
      opt.vmaf_model_path = v;
      opt.vmaf_model_from_vmaf_model = true;
    } else if (a == "--decode") {
      const auto v = need("--decode");
      if (!parse_device_choice(v, opt.decode_request)) {
        std::fprintf(stderr, "pixvqa: unsupported --decode %s (use auto, cpu, or gpu)\n", v.c_str());
        return 2;
      }
    } else if (a == "--input") {
      const auto v = need("--input");
      InputKind kind = InputKind::Auto;
      if (v == "raw") kind = InputKind::Raw;
      else if (v == "compressed") kind = InputKind::Compressed;
      else if (v != "auto") {
        std::fprintf(stderr, "pixvqa: unsupported --input %s (use auto, raw, or compressed)\n",
                     v.c_str());
        return 2;
      }
      if (bind) bind->input = kind;
      else {
        ref_side.input = kind;
        test_side.input = kind;
      }
    } else if (a == "--fps") {
      const auto v = need("--fps");
      if (!parse_fps(v, opt.fps_num, opt.fps_den)) {
        std::fprintf(stderr,
                     "pixvqa: invalid --fps %s (use a positive integer or num/den, "
                     "for example 60 or 60000/1001)\n",
                     v.c_str());
        return 2;
      }
      opt.fps_set = true;
      mark_cvqahdr_raw_option(cvqahdr, a);
    } else if (a == "--threads") {
      const auto v = need("--threads");
      if (v == "auto") {
        opt.threads_auto = true;
      } else {
        char* end = nullptr;
        const long n = std::strtol(v.c_str(), &end, 10);
        if (end == v.c_str() || (end && *end) || n < 1 || n > 1000000) {
          std::fprintf(stderr, "pixvqa: --threads expects auto or a positive integer\n");
          return 2;
        }
        opt.threads_auto = false;
        opt.threads = static_cast<int>(n);
      }
    } else if (a == "-n" || a == "--frames") {
      long frames = 0;
      if (!parse_cli_int(need(a.c_str()), frames) || frames < 1 || frames > 100000000L) {
        std::fprintf(stderr, "pixvqa: --frames expects a positive integer\n");
        return 2;
      }
      opt.max_frames = static_cast<int>(frames);
    } else if (a == "-v") {
      long level = 0;
      if (!parse_cli_int(need("-v"), level) || level < 0 || level > 9) {
        std::fprintf(stderr, "pixvqa: -v expects an integer from 0 to 9\n");
        return 2;
      }
      opt.verbosity = static_cast<int>(level);
    } else if (a == "-r") {
      ref_side.path = need("-r");
      bind = &ref_side;
    } else if (a == "-t") {
      test_side.path = need("-t");
      bind = &test_side;
    } else if (a == "-s") {
      const auto s = need("-s");
      if (bind) apply_size(*bind, s);
      else apply_size_both(ref_side, test_side, s);
      mark_cvqahdr_raw_option(cvqahdr, a);
    } else if (a == "-f") {
      const auto s = need("-f");
      if (bind) apply_format(*bind, s);
      else apply_format_both(ref_side, test_side, s);
      mark_cvqahdr_raw_option(cvqahdr, a);
    } else if (a == "--cs") {
      const auto v = need("--cs");
      if (!known_cs(v)) {
        std::fprintf(stderr, "unsupported --cs %s\n", v.c_str());
        return 2;
      }
      opt.color_space = v;
      opt.color_space_set = true;
    } else if (a == "--tf") {
      const auto v = need("--tf");
      if (!known_tf(v)) {
        std::fprintf(stderr, "unsupported --tf %s\n", v.c_str());
        return 2;
      }
      opt.transfer = v;
      opt.transfer_set = true;
    } else if (a == "--ref-tf" || a == "--dist-tf") {
      const auto v = need(a.c_str());
      if (!known_tf(v)) {
        std::fprintf(stderr, "unsupported %s %s\n", a.c_str(), v.c_str());
        return 2;
      }
      if (a == "--ref-tf") {
        opt.ref_transfer = v;
        opt.ref_transfer_set = true;
      } else {
        opt.dist_transfer = v;
        opt.dist_transfer_set = true;
      }
    } else if (a == "--ref-cs" || a == "--dist-cs") {
      const auto v = need(a.c_str());
      if (!known_cs(v)) {
        std::fprintf(stderr, "unsupported %s %s\n", a.c_str(), v.c_str());
        return 2;
      }
      if (a == "--ref-cs") {
        opt.ref_color_space = v;
        opt.ref_color_set = true;
      } else {
        opt.dist_color_space = v;
        opt.dist_color_set = true;
      }
    } else if (a == "--hlg-peak") {
      const auto v = need("--hlg-peak");
      char* end = nullptr;
      const double peak = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || (end && *end) || !std::isfinite(peak) || !(peak > 0.0) || peak > 10000.0) {
        std::fprintf(stderr, "pixvqa: --hlg-peak expects cd/m2 in (0, 10000]\n");
        return 2;
      }
      opt.hlg_peak = peak;
      opt.hlg_peak_set = true;
    } else if (a.compare(0, 6, "--hdr-") == 0 || a.compare(0, 10, "--ref-hdr-") == 0 ||
               a.compare(0, 11, "--dist-hdr-") == 0) {
      const int hdr_rc = parse_hdr_option(opt, a, need(a.c_str()));
      if (hdr_rc == 1) {
        usage(argv[0]);
        return 1;
      }
      if (hdr_rc != 0) return hdr_rc;
    } else if (a == "--seek") {
      long seek = 0;
      if (!parse_cli_int(need("--seek"), seek)) {
        std::fprintf(stderr, "pixvqa: --seek expects an integer\n");
        return 2;
      }
      if (seek != 0) {
        std::fprintf(stderr, "unsupported: nonzero --seek\n");
        return 2;
      }
    } else if (a == "--wsssim" || a == "--wspsnr" || a == "--rfilter" || a == "--tfilter" ||
               a == "--geom" || a == "--mt" || a == "--fif") {
      std::fprintf(stderr, "unsupported argument in this build: %s\n", a.c_str());
      return 2;
    } else if (a == "--") {
      continue;
    } else {
      std::fprintf(stderr, "PARSE ERROR: Argument: %s\n", a.c_str());
      std::fprintf(stderr, "             Couldn't find match for argument\n\n");
      usage(argv[0]);
      return 1;
    }
  }

  opt.ref_path = ref_side.path;
  opt.test_path = test_side.path;
  opt.ref_input = ref_side.input;
  opt.test_input = test_side.input;
  if (!apply_processing_defaults(opt, cvvdp, cvqahdr)) return 2;
  const bool native = native_selected(opt);
  const bool learned = cvvdp.selected || cvqahdr.selected;
  if (cvvdp.options_seen && !cvvdp.selected) {
    std::fprintf(stderr, "pixvqa: --cvvdp-* options require --metrics cvvdp\n");
    return 2;
  }
  if (cvqahdr.options_seen && !cvqahdr.selected) {
    std::fprintf(stderr, "pixvqa: --cvqahdr-* options require --metrics cvqa-hdr-nr\n");
    return 2;
  }
  if ((opt.qcs_range_seen & kQcsCvqaHdrNrBit) != 0 && !cvqahdr.selected) {
    std::fprintf(stderr, "pixvqa: --qcs-range cvqa-hdr-nr requires cvqa-hdr-nr in --metrics\n");
    return 2;
  }
  if ((opt.qcs_range_seen & kQcsCvvdpBit) != 0 && !cvvdp.selected) {
    std::fprintf(stderr, "pixvqa: --qcs-range cvvdp requires cvvdp in --metrics\n");
    return 2;
  }
  if (!learned_config_path.empty() && !learned) {
    std::fprintf(stderr, "pixvqa: --learned-config requires cvvdp or cvqa-hdr-nr in --metrics\n");
    return 2;
  }
  if (learned) {
    const std::vector<const MetricCaps*> caps = selected_metric_caps(opt, cvvdp, cvqahdr);
    if (!native && !native_only_option.empty()) {
      std::fprintf(stderr,
                   "pixvqa: %s is incompatible with %s alone; it applies only to native metrics\n",
                   native_only_option.c_str(), cvvdp.selected ? "cvvdp" : "cvqa-hdr-nr");
      return 2;
    }
    for (const MetricCaps* c : caps) {
      if (opt.max_frames > 0 && !c->honors_frames) {
        std::fprintf(stderr,
                     "pixvqa: --frames is incompatible with %s; its official sampler reads the "
                     "whole clip\n",
                     c->name);
        return 2;
      }
    }
    if (cvqahdr.selected && !native && !cvqahdr.raw_option.empty()) {
      std::fprintf(stderr,
                   "pixvqa: %s is incompatible with cvqa-hdr-nr; it reads compressed video only "
                   "and needs no raw YUV geometry\n",
                   cvqahdr.raw_option.c_str());
      return 2;
    }
    if (opt.qcs_range_set && !opt.enable_qcs) {
      std::fprintf(stderr, "pixvqa: --qcs-range requires --qcs\n");
      return 2;
    }
    if (opt.test_path.empty()) {
      std::fprintf(stderr, "pixvqa: -t <distorted> is required\n");
      return 2;
    }
    const std::string full_reference = join_metric_names(caps, true);
    if (!full_reference.empty() && opt.ref_path.empty()) {
      std::fprintf(stderr, "pixvqa: -r <reference> is required by full-reference metrics: %s\n",
                   full_reference.c_str());
      return 2;
    }
    if (full_reference.empty() && !opt.ref_path.empty()) {
      std::fprintf(stderr, "pixvqa: info: cvqa-hdr-nr is no-reference; ignoring -r %s\n",
                   opt.ref_path.c_str());
      opt.ref_path.clear();
      ref_side = SideSpec();
    }
    if (cvqahdr.selected && !native && path_ends_with_yuv(opt.test_path)) {
      std::fprintf(stderr,
                   "pixvqa: cvqa-hdr-nr supports compressed video only; raw YUV is unsupported\n");
      return 2;
    }
    if (!resolve_learned_config(learned_config_path)) return 2;
    apply_learned_config(cvvdp, cvqahdr);
    if (cvvdp.selected && !validate_cvvdp_options(cvvdp)) return 2;
    if (cvqahdr.selected && !validate_cvqahdr_options(cvqahdr)) return 2;
    if (!native) {
      auto plan = std::make_unique<LearnedPlan>();
      plan->cvvdp = cvvdp;
      plan->cvqahdr = cvqahdr;
      if (cvvdp.selected) {
        inherit_side_geometry(ref_side, test_side);
        if (cvvdp_raw_requested(ref_side, test_side, opt)) {
          if (!prepare_cvvdp_raw(ref_side, test_side, opt, cvvdp)) return 2;
          plan->cvvdp_raw = true;
          plan->raw_width = ref_side.format.width;
          plan->raw_height = ref_side.format.height;
        }
      }
      plan->opt = opt;
      if (!(cvvdp.selected && cvqahdr.selected) && !opt.enable_qcs) {
        return run_learned_standalone(*plan);
      }
      return run_learned_report(*plan);
    }
    g_learned = std::make_unique<LearnedPlan>();
    g_learned->cvvdp = cvvdp;
    g_learned->cvqahdr = cvqahdr;
  }
  if (opt.transfer_set && opt.hdr.transfer_set) {
    std::fprintf(stderr, "pixvqa: --tf and --hdr-transfer both set\n");
    return 2;
  }
  if (opt.ref_transfer_set && opt.ref_hdr.transfer_set) {
    std::fprintf(stderr, "pixvqa: --ref-tf and --ref-hdr-transfer both set\n");
    return 2;
  }
  if (opt.dist_transfer_set && opt.dist_hdr.transfer_set) {
    std::fprintf(stderr, "pixvqa: --dist-tf and --dist-hdr-transfer both set\n");
    return 2;
  }
  if (opt.hlg_peak_set && opt.hdr.peak_set) {
    std::fprintf(stderr, "pixvqa: --hlg-peak and --hdr-peak-nits both set\n");
    return 2;
  }

  const char* cuda_info = pix_getenv("CUDA_INFO");
  if (cuda_info && cuda_info[0] == '1' && cuda_info[1] == '\0') {
#ifdef PIXVQA_HAS_CUDA
    cuda_print_device_info();
    return 0;
#else
    std::fprintf(stderr, "pixvqa: this build has no GPU metrics\n");
    return 2;
#endif
  }

  inherit_side_geometry(ref_side, test_side);

  print_banner();
  if (!opt.ref_path.empty()) std::printf("%s\n", opt.ref_path.c_str());
  if (!opt.test_path.empty()) std::printf("%s\n", opt.test_path.c_str());
  std::printf("\n");

  if (opt.ref_path.empty() || opt.test_path.empty()) {
    usage(argv[0]);
    return 2;
  }
  if (prepare_formats(opt, ref_side, test_side) != 0) return 2;
  if (!ref_side.has_size || !test_side.has_size || !ref_side.has_fmt || !test_side.has_fmt) {
    usage(argv[0]);
    return 2;
  }
  if (!formats_equal(ref_side.format, test_side.format)) {
    std::fprintf(stderr,
                 "pixvqa: reference and test -s/-f disagree "
                 "(%dx%d %s:%d vs %dx%d %s:%d)\n",
                 ref_side.format.width, ref_side.format.height, chroma_token(ref_side.format.chroma),
                 ref_side.format.bit_depth, test_side.format.width, test_side.format.height,
                 chroma_token(test_side.format.chroma), test_side.format.bit_depth);
    return 2;
  }
  opt.format = ref_side.format;

  if (opt.qcs_range_set && !opt.enable_qcs) {
    std::fprintf(stderr, "pixvqa: --qcs-range requires --qcs\n");
    return 2;
  }
  if (!opt.enable_psnr && !opt.enable_ssim && !opt.enable_msssim && !opt.enable_xpsnr &&
      !opt.enable_vmaf && !pu21_selected(opt)) {
    std::fprintf(stderr,
                 "pixvqa: --metrics is required (psnr, xpsnr, ssim, msssim, vmaf, "
                 "pu21-psnr, pu21-ssim, pu21-msssim)\n");
    return 2;
  }
  if (resolve_execution(opt) != 0) return 2;
  if (opt.vmaf_cpu_notice) {
    std::fprintf(stderr,
                 "pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf.\n");
  }
  if (opt.enable_vmaf) {
#ifndef PIXVQA_HAS_VMAF
    std::fprintf(stderr,
                 "pixvqa: this build has no libvmaf; refusing vmaf "
                 "(build deps with scripts/build_libvmaf.sh)\n");
    return 2;
#else
    if (opt.vmaf_model_path.empty()) {
      std::fprintf(stderr,
                   "pixvqa: vmaf requires --model <path-to-vmaf-model.json> "
                   "(example: --model model/vmaf_v1.0.16_3d0h.json)\n");
      return 2;
    }
    auto explicit_hdr = [](bool set, const std::string& token) {
      return set && (token == "st2084" || token == "stdb67" || token == "pq" || token == "hlg");
    };
    const bool explicit_hdr_flag =
        explicit_hdr(opt.transfer_set, opt.transfer) || explicit_hdr(opt.ref_transfer_set, opt.ref_transfer) ||
        explicit_hdr(opt.dist_transfer_set, opt.dist_transfer) ||
        explicit_hdr(opt.hdr.transfer_set, opt.hdr.transfer) ||
        explicit_hdr(opt.ref_hdr.transfer_set, opt.ref_hdr.transfer) ||
        explicit_hdr(opt.dist_hdr.transfer_set, opt.dist_hdr.transfer);
    if (opt.format.bit_depth != 8 && opt.format.bit_depth != 10) {
      std::fprintf(stderr, "pixvqa: vmaf supports only 8-bit or 10-bit planar YUV\n");
      return 2;
    }
    try {
      const std::string resolved = vmaf_resolve_model_path(opt.vmaf_model_path);
      {
        std::ifstream probe(resolved);
        if (!probe) {
          std::fprintf(stderr, "pixvqa: VMAF model file not found or not readable: %s\n",
                       resolved.c_str());
          return 2;
        }
      }
      const bool hdr_model = vmaf_inspect_model_file(resolved).has_conversion;
      if (explicit_hdr_flag && !hdr_model) {
        std::fprintf(stderr,
                     "pixvqa: vmaf rejects explicit HDR transfer "
                     "(SDR-trained VMAF v1 model; not a validated PQ/HLG score)\n");
        return 2;
      }
      const std::string digest = vmaf_file_sha256(resolved);
      int min_w = 0, min_h = 0;
      if (vmaf_min_size_for_sha256(digest, min_w, min_h) &&
          (opt.format.width < min_w || opt.format.height < min_h)) {
        std::fprintf(stderr,
                     "pixvqa: VMAF model %s needs at least %dx%d luma "
                     "(got %dx%d; SpEED/cambi reject smaller frames)\n",
                     resolved.c_str(), min_w, min_h, opt.format.width, opt.format.height);
        return 2;
      }
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "pixvqa: %s\n", ex.what());
      return 2;
    }
#endif
  } else if (!opt.vmaf_model_path.empty()) {
    std::fprintf(stderr, "pixvqa: --model and --vmaf-model require vmaf in --metrics\n");
    return 2;
  }
  if (adopt_frame_rate(opt, ref_side, test_side) != 0) return 2;
  if (opt.enable_xpsnr && !xpsnr_geometry_supported(opt.format.width, opt.format.height)) {
    std::fprintf(stderr,
                 "pixvqa: XPSNR does not support %dx%d (odd luma edge above 2048*1152 "
                 "samples; the downsampled high-pass would read past the plane)\n",
                 opt.format.width, opt.format.height);
    return 2;
  }
  if (opt.enable_msssim && opt.msssim_gpu && opt.cuda_prec != CudaPrecision::Fp64) {
    const char* mode = opt.cuda_prec == CudaPrecision::Mixed ? "mixed" : "fp32";
    std::fprintf(stderr,
                 "pixvqa: warning: GPU MS-SSIM is using %s. "
                 "FP64 is the validated MS-SSIM precision. "
                 "FP32 and mixed precision can deviate on flat or nearly flat images.\n",
                 mode);
  }
  if (g_learned) {
    g_learned->opt = opt;
    if (!prepare_learned_with_native(*g_learned, ref_side, test_side)) return 2;
    if (preflight_learned(*g_learned) != 0) return 2;
  }
  opt.peak = (1 << opt.format.bit_depth) - 1;
  const double L = opt.peak;
  const int bd = opt.format.bit_depth;

  try {
    YuvReader ref_rd, test_rd;
#ifdef PIXVQA_HAS_FFMPEG
    CompressedTestReader ref_dec, test_dec;
#endif
    try {
      if (opt.ref_decode == DecodeMode::Raw) {
        ref_rd.open(opt.ref_path, opt.format);
      }
#ifdef PIXVQA_HAS_FFMPEG
      else {
        std::string dec_err;
        if (!ref_dec.open(opt.ref_path, opt.format, opt.ref_decode, dec_err, "reference")) {
          std::fprintf(stderr, "pixvqa: %s\n", dec_err.c_str());
          return ref_dec.status() == 0 ? 2 : ref_dec.status();
        }
      }
#endif
      if (opt.decode == DecodeMode::Raw) {
        test_rd.open(opt.test_path, opt.format);
      }
#ifdef PIXVQA_HAS_FFMPEG
      else {
        std::string dec_err;
        if (!test_dec.open(opt.test_path, opt.format, opt.decode, dec_err, "test")) {
          std::fprintf(stderr, "pixvqa: %s\n", dec_err.c_str());
          return test_dec.status() == 0 ? 2 : test_dec.status();
        }
      }
#endif
    } catch (const std::exception& ex) {
      const std::string msg = ex.what();
      if (msg.find("failed to open") != std::string::npos) {
        std::printf("failed to open Input YUV file\n");
        return 1;
      }
      throw;
    }

    std::printf("%s image/video comparison - v%s\n\n", PIXVQA_PRODUCT_NAME, PIXVQA_VERSION);
    std::printf("Input: %dx%d %s (%db,%db,%db) %s %s\n", opt.format.width, opt.format.height,
                chroma_token(opt.format.chroma), bd, bd, bd, opt.color_space.c_str(),
                transfer_display(opt.transfer));
    std::printf("  %s\n  %s\n\n", opt.ref_path.c_str(), opt.test_path.c_str());
    std::fflush(stdout);

    Frame ref_fr, test_fr;
    double sum_mse_y = 0, sum_mse_u = 0, sum_mse_v = 0;
    PlaneScore sum_ssim{}, sum_msssim{};
    int nframes = 0;
    XpsnrMeter xpsnr;
    const bool xpsnr_cuda = opt.enable_xpsnr && opt.xpsnr_backend == XpsnrBackend::Cuda;
    if (opt.enable_xpsnr) {
      const int xrate = xpsnr_integer_rate(opt.fps_num, opt.fps_den);
      xpsnr.start(opt.format, xrate, opt.threads);
#ifdef PIXVQA_HAS_CUDA
      if (xpsnr_cuda) cuda_xpsnr_reset(opt.format, xrate);
#endif
    }
    std::string vmaf_color_ref;
    std::string vmaf_color_dist;
    {
      HdrSide ref_meta;
      HdrSide dist_meta;
#ifdef PIXVQA_HAS_FFMPEG
      if (opt.ref_decode != DecodeMode::Raw) ref_meta = ref_dec.stream_hdr();
      if (opt.decode != DecodeMode::Raw) dist_meta = test_dec.stream_hdr();
#endif
      if (report_hdr(ref_meta, dist_meta, opt, false, nullptr, nullptr, &vmaf_color_ref,
                     &vmaf_color_dist) != 0)
        return 2;
    }
#ifdef PIXVQA_HAS_VMAF
    VmafMeter vmaf;
    // When VMAF is enabled, defer per-frame printing until after flush so
    // motion3-backed scores can appear on the same ordered frame lines.
    struct DeferredFrame {
      PsnrFrame psnr{};
      PlaneScore ssim{};
      PlaneScore msssim{};
      XpsnrFrame xframe{};
      bool has_xpsnr = false;
      Pu21FrameScore pu21{};
    };
    std::vector<DeferredFrame> deferred;
    if (opt.enable_vmaf) {
      vmaf.start(opt.format, opt.vmaf_model_path, opt.threads, vmaf_color_ref, vmaf_color_dist);
      print_vmaf_meta(vmaf.meta());
      if (opt.verbosity >= 1) deferred.reserve(64);
    }
#endif

    const char* prof_env = pix_getenv("PROFILE");
    const bool prof = prof_env && prof_env[0] == '1';
    double io_s = 0.0;
    double compute_s = 0.0;
    QcsAccum qcs_psnr{opt.qcs_psnr_lo, opt.qcs_psnr_hi, 0, 0};
    QcsAccum qcs_ssim{opt.qcs_ssim_lo, opt.qcs_ssim_hi, 0, 0};
    QcsAccum qcs_msssim{opt.qcs_msssim_lo, opt.qcs_msssim_hi, 0, 0};
    QcsAccum qcs_xpsnr{opt.qcs_xpsnr_lo, opt.qcs_xpsnr_hi, 0, 0};
    QcsAccum qcs_vmaf{opt.qcs_vmaf_lo, opt.qcs_vmaf_hi, 0, 0};
    QcsAccum qcs_pu21_psnr{opt.qcs_pu21_psnr_lo, opt.qcs_pu21_psnr_hi, 0, 0};
    QcsAccum qcs_pu21_ssim{opt.qcs_pu21_ssim_lo, opt.qcs_pu21_ssim_hi, 0, 0};
    QcsAccum qcs_pu21_msssim{opt.qcs_pu21_msssim_lo, opt.qcs_pu21_msssim_hi, 0, 0};
    const char* qcs_trace = pix_getenv("QCS_TRACE");
    const bool trace_qcs = qcs_trace && qcs_trace[0] == '1';
    print_execution_summary(opt);
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    bool hdr_reported = false;
    HdrEffective ref_hdr_eff;
    HdrEffective dist_hdr_eff;
    bool pu21_clamp_noted = false;
    double sum_pu_mse = 0.0;
    double sum_pu_ssim = 0.0;
    double sum_pu_msssim = 0.0;

    while (opt.max_frames < 0 || nframes < opt.max_frames) {
      const auto t_read = std::chrono::steady_clock::now();
      std::string dec_err;
      bool ok_r = false;
      if (opt.ref_decode == DecodeMode::Raw) {
        ok_r = ref_rd.read_frame(ref_fr);
      }
#ifdef PIXVQA_HAS_FFMPEG
      else {
        // The reference is always copied to the host. CUDA scoring has no
        // device-reference path, and the conversion buffer is shared.
        ok_r = ref_dec.read_frame(ref_fr, dec_err, true);
        if (!dec_err.empty()) {
          std::fprintf(stderr, "pixvqa: %s\n", dec_err.c_str());
          return ref_dec.status() == 0 ? 1 : ref_dec.status();
        }
      }
#endif
      bool ok_t = false;
      if (opt.decode == DecodeMode::Raw) {
        ok_t = test_rd.read_frame(test_fr);
      }
#ifdef PIXVQA_HAS_FFMPEG
      else {
        // A CPU metric, including VMAF, needs a host copy. GPU metrics can still
        // read the decoder planes when the distorted input is the GPU decoder.
        const bool host_metric = (opt.enable_psnr && !opt.psnr_gpu) ||
                                 (opt.enable_ssim && !opt.ssim_gpu) ||
                                 (opt.enable_msssim && !opt.msssim_gpu) ||
                                 (opt.enable_xpsnr && !xpsnr_cuda) || opt.enable_vmaf ||
                                 pu21_selected(opt);
        // 10-bit NVDEC stays on the P010 device path. 8-bit NVDEC is NV12 and
        // is deinterleaved on the host, then uploaded like any other host frame.
        const bool resident = opt.decode == DecodeMode::Nvdec && opt.use_cuda && !host_metric &&
                              opt.format.bit_depth == 10;
        ok_t = test_dec.read_frame(test_fr, dec_err, !resident);
        if (!dec_err.empty()) {
          std::fprintf(stderr, "pixvqa: %s\n", dec_err.c_str());
          return test_dec.status() == 0 ? 1 : test_dec.status();
        }
      }
#endif
      if (!ok_r && !ok_t) break;
      if (ok_r != ok_t) {
        std::fprintf(stderr, "ref/test frame count mismatch at frame %d\n", nframes);
        return 1;
      }
      if (!hdr_reported) {
        HdrSide ref_meta;
        HdrSide dist_meta;
#ifdef PIXVQA_HAS_FFMPEG
        if (opt.ref_decode != DecodeMode::Raw) ref_meta = ref_dec.stream_hdr();
        if (opt.decode != DecodeMode::Raw) dist_meta = test_dec.stream_hdr();
#endif
        if (report_hdr(ref_meta, dist_meta, opt, true, &ref_hdr_eff, &dist_hdr_eff, nullptr,
                       nullptr) != 0)
          return 2;
        hdr_reported = true;
      }
      const auto t_comp = std::chrono::steady_clock::now();

      PsnrFrame psnr{};
      PlaneScore ssim{}, msssim{};
      XpsnrFrame xframe;
#ifdef PIXVQA_HAS_CUDA
      const bool want_cuda = opt.psnr_gpu || opt.ssim_gpu || opt.msssim_gpu || xpsnr_cuda;
      if (want_cuda) {
        CudaXpsnrRequest xreq;
        if (xpsnr_cuda) xreq.integer_rate = xpsnr.integer_frame_rate();
        const char* score_device = pix_getenv("SCORE_DEVICE");
        if (score_device && score_device[0] == '1' && score_device[1] == '\0') {
          cuda_score_device_frame(&ref_fr, nullptr, &test_fr, nullptr, opt.format, opt.peak, opt.cuda_prec, opt.psnr_gpu, opt.ssim_gpu, opt.msssim_gpu, opt.psnr_gpu ? &psnr : nullptr, opt.ssim_gpu ? &ssim : nullptr, opt.msssim_gpu ? &msssim : nullptr, xpsnr_cuda ? &xreq : nullptr);
        } else if (score_device && std::strcmp(score_device, "pitch") == 0) {
          DevicePlanarFrame bad{};
          bad.y.width = opt.format.width;
          bad.y.height = opt.format.height;
          bad.y.pitch_samples = opt.format.width + 1;
          try {
            cuda_score_device_frame(&ref_fr, nullptr, nullptr, &bad, opt.format, opt.peak, opt.cuda_prec, opt.psnr_gpu, opt.ssim_gpu, opt.msssim_gpu, opt.psnr_gpu ? &psnr : nullptr, opt.ssim_gpu ? &ssim : nullptr, opt.msssim_gpu ? &msssim : nullptr, xpsnr_cuda ? &xreq : nullptr);
          } catch (const std::exception& ex) {
            std::fprintf(stderr, "pixvqa: %s\n", ex.what());
            return 2;
          }
          std::fprintf(stderr, "pixvqa: device pitch check did not fail\n");
          return 1;
        } else if (opt.decode == DecodeMode::Nvdec && opt.use_cuda && opt.format.bit_depth == 10) {
          const DevicePlanarFrame dev = cuda_metric_test_frame(opt.format);
          cuda_score_device_frame(&ref_fr, nullptr, nullptr, &dev, opt.format, opt.peak, opt.cuda_prec, opt.psnr_gpu, opt.ssim_gpu, opt.msssim_gpu, opt.psnr_gpu ? &psnr : nullptr, opt.ssim_gpu ? &ssim : nullptr, opt.msssim_gpu ? &msssim : nullptr, xpsnr_cuda ? &xreq : nullptr);
        } else {
          cuda_score_frame(ref_fr, test_fr, opt.format, opt.peak, opt.cuda_prec, opt.psnr_gpu, opt.ssim_gpu, opt.msssim_gpu, opt.psnr_gpu ? &psnr : nullptr, opt.ssim_gpu ? &ssim : nullptr, opt.msssim_gpu ? &msssim : nullptr, xpsnr_cuda ? &xreq : nullptr);
        }
        if (xpsnr_cuda) xframe = xpsnr.adopt_wsse(xreq.wsse_y, xreq.wsse_u, xreq.wsse_v);
        if (opt.psnr_gpu) {
          sum_mse_y += psnr.mse.y;
          sum_mse_u += psnr.mse.u;
          sum_mse_v += psnr.mse.v;
        }
        if (opt.ssim_gpu) {
          sum_ssim.y += ssim.y;
          sum_ssim.u += ssim.u;
          sum_ssim.v += ssim.v;
          sum_ssim.avg += ssim.avg;
        }
        if (opt.msssim_gpu) accumulate(sum_msssim, msssim);
      }
#endif
      if (opt.enable_psnr && !opt.psnr_gpu) {
        psnr = compute_psnr_frame(ref_fr, test_fr, opt.format, opt.peak);
        sum_mse_y += psnr.mse.y;
        sum_mse_u += psnr.mse.u;
        sum_mse_v += psnr.mse.v;
      }
      if (opt.enable_ssim && !opt.ssim_gpu) {
        ssim = compute_ssim_frame(ref_fr, test_fr, opt.format, L, opt.kernel, opt.threads);
        sum_ssim.y += ssim.y;
        sum_ssim.u += ssim.u;
        sum_ssim.v += ssim.v;
        sum_ssim.avg += ssim.avg;
      }
      if (opt.enable_msssim && !opt.msssim_gpu) {
        msssim = score_msssim(ref_fr, test_fr, opt.format, opt.peak, opt.threads);
        accumulate(sum_msssim, msssim);
      }

      Pu21FrameScore pu21{};
      if (pu21_selected(opt)) {
        std::string pu_err;
#ifdef PIXVQA_HAS_CUDA
        if (opt.pu21_gpu) {
          pu_err = cuda_score_pu21(ref_fr, test_fr, opt.format, ref_hdr_eff, dist_hdr_eff, opt.enable_pu21_psnr,
                                   opt.enable_pu21_ssim, opt.enable_pu21_msssim, pu21);
        } else
#endif
        pu_err = pu21_score_frame(ref_fr, test_fr, opt.format, ref_hdr_eff, dist_hdr_eff, opt.enable_pu21_psnr,
                                  opt.enable_pu21_ssim, opt.enable_pu21_msssim, opt.threads, pu21);
        if (!pu_err.empty()) {
          std::fprintf(stderr, "pixvqa: %s\n", pu_err.c_str());
          return 2;
        }
        if (pu21.clamped && !pu21_clamp_noted) {
          std::fprintf(stderr,
                       "pixvqa: PU21 clamped samples outside [0.005, 10000] cd/m2\n");
          pu21_clamp_noted = true;
        }
        if (opt.enable_pu21_psnr) sum_pu_mse += pu21.mse;
        if (opt.enable_pu21_ssim) sum_pu_ssim += pu21.ssim;
        if (opt.enable_pu21_msssim) sum_pu_msssim += pu21.msssim;
      }
      if (opt.enable_xpsnr && !xpsnr_cuda) {
        xframe = xpsnr.push(ref_fr, test_fr);
      }
      if (opt.enable_xpsnr) {
        const char* xdbg = pix_getenv("XPSNR_DEBUG");
        if (xdbg && xdbg[0] == '1') {
          std::fprintf(stderr, "xpsnr_wsse %d %llu %llu %llu\n", nframes,
                       static_cast<unsigned long long>(xframe.wsse_y),
                       static_cast<unsigned long long>(xframe.wsse_u),
                       static_cast<unsigned long long>(xframe.wsse_v));
        }
      }
#ifdef PIXVQA_HAS_VMAF
      if (opt.enable_vmaf) {
        vmaf.push(ref_fr, test_fr);
      }
#endif

      if (prof) {
        const auto t_done = std::chrono::steady_clock::now();
        io_s += std::chrono::duration<double>(t_comp - t_read).count();
        compute_s += std::chrono::duration<double>(t_done - t_comp).count();
      }
      if (opt.enable_qcs) {
        if (opt.enable_psnr) {
          qcs_psnr.add(psnr.psnr.avg);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample psnr %.17g\n", psnr.psnr.avg);
        }
        if (opt.enable_ssim) {
          qcs_ssim.add(ssim.avg);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample ssim %.17g\n", ssim.avg);
        }
        if (opt.enable_msssim) {
          qcs_msssim.add(msssim.avg);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample msssim %.17g\n", msssim.avg);
        }
        if (opt.enable_xpsnr) {
          qcs_xpsnr.add(xframe.avg);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample xpsnr %.17g\n", xframe.avg);
        }
        if (opt.enable_pu21_psnr) {
          qcs_pu21_psnr.add(pu21.psnr);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample pu21-psnr %.17g\n", pu21.psnr);
        }
        if (opt.enable_pu21_ssim) {
          qcs_pu21_ssim.add(pu21.ssim);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample pu21-ssim %.17g\n", pu21.ssim);
        }
        if (opt.enable_pu21_msssim) {
          qcs_pu21_msssim.add(pu21.msssim);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample pu21-msssim %.17g\n", pu21.msssim);
        }
      }
      if (opt.verbosity >= 1) {
#ifdef PIXVQA_HAS_VMAF
        if (opt.enable_vmaf) {
          DeferredFrame d;
          d.psnr = psnr;
          d.ssim = ssim;
          d.msssim = msssim;
          d.xframe = xframe;
          d.has_xpsnr = opt.enable_xpsnr;
          d.pu21 = pu21;
          deferred.push_back(d);
        } else
#endif
        {
          std::printf("%-3d:", nframes);
          if (opt.enable_psnr) print_psnr(psnr.psnr, psnr.mse);
          if (opt.enable_ssim) print_metric("SSIM", ssim);
          if (opt.enable_msssim) print_msssim(msssim);
          if (opt.enable_xpsnr) print_xpsnr(xframe);
          print_pu21_scores(opt, pu21);
          std::printf("\n");
        }
      }
      ++nframes;
    }

    if (nframes <= 0) {
      std::fprintf(stderr,
                   "pixvqa: no frames read (empty input or FIFO closed before the first frame)\n");
      return 1;
    }

#ifdef PIXVQA_HAS_VMAF
    if (opt.enable_vmaf) {
      vmaf.finish();
      if (opt.enable_qcs) {
        const auto& vframes = vmaf.frames();
        for (int i = 0; i < nframes; ++i) {
          const double score = vframes[static_cast<size_t>(i)].score;
          qcs_vmaf.add(score);
          if (trace_qcs) std::fprintf(stderr, "qcs_sample vmaf %.17g\n", score);
        }
      }
      if (opt.verbosity >= 1) {
        const auto& vframes = vmaf.frames();
        for (int i = 0; i < nframes; ++i) {
          const DeferredFrame& d = deferred[static_cast<size_t>(i)];
          std::printf("%-3d:", i);
          if (opt.enable_psnr) print_psnr(d.psnr.psnr, d.psnr.mse);
          if (opt.enable_ssim) print_metric("SSIM", d.ssim);
          if (opt.enable_msssim) print_msssim(d.msssim);
          if (d.has_xpsnr) print_xpsnr(d.xframe);
          print_pu21_scores(opt, d.pu21);
          print_vmaf(vframes[static_cast<size_t>(i)]);
          std::printf("\n");
        }
      }
      const char* feat_env = pix_getenv("VMAF_FEATURES");
      if (feat_env && feat_env[0] == '1') {
        const auto& vframes = vmaf.frames();
        for (int i = 0; i < nframes; ++i) {
          const VmafFrame& fr = vframes[static_cast<size_t>(i)];
          if (fr.features_valid) {
            std::fprintf(stderr,
                         "vmaf_features %d score=%.17g cambi=%.17g speed_chroma=%.17g "
                         "adm3=%.17g motion3=%.17g\n",
                         i, fr.score, fr.cambi, fr.speed_chroma, fr.adm3, fr.motion3);
          } else {
            std::fprintf(stderr, "vmaf_features %d score=%.17g features=unavailable\n", i,
                         fr.score);
          }
        }
      }
    }
#endif

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    if (g_learned) execute_learned(*g_learned);

    std::printf("\n\nSummary:\n\n");
    if (opt.enable_psnr) {
      auto summary = psnr_from_mean_mse(sum_mse_y / nframes, sum_mse_u / nframes,
                                        sum_mse_v / nframes, opt.format.chroma, opt.peak);
      print_psnr(summary.psnr, summary.mse);
      std::printf("\n");
    }
    if (opt.enable_ssim) {
      PlaneScore s{sum_ssim.y / nframes, sum_ssim.u / nframes, sum_ssim.v / nframes,
                   sum_ssim.avg / nframes};
      print_metric("SSIM", s);
      std::printf("\n");
    }
    if (opt.enable_msssim) {
      PlaneScore s{sum_msssim.y / nframes, sum_msssim.u / nframes, sum_msssim.v / nframes,
                   sum_msssim.avg / nframes};
      print_msssim(s);
      std::printf("\n");
    }
    if (opt.enable_xpsnr) {
      print_xpsnr(xpsnr.summary().score);
      std::printf("\n");
    }
    if (pu21_selected(opt)) {
      Pu21FrameScore summary;
      if (opt.enable_pu21_psnr) {
        summary.mse = sum_pu_mse / nframes;
        summary.psnr = summary.mse == 0.0 ? std::numeric_limits<double>::infinity()
                                          : 10.0 * std::log10((256.0 * 256.0) / summary.mse);
      }
      summary.ssim = sum_pu_ssim / nframes;
      summary.msssim = sum_pu_msssim / nframes;
      print_pu21_scores(opt, summary);
      std::printf("\n");
    }
#ifdef PIXVQA_HAS_VMAF
    if (opt.enable_vmaf) {
      const VmafSummary vs = vmaf.summary();
      // Summary: all official libvmaf temporal pool methods.
      std::printf(
          "[ VMAF mean:%.6f harmonic:%.6f min:%.6f max:%.6f frames:%d ]\n", vs.mean,
          vs.harmonic_mean, vs.min, vs.max, vs.frames);
    }
#endif
    if (g_learned) print_learned_summary(*g_learned);
    if (opt.enable_qcs) {
      if (opt.enable_psnr) print_qcs("PSNR-QCS", qcs_psnr);
      if (opt.enable_ssim) print_qcs("SSIM-QCS", qcs_ssim);
      if (opt.enable_msssim) print_qcs("MS-SSIM-QCS", qcs_msssim);
      if (opt.enable_xpsnr) print_qcs("XPSNR-QCS", qcs_xpsnr);
      if (opt.enable_vmaf) print_qcs("VMAF-QCS", qcs_vmaf);
      if (opt.enable_pu21_psnr) print_qcs("PU21-PSNR-QCS", qcs_pu21_psnr);
      if (opt.enable_pu21_ssim) print_qcs("PU21-SSIM-QCS", qcs_pu21_ssim);
      if (opt.enable_pu21_msssim) print_qcs("PU21-MS-SSIM-QCS", qcs_pu21_msssim);
      if (g_learned) print_learned_qcs(*g_learned);
    }
    std::printf("\nTotal frames:%d  fps:%.2f\n\n", nframes, nframes / std::max(sec, 1e-9));
    if (prof) {
      std::fprintf(stderr, "[PIXVQA_PROFILE] io=%.1fms compute=%.1fms\n", io_s * 1e3, compute_s * 1e3);
    }
    pixvqa_dump_ssim_profile();
#ifdef PIXVQA_HAS_CUDA
    if (opt.use_cuda) cuda_dump_profile();
#endif
    return g_learned ? learned_exit_status(*g_learned) : 0;
  } catch (const std::exception& ex) {
    const std::string msg = ex.what();
    if (msg.find("short YUV frame") != std::string::npos) {
      std::fprintf(stderr, "pixvqa: %s\n", ex.what());
      return 1;
    }
    std::fprintf(stderr, "pixvqa: %s\n", ex.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int status = run_cli(argc, argv);
  // A native runtime failure keeps its partial per-frame output; learned metrics
  // that passed preflight still run so their results are not lost.
  if (status == 1 && g_learned && g_learned->preflight_passed && !g_learned->executed) {
    std::fflush(stdout);
    execute_learned(*g_learned);
    std::printf("\n\nSummary:\n\n[ NATIVE status:failed ]\n");
    print_learned_summary(*g_learned);
    if (g_learned->opt.enable_qcs) print_learned_qcs(*g_learned);
    std::printf("\n");
  }
  g_learned.reset();
  return status;
}
