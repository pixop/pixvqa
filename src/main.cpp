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
#include <string>
#include <sys/stat.h>
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
      "1. Usage\n"
      "   Compare a reference with a reconstructed picture sequence.\n"
      "   --metrics is required. Names: psnr, xpsnr, ssim, msssim, vmaf,\n"
      "   pu21-psnr, pu21-ssim, pu21-msssim. See --help-hdr and docs/PU21.md.\n"
      "   --help and --version do not read inputs and do not initialize a GPU.\n"
      "   A compressed input takes resolution, pixel format, bit depth, and\n"
      "   chroma subsampling from the stream. Raw YUV still requires -s and -f.\n"
      "\n"
      "2. Input files and pixel formats\n"
      "   -r, -t              Reference and test paths. Regular files or FIFOs.\n"
      "   -s WxH              Luma size. Required for each raw input. Optional for\n"
      "                       a compressed input, where it is checked against the\n"
      "                       decoded picture. The two sides must match.\n"
      "   -f 420p:10          420p, 422p, or 444p, with :8 or :10. Default bits are 10\n"
      "                       only when -f is given without a bit depth. Required for\n"
      "                       each raw input. A compressed input uses the decoded\n"
      "                       layout, and an explicit -f must match it.\n"
      "                       4:2:0 requires even width and height.\n"
      "                       Samples are planar, little-endian, low bits occupied.\n"
      "   --decode auto|cpu|gpu\n"
      "                       Default auto. A regular file is compressed when\n"
      "                       FFmpeg opens a video stream in it. Otherwise it is\n"
      "                       raw planar YUV. A FIFO stays raw. There is no codec\n"
      "                       or container allowlist. --decode cpu and --decode gpu\n"
      "                       select the decoder for a file FFmpeg recognizes.\n"
      "                       They stay raw when FFmpeg finds no video stream.\n"
      "                       cpu uses a software decoder for that stream's codec.\n"
      "                       AV1 software decoding uses libdav1d when this FFmpeg\n"
      "                       build has it. VVC uses FFmpeg's native vvc decoder.\n"
      "                       gpu uses NVDEC for HEVC 10-bit 4:2:0 and for AV1\n"
      "                       8-bit or 10-bit 4:2:0. VVC and any other codec are\n"
      "                       rejected. gpu is not replaced by software.\n"
      "                       auto prefers software. AV1 uses NVDEC only when no\n"
      "                       software AV1 decoder is linked. VVC stays on software.\n"
      "   --input auto|raw|compressed\n"
      "                       Selects how the current input, or both inputs when\n"
      "                       given before -r, is classified. auto is the default.\n"
      "                       raw reads planar YUV and does not probe. compressed\n"
      "                       requires a video stream and does not fall back to raw.\n"
      "                       Software pictures must already be planar 8-bit or\n"
      "                       10-bit 4:2:0, 4:2:2, or 4:4:4. Other decoded layouts\n"
      "                       are rejected by name. Samples are not converted.\n"
      "                       The two inputs are decoded independently and paired in\n"
      "                       presentation order. Timestamps are not used to drop,\n"
      "                       repeat, or resynchronize frames. A length or geometry\n"
      "                       mismatch is an error.\n"
      "\n"
      "3. Metrics\n"
      "   --metrics LIST      Comma-separated names. psnr prints PSNR and MSE.\n"
      "                       ssim is classical single-scale SSIM (11x11 Gaussian,\n"
      "                       sigma 1.5, K1=0.01, K2=0.03), printed as\n"
      "                       [ SSIM  y: u: v: avg: ].\n"
      "                       msssim is the five-scale product MS-SSIM, printed as\n"
      "                       [ MS-SSIM  y: u: v: avg: ]. ssim does not imply msssim.\n"
      "                       xpsnr requires --fps when no single constant frame rate\n"
      "                       can be read from the compressed inputs, and prints\n"
      "                       Y U V avg and min.\n"
      "                       vmaf is Official Netflix VMAF v1 via libvmaf (CPU).\n"
      "   --model PATH        Required when vmaf is selected. VMAF model JSON.\n"
      "                       Absolute or relative to the working directory.\n"
      "                       No default, preset, or automatic model selection.\n"
      "   --vmaf-model PATH   Alias for --model. Rejected if both disagree.\n"
      "   Displayed names are PSNR, XPSNR, SSIM, MS-SSIM, and VMAF.\n"
      "   GPU PSNR, SSIM, MS-SSIM, XPSNR, and PU21 accept 8-bit and 10-bit.\n"
      "   GPU PSNR, SSIM, and MS-SSIM use FP64.\n"
      "   VMAF always runs on the CPU, including when other metrics use the GPU\n"
      "   and when -t was decoded on the GPU. An SDR model rejects PQ and HLG.\n"
      "   An HDR VMAF v1.1.8 model requires explicit PQ, BT.2020 primaries,\n"
      "   the bt2020nc matrix, and limited range on both inputs, from stream\n"
      "   tags or --hdr-transfer, --hdr-primaries, --hdr-matrix, and --hdr-range.\n"
      "   HLG and full-range PQ stay rejected. A GPU request still runs HDR VMAF\n"
      "   on CPU libvmaf and warns. It does not compute HDR VMAF on CUDA.\n"
      "   HDR scores keep the model clip, including [0,110].\n"
      "   pu21-psnr (PU21-PSNR), pu21-ssim (PU21-SSIM), and pu21-msssim\n"
      "   (PU21-MS-SSIM) accept 8-bit and 10-bit PQ or HLG on CPU and CUDA.\n"
      "\n"
      "4. Execution\n"
      "   --metric-device auto|cpu|gpu\n"
      "                       Default auto. cpu keeps every selected metric on the CPU.\n"
      "                       gpu uses a validated GPU implementation for each selected\n"
      "                       metric that has one. A format the GPU metric does not\n"
      "                       support is an error, with no CPU substitute.\n"
      "                       VMAF has no GPU implementation. Selecting vmaf with gpu\n"
      "                       uses CPU libvmaf and prints a warning, including when\n"
      "                       vmaf is the only metric. Other selected GPU metrics stay\n"
      "                       on the GPU. The summary names the requested device and\n"
      "                       the actual device of each metric.\n"
      "                       auto does not benchmark at startup. It uses the CPU unless\n"
      "                       a checked profile applies. The only profile is 10-bit\n"
      "                       3840x2160 or larger, GPU decoding, no VMAF, and a CUDA\n"
      "                       build: PSNR, SSIM, MS-SSIM, XPSNR, and PU21 then use the GPU.\n"
      "                       That profile is one 4K HEVC comparison. It is not a claim\n"
      "                       that a GPU is faster in general.\n"
      "   --threads auto|N    Default auto. auto counts the processors visible to this\n"
      "                       process, including affinity limits, and uses a conservative\n"
      "                       budget (half of those processors, at most 8, at least 1).\n"
      "                       That budget is not an optimum. CPU metrics share it\n"
      "                       sequentially. Software decoding keeps its own fixed pool\n"
      "                       of 8 threads on each compressed input; auto then leaves\n"
      "                       every such pool room when it can.\n"
      "                       N is a positive integer for the metric workers only.\n"
      "   An execution summary lists each input as raw, cpu, or gpu, and each metric backend.\n"
      "\n"
      "5. QCS\n"
      "   Quality Consistency Score (QCS) measures how consistently video quality\n"
      "   remains above selected quality thresholds over time, using the normalized\n"
      "   area under the survival curve of frame-level quality scores.\n"
      "   It is not a ninth quality metric. It is an optional descriptor of the\n"
      "   selected base scores. Every metric in --metrics has a QCS value,\n"
      "   whether that metric runs on the CPU or the GPU.\n"
      "   --qcs               Print one QCS value for each selected metric.\n"
      "   --qcs-range LIST    Repeatable. LIST is metric=lower:upper, or several\n"
      "                       such entries separated by commas. The metric is a\n"
      "                       --metrics name: psnr, xpsnr, ssim, msssim, vmaf,\n"
      "                       pu21-psnr, pu21-ssim, or pu21-msssim. Requires --qcs\n"
      "                       and lower < upper. The same metric with a different\n"
      "                       interval is an error.\n"
      "   Defaults: psnr and xpsnr [30,50]; ssim and msssim [0.8,1.0];\n"
      "   vmaf [70,100] for every SDR and HDR model; pu21-psnr [30,50];\n"
      "   pu21-ssim and pu21-msssim [0.8,1.0].\n"
      "   Each frame is one sample. Timestamps are not read, so this matches\n"
      "   viewing time only when every frame has the same duration. Frame order\n"
      "   does not change QCS.\n"
      "   A score at or below the lower limit contributes 0%. A score at or\n"
      "   above the upper limit contributes 100%. +inf contributes 100%.\n"
      "   -inf contributes 0%. NaN is omitted. If no valid sample remains,\n"
      "   the value is unavailable. The frame score is not clipped or\n"
      "   rescaled before that contribution. A VMAF score above 100, which\n"
      "   some models allow up to their own clip of 110, contributes 100%\n"
      "   when the upper bound is 100.\n"
      "   Identical PSNR is +inf internally and is printed 999.99.\n"
      "   Identical XPSNR is printed inf. Identical PU21-PSNR is printed 999.99.\n"
      "   Those frames still count in QCS.\n"
      "\n"
      "6. Frame rate and frame limits\n"
      "   --fps N or N/D      Positive integer or rational. xpsnr requires --fps\n"
      "                       for raw YUV, and for compressed input whose frame rate\n"
      "                       is missing, variable, or different on the two sides.\n"
      "                       A matching constant rate from a compressed input is\n"
      "                       used when --fps is omitted. An explicit --fps is used\n"
      "                       as given. 59.94 is truncated to 59. Resolution and\n"
      "                       frame rate never switch the VMAF model.\n"
      "   -n N, --frames N    Stop after N frames. N is a positive integer.\n"
      "   -v 0                Summary only. -v 1 (default) prints each frame.\n"
      "\n"
      "7. General options\n"
      "   --cs, --tf          Legacy color and transfer override for both inputs.\n"
      "                       Accepted values are listed by --help-hdr.\n"
      "   --version           Name, development version, build kind, and decode support.\n"
      "   -h, --help          This text. Exit 0.\n"
      "   --help-hdr          HDR options, accepted values, and defaults. Exit 0.\n"
      "   Unknown options are an error. --wsssim, --wspsnr, --rfilter, --tfilter,\n"
      "   --geom, and nonzero --seek are unsupported.\n"
      "\n"
      "8. Examples\n"
      "   pixvqa --metrics psnr,xpsnr,ssim,msssim,vmaf --fps 60 \\\n"
      "       --model model/vmaf_v1.0.16_3d0h.json \\\n"
      "       -r reference.yuv -s 320x240 -f 420p:10 \\\n"
      "       -t reconstructed.yuv -s 320x240 -f 420p:10\n"
      "   pixvqa --metrics psnr,ssim,msssim --metric-device cpu --threads auto \\\n"
      "       -r reference.yuv -s 176x176 -f 420p:10 \\\n"
      "       -t reconstructed.yuv -s 176x176 -f 420p:10\n"
      "   pixvqa --metrics psnr,ssim,msssim --threads 4 \\\n"
      "       -r reference.yuv -s 176x176 -f 420p:10 \\\n"
      "       -t reconstructed.yuv -s 176x176 -f 420p:10\n"
      "   pixvqa --metrics psnr,xpsnr,ssim,msssim --decode gpu --metric-device gpu \\\n"
      "       --fps 60 -r reference.yuv -s 3840x2160 -f 420p:10 \\\n"
      "       -t encoded.hevc -s 3840x2160 -f 420p:10\n"
      "   pixvqa --metrics psnr --qcs --qcs-range psnr=40:55 \\\n"
      "       -r reference.yuv -s 176x176 -f 420p:10 \\\n"
      "       -t reconstructed.yuv -s 176x176 -f 420p:10\n"
      "   pixvqa --metrics vmaf --qcs --qcs-range vmaf=60:90 \\\n"
      "       --model model/vmaf_v1.0.16_3d0h.json \\\n"
      "       -r reference.yuv -s 320x240 -f 420p:10 \\\n"
      "       -t reconstructed.yuv -s 320x240 -f 420p:10\n",
      stdout);
}

void print_help_hdr() {
  std::fputs(
      "HDR parameters\n"
      "\n"
      "PU21 metrics: pu21-psnr, pu21-ssim, pu21-msssim. PQ or HLG, 8-bit or\n"
      "10-bit, on CPU or CUDA. CPU is the reference. --metric-device gpu selects\n"
      "CUDA. Processing, the HLG gamma formula, and examples: docs/PU21.md.\n"
      "PU21 is the banding_glare encoding of Mantiuk and Azimi, PCS 2021.\n"
      "DOI: https://doi.org/10.1109/PCS50896.2021.9477471\n"
      "Reference code: https://github.com/gfxdisp/pu21\n"
      "\n"
      "Order for every field: per-input option, shared option, stream metadata,\n"
      "documented default, then an error if the value is still ambiguous.\n"
      "A 10-bit frame, a filename, or BT.2020 primaries do not select PQ or HLG.\n"
      "Contradictory metadata inside one stream is an error.\n"
      "Reference and distorted inputs are resolved independently.\n"
      "\n"
      "When the transfer is pq or hlg and a field is missing:\n"
      "  primaries    bt2020\n"
      "  matrix       bt2020nc (non-constant luminance)\n"
      "  range        limited\n"
      "  chroma-loc   left (4:2:0 horizontal co-sited, vertical midway)\n"
      "  chroma-up    bilinear\n"
      "  HLG peak     1000 cd/m2\n"
      "  HLG gamma    1.2 at 1000 cd/m2; otherwise derived from that peak\n"
      "  HLG black    0 cd/m2\n"
      "PQ stays on its absolute scale. Set --hdr-peak-nits to clip an evaluation\n"
      "display. That value is separate from the mastering-display peak, MaxCLL,\n"
      "MaxFALL, and the content peak. Those are not read.\n"
      "HLG peak is the evaluation display. Another peak changes the derived gamma\n"
      "unless --hdr-hlg-gamma is set.\n"
      "\n"
      "Shared options:\n"
      "  --hdr-transfer auto|pq|hlg|sdr|linear\n"
      "  --hdr-primaries auto|bt2020|bt709\n"
      "  --hdr-matrix auto|bt2020nc|bt709\n"
      "  --hdr-range auto|limited|full\n"
      "  --hdr-chroma-loc auto|left|center|topleft\n"
      "  --hdr-chroma-up bilinear|nearest\n"
      "  --hdr-peak-nits auto|N        N in (0, 10000] cd/m2\n"
      "  --hdr-black-nits auto|N       HLG only, in [0, peak)\n"
      "  --hdr-hlg-gamma auto|N        HLG only, in (0, 10]\n"
      "Prefix ref- or dist- to set one input, for example --ref-hdr-transfer pq.\n"
      "auto leaves the field unset so metadata or the default applies.\n"
      "\n"
      "Legacy aliases, still accepted:\n"
      "  --tf sdr|linear|st2084|stdb67     st2084 is pq, stdb67 is hlg\n"
      "  --cs bt709l|bt709f|bt2020l|bt2020f|bt601l|bt601f\n"
      "  --hlg-peak N                      HLG evaluation peak only\n"
      "  --ref-tf, --dist-tf, --ref-cs, --dist-cs\n"
      "bt601 is rejected for PQ and HLG. A per-input value replaces the shared\n"
      "value for that input. Two controls of the same scope are an error:\n"
      "--tf with --hdr-transfer, --ref-tf with --ref-hdr-transfer, and\n"
      "--hlg-peak with --hdr-peak-nits. --hlg-peak with --ref-hdr-peak-nits\n"
      "keeps the shared peak for the other input.\n"
      "PSNR, SSIM, MS-SSIM, and XPSNR compare stored sample codes.\n"
      "An SDR VMAF model rejects PQ and HLG. An HDR VMAF model requires\n"
      "limited-range PQ. See --help.\n",
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
  if (std::fabs(x) >= 2.0 && std::fabs(x - std::round(x)) < 1e-9) {
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

bool add_metric_name(Options& opt, const std::string& name) {
  if (name.empty()) {
    std::fprintf(stderr, "pixvqa: empty --metrics entry\n");
    return false;
  }
  if (name == "psnr") opt.enable_psnr = true;
  else if (name == "xpsnr") opt.enable_xpsnr = true;
  else if (name == "ssim") opt.enable_ssim = true;
  else if (name == "msssim") opt.enable_msssim = true;
  else if (name == "vmaf") opt.enable_vmaf = true;
  else if (name == "pu21-psnr") opt.enable_pu21_psnr = true;
  else if (name == "pu21-ssim") opt.enable_pu21_ssim = true;
  else if (name == "pu21-msssim") opt.enable_pu21_msssim = true;
  else {
    std::fprintf(stderr,
                 "pixvqa: unsupported metric %s (supported: psnr, xpsnr, ssim, msssim, vmaf, "
                 "pu21-psnr, pu21-ssim, pu21-msssim)\n",
                 name.c_str());
    return false;
  }
  return true;
}

bool parse_metrics_list(Options& opt, const std::string& list) {
  std::string cur;
  auto flush = [&](const std::string& name) { return add_metric_name(opt, name); };
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
    std::fprintf(stderr, "pixvqa: compressed input requires a decoder; this build has none\n");
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
  if (opt.enable_vmaf && !opt.vmaf_model_path.empty()) {
    try {
      hdr_model = vmaf_inspect_model_file(vmaf_resolve_model_path(opt.vmaf_model_path)).has_conversion;
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "pixvqa: %s\n", ex.what());
      return 2;
    }
  }
  if (opt.enable_vmaf && hdr_model) {
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
                 "pixvqa: %s: this build cannot detect compressed metadata and has no decoder\n",
                 role);
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

int main(int argc, char** argv) {
  Options opt;
  SideSpec ref_side, test_side;
  SideSpec* bind = nullptr;

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

    if (a == "--metrics") {
      if (!parse_metrics_list(opt, need("--metrics"))) return 2;
    } else if (a == "--metric-device") {
      const auto v = need("--metric-device");
      if (!parse_device_choice(v, opt.metric_device)) {
        std::fprintf(stderr, "pixvqa: unsupported --metric-device %s (use auto, cpu, or gpu)\n",
                     v.c_str());
        return 2;
      }
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
    } else if (a == "-f") {
      const auto s = need("-f");
      if (bind) apply_format(*bind, s);
      else apply_format_both(ref_side, test_side, s);
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

  // Inherit unset side geometry/format from the other side.
  if (!ref_side.has_size && test_side.has_size) {
    ref_side.format.width = test_side.format.width;
    ref_side.format.height = test_side.format.height;
    ref_side.has_size = true;
  }
  if (!test_side.has_size && ref_side.has_size) {
    test_side.format.width = ref_side.format.width;
    test_side.format.height = ref_side.format.height;
    test_side.has_size = true;
  }
  if (!ref_side.has_fmt && test_side.has_fmt) {
    ref_side.format.chroma = test_side.format.chroma;
    ref_side.format.bit_depth = test_side.format.bit_depth;
    ref_side.has_fmt = true;
  }
  if (!test_side.has_fmt && ref_side.has_fmt) {
    test_side.format.chroma = ref_side.format.chroma;
    test_side.format.bit_depth = ref_side.format.bit_depth;
    test_side.has_fmt = true;
  }

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

    std::printf("\n\nSummary results:\n----------------\n\n");
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
    if (opt.enable_qcs) {
      if (opt.enable_psnr) print_qcs("PSNR-QCS", qcs_psnr);
      if (opt.enable_ssim) print_qcs("SSIM-QCS", qcs_ssim);
      if (opt.enable_msssim) print_qcs("MS-SSIM-QCS", qcs_msssim);
      if (opt.enable_xpsnr) print_qcs("XPSNR-QCS", qcs_xpsnr);
      if (opt.enable_vmaf) print_qcs("VMAF-QCS", qcs_vmaf);
      if (opt.enable_pu21_psnr) print_qcs("PU21-PSNR-QCS", qcs_pu21_psnr);
      if (opt.enable_pu21_ssim) print_qcs("PU21-SSIM-QCS", qcs_pu21_ssim);
      if (opt.enable_pu21_msssim) print_qcs("PU21-MS-SSIM-QCS", qcs_pu21_msssim);
    }
    std::printf("\nTotal frames:%d  fps:%.2f\n\n", nframes, nframes / std::max(sec, 1e-9));
    if (prof) {
      std::fprintf(stderr, "[PIXVQA_PROFILE] io=%.1fms compute=%.1fms\n", io_s * 1e3, compute_s * 1e3);
    }
    pixvqa_dump_ssim_profile();
#ifdef PIXVQA_HAS_CUDA
    if (opt.use_cuda) cuda_dump_profile();
#endif
    return 0;
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
