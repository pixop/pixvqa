#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace pixvqa {

// Official variables are PIXVQA_<suffix>.
inline const char* pix_getenv(const char* suffix) {
  char key[64];
  std::snprintf(key, sizeof key, "PIXVQA_%s", suffix);
  return std::getenv(key);
}


enum class ChromaFormat { YUV420, YUV422, YUV444 };

struct VideoFormat {
  int width = 0;
  int height = 0;
  int bit_depth = 10;
  ChromaFormat chroma = ChromaFormat::YUV420;

  int chroma_width() const {
    switch (chroma) {
      case ChromaFormat::YUV420:
      case ChromaFormat::YUV422:
        return width / 2;
      case ChromaFormat::YUV444:
        return width;
    }
    return width;
  }
  int chroma_height() const {
    switch (chroma) {
      case ChromaFormat::YUV420:
        return height / 2;
      case ChromaFormat::YUV422:
      case ChromaFormat::YUV444:
        return height;
    }
    return height;
  }
  size_t plane_samples(char plane) const {
    if (plane == 'y' || plane == 'Y') return static_cast<size_t>(width) * height;
    return static_cast<size_t>(chroma_width()) * chroma_height();
  }
  size_t frame_bytes() const {
    const size_t bytes_per = (bit_depth > 8) ? 2 : 1;
    return (plane_samples('y') + plane_samples('u') + plane_samples('v')) * bytes_per;
  }
};

struct PlaneView {
  const uint16_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;  // samples
};

struct PlaneBuffer {
  std::vector<uint16_t> storage;
  int width = 0;
  int height = 0;
  int stride = 0;

  void allocate(int w, int h) {
    width = w;
    height = h;
    stride = w;
    storage.assign(static_cast<size_t>(w) * h, 0);
  }
  uint16_t* ptr() { return storage.data(); }
  const uint16_t* ptr() const { return storage.data(); }
  PlaneView view() const { return PlaneView{ptr(), width, height, stride}; }
};

struct Frame {
  PlaneBuffer y, u, v;
  void allocate(const VideoFormat& fmt) {
    y.allocate(fmt.width, fmt.height);
    u.allocate(fmt.chroma_width(), fmt.chroma_height());
    v.allocate(fmt.chroma_width(), fmt.chroma_height());
  }
};

struct PlaneScore {
  double y = 0.0;
  double u = 0.0;
  double v = 0.0;
  double avg = 0.0;
};

struct PsnrFrame {
  PlaneScore psnr;
  PlaneScore mse;
};

// Scalar = debug reference; Avx2 = Phase-2 strip-tiled (default);
// Avx2Full = Phase-1 full-frame path retained for regression.
enum class SsimKernelMode { Scalar, Avx2, Avx2Full };

// CUDA metric precision. Fp32/Mixed filter in float32. All three reduce in float64.
// Fp64 filters in float64. Mixed evaluates the SSIM formula in float64.
enum class CudaPrecision { Fp32, Fp64, Mixed };

// XPSNR device selection. Cpu is the default, including when PSNR and
// MS-SSIM already run on CUDA. Cuda is opt-in and does not use CudaPrecision.
enum class XpsnrBackend { Cpu, Cuda };

// VMAF execution. Cpu is the only implemented backend. A GPU metric-device
// request keeps this value and warns; it is never reported as CUDA.
enum class VmafBackend { Cpu, Cuda };

// raw: both inputs are planar YUV. software/nvdec decode -t only.
enum class DecodeMode { Raw, Software, Nvdec };

// Public device request. Auto is resolved after the format is known.
enum class DeviceChoice { Auto, Cpu, Gpu };
enum class InputKind { Auto, Raw, Compressed };

// Split HDR overrides. "auto" is represented by leaving the corresponding set flag false.
struct HdrCli {
  bool transfer_set = false;
  std::string transfer;
  bool primaries_set = false;
  std::string primaries;
  bool matrix_set = false;
  std::string matrix;
  bool range_set = false;
  std::string range;
  bool chroma_loc_set = false;
  std::string chroma_loc;
  bool chroma_up_set = false;
  std::string chroma_up;
  bool peak_set = false;
  double peak = 0;
  bool black_set = false;
  double black = 0;
  bool gamma_set = false;
  double gamma = 0;
};

struct Options {
  std::string ref_path;
  std::string test_path;
  VideoFormat format;
  bool enable_psnr = false;
  bool enable_ssim = false;
  bool enable_msssim = false;
  bool enable_xpsnr = false;
  bool enable_vmaf = false;
  bool enable_pu21_psnr = false;
  bool enable_pu21_ssim = false;
  bool enable_pu21_msssim = false;
  bool pu21_gpu = false;
  // Required when enable_vmaf. Filesystem path to a VMAF model JSON.
  // Set by --model or the backward-compatible alias --vmaf-model.
  std::string vmaf_model_path;
  bool vmaf_model_from_model = false;       // --model was used
  bool vmaf_model_from_vmaf_model = false;  // --vmaf-model was used
  VmafBackend vmaf_backend = VmafBackend::Cpu;
  bool enable_qcs = false;
  bool qcs_range_set = false;
  // Exact-survival ranges. XPSNR inherits the PSNR default by convention.
  double qcs_psnr_lo = 30.0, qcs_psnr_hi = 50.0;
  double qcs_xpsnr_lo = 30.0, qcs_xpsnr_hi = 50.0;
  double qcs_ssim_lo = 0.8, qcs_ssim_hi = 1.0;
  double qcs_msssim_lo = 0.8, qcs_msssim_hi = 1.0;
  // One range for every VMAF model. HDR scores above 100 stay unclipped.
  double qcs_vmaf_lo = 70.0, qcs_vmaf_hi = 100.0;
  // PU21-PSNR uses peak 256, the PU code of about 100 cd/m2, so its dB
  // band matches PSNR rather than VMAF. PU21-SSIM and PU21-MS-SSIM are
  // similarities on that same peak, so they use the SSIM band.
  double qcs_pu21_psnr_lo = 30.0, qcs_pu21_psnr_hi = 50.0;
  double qcs_pu21_ssim_lo = 0.8, qcs_pu21_ssim_hi = 1.0;
  double qcs_pu21_msssim_lo = 0.8, qcs_pu21_msssim_hi = 1.0;
  XpsnrBackend xpsnr_backend = XpsnrBackend::Cpu;
  bool fps_set = false;
  int fps_num = 0;
  int fps_den = 1;
  // Resolved worker count for CPU metrics. --threads auto fills this in.
  int threads = 1;
  bool threads_auto = true;
  int logical_visible = 1;
  int verbosity = 1;
  int max_frames = -1;  // <0 = all
  SsimKernelMode kernel = SsimKernelMode::Scalar;
  DeviceChoice metric_device = DeviceChoice::Auto;
  DeviceChoice decode_request = DeviceChoice::Auto;
  // auto probes with libavformat. raw forces planar YUV. compressed requires
  // a video stream and does not fall back to raw.
  InputKind ref_input = InputKind::Auto;
  InputKind test_input = InputKind::Auto;
  bool psnr_gpu = false;
  bool ssim_gpu = false;
  bool msssim_gpu = false;
  bool vmaf_cpu_notice = false;
  bool use_cuda = false;
  bool cuda_prec_set = false;
  CudaPrecision cuda_prec = CudaPrecision::Fp64;
  int strip_height = 0;  // 0 = library default; the public CLI does not set this
  double peak = 1023.0;  // 10-bit
  // Color overrides. They do not enter PSNR, SSIM, MS-SSIM, or XPSNR.
  // HDR VMAF uses an explicit complete description of each input. SDR VMAF
  // leaves picture color unset. When set, these replace stream metadata.
  std::string color_space = "bt709l";
  std::string transfer = "sdr";
  bool color_space_set = false;
  bool transfer_set = false;
  bool ref_color_set = false;
  bool dist_color_set = false;
  bool ref_transfer_set = false;
  bool dist_transfer_set = false;
  std::string ref_color_space;
  std::string dist_color_space;
  std::string ref_transfer;
  std::string dist_transfer;
  bool hlg_peak_set = false;
  double hlg_peak = 0.0;
  HdrCli hdr;
  HdrCli ref_hdr;
  HdrCli dist_hdr;
  // decode is the distorted input. ref_decode is the reference input.
  // Raw means the file is planar YUV. Software and Nvdec are independent.
  DecodeMode decode = DecodeMode::Raw;
  DecodeMode ref_decode = DecodeMode::Raw;
};

inline double yuv_sample_weight_avg(double y, double u, double v, ChromaFormat fmt) {
  switch (fmt) {
    case ChromaFormat::YUV420:
      return (4.0 * y + u + v) / 6.0;
    case ChromaFormat::YUV422:
      return (2.0 * y + u + v) / 4.0;
    case ChromaFormat::YUV444:
      return (y + u + v) / 3.0;
  }
  return (4.0 * y + u + v) / 6.0;
}

}  // namespace pixvqa
