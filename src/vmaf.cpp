#include "pixvqa/vmaf.hpp"

#ifndef PIXVQA_HAS_VMAF
#error "src/vmaf.cpp requires -DPIXVQA_HAS_VMAF=1 and a linked libvmaf"
#endif

#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
#include <libvmaf/version.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace pixvqa {
namespace {

namespace fs = std::filesystem;

struct ModelInfo {
  const char* id;
  const char* sha256;
  double clip_hi;
  bool hfr;
  int min_w;
  int min_h;
  const char* feat_cambi;
  const char* feat_speed_chroma;
  const char* feat_adm3;
  const char* feat_motion3;
};

// SHA-256 of the JSON files at Netflix/vmaf tag v3.2.1
// (commit f85a853692a8c730d0270cd733c8bb30b5b93b7c).
constexpr ModelInfo kModels[] = {
    {"vmaf_v1.0.16_3d0h",
     "e4cf8c147e1368b35497d772920bc92f98c1ad7853c1033d8a836947f427140e", 100.0, false, 320, 240,
     "cambi_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02", "integer_motion3_mmxv_18"},
    {"vmaf_v1.0.16_5d0h",
     "64e12d9c1e4ed5d79da7fa515b82fac7247a9e2ccc91e0ebce31d05489500385", 100.0, false, 1920, 1080,
     "cambi_hrs_1080_cmxv_17_vlt_0.06",
     "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5", "integer_motion3_mmxv_18"},
    {"vmaf_v1.0.16_1d5h_2160",
     "634d36e0502cb797384726cdd1a106a2c207827595a9e4958d3dae1bd517b139", 100.0, false, 320, 240,
     "cambi_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160",
     "integer_motion3_mmxv_18"},
    {"vmaf_v1.0.16_3d0h_2160",
     "77f552fd8d5b6c6a2166bfc4470c9368ec68d2425ed40f1cc235b0cc500f33c2", 110.0, false, 1920, 1080,
     "cambi_hrs_1080_cmxv_17_vlt_0.06",
     "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160", "integer_motion3_mmxv_18"},
    {"vmaf_v1.0.16_hfr_3d0h",
     "dcfdfab1c00685c1e5f75b6f1078bb4a44c722654e4de090fa5c66784c75241b", 100.0, true, 320, 240,
     "cambi_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02", "integer_motion3_mffw_mmxv_18_mma"},
    {"vmaf_v1.0.16_hfr_5d0h",
     "590f612c87df94badca58e563b318fa9958eac631ab8aadd73bff8cc1adfc515", 100.0, true, 1920, 1080,
     "cambi_hrs_1080_cmxv_17_vlt_0.06",
     "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5",
     "integer_motion3_mffw_mmxv_18_mma"},
    {"vmaf_v1.0.16_hfr_1d5h_2160",
     "0371db55d3de81a760125f6d65eb4386f1d38af6cf1d00c2f48158e3596246d4", 100.0, true, 320, 240,
     "cambi_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160",
     "integer_motion3_mffw_mmxv_18_mma"},
    {"vmaf_v1.0.16_hfr_3d0h_2160",
     "1bc502f00aeee11433e2d09468186bbf1bbce56ee650c9bade06c7435ebdd4db", 110.0, true, 1920, 1080,
     "cambi_hrs_1080_cmxv_17_vlt_0.06",
     "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5",
     "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160",
     "integer_motion3_mffw_mmxv_18_mma"},
};

const ModelInfo* find_by_sha256(const std::string& digest) {
  for (const auto& m : kModels) {
    if (digest == m.sha256) return &m;
  }
  return nullptr;
}

// Parse model_dict.score_clip = [lo, hi] from a VMAF model JSON.
// Ignores param_dict.score_clip when it is null (or any non-array value).
// Returns false when no usable [lo, hi] array is present.
bool parse_score_clip_from_json(const std::string& path, double& lo, double& hi) {
  std::ifstream in(path);
  if (!in) return false;
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  // Prefer the last "score_clip" whose value is a JSON array (not null).
  size_t pos = 0;
  size_t best = std::string::npos;
  while (true) {
    const size_t found = text.find("\"score_clip\"", pos);
    if (found == std::string::npos) break;
    size_t i = found + 12;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' ||
                               text[i] == '\r' || text[i] == ':')) {
      ++i;
    }
    if (i < text.size() && text[i] == '[') best = found;
    pos = found + 12;
  }
  if (best == std::string::npos) return false;
  size_t i = best + 12;
  while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' ||
                             text[i] == '\r' || text[i] == ':')) {
    ++i;
  }
  if (i >= text.size() || text[i] != '[') return false;
  ++i;
  try {
    size_t end1 = 0;
    lo = std::stod(text.substr(i), &end1);
    i += end1;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' ||
                               text[i] == '\r' || text[i] == ',')) {
      ++i;
    }
    size_t end2 = 0;
    hi = std::stod(text.substr(i), &end2);
  } catch (...) {
    return false;
  }
  return true;
}

// Minimal SHA-256 (public-domain style) for model file digests.
struct Sha256Ctx {
  uint64_t bitlen = 0;
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint8_t data[64]{};
  size_t datalen = 0;
};

uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

void sha256_transform(Sha256Ctx& ctx, const uint8_t data[64]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
      0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
      0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
      0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
      0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
      0xc67178f2u};
  uint32_t m[64];
  for (int i = 0; i < 16; ++i) {
    m[i] = (uint32_t(data[i * 4]) << 24) | (uint32_t(data[i * 4 + 1]) << 16) |
           (uint32_t(data[i * 4 + 2]) << 8) | uint32_t(data[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
    const uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
    m[i] = m[i - 16] + s0 + m[i - 7] + s1;
  }
  uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
  uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + S1 + ch + k[i] + m[i];
    const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = S0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  ctx.state[0] += a;
  ctx.state[1] += b;
  ctx.state[2] += c;
  ctx.state[3] += d;
  ctx.state[4] += e;
  ctx.state[5] += f;
  ctx.state[6] += g;
  ctx.state[7] += h;
}

void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ctx.data[ctx.datalen++] = data[i];
    if (ctx.datalen == 64) {
      sha256_transform(ctx, ctx.data);
      ctx.bitlen += 512;
      ctx.datalen = 0;
    }
  }
}

std::string sha256_final_hex(Sha256Ctx& ctx) {
  size_t i = ctx.datalen;
  ctx.data[i++] = 0x80;
  if (i > 56) {
    while (i < 64) ctx.data[i++] = 0;
    sha256_transform(ctx, ctx.data);
    i = 0;
  }
  while (i < 56) ctx.data[i++] = 0;
  ctx.bitlen += static_cast<uint64_t>(ctx.datalen) * 8;
  for (int j = 0; j < 8; ++j) {
    ctx.data[63 - j] = static_cast<uint8_t>((ctx.bitlen >> (8 * j)) & 0xff);
  }
  sha256_transform(ctx, ctx.data);
  char out[65];
  for (int j = 0; j < 8; ++j) {
    std::snprintf(out + j * 8, 9, "%08x", ctx.state[j]);
  }
  return std::string(out, 64);
}

enum VmafPixelFormat to_pix_fmt(ChromaFormat c) {
  switch (c) {
    case ChromaFormat::YUV420:
      return VMAF_PIX_FMT_YUV420P;
    case ChromaFormat::YUV422:
      return VMAF_PIX_FMT_YUV422P;
    case ChromaFormat::YUV444:
      return VMAF_PIX_FMT_YUV444P;
  }
  return VMAF_PIX_FMT_UNKNOWN;
}

void check_err(int err, const char* what) {
  if (err) {
    throw std::runtime_error(std::string("libvmaf ") + what + " failed (" +
                             std::to_string(err) + ")");
  }
}

void copy_plane_u16(const PlaneBuffer& src, uint16_t* dst, ptrdiff_t stride_bytes, int w, int h) {
  const int stride_samples =
      static_cast<int>(stride_bytes / static_cast<ptrdiff_t>(sizeof(uint16_t)));
  for (int y = 0; y < h; ++y) {
    std::memcpy(dst + static_cast<size_t>(y) * stride_samples,
                src.ptr() + static_cast<size_t>(y) * src.stride,
                static_cast<size_t>(w) * sizeof(uint16_t));
  }
}

void copy_plane_u8_from_u16(const PlaneBuffer& src, uint8_t* dst, ptrdiff_t stride_bytes, int w,
                            int h) {
  for (int y = 0; y < h; ++y) {
    const uint16_t* row = src.ptr() + static_cast<size_t>(y) * src.stride;
    uint8_t* out = dst + static_cast<size_t>(y) * stride_bytes;
    for (int x = 0; x < w; ++x) out[x] = static_cast<uint8_t>(row[x] & 0xff);
  }
}

}  // namespace

std::string vmaf_file_sha256(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to read VMAF model file: " + path);
  Sha256Ctx ctx;
  uint8_t buf[8192];
  while (in) {
    in.read(reinterpret_cast<char*>(buf), sizeof buf);
    const auto n = static_cast<size_t>(in.gcount());
    if (n) sha256_update(ctx, buf, n);
  }
  return sha256_final_hex(ctx);
}

std::string vmaf_resolve_model_path(const std::string& requested) {
  if (requested.empty()) throw std::runtime_error("empty VMAF model path");
  std::error_code ec;
  fs::path p = fs::absolute(fs::path(requested), ec);
  if (ec) throw std::runtime_error("cannot resolve VMAF model path: " + requested);
  return p.lexically_normal().string();
}

bool vmaf_min_size_for_sha256(const std::string& sha256, int& min_w, int& min_h) {
  const ModelInfo* info = find_by_sha256(sha256);
  if (!info) return false;
  min_w = info->min_w;
  min_h = info->min_h;
  return true;
}

VmafMeter::VmafMeter() = default;

VmafMeter::~VmafMeter() { destroy(); }

void VmafMeter::destroy() {
  // Close the VMAF context before destroying the model. Feature extractors
  // registered from the model may still hold references until vmaf_close().
  if (vmaf_) {
    vmaf_close(static_cast<VmafContext*>(vmaf_));
    vmaf_ = nullptr;
  }
  if (model_) {
    vmaf_model_destroy(static_cast<VmafModel*>(model_));
    model_ = nullptr;
  }
  started_ = false;
  finished_ = false;
  submitted_ = 0;
  pooled_mean_ = 0.0;
  pooled_harmonic_ = 0.0;
  pooled_min_ = 0.0;
  pooled_max_ = 0.0;
  frames_.clear();
  feat_cambi_.clear();
  feat_speed_chroma_.clear();
  feat_adm3_.clear();
  feat_motion3_.clear();
  meta_ = VmafMeta{};
}

void VmafMeter::ensure_started() const {
  if (!started_) throw std::runtime_error("VmafMeter::start was not called");
}

void VmafMeter::start(const VideoFormat& fmt, const std::string& model_path, int n_threads) {
  destroy();
  if (fmt.bit_depth != 8 && fmt.bit_depth != 10) {
    throw std::runtime_error("VMAF accepts only 8-bit or 10-bit planar YUV");
  }
  if (fmt.width <= 0 || fmt.height <= 0) {
    throw std::runtime_error("VMAF requires positive frame dimensions");
  }

  meta_.model_path_requested = model_path;
  meta_.model_path_resolved = vmaf_resolve_model_path(model_path);
  meta_.libvmaf_version = vmaf_version();
  meta_.backend = "cpu";

  if (!fs::is_regular_file(meta_.model_path_resolved)) {
    throw std::runtime_error("VMAF model file not found or not a regular file: " +
                             meta_.model_path_resolved);
  }
  {
    std::ifstream probe(meta_.model_path_resolved);
    if (!probe) {
      throw std::runtime_error("VMAF model file is not readable: " + meta_.model_path_resolved);
    }
  }

  meta_.model_sha256 = vmaf_file_sha256(meta_.model_path_resolved);
  meta_.model_id = fs::path(meta_.model_path_resolved).stem().string();

  // Prefer the score_clip array from the loaded JSON. Never invent 100 for unknowns.
  double json_lo = 0.0, json_hi = 0.0;
  const bool json_clip = parse_score_clip_from_json(meta_.model_path_resolved, json_lo, json_hi);
  if (json_clip) {
    meta_.score_clip_known = true;
    meta_.score_clip_lo = json_lo;
    meta_.score_clip_hi = json_hi;
  } else {
    meta_.score_clip_known = false;
    meta_.score_clip_lo = 0.0;
    meta_.score_clip_hi = 0.0;
  }

  const ModelInfo* known = find_by_sha256(meta_.model_sha256);
  if (known) {
    meta_.known_v1_model = true;
    meta_.model_id = known->id;
    meta_.hfr_model = known->hfr;
    feat_cambi_ = known->feat_cambi;
    feat_speed_chroma_ = known->feat_speed_chroma;
    feat_adm3_ = known->feat_adm3;
    feat_motion3_ = known->feat_motion3;
    // Known table is a fallback only when JSON parsing failed.
    if (!meta_.score_clip_known) {
      meta_.score_clip_known = true;
      meta_.score_clip_lo = 0.0;
      meta_.score_clip_hi = known->clip_hi;
    }
  } else {
    meta_.known_v1_model = false;
    meta_.hfr_model = meta_.model_id.find("_hfr_") != std::string::npos;
  }

  fmt_ = fmt;
  threads_ = n_threads < 1 ? 1 : n_threads;

  VmafConfiguration cfg{};
  cfg.log_level = VMAF_LOG_LEVEL_ERROR;
  cfg.n_threads = static_cast<unsigned>(threads_);
  cfg.n_subsample = 0;
  cfg.cpumask = 0;
  cfg.gpumask = 0;

  VmafContext* vmaf = nullptr;
  check_err(vmaf_init(&vmaf, cfg), "vmaf_init");
  vmaf_ = vmaf;

  VmafModelConfig model_cfg{};
  model_cfg.name = "vmaf";
  model_cfg.flags = 0;
  VmafModel* model = nullptr;
  const int load_err =
      vmaf_model_load_from_path(&model, &model_cfg, meta_.model_path_resolved.c_str());
  if (load_err || !model) {
    if (vmaf_) {
      vmaf_close(static_cast<VmafContext*>(vmaf_));
      vmaf_ = nullptr;
    }
    throw std::runtime_error("libvmaf failed to load model from path: " +
                             meta_.model_path_resolved + " (" + std::to_string(load_err) + ")");
  }
  model_ = model;

  const int feat_err = vmaf_use_features_from_model(vmaf, model);
  if (feat_err) {
    if (vmaf_) {
      vmaf_close(static_cast<VmafContext*>(vmaf_));
      vmaf_ = nullptr;
    }
    if (model_) {
      vmaf_model_destroy(static_cast<VmafModel*>(model_));
      model_ = nullptr;
    }
    throw std::runtime_error(
        "libvmaf could not register feature extractors for model: " + meta_.model_path_resolved +
        " (" + std::to_string(feat_err) + ")");
  }

  started_ = true;
  finished_ = false;
  submitted_ = 0;
  frames_.clear();
}

void VmafMeter::fill_picture(const Frame& src, void* pic_out) {
  auto* pic = static_cast<VmafPicture*>(pic_out);
  const enum VmafPixelFormat pix_fmt = to_pix_fmt(fmt_.chroma);
  const unsigned bpc = static_cast<unsigned>(fmt_.bit_depth);
  check_err(vmaf_picture_alloc(pic, pix_fmt, bpc, static_cast<unsigned>(fmt_.width),
                               static_cast<unsigned>(fmt_.height)),
            "vmaf_picture_alloc");

  if (fmt_.bit_depth > 8) {
    copy_plane_u16(src.y, static_cast<uint16_t*>(pic->data[0]), pic->stride[0], src.y.width,
                   src.y.height);
    copy_plane_u16(src.u, static_cast<uint16_t*>(pic->data[1]), pic->stride[1], src.u.width,
                   src.u.height);
    copy_plane_u16(src.v, static_cast<uint16_t*>(pic->data[2]), pic->stride[2], src.v.width,
                   src.v.height);
  } else {
    copy_plane_u8_from_u16(src.y, static_cast<uint8_t*>(pic->data[0]), pic->stride[0], src.y.width,
                           src.y.height);
    copy_plane_u8_from_u16(src.u, static_cast<uint8_t*>(pic->data[1]), pic->stride[1], src.u.width,
                           src.u.height);
    copy_plane_u8_from_u16(src.v, static_cast<uint8_t*>(pic->data[2]), pic->stride[2], src.v.width,
                           src.v.height);
  }
}

void VmafMeter::push(const Frame& reference, const Frame& distorted) {
  ensure_started();
  if (finished_) throw std::runtime_error("VmafMeter::push after finish");

  VmafPicture ref{};
  VmafPicture dist{};
  fill_picture(reference, &ref);
  try {
    fill_picture(distorted, &dist);
  } catch (...) {
    vmaf_picture_unref(&ref);
    throw;
  }

  const unsigned index = static_cast<unsigned>(submitted_);
  const int err = vmaf_read_pictures(static_cast<VmafContext*>(vmaf_), &ref, &dist, index);
  if (err) {
    if (ref.ref) vmaf_picture_unref(&ref);
    if (dist.ref) vmaf_picture_unref(&dist);
    check_err(err, "vmaf_read_pictures");
  }
  ++submitted_;
}

void VmafMeter::finish() {
  ensure_started();
  if (finished_) return;
  if (submitted_ <= 0) throw std::runtime_error("VmafMeter::finish with no frames");

  check_err(vmaf_read_pictures(static_cast<VmafContext*>(vmaf_), nullptr, nullptr, 0),
            "vmaf_read_pictures flush");

  auto* vmaf = static_cast<VmafContext*>(vmaf_);
  auto* model = static_cast<VmafModel*>(model_);
  frames_.assign(static_cast<size_t>(submitted_), VmafFrame{});
  const bool have_feats = !feat_cambi_.empty();

  for (int i = 0; i < submitted_; ++i) {
    VmafFrame& fr = frames_[static_cast<size_t>(i)];
    const unsigned idx = static_cast<unsigned>(i);
    check_err(vmaf_score_at_index(vmaf, model, &fr.score, idx), "vmaf_score_at_index");
    if (have_feats) {
      check_err(vmaf_feature_score_at_index(vmaf, feat_cambi_.c_str(), &fr.cambi, idx),
                "feature cambi");
      check_err(
          vmaf_feature_score_at_index(vmaf, feat_speed_chroma_.c_str(), &fr.speed_chroma, idx),
          "feature speed_chroma");
      check_err(vmaf_feature_score_at_index(vmaf, feat_adm3_.c_str(), &fr.adm3, idx),
                "feature adm3");
      check_err(vmaf_feature_score_at_index(vmaf, feat_motion3_.c_str(), &fr.motion3, idx),
                "feature motion3");
      fr.features_valid = true;
    }
    fr.valid = true;
  }

  const unsigned lo = 0;
  const unsigned hi = static_cast<unsigned>(submitted_ - 1);
  check_err(vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN, &pooled_mean_, lo, hi),
            "vmaf_score_pooled mean");
  check_err(vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_HARMONIC_MEAN, &pooled_harmonic_, lo,
                              hi),
            "vmaf_score_pooled harmonic_mean");
  check_err(vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MIN, &pooled_min_, lo, hi),
            "vmaf_score_pooled min");
  check_err(vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MAX, &pooled_max_, lo, hi),
            "vmaf_score_pooled max");
  finished_ = true;
}

VmafSummary VmafMeter::summary() const {
  ensure_started();
  if (!finished_) throw std::runtime_error("VmafMeter::summary before finish");
  VmafSummary out;
  out.frames = submitted_;
  out.mean = pooled_mean_;
  out.harmonic_mean = pooled_harmonic_;
  out.min = pooled_min_;
  out.max = pooled_max_;
  return out;
}

}  // namespace pixvqa
