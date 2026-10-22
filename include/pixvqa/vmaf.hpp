#pragma once

#include "pixvqa/types.hpp"

#include <string>
#include <vector>

namespace pixvqa {

// Model JSON logical names (SVM inputs). Collector keys append option suffixes.
inline constexpr const char* kVmafFeatureCambiLogical = "Cambi_feature_cambi_score";
inline constexpr const char* kVmafFeatureSpeedChromaLogical =
    "Speed_chroma_feature_speed_chroma_uv_score";
inline constexpr const char* kVmafFeatureAdm3Logical = "VMAF_integer_feature_adm3_score";
inline constexpr const char* kVmafFeatureMotion3Logical =
    "VMAF_integer_feature_motion3_score";

struct VmafFrame {
  double score = 0.0;
  double cambi = 0.0;
  double speed_chroma = 0.0;
  double adm3 = 0.0;
  double motion3 = 0.0;
  bool valid = false;
  bool features_valid = false;
};

struct VmafSummary {
  // Temporal pools from libvmaf vmaf_score_pooled (official enums only).
  double mean = 0.0;            // VMAF_POOL_METHOD_MEAN (arithmetic)
  double harmonic_mean = 0.0;   // VMAF_POOL_METHOD_HARMONIC_MEAN
  double min = 0.0;             // VMAF_POOL_METHOD_MIN
  double max = 0.0;             // VMAF_POOL_METHOD_MAX
  int frames = 0;
};

struct VmafMeta {
  std::string model_path_requested;
  std::string model_path_resolved;
  std::string model_id;  // filename stem, for display
  std::string libvmaf_version;
  std::string backend = "cpu";
  std::string model_sha256;
  // Score clipping metadata only. Does not change libvmaf prediction.
  // When score_clip_known is false, lo/hi are unspecified (do not invent 100).
  bool score_clip_known = false;
  double score_clip_lo = 0.0;
  double score_clip_hi = 0.0;
  bool hfr_model = false;
  bool known_v1_model = false;
};

// Look up SpEED/cambi minimum luma size for a model JSON identified by SHA-256.
// Returns false when the digest is not one of the eight known v1.0.16 files.
bool vmaf_min_size_for_sha256(const std::string& sha256, int& min_w, int& min_h);

// SHA-256 (lowercase hex) of a regular file. Throws on I/O failure.
std::string vmaf_file_sha256(const std::string& path);

// Resolve to an absolute path (cwd-relative). Throws if the path is empty.
std::string vmaf_resolve_model_path(const std::string& requested);

// CPU adapter over the official libvmaf 3.2.1 C API.
// Lifecycle: start → push (in order) → finish → frame()/summary() → destroy.
// finish() flushes temporal extractors (motion3) before scores are collected.
class VmafMeter {
 public:
  VmafMeter();
  ~VmafMeter();

  VmafMeter(const VmafMeter&) = delete;
  VmafMeter& operator=(const VmafMeter&) = delete;

  // Loads the model JSON at model_path via vmaf_model_load_from_path.
  // n_threads is passed to VmafConfiguration. Format bit_depth must be 8 or 10.
  void start(const VideoFormat& fmt, const std::string& model_path, int n_threads = 1);

  // Copy host Frame planes into a VmafPicture pair and submit them.
  // 10-bit: preserve uint16 sample values. 8-bit: narrow to uint8 (no 10-bit promote).
  void push(const Frame& reference, const Frame& distorted);

  // End-of-stream flush + per-frame score/feature retrieval + pooled mean.
  void finish();

  const VmafMeta& meta() const { return meta_; }
  const std::vector<VmafFrame>& frames() const { return frames_; }
  VmafSummary summary() const;
  bool finished() const { return finished_; }
  int submitted() const { return submitted_; }

 private:
  void ensure_started() const;
  void destroy();
  void fill_picture(const Frame& src, void* pic_out);

  VideoFormat fmt_{};
  VmafMeta meta_{};
  int threads_ = 1;
  int submitted_ = 0;
  bool started_ = false;
  bool finished_ = false;
  double pooled_mean_ = 0.0;
  double pooled_harmonic_ = 0.0;
  double pooled_min_ = 0.0;
  double pooled_max_ = 0.0;
  std::vector<VmafFrame> frames_;

  // Collector feature keys for the loaded model (empty when unknown).
  std::string feat_cambi_;
  std::string feat_speed_chroma_;
  std::string feat_adm3_;
  std::string feat_motion3_;

  void* vmaf_ = nullptr;
  void* model_ = nullptr;
};

}  // namespace pixvqa
