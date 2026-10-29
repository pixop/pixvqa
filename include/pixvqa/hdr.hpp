#pragma once

#include <string>
#include <vector>

namespace pixvqa {

// One input's color description. PQ and HLG are never inferred from bit depth.
enum class HdrTransfer {
  Unspecified,
  Sdr,
  Bt709,
  Bt2020_10,
  Smpte170m,
  Linear,
  Pq,
  Hlg,
  Other
};
enum class HdrPrimaries { Unspecified, Bt709, Bt2020, Bt601, Other };
enum class HdrMatrix { Unspecified, Bt709, Bt2020Ncl, Bt601, Other };
enum class HdrRange { Unspecified, Limited, Full };
enum class HdrSource { Absent, Stream, Override, Assumption };
enum class HdrChromaLoc { Left, Center, TopLeft, Unspecified };
enum class HdrChromaUp { Bilinear, Nearest };

struct HdrSide {
  HdrTransfer transfer = HdrTransfer::Unspecified;
  HdrPrimaries primaries = HdrPrimaries::Unspecified;
  HdrMatrix matrix = HdrMatrix::Unspecified;
  HdrRange range = HdrRange::Unspecified;
  HdrSource transfer_source = HdrSource::Absent;
  HdrSource primaries_source = HdrSource::Absent;
  HdrSource matrix_source = HdrSource::Absent;
  HdrSource range_source = HdrSource::Absent;
  // Set when two specified observations of this input disagree. An override
  // of that field clears the flag. A remaining flag rejects the run.
  bool transfer_conflict = false;
  bool primaries_conflict = false;
  bool matrix_conflict = false;
  bool range_conflict = false;
  HdrChromaLoc chroma_loc = HdrChromaLoc::Unspecified;
  HdrSource chroma_loc_source = HdrSource::Absent;
  bool chroma_loc_conflict = false;
  bool chroma_loc_unsupported = false;
};

const char* hdr_transfer_name(HdrTransfer v);
const char* hdr_primaries_name(HdrPrimaries v);
const char* hdr_matrix_name(HdrMatrix v);
const char* hdr_range_name(HdrRange v);
const char* hdr_source_name(HdrSource v);

bool hdr_parse_transfer(const std::string& token, HdrTransfer& out);
bool hdr_parse_colorspace(const std::string& token, HdrPrimaries& primaries, HdrMatrix& matrix,
                          HdrRange& range);

// Record one observation from a container or a decoded picture. Unspecified
// leaves the current value alone. A second specified value that differs sets
// the conflict flag and keeps the first value.
void hdr_observe(HdrSide& side, HdrTransfer transfer, HdrPrimaries primaries, HdrMatrix matrix,
                 HdrRange range);
void hdr_observe_chroma(HdrSide& side, HdrChromaLoc loc, bool unsupported);

// Replace fields the user set. Clears the conflict flag on each replaced field.
// Returns a diagnostic when a token is not recognized.
std::string hdr_apply_override(HdrSide& side, bool transfer_set, const std::string& transfer,
                               bool color_set, const std::string& color_space);

// Empty when the pair can continue. A disagreement inside one stream is an
// error. An explicit unsupported PQ/HLG color system is an error. Reference
// and distorted inputs may use different transfers. Missing fields are filled
// later by hdr_resolve. --hlg-peak is rejected when neither input is HLG.
std::string hdr_reject_reason(const HdrSide& reference, const HdrSide& distorted, bool hlg_peak_set,
                              double hlg_peak_nits);

// Display and chroma choices for one input. Peak, black, and gamma are applied
// only when the resolved transfer uses them.
struct HdrKnobs {
  bool peak_set = false;
  double peak_nits = 0;
  bool legacy_hlg_peak = false;
  double legacy_hlg_peak_nits = 0;
  bool black_set = false;
  double black_nits = 0;
  bool gamma_set = false;
  double gamma = 0;
  bool chroma_loc_set = false;
  HdrChromaLoc chroma_loc = HdrChromaLoc::Left;
  bool chroma_up_set = false;
  HdrChromaUp chroma_up = HdrChromaUp::Bilinear;
  // PU21 requires an explicit or streamed pq/hlg transfer. It does not infer one.
  bool require_hdr_transfer = false;
};

struct HdrEffective {
  HdrSide side;
  bool pq_clip = false;
  double peak_nits = 10000;
  HdrSource peak_source = HdrSource::Absent;
  double black_nits = 0;
  HdrSource black_source = HdrSource::Assumption;
  double hlg_gamma = 1.2;
  HdrSource gamma_source = HdrSource::Assumption;
  HdrChromaLoc chroma_loc = HdrChromaLoc::Left;
  HdrSource chroma_loc_source = HdrSource::Assumption;
  HdrChromaUp chroma_up = HdrChromaUp::Bilinear;
  HdrSource chroma_up_source = HdrSource::Assumption;
  std::vector<std::string> notices;
};

// Fills PQ/HLG defaults: BT.2020 primaries, BT.2020 non-constant-luminance
// matrix, limited range, HLG peak 1000 cd/m2, and the BT.2100 gamma for that
// peak. Returns a diagnostic when the result is still unusable.
std::string hdr_resolve(const HdrSide& side, const HdrKnobs& knobs, HdrEffective& out);

const char* hdr_chroma_loc_name(HdrChromaLoc v);
const char* hdr_chroma_up_name(HdrChromaUp v);
bool hdr_parse_chroma_loc(const std::string& token, HdrChromaLoc& out);
bool hdr_parse_chroma_up(const std::string& token, HdrChromaUp& out);
bool hdr_parse_primaries_token(const std::string& token, HdrPrimaries& out);
bool hdr_parse_matrix_token(const std::string& token, HdrMatrix& out);
bool hdr_parse_range_token(const std::string& token, HdrRange& out);

// Display-referred linear RGB in cd/m2.
// PQ is the ST 2084 EOTF on R'G'B'. An explicit peak clips each component.
// HLG is the BT.2100 inverse OETF, then the reference OOTF
// Rd = (Lw-Lb) * Ys^(gamma-1) * Rs + Lb, and the same for G and B.
// Lb = 0 reduces to Rs * (Lw * Ys^gamma) / Ys. A zero scene maps to black
// on every channel.
// Nonlinear components outside [0, 1] are clamped before the nonlinear step.
bool hdr_display_rgb(const HdrEffective& cfg, double y, double cb, double cr, double& r_nits,
                     double& g_nits, double& b_nits);

std::string hdr_display_text(const char* role, const HdrEffective& side);

std::string hdr_side_line(const char* role, const HdrSide& side);

// Empty unless the effective transfer is PQ or HLG. States the display transform.
std::string hdr_display_line(const HdrSide& side, double hlg_peak_nits);

// BT.2100 system gamma for a nominal peak Lw: 1.2 + 0.42 log10(Lw/1000).
double hdr_hlg_system_gamma(double peak_nits);

// Display luminance in cd/m2.
// y is nominal luma in about [0, 1]. cb and cr are centered, neutral at 0.
// PQ is the ST 2084 EOTF on R'G'B', then the matrix luminance weights.
// HLG is the BT.2100 inverse OETF, then Lw * Ys^gamma on the scene luminance.
// Components outside [0, 1] are clamped before the nonlinear step.
// Returns false when the transfer or matrix has no display transform.
bool hdr_display_cdm2(HdrTransfer transfer, HdrMatrix matrix, double y, double cb, double cr,
                      double hlg_peak_nits, double& nits);

// Expand a stored code to the nominal values above. Limited range uses the
// 16/219 luma and 128/224 chroma scaling of the code's bit depth.
bool hdr_codes_to_nominal(int bit_depth, HdrRange range, double y, double cb, double cr, double& yn,
                          double& cb_n, double& cr_n);
bool hdr_codes_to_nominal(int bit_depth, HdrRange range, int y, int cb, int cr, double& yn,
                          double& cb_n, double& cr_n);

}  // namespace pixvqa
