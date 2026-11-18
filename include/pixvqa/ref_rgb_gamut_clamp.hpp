#pragma once

#include "pixvqa/hdr.hpp"
#include "pixvqa/types.hpp"

#include <string>

namespace pixvqa {

// Opt-in reference-only Y'CbCr -> R'G'B' gamut clamp in the nonlinear code-value domain
// (no PQ/HLG EOTF). Matrix and range are explicit for raw input or taken from reference
// stream metadata when unambiguous.
struct RefRgbGamutClampConfig {
  HdrMatrix matrix = HdrMatrix::Unspecified;
  HdrRange range = HdrRange::Unspecified;
  HdrChromaLoc chroma_loc = HdrChromaLoc::Left;
  HdrChromaUp chroma_up = HdrChromaUp::Bilinear;
  // Optional SMPTE 240M coefficients; when false, matrix must be in HdrMatrix.
  bool smpte240m = false;
};

// Parses bt709, bt2020nc, bt601, smpte170m, bt470bg, smpte240m. Rejects bt2020c/cl.
bool ref_parse_gamut_matrix_token(const std::string& token, RefRgbGamutClampConfig& out);

// Empty on success. Resolves matrix/range/chroma for the enabled transform.
std::string ref_gamut_clamp_resolve(const Options& opt, bool ref_raw, const HdrSide& ref_stream,
                                    RefRgbGamutClampConfig& cfg);

// In-place transform on decoded reference samples before full-reference metrics.
// Returns an error message on failure.
std::string ref_gamut_clamp_frame(Frame& frame, const VideoFormat& fmt,
                                  const RefRgbGamutClampConfig& cfg);

std::string ref_gamut_clamp_summary(const RefRgbGamutClampConfig& cfg);

}  // namespace pixvqa