#pragma once

#include "pixvqa/hdr.hpp"

#include <string>

namespace pixvqa {

struct MetricTransferCli {
  bool preset_hdr = false;
  std::string metric_group;
  bool hdr_transfer_set = false;
  std::string hdr_transfer;
  bool transfer_set = false;
  std::string transfer;
  bool ref_hdr_transfer_set = false;
  std::string ref_hdr_transfer;
  bool dist_hdr_transfer_set = false;
  std::string dist_hdr_transfer;
  bool ref_transfer_set = false;
  std::string ref_transfer;
  bool dist_transfer_set = false;
  std::string dist_transfer;
};

struct MetricSideProbe {
  std::string path;
  bool compressed = false;
};

struct CvvdpResolveState {
  bool selected = false;
  bool interpretation_set = false;
  std::string interpretation;
  bool display_set = false;
  std::string display;
  bool config_resolved = false;
};

std::string transfer_family_token(const std::string& token);
std::string hdr_side_transfer_family(const HdrSide& side);
const char* default_cvvdp_display_for_family(const std::string& family);

// Fills interpretation/display when empty; validates explicit conflicts. Sets config_resolved.
bool resolve_cvvdp_parameters(const MetricTransferCli& cli, const MetricSideProbe& reference,
                              const MetricSideProbe& test, CvvdpResolveState& cvvdp,
                              std::string& err);

}  // namespace pixvqa
