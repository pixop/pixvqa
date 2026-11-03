#include "pixvqa/metric_resolve.hpp"

#ifdef PIXVQA_HAS_FFMPEG
#include "pixvqa/decode.hpp"
#endif

#include <cstdio>
#include <string>

namespace pixvqa {

namespace {

std::string normalized_shared_transfer(const MetricTransferCli& cli) {
  if (cli.hdr_transfer_set) return cli.hdr_transfer;
  if (cli.transfer_set) return cli.transfer;
  if (cli.ref_hdr_transfer_set && cli.dist_hdr_transfer_set &&
      transfer_family_token(cli.ref_hdr_transfer) != transfer_family_token(cli.dist_hdr_transfer)) {
    return "__conflict__";
  }
  if (cli.ref_hdr_transfer_set) return cli.ref_hdr_transfer;
  if (cli.dist_hdr_transfer_set) return cli.dist_hdr_transfer;
  if (cli.ref_transfer_set && cli.dist_transfer_set &&
      transfer_family_token(cli.ref_transfer) != transfer_family_token(cli.dist_transfer)) {
    return "__conflict__";
  }
  if (cli.ref_transfer_set) return cli.ref_transfer;
  if (cli.dist_transfer_set) return cli.dist_transfer;
  return {};
}

std::string preset_transfer_family(const MetricTransferCli& cli) {
  if (cli.preset_hdr) return "pq";
  if (cli.metric_group == "pq" || cli.metric_group == "hlg" || cli.metric_group == "sdr") {
    return cli.metric_group;
  }
  return {};
}

bool probe_side_family(const MetricSideProbe& side, std::string& family, std::string& err) {
  family.clear();
  if (!side.compressed || side.path.empty()) return true;
#ifdef PIXVQA_HAS_FFMPEG
  HdrSide hdr;
  // Missing compressed transfer tags are a hard error: the Python adapter
  // requires each compressed side to carry compatible color metadata.
  if (!probe_stream_hdr(side.path, hdr, err)) return false;
  family = hdr_side_transfer_family(hdr);
  if (family.empty()) {
    err = "compressed input '" + side.path + "' has no reliable PQ/HLG/SDR transfer in stream metadata";
    return false;
  }
  return true;
#else
  (void)side;
  err = "cannot read transfer from compressed metadata without an FFmpeg-enabled pixvqa build";
  return false;
#endif
}

void log_resolution(const char* field, const std::string& value, const char* source) {
  std::fprintf(stderr, "pixvqa: info: cvvdp %s %s (%s)\n", field, value.c_str(), source);
}

}  // namespace

std::string transfer_family_token(const std::string& token) {
  if (token.empty() || token == "auto") return {};
  if (token == "pq" || token == "st2084") return "pq";
  if (token == "hlg" || token == "stdb67") return "hlg";
  if (token == "sdr" || token == "bt709") return "sdr";
  if (token == "linear") return "linear";
  return {};
}

std::string hdr_side_transfer_family(const HdrSide& side) {
  if (side.transfer_conflict) return {};
  switch (side.transfer) {
    case HdrTransfer::Pq:
      return "pq";
    case HdrTransfer::Hlg:
      return "hlg";
    case HdrTransfer::Sdr:
    case HdrTransfer::Bt709:
    case HdrTransfer::Smpte170m:
    case HdrTransfer::Bt2020_10:
      return "sdr";
    default:
      return {};
  }
}

const char* default_cvvdp_display_for_family(const std::string& family) {
  if (family == "pq") return "standard_hdr_pq";
  if (family == "hlg") return "standard_hdr_hlg";
  if (family == "sdr") return "standard_fhd";
  return nullptr;
}

bool resolve_cvvdp_parameters(const MetricTransferCli& cli, const MetricSideProbe& reference,
                              const MetricSideProbe& test, CvvdpResolveState& cvvdp,
                              std::string& err) {
  err.clear();
  if (!cvvdp.selected) {
    cvvdp.config_resolved = true;
    return true;
  }

  const std::string preset = preset_transfer_family(cli);
  const std::string shared = normalized_shared_transfer(cli);
  if (shared == "__conflict__") {
    err = "reference and test HDR transfer overrides disagree; use one shared --hdr-transfer";
    return false;
  }
  const std::string shared_family = transfer_family_token(shared);

  if (cvvdp.interpretation_set && !preset.empty() &&
      transfer_family_token(cvvdp.interpretation) != preset) {
    err = std::string("--cvvdp-interpretation ") + cvvdp.interpretation + " conflicts with --metrics " +
          preset;
    return false;
  }
  if (cvvdp.interpretation_set && !shared_family.empty() &&
      transfer_family_token(cvvdp.interpretation) != shared_family) {
    err = std::string("--cvvdp-interpretation ") + cvvdp.interpretation +
          " conflicts with explicit HDR transfer " + shared;
    return false;
  }
  if (!preset.empty() && !shared_family.empty() && preset != shared_family) {
    err = std::string("metric preset uses ") + preset + "; explicit HDR transfer " + shared + " conflicts";
    return false;
  }

  std::string family;
  std::string family_source;
  if (cvvdp.interpretation_set) {
    family = transfer_family_token(cvvdp.interpretation);
    family_source = "explicit --cvvdp-interpretation";
  } else if (!shared_family.empty()) {
    family = shared_family;
    family_source = "explicit HDR transfer";
  } else if (!preset.empty()) {
    family = preset;
    family_source = cli.preset_hdr ? "--preset hdr" : "--metrics " + cli.metric_group;
  } else if (!cvvdp.interpretation.empty()) {
    family = transfer_family_token(cvvdp.interpretation);
    if (!family.empty()) family_source = "metric preset defaults";
  }

  std::string ref_family;
  std::string test_family;
  if (family.empty()) {
    if (reference.compressed) {
      if (!probe_side_family(reference, ref_family, err)) return false;
    }
    if (test.compressed) {
      if (!probe_side_family(test, test_family, err)) return false;
    }
    if (!ref_family.empty() && !test_family.empty() && ref_family != test_family) {
      err = "reference and test compressed streams disagree on transfer (" + ref_family + " vs " +
            test_family + "); set --hdr-transfer explicitly";
      return false;
    }
    if (!test_family.empty() && !reference.compressed) {
      family = test_family;
      family_source =
          "compressed test metadata (raw reference assumed to use the same transfer; set --hdr-transfer "
          "to override)";
    } else if (!ref_family.empty() && !test.compressed) {
      family = ref_family;
      family_source =
          "compressed reference metadata (raw test assumed to use the same transfer; set --hdr-transfer "
          "to override)";
    } else if (!ref_family.empty() && !test_family.empty()) {
      family = ref_family;
      family_source = "compressed stream metadata";
    }
  }

  if (family.empty()) {
    err =
        "could not determine CVVDP transfer; set --hdr-transfer pq|hlg|sdr, select --metrics pq|hlg|sdr, "
        "or pass --cvvdp-interpretation with a matching --cvvdp-display";
    return false;
  }
  if (family == "linear") {
    err = "CVVDP does not support linear-light interpretation; use pq, hlg, or sdr";
    return false;
  }

  if (cvvdp.interpretation.empty()) {
    cvvdp.interpretation = family;
    log_resolution("interpretation", cvvdp.interpretation, family_source.empty() ? "default" : family_source.c_str());
  }

  if (cvvdp.display.empty()) {
    const char* display = default_cvvdp_display_for_family(cvvdp.interpretation);
    if (!display) {
      err = std::string("cvvdp needs --cvvdp-display for interpretation ") + cvvdp.interpretation;
      return false;
    }
    cvvdp.display = display;
    log_resolution("display", cvvdp.display, "default for transfer");
  }

  cvvdp.config_resolved = true;
  return true;
}

}  // namespace pixvqa
