#pragma once

// Human-readable child-process diagnostics only. The learned-metric report
// protocol travels on stdout and must never pass through this formatter.
// Keeping the formatter header-only makes its byte-stream behavior testable
// without FFmpeg, libvmaf, torch, or a CUDA runtime.

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace pixvqa::backend_log {

inline bool begins(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

inline bool contains_ignore_case(std::string_view value, std::string_view needle) {
  return std::search(value.begin(), value.end(), needle.begin(), needle.end(),
                     [](char a, char b) {
                       return std::tolower(static_cast<unsigned char>(a)) ==
                              std::tolower(static_cast<unsigned char>(b));
                     }) != value.end();
}

inline std::string component_for(std::string_view backend) {
  if (backend == "CVVDP") return "pixvqa-cvvdp";
  if (backend == "CVQA-HDR-NR") return "pixvqa-cvqa-hdr-nr";
  return "pixvqa";
}

inline bool structured_severity(std::string_view text) {
  return begins(text, "info: ") || begins(text, "warning: ") || begins(text, "error: ");
}

class Relay {
 public:
  explicit Relay(std::string_view backend) : component_(component_for(backend)) {}

  template <typename Emit>
  void consume(const char* data, size_t size, Emit emit) {
    for (size_t i = 0; i < size; ++i) {
      const char ch = data[i];
      if (ch == '\n') {
        emit(format_line(line_) + "\n");
        line_.clear();
      } else {
        line_.push_back(ch);
        // Third-party progress streams may update using only carriage returns.
        // Bound retained stderr even if a backend never emits a newline.
        if (line_.size() >= 16384) {
          emit(format_line(line_) + "\n");
          line_.clear();
        }
      }
    }
  }

  template <typename Emit>
  void finish(Emit emit) {
    if (!line_.empty()) {
      emit(format_line(line_) + "\n");
      line_.clear();
    }
  }

  std::string format_line(std::string line) const {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) return "";

    // Our adapters already provide a canonical, structured prefix. Keep their
    // message and severity, including original diagnostic details.
    const std::string legacy = "pixvqa-cvqahdr: ";
    if (begins(line, legacy)) {
      line = "pixvqa-cvqa-hdr-nr: " + line.substr(legacy.size());
    }
    const std::string own = component_ + ": ";
    if (begins(line, own)) {
      if (structured_severity(std::string_view(line).substr(own.size()))) return line;
      // Old adapters did not give errors a severity.
      return own + "error: " + line.substr(own.size());
    }
    if (begins(line, "pixvqa:")) return line;  // Pre-existing core diagnostic.

    std::string_view message(line);
    if (contains_ignore_case(message, "no pretrained weights loaded")) {
      // open_clip emits this during pretrained=False construction, before
      // PixVQA loads its own checkpoint with strict=True. It is not a failed
      // model load; genuine checkpoint failures still surface separately.
      // Drop the upstream WARNING label here so it does not contradict our
      // informational severity. Retain the original model name and text.
      const auto marker = line.find("No pretrained weights loaded");
      const std::string detail = marker == std::string::npos ? line : line.substr(marker);
      return component_ + ": info: upstream open_clip: " + detail;
    }
    const char* level = "info";
    if (begins(message, "[ERROR]") || begins(message, "ERROR:") ||
               begins(message, "Error:") || begins(message, "Traceback (most recent call last):") ||
               contains_ignore_case(message, " exception:") ||
               contains_ignore_case(message, "error:")) {
      level = "error";
    } else if (begins(message, "[WARNING]") || begins(message, "WARNING:") ||
               begins(message, "Warning:") || contains_ignore_case(message, "warning:") ||
               contains_ignore_case(message, "the mean color value is less than 1")) {
      level = "warning";
    }
    // Never discard unrecognized upstream text: it may contain useful
    // photometric diagnostics or exceptions from third-party packages.
    return component_ + ": " + level + ": upstream: " + line;
  }

 private:
  std::string component_;
  std::string line_;
};

}  // namespace pixvqa::backend_log
