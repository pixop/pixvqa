#pragma once

#include "pixvqa/types.hpp"

#include <cstdio>
#include <string>

namespace pixvqa {

class YuvReader {
 public:
  YuvReader() = default;
  ~YuvReader();

  YuvReader(const YuvReader&) = delete;
  YuvReader& operator=(const YuvReader&) = delete;

  bool open(const std::string& path, const VideoFormat& fmt);
  // Returns false on EOF. Throws on short/partial reads that are not EOF.
  bool read_frame(Frame& out);
  void close();
  const std::string& path() const { return path_; }
  bool is_fifo() const { return is_fifo_; }

 private:
  FILE* fp_ = nullptr;
  VideoFormat fmt_{};
  std::string path_;
  bool is_fifo_ = false;
  std::vector<uint8_t> scratch_;
};

bool parse_size(const std::string& s, int& w, int& h);
bool parse_format(const std::string& s, ChromaFormat& chroma, int& bit_depth);

}  // namespace pixvqa
