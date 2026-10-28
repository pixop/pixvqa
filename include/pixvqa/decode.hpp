#pragma once

#include "pixvqa/hdr.hpp"
#include "pixvqa/types.hpp"

#include <string>

namespace pixvqa {

// One independent compressed-video decoder. Use a separate instance for each input.
// The codec comes from the stream FFmpeg opens. Instances do not share packets,
// frames, or codec state. Pictures are returned in presentation order. The caller
// pairs them in that order and does not drop or repeat frames to force a match.
class CompressedTestReader {
 public:
  CompressedTestReader();
  ~CompressedTestReader();

  CompressedTestReader(const CompressedTestReader&) = delete;
  CompressedTestReader& operator=(const CompressedTestReader&) = delete;

  // Open failures set status 2. err is a single-line diagnostic.
  // role is "test" or "reference" and appears in diagnostics. The decode dump
  // is opened only for role "test", so two readers do not share one file.
  bool open(const std::string& path, const VideoFormat& expected, DecodeMode mode, std::string& err,
            const char* role = "test");

  // true: one picture in presentation order, tightly packed uint16.
  // false and err empty: clean EOF after drain.
  // false and err set: failure, status 1 (or 2 if the first picture disagrees
  // with -s/-f).
  // copy_to_host is false only for the NVDEC GPU-resident path. The converted
  // planes stay in the metric device buffers and out is left untouched.
  bool read_frame(Frame& out, std::string& err, bool copy_to_host = true);

  void close();
  int status() const;
  int frames_read() const;
  // Transfer, primaries, matrix, and range seen on this input. Raw files have none.
  HdrSide stream_hdr() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

// True when FFmpeg can open a video stream in this regular file. A failure
// leaves the file as raw planar YUV. FIFOs are never probed. When codec_name
// is set, it receives FFmpeg's name for that stream (av1, vvc, hevc, ...).
bool probe_video_input(const std::string& path, std::string* codec_name = nullptr);

struct CompressedInfo {
  VideoFormat format;
  bool size_known = false;
  bool layout_known = false;
  int fps_num = 0;
  int fps_den = 1;
  int nominal_num = 0;
  int nominal_den = 1;
  FrameRateKind rate = FrameRateKind::Missing;
  std::string codec;
};

// Demuxer metadata, plus one software picture when the stream header has no
// supported pixel format or size. Does not convert samples. err is one line.
bool detect_compressed_info(const std::string& path, CompressedInfo& out, std::string& err);

// True when this process's libavcodec registered libdav1d or libaom-av1.
bool av1_software_available();

}  // namespace pixvqa
