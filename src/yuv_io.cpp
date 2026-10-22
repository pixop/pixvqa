#include "pixvqa/yuv_io.hpp"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace pixvqa {

namespace {

// Read exactly nbytes, or return bytes_read==0 on clean EOF before any data.
// Throws if a partial frame is observed.
size_t read_full_or_eof(FILE* fp, void* buf, size_t nbytes) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < nbytes) {
    size_t n = std::fread(p + got, 1, nbytes - got, fp);
    if (n == 0) {
      if (std::ferror(fp)) {
        throw std::runtime_error(std::string("read error: ") + std::strerror(errno));
      }
      if (got == 0) return 0;  // clean EOF
      throw std::runtime_error("short YUV frame read (truncated/corrupt stream)");
    }
    got += n;
  }
  return got;
}

}  // namespace

YuvReader::~YuvReader() { close(); }

bool YuvReader::open(const std::string& path, const VideoFormat& fmt) {
  close();
  path_ = path;
  fmt_ = fmt;
  if (fmt_.bit_depth != 8 && fmt_.bit_depth != 10) {
    throw std::runtime_error("only 8-bit and 10-bit planar little-endian is supported in this build");
  }
  if (fmt_.width % 2 != 0 || fmt_.height % 2 != 0) {
    if (fmt_.chroma == ChromaFormat::YUV420) {
      throw std::runtime_error("4:2:0 requires even width and height");
    }
  }
  fp_ = std::fopen(path.c_str(), "rb");
  if (!fp_) {
    throw std::runtime_error("failed to open " + path + ": " + std::strerror(errno));
  }
  setvbuf(fp_, nullptr, _IOFBF, 1 << 20);

  struct stat st {};
  if (fstat(fileno(fp_), &st) == 0) {
    is_fifo_ = S_ISFIFO(st.st_mode) != 0;
  }
  scratch_.resize(fmt_.frame_bytes());
  return true;
}

void YuvReader::close() {
  if (fp_) {
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

bool YuvReader::read_frame(Frame& out) {
  if (!fp_) throw std::runtime_error("YuvReader not open");
  out.allocate(fmt_);
  size_t n = read_full_or_eof(fp_, scratch_.data(), scratch_.size());
  if (n == 0) return false;
  const size_t ny = fmt_.plane_samples('y');
  const size_t nu = fmt_.plane_samples('u');
  const size_t nv = fmt_.plane_samples('v');
  if (fmt_.bit_depth == 8) {
    const auto* src = scratch_.data();
    auto expand = [](uint16_t* dst, const uint8_t* s, size_t count) {
      for (size_t i = 0; i < count; ++i) dst[i] = s[i];
    };
    expand(out.y.ptr(), src, ny);
    expand(out.u.ptr(), src + ny, nu);
    expand(out.v.ptr(), src + ny + nu, nv);
    return true;
  }
  const auto* src = reinterpret_cast<const uint16_t*>(scratch_.data());
  std::memcpy(out.y.ptr(), src, ny * 2);
  std::memcpy(out.u.ptr(), src + ny, nu * 2);
  std::memcpy(out.v.ptr(), src + ny + nu, nv * 2);
  return true;
}

bool parse_size(const std::string& s, int& w, int& h) {
  const auto x = s.find('x');
  if (x == std::string::npos) return false;
  try {
    w = std::stoi(s.substr(0, x));
    h = std::stoi(s.substr(x + 1));
  } catch (...) {
    return false;
  }
  return w > 0 && h > 0;
}

bool parse_format(const std::string& s, ChromaFormat& chroma, int& bit_depth) {
  std::string t = s;
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  bit_depth = 10;
  auto colon = t.find(':');
  if (colon != std::string::npos) {
    const std::string bits = t.substr(colon + 1);
    size_t idx = 0;
    try {
      bit_depth = std::stoi(bits, &idx);
    } catch (...) {
      return false;
    }
    if (idx != bits.size()) return false;
    t = t.substr(0, colon);
  }
  if (t.find("444") != std::string::npos) chroma = ChromaFormat::YUV444;
  else if (t.find("422") != std::string::npos) chroma = ChromaFormat::YUV422;
  else if (t.find("420") != std::string::npos) chroma = ChromaFormat::YUV420;
  else return false;
  return bit_depth == 8 || bit_depth == 10;
}

}  // namespace pixvqa
