#include "pixvqa/decode.hpp"

#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
#ifdef PIXVQA_HAS_NVDEC
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#endif
}

#ifdef PIXVQA_HAS_NVDEC
#include "pixvqa/cuda_backend.hpp"
#endif

namespace pixvqa {
namespace {

std::string av_error(int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof(buf));
  return buf;
}

void ffmpeg_log(void* /*avcl*/, int level, const char* fmt, va_list ap) {
  if (level > av_log_get_level()) return;
  char line[1024];
  std::vsnprintf(line, sizeof(line), fmt, ap);
  std::fputs(line, stderr);
}

HdrTransfer map_trc(int v) {
  switch (static_cast<AVColorTransferCharacteristic>(v)) {
    case AVCOL_TRC_RESERVED0:
    case AVCOL_TRC_UNSPECIFIED:
      return HdrTransfer::Unspecified;
    case AVCOL_TRC_BT709:
      return HdrTransfer::Bt709;
    case AVCOL_TRC_BT2020_10:
    case AVCOL_TRC_BT2020_12:
      return HdrTransfer::Bt2020_10;
    case AVCOL_TRC_SMPTE170M:
      return HdrTransfer::Smpte170m;
    case AVCOL_TRC_GAMMA22:
    case AVCOL_TRC_GAMMA28:
      return HdrTransfer::Other;
    case AVCOL_TRC_LINEAR:
      return HdrTransfer::Linear;
    case AVCOL_TRC_SMPTE2084:
      return HdrTransfer::Pq;
    case AVCOL_TRC_ARIB_STD_B67:
      return HdrTransfer::Hlg;
    default:
      return HdrTransfer::Other;
  }
}

HdrPrimaries map_primaries(int v) {
  switch (static_cast<AVColorPrimaries>(v)) {
    case AVCOL_PRI_RESERVED0:
    case AVCOL_PRI_UNSPECIFIED:
      return HdrPrimaries::Unspecified;
    case AVCOL_PRI_BT709:
      return HdrPrimaries::Bt709;
    case AVCOL_PRI_BT2020:
      return HdrPrimaries::Bt2020;
    case AVCOL_PRI_SMPTE170M:
    case AVCOL_PRI_BT470BG:
    case AVCOL_PRI_BT470M:
      return HdrPrimaries::Bt601;
    default:
      return HdrPrimaries::Other;
  }
}

HdrMatrix map_matrix(int v) {
  switch (static_cast<AVColorSpace>(v)) {
    case AVCOL_SPC_UNSPECIFIED:
      return HdrMatrix::Unspecified;
    case AVCOL_SPC_RGB:
      return HdrMatrix::Other;
    case AVCOL_SPC_BT709:
      return HdrMatrix::Bt709;
    case AVCOL_SPC_BT2020_NCL:
      return HdrMatrix::Bt2020Ncl;
    case AVCOL_SPC_SMPTE170M:
    case AVCOL_SPC_BT470BG:
      return HdrMatrix::Bt601;
    default:
      return HdrMatrix::Other;
  }
}

HdrRange map_range(int v) {
  switch (static_cast<AVColorRange>(v)) {
    case AVCOL_RANGE_UNSPECIFIED:
      return HdrRange::Unspecified;
    case AVCOL_RANGE_MPEG:
      return HdrRange::Limited;
    case AVCOL_RANGE_JPEG:
      return HdrRange::Full;
    default:
      return HdrRange::Unspecified;
  }
}

void observe_chroma_location(HdrSide& side, int loc) {
  bool unsupported = false;
  HdrChromaLoc mapped = HdrChromaLoc::Unspecified;
  switch (static_cast<AVChromaLocation>(loc)) {
    case AVCHROMA_LOC_UNSPECIFIED:
      break;
    case AVCHROMA_LOC_LEFT:
      mapped = HdrChromaLoc::Left;
      break;
    case AVCHROMA_LOC_CENTER:
      mapped = HdrChromaLoc::Center;
      break;
    case AVCHROMA_LOC_TOPLEFT:
      mapped = HdrChromaLoc::TopLeft;
      break;
    default:
      unsupported = true;
      break;
  }
  hdr_observe_chroma(side, mapped, unsupported);
}

void copy_plane_bytes(uint16_t* dst, int width, int height, const uint8_t* src, int linesize, int bytes,
                      std::string& err) {
  if (!src || linesize < width * bytes) {
    err = "decoded plane stride is smaller than the picture width";
    return;
  }
  if (bytes == 2) {
    const size_t row = static_cast<size_t>(width) * sizeof(uint16_t);
    for (int y = 0; y < height; ++y) {
      std::memcpy(dst + static_cast<size_t>(y) * static_cast<size_t>(width),
                  src + static_cast<size_t>(y) * static_cast<size_t>(linesize), row);
    }
    return;
  }
  for (int y = 0; y < height; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * static_cast<size_t>(linesize);
    uint16_t* out = dst + static_cast<size_t>(y) * static_cast<size_t>(width);
    for (int x = 0; x < width; ++x) out[x] = row[x];
  }
}

struct SwLayout {
  int bit_depth = 0;
  ChromaFormat chroma = ChromaFormat::YUV420;
  bool jpeg_full = false;
};

bool software_layout(AVPixelFormat fmt, SwLayout& out) {
  out = SwLayout{};
  switch (fmt) {
    case AV_PIX_FMT_YUV420P:
      out = {8, ChromaFormat::YUV420, false};
      return true;
    case AV_PIX_FMT_YUV422P:
      out = {8, ChromaFormat::YUV422, false};
      return true;
    case AV_PIX_FMT_YUV444P:
      out = {8, ChromaFormat::YUV444, false};
      return true;
    case AV_PIX_FMT_YUVJ420P:
      out = {8, ChromaFormat::YUV420, true};
      return true;
    case AV_PIX_FMT_YUVJ422P:
      out = {8, ChromaFormat::YUV422, true};
      return true;
    case AV_PIX_FMT_YUVJ444P:
      out = {8, ChromaFormat::YUV444, true};
      return true;
    case AV_PIX_FMT_YUV420P10LE:
      out = {10, ChromaFormat::YUV420, false};
      return true;
    case AV_PIX_FMT_YUV422P10LE:
      out = {10, ChromaFormat::YUV422, false};
      return true;
    case AV_PIX_FMT_YUV444P10LE:
      out = {10, ChromaFormat::YUV444, false};
      return true;
    default:
      return false;
  }
}

const char* chroma_name(ChromaFormat c) {
  if (c == ChromaFormat::YUV422) return "4:2:2";
  if (c == ChromaFormat::YUV444) return "4:4:4";
  return "4:2:0";
}

std::string layout_text(const SwLayout& layout) {
  return std::to_string(layout.bit_depth) + "-bit " + chroma_name(layout.chroma);
}

const AVCodec* software_decoder(AVCodecID id) {
  // The native "av1" decoder asks for a hardware frame and returns ENOSYS.
  // A registered native decoder is not treated as a working software decoder.
  if (id == AV_CODEC_ID_AV1) {
    if (const AVCodec* dav1d = avcodec_find_decoder_by_name("libdav1d")) return dav1d;
    if (const AVCodec* aom = avcodec_find_decoder_by_name("libaom-av1")) return aom;
    return nullptr;
  }
  if (id == AV_CODEC_ID_VVC) return avcodec_find_decoder_by_name("vvc");
  return avcodec_find_decoder(id);
}

#ifdef PIXVQA_HAS_NVDEC
bool copy_nv12_planar(Frame& out, const VideoFormat& fmt, const AVFrame* sw, std::string& err) {
  if (!sw || sw->format != AV_PIX_FMT_NV12 || !sw->data[0] || !sw->data[1]) {
    err = "NVDEC NV12 download did not produce an nv12 picture";
    return false;
  }
  if (sw->linesize[0] < fmt.width || sw->linesize[1] < fmt.width) {
    err = "NVDEC NV12 stride is smaller than the picture width";
    return false;
  }
  out.allocate(fmt);
  for (int y = 0; y < fmt.height; ++y) {
    const uint8_t* row = sw->data[0] + static_cast<ptrdiff_t>(y) * sw->linesize[0];
    uint16_t* dst = out.y.ptr() + static_cast<ptrdiff_t>(y) * fmt.width;
    for (int x = 0; x < fmt.width; ++x) dst[x] = row[x];
  }
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  for (int y = 0; y < ch; ++y) {
    const uint8_t* row = sw->data[1] + static_cast<ptrdiff_t>(y) * sw->linesize[1];
    uint16_t* u = out.u.ptr() + static_cast<ptrdiff_t>(y) * cw;
    uint16_t* v = out.v.ptr() + static_cast<ptrdiff_t>(y) * cw;
    for (int x = 0; x < cw; ++x) {
      u[x] = row[x * 2];
      v[x] = row[x * 2 + 1];
    }
  }
  return true;
}
#endif

std::string pix_name(AVPixelFormat fmt) {
  const char* name = av_get_pix_fmt_name(fmt);
  return name ? name : "unknown";
}

}  // namespace

bool av1_software_available() {
  return avcodec_find_decoder_by_name("libdav1d") != nullptr ||
         avcodec_find_decoder_by_name("libaom-av1") != nullptr;
}

bool probe_video_input(const std::string& path, std::string* codec_name) {
  if (codec_name) codec_name->clear();
  struct stat st;
  if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
  AVFormatContext* fmt = nullptr;
  const int saved = av_log_get_level();
  av_log_set_level(AV_LOG_QUIET);
  const int opened = avformat_open_input(&fmt, path.c_str(), nullptr, nullptr);
  bool video = false;
  if (opened >= 0) {
    const int best = avformat_find_stream_info(fmt, nullptr) >= 0
                          ? av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0)
                          : AVERROR_STREAM_NOT_FOUND;
    if (best >= 0) {
      video = true;
      if (codec_name) *codec_name = avcodec_get_name(fmt->streams[best]->codecpar->codec_id);
    }
    avformat_close_input(&fmt);
  }
  av_log_set_level(saved);
  return video;
}

struct CompressedTestReader::Impl {
  AVFormatContext* fmt = nullptr;
  AVCodecContext* ctx = nullptr;
  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;
  int video_index = -1;
  bool draining = false;
  bool input_eof = false;
  int status = 0;
  int frames = 0;
  std::string role = "test";
  VideoFormat expected{};
  FILE* dump = nullptr;
  bool profile = false;
  double decode_s = 0.0;
  double p010_s = 0.0;
  double download_s = 0.0;
  bool logged_threads = false;
  bool nvdec = false;
  std::string codec_name = "video";
  HdrSide hdr{};
#ifdef PIXVQA_HAS_NVDEC
  AVBufferRef* hw_device = nullptr;
  CUstream dec_stream = nullptr;
  uintptr_t primary = 0;
#endif

  ~Impl() { reset(); }

  void reset() {
    if (profile) {
      std::fprintf(stderr,
                   "pixvqa: decode_profile seconds=%.6f p010_seconds=%.6f download_seconds=%.6f frames=%d nvdec=%d\n",
                   decode_s, p010_s, download_s, frames, nvdec ? 1 : 0);
      profile = false;
    }
#ifdef PIXVQA_HAS_NVDEC
    if (hw_device) av_buffer_unref(&hw_device);
    dec_stream = nullptr;
    primary = 0;
#endif
    nvdec = false;
    if (dump) {
      std::fclose(dump);
      dump = nullptr;
    }
    if (pkt) {
      av_packet_free(&pkt);
    }
    if (frame) {
      av_frame_free(&frame);
    }
    if (ctx) {
      avcodec_free_context(&ctx);
    }
    if (fmt) {
      avformat_close_input(&fmt);
    }
    draining = false;
    input_eof = false;
  }
};

CompressedTestReader::CompressedTestReader() : impl_(new Impl) {}

CompressedTestReader::~CompressedTestReader() { delete impl_; }

int CompressedTestReader::status() const { return impl_ ? impl_->status : 2; }

int CompressedTestReader::frames_read() const { return impl_ ? impl_->frames : 0; }

HdrSide CompressedTestReader::stream_hdr() const { return impl_ ? impl_->hdr : HdrSide{}; }

void CompressedTestReader::close() {
  if (impl_) impl_->reset();
}

bool CompressedTestReader::open(const std::string& path, const VideoFormat& expected, DecodeMode mode,
                                std::string& err, const char* role) {
  err.clear();
  impl_->status = 2;
  impl_->role = (role && role[0]) ? role : "test";
  impl_->expected = expected;
  if (mode != DecodeMode::Software && mode != DecodeMode::Nvdec) {
    err = "unsupported decode mode";
    return false;
  }
#ifndef PIXVQA_HAS_NVDEC
  if (mode == DecodeMode::Nvdec) {
    err = "NVDEC decoding is not compiled into this binary";
    return false;
  }
#endif
  if (expected.bit_depth != 8 && expected.bit_depth != 10) {
    err = "software decoding supports 8-bit and 10-bit planar YUV only";
    return false;
  }

  static bool log_ready = false;
  if (!log_ready) {
    av_log_set_level(AV_LOG_ERROR);
    av_log_set_callback(ffmpeg_log);
    log_ready = true;
  }

  const char* prof = pix_getenv("DECODE_PROFILE");
  impl_->profile = prof && prof[0] == '1';
  const char* dump_path = pix_getenv("DECODE_YUV");
  if (dump_path && dump_path[0] && impl_->role == "test") {
    impl_->dump = std::fopen(dump_path, "wb");
    if (!impl_->dump) {
      err = std::string("could not open decode dump ") + dump_path;
      return false;
    }
  }

  int ret = avformat_open_input(&impl_->fmt, path.c_str(), nullptr, nullptr);
  if (ret < 0) {
    err = "failed to open " + impl_->role + " input: " + av_error(ret);
    return false;
  }
  ret = avformat_find_stream_info(impl_->fmt, nullptr);
  if (ret < 0) {
    err = "failed to read " + impl_->role + " stream information: " + av_error(ret);
    impl_->reset();
    return false;
  }
  ret = av_find_best_stream(impl_->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (ret < 0) {
    err = impl_->role + " input has no video stream: " + av_error(ret);
    impl_->reset();
    return false;
  }
  impl_->video_index = ret;
  AVStream* stream = impl_->fmt->streams[impl_->video_index];
  const AVCodecParameters* par = stream->codecpar;
  impl_->codec_name = avcodec_get_name(par->codec_id);
  const AVCodec* codec = nullptr;
  if (mode == DecodeMode::Nvdec) {
    if (expected.chroma != ChromaFormat::YUV420 ||
        (expected.bit_depth != 8 && expected.bit_depth != 10)) {
      err = "GPU decoding supports 8-bit and 10-bit 4:2:0 only";
      impl_->reset();
      return false;
    }
    if (par->codec_id == AV_CODEC_ID_HEVC) {
      if (expected.bit_depth != 10) {
        err = "GPU decoding supports HEVC at 420p:10 only";
        impl_->reset();
        return false;
      }
      codec = avcodec_find_decoder_by_name("hevc");
    } else if (par->codec_id == AV_CODEC_ID_AV1) {
      codec = avcodec_find_decoder_by_name("av1_cuvid");
      if (!codec) {
        err = "AV1 GPU decoding requires the av1_cuvid decoder in this FFmpeg build";
        impl_->reset();
        return false;
      }
    } else if (par->codec_id == AV_CODEC_ID_VVC) {
      err = impl_->role + " codec is vvc; GPU decoding does not support VVC on this NVIDIA NVDEC path";
      impl_->reset();
      return false;
    } else {
      err = impl_->role + " codec is " + impl_->codec_name + "; GPU decoding supports HEVC and AV1 only";
      impl_->reset();
      return false;
    }
  } else {
    codec = software_decoder(par->codec_id);
  }
  if (!codec) {
    if (mode != DecodeMode::Nvdec && par->codec_id == AV_CODEC_ID_AV1) {
      err = "AV1 software decoding requires libdav1d or libaom-av1; the native av1 decoder is not used";
    } else {
      err = impl_->role + " codec " + impl_->codec_name + " has no decoder in this FFmpeg build";
    }
    impl_->reset();
    return false;
  }
  impl_->ctx = avcodec_alloc_context3(codec);
  if (!impl_->ctx) {
    err = "failed to allocate the " + impl_->codec_name + " decoder";
    impl_->reset();
    return false;
  }
  ret = avcodec_parameters_to_context(impl_->ctx, par);
  if (ret < 0) {
    err = "failed to copy " + impl_->codec_name + " parameters: " + av_error(ret);
    impl_->reset();
    return false;
  }
  impl_->ctx->pkt_timebase = stream->time_base;
  if (mode == DecodeMode::Nvdec) {
#ifdef PIXVQA_HAS_NVDEC
    try {
      cuda_ensure_primary_context();
    } catch (const std::exception& ex) {
      err = ex.what();
      impl_->reset();
      return false;
    }
    impl_->primary = cuda_primary_context_handle();
    // The runtime primary context is already current. Retaining it again with
    // FFmpeg's own flags fails once that context exists. Use the current one
    // and require the pointers to match.
    const int hw = av_hwdevice_ctx_create(&impl_->hw_device, AV_HWDEVICE_TYPE_CUDA, "0", nullptr,
                                          AV_CUDA_USE_CURRENT_CONTEXT);
    if (hw < 0) {
      err = "failed to open the CUDA device for NVDEC: " + av_error(hw);
      impl_->reset();
      return false;
    }
    auto* hwdev = reinterpret_cast<AVHWDeviceContext*>(impl_->hw_device->data);
    auto* cu = reinterpret_cast<AVCUDADeviceContext*>(hwdev->hwctx);
    if (!cu || reinterpret_cast<uintptr_t>(cu->cuda_ctx) != impl_->primary) {
      err = "FFmpeg CUDA context does not match the NVIDIA primary context";
      impl_->reset();
      return false;
    }
    impl_->dec_stream = cu->stream;
    impl_->ctx->hw_device_ctx = av_buffer_ref(impl_->hw_device);
    impl_->ctx->get_format = [](AVCodecContext*, const enum AVPixelFormat* pix) -> AVPixelFormat {
      for (const enum AVPixelFormat* p = pix; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) return *p;
      }
      return AV_PIX_FMT_NONE;
    };
    impl_->ctx->thread_count = 1;
    impl_->nvdec = true;
    std::fprintf(stderr, "pixvqa: nvdec primary context matches FFmpeg cuda_ctx\n");
#else
    err = "NVDEC decoding is not compiled into this binary";
    impl_->reset();
    return false;
#endif
  } else {
    // Fixed after the software pixel oracle. Do not retune from --threads.
    impl_->ctx->thread_count = 8;
    impl_->ctx->thread_type = FF_THREAD_FRAME;
  }
  AVDictionary* open_opts = nullptr;
  if (!impl_->nvdec && codec->name && std::strcmp(codec->name, "libdav1d") == 0) {
    // Library defaults. -1 applies film grain only when the bitstream has it,
    // and selects the default operating point. Both sides of a comparison use this.
    av_dict_set(&open_opts, "filmgrain", "-1", 0);
    av_dict_set(&open_opts, "oppoint", "-1", 0);
  }
  ret = avcodec_open2(impl_->ctx, codec, &open_opts);
  av_dict_free(&open_opts);
  if (ret < 0) {
    err = "failed to open the " + impl_->codec_name + " decoder: " + av_error(ret);
    impl_->reset();
    return false;
  }
  if (par->width > 0 && par->height > 0 &&
      (par->width != expected.width || par->height != expected.height)) {
    err = impl_->role + " picture is " + std::to_string(par->width) + "x" + std::to_string(par->height) +
          ", expected " + std::to_string(expected.width) + "x" + std::to_string(expected.height);
    impl_->reset();
    return false;
  }
  if (!impl_->nvdec && par->format != AV_PIX_FMT_NONE) {
    SwLayout layout;
    const auto fmt = static_cast<AVPixelFormat>(par->format);
    if (!software_layout(fmt, layout)) {
      err = impl_->role + " pixel format is " + pix_name(fmt) +
            "; supported software layouts are planar 8-bit and 10-bit 4:2:0, 4:2:2, and 4:4:4";
      impl_->reset();
      return false;
    }
    if (layout.bit_depth != expected.bit_depth || layout.chroma != expected.chroma) {
      err = impl_->role + " pixel format is " + pix_name(fmt) + " (" + layout_text(layout) +
            "); -f asks for " + layout_text(SwLayout{expected.bit_depth, expected.chroma, false});
      impl_->reset();
      return false;
    }
  }
  impl_->pkt = av_packet_alloc();
  impl_->frame = av_frame_alloc();
  if (!impl_->pkt || !impl_->frame) {
    err = "failed to allocate decoder packet or frame";
    impl_->reset();
    return false;
  }
  impl_->status = 0;
  hdr_observe(impl_->hdr, map_trc(par->color_trc), map_primaries(par->color_primaries),
              map_matrix(par->color_space), map_range(par->color_range));
  observe_chroma_location(impl_->hdr, par->chroma_location);
  std::fprintf(stderr, "pixvqa: %s decoder=%s backend=%s\n", impl_->role.c_str(), codec->name,
               impl_->nvdec ? "nvdec" : "software");
  return true;
}

bool CompressedTestReader::read_frame(Frame& out, std::string& err, bool copy_to_host) {
  err.clear();
  if (!copy_to_host && !impl_->nvdec) {
    err = "skipping the host copy is only valid for NVDEC";
    impl_->status = 2;
    return false;
  }
  Impl& d = *impl_;
  const auto t0 = std::chrono::steady_clock::now();
  auto finish_time = [&]() {
    if (!d.profile) return;
    const auto t1 = std::chrono::steady_clock::now();
    d.decode_s += std::chrono::duration<double>(t1 - t0).count();
  };

  for (;;) {
    int ret = avcodec_receive_frame(d.ctx, d.frame);
    if (ret == AVERROR(EAGAIN)) {
      if (d.draining) {
        d.status = 1;
        err = d.codec_name + " decoder stalled while draining";
        finish_time();
        return false;
      }
      if (!d.input_eof) {
        ret = av_read_frame(d.fmt, d.pkt);
        if (ret == AVERROR_EOF) {
          d.input_eof = true;
        } else if (ret < 0) {
          d.status = 1;
          err = "failed to read a " + d.role + " packet: " + av_error(ret);
          finish_time();
          return false;
        } else if (d.pkt->stream_index != d.video_index) {
          av_packet_unref(d.pkt);
          continue;
        } else {
          const int sent = avcodec_send_packet(d.ctx, d.pkt);
          av_packet_unref(d.pkt);
          if (sent < 0) {
            d.status = 1;
            if (sent == AVERROR(ENOSYS)) {
              err = "the " + d.codec_name +
                    " decoder in this FFmpeg build cannot produce a software frame: " +
                    av_error(sent);
            } else {
              err = "failed to send a " + d.codec_name + " packet: " + av_error(sent);
            }
            finish_time();
            return false;
          }
          continue;
        }
      }
      if (d.input_eof && !d.draining) {
        const int sent = avcodec_send_packet(d.ctx, nullptr);
        d.draining = true;
        if (sent < 0) {
          d.status = 1;
          err = "failed to drain the " + d.codec_name + " decoder: " + av_error(sent);
          finish_time();
          return false;
        }
      }
      continue;
    }
    if (ret == AVERROR_EOF) {
      finish_time();
      return false;
    }
    if (ret < 0) {
      d.status = 1;
      err = d.codec_name + " decoder failed on a picture: " + av_error(ret);
      finish_time();
      return false;
    }

    const int first = d.frames == 0;
    hdr_observe(d.hdr, map_trc(d.frame->color_trc), map_primaries(d.frame->color_primaries),
                map_matrix(d.frame->colorspace), map_range(d.frame->color_range));
    observe_chroma_location(d.hdr, d.frame->chroma_location);
#ifdef PIXVQA_HAS_NVDEC
    if (d.nvdec) {
      if (d.frame->format != AV_PIX_FMT_CUDA || !d.frame->hw_frames_ctx) {
        err = "NVDEC picture is not a CUDA hardware frame";
        d.status = first ? 2 : 1;
        av_frame_unref(d.frame);
        finish_time();
        return false;
      }
      auto* frames = reinterpret_cast<AVHWFramesContext*>(d.frame->hw_frames_ctx->data);
      const bool size_ok = d.frame->width == d.expected.width && d.frame->height == d.expected.height;
      const bool p010 = frames->sw_format == AV_PIX_FMT_P010LE && d.expected.bit_depth == 10;
      const bool nv12 = frames->sw_format == AV_PIX_FMT_NV12 && d.expected.bit_depth == 8;
      if ((!p010 && !nv12) || !size_ok) {
        const char* name = av_get_pix_fmt_name(frames->sw_format);
        err = std::string("NVDEC picture is ") + std::to_string(d.frame->width) + "x" +
              std::to_string(d.frame->height) + " " + (name ? name : "unknown") +
              "; 8-bit nv12 or 10-bit p010 at " + std::to_string(d.expected.width) + "x" +
              std::to_string(d.expected.height) + " is required";
        d.status = first ? 2 : 1;
        av_frame_unref(d.frame);
        finish_time();
        return false;
      }
      if (nv12 && !copy_to_host) {
        err = "8-bit NVDEC pictures are NV12 and are deinterleaved on the host; GPU-resident metrics stay on the 10-bit P010 path";
        d.status = 2;
        av_frame_unref(d.frame);
        finish_time();
        return false;
      }
      CUcontext cur = nullptr;
      cuCtxGetCurrent(&cur);
      if (reinterpret_cast<uintptr_t>(cur) != d.primary &&
          cuCtxSetCurrent(reinterpret_cast<CUcontext>(d.primary)) != CUDA_SUCCESS) {
        err = "could not make the shared CUDA primary context current";
        d.status = 1;
        av_frame_unref(d.frame);
        finish_time();
        return false;
      }
      if (cuStreamSynchronize(d.dec_stream) != CUDA_SUCCESS) {
        err = "failed to synchronize the NVDEC stream";
        d.status = 1;
        av_frame_unref(d.frame);
        finish_time();
        return false;
      }
      if (nv12) {
        AVFrame* sw = av_frame_alloc();
        const int transferred = sw ? av_hwframe_transfer_data(sw, d.frame, 0) : AVERROR(ENOMEM);
        if (transferred < 0 || !copy_nv12_planar(out, d.expected, sw, err)) {
          if (err.empty()) err = "failed to download the NVDEC NV12 picture: " + av_error(transferred);
          av_frame_free(&sw);
          d.status = 1;
          av_frame_unref(d.frame);
          finish_time();
          return false;
        }
        av_frame_free(&sw);
        av_frame_unref(d.frame);
        ++d.frames;
        finish_time();
        return true;
      }
      const auto conv_t0 = std::chrono::steady_clock::now();
      try {
        P010DeviceSurface surface;
        surface.y = d.frame->data[0];
        surface.y_pitch_bytes = d.frame->linesize[0];
        surface.uv = d.frame->data[1];
        surface.uv_pitch_bytes = d.frame->linesize[1];
        cuda_convert_p010_to_test_planes(surface, d.expected);
        if (d.profile) {
          d.p010_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - conv_t0).count();
        }
        if (copy_to_host) {
          const auto dl_t0 = std::chrono::steady_clock::now();
          cuda_download_test_planes(out, d.expected);
          if (d.profile) {
            d.download_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_t0).count();
          }
        }
      } catch (const std::exception& ex) {
        err = ex.what();
        d.status = 1;
        av_frame_unref(d.frame);
        finish_time();
        return false;
      }
      if (d.profile) {
        if (!d.logged_threads) {
          std::fprintf(stderr, "pixvqa: nvdec p010 linesize_bytes y=%d uv=%d pitch_samples=%d\n",
                       d.frame->linesize[0], d.frame->linesize[1], d.expected.width);
          d.logged_threads = true;
        }
      }
      av_frame_unref(d.frame);
      if (!copy_to_host) {
        if (d.dump) {
          err = "GPU-resident NVDEC does not download the test picture for PIXVQA_DECODE_YUV";
          d.status = 2;
          finish_time();
          return false;
        }
        ++d.frames;
        finish_time();
        return true;
      }
      if (d.dump) {
        const size_t ys = static_cast<size_t>(d.expected.width) * d.expected.height;
        const size_t cs = static_cast<size_t>(d.expected.chroma_width()) * d.expected.chroma_height();
        if (std::fwrite(out.y.ptr(), sizeof(uint16_t), ys, d.dump) != ys ||
            std::fwrite(out.u.ptr(), sizeof(uint16_t), cs, d.dump) != cs ||
            std::fwrite(out.v.ptr(), sizeof(uint16_t), cs, d.dump) != cs) {
          err = "failed to write the decoded picture dump";
          d.status = 1;
          finish_time();
          return false;
        }
      }
      ++d.frames;
      finish_time();
      return true;
    }
#endif
    SwLayout layout;
    const auto picture_fmt = static_cast<AVPixelFormat>(d.frame->format);
    if (!software_layout(picture_fmt, layout) || layout.bit_depth != d.expected.bit_depth ||
        layout.chroma != d.expected.chroma || d.frame->width != d.expected.width ||
        d.frame->height != d.expected.height) {
      err = "decoded picture is " + std::to_string(d.frame->width) + "x" +
            std::to_string(d.frame->height) + " " + pix_name(picture_fmt);
      if (software_layout(picture_fmt, layout)) {
        err += " (" + layout_text(layout) + ")";
      }
      err += "; -f asks for " + layout_text(SwLayout{d.expected.bit_depth, d.expected.chroma, false}) + " " +
             std::to_string(d.expected.width) + "x" + std::to_string(d.expected.height);
      if (!software_layout(picture_fmt, layout)) {
        err += "; supported software layouts are planar 8-bit and 10-bit 4:2:0, 4:2:2, and 4:4:4";
      }
      d.status = first ? 2 : 1;
      av_frame_unref(d.frame);
      finish_time();
      return false;
    }
    if (layout.jpeg_full && d.hdr.range == HdrRange::Unspecified) {
      hdr_observe(d.hdr, HdrTransfer::Unspecified, HdrPrimaries::Unspecified, HdrMatrix::Unspecified,
                  HdrRange::Full);
    }
    if (d.frame->linesize[0] < 0 || d.frame->linesize[1] < 0 || d.frame->linesize[2] < 0) {
      d.status = 1;
      err = "decoded picture has a negative stride";
      av_frame_unref(d.frame);
      finish_time();
      return false;
    }

    const int sample_bytes = layout.bit_depth > 8 ? 2 : 1;
    out.allocate(d.expected);
    copy_plane_bytes(out.y.ptr(), d.expected.width, d.expected.height, d.frame->data[0],
                     d.frame->linesize[0], sample_bytes, err);
    if (err.empty()) {
      copy_plane_bytes(out.u.ptr(), d.expected.chroma_width(), d.expected.chroma_height(), d.frame->data[1],
                       d.frame->linesize[1], sample_bytes, err);
    }
    if (err.empty()) {
      copy_plane_bytes(out.v.ptr(), d.expected.chroma_width(), d.expected.chroma_height(), d.frame->data[2],
                       d.frame->linesize[2], sample_bytes, err);
    }
    if (d.dump && err.empty()) {
      const size_t ys = static_cast<size_t>(d.expected.width) * d.expected.height;
      const size_t cs = static_cast<size_t>(d.expected.chroma_width()) * d.expected.chroma_height();
      if (std::fwrite(out.y.ptr(), sizeof(uint16_t), ys, d.dump) != ys ||
          std::fwrite(out.u.ptr(), sizeof(uint16_t), cs, d.dump) != cs ||
          std::fwrite(out.v.ptr(), sizeof(uint16_t), cs, d.dump) != cs) {
        err = "failed to write the decoded picture dump";
      }
    }
    av_frame_unref(d.frame);
    if (!err.empty()) {
      d.status = 1;
      finish_time();
      return false;
    }
    if (!d.logged_threads && d.profile) {
      std::fprintf(stderr, "pixvqa: software decoder thread_type=frame thread_count=%d codec=%s\n",
                   d.ctx->thread_count, d.codec_name.c_str());
      d.logged_threads = true;
    }
    ++d.frames;
    finish_time();
    return true;
  }
}

}  // namespace pixvqa
