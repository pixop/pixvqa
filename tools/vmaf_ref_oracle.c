/* Independent double-precision VMAF oracle using official libvmaf C API.
 * Prints JSON with %.17g decimal floats for scores, features, and all four
 * official temporal pools (mean, harmonic_mean, min, max).
 *
 * Usage:
 *   vmaf_ref_oracle --model PATH --ref REF.yuv --dist DIST.yuv \
 *     --width W --height H --bitdepth 8|10 --frames N --pixfmt 420 \
 *     --feat-cambi KEY --feat-speed KEY --feat-adm3 KEY --feat-motion3 KEY
 */
#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
#include <libvmaf/version.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char* msg) {
  fprintf(stderr, "vmaf_ref_oracle: %s\n", msg);
  exit(1);
}

static char* require(int* i, int argc, char** argv, const char* opt) {
  if (*i + 1 >= argc) die(opt);
  return argv[++(*i)];
}

static void fill_picture(VmafPicture* pic, FILE* f, int w, int h, int bitdepth, enum VmafPixelFormat fmt) {
  if (vmaf_picture_alloc(pic, fmt, (unsigned)bitdepth, (unsigned)w, (unsigned)h))
    die("vmaf_picture_alloc");
  const int cw = (fmt == VMAF_PIX_FMT_YUV420P) ? w / 2 : (fmt == VMAF_PIX_FMT_YUV422P) ? w / 2 : w;
  const int ch = (fmt == VMAF_PIX_FMT_YUV420P) ? h / 2 : h;
  const size_t y_bytes = (size_t)w * (size_t)h * (bitdepth > 8 ? 2 : 1);
  const size_t c_bytes = (size_t)cw * (size_t)ch * (bitdepth > 8 ? 2 : 1);
  unsigned char* tmp = (unsigned char*)malloc(y_bytes);
  if (!tmp) die("oom");
  if (fread(tmp, 1, y_bytes, f) != y_bytes) die("short Y");
  if (bitdepth > 8) {
    uint16_t* dst = (uint16_t*)pic->data[0];
    const uint16_t* src = (const uint16_t*)tmp;
    for (int y = 0; y < h; ++y)
      memcpy((unsigned char*)dst + y * pic->stride[0], src + y * w, (size_t)w * 2);
  } else {
    uint8_t* dst = (uint8_t*)pic->data[0];
    for (int y = 0; y < h; ++y)
      memcpy(dst + y * pic->stride[0], tmp + y * w, (size_t)w);
  }
  for (int p = 1; p <= 2; ++p) {
    if (fread(tmp, 1, c_bytes, f) != c_bytes) die("short chroma");
    if (bitdepth > 8) {
      uint16_t* dst = (uint16_t*)pic->data[p];
      const uint16_t* src = (const uint16_t*)tmp;
      for (int y = 0; y < ch; ++y)
        memcpy((unsigned char*)dst + y * pic->stride[p], src + y * cw, (size_t)cw * 2);
    } else {
      uint8_t* dst = (uint8_t*)pic->data[p];
      for (int y = 0; y < ch; ++y)
        memcpy(dst + y * pic->stride[p], tmp + y * cw, (size_t)cw);
    }
  }
  free(tmp);
}

int main(int argc, char** argv) {
  const char* model = NULL;
  const char* ref_path = NULL;
  const char* dist_path = NULL;
  const char* feat_cambi = NULL;
  const char* feat_speed = NULL;
  const char* feat_adm3 = NULL;
  const char* feat_motion3 = NULL;
  int w = 0, h = 0, bitdepth = 10, frames = 0, threads = 4;
  enum VmafPixelFormat pix = VMAF_PIX_FMT_YUV420P;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--model")) model = require(&i, argc, argv, "--model");
    else if (!strcmp(argv[i], "--ref")) ref_path = require(&i, argc, argv, "--ref");
    else if (!strcmp(argv[i], "--dist")) dist_path = require(&i, argc, argv, "--dist");
    else if (!strcmp(argv[i], "--width")) w = atoi(require(&i, argc, argv, "--width"));
    else if (!strcmp(argv[i], "--height")) h = atoi(require(&i, argc, argv, "--height"));
    else if (!strcmp(argv[i], "--bitdepth")) bitdepth = atoi(require(&i, argc, argv, "--bitdepth"));
    else if (!strcmp(argv[i], "--frames")) frames = atoi(require(&i, argc, argv, "--frames"));
    else if (!strcmp(argv[i], "--threads")) threads = atoi(require(&i, argc, argv, "--threads"));
    else if (!strcmp(argv[i], "--feat-cambi")) feat_cambi = require(&i, argc, argv, "--feat-cambi");
    else if (!strcmp(argv[i], "--feat-speed")) feat_speed = require(&i, argc, argv, "--feat-speed");
    else if (!strcmp(argv[i], "--feat-adm3")) feat_adm3 = require(&i, argc, argv, "--feat-adm3");
    else if (!strcmp(argv[i], "--feat-motion3")) feat_motion3 = require(&i, argc, argv, "--feat-motion3");
    else if (!strcmp(argv[i], "--pixfmt")) {
      const char* p = require(&i, argc, argv, "--pixfmt");
      if (!strcmp(p, "420")) pix = VMAF_PIX_FMT_YUV420P;
      else if (!strcmp(p, "422")) pix = VMAF_PIX_FMT_YUV422P;
      else if (!strcmp(p, "444")) pix = VMAF_PIX_FMT_YUV444P;
      else die("pixfmt");
    } else die("unknown arg");
  }
  if (!model || !ref_path || !dist_path || w <= 0 || h <= 0 || frames <= 0) die("missing args");
  if (bitdepth != 8 && bitdepth != 10) die("bitdepth");

  FILE* fr = fopen(ref_path, "rb");
  FILE* fd = fopen(dist_path, "rb");
  if (!fr || !fd) die("open yuv");

  VmafConfiguration cfg = {0};
  cfg.log_level = VMAF_LOG_LEVEL_ERROR;
  cfg.n_threads = (unsigned)(threads < 1 ? 1 : threads);
  VmafContext* vmaf = NULL;
  if (vmaf_init(&vmaf, cfg)) die("vmaf_init");
  VmafModelConfig mcfg = {0};
  mcfg.name = "vmaf";
  VmafModel* m = NULL;
  if (vmaf_model_load_from_path(&m, &mcfg, model) || !m) die("model load");
  if (vmaf_use_features_from_model(vmaf, m)) die("features");

  for (int i = 0; i < frames; ++i) {
    VmafPicture r = {0}, d = {0};
    fill_picture(&r, fr, w, h, bitdepth, pix);
    fill_picture(&d, fd, w, h, bitdepth, pix);
    if (vmaf_read_pictures(vmaf, &r, &d, (unsigned)i)) die("read_pictures");
  }
  if (vmaf_read_pictures(vmaf, NULL, NULL, 0)) die("flush");

  printf("{\n  \"libvmaf\": \"%s\",\n  \"frames\": [\n", vmaf_version());
  for (int i = 0; i < frames; ++i) {
    double score = 0, cambi = 0, speed = 0, adm3 = 0, motion3 = 0;
    if (vmaf_score_at_index(vmaf, m, &score, (unsigned)i)) die("score");
    if (feat_cambi && vmaf_feature_score_at_index(vmaf, feat_cambi, &cambi, (unsigned)i)) die("cambi");
    if (feat_speed && vmaf_feature_score_at_index(vmaf, feat_speed, &speed, (unsigned)i)) die("speed");
    if (feat_adm3 && vmaf_feature_score_at_index(vmaf, feat_adm3, &adm3, (unsigned)i)) die("adm3");
    if (feat_motion3 && vmaf_feature_score_at_index(vmaf, feat_motion3, &motion3, (unsigned)i))
      die("motion3");
    printf("    {\"i\":%d,\"vmaf\":%.17g,\"cambi\":%.17g,\"speed_chroma\":%.17g,"
           "\"adm3\":%.17g,\"motion3\":%.17g}%s\n",
           i, score, cambi, speed, adm3, motion3, (i + 1 < frames) ? "," : "");
  }
  const unsigned lo = 0;
  const unsigned hi = (unsigned)(frames - 1);
  double pooled_mean = 0, pooled_harmonic = 0, pooled_min = 0, pooled_max = 0;
  if (vmaf_score_pooled(vmaf, m, VMAF_POOL_METHOD_MEAN, &pooled_mean, lo, hi))
    die("pooled mean");
  if (vmaf_score_pooled(vmaf, m, VMAF_POOL_METHOD_HARMONIC_MEAN, &pooled_harmonic, lo, hi))
    die("pooled harmonic_mean");
  if (vmaf_score_pooled(vmaf, m, VMAF_POOL_METHOD_MIN, &pooled_min, lo, hi))
    die("pooled min");
  if (vmaf_score_pooled(vmaf, m, VMAF_POOL_METHOD_MAX, &pooled_max, lo, hi))
    die("pooled max");
  printf(
      "  ],\n"
      "  \"frames_count\": %d,\n"
      "  \"pooled\": {\n"
      "    \"mean\": %.17g,\n"
      "    \"harmonic_mean\": %.17g,\n"
      "    \"min\": %.17g,\n"
      "    \"max\": %.17g\n"
      "  },\n"
      "  \"pooled_mean\": %.17g\n"
      "}\n",
      frames, pooled_mean, pooled_harmonic, pooled_min, pooled_max, pooled_mean);

  vmaf_close(vmaf);
  vmaf_model_destroy(m);
  fclose(fr);
  fclose(fd);
  return 0;
}
