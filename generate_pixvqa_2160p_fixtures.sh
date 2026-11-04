#!/usr/bin/env bash
# PixVQA real-source fixture generator: 3 x 60-frame UHD 4:2:0 10-bit YUV,
# HEVC (NVENC if available), AV1 (SVT-AV1 preferred), VVC (VVenC).
# Nothing is tone-mapped, resized, or intentionally color-converted.
# All outputs are written flat into D:\tmp\2160p_420 (WSL: /mnt/d/tmp/2160p_420).
set -Eeuo pipefail

DIR="${PIXVQA_FIXTURE_DIR:-/mnt/d/tmp/2160p_420}"
DEPS="${PIXVQA_DEPS:-$HOME/codeccomp/metrics/pixvqa/deps}"
FRAMES="${FRAMES:-60}"
WIDTH=3840
HEIGHT=2160
QP="${QP:-32}"
PQ_MATRIX="${PQ_MATRIX:-bt2020nc}"  # BT.2100 PQ, BT.2020 non-constant luminance.
HLG_MATRIX="${HLG_MATRIX:-bt2020nc}"
FFMPEG="${FFMPEG:-$(command -v ffmpeg || true)}"
FFPROBE="${FFPROBE:-$(command -v ffprobe || true)}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
info() { printf '\n==> %s\n' "$*"; }
[[ -d "$DIR" ]] || fail "Missing directory: $DIR (is D: mounted in WSL?)"
[[ -x "$FFMPEG" && -x "$FFPROBE" ]] || fail 'ffmpeg and ffprobe are required.'
[[ "$FRAMES" =~ ^[0-9]+$ ]] && (( FRAMES >= 1 )) || fail 'FRAMES must be a positive integer.'
[[ "$QP" =~ ^[0-9]+$ ]] && (( QP >= 0 && QP <= 51 )) || fail 'QP must be between 0 and 51 (HEVC-compatible range).'
case "$PQ_MATRIX" in bt2020nc|bt2020c) ;; *) fail 'PQ_MATRIX must be bt2020nc or bt2020c';; esac
case "$HLG_MATRIX" in bt2020nc|bt2020c) ;; *) fail 'HLG_MATRIX must be bt2020nc or bt2020c';; esac

# Files have to be truly 4:2:0 planar little-endian 10-bit in 16-bit containers.
BYTES_PER_FRAME=$(( WIDTH * HEIGHT * 3 ))
EXPECTED_BYTES=$(( FRAMES * BYTES_PER_FRAME ))

SDR_SOURCE="${SDR_SOURCE:-$DIR/NeptuneFountain2_3840x2160_60fps_10bit_420.yuv}"
# PQ source from user's filename (possibly without .yuv extension).
if [[ -n "${PQ_SOURCE:-}" ]]; then
  pq_source="$PQ_SOURCE"
else
  shopt -s nullglob
  pq_matches=( "$DIR"/NeptuneFountain2_3840x2160_pq_bt2020l_420p_10b_60* )
  shopt -u nullglob
  (( ${#pq_matches[@]} == 1 )) || fail "Expected one PQ source matching NeptuneFountain2_3840x2160_pq_bt2020l_420p_10b_60*, found ${#pq_matches[@]}. Set PQ_SOURCE explicitly."
  pq_source="${pq_matches[0]}"
fi
PQ_SOURCE="$pq_source"
HLG_SOURCE="${HLG_SOURCE:-$DIR/LG_4K_OLED_Wonders-184055-tmu_hlg10_hevc.mkv}"
for src in "$SDR_SOURCE" "$PQ_SOURCE" "$HLG_SOURCE"; do
  [[ -f "$src" ]] || fail "Missing source: $src"
done

# Prevent accidentally reading prior outputs as source material.
for src in "$SDR_SOURCE" "$PQ_SOURCE"; do
  bytes=$(stat -c '%s' "$src")
  (( bytes >= EXPECTED_BYTES )) || fail "Raw source too short ($bytes bytes; need $EXPECTED_BYTES): $src"
  (( bytes % BYTES_PER_FRAME == 0 )) || fail "Raw source is not an integer number of 3840x2160 yuv420p10le frames: $src"
done

# Select encoding backends before writing outputs.
if "$FFMPEG" -hide_banner -encoders 2>/dev/null | grep -E '^[[:space:]]*V[^[:space:]]*[[:space:]]+hevc_nvenc([[:space:]]|$)'; then
  HEVC_ENCODER=hevc_nvenc
elif "$FFMPEG" -hide_banner -encoders 2>/dev/null | grep -E '^[[:space:]]*V[^[:space:]]*[[:space:]]+libx265([[:space:]]|$)'; then
  HEVC_ENCODER=libx265
else
  fail 'No HEVC encoder (hevc_nvenc or libx265) in selected FFmpeg.'
fi
if "$FFMPEG" -hide_banner -encoders 2>/dev/null | grep -E '^[[:space:]]*V[^[:space:]]*[[:space:]]+libsvtav1([[:space:]]|$)'; then
  AV1_ENCODER=libsvtav1
elif "$FFMPEG" -hide_banner -encoders 2>/dev/null | grep -E '^[[:space:]]*V[^[:space:]]*[[:space:]]+av1_nvenc([[:space:]]|$)'; then
  AV1_ENCODER=av1_nvenc
else
  fail 'No AV1 encoder (libsvtav1 or av1_nvenc) in selected FFmpeg.'
fi

VVENC="${VVENC:-}"
if [[ -z "$VVENC" ]]; then
  VVENC=$(command -v vvencapp || true)
fi
if [[ -z "$VVENC" ]]; then
  # Search in the known dependencies directory (not assumed at its root).
  VVENC=$(find "$DEPS" -type f -name vvencapp -perm /111 -print -quit 2>/dev/null || true)
fi
if [[ -z "$VVENC" || ! -x "$VVENC" ]]; then
  vvsrc="$DEPS/vvenc-1.13.1"
  command -v cmake >/dev/null || fail "vvencapp not found and cmake is unavailable. Build VVenC from $vvsrc or set VVENC."
  [[ -f "$vvsrc/CMakeLists.txt" ]] || fail "vvencapp not found and VVenC source missing at $vvsrc. Set VVENC explicitly."
  info 'VVenC executable missing; compiling vvencapp once from the existing source tree'
  vvbuild="$DEPS/vvenc-local-build"
  cmake -S "$vvsrc" -B "$vvbuild" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$vvbuild" --target vvencapp -j 8
  VVENC=$(find "$vvbuild" -type f -name vvencapp -perm /111 -print -quit 2>/dev/null || true)
fi
[[ -n "$VVENC" && -x "$VVENC" ]] || fail "Cannot find compiled vvencapp. Set VVENC=/absolute/path/to/vvencapp."

info "Source directory: $DIR"
printf 'FFmpeg: %s\nVVenC: %s\nHEVC: %s\nAV1: %s\nFrames: %s, QP: %s\n' \
  "$FFMPEG" "$VVENC" "$HEVC_ENCODER" "$AV1_ENCODER" "$FRAMES" "$QP"
printf 'PQ matrix: %s (source name bt2020l may indicate constant-luminance)\n' "$PQ_MATRIX"
printf 'HLG matrix: %s\n' "$HLG_MATRIX"
printf 'Each 60-frame UHD raw YUV occupies %.2f GiB.\n' "$(awk "BEGIN {print $EXPECTED_BYTES/1073741824}")"
printf 'Outputs are skipped if they already exist with nonzero size.\n'

# Flat output list: sdr/pq/hlg_ref_2160p_420p10.yuv, plus each codec.
# Never truncate a pre-existing output. Remove a selected output manually to redo it.
write_raw() {
  local mode=$1 src=$2 dst="$DIR/${1}_ref_2160p_420p10.yuv"
  if [[ -e "$dst" ]]; then
    [[ $(stat -c '%s' "$dst") -eq "$EXPECTED_BYTES" ]] || fail "Existing raw output has wrong size, please inspect/remove: $dst"
    info "SKIP existing YUV: $dst"
  elif [[ "$mode" == hlg ]]; then
    info 'Decoding exactly 60 HLG source frames to 10-bit YUV (no scaling/tone mapping)'
    local w h fmt trc
    w=$($FFPROBE -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$src" | head -n1)
    h=$($FFPROBE -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$src" | head -n1)
    [[ "$w" == "$WIDTH" && "$h" == "$HEIGHT" ]] || fail "HLG source is ${w}x${h}, expected ${WIDTH}x${HEIGHT}; refusing to rescale."
    fmt=$($FFPROBE -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$src" | head -n1)
    [[ "$fmt" == yuv420p10le || "$fmt" == p010le ]] || fail "HLG source pixel format is $fmt, not known 10-bit 4:2:0. Refusing implicit bit-depth/chroma conversion."
    trc=$($FFPROBE -v error -select_streams v:0 -show_entries stream=color_transfer -of csv=p=0 "$src" | head -n1)
    [[ "$trc" == arib-std-b67 || "$trc" == unknown || -z "$trc" ]] || fail "HLG source is tagged $trc, not HLG."
    if [[ "$trc" != arib-std-b67 ]]; then printf 'WARNING: HLG source transfer not tagged; trusting user-provided HLG classification.\n' >&2; fi
    local tmp="$dst.partial"
    [[ ! -e "$tmp" ]] || fail "Previous partial output exists: $tmp"
    "$FFMPEG" -hide_banner -loglevel warning -nostdin -i "$src" -map 0:v:0 -an -sn \
      -frames:v "$FRAMES" -pix_fmt yuv420p10le -f rawvideo "$tmp"
    [[ $(stat -c '%s' "$tmp") -eq "$EXPECTED_BYTES" ]] || fail 'HLG decoded output size incorrect (source may have too few frames).'
    mv "$tmp" "$dst"
  else
    info "Trimming first $FRAMES native raw frames: $src"
    local tmp="$dst.partial"
    [[ ! -e "$tmp" ]] || fail "Previous partial output exists: $tmp"
    head -c "$EXPECTED_BYTES" "$src" > "$tmp"
    [[ $(stat -c '%s' "$tmp") -eq "$EXPECTED_BYTES" ]] || fail 'Incomplete raw copy'
    mv "$tmp" "$dst"
  fi
}

# source frame rate of HLG compressed source is preserved; SDR and PQ are 60fps.
HLG_FPS=$($FFPROBE -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$HLG_SOURCE" | head -n1)
[[ "$HLG_FPS" =~ ^[0-9]+(/[0-9]+)?$ && "$HLG_FPS" != 0/0 ]] || fail "Invalid HLG frame rate: $HLG_FPS"

encode_ffmpeg() {
  local mode=$1 codec=$2 fps=$3 prim=$4 trc=$5 matrix=$6
  local src="$DIR/${mode}_ref_2160p_420p10.yuv"
  local dst="$DIR/${mode}_${codec}_qp${QP}.mp4"
  if [[ -e "$dst" ]]; then
    [[ -s "$dst" ]] || fail "Empty previous output exists: $dst"
    info "SKIP existing encoded file: $dst"
    return
  fi
  local -a enc=()
  if [[ "$codec" == hevc ]]; then
    if [[ "$HEVC_ENCODER" == hevc_nvenc ]]; then
      enc=(-c:v hevc_nvenc -profile:v main10 -preset p4 -rc constqp -qp "$QP" -pix_fmt p010le)
    else
      enc=(-c:v libx265 -preset medium -qp "$QP" -pix_fmt yuv420p10le -x265-params "aq-mode=0:repeat-headers=1")
    fi
  else
    if [[ "$AV1_ENCODER" == libsvtav1 ]]; then
      enc=(-c:v libsvtav1 -preset 8 -qp "$QP" -b:v 0 -pix_fmt yuv420p10le)
    else
      enc=(-c:v av1_nvenc -preset p4 -rc constqp -qp "$QP" -pix_fmt p010le)
    fi
  fi
  info "Encoding $mode -> $codec (QP $QP; FPS $fps)"
  local tmp="$DIR/.${mode}_${codec}_qp${QP}.partial.mp4"
  [[ ! -e "$tmp" ]] || fail "Previous partial output exists: $tmp"
  "$FFMPEG" -hide_banner -loglevel warning -nostdin \
    -f rawvideo -pixel_format yuv420p10le -video_size "${WIDTH}x${HEIGHT}" -framerate "$fps" -i "$src" \
    -vf "format=p010le,setparams=range=limited:color_primaries=${prim}:color_trc=${trc}:colorspace=${matrix}" \
    -frames:v "$FRAMES" -an -sn "${enc[@]}" \
    -color_range tv -color_primaries "$prim" -color_trc "$trc" -colorspace "$matrix" \
    -movflags +faststart "$tmp"
  [[ -s "$tmp" ]] || fail "Encoder output empty: $tmp"
  mv "$tmp" "$dst"
}

encode_vvc() {
  local mode=$1 fps=$2 prim=$3 trc=$4 matrix=$5 signal=$6
  local src="$DIR/${mode}_ref_2160p_420p10.yuv"
  local dst="$DIR/${mode}_vvc_qp${QP}.266"
  if [[ -e "$dst" ]]; then
    [[ -s "$dst" ]] || fail "Empty previous output exists: $dst"
    info "SKIP existing VVC: $dst"
    return
  fi
  info "Encoding $mode -> VVC via VVenC (QP $QP; FPS $fps)"
  local tmp="$DIR/.${mode}_vvc_qp${QP}.partial.266"
  [[ ! -e "$tmp" ]] || fail "Previous partial output exists: $tmp"
  local -a hdr_sdr=()
  case "$signal" in
    sdr) hdr_sdr=(--sdr sdr_709) ;;
    pq) hdr_sdr=(--hdr pq_2020) ;;
    hlg) hdr_sdr=(--hdr hlg_2020) ;;
  esac
  "$VVENC" -i "$src" -s "${WIDTH}x${HEIGHT}" --fps "$fps" --format yuv420_10 \
    --frames "$FRAMES" --preset fast --qp "$QP" --qpa 0 \
    "${hdr_sdr[@]}" \
    --additional "ColourDescriptionPresent=1:ColourPrimaries=${prim}:TransferCharacteristics=${trc}:MatrixCoefficients=${matrix}:VideoFullRange=0" \
    -o "$tmp"
  [[ -s "$tmp" ]] || fail "VVenC output empty: $tmp"
  mv "$tmp" "$dst"
}

write_raw sdr "$SDR_SOURCE"
write_raw pq "$PQ_SOURCE"
write_raw hlg "$HLG_SOURCE"

for mode in sdr pq hlg; do
  case "$mode" in
    sdr) fps=60; prim=bt709; trc=bt709; matrix=bt709 ;;
    pq)  fps=60; prim=bt2020; trc=smpte2084; matrix="$PQ_MATRIX" ;;
    hlg) fps="$HLG_FPS"; prim=bt2020; trc=arib-std-b67; matrix="$HLG_MATRIX" ;;
  esac
  encode_ffmpeg "$mode" hevc "$fps" "$prim" "$trc" "$matrix"
  encode_ffmpeg "$mode" av1 "$fps" "$prim" "$trc" "$matrix"
  encode_vvc "$mode" "$fps" "$prim" "$trc" "$matrix" "$mode"
done

# Raw VVC is the canonical output. Attempt MP4 remux where system FFmpeg supports
# VVC demuxing/muxing; failure is nonfatal, and never corrupts the .266 bitstream.
for mode in sdr pq hlg; do
  case "$mode" in sdr|pq) fps=60 ;; hlg) fps="$HLG_FPS" ;; esac
  elementary="$DIR/${mode}_vvc_qp${QP}.266"
  mp4="$DIR/${mode}_vvc_qp${QP}.mp4"
  tmp="$DIR/.${mode}_vvc_qp${QP}.partial.mp4"
  if [[ -e "$mp4" ]]; then
    info "SKIP existing VVC MP4: $mp4"
  elif [[ -e "$tmp" ]]; then
    fail "Previous partial output exists: $tmp"
  elif "$FFMPEG" -hide_banner -loglevel error -nostdin -f vvc -r "$fps" -i "$elementary" \
      -map 0:v:0 -c:v copy -movflags +faststart "$tmp"; then
    mv "$tmp" "$mp4"
    info "VVC MP4 created: $mp4"
  else
    rm -f "$tmp"
    printf 'WARNING: Selected FFmpeg cannot remux VVC .266 to MP4. Kept %s.\n' "$elementary" >&2
  fi
done

info 'Probe metadata of MP4 outputs (10-bit + tags should match the source class)'
for mode in sdr pq hlg; do
  for codec in hevc av1; do
    file="$DIR/${mode}_${codec}_qp${QP}.mp4"
    echo "--- $(basename "$file") ---"
    "$FFPROBE" -v error -select_streams v:0 \
      -show_entries stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,color_range,color_space,color_transfer,color_primaries \
      -of default=noprint_wrappers=1 "$file"
  done
done
info 'VVC .266 files are raw VVC elementary bitstreams, not MP4 containers.'
printf 'Done. Files in: %s\n' "$DIR"
printf 'Reminder: QP=32 is codec-specific, not equal visual quality across codecs.\n'
