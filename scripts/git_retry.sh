#!/usr/bin/env bash
# Retry PixVQA dependency Git operations. The second and later attempts force
# HTTP/1.1. git.ffmpeg.org also falls back to the official GitHub mirror.
# Set PIXVQA_GIT_RETRY_DELAY=0 to skip the pause between attempts.

pixvqa_git_pause() {
  local seconds="$1"
  if [[ "${PIXVQA_GIT_RETRY_DELAY:-1}" == 0 ]]; then
    return 0
  fi
  sleep "$seconds"
}

pixvqa_git_clone() {
  local ref="$1" url="$2" dest="$3"
  local -a urls=("$url")
  local mirror attempt
  [[ -n "$dest" && "$dest" != "/" && "$dest" != "." ]] || return 1
  if [[ "$url" == "https://git.ffmpeg.org/ffmpeg.git" ]]; then
    urls+=("https://github.com/FFmpeg/FFmpeg.git")
  fi
  for mirror in "${urls[@]}"; do
    for attempt in 1 2 3; do
      rm -rf -- "$dest"
      if git clone --depth 1 --branch "$ref" "$mirror" "$dest"; then
        return 0
      fi
      rm -rf -- "$dest"
      printf 'git_retry: clone of %s failed (attempt %s); trying HTTP/1.1\n' "$mirror" "$attempt" >&2
      if git -c http.version=HTTP/1.1 clone --depth 1 --branch "$ref" "$mirror" "$dest"; then
        return 0
      fi
      rm -rf -- "$dest"
      pixvqa_git_pause "$attempt"
    done
  done
  printf 'git_retry: giving up on clone of %s\n' "$url" >&2
  return 1
}

pixvqa_git_fetch() {
  local dest="$1"
  local attempt
  shift
  for attempt in 1 2 3; do
    if git -C "$dest" "$@"; then
      return 0
    fi
    printf 'git_retry: fetch in %s failed (attempt %s); trying HTTP/1.1\n' "$dest" "$attempt" >&2
    if git -c http.version=HTTP/1.1 -C "$dest" "$@"; then
      return 0
    fi
    pixvqa_git_pause "$attempt"
  done
  printf 'git_retry: giving up on fetch in %s\n' "$dest" >&2
  return 1
}
