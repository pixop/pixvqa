#!/usr/bin/env bash
# Two independent checkouts must not install into each other.
# A custom PREFIX that is not a PixVQA checkout is preserved.
# Git dependency clones retry and fall back to HTTP/1.1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bash -n scripts/install_prefix.sh
bash -n scripts/git_retry.sh
bash -n scripts/install_choices.sh
bash -n scripts/ci/verify_install.sh
bash -n scripts/ci/prepare_vvc_fixture.sh
bash -n install_pixvqa.sh
# shellcheck disable=SC1091
source "$ROOT/scripts/install_prefix.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/git_retry.sh"

grep -q 'pixvqa_select_prefix' install_pixvqa.sh || fail 'installer does not select a prefix'
grep -q 'pixvqa_run_installed_smoke' install_pixvqa.sh || fail 'installer smoke does not use the installed binary'
grep -q 'pixvqa_git_clone' install_pixvqa.sh || fail 'installer clones without the retry helper'
grep -q 'PIXVQA_QA_CPU' scripts/run_qa.py || fail 'QA runner ignores the installed binary'
grep -q 'pixvqa_choice' install_pixvqa.sh || fail 'installer has no noninteractive choice'
grep -q 'Install dav1d' install_pixvqa.sh || fail 'installer dropped the interactive dav1d prompt'
# shellcheck disable=SC1091
source "$ROOT/scripts/install_choices.sh"
(
    unset PIXVQA_ENABLE_DAV1D PIXVQA_ENABLE_NVIDIA
    [[ "$(pixvqa_choice PIXVQA_ENABLE_DAV1D)" == unset ]]
    [[ "$(PIXVQA_ENABLE_DAV1D=1 pixvqa_choice PIXVQA_ENABLE_DAV1D)" == yes ]]
    [[ "$(PIXVQA_ENABLE_DAV1D=yes pixvqa_choice PIXVQA_ENABLE_DAV1D)" == yes ]]
    [[ "$(PIXVQA_ENABLE_NVIDIA=0 pixvqa_choice PIXVQA_ENABLE_NVIDIA)" == no ]]
    [[ "$(PIXVQA_ENABLE_NVIDIA=no pixvqa_choice PIXVQA_ENABLE_NVIDIA)" == no ]]
    [[ "$(PIXVQA_ENABLE_DAV1D=maybe pixvqa_choice PIXVQA_ENABLE_DAV1D)" == invalid ]]
    [[ "$(PIXVQA_ENABLE_DAV1D= pixvqa_choice PIXVQA_ENABLE_DAV1D)" == invalid ]]
)
echo "OK   noninteractive installer choices"

make_checkout() {
  local dir="$1"
  mkdir -p "$dir/scripts" "$dir/bin"
  : > "$dir/install_pixvqa.sh"
  : > "$dir/build.sh"
  : > "$dir/scripts/install.sh"
  printf 'keep\n' > "$dir/KEEP"
  printf 'old\n' > "$dir/bin/pixvqa"
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
a="$work/checkout-a"
b="$work/checkout-b"
make_checkout "$a"
make_checkout "$b"
custom="$work/custom-prefix"
snapshot() { sha256sum "$1/KEEP" "$1/bin/pixvqa"; }
before_a="$(snapshot "$a")"
before_b="$(snapshot "$b")"

selected="$(PREFIX="$custom" pixvqa_select_prefix "$b")"
[[ "$selected" == "$(cd "$(dirname "$custom")" && pwd -P)/$(basename "$custom")" ]] || fail "custom prefix became $selected"
selected="$(PREFIX="$a" pixvqa_select_prefix "$b")"
[[ "$selected" == "$(cd "$b" && pwd -P)" ]] || fail "foreign checkout prefix became $selected"
selected="$(PREFIX="$a/nested" pixvqa_select_prefix "$b")"
[[ "$selected" == "$(cd "$b" && pwd -P)" ]] || fail "prefix inside another checkout became $selected"
selected="$(unset PREFIX; pixvqa_select_prefix "$b")"
[[ "$selected" == "$(cd "$b" && pwd -P)" ]] || fail "default prefix became $selected"
selected="$(PREFIX="$b" pixvqa_select_prefix "$b")"
[[ "$selected" == "$(cd "$b" && pwd -P)" ]] || fail "same-checkout prefix became $selected"
echo "OK   prefix selection ignores another checkout and keeps a custom prefix"

tiny="$work/tiny-pixvqa"
printf '#!/bin/sh\nexit 0\n' > "$tiny"
chmod +x "$tiny"
PREFIX="$custom" PIXVQA_BIN="$tiny" bash scripts/install.sh >/dev/null
[[ -x "$custom/bin/pixvqa" ]] || fail 'custom prefix has no bin/pixvqa'
[[ "$(snapshot "$a")" == "$before_a" ]] || fail 'custom install modified checkout A'
[[ "$(snapshot "$b")" == "$before_b" ]] || fail 'custom install modified checkout B'

resolved="$(PREFIX="$a" pixvqa_select_prefix "$b")"
PREFIX="$resolved" PIXVQA_BIN="$tiny" bash scripts/install.sh >/dev/null
[[ -x "$b/bin/pixvqa" ]] || fail 'current checkout was not installed'
[[ "$(snapshot "$a")" == "$before_a" ]] || fail 'installing checkout B modified checkout A'
[[ ! -x "$a/bin/pixvqa" || "$(cat "$a/bin/pixvqa")" == old ]] || fail 'checkout A executable was replaced'
echo "OK   installing one checkout does not modify the other"

stub="$work/bin"
mkdir -p "$stub"
cat > "$stub/python3" << 'EOF'
#!/bin/sh
printf '%s\n' "$PIXVQA_QA_CPU" > "$STUB_OUT"
if [ -n "${LD_LIBRARY_PATH+x}" ]; then
  printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH" >> "$STUB_OUT"
else
  printf 'LD_LIBRARY_PATH unset\n' >> "$STUB_OUT"
fi
printf '%s\n' "$*" >> "$STUB_OUT"
exit 0
EOF
chmod +x "$stub/python3"
smoke_out="$work/smoke-out"
: > "$smoke_out"
PATH="$stub:$PATH" STUB_OUT="$smoke_out" LD_LIBRARY_PATH="/tmp/should-not-leak" \
  pixvqa_run_installed_smoke "$b" "$b"
grep -qx "$(readlink -f "$b/bin/pixvqa")" "$smoke_out" || fail "smoke did not target the installed executable"
grep -qx 'LD_LIBRARY_PATH unset' "$smoke_out" || fail 'smoke kept LD_LIBRARY_PATH'
grep -q 'scripts/run_qa.py --profile smoke' "$smoke_out" || fail 'smoke command was not run_qa.py'
echo "OK   smoke QA targets the installed executable with LD_LIBRARY_PATH unset"

git_log="$work/git-log"
mkdir -p "$stub"
cat > "$stub/git" << 'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$GIT_LOG"
case " $* " in
  *" https://git.ffmpeg.org/ffmpeg.git "*) exit 1 ;;
esac
case " $* " in
  *" http.version=HTTP/1.1 "*) exit 0 ;;
esac
exit 1
EOF
chmod +x "$stub/git"
: > "$git_log"
PATH="$stub:/usr/bin:/bin" GIT_LOG="$git_log" PIXVQA_GIT_RETRY_DELAY=0 \
  pixvqa_git_clone n8.1.3 https://git.ffmpeg.org/ffmpeg.git "$work/ffmpeg-src"
grep -q 'http.version=HTTP/1.1' "$git_log" || fail 'FFmpeg clone did not fall back to HTTP/1.1'
grep -q 'https://github.com/FFmpeg/FFmpeg.git' "$git_log" || fail 'FFmpeg clone did not try the GitHub mirror'
echo "OK   FFmpeg clone retries over HTTP/1.1 and uses the GitHub mirror"

: > "$git_log"
fetch_dir="$work/fetch-src"
mkdir -p "$fetch_dir"
PATH="$stub:/usr/bin:/bin" GIT_LOG="$git_log" PIXVQA_GIT_RETRY_DELAY=0 \
  pixvqa_git_fetch "$fetch_dir" fetch --depth 1 origin abcdef
grep -q 'http.version=HTTP/1.1' "$git_log" || fail 'fetch did not fall back to HTTP/1.1'
head -n 1 "$git_log" | grep -q 'http.version=HTTP/1.1' && fail 'fetch skipped the first plain attempt'
echo "OK   git fetch retries with an HTTP/1.1 fallback"
echo "All install-prefix checks passed"
