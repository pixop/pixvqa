#!/usr/bin/env bash
# Build pinned VVenC and VVdeC, then write a small synthetic VVC stream.
# This is not the Big Buck Bunny fixture and is safe to generate in CI.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source "$root/scripts/git_retry.sh"

prefix="$root/deps/ci-vvc"
src="$prefix/src"
out="$root/reports/qa/vvc-fixture"
mkdir -p "$src" "$out"
export PIXVQA_GIT_RETRY_DELAY="${PIXVQA_GIT_RETRY_DELAY:-2}"
log="$out/prepare.log"
exec > >(tee -a "$log") 2>&1

build_tool() {
    local name="$1" tag="$2" url="$3"
    if [[ -x "$prefix/bin/$name" ]]; then
        printf '%s already installed at %s\n' "$name" "$prefix/bin/$name"
        return 0
    fi
    pixvqa_git_clone "$tag" "$url" "$src/$name"
    cmake -S "$src/$name" -B "$src/$name/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix"
    cmake --build "$src/$name/build" -j"$(nproc)"
    cmake --install "$src/$name/build"
    if [[ ! -x "$prefix/bin/$name" ]]; then
        found="$(find "$src/$name/build" -type f -name "$name" -perm -111 | head -n 1)"
        [[ -n "$found" ]] || return 1
        mkdir -p "$prefix/bin"
        cp "$found" "$prefix/bin/$name"
        chmod 755 "$prefix/bin/$name"
    fi
}

build_tool vvencapp v1.14.0 https://github.com/fraunhoferhhi/vvenc.git
build_tool vvdecapp v3.2.1 https://github.com/fraunhoferhhi/vvdec.git
[[ -x "$prefix/bin/vvencapp" && -x "$prefix/bin/vvdecapp" ]]

width=176
height=144
frames=4
yuv="$out/source.yuv"
bit="$out/synthetic.vvc"
ref="$out/reference.yuv"
python3 - "$yuv" "$width" "$height" "$frames" <<'PY'
import sys
from pathlib import Path
dest, width, height, frames = sys.argv[1:]
width, height, frames = int(width), int(height), int(frames)
mask = (1 << 10) - 1
with Path(dest).open("wb") as out:
    for index in range(frames):
        def plane(w, h, base):
            raw = bytearray()
            for row in range(h):
                for col in range(w):
                    value = (base + index * 17 + row * 3 + col * 5) & mask
                    raw += int(value).to_bytes(2, "little")
            return bytes(raw)
        out.write(plane(width, height, 40))
        out.write(plane(width // 2, height // 2, 80))
        out.write(plane(width // 2, height // 2, 140))
PY

"$prefix/bin/vvencapp" --preset faster \
    --input "$yuv" --size "${width}x${height}" --format yuv420_10 \
    --framerate 24 --frames "$frames" --qp 32 --threads 1 \
    --intraperiod "$frames" --output "$bit"
"$prefix/bin/vvdecapp" --bitstream "$bit" --output "$ref" --frames "$frames"
[[ -s "$bit" && -s "$ref" ]]
python3 - "$ref" "$width" "$height" <<'PY'
import sys
from pathlib import Path
path, width, height = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
frame = width * height * 3
size = path.stat().st_size
if size == 0 or size % frame:
    raise SystemExit(f"reference is not yuv420p10le {width}x{height}: {size} bytes")
print(f"synthetic VVC reference frames: {size // frame}")
PY

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'fixture=%s\nreference=%s\nsize=%sx%s\n' "$bit" "$ref" "$width" "$height" >> "$GITHUB_OUTPUT"
fi
printf 'PIXVQA_VVC_FIXTURE=%s\nPIXVQA_VVC_REFERENCE=%s\nPIXVQA_VVC_SIZE=%sx%s\n' \
    "$bit" "$ref" "$width" "$height" > "$out/fixture.env"
printf 'synthetic VVC fixture ready: %s\n' "$bit"
