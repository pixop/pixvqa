#!/usr/bin/env bash
# Build pinned zimg release-3.0.6 (commit f819b14) into deps/zimg-3.0.6.
# HDR VMAF picture conversion links this library. CUDA is not involved.
# GNU make, gcc, g++, nasm, pkg-config, git, and perl must be on PATH.
# If autoconf/automake/libtool/m4 are absent, this script downloads the
# Ubuntu 24.04 packages named below into deps/src (gitignored) and uses them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_TAG="release-3.0.6"
PIN_COMMIT="f819b14e8f39d1282400b0d9543e8ef73c1b2bbd"
SRC="${ZIMG_SRC:-$ROOT/deps/src/zimg}"
PREFIX="${ZIMG_PREFIX:-$ROOT/deps/zimg-3.0.6}"
HOST="${ROOT}/deps/src/host-autotools"
DEBS="${ROOT}/deps/src/host-debs"

if [[ -n "${GCC_ENV:-}" ]]; then
  export PATH="${GCC_ENV}/bin:${PATH}"
elif [[ -d "${HOME}/gcc-env/bin" ]]; then
  export PATH="${HOME}/gcc-env/bin:${PATH}"
fi

for tool in git gcc g++ nasm pkg-config perl; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "build_zimg: missing required tool '$tool' on PATH" >&2
    exit 1
  fi
done
if ! command -v make >/dev/null 2>&1; then
  echo "build_zimg: GNU make is required on PATH" >&2
  exit 1
fi

ensure_autotools() {
  if command -v autoconf >/dev/null 2>&1 && command -v automake >/dev/null 2>&1 &&
     command -v libtoolize >/dev/null 2>&1 && command -v m4 >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p "$DEBS" "$HOST"
  local base="http://archive.ubuntu.com/ubuntu/pool/main"
  declare -A url=(
    [autoconf_2.71-3_all.deb]="$base/a/autoconf/autoconf_2.71-3_all.deb"
    [automake_1.16.5-1.3ubuntu1_all.deb]="$base/a/automake-1.16/automake_1.16.5-1.3ubuntu1_all.deb"
    [libtool_2.4.7-7build1_all.deb]="$base/libt/libtool/libtool_2.4.7-7build1_all.deb"
    [m4_1.4.19-4build1_amd64.deb]="$base/m/m4/m4_1.4.19-4build1_amd64.deb"
    [autotools-dev_20220109.1_all.deb]="$base/a/autotools-dev/autotools-dev_20220109.1_all.deb"
  )
  declare -A sha=(
    [autoconf_2.71-3_all.deb]=cc3f9f7a1e576173fb59c36652c0a67c6426feae752b352404ba92dfcb1b26c9
    [automake_1.16.5-1.3ubuntu1_all.deb]=5ae9a98e73545002cd891f028859941af2a3c760cb6190e635c7ef36953912de
    [libtool_2.4.7-7build1_all.deb]=9d1d707179675d38e024bb13613b1d99e0d33fa6c45e5f3bcba19340781781d3
    [m4_1.4.19-4build1_amd64.deb]=30ea715845a1863abd55d11e47ab91513130bf347852d4cdeae69e9d98fa795c
    [autotools-dev_20220109.1_all.deb]=d909f0327b09d9a9136239caca975df89782fa28efd721c4eb4caea422d3fc5a
  )
  rm -rf "$HOST"
  mkdir -p "$HOST"
  local name
  for name in "${!url[@]}"; do
    if [[ ! -f "$DEBS/$name" ]] || ! echo "${sha[$name]}  $DEBS/$name" | sha256sum -c - >/dev/null 2>&1; then
      echo "build_zimg: fetching $name"
      curl -fsSL -o "$DEBS/$name" "${url[$name]}"
      echo "${sha[$name]}  $DEBS/$name" | sha256sum -c -
    fi
    dpkg-deb -x "$DEBS/$name" "$HOST"
  done
  # The Ubuntu packages install automake/aclocal via postinst alternatives,
  # and their scripts embed /usr/share. Point those paths at the extract.
  ln -sfn automake-1.16 "$HOST/usr/bin/automake"
  ln -sfn aclocal-1.16 "$HOST/usr/bin/aclocal"
  # automake and libtool ship config.guess as a symlink into autotools-dev.
  local link base
  while IFS= read -r link; do
    base="$(basename "$link")"
    rm -f "$link"
    cp "$HOST/usr/share/misc/$base" "$link"
  done < <(find "$HOST/usr/share" -type l \( -name config.guess -o -name config.sub \))
  # Package scripts and autom4te.cfg embed /usr/share. Rewrite the extract once.
  local path
  while IFS= read -r -d '' path; do
    if grep -I -q '/usr/share' "$path"; then
      sed -i "s|/usr/share|${HOST}/usr/share|g" "$path"
    fi
  done < <(find "$HOST/usr" -type f -print0)
  local script
  for script in "$HOST"/usr/bin/*; do
    if [[ -L "$script" ]]; then
      continue
    fi
    if head -c 80 "$script" | grep -q -E 'perl|/bin/sh'; then
      sed -i \
        -e "s|/usr/bin/autom4te|${HOST}/usr/bin/autom4te|g" \
        -e "s|/usr/bin/autoconf|${HOST}/usr/bin/autoconf|g" \
        -e "s|/usr/bin/autoheader|${HOST}/usr/bin/autoheader|g" \
        -e "s|/usr/bin/automake|${HOST}/usr/bin/automake|g" \
        -e "s|/usr/bin/aclocal|${HOST}/usr/bin/aclocal|g" \
        -e "s|/usr/bin/m4|${HOST}/usr/bin/m4|g" \
        "$script"
    fi
  done
  export PATH="$HOST/usr/bin:${PATH}"
  export ACLOCAL_PATH="$HOST/usr/share/aclocal${ACLOCAL_PATH:+:$ACLOCAL_PATH}"
  export M4="$HOST/usr/bin/m4"
  export AUTOM4TE="$HOST/usr/bin/autom4te"
  export AUTOCONF="$HOST/usr/bin/autoconf"
  export AUTOHEADER="$HOST/usr/bin/autoheader"
  export AUTOMAKE="$HOST/usr/bin/automake"
  export ACLOCAL="$HOST/usr/bin/aclocal"
  export autom4te_perllibdir="$HOST/usr/share/autoconf"
  export AC_MACRODIR="$HOST/usr/share/autoconf"
  export trailer_m4="$HOST/usr/share/autoconf/autoconf/trailer.m4"
  hash -r
}

ensure_autotools
for tool in autoconf automake libtoolize m4; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "build_zimg: autotools tool '$tool' is still missing" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/deps/src"
if [[ ! -d "$SRC/.git" ]]; then
  git init -q -b main "$SRC"
  git -C "$SRC" remote add origin https://github.com/sekrit-twc/zimg.git
fi
git -C "$SRC" fetch --depth 1 origin "$PIN_COMMIT"
git -C "$SRC" checkout --detach --force FETCH_HEAD
ACTUAL="$(git -C "$SRC" rev-parse HEAD)"
echo "zimg source HEAD=$ACTUAL"
if [[ "$ACTUAL" != "$PIN_COMMIT" ]]; then
  echo "build_zimg: checkout $ACTUAL != $PIN_COMMIT" >&2
  exit 1
fi
echo "Verified zimg $PIN_TAG == $PIN_COMMIT"

if [[ ! -f "$SRC/configure" || ! -f "$SRC/config.guess" || ! -f "$SRC/config.sub" ]]; then
  (cd "$SRC" && ./autogen.sh)
fi

rm -rf "$PREFIX"
mkdir -p "$PREFIX"
(
  cd "$SRC"
  ./configure --prefix="$PREFIX" --enable-static --disable-shared
  make -j"$(nproc)"
  make install
)

PC_DIR=""
if [[ -f "$PREFIX/lib64/pkgconfig/zimg.pc" ]]; then
  PC_DIR="$PREFIX/lib64/pkgconfig"
elif [[ -f "$PREFIX/lib/pkgconfig/zimg.pc" ]]; then
  PC_DIR="$PREFIX/lib/pkgconfig"
else
  echo "build_zimg: zimg.pc was not installed under $PREFIX" >&2
  exit 1
fi
export PKG_CONFIG_PATH="${PC_DIR}${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
ZVER="$(pkg-config --modversion zimg)"
{
  echo "zimg_tag=$PIN_TAG"
  echo "zimg_commit=$PIN_COMMIT"
  echo "zimg_version=$ZVER"
  echo "zimg_prefix=$PREFIX"
  echo "zimg_pc=$PC_DIR/zimg.pc"
  echo "configure=--enable-static --disable-shared"
} > "$PREFIX/pixvqa-zimg-stamp.txt"
echo "zimg version: $ZVER"
echo "zimg libs: $(pkg-config --libs --static zimg)"
echo "zimg install OK: $PREFIX"
