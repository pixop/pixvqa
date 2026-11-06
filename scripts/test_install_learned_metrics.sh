#!/usr/bin/env bash
# Fast offline checks for scripts/install_learned_metrics.sh defaults and config
# generation. Uses a throwaway tree and mocked venvs; does not run pip or download.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bash -n scripts/install_learned_metrics.sh
bash -n scripts/setup_learned_env.sh

# Defaults appear in the installer (repo-local .venv and deps/ColorVideoVDP).
grep -Eq 'PIXVQA_LEARNED_ENV_DIR.*\$\{[^}]*root[^}]*\}/\.venv|env_dir=.*\.venv' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh missing default repo-local .venv'
grep -q 'deps/ColorVideoVDP' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh missing default deps/ColorVideoVDP path'
grep -q 'learned_venv_helpers.sh' scripts/setup_learned_env.sh ||
  fail 'setup_learned_env.sh must source learned_venv_helpers.sh'
grep -q 'learned_venv_create_flags' scripts/setup_learned_env.sh ||
  fail 'setup_learned_env.sh must call learned_venv_create_flags'
grep -q 'pytorchvideo==' python/requirements/learned-env.txt ||
  fail 'learned-env.txt must pin pytorchvideo for SimpleVQA'
grep -q 'SimpleVQA' docs/LEARNED_METRICS.md || fail 'LEARNED_METRICS.md must mention SimpleVQA shared env'
grep -q 'refusing to create a virtualenv under NTFS' scripts/install_learned_metrics.sh &&
  fail 'install_learned_metrics.sh must not blanket-refuse NTFS; use venv --copies instead'

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fake_root="$work/pixvqa"
mkdir -p "$fake_root/scripts" "$fake_root/config"
cp -- "$ROOT/scripts/install_learned_metrics.sh" "$fake_root/scripts/install_learned_metrics.sh"
cp -- "$ROOT/config/learned_metrics.example.toml" "$fake_root/config/learned_metrics.example.toml"

commit=166f5ae95ecbd841ae727984cd237de213875399

make_pinned_venv() {
  local dir="$1"
  mkdir -p "$dir/bin"
  printf '#!/usr/bin/env sh\nprintf "stub\\n"\n' > "$dir/bin/python"
  chmod +x "$dir/bin/python"
  printf '%s\n' "$commit" > "$dir/COLORVIDEOVDP_COMMIT"
}

# 1) Fresh example: default env is fake_root/.venv (pinned), config is created with absolute python.
make_pinned_venv "$fake_root/.venv"
(
  cd "$fake_root"
  unset PIXVQA_LEARNED_ENV_DIR PIXVQA_CVVDP_SOURCE || true
  bash scripts/install_learned_metrics.sh
) >/dev/null
[[ -f "$fake_root/config/learned_metrics.local.toml" ]] || fail 'local config not created'
grep -Fq "python = \"$fake_root/.venv/bin/python\"" "$fake_root/config/learned_metrics.local.toml" ||
  fail 'created config does not pin absolute interpreter path'
before="$(cat "$fake_root/config/learned_metrics.local.toml")"

# 2) Re-run preserves config; still succeeds.
(
  cd "$fake_root"
  unset PIXVQA_LEARNED_ENV_DIR PIXVQA_CVVDP_SOURCE || true
  bash scripts/install_learned_metrics.sh
) >/dev/null
[[ "$(cat "$fake_root/config/learned_metrics.local.toml")" == "$before" ]] || fail 're-run modified existing local config'

# 3) Override env dir: second tree, custom pinned path, config matches override.
fake2="$work/pixvqa2"
mkdir -p "$fake2/scripts" "$fake2/config"
cp -- "$ROOT/scripts/install_learned_metrics.sh" "$fake2/scripts/install_learned_metrics.sh"
cp -- "$ROOT/config/learned_metrics.example.toml" "$fake2/config/learned_metrics.example.toml"
custom_venv="$work/custom-learned"
make_pinned_venv "$custom_venv"
(
  cd "$fake2"
  export PIXVQA_LEARNED_ENV_DIR="$custom_venv"
  export PIXVQA_CVVDP_SOURCE="$work/unused-cvvdp"
  bash scripts/install_learned_metrics.sh
) >/dev/null
grep -Fq "python = \"$custom_venv/bin/python\"" "$fake2/config/learned_metrics.local.toml" ||
  fail 'override PIXVQA_LEARNED_ENV_DIR not reflected in generated config'

# 4) Mismatch warning: existing config points at a different python than selected env.
fake3="$work/pixvqa3"
mkdir -p "$fake3/scripts" "$fake3/config"
cp -- "$ROOT/scripts/install_learned_metrics.sh" "$fake3/scripts/install_learned_metrics.sh"
cp -- "$ROOT/config/learned_metrics.example.toml" "$fake3/config/learned_metrics.example.toml"
make_pinned_venv "$fake3/.venv"
other_py="$work/other-env/bin/python"
mkdir -p "$(dirname -- "$other_py")"
printf '#!/bin/sh\nexit 0\n' > "$other_py"
chmod +x "$other_py"
cat > "$fake3/config/learned_metrics.local.toml" <<EOF
[learned]
python = "$other_py"
adapter_path = "../python"
path = "~/bin"
EOF
out="$work/mismatch.out"
(
  cd "$fake3"
  unset PIXVQA_LEARNED_ENV_DIR PIXVQA_CVVDP_SOURCE || true
  bash scripts/install_learned_metrics.sh
) >"$out" 2>&1 || fail 'mismatch scenario should not fail the installer'
grep -qi 'lists python=' "$out" || fail 'expected python mismatch warning when config differs from env'
grep -q 'preserving existing' "$out" || fail 'expected preserve message for existing config'

# 5) WSL-style mount: fake findmnt reports 9p → learned_venv_create_flags returns --copies
stub_bin="$work/stub-bin"
mkdir -p "$stub_bin"
cat > "$stub_bin/findmnt" <<'EOF'
#!/bin/sh
# Minimal findmnt for learned_venv_on_wsl_network_fs (see `man findmnt`).
if [ "$1" = -n ] && [ "$2" = -o ] && [ "$3" = FSTYPE ] && [ "$4" = -T ]; then
  printf '%s\n' "9p"
elif [ "$1" = -n ] && [ "$2" = -o ] && [ "$3" = SOURCE ] && [ "$4" = -T ]; then
  printf '%s\n' '\\?\ADVHS{...}'
fi
EOF
chmod +x "$stub_bin/findmnt"
# shellcheck source=learned_venv_helpers.sh
# shellcheck disable=SC1091
source "$ROOT/scripts/learned_venv_helpers.sh"
mkdir -p "$work/wsl-layout"
wsl_venv="$work/wsl-layout/.venv"
flags="$(PATH="$stub_bin:$PATH" learned_venv_create_flags "$wsl_venv")"
[[ "$flags" == "--copies" ]] || fail "expected --copies on simulated 9p mount, got ${flags:-empty}"
flags_symlink="$(PATH="$stub_bin:$PATH" PIXVQA_LEARNED_VENV_USE_SYMLINKS=1 learned_venv_create_flags "$wsl_venv")"
[[ -z "$flags_symlink" ]] || fail "expected empty venv flags when PIXVQA_LEARNED_VENV_USE_SYMLINKS=1"

echo "OK   install_learned_metrics offline tests passed"
