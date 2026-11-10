#!/usr/bin/env bash
# Fast offline checks for scripts/install_learned_metrics.sh defaults, config merge,
# and pixvqa_learned_install. Uses throwaway trees; no pip or network when offline=1.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bash -n scripts/install_learned_metrics.sh
bash -n scripts/setup_learned_env.sh
bash -n scripts/setup_hdr_chipqa_envs.sh
bash -n scripts/ensure_chipqa_inference_python.sh

grep -Eq 'PIXVQA_LEARNED_ENV_DIR.*\$\{[^}]*root[^}]*\}/\.venv|env_dir=.*\.venv' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh missing default repo-local .venv'
grep -q 'deps/ColorVideoVDP' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh missing default deps/ColorVideoVDP path'
grep -q 'pixvqa_learned_install' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh must invoke pixvqa_learned_install'
grep -Fq 'PIXVQA_LEARNED_CONFIG_PYTHON:-$selected_python' scripts/install_learned_metrics.sh ||
  fail 'config installer must use managed learned Python, not Ubuntu system Python 3.10'
grep -q 'ensure_cvvdp_dependencies' scripts/install_learned_metrics.sh ||
  fail 'installer must repair CVVDP metadata and dependency closure'
grep -q 'ensure_simplevqa_dependencies' scripts/install_learned_metrics.sh ||
  fail 'installer must verify and repair SimpleVQA modules in reused venv'
grep -q 'setup_hdr_chipqa_envs' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh must provision HDR-ChipQA envs'
grep -q 'ensure_learned_ffmpeg_cli' scripts/install_learned_metrics.sh ||
  fail 'install_learned_metrics.sh must provision pinned ffmpeg/ffprobe'
bash -n scripts/ensure_learned_ffmpeg_cli.sh
bash -n scripts/build_learned_ffmpeg_cli.sh
grep -q 'learned_venv_helpers.sh' scripts/setup_learned_env.sh ||
  fail 'setup_learned_env.sh must source learned_venv_helpers.sh'
grep -q 'pytorchvideo==' python/requirements/learned-env.txt ||
  fail 'learned-env.txt must pin pytorchvideo for SimpleVQA'
grep -q 'SimpleVQA' docs/LEARNED_METRICS.md || fail 'LEARNED_METRICS.md must mention SimpleVQA'
grep -q 'refusing to create a virtualenv under NTFS' scripts/install_learned_metrics.sh &&
  fail 'install_learned_metrics.sh must not blanket-refuse NTFS; use venv --copies instead'

python3 scripts/test_install_learned_assets.py

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fake_root="$work/pixvqa"
mkdir -p "$fake_root/scripts" "$fake_root/config" "$fake_root/python"
cp -r -- "$ROOT/scripts/." "$fake_root/scripts/"
cp -r -- "$ROOT/python/." "$fake_root/python/"
cp -- "$ROOT/config/learned_metrics.example.toml" "$fake_root/config/learned_metrics.example.toml"
# Throwaway venvs have no torch; stub SlowFast warm-up for config-merge dry-runs only.
cat > "$fake_root/python/pixvqa_learned_install/backbones.py" <<'EOF'
"""Offline test stub: config-merge dry-runs must not require torch in throwaway venvs."""
def main() -> int:
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
EOF

commit=166f5ae95ecbd841ae727984cd237de213875399

make_pinned_venv() {
  local dir="$1"
  # Real venv required: install_learned_metrics.sh calls assert_learned_venv_isolated
  # (rejects shell-wrapper interpreters from the pre-isolation era).
  local py="${PIXVQA_LEARNED_CONFIG_PYTHON:-python3}"
  "$py" -m venv "$dir"
  printf '%s\n' "$commit" > "$dir/COLORVIDEOVDP_COMMIT"
}

make_chipqa_infer() {
  local dir="$1"
  mkdir -p "$dir/bin"
  cat > "$dir/bin/python" <<'PY'
#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sklearn.__version__* ]]; then exit 0; fi
  if [[ "$2" == *joblib.load* ]]; then exit 0; fi
  if [[ "$2" == *sys.version_info* ]]; then exit 0; fi
fi
exit 0
PY
  chmod +x "$dir/bin/python"
  printf '1.0.2\n' > "$dir/SKLEARN_PIN"
}

stub_ensure_learned() {
  cat > "$fake_root/scripts/ensure_learned_python.sh" <<'EOF'
#!/usr/bin/env bash
printf '/bin/true\n'
EOF
  chmod +x "$fake_root/scripts/ensure_learned_python.sh"
}

stub_chipqa_stack() {
  cat > "$fake_root/scripts/ensure_learned_ffmpeg_cli.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$fake_root/scripts/ensure_learned_ffmpeg_cli.sh"
  cat > "$fake_root/scripts/ensure_cvvdp_dependencies.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$fake_root/scripts/ensure_cvvdp_dependencies.sh"
  cat > "$fake_root/scripts/ensure_simplevqa_dependencies.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$fake_root/scripts/ensure_simplevqa_dependencies.sh"
  cat > "$fake_root/scripts/ensure_chipqa_inference_python.sh" <<'EOF'
#!/usr/bin/env bash
printf '/usr/bin/true\n'
EOF
  chmod +x "$fake_root/scripts/ensure_chipqa_inference_python.sh"
  cat > "$fake_root/scripts/setup_hdr_chipqa_envs.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$2/bin/python"
EOF
  chmod +x "$fake_root/scripts/setup_hdr_chipqa_envs.sh"
}

# 1) Fresh tree: pinned .venv, offline asset pass creates config paths.
make_pinned_venv "$fake_root/.venv"
make_chipqa_infer "$fake_root/.venv-chipqa-inference"
stub_ensure_learned
stub_chipqa_stack
(
  cd "$fake_root"
  export PIXVQA_LEARNED_INSTALL_OFFLINE=1
  export PIXVQA_LEARNED_CONFIG_PYTHON="$(command -v python3)"
  unset PIXVQA_LEARNED_ENV_DIR PIXVQA_CVVDP_SOURCE || true
  bash scripts/install_learned_metrics.sh
) >/dev/null 2>&1 || true  # offline without deps may exit 1; config merge must still run

[[ -f "$fake_root/config/learned_metrics.local.toml" ]] || fail 'local config not created/updated'
grep -Eq 'python\s*=' "$fake_root/config/learned_metrics.local.toml" ||
  fail 'config missing learned python path'
before="$(cat "$fake_root/config/learned_metrics.local.toml")"

# 2) Re-run preserves user-facing keys (checkpoint override).
echo 'user_checkpoint="/keep/me.pth"' >> "$fake_root/config/learned_metrics.local.toml"
(
  cd "$fake_root"
  export PIXVQA_LEARNED_INSTALL_OFFLINE=1
  export PIXVQA_LEARNED_CONFIG_PYTHON="$(command -v python3)"
  bash scripts/install_learned_metrics.sh
) >/dev/null 2>&1 || true
grep -q 'user_checkpoint="/keep/me.pth"' "$fake_root/config/learned_metrics.local.toml" ||
  fail 're-run modified unrelated config lines'

# 3) WSL venv --copies helper (unchanged).
stub_bin="$work/stub-bin"
mkdir -p "$stub_bin"
cat > "$stub_bin/findmnt" <<'EOF'
#!/bin/sh
if [ "$1" = -n ] && [ "$2" = -o ] && [ "$3" = FSTYPE ] && [ "$4" = -T ]; then
  printf '%s\n' "9p"
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

echo "OK   install_learned_metrics offline tests passed"
