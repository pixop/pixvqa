#!/usr/bin/env bash
# Choose the PixVQA install prefix and run smoke QA on that installed binary.
# A PREFIX that is another PixVQA checkout, or a directory inside one, is
# ignored. Any other PREFIX is a custom location and is preserved.

pixvqa_checkout_root() {
  local dir="$1"
  while [[ -n "$dir" && "$dir" != "/" ]]; do
    if [[ -f "$dir/install_pixvqa.sh" && -f "$dir/build.sh" && -f "$dir/scripts/install.sh" ]]; then
      (cd -- "$dir" && pwd -P)
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done
  return 1
}

pixvqa_select_prefix() {
  local src="$1" requested probe root parent base
  src="$(cd -- "$src" && pwd -P)"
  requested="${PREFIX-}"
  if [[ -z "$requested" ]]; then
    printf '%s\n' "$src"
    return 0
  fi
  if [[ -d "$requested" ]]; then
    probe="$(cd -- "$requested" && pwd -P)"
  else
    parent="$(dirname -- "$requested")"
    if [[ -d "$parent" ]]; then
      probe="$(cd -- "$parent" && pwd -P)"
    else
      probe=""
    fi
  fi
  root=""
  if [[ -n "$probe" ]]; then
    root="$(pixvqa_checkout_root "$probe" || true)"
  fi
  if [[ -n "$root" && "$root" != "$src" ]]; then
    printf 'select_prefix: PREFIX=%s belongs to another PixVQA checkout (%s). Installing into %s.\n' \
      "$requested" "$root" "$src" >&2
    printf '%s\n' "$src"
    return 0
  fi
  if [[ -d "$requested" ]]; then
    printf '%s\n' "$(cd -- "$requested" && pwd -P)"
    return 0
  fi
  parent="$(dirname -- "$requested")"
  base="$(basename -- "$requested")"
  if [[ -d "$parent" ]]; then
    printf '%s/%s\n' "$(cd -- "$parent" && pwd -P)" "$base"
    return 0
  fi
  printf '%s\n' "$requested"
}

pixvqa_run_installed_smoke() {
  local src="$1" prefix="$2" bin bin_real prefix_real
  bin="$prefix/bin/pixvqa"
  [[ -x "$bin" ]] || {
    printf 'STOP: installed executable missing: %s\n' "$bin" >&2
    return 1
  }
  bin_real="$(readlink -f "$bin")"
  prefix_real="$(cd -- "$prefix" && pwd -P)"
  case "$bin_real" in
    "$prefix_real"/*) ;;
    *)
      printf 'STOP: smoke binary %s is outside %s\n' "$bin_real" "$prefix_real" >&2
      return 1
      ;;
  esac
  (
    cd -- "$src"
    env -u LD_LIBRARY_PATH PIXVQA_QA_CPU="$bin_real" python3 scripts/run_qa.py --profile smoke
  )
}
