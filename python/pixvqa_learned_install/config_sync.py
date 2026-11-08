"""Non-destructive, section-aware synchronization of learned metric paths."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
KEY = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*=")

def _entries(lines: list[str]) -> dict[tuple[str, str], tuple[int, str]]:
    out: dict[tuple[str, str], tuple[int, str]] = {}
    section = ""
    for i, line in enumerate(lines):
        heading = SECTION.match(line)
        if heading:
            section = heading.group(1)
            continue
        if line.lstrip().startswith("#"):
            continue
        match = KEY.match(line)
        if match and section:
            try:
                value = json.loads(line.split("=", 1)[1].strip())
            except (ValueError, TypeError):
                value = None
            out[(section, match.group(1))] = (i, value if isinstance(value, str) else "")
    return out

def _put(lines: list[str], section: str, key: str, value: str) -> list[str]:
    entries = _entries(lines)
    line = f"{key} = {json.dumps(value)}"
    if (section, key) in entries:
        lines[entries[(section, key)][0]] = line
        return lines
    start = next((i for i, text in enumerate(lines) if SECTION.match(text) and SECTION.match(text).group(1) == section), None)
    if start is None:
        return lines + ["", f"[{section}]", line]
    return lines[:start+1] + [line] + lines[start+1:]

def merge_config(example: Path, destination: Path, defaults: dict[str, str]) -> tuple[bool, list[str]]:
    template_lines = example.read_text(encoding="utf-8").splitlines()
    template = _entries(template_lines)
    created = not destination.exists()
    lines = template_lines if created else destination.read_text(encoding="utf-8").splitlines()
    messages: list[str] = []
    changed = created
    for name, value in defaults.items():
        section, key = name.split(".", 1)
        current = _entries(lines).get((section, key))
        templ = template.get((section, key))
        # Preserve deliberate custom choices; replace only absent, empty or
        # untouched stock example values (which may point to missing assets).
        if current is not None and current[1] and (templ is None or current[1] != templ[1]):
            # Earlier installers dereferenced venv/bin/python to the host Python
            # before writing local config. Repair that generated value when it
            # matches the physical target of the selected managed venv Python.
            # Leave unrelated custom interpreter paths untouched.
            interpreter_key = name == "learned.python" or name in (
                "hdr-chipqa.feature_python", "hdr-chipqa.inference_python"
            )
            legacy_host_path = str(Path(value).resolve()) if interpreter_key else ""
            legacy_base = False
            if name == "learned.python" and current[1] != value and Path(value).is_file():
                # A previous installer persisted the base Python 3.12 executable
                # instead of .venv/bin/python. Repair only that identifiable
                # erroneous configuration, not a working custom virtualenv.
                current_path = Path(current[1])
                if current_path.is_file():
                    try:
                        result = subprocess.run(
                            [str(current_path), "-c", "import sys; print(sys.prefix == sys.base_prefix)"],
                            capture_output=True, text=True, timeout=8,
                        )
                        legacy_base = result.returncode == 0 and result.stdout.strip() == "True"
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            if not ((interpreter_key and current[1] == legacy_host_path and current[1] != value) or legacy_base):
                messages.append(f"preserved [{section}].{key}")
                continue
            messages.append(f"repaired legacy interpreter path [{section}].{key}")
        lines = _put(lines, section, key, value)
        changed = True
        messages.append(f"configured [{section}].{key}")
    if changed:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temp.replace(destination)
    return created, messages
