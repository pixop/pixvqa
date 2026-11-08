"""Non-destructive, section-aware synchronization of learned metric paths."""
from __future__ import annotations

import json
import re
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
            messages.append(f"preserved [{section}].{key}")
            continue
        lines = _put(lines, section, key, value)
        changed = True
        messages.append(f"configured [{section}].{key}")
    if changed:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temp.replace(destination)
    return created, messages
