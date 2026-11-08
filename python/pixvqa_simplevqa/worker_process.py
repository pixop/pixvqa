"""Child process for persistent SimpleVQA worker (import once, score many)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path


def _load_main():
    root = Path(os.environ["PIXVQA_SVQA_WORKER_ROOT"])
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    spec = importlib.util.spec_from_file_location("svqa_staged_demo", root / "test_demo.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load staged test_demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svqa_staged_demo"] = mod
    spec.loader.exec_module(mod)
    return mod.main


def main() -> int:
    score_fn = _load_main()
    sys.stdout.write(json.dumps({"status": "ready", "import_s": "done"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "score":
            continue
        req_id = str(req.get("id", ""))
        video = str(req.get("video", ""))
        output = str(req.get("output", ""))
        if not req_id:
            sys.stdout.write(
                json.dumps({"status": "error", "id": "", "error": "missing request id"})
                + "\n"
            )
            sys.stdout.flush()
            continue
        resp_id = req_id
        try:
            config = types.SimpleNamespace(
                method_name="single-scale",
                dist=video,
                output=output,
                is_gpu=True,
            )
            score_fn(config)
            out_text = Path(output).read_text(encoding="utf-8").strip()
            score_s = out_text.rsplit(",", 1)[-1]
            score = float(score_s)
            sys.stdout.write(
                json.dumps({"status": "ok", "id": resp_id, "score": score}) + "\n"
            )
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"status": "error", "id": resp_id, "error": str(exc)}) + "\n"
            )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
