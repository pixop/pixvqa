#!/usr/bin/env python3
"""Fast mock tests for SimpleVQA persistent worker lifecycle and I/O hardening."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_simplevqa.adapter import InferenceError  # noqa: E402
from pixvqa_simplevqa import worker as worker_mod  # noqa: E402
from pixvqa_simplevqa.worker import (  # noqa: E402
    _WorkerSession,
    run_via_worker,
    shutdown_worker,
)


def _write_script(tmp: Path, body: str) -> Path:
    path = tmp / f"fake_child_{os.getpid()}_{id(body)}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _minimal_repo(tmp: Path) -> tuple[Path, Path]:
    repo = tmp / "repo"
    repo.mkdir()
    (repo / "test_demo.py").write_text("# mock upstream\npass\n", encoding="utf-8")
    (repo / "model").mkdir()
    (repo / "model" / "UGC_BVQA_model.py").write_text("# mock\n", encoding="utf-8")
    ckpt = tmp / "ckpt.pth"
    ckpt.write_bytes(b"fake")
    return repo, ckpt


def _stub_overlay(src: Path) -> str:
    return "# staged overlay\n"


def test_startup_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import time
            time.sleep(30)
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "clip.mp4"
        video.write_bytes(b"x")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            try:
                sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=0.6)
                sess.score(video, timeout=5)
            except InferenceError as exc:
                assert "startup timed out" in str(exc).lower()
            else:
                raise AssertionError("expected startup timeout")
            finally:
                if "sess" in dir():
                    sess.close()


def test_response_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import json, sys, time
            sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                if req.get("cmd") == "score":
                    time.sleep(30)
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "clip.mp4"
        video.write_bytes(b"x")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=10)
            try:
                try:
                    sess.score(video, timeout=0.6)
                except InferenceError as exc:
                    assert "timed out" in str(exc).lower()
                else:
                    raise AssertionError("expected response timeout")
                assert sess._proc is None or sess._proc.poll() is not None or sess._broken
            finally:
                sess.close()


def test_broken_pipe_on_request():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import json, sys
            sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                if line.strip():
                    break
            raise SystemExit(3)
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "clip.mp4"
        video.write_bytes(b"x")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=10)
            try:
                try:
                    sess.score(video, timeout=5)
                except InferenceError:
                    pass
                else:
                    raise AssertionError("expected failure on broken pipe")
            finally:
                sess.close()


def test_malformed_response():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import json, sys
            sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                if req.get("cmd") == "score":
                    sys.stdout.write("not-json\\n")
                    sys.stdout.write(json.dumps({"status": "ok", "id": "wrong-id", "score": 1.0}) + "\\n")
                    sys.stdout.flush()
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "clip.mp4"
        video.write_bytes(b"x")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=10)
            try:
                try:
                    sess.score(video, timeout=2)
                except InferenceError as exc:
                    assert "timed out" in str(exc).lower()
                else:
                    raise AssertionError("expected timeout on wrong/malformed id")
            finally:
                sess.close()


def test_restart_after_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        counter = tmp / "counter.txt"
        counter.write_text("0", encoding="utf-8")
        script = _write_script(
            tmp,
            f"""
            import json, sys
            from pathlib import Path
            ctr = Path({str(counter)!r})
            n = int(ctr.read_text())
            ctr.write_text(str(n + 1))
            if n == 0:
                sys.stdout.write(json.dumps({{"status": "ready"}}) + "\\n")
                sys.stdout.flush()
                for line in sys.stdin:
                    if line.strip():
                        raise SystemExit(9)
            else:
                sys.stdout.write(json.dumps({{"status": "ready"}}) + "\\n")
                sys.stdout.flush()
                for line in sys.stdin:
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    if req.get("cmd") == "shutdown":
                        break
                    if req.get("cmd") == "score":
                        out = req["output"]
                        vid = req["video"]
                        Path(out).write_text(f"{{vid}},0.42\\n")
                        sys.stdout.write(json.dumps({{
                            "status": "ok", "id": req["id"], "score": 0.42
                        }}) + "\\n")
                        sys.stdout.flush()
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "clip.mp4"
        video.write_bytes(b"x")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            with patch("pixvqa_simplevqa.worker.parse_score", return_value=0.42):
                sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=10)
                try:
                    try:
                        sess.score(video, timeout=5)
                    except InferenceError:
                        pass
                    score = sess.score(video, timeout=10)
                    assert score == 0.42
                    assert int(counter.read_text()) >= 2
                finally:
                    sess.close()


def test_request_id_path_collision():
    """Distinct paths with the same basename must not cross-match responses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import json, sys, time
            from pathlib import Path
            sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                if req.get("cmd") == "score":
                    time.sleep(0.05)
                    vid = req["video"]
                    out = req["output"]
                    score = 1.1 if "/dir_a/" in vid else 2.2
                    Path(out).write_text(f"{vid},{score}\\n")
                    sys.stdout.write(json.dumps({
                        "status": "ok", "id": req["id"], "score": score
                    }) + "\\n")
                    sys.stdout.flush()
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        (tmp / "dir_a").mkdir()
        (tmp / "dir_b").mkdir()
        vid_a = tmp / "dir_a" / "same.mp4"
        vid_b = tmp / "dir_b" / "same.mp4"
        vid_a.write_bytes(b"a")
        vid_b.write_bytes(b"b")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            with patch("pixvqa_simplevqa.worker.parse_score", side_effect=lambda t, v: float(t.rsplit(",", 1)[-1])):
                sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=10)
                try:
                    sa = sess.score(vid_a, timeout=10)
                    sb = sess.score(vid_b, timeout=10)
                    assert sa == 1.1
                    assert sb == 2.2
                finally:
                    sess.close()


def test_shutdown_cleans_child():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import json, sys, time
            sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                time.sleep(0.1)
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            sess = _WorkerSession(repo, ckpt, child_script=script, startup_timeout=10)
            proc = sess._proc
            assert proc is not None and proc.poll() is None
            sess.close()
            assert proc.poll() is not None
        shutdown_worker()
        assert worker_mod._SESSION is None


def test_run_via_worker_reuses_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = _write_script(
            tmp,
            """
            import json, sys
            from pathlib import Path
            sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")
            sys.stdout.flush()
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                if req.get("cmd") == "score":
                    out = req["output"]
                    vid = req["video"]
                    Path(out).write_text(f"{vid},0.5\\n")
                    sys.stdout.write(json.dumps({
                        "status": "ok", "id": req["id"], "score": 0.5
                    }) + "\\n")
                    sys.stdout.flush()
            """,
        )
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "v.mp4"
        video.write_bytes(b"v")
        shutdown_worker()
        with patch.object(worker_mod, "cuda_single_decode_overlay_test_demo", _stub_overlay):
            with patch.object(worker_mod, "_CHILD_SCRIPT", script):
                with patch("pixvqa_simplevqa.worker.parse_score", return_value=0.5):
                    s1 = run_via_worker(repo, ckpt, video, timeout=30)
                    proc1 = worker_mod._SESSION._proc if worker_mod._SESSION else None
                    s2 = run_via_worker(repo, ckpt, video, timeout=30)
                    proc2 = worker_mod._SESSION._proc if worker_mod._SESSION else None
                    assert s1 == 0.5 and s2 == 0.5
                    assert proc1 is not None and proc1 is proc2
        shutdown_worker()


if __name__ == "__main__":
    tests = [
        test_startup_timeout,
        test_response_timeout,
        test_broken_pipe_on_request,
        test_malformed_response,
        test_restart_after_crash,
        test_request_id_path_collision,
        test_shutdown_cleans_child,
        test_run_via_worker_reuses_session,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)
    print(f"PASS: {len(tests)} worker mock tests")
