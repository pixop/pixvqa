"""Bounded-memory, FFmpeg-backed BGR24 capture for learned VQA adapters.

Implements the sequential cv2.VideoCapture subset used by the pinned
SimpleVQA upstream functions and CVQA-HDR-NR's local preprocessor.
No OpenCV video decoder fallback, seeks, or frame-rate conversion.
"""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

from pixvqa_video.ffmpeg_bins import FFmpegBinError, resolve_ffmpeg_bin, resolve_ffprobe_bin
from pixvqa_video.frame_count import probe_checked, resolve_presentation_frame_count


class FFmpegCapture:
    """Sequential FFmpeg BGR24 pipe with explicit subprocess lifecycle."""

    def __init__(self, path=None, *args):
        self._proc = None
        self._width = 0
        self._height = 0
        self._frames = 0
        self.advertised_frame_count = None
        self.frame_count_basis = None
        self._fps = 0.0
        self._read_frames = 0
        self._released = True
        if path is not None:
            if not self.open(path, *args):
                raise RuntimeError(f"FFmpegCapture failed to open {path!r}")

    def open(self, path, *args):
        # Preserve the upstream no-arg constructor + open(path) pattern.
        # Backend options are not accepted; FFmpeg is the only decoder.
        if args:
            raise ValueError("FFmpegCapture does not accept backend options")
        self.release()
        source = os.fspath(path)
        try:
            ffprobe = resolve_ffprobe_bin()
            ffmpeg = resolve_ffmpeg_bin()
        except FFmpegBinError as exc:
            raise RuntimeError(str(exc)) from exc
        meta = json.loads(probe_checked(ffprobe, [
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,avg_frame_rate:format=format_name",
            "-of", "json", source,
        ]))
        streams = meta.get("streams") or []
        if not streams:
            raise ValueError(f"ffprobe found no video stream: {source}")
        stream = streams[0]
        self._width = int(stream["width"])
        self._height = int(stream["height"])
        self._fps = float(Fraction(stream["avg_frame_rate"]))
        if self._fps <= 0 or self._width <= 0 or self._height <= 0:
            raise ValueError("invalid FFmpeg video geometry or frame rate")
        count = stream.get("nb_frames")
        self.advertised_frame_count = None if count in (None, "N/A") else int(count)
        self._frames, self.frame_count_basis = resolve_presentation_frame_count(
            ffprobe, source, self.advertised_frame_count,
            (meta.get("format") or {}).get("format_name"),
        )
        self._command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-nostdin",
            "-threads", "1", "-i", source, "-map", "0:v:0",
            "-an", "-sn", "-dn", "-fps_mode", "passthrough",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
        ]
        self._read_frames = 0
        self._released = False
        return True

    def isOpened(self):
        return not self._released

    def getBackendName(self):
        return "FFMPEG_PIPE"

    def get(self, prop):
        import cv2

        return {
            cv2.CAP_PROP_FRAME_COUNT: float(self._frames),
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: float(self._width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._height),
        }.get(prop, 0.0)

    def _finish_decode(self):
        """Validate completion even when the caller reads exactly the count."""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        remaining = proc.stdout.read(1)
        if remaining:
            # Do NOT wait while the child can still be writing to a full pipe.
            self.release()
            raise RuntimeError("FFmpegCapture decoded more frames than the presentation count")
        status = proc.wait()
        proc.stdout.close()
        self._proc = None
        if status != 0:
            self._released = True
            raise RuntimeError(f"FFmpegCapture decoder failed at EOF (exit={status})")

    def read(self):
        if not self.isOpened() or self._read_frames >= self._frames:
            return False, None
        if self._proc is None:
            self._proc = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        assert self._proc.stdout is not None
        size = self._width * self._height * 3
        data = self._proc.stdout.read(size)
        if len(data) != size:
            status = self._proc.wait()
            self.release()
            raise RuntimeError(
                f"FFmpegCapture truncated frame {self._read_frames}: "
                f"expected {size} bytes, got {len(data)} (exit={status})"
            )
        self._read_frames += 1
        if self._read_frames == self._frames:
            self._finish_decode()
        import numpy as np

        return True, np.frombuffer(data, dtype=np.uint8).reshape(
            self._height, self._width, 3
        ).copy()

    def grab(self):
        ok, _ = self.read()
        return ok

    def release(self):
        proc = self._proc
        self._proc = None
        self._released = True
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
