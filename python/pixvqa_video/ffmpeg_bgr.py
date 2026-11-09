"""Bounded-memory, FFmpeg-backed BGR24 capture for learned VQA adapters.

Implements only the cv2.VideoCapture API used by the pinned upstream models.
It intentionally does not expose seek or arbitrary capture properties.
Use PIXVQA_LEARNED_DECODER=opencv to compare against the original decoder.
"""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

import numpy as np


class FFmpegCapture:
    """Sequential FFmpeg BGR24 pipe with explicit subprocess lifecycle."""

    def __init__(self, path=None, *args):
        self._proc = None
        self._width = 0
        self._height = 0
        self._frames = 0
        self._fps = 0.0
        self._read_frames = 0
        self._released = True
        if path is not None:
            if not self.open(path, *args):
                raise RuntimeError(f"FFmpegCapture failed to open {path!r}")

    def open(self, path, *args):
        import cv2

        if args and args[0] != cv2.CAP_FFMPEG:
            raise ValueError("FFmpegCapture accepts only the FFmpeg backend")
        self.release()
        source = os.fspath(path)
        ffprobe = os.environ.get("PIXVQA_FFPROBE_BIN", "ffprobe")
        ffmpeg = os.environ.get("PIXVQA_FFMPEG_BIN", "ffmpeg")
        meta = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
             "-of", "json", source],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(meta.stdout).get("streams") or []
        if not streams:
            raise ValueError(f"ffprobe found no video stream: {source}")
        stream = streams[0]
        self._width = int(stream["width"])
        self._height = int(stream["height"])
        self._fps = float(Fraction(stream["avg_frame_rate"]))
        if self._fps <= 0 or self._width <= 0 or self._height <= 0:
            raise ValueError("invalid FFmpeg video geometry or frame rate")
        count = stream.get("nb_frames")
        if count in (None, "N/A"):
            counted = subprocess.run(
                [ffprobe, "-v", "error", "-count_frames",
                 "-select_streams", "v:0", "-show_entries",
                 "stream=nb_read_frames", "-of", "default=nw=1:nk=1", source],
                capture_output=True, text=True, check=True,
            )
            count = counted.stdout.strip()
        self._frames = int(count)
        if self._frames <= 0:
            raise ValueError("FFmpegCapture requires a known positive frame count")
        self._proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-threads", "1", "-i", source, "-map", "0:v:0",
             "-an", "-sn", "-dn", "-fps_mode", "passthrough",
             "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._read_frames = 0
        self._released = False
        return True

    def isOpened(self):
        return self._proc is not None and not self._released

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

    def read(self):
        if not self.isOpened():
            return False, None
        if self._read_frames >= self._frames:
            return False, None
        assert self._proc is not None and self._proc.stdout is not None
        size = self._width * self._height * 3
        data = self._proc.stdout.read(size)
        if len(data) != size:
            raise RuntimeError(
                f"FFmpegCapture truncated frame {self._read_frames}: "
                f"expected {size} bytes, got {len(data)}"
            )
        self._read_frames += 1
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
