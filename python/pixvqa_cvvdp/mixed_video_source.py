"""Mixed raw YUV + compressed video for ColorVideoVDP's predict_video_source API.

Upstream video_source_file picks one reader for both paths; FFmpeg cannot open
headerless .yuv when either path is not .yuv. This source uses YUVReader on raw
sides and the official ffmpeg video_reader on compressed sides, then applies the
same display-model transform as video_source_yuv_file / video_source_video_file.
"""

from __future__ import annotations

import logging

from pycvvdp.video_source import reshuffle_dims, video_source_dm
from pycvvdp.video_source_yuv import YUVReader
from pycvvdp.vq_exception import vq_exception


class PixvqaMixedVideoSource(video_source_dm):
    """Independent raw and compressed frame sources with a shared display model."""

    def __init__(
        self,
        test_path: str,
        reference_path: str,
        *,
        ref_is_raw: bool,
        test_is_raw: bool,
        display_photometry: str,
        frames: int = -1,
        fps: float | None = None,
        ffmpeg_cc: bool = False,
        verbose: bool = False,
    ) -> None:
        if ref_is_raw == test_is_raw:
            raise ValueError("PixvqaMixedVideoSource requires one raw and one compressed side")
        self.test_path = test_path
        self.reference_path = reference_path
        self.ref_is_raw = ref_is_raw
        self.test_is_raw = test_is_raw
        self.in_frames = frames
        self.fps = fps
        self.ffmpeg_cc = ffmpeg_cc
        self.verbose = verbose
        self._ref = None
        self._test = None
        self.frames = 0
        super().__init__(display_photometry=display_photometry, config_paths=[])

    def init_readers(self) -> None:
        if self._test is not None:
            return
        from pycvvdp.video_source_file import video_reader

        if self.ref_is_raw:
            self._ref = YUVReader(self.reference_path)
            self._test = video_reader(
                self.test_path, self.in_frames, verbose=self.verbose,
            )
        else:
            self._ref = video_reader(
                self.reference_path, self.in_frames, verbose=self.verbose,
            )
            self._test = YUVReader(self.test_path)

        ref_count = int(self._ref.frames)
        test_count = int(self._test.frames)
        if test_count == -1 and ref_count == -1:
            raise vq_exception(
                "Neither test nor reference video contains frame-count metadata; pass --nframes"
            )
        if test_count == -1:
            self.frames = ref_count
        elif ref_count == -1:
            self.frames = test_count
        else:
            self.frames = min(test_count, ref_count)
        if self.in_frames > 0:
            self.frames = min(self.frames, self.in_frames)
        if test_count != -1 and ref_count != -1 and test_count != ref_count:
            logging.warning(
                "Test and reference videos contain different frame counts (%s and %s). "
                "Comparing %s frames.",
                test_count,
                ref_count,
                self.frames,
            )

        ref_fps = float(self._ref.avg_fps)
        test_fps = float(self._test.avg_fps)
        if self.fps is None:
            self.fps = test_fps if not self.test_is_raw else ref_fps
        if abs(ref_fps - test_fps) > 1e-3:
            raise vq_exception(
                f"Test and reference frame rates differ: test {test_fps} fps, reference {ref_fps} fps"
            )

        for side, path, reader in (
            ("test", self.test_path, self._test),
            ("reference", self.reference_path, self._ref),
        ):
            logging.debug(
                "Mixed CVVDP %s '%s': [%sx%s], fps: %s, frames: %s",
                side,
                path,
                reader.width,
                reader.height,
                self.fps,
                self.frames,
            )

    def get_video_size(self):
        self.init_readers()
        return (self._test.height, self._test.width, self.frames)

    def get_frames_per_second(self) -> float:
        self.init_readers()
        return float(self.fps)

    def get_test_frame(self, frame, device, colorspace="Y"):
        self.init_readers()
        if self.test_is_raw:
            return self._frame_from_yuv(self._test, frame, device, colorspace)
        return self._frame_from_video(self._test, frame, device, colorspace)

    def get_reference_frame(self, frame, device, colorspace="Y"):
        self.init_readers()
        if self.ref_is_raw:
            return self._frame_from_yuv(self._ref, frame, device, colorspace)
        return self._frame_from_video(self._ref, frame, device, colorspace)

    def _frame_from_yuv(self, reader: YUVReader, frame: int, device, colorspace: str):
        rgb = reader.get_frame_rgb_tensor(frame, device)
        tensor = reshuffle_dims(rgb, in_dims="HWC", out_dims="BCFHW")
        return self.apply_dm_and_color_transform(tensor, colorspace)

    def _frame_from_video(self, reader, frame: int, device, colorspace: str):
        if frame != reader.curr_frame + 1:
            raise vq_exception(
                "Video can be currently only read frame-by-frame. Random access not implemented."
            )
        frame_np = reader.get_frame()
        if frame_np is None:
            raise vq_exception(
                f'Could not read frame {frame} of "{reader.fname}". Try passing "--nframes".'
            )
        rgb = reader.unpack(frame_np, device)
        tensor = reshuffle_dims(rgb, in_dims="HWC", out_dims="BCFHW")
        return self.apply_dm_and_color_transform(tensor, colorspace)

