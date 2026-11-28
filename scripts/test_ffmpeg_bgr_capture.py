#!/usr/bin/env python3
"""Shared learned-decoder regression: no checkpoint or GPU needed.

Default: real fixture tests against the pinned learned FFmpeg pair.
PIXVQA_CAPTURE_TEST_HOST=1: inject the encoder-capable host pair for regular
native full QA (test only; production pin enforcement is unchanged).
--unit-only: subprocess/count contracts without FFmpeg, for hosted smoke CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from pixvqa_video import ffmpeg_bgr, frame_count
from pixvqa_video.ffmpeg_bgr import FFmpegCapture
from pixvqa_cvqahdr.adapter import sampling_plan, sample_schedule
from pixvqa_simplevqa import adapter as simple

# Only enum values are needed by get(); no OpenCV decoder or learned stack.
CV2 = SimpleNamespace(CAP_PROP_FRAME_COUNT=7, CAP_PROP_FRAME_WIDTH=3,
                      CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5)


def fake_decoder(frames: int, nbytes: int, exit_code: int = 0) -> FFmpegCapture:
    cap = FFmpegCapture()
    cap._width = 2
    cap._height = 2
    cap._frames = frames
    cap._released = False
    cap._command = [sys.executable, "-c",
                    f"import sys; sys.stdout.buffer.write(bytes({nbytes})); sys.stdout.flush(); sys.exit({exit_code})"]
    return cap


class PipeContracts(unittest.TestCase):
    def test_exact_count_checks_eof_on_last_read(self):
        with fake_decoder(2, 24) as cap:
            self.assertTrue(cap.read()[0])
            self.assertTrue(cap.read()[0])
            self.assertIsNone(cap._proc)
            self.assertEqual(cap.read(), (False, None))
            self.assertEqual(cap.read(), (False, None))

    def test_unreconciled_early_clean_eof_is_failure(self):
        with fake_decoder(2, 12) as cap:
            self.assertTrue(cap.read()[0])
            with self.assertRaisesRegex(RuntimeError, r"truncated frame 1.*got 0.*exit=0"):
                cap.read()
            self.assertFalse(cap.isOpened())

    def test_partial_frame_is_failure(self):
        with fake_decoder(2, 17) as cap:
            self.assertTrue(cap.read()[0])
            with self.assertRaisesRegex(RuntimeError, r"truncated frame 1.*got 5"):
                cap.read()

    def test_decoder_nonzero_exit_at_boundary_is_failure(self):
        with fake_decoder(2, 24, 7) as cap:
            self.assertTrue(cap.read()[0])
            with self.assertRaisesRegex(RuntimeError, r"decoder failed at EOF.*exit=7"):
                cap.read()
            self.assertFalse(cap.isOpened())

    def test_decoder_nonzero_exit_before_boundary_is_failure(self):
        with fake_decoder(2, 12, 9) as cap:
            self.assertTrue(cap.read()[0])
            with self.assertRaisesRegex(RuntimeError, r"truncated frame 1.*exit=9"):
                cap.read()

    def test_extra_frames_do_not_deadlock(self):
        # Output exceeds pipe capacity. Waiting before closing the pipe hangs.
        with fake_decoder(1, 1024 * 1024) as cap:
            with self.assertRaisesRegex(RuntimeError, "more frames"):
                cap.read()
            self.assertIsNone(cap._proc)
            self.assertFalse(cap.isOpened())

    def test_prefix_can_release_without_requiring_full_decode(self):
        with fake_decoder(100, 1200) as cap:
            self.assertTrue(cap.read()[0])
        self.assertFalse(cap.isOpened())
        self.assertIsNone(cap._proc)


class CountContracts(unittest.TestCase):
    def test_no_scan_for_known_non_mov(self):
        with patch.object(frame_count, "_mp4_packet_summary") as scan, patch.object(frame_count, "_decoded_count") as decode:
            self.assertEqual(frame_count._resolve_count("probe", "file", 12, False), (12, "container"))
            scan.assert_not_called()
            decode.assert_not_called()

    def test_ordinary_mp4_does_not_decode_for_preflight(self):
        with patch.object(frame_count, "_mp4_packet_summary", return_value=(12, False)), patch.object(frame_count, "_decoded_count") as decode:
            self.assertEqual(frame_count._resolve_count("probe", "file", 12, True), (12, "container"))
            decode.assert_not_called()

    def test_edit_list_count_not_packet_subtraction(self):
        for packet_count, discarded in ((47, True), (32, False)):
            with self.subTest(packets=packet_count, discard=discarded), patch.object(frame_count, "_mp4_packet_summary", return_value=(packet_count, discarded)), patch.object(frame_count, "probe_checked", return_value="47\n") as probe, patch.object(frame_count, "_decoded_count", return_value=32):
                self.assertEqual(frame_count._resolve_count("probe", "file", 47, True), (32, "mp4_edit_list_decoded_count"))
                self.assertIn("-ignore_editlist", probe.call_args.args[1])

    def test_coded_track_mismatch_not_normalized(self):
        with patch.object(frame_count, "_mp4_packet_summary", return_value=(32, False)), patch.object(frame_count, "probe_checked", return_value="32\n"), patch.object(frame_count, "_decoded_count") as decode:
            with self.assertRaisesRegex(RuntimeError, "coded track count mismatch"):
                frame_count._resolve_count("probe", "file", 47, True)
            decode.assert_not_called()

    def test_unknown_count_counts_decoded_frames(self):
        with patch.object(frame_count, "_decoded_count", return_value=32):
            self.assertEqual(frame_count._resolve_count("probe", "file", None, True), (32, "decoded_count"))

    def test_probe_errors_rejected_even_with_zero_exit(self):
        with patch.object(frame_count.subprocess, "run", return_value=subprocess.CompletedProcess([],0,"32","corrupt slice")):
            with self.assertRaisesRegex(RuntimeError, "corrupt slice"):
                frame_count.probe_checked("probe", [])

    def test_cached_count_invalidated_on_file_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.mp4"
            path.write_bytes(b"x")
            with patch.object(frame_count, "_resolve_count", side_effect=[(32,"decoded_count"),(31,"decoded_count")]) as count:
                self.assertEqual(frame_count.resolve_presentation_frame_count("probe", path, None, "mp4")[0],32)
                self.assertEqual(frame_count.resolve_presentation_frame_count("probe", path, None, "mp4")[0],32)
                self.assertEqual(count.call_count,1)
                path.write_bytes(b"xx")
                self.assertEqual(frame_count.resolve_presentation_frame_count("probe", path, None, "mp4")[0],31)
                self.assertEqual(count.call_count,2)

    def test_simplevqa_uses_presentation_count_for_minimum_length(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"edit.mp4"
            path.write_bytes(b"x")
            stream={"nb_frames":"110","avg_frame_rate":"12/1","codec_name":"h264",
                    "width":96,"height":64,"color_transfer":"bt709"}
            fmt={"duration":"9.2","format_name":"mov,mp4,m4a,3gp,3g2,mj2"}
            def response(obj):
                return subprocess.CompletedProcess([],0,json.dumps(obj),"")
            for actual, accepted in ((95,False),(100,True)):
                with patch.object(simple,"_run_ffprobe",side_effect=[response({"streams":[stream]}),response({"format":fmt})]), patch.object(simple,"resolve_presentation_frame_count",return_value=(actual,"mp4_edit_list_decoded_count")):
                    if accepted:
                        record=simple.inspect_video(path,ffprobe="probe")
                        self.assertEqual(record["frame_count"],actual)
                        self.assertEqual(record["container_frame_count"],110)
                    else:
                        with self.assertRaisesRegex(simple.SetupError,"at least 8 spatial"):
                            simple.inspect_video(path,ffprobe="probe")


def host_pair() -> tuple[str,str]:
    ffmpeg = (os.environ.get("PIXVQA_TEST_FIXTURE_FFMPEG") or
              os.environ.get("FFMPEG") or shutil.which("ffmpeg"))
    if not ffmpeg:
        raise RuntimeError("host FFmpeg required to encode synthetic H.264 edit-list fixture")
    ffmpeg = str(Path(ffmpeg).absolute())
    ffprobe = os.environ.get("PIXVQA_TEST_FIXTURE_FFPROBE") or str(Path(ffmpeg).with_name("ffprobe"))
    if not Path(ffprobe).is_file():
        raise RuntimeError(f"host ffprobe missing: {ffprobe}; set PIXVQA_TEST_FIXTURE_FFPROBE")
    return ffmpeg,ffprobe


def run_command(tool, args, **kw):
    return subprocess.run([tool, "-hide_banner", "-loglevel", "error", *args],
                          check=True, timeout=90, **kw)


def probe_counts(probe, path):
    result = subprocess.run([probe,"-v","error","-count_frames","-select_streams","v:0",
                             "-show_entries","stream=nb_frames,nb_read_frames","-of","json",str(path)],
                            capture_output=True,text=True,check=True,timeout=90)
    if result.stderr.strip():
        raise RuntimeError(result.stderr)
    stream=json.loads(result.stdout)["streams"][0]
    return int(stream["nb_frames"]),int(stream["nb_read_frames"])


def decode_hashes(path, expected_advertised, expected_presented, expected_basis):
    with patch.dict(sys.modules, {"cv2": CV2}), FFmpegCapture(str(path)) as cap:
        assert cap.isOpened()
        assert cap.getBackendName()=="FFMPEG_PIPE"
        assert cap.advertised_frame_count==expected_advertised
        assert cap.frame_count_basis==expected_basis
        assert int(cap.get(CV2.CAP_PROP_FRAME_COUNT))==expected_presented
        assert int(cap.get(CV2.CAP_PROP_FRAME_WIDTH))==96
        assert int(cap.get(CV2.CAP_PROP_FRAME_HEIGHT))==64
        fps=cap.get(CV2.CAP_PROP_FPS)
        hashes=[]
        for _ in range(expected_presented):
            ok,frame=cap.read()
            assert ok and frame.shape==(64,96,3)
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        assert cap.read()==(False,None)
        assert cap._proc is None
        return hashes,fps


def real_fixtures():
    encoder,host_probe=host_pair()
    host_mode=os.environ.get("PIXVQA_CAPTURE_TEST_HOST")=="1"
    if host_mode:
        decoder,probe=encoder,host_probe
        print("Fixture integration: host FFmpeg pair injected in test only")
    else:
        decoder,probe=ffmpeg_bgr.resolve_ffmpeg_bin(),ffmpeg_bgr.resolve_ffprobe_bin()
        print("Fixture integration: pinned learned FFmpeg pair")
    with tempfile.TemporaryDirectory(prefix="pixvqa_capture_regression_") as d, patch.object(ffmpeg_bgr,"resolve_ffmpeg_bin",return_value=decoder), patch.object(ffmpeg_bgr,"resolve_ffprobe_bin",return_value=probe):
        root=Path(d)
        ordinary=root/"ordinary.mp4"
        run_command(encoder,["-y","-f","lavfi","-i","testsrc2=size=96x64:rate=6:duration=2",
                             "-frames:v","12","-c:v","mpeg4",str(ordinary)])
        assert probe_counts(probe,ordinary)==(12,12)
        a,fps=decode_hashes(ordinary,12,12,"container")
        b,_=decode_hashes(ordinary,12,12,"container")
        assert fps==6.0 and a==b
        with FFmpegCapture(str(ordinary)) as early:
            assert early.read()[0]
        print("PASS: ordinary MP4, deterministic pixels, prefix release")

        source=root/"source.mp4"
        edited=root/"edited.mp4"
        run_command(encoder,["-y","-f","lavfi","-i","testsrc2=size=96x64:rate=12:duration=5",
                             "-c:v","libx264","-g","60","-keyint_min","60","-sc_threshold","0",
                             "-bf","2","-pix_fmt","yuv420p",str(source)])
        run_command(encoder,["-y","-ss","1.25","-i",str(source),"-t","2.5","-map","0:v:0",
                             "-c","copy","-avoid_negative_ts","disabled","-use_editlist","1",str(edited)])
        advertised,presented=probe_counts(probe,edited)
        assert advertised>presented, (advertised,presented)
        a,fps=decode_hashes(edited,advertised,presented,"mp4_edit_list_decoded_count")
        b,_=decode_hashes(edited,advertised,presented,"mp4_edit_list_decoded_count")
        assert a==b and fps==12.0
        # Independent FFmpeg playback oracle (no ignore_editlist, no resampling).
        raw=run_command(decoder,["-xerror","-nostdin","-threads","1","-i",str(edited),
                                 "-map","0:v:0","-an","-sn","-dn","-fps_mode","passthrough",
                                 "-pix_fmt","bgr24","-f","rawvideo","-"],capture_output=True).stdout
        size=96*64*3
        assert len(raw)==presented*size
        assert a==[hashlib.sha256(raw[i:i+size]).hexdigest() for i in range(0,len(raw),size)]
        plan=sampling_plan(presented,fps)
        assert plan["sampled_frames"]==presented//12
        assert plan["sampled_frames"]<advertised//12, (advertised,presented)
        assert [x["frame_index"] for x in sample_schedule(plan)]==list(range(6,12*(presented//12),12))
        print(f"PASS: synthetic MP4 edit list: container={advertised}, playback={presented}, "
              f"CVQA midpoints={plan['sampled_frames']}; exact BGR24 oracle parity")

        # Damaged input with moov at the beginning: preserve readable metadata,
        # then truncate the coded media. A count discrepancy must not hide damage.
        damaged=root/"damaged.mp4"
        run_command(encoder,["-y","-i",str(source),"-c","copy","-movflags","+faststart",str(damaged)])
        contents=damaged.read_bytes()
        damaged.write_bytes(contents[:len(contents)*2//3])
        try:
            with FFmpegCapture(str(damaged)) as cap:
                while cap.read()[0]:
                    pass
        except (RuntimeError, ValueError, subprocess.CalledProcessError):
            print("PASS: truncated MP4 rejected")
        else:
            raise AssertionError("truncated MP4 silently accepted")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only",action="store_true")
    args=parser.parse_args()
    suite=unittest.TestSuite()
    for cls in (PipeContracts,CountContracts):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        return 1
    if not args.unit_only:
        real_fixtures()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
