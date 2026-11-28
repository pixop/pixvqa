"""Resolve presentation-frame counts without changing FFmpeg's playback policy.

MP4 nb_frames counts track samples, including edit-list preroll. Packet
discard flags (or a change in packet count) identify files needing an exact
decode count. Normal MP4s need only a packet scan, not a second video decode.
Only regular, unchanged local files are cached within the current process.
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from functools import lru_cache

_MOV_FORMATS = frozenset(("mov", "mp4", "m4a", "3gp", "3g2", "mj2"))


def is_mov_container(format_name: str | None) -> bool:
    return bool(_MOV_FORMATS.intersection((format_name or "").split(",")))


def probe_checked(ffprobe: str, args: list[str]) -> str:
    """Reject error-level diagnostics even if ffprobe returns zero."""
    result = subprocess.run(
        [ffprobe, "-v", "error", *args], capture_output=True, text=True,
    )
    if result.returncode or result.stderr.strip():
        raise RuntimeError(
            f"ffprobe failed (exit={result.returncode}): "
            f"{result.stderr.strip()[-1000:] or 'no diagnostic'}"
        )
    return result.stdout


def _positive_count(value: str, label: str) -> int:
    try:
        count = int(value.strip())
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"ffprobe returned invalid {label}: {value!r}") from exc
    if count <= 0:
        raise RuntimeError(f"ffprobe returned non-positive {label}: {count}")
    return count


def _mp4_packet_summary(ffprobe: str, source: str) -> tuple[int, bool]:
    """Stream packet flags rather than accumulating a packet list in memory."""
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=flags", "-of", "csv=p=0", source,
    ]
    count = 0
    discarded = False
    corrupt = False
    # Keep stderr off a PIPE: a noisy corrupt stream must not deadlock stdout.
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                              text=True) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                flags = line.strip().split(",", 1)[0]
                if not flags:
                    continue
                count += 1
                discarded |= "D" in flags
                corrupt |= "C" in flags
            status = proc.wait()
        errors.seek(0, os.SEEK_END)
        errors.seek(max(0, errors.tell() - 1000))
        diagnostic = errors.read().decode("utf-8", errors="replace").strip()
    if status or diagnostic or corrupt:
        raise RuntimeError(
            f"ffprobe packet scan failed (exit={status}, corrupt={corrupt}): "
            f"{diagnostic or 'no diagnostic'}"
        )
    return count, discarded


def _decoded_count(ffprobe: str, source: str) -> int:
    return _positive_count(probe_checked(ffprobe, [
        "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", source,
    ]), "decoded frame count")


def _resolve_count(ffprobe: str, source: str, advertised: int | None,
                   mov: bool) -> tuple[int, str]:
    if advertised is None:
        return _decoded_count(ffprobe, source), "decoded_count"
    if advertised <= 0:
        raise RuntimeError("FFmpegCapture requires a known positive frame count")
    if not mov:
        return advertised, "container"
    packets, discarded = _mp4_packet_summary(ffprobe, source)
    if not discarded and packets == advertised:
        return advertised, "container"
    # Prove the underlying track is complete before treating a playback-count
    # discrepancy as an edit-list effect. Never use ignore_editlist for scoring.
    coded = _positive_count(probe_checked(ffprobe, [
        "-ignore_editlist", "1", "-count_packets", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_packets", "-of", "default=nw=1:nk=1", source,
    ]), "unrestricted coded packet count")
    if coded != advertised:
        raise RuntimeError(
            f"MP4 coded track count mismatch: container={advertised}, "
            f"packets_without_edit_list={coded}"
        )
    return _decoded_count(ffprobe, source), "mp4_edit_list_decoded_count"


def _fingerprint(source: str) -> tuple[int, ...] | None:
    try:
        st = os.stat(source)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


@lru_cache(maxsize=32)
def _cached_count(ffprobe: str, source: str, advertised: int | None,
                  mov: bool, fingerprint: tuple[int, ...]) -> tuple[int, str]:
    return _resolve_count(ffprobe, source, advertised, mov)


def resolve_presentation_frame_count(ffprobe: str, path: str | os.PathLike,
                                     advertised: int | None,
                                     format_name: str | None) -> tuple[int, str]:
    """Return a positive presentation count and its provenance.

    The raw MP4 sample count remains available to callers as separate metadata.
    Cached counts are invalidated by file identity, size or timestamp changes.
    """
    source = os.fspath(path)
    mov = is_mov_container(format_name)
    if advertised is not None and not mov:
        return _resolve_count(ffprobe, source, advertised, mov)
    fingerprint = _fingerprint(source)
    if fingerprint is None:
        return _resolve_count(ffprobe, source, advertised, mov)
    source = os.path.abspath(source)
    count = _cached_count(ffprobe, source, advertised, mov, fingerprint)
    if _fingerprint(source) != fingerprint:
        raise RuntimeError("video changed during presentation-frame-count probe")
    return count
