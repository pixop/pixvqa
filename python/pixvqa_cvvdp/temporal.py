"""Frame-indexed CVVDP quality derived from the official run's own features.

The official CLI is run once over the whole video with its ``--features`` flag.
That flag writes ``stats['Q_per_ch']`` after the JOD has been computed, so it
cannot change the score. ``Q_per_ch[batch, channel, frame, band]`` is each
frame's distortion after the sustained and transient temporal filters (which
read neighbouring frames, with replicate padding at the clip edges), the
masking model, and spatial pooling within each band.

The official ``do_pooling_and_jods`` then pools bands (beta_sch), channels
(beta_tch), and frames (normalized Lp, beta_t), and maps the result to JOD with
``met2jod``. This module stops before the frame pool:

    q[f] = channel_pool(band_pool(weighted Q_per_ch[:, :, f, :]))
    jod_equivalent[f] = met2jod(q[f])

``jod_equivalent[f]`` is the JOD that CVVDP assigns to a video whose pooled
distortion stays at q[f] for its whole duration, because the normalized Lp mean
of a constant sequence is that constant. It is an extension of the published
model: not an official per-frame JOD and not the score of frame f evaluated on
its own. The official JOD is met2jod(Lp_mean(q)), not the mean of
jod_equivalent. Both identities are checked on every run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

TEMPORAL_NOTE = (
    "jod_equivalent[f] = met2jod(q[f]), where q[f] is the official band- and channel-pooled "
    "distortion of frame f from the single full-video CVVDP pass (temporal filtering preserved). "
    "It is the JOD of a video that sustained frame f's distortion; an extension of the published "
    "model, not an official per-frame JOD."
)


def features_path(output_dir: Path, test_path: str) -> Path:
    """Where run_cvvdp writes --features: <output-dir>/<test basename>_fmap.json."""
    stem = Path(test_path).name
    dot = stem.rfind(".")
    if dot > 0:
        stem = stem[:dot]
    return output_dir / f"{stem}_fmap.json"


def load_q_per_ch(path: Path, torch, device):
    """Rebuild Q_per_ch[batch, channel, frame, band] exactly from the features JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    channels = {}
    for key, value in data.items():
        if key.startswith("t") and "_b" in key:
            cc, bb = key[1:].split("_b", 1)
            if cc.isdigit() and bb.isdigit():
                channels[(int(cc), int(bb))] = value
    if not channels:
        raise ValueError(f"{path} has no t<channel>_b<band> features")
    n_ch = 1 + max(cc for cc, _ in channels)
    n_band = 1 + max(bb for _, bb in channels)
    if len(channels) != n_ch * n_band:
        raise ValueError(f"{path} has an incomplete channel/band grid")
    # Each entry is Q_per_ch[:, cc, :, bb].tolist(): [batch][frame]. float32 values
    # survive the JSON round trip exactly.
    stacked = [[channels[(cc, bb)] for bb in range(n_band)] for cc in range(n_ch)]
    q = torch.tensor(stacked, dtype=torch.float32, device=device)  # [ch, band, batch, frame]
    q = q.permute(2, 0, 3, 1).contiguous()  # [batch, ch, frame, band]
    return q, data


def channel_pooled_distortion(metric, q_per_ch, torch):
    """do_pooling_and_jods up to, but excluding, the pooling across frames.

    Mirrors pycvvdp.cvvdp_metric.cvvdp.do_pooling_and_jods at the pinned commit,
    using the metric's own parameters and lp_norm.
    """
    no_channels = q_per_ch.shape[1]
    no_bands = q_per_ch.shape[3]
    per_ch_w = metric.get_ch_weights(no_channels)
    per_sband_w = torch.ones((1, no_channels, 1, no_bands), dtype=torch.float32, device=metric.device)
    per_sband_w[:, :, 0, -1] = metric.baseband_weight[0:no_channels]
    q_sc = metric.lp_norm(q_per_ch * per_ch_w * per_sband_w, metric.beta_sch, dim=3, normalize=False)
    if metric.block_channels is not None:
        return metric.lp_norm(q_sc[metric.block_channels[0:no_channels], ...], metric.beta_tch, dim=1,
                              normalize=False)
    return metric.lp_norm(q_sc, metric.beta_tch, dim=1, normalize=False)


def frame_sequence(features_file: Path, display: str, device: str, official_jod: float) -> dict:
    """Per-frame JOD-equivalent sequence plus the parity checks against the official JOD."""
    import torch
    import pycvvdp

    torch_device = torch.device(device)
    metric = pycvvdp.cvvdp(display_name=display, device=torch_device, quiet=True, temp_padding="replicate")
    if getattr(metric, "temp_resample", False):
        raise ValueError("CVVDP temporal resampling is enabled; frames no longer map to decoded frames")
    with torch.no_grad():
        q_per_ch, data = load_q_per_ch(features_file, torch, torch_device)
        if q_per_ch.shape[0] != 1:
            raise ValueError("CVVDP batch features are unsupported")
        n_frames = q_per_ch.shape[2]
        upstream_jod = metric.do_pooling_and_jods(q_per_ch).item()
        q_tc = channel_pooled_distortion(metric, q_per_ch, torch)
        if n_frames > 1:
            mirrored_jod = metric.met2jod(
                metric.lp_norm(q_tc, metric.beta_t, dim=2, normalize=True).squeeze()).item()
        else:
            mirrored_jod = float("nan")
        q_frames = q_tc.reshape(-1)
        jod_frames = metric.met2jod(q_frames)
    fps = float(data.get("frames_per_second") or 0.0)
    distortion = [float(v) for v in q_frames.cpu().tolist()]
    jod_equivalent = [float(v) for v in jod_frames.cpu().tolist()]
    features_parity = upstream_jod == official_jod
    mirror_parity = mirrored_jod == upstream_jod
    samples = [
        {
            "frame_index": f,
            "time_seconds": f / fps if fps > 0 else None,
            "distortion": distortion[f],
            "jod_equivalent": jod_equivalent[f],
        }
        for f in range(n_frames)
    ]
    finite = all(math.isfinite(s["jod_equivalent"]) for s in samples)
    return {
        "basis": "per-frame-jod-equivalent",
        "method": "met2jod(channel_pool(band_pool(Q_per_ch[:, :, f, :]))) from the official --features",
        "note": TEMPORAL_NOTE,
        "frame_count": n_frames,
        "frames_per_second": fps,
        "dtype": "float32",
        "device": device,
        "official_jod": official_jod,
        "upstream_pooling_jod": upstream_jod,
        "mirrored_pooling_jod": mirrored_jod,
        "features_parity": features_parity,
        "mirror_parity": mirror_parity,
        "pooled_parity": features_parity and mirror_parity,
        "single_frame": n_frames == 1,
        "qcs_eligible": features_parity and mirror_parity and n_frames > 1 and finite,
        "mean_jod_equivalent": sum(jod_equivalent) / n_frames if n_frames else None,
        "samples": samples,
    }
