# Quality Consistency Score (QCS)

**Publication:** S. Sanz-Rodríguez and J. Frydensbjerg, "[Quality Consistency Score (QCS): A Survival-Based Reliability Descriptor for Video Quality Assessment](https://ieeexplore.ieee.org/document/11629873/)," *IEEE International Conference on Image Processing (ICIP)*, 2026.

Quality Consistency Score (QCS) is a metric-agnostic descriptor of **temporal quality reliability**. It measures the proportion of viewing time during which video quality meets or exceeds specified thresholds, revealing degradations that may be hidden by average quality metrics or conventional rate-distortion (RD/BD-rate) analysis.

## Definition

Let $Q(t)$ represent the frame-level quality scores and $S(x)=P(Q\ge x)$ their survival function. QCS is the normalized area under the survival curve over a quality interval $[a,b]$:
$$
\mathrm{QCS}_{[a,b]} = \frac{1}{b-a}\int_a^b S(x)\,dx
$$
QCS ranges from 0% to 100% when expressed as a percentage. Scores at or below $a$ contribute 0%, scores at or above $b$ contribute 100%, and intermediate scores contribute proportionally. NaN values are excluded. +inf contributes 100% and -inf contributes 0%. A VMAF score above 100 is kept. With the default upper bound of 100 it contributes 100%.

Each frame contributes equally (constant-frame-rate assumption). QCS describes the **distribution of quality scores**, not their temporal ordering.

## Default QCS intervals

| Metric | Interval |
|---|---|
| PSNR / XPSNR | [30, 50] dB |
| SSIM / MS-SSIM | [0.8, 1.0] |
| VMAF (SDR / HDR) | [70, 100] |
| PU21-PSNR | [30, 50] dB |
| PU21-SSIM / PU21-MS-SSIM | [0.8, 1.0] |
| `cvvdp` (per-frame JOD-equivalent) | [0, 10] |
| `cvqa-hdr-nr` (sampled frames) | [0, 10] |

The [0, 10] interval applies only to QCS threshold integration. It does not clamp or rescale the reported CVVDP JOD or NR `quality_score`, and pooled values outside [0, 10] are reported unchanged.

VMAF uses one interval for every SDR and HDR model. PU21-PSNR uses the PSNR interval because its peak is 256, about 100 cd/m², which is an 8-bit-style decibel scale. PU21-SSIM and PU21-MS-SSIM use the SSIM interval because they are similarities on that same peak.

## Usage

Enable QCS with `--qcs`. Override a default interval with `--qcs-range metric=lower:upper`. `--qcs-range` requires `--qcs`. Several intervals may be comma-separated in one option, and the option may be repeated. The two forms can be combined. Naming one metric again with a different interval is an error; the same interval is unchanged. Every selected native metric receives a QCS value, whether that metric runs on the CPU or the GPU. `cvvdp` receives one from its per-frame JOD-equivalent sequence and `cvqa-hdr-nr` from its sampled frames.
```bash
pixvqa --metrics psnr,vmaf \
  --model model/vmaf_v1.0.16_3d0h.json \
  --qcs --qcs-range psnr=40:55,vmaf=30:100 \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```
Repeating the option is equivalent: `--qcs-range psnr=40:55 --qcs-range vmaf=30:100`.

QCS is computed independently for each selected metric. The name in `--qcs-range` is the `--metrics` name, and the bounds must satisfy lower < upper. Identical PSNR is printed `999.99`, identical XPSNR is printed `inf`, and identical PU21-PSNR is printed `999.99`. All three still count in QCS.

## Learned metrics

QCS is never computed from a pooled video score, and frame scores are never fabricated. Both learned sequences come from the single official full-video pass, so the pooled scores are unchanged.

- **`cvvdp`: per-frame JOD-equivalent, default [0, 10].** CVVDP publishes one pooled JOD per video. Upstream computes it from a per-frame, per-band, per-channel distortion tensor, `Q_per_ch`, that is already temporally filtered: the sustained and transient channels are produced by temporal filters over the whole sequence before spatial pooling. With `--qcs`, the adapter asks the official `cvvdp` command for that tensor (`--features`) in the same run that produces the pooled JOD. It then applies upstream's band and channel pooling to each frame, followed by upstream's `met2jod`:

  $$
  \mathrm{JOD}_{eq}(f) = \mathrm{met2jod}\Big(\big\| w_c\, \| w_b\, Q_{c,f,b} \|_{\beta_{sch}, b} \big\|_{\beta_{tch}, c}\Big)
  $$

  The official JOD is $\mathrm{met2jod}(\| \cdot \|_{\beta_t, f})$ of the same per-frame values: a normalized $L_{\beta_t}$ mean over frames ($\beta_t = 2$). $\mathrm{JOD}_{eq}(f)$ is therefore the JOD that CVVDP would report for a video whose every frame had frame $f$'s distortion, because the normalized mean of a constant is that constant. Upstream's distogram export applies `met2jod` to per-frame values in the same way. The frames are not evaluated independently, and the temporal filter state is the official full-video state. The adapter checks two parities before QCS is reported: upstream `do_pooling_and_jods` applied to the exported tensor must reproduce the official JOD, and the PixVQA per-frame pooling, re-pooled over frames, must reproduce upstream. Otherwise the line is `value:unavailable` with a reason.

  ```text
  [ CVVDP-QCS value:74.247569 range:[0,10] frames:120 basis:per-frame-jod-equivalent ]
  ```

  This is an **extension**: CVVDP was calibrated against video-level judgments, not per-frame judgments. Because the pooling over frames is an L2 mean of distortion before the `met2jod` compression, the mean of $\mathrm{JOD}_{eq}$ is close to the pooled JOD but not equal to it. `--qcs-range cvvdp=lower:upper` changes the interval. Reasons for `unavailable`:

  | Reason | Meaning |
  |---|---|
  | `backend-failed` | The CVVDP backend did not complete |
  | `no-frame-sequence` | The backend reported no frame sequence |
  | `pooled-parity-mismatch` | The exported tensor or the PixVQA pooling did not reproduce the official JOD |
  | `single-frame` | One frame: CVVDP uses its image path and there is no temporal sequence |
  | `frame-count-mismatch` | The sequence length differs from the evaluated frame count |
  | `no-finite-samples` | No finite per-frame value was available |

  See [ColorVideoVDP](COLORVIDEOVDP.md#per-frame-qcs).
- **`cvqa-hdr-nr`: sampled frames, default [0, 10].** `--qcs` normalizes each calibrated sample into that band with the same integral as the native metrics. Override it with `--qcs-range cvqa-hdr-nr=lower:upper`. The adapter captures the official model's raw output for each sampled frame with a forward hook and calibrates each one with the official logistic. The official sampler keeps about one frame per second, so the result describes consistency across sampled frames, not every frame:

  ```text
  [ CVQA-HDR-NR-QCS value:... range:[0,10] samples:2 basis:sampled-frames ]
  ```

  The pooled `quality_score` is unchanged: it is still the logistic of the mean raw score. When QCS cannot be computed, the line is `value:unavailable` with a reason:

  | Reason | Meaning |
  |---|---|
  | `backend-failed` | The NR backend did not complete |
  | `pooled-parity-mismatch` | The hooked samples did not reproduce the official pooled raw score |
  | `padded-samples` | OpenCV decoded fewer frames than reported, and upstream padded the sample buffer by repeating a frame |
  | `sample-count-mismatch` | The sample count differs from the sampling plan |
  | `no-finite-samples` | No finite calibrated sample was available |

  Per-sample calibration extends the published model, which was trained and validated as a video-level predictor. It is not an officially validated per-frame quality predictor. See [CompressedVQA-HDR NR](COMPRESSEDVQA_HDR.md#sampled-frame-qcs).

## Reference implementation

The separate [Quality Consistency Score research repository](https://github.com/pixop/Quality-Consistency-Score) provides a Python reference implementation, sample score traces, and a demonstration notebook. The `--qcs` option described above is the integration in PixVQA.

