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

VMAF uses one interval for every SDR and HDR model. PU21-PSNR uses the PSNR interval because its peak is 256, about 100 cd/m², which is an 8-bit-style decibel scale. PU21-SSIM and PU21-MS-SSIM use the SSIM interval because they are similarities on that same peak.

## Usage

Enable QCS with `--qcs`. Override a default interval with `--qcs-range metric=lower:upper`. `--qcs-range` requires `--qcs`. Several intervals may be comma-separated in one option, and the option may be repeated. The two forms can be combined. Naming one metric again with a different interval is an error; the same interval is unchanged. Every selected metric receives a QCS value, whether that metric runs on the CPU or the GPU.
```bash
pixvqa --metrics psnr,vmaf \
  --model model/vmaf_v1.0.16_3d0h.json \
  --qcs --qcs-range psnr=40:55,vmaf=30:100 \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```
Repeating the option is equivalent: `--qcs-range psnr=40:55 --qcs-range vmaf=30:100`.

QCS is computed independently for each selected metric. The name in `--qcs-range` is the `--metrics` name, and the bounds must satisfy lower < upper. Identical PSNR is printed `999.99`, identical XPSNR is printed `inf`, and identical PU21-PSNR is printed `999.99`. All three still count in QCS.

## Reference implementation

The separate [Quality Consistency Score research repository](https://github.com/pixop/Quality-Consistency-Score) provides a Python reference implementation, sample score traces, and a demonstration notebook. The `--qcs` option described above is the integration in PixVQA.

