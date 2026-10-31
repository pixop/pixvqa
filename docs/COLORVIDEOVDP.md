# Optional ColorVideoVDP (CVVDP)

CVVDP is an optional external backend. It is not linked into PixVQA and is not
installed by PixVQA's build or install scripts. The native executable launches
the isolated Python adapter with an explicit argument vector; it never builds a
shell command.

The recommended runtime is the shared learned-metric environment, which also
runs `cvqa-hdr-nr`. It pins Python 3.12, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, and CVVDP 0.5.7 from commit
`166f5ae95ecbd841ae727984cd237de213875399`. See
[Learned-metric environment](LEARNED_ENVIRONMENT.md). PixVQA's build does not
install these packages, download model files, or vendor the upstream source
tree.

The adapter is dependency-free. Either put `python/` on `PYTHONPATH` through
the [learned-metric config](LEARNED_METRICS_CONFIG.md) (`adapter_path`), or
install it into the environment with `python -m pip install ./python`.
CVVDP decodes compressed inputs with `ffmpeg` and `ffprobe`, which must be on
`PATH`. The config's `[learned] path` can prepend their directory. The adapter
uses `--cvvdp-bin` or `[cvvdp] executable` if set. Otherwise it uses the `cvvdp`
entry point installed beside the interpreter, and then `PATH`.

```bash
pixvqa --metrics cvvdp \
  -r reference.mkv \
  -t distorted.mkv \
  --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq \
  --cvvdp-python /opt/cvvdp/bin/python \
  --cvvdp-output-json result.cvvdp.json
```

The interpreter is chosen in this order: `--cvvdp-python`, then
`PIXVQA_CVVDP_PYTHON`, then `[cvvdp] python`, then `[learned] python` in the
config, then `python3`. With a config in place, `--cvvdp-python` can be
omitted. `--cvvdp-bin` optionally points the adapter at a specific official
CVVDP executable. `--frames` becomes CVVDP's `--nframes`.

`--cvvdp-gpu-mem GB` optionally passes CVVDP's `--gpu-mem` limit. The unit is
decimal GB (for example, `--cvvdp-gpu-mem 4` is 4,000,000,000 bytes). Upstream
uses it only on CUDA, to cap the block-processing memory budget at the smaller
of the supplied amount and its own estimate of free GPU memory after its safety
reserve. Omit it to retain upstream's default free-memory heuristic. The JSON
record retains the requested `gpu_mem_gb` (or `null` when omitted).

CVVDP can run alone or mixed with native metrics and `cvqa-hdr-nr`; see
[mixed execution](CLI_USAGE.md#mixed-native-and-learned-metrics). It runs after
the native metrics and before NR. It needs both inputs raw YUV420p10le (PQ or HLG) or both compressed. `--device cpu|gpu|auto` (alias of `--metric-device`) selects
the device. `gpu` requires CUDA and fails in preflight without it. `auto` uses CUDA when PyTorch sees
a device, else the CPU with a notice. The adapter always passes the official CLI an explicit `cpu` or
`cuda`, because the CLI itself silently runs a `cuda` request on the CPU. It then checks the CLI's
`Running on device:` log line. Pooled JOD was identical on CPU and CUDA in every validation run; see
[LEARNED_DEVICES.md](LEARNED_DEVICES.md). When no native metric is selected, the native decoder,
HDR override, VMAF-model, thread, and verbosity options are rejected. When
native metrics are selected, those options apply only to them. `-s`, `-f`, and
`--fps` are accepted for raw input.

`--preset hdr` and `--metrics pq` fill `--cvvdp-interpretation pq` and
`--cvvdp-display standard_hdr_pq` when they are omitted, and reject an explicit
`sdr` or `hlg` interpretation. `--metrics hlg` fills
`--cvvdp-interpretation hlg` and `--cvvdp-display standard_hdr_hlg`, and
rejects `standard_hdr_pq`: that display would apply the PQ EOTF to HLG codes.
`--metrics sdr` fills `--cvvdp-interpretation sdr` but requires an explicit
`--cvvdp-display` (`standard_fhd` or `standard_4k`), because the SDR display
model is not inferred from resolution.

## Per-frame QCS

With `--qcs`, CVVDP reports a QCS over the default interval [0, 10]. Override
it with `--qcs-range cvvdp=lower:upper`. The pooled JOD is the official value
and is never clamped or rescaled.

CVVDP has no official per-frame JOD. Its pooled JOD is computed from
`Q_per_ch[batch, channel, frame, band]` in `cvvdp.process_block_of_frames`,
upstream's per-frame distortion. It is already temporally filtered: the
sustained and transient channels come from temporal filters applied over the
whole sequence, with replicate padding, before spatial pooling.
`do_pooling_and_jods` then:

1. weights the channels (chromatic 1, transient 0.808) and the baseband;
2. takes an unnormalized $L_4$ norm over bands, then over channels;
3. takes a normalized $L_2$ mean over frames;
4. maps the result to JOD with `met2jod` (power law, linear below 0.1).

With `--qcs`, the adapter adds the official `--features` option to the same
`cvvdp` run. CVVDP writes `Q_per_ch` to `<test>_fmap.json` after computing the
JOD, so the run, its temporal filter state, and its JOD are unchanged. The
adapter reloads that tensor in float32 and applies steps 1, 2, and 4 to each
frame, using the parameters of a `pycvvdp.cvvdp` object built for the same
display. The per-frame result is

$$
\mathrm{JOD}_{eq}(f) = \mathrm{met2jod}(D_f), \qquad
\mathrm{JOD} = \mathrm{met2jod}\big(\|D\|_{2,\text{mean}}\big)
$$

where $D_f$ is frame $f$'s pooled distortion. $\mathrm{JOD}_{eq}(f)$ is the JOD
that CVVDP would assign to a video with frame $f$'s distortion in every frame,
because the normalized mean of a constant is that constant. The distogram that
upstream exports applies `met2jod` per frame in the same way.

Three checks guard the sequence, and the record stores each result:

| Check | Requirement |
|---|---|
| `features_parity` | Upstream `do_pooling_and_jods` on the exported tensor reproduces the CSV JOD |
| `mirror_parity` | PixVQA's per-frame pooling, re-pooled over frames, reproduces upstream |
| `pooled_parity` | Both of the above |

Without `pooled_parity`, or when the sequence length differs from the evaluated
frame count, the QCS line is `value:unavailable` with a reason. The persistent
JSON record has a `temporal` object with the method, the note, the per-frame
`distortion` and `jod_equivalent`, and the parity values. The adapter rejects
temporal resampling (`temp_resample`) and batched input, because neither
appears in the official CLI path.

Approximations and extensions:

- $\mathrm{JOD}_{eq}$ is an extension. CVVDP was calibrated on video-level
  judgments. A per-frame value describes how frame $f$'s distortion maps
  through the model, not a subjective per-frame score.
- Frames are not evaluated independently. Each frame's value depends on its
  neighbours through the official temporal filters.
- The mean of $\mathrm{JOD}_{eq}$ is not the pooled JOD, because the frame
  pooling ($L_2$ of distortion) happens before the concave `met2jod`. For
  BasketballGame 1080p crf20 against crf40, the pooled JOD is
  7.415657997131348 and the mean $\mathrm{JOD}_{eq}$ is 7.4248.
- The tensor is reloaded from JSON as float32, the dtype upstream computes in.
  float32 values survive the JSON round trip, and both parity checks require
  exact equality with the official float32 JOD. They hold exactly on the
  BasketballGame validation pair.
- One-frame input uses CVVDP's image path. QCS is then `unavailable` with
  `single-frame`.

In mixed runs PixVQA first runs `python -m pixvqa_cvvdp --check` with the same
arguments. That check validates inputs, metadata, interpretation, display, and
the CVVDP executable, and it exits 2 on failure without running the metric. It
then runs the metric with `--protocol lines`, a line-oriented stdout format
that carries `jod` at full precision. The summary line is
`[ CVVDP jod:<value> frames:<n> display:<display> interpretation:<i> ]`.
Adapter diagnostics and CVVDP's own output go to stderr.

If `--cvvdp-output-json` is set, that exact file and its sibling CSV are
preserved on both success and backend failure. Without it, PixVQA creates a
private temporary directory for the adapter's JSON and CSV and removes that
directory after the child exits, whether it succeeds or fails. The success JSON
then contains `"output_json": null`; specify `--cvvdp-output-json` when the
full reproducibility record must be retained.

Interpretation and display are deliberately explicit. `pq` requires both
inputs to be tagged SMPTE 2084 and BT.2020 and permits only
`standard_hdr_pq`. `hlg` requires both inputs to be tagged HLG
(`arib-std-b67`), BT.2020 primaries, `bt2020nc`, and limited range (`tv`),
and permits only `standard_hdr_hlg`. A PQ tag is not accepted as HLG, and an
HLG tag is not accepted as PQ. `sdr` requires explicit BT.709 or sRGB tags
and permits `standard_fhd` or `standard_4k`. The adapter does not infer the
transfer from bit depth, resolution, filenames, or metadata defaults, and it
does not modify the official CVVDP algorithm.

Upstream decodes compressed video to display-encoded YCbCr and chooses the
RGB matrix from the file's matrix (`bt2020nc` or BT.709). The EOTF comes from
the display model, not from the file: `standard_hdr_hlg` is `BT.2020-HLG` in
`display_models.json` at commit `166f5ae`, and `display_model.py` linearizes
it with `hlg2lin`. The file reader's only transfer check is a warning when a
PQ file is paired with a non-PQ display. The adapter turns that pairing, and
the reverse, into an error before CVVDP starts.

For `--metrics cvvdp` alone, stdout on success is one JSON object containing `metric`, `jod`,
`interpretation`, `display`, `device`, elapsed time, and the record path. The
record contains the command argument vector, pin identifiers, `ffprobe`
metadata, captured backend output, and the result CSV path. It also records
the installed CVVDP, PyTorch, and torchvision versions. A VCS commit is marked
verified only when the installed package exposes a matching commit through
Python package metadata; PixVQA does not claim that an expected pin is the
installed commit. JOD is not a QCS,
VMAF, PSNR, or SSIM score; larger JOD means a smaller visible difference, and
an identical pair is near 10 JOD.

## Raw YUV420p10le PQ and HLG

This phase accepts one raw layout: planar YUV 4:2:0, stored as little-endian
`uint16` with the 10-bit code in the low bits (`0..1023`). The samples are
limited-range BT.2020 non-constant-luminance YCbCr. `--cvvdp-interpretation pq`
selects the PQ EOTF and `--cvvdp-interpretation hlg` selects the HLG EOTF.
The filename token is `2020` for both: upstream uses it for the matrix, and
the display model supplies the EOTF. SDR raw YUV stays unsupported.
PixVQA does not convert them. The official `YUVReader` expands limited-range
codes itself: luma uses `(code - 64) / 876` and chroma uses `(code - 512) / 896`,
then clips. Full-range samples are therefore not represented correctly. Codes
outside the 10-bit range are rejected before ColorVideoVDP starts. Legal-range
limits (luma 64–940, chroma 64–960) are not enforced; the official reader clips
the expanded result.

```bash
pixvqa --metrics cvvdp \
  -r reference.yuv \
  -t reconstructed.yuv \
  -s 1920x1080 -f 420p:10 --fps 60 \
  --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq
```

`-s`, `-f`, and `--fps` use the normal PixVQA rules, including one shared size
and format for both inputs. `--fps` may be an integer or a rational. The
official filename stores frame rate to three decimal places, so `60000/1001`
becomes `59.94`. Both files must exist, share that geometry, contain a positive
whole number of frames, and contain at least `--frames` frames when that option
is set.

The adapter creates a private temporary directory and symlinks each original
file to a metadata-bearing name such as
`reference_1920x1080_10b_420_2020_60fps.yuv`. It does not rename, modify, or
copy the YUV. The name is checked with a strict parser that rejects missing
fields, and with `pycvvdp.video_source_yuv.decode_video_props()` when that
function can be imported. The official `cvvdp` CLI then receives those symlink
paths and selects its raw reader because the names end in `.yuv`. The symlinks
and directory are removed after success and after failure. A user-specified
`--cvvdp-output-json` file is not part of that cleanup.

SDR raw YUV is not supported. In ColorVideoVDP 0.5.7 the raw reader's non-2020
matrix is the BT.601 matrix (`1.402`, `-0.344136`, `-0.714136`, `1.772`), not
the BT.709 matrix used for compressed inputs. 8-bit, 4:2:2, 4:4:4, and
big-endian storage are rejected.
