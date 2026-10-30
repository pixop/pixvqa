# Optional ColorVideoVDP (CVVDP)

CVVDP is an optional external backend. It is not linked into PixVQA and is not
installed by PixVQA's build or install scripts. The native executable launches
the isolated Python adapter with an explicit argument vector; it never builds a
shell command.

Install the adapter package from this checkout into a separate Python
environment, then install the official ColorVideoVDP package at the pinned
revision in `python/pixvqa_cvvdp/__init__.py`. The validated prototype used
Python 3.12, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and CVVDP 0.5.7
from commit `166f5ae95ecbd841ae727984cd237de213875399`. PixVQA does not install
these packages, download model files, or vendor the upstream source tree.

Install the dependency-free adapter into that isolated environment:

```bash
/opt/cvvdp/bin/python -m pip install ./python
```

Then ensure `ffprobe` and the official `cvvdp` executable are on that
environment's `PATH`.

```bash
pixvqa --metrics cvvdp \
  -r reference.mkv \
  -t distorted.mkv \
  --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq \
  --cvvdp-python /opt/cvvdp/bin/python \
  --cvvdp-output-json result.cvvdp.json
```

`--cvvdp-python` overrides `PIXVQA_CVVDP_PYTHON`; otherwise PixVQA uses
`python3`. `--cvvdp-bin` optionally points the adapter at a specific official
CVVDP executable. `--frames` becomes CVVDP's `--nframes`.

CVVDP accepts only `--metrics cvvdp`, never a mixed native/CVVDP metric list.
It accepts `--metric-device cpu` or `gpu`; `auto` selects the adapter's CUDA
default. Native decoder, HDR override, VMAF-model, QCS, thread, and verbosity
options are incompatible. `-s`, `-f`, and `--fps` are accepted for raw input.

If `--cvvdp-output-json` is set, that exact file and its sibling CSV are
preserved on both success and backend failure. Without it, PixVQA creates a
private temporary directory for the adapter's JSON and CSV and removes that
directory after the child exits, whether it succeeds or fails. The success JSON
then contains `"output_json": null`; specify `--cvvdp-output-json` when the
full reproducibility record must be retained.

Interpretation and display are deliberately explicit. `pq` requires both
inputs to be tagged SMPTE 2084 and BT.2020 and currently permits only
`standard_hdr_pq`. `sdr` requires explicit BT.709 or sRGB tags and permits
`standard_fhd` or `standard_4k`. The adapter does not infer HDR from bit depth,
resolution, filenames, or metadata defaults, and it does not modify the
official CVVDP algorithm.

On success, stdout is one JSON object containing `metric`, `jod`,
`interpretation`, `display`, `device`, elapsed time, and the record path. The
record contains the command argument vector, pin identifiers, `ffprobe`
metadata, captured backend output, and the result CSV path. It also records
the installed CVVDP, PyTorch, and torchvision versions. A VCS commit is marked
verified only when the installed package exposes a matching commit through
Python package metadata; PixVQA does not claim that an expected pin is the
installed commit. JOD is not a QCS,
VMAF, PSNR, or SSIM score; larger JOD means a smaller visible difference, and
an identical pair is near 10 JOD.

## Raw YUV420p10le PQ

This phase accepts one raw layout: planar YUV 4:2:0, stored as little-endian
`uint16` with the 10-bit code in the low bits (`0..1023`). The samples are
limited-range BT.2020 non-constant-luminance YCbCr with a PQ interpretation.
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
