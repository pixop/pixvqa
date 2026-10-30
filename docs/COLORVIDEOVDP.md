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
default. Native decoder, raw-YUV, HDR override, VMAF-model, QCS, thread, and
verbosity options are incompatible.

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

This first integration supports only files accepted and validated by the
official CVVDP CLI. Raw YUV is deliberately unsupported. Future YUV420p8/p10
support needs an official-CVVDP-compatible raw-video invocation, explicit
width/height/pixel-format/frame-rate transfer, colorimetry and range metadata,
paired-frame-count validation, and parity validation against decoded fixtures.
