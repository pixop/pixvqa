# VVC software-decoding regression fixture

The regression uses the **external** `bbb_360p.vvc` elementary bitstream from
https://dl.photoprism.app/samples/Formats/Video/H.266-VVC/bbb_360p.vvc

- Filename: `bbb_360p.vvc`
- SHA-256: `eb1191c46b50cb8b2b631194f74a7a9152cced29e0f7d1278d7723420e83d44d`
- Expected coded format: VVC, 480 x 360, yuv420p10le
- Frame count: determined from an independent software decode, not assumed
- License/redistribution: **not verified**. Do **not commit or distribute** this
  bitstream in a public repository until explicit redistribution rights are
  established. The code and documentation do not include its bytes.

To stage this file locally from the repository root:

```bash
mkdir -p tests/fixtures/vvc
cp /tmp/pixvqa-vvc-test/bbb_360p.vvc tests/fixtures/vvc/bbb_360p.vvc
sha256sum tests/fixtures/vvc/bbb_360p.vvc
```

Set an independent modern FFmpeg with native VVC decoding, for example:

```bash
export PIXVQA_VVC_FFMPEG="$HOME/codeccomp/encoders/nvenc/bin/ffmpeg"
export PIXVQA_CHECK_GPU=1
python3 scripts/run_qa.py --profile hardware
```

The QA script verifies SHA-256, decodes with independent FFmpeg, and compares
PixVQA's software VVC output against the independently decoded frames at PSNR
999.99. It also tests explicit GPU VVC rejection. This checks decoder
agreement, **not** an independent known-good pixel oracle or all VVC conformance.

For a truly self-contained open-source QA, replace the fixture with a small
redistributable, properly licensed stream and pin its checksum and known-good
reference YUV hash.

A missing `bbb_360p.vvc` is a **failure**, not a skip. CI does not download it.
`scripts/ci/prepare_vvc_fixture.sh` builds VVenC 1.14.0 and VVdeC 3.2.1, writes
a deterministic 176x144 10-bit source, encodes four frames with `--threads 1`,
and decodes that bitstream with VVdeC. The products land in
`reports/qa/vvc-fixture/`, which is not committed. Load them with:

```bash
bash scripts/ci/prepare_vvc_fixture.sh
set -a
source reports/qa/vvc-fixture/fixture.env
set +a
```

`fixture.env` sets `PIXVQA_VVC_FIXTURE`, `PIXVQA_VVC_SIZE` (`176x144`), and
`PIXVQA_VVC_REFERENCE`. The test compares PixVQA with that VVdeC
reconstruction. It does not pin the synthetic bitstream hash, and this path
does not replace the SHA-256 check on `bbb_360p.vvc`. If generation fails, or
neither the default fixture nor these three variables is present, the VVC case
fails. It is not skipped.
