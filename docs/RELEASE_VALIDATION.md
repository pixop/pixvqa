
# PixVQA Release Validation

PixVQA provides automated release qualification through Make targets.

These commands build and test PixVQA but do not create Git commits,
move tags, publish artifacts, or modify existing releases.

## Commands

### CPU qualification

Run:

    make release-check-cpu

Builds the CPU and FFmpeg software-decoder variants and executes
the full QA suite.

### Complete qualification

Run:

    make release-check

Executes eleven validation stages:

1. CPU build
2. FFmpeg software-decoder build
3. Full CUDA/NVDEC build
4. Full QA suite
5. GPU release qualification
6. CUDA numerical regression
7. CUDA PSNR/SSIM regression
8. CUDA PU21 regression
9. CUDA XPSNR WSSE regression
10. UHD XPSNR fixture preparation
11. UHD XPSNR concurrent-jobs regression

The complete workflow requires a working NVIDIA/CUDA environment
and the dependencies required by the existing build and QA scripts.

## Reports

All reports are stored under:

    reports/release/

The consolidated JSON report is:

    reports/release/summary.json

Each stage produces an individual log file.

The QA runner also generates structured results under:

    reports/release/qa/

The consolidated report includes:

- Git commit and execution profile
- Stage statuses and return codes
- Stage durations and log paths
- QA test counts and coverage gaps
- GPU qualification counts
- Final release-verification verdict

## Release Verdicts

PASS:
All required stages completed successfully, without mandatory
QA or GPU checks being skipped.

FAIL:
A required stage failed, an expected stage was not completed,
or report processing encountered an error.

INCOMPLETE:
The stages completed successfully, but mandatory QA or GPU
coverage was incomplete.

Informational QA coverage gaps may still be reported when
the overall verdict is PASS.

## UHD XPSNR Fixtures

The UHD regression uses four YUV420p10le sequences:

- Resolution: 3840 x 2160
- Frames: 48 per sequence
- Format: planar 10-bit YUV420

Fixture generator:

    scripts/prepare_xpsnr_fixtures.py

Default cache:

    /tmp/pixvqa_p4/

The generator validates:

- Expected file sizes
- SHA-256 checksums
- Declared generator configuration signature

Valid fixtures are reused. Missing or invalid fixtures are
regenerated, and the manifest is updated.

The complete cache occupies approximately 4.78 GB.

The generator signature does not pin the FFmpeg binary version
or guarantee byte-identical output across different environments.

## Release Policy

Successful qualification is a prerequisite for publishing
a release, not an automatic publication step.

Review the consolidated report and remaining coverage gaps
before creating a new release tag.

Existing published Git tags must not be moved.
