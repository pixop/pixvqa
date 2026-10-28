# Independent Linux installation validation (release gate)

This procedure validates two **independent** workflows: the interactive installer (`install_pixvqa.sh`) and the manual instructions (`docs/INSTALL.md`). A successful rerun in an already populated WSL2 distribution is a **resume/regression test**, not a clean installation.

## Preserve the existing working installation

Do **not** delete `$HOME/pixvqa`, `$HOME/pixvqa-deps`, `$HOME/dav1d-1.5.1`, `$HOME/ffmpeg-n8.1.3`, the `deps/` or `build/` trees, or any Git checkout in the working environment. Never run `git clean -fdx`, `git reset --hard`, or force-checkout user source files. Take a reproducible repository commit/snapshot for both new test instances. The test instances must be isolated from the existing home directories and must not share mounted build/dependency directories.

## Test systems

Recommended: two fresh Ubuntu 22.04 or Ubuntu 24.04 virtual machines, or two independent WSL2 distributions with **separate Linux filesystems and home directories**, one named AUTO and one named MANUAL. For GPU validation, provision drivers/CUDA 12.8 in each instance following NVIDIA's WSL/Linux instructions. WSL2 reuses the Windows NVIDIA driver but needs a Linux CUDA toolkit for `nvcc`. Record OS version, GPU model, `nvidia-smi`, compiler, CUDA toolkit and project commit/snapshot. Do not reuse an old `$HOME/pixvqa-deps` prefix between runs.

An alternative clean-room CPU-only test can use Ubuntu containers/VMs (no GPU); such a run does not validate NVDEC or CUDA. `/mnt/d` may contain an existing repository, but the tests should clone/copy source into the **new Linux filesystem**, not reuse generated `build/` or `deps/` from that checkout. Consider the project under test immutable during each test.

## Before either run

1. Review the snapshot for approved publication. Ensure `tests/fixtures/vvc/bbb_360p.vvc` is **not** included because redistribution permission is unverified. Also exclude root-level duplicate `libvmaf_flags.sh`, installation logs and generated files.
2. Confirm `bash -n` succeeds for `install_pixvqa.sh`, `build.sh`, `build_cuda.sh`, `build_decode.sh`, `scripts/build_libvmaf.sh`, `scripts/build_zimg.sh`, `scripts/deps_layout.sh`, `scripts/libvmaf_flags.sh`, `scripts/install.sh`, `scripts/verify_installed_pixvqa.sh`, `scripts/finish_install.sh`, `scripts/install_choices.sh`, `scripts/ci/verify_install.sh`, `scripts/ci/prepare_vvc_fixture.sh`, `scripts/test_deps_layout.sh` and `scripts/test_install_layout.sh`. Run `bash scripts/test_deps_layout.sh` and `bash scripts/test_install_layout.sh` before either installation. The first checks legacy and isolated dependency paths. The second installs each available build into a temporary prefix and checks a clean-shell `./bin/pixvqa`. Neither installs packages or touches `$HOME/dav1d-1.5.1`.
3. Verify `git status --short` in the snapshot and record file checksums/commit ID. A clean installation tests the exact same snapshot under AUTO and MANUAL.

## AUTO instance

Start with a new distribution/VM and a fresh checkout. The system may already have the NVIDIA driver and CUDA toolkit when testing GPU installation; document these explicitly. In the checkout root:

On a machine that already has `$HOME/dav1d-1.5.1` or `$HOME/ffmpeg-n8.1.3`, set `PIXVQA_DEPS_ROOT` so the guided installer does not reuse or overwrite those prefixes. `apt` packages, the NVIDIA driver, and the CUDA toolkit stay outside that directory. Omitting the variable preserves the legacy home paths.

```bash
set -o pipefail
PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh 2>&1 | tee auto_install.log
```

For the full NVIDIA path answer `y` to dav1d and `y` to NVIDIA, or export `PIXVQA_ENABLE_DAV1D=1` and `PIXVQA_ENABLE_NVIDIA=1` before starting the installer. `0` selects no. Unset variables still prompt. There is no `--with-dav1d` flag. Assert that the installed executable works from a **new shell**:

The installer installs into the checkout that contains `install_pixvqa.sh`. A `PREFIX` inherited from another PixVQA checkout is ignored. A `PREFIX` that is not a PixVQA checkout is honored. The default installed program is `./bin/pixvqa`. Its smoke QA runs that executable with `LD_LIBRARY_PATH` unset, so private libraries must resolve through RPATH. The apt set includes `perl`, `autoconf`, `automake`, `libtool`, and `m4` from the same Ubuntu release. The final NVDEC link requires `/usr/bin/g++`. Set `PIXVQA_KEEP_BUILD=1` before the installer to retain `build/`.

```bash
env -u LD_LIBRARY_PATH ./bin/pixvqa --version
env -u LD_LIBRARY_PATH ldd ./bin/pixvqa | grep 'not found' && exit 1 || true
env -u LD_LIBRARY_PATH python3 scripts/run_qa.py --profile smoke
```

Check the actual `ldd` output if the command is inconclusive. The full build must print `decode: software nvdec` and `libvmaf: enabled` during compilation. To validate VMAF numerically, run a VMAF case with an installed model, not simply `--help`.

## MANUAL instance

Start with a **second** fresh distribution/VM and the **same source snapshot**, but no PixVQA-specific dependencies. Follow `docs/INSTALL.md` from Section 1 through Section 8 exactly, without running `install_pixvqa.sh` or copying AUTO's generated dependencies. Section 0 sets `PIXVQA_MANUAL_ROOT` to `$PIXVQA_SRC/deps/manual` unless you assign another checkout-local directory. Do not point it at AUTO's `PIXVQA_DEPS_ROOT`. Set `PIXVQA_CHECKOUT` to the new checkout root in Section 0. For the NVIDIA variant, select NVDEC configuration before building FFmpeg, discover `dav1d.pc`, and ensure `CUDART_LIBDIR`, `LIBRARY_PATH`, and `PIXVQA_FULL_LINK_CXX=/usr/bin/g++` are set as documented. Section 7 installs the same `bin/pixvqa` layout as the guided installer and checks it with `LD_LIBRARY_PATH` unset.

Capture `manual_install.log`, `env -u LD_LIBRARY_PATH ./bin/pixvqa --version`, `ldd` and smoke QA result. Neither run may rely on the other instance's compiled artifacts.

## Hardware and media QA (each GPU instance)

The hardware profile includes a pre-encoded third-party VVC fixture, which is **not** redistributed with the public sources until rights are resolved. Stage it separately as described in `tests/fixtures/vvc/README.md`; verify its SHA-256 before the run. Also supply a modern external FFmpeg with native VVC decoding (the FFmpeg n8.1.3 private decode-library build does not install an FFmpeg CLI and has encoders disabled). Then:

```bash
export PIXVQA_VVC_FFMPEG=/absolute/path/to/modern/ffmpeg
export PIXVQA_CHECK_GPU=1
python3 scripts/run_qa.py --profile hardware
```

Expected release gate on supported GPU: smoke `PASS 16 FAIL 0 SKIP 0`; hardware `PASS 3 FAIL 0 SKIP 0`. Run `--profile full` separately when its VMAF CLI and VVdeC prerequisites are available; log every skipped case. A hardware PASS with a locally staged fixture is not proof of a self-contained public checkout.

## Matrix and release verdict

| Workflow | dav1d | NVIDIA | Status | Purpose |
|---|---|---|---|---|
| Automatic fresh | Yes | Yes | NOT RUN | Full GPU/NVDEC installation |
| Manual fresh | Yes | Yes | NOT RUN | Independent full GPU/NVDEC installation |
| Automatic fresh | Yes | No | NOT RUN | Software-decoding-only branch |
| Automatic fresh | No | No | NOT RUN | Raw-YUV-only branch |
| Automatic | No | Yes | Unsupported/rejected | Intentional installer behavior |

Mark **PASS only after the specific fresh-environment run**, preserving logs and QA JSON as evidence. The previously successful RTX 4070 run was on a reused installation, so it is not a substitute for either first two rows. Validation on Ubuntu 22.04 does not automatically establish Ubuntu 24.04 compatibility or other GPU architectures.
