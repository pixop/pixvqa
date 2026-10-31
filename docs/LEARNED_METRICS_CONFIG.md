# Learned-metric configuration

`cvvdp` and `cvqa-hdr-nr` need machine-specific paths: a Python interpreter,
the adapter packages, `ffmpeg`/`ffprobe`, the CompressedVQA-HDR checkout, and
its model assets. A small TOML file stores them so that commands stay short and
no workstation path is compiled into PixVQA.

```bash
cp config/learned_metrics.example.toml config/learned_metrics.local.toml
# edit the paths, then:
./bin/pixvqa --preset hdr --qcs -r reference.mp4 -t distorted.mp4
```

`config/*.local.toml` is ignored by Git. Only the portable
[`config/learned_metrics.example.toml`](../config/learned_metrics.example.toml)
is tracked.

## Which file is read

The config is read only when `cvvdp` or `cvqa-hdr-nr` is selected. The first
match wins:

1. `--learned-config PATH`. The file must exist and parse.
2. `PIXVQA_LEARNED_CONFIG=PATH`. The file must exist and parse.
   `PIXVQA_LEARNED_CONFIG=none` disables config lookup; the test scripts use it.
3. `config/learned_metrics.local.toml` in the checkout that holds the
   executable: `<exe dir>/../config/` (for `bin/pixvqa`) or
   `<exe dir>/../../config/` (for `build/<kind>/pixvqa`). The executable is
   resolved through `/proc/self/exe`, so the working directory does not matter.
   A missing file here is not an error.

When a file is read, PixVQA prints `pixvqa: info: learned-metric config <path>`
on stderr.

## Keys

| Section and key | Meaning |
|---|---|
| `[learned] python` | Interpreter for both adapters |
| `[learned] adapter_path` | Directory holding `pixvqa_cvvdp` and `pixvqa_cvqahdr`. Appended to `PYTHONPATH` |
| `[learned] path` | Directory prepended to `PATH` for the backends, for example the one with `ffmpeg` and `ffprobe` |
| `[cvvdp] python` | Interpreter for `cvvdp` only |
| `[cvvdp] executable` | Official `cvvdp` entry point. Default: beside the interpreter, then `PATH` |
| `[cvqa-hdr-nr] python` | Interpreter for `cvqa-hdr-nr` only |
| `[cvqa-hdr-nr] repository` | Upstream CompressedVQA-HDR checkout |
| `[cvqa-hdr-nr] checkpoint` | `NR_HDR_VQA.pth` |
| `[cvqa-hdr-nr] profile` | `NR_HDR_VQA.npy` |

The file is a strict TOML subset: `#` comments, the three sections above, and
`key = "string"` or `key = 'string'` lines. An unknown section, an unknown or
duplicate key, or an unquoted value is an error that names the file and line;
PixVQA exits 2 before any metric runs.

Path values:

- `~/` expands to `$HOME`.
- A relative value that contains `/` is relative to the config file's directory.
  `adapter_path = "../python"` therefore points at the checkout's `python/`.
- A value without `/`, such as `python3`, is a command looked up on `PATH`.

## Precedence

For every value, from highest to lowest:

1. The command-line option (`--cvvdp-python`, `--cvvdp-bin`,
   `--cvqahdr-python`, `--cvqahdr-repo`, `--cvqahdr-checkpoint`,
   `--cvqahdr-profile`).
2. For interpreters only, `PIXVQA_CVVDP_PYTHON` or `PIXVQA_CVQAHDR_PYTHON`.
3. The metric's own section (`[cvvdp]`, `[cvqa-hdr-nr]`).
4. `[learned] python`, for interpreters.
5. The built-in default: `python3` for interpreters; for `cvvdp` the entry
   point beside the interpreter, then `PATH`. The NR assets have no default and
   remain required.

`PYTHONPATH` and `PATH` already set in the environment keep priority over
`adapter_path`. `[learned] path` is prepended, so its `ffmpeg` is found before
a system one.

The config does not change metric behaviour. It supplies paths only:
interpretation, display, devices, and QCS intervals remain options.
