# Project license

PixVQA does not yet have a project license. This note is not a selection and it is not a `LICENSE` file.

The comparison is between MIT and BSD-3-Clause. Either can cover only the code the copyright holder has the right to license.

## MIT

A recipient may use, modify, and distribute the covered code. The copyright and permission notice stay with the copies. There is no patent grant and no endorsement clause.

## BSD-3-Clause

The same permissions, plus a requirement that the copyright holder's name is not used to endorse a derived product without permission. There is no patent grant.

## Dependencies that stay under their own terms

- FFmpeg, when a user builds it for this program, is LGPL 2.1 or later if configured without `--enable-gpl` and without `--enable-version3`. This tree does not vendor FFmpeg.
- libdav1d is BSD-2-Clause.
- libvmaf and the model JSON files are BSD-2-Clause-Patent. See `third_party/netflix-vmaf/`.
- The PU21 formula attribution is BSD-3-Clause. See `licenses/PU21_BSD.txt`.

MIT and BSD-3-Clause both allow those notices to remain beside the project license. Neither document replaces them.

Apache-2.0 is not the comparison here. Its patent clause is an open question next to LGPL-2.1, which is the license of an FFmpeg build that does not pass `--enable-version3`.
