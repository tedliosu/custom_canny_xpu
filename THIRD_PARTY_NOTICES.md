# Third-party notices

This file records third-party source incorporated into or adapted by this
repository. The repository's Apache License 2.0 does not replace the license
terms reproduced below for the applicable material.

## scikit-image Canny double hysteresis

`_canny_double_hysteresis.py` contains a backend-generalized adaptation of the
double-thresholding and edge-tracking logic from
`skimage/feature/_canny.py` in the scikit-image project.

The maintainer recalls consulting the then-current scikit-image main branch
rather than a pinned release. The adaptation was in use by May 16, 2025, but
the exact upstream commit was not recorded and is no longer expected to be
recoverable. This notice therefore records the best available provenance
without assigning a speculative release or commit.

Upstream project: <https://github.com/scikit-image/scikit-image>

Upstream license record:
<https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt>

Copyright notices associated with scikit-image and
`skimage/feature/_canny.py`:

```text
Copyright 2009-2022 the scikit-image team
Copyright 2003-2009 Massachusetts Institute of Technology
Copyright 2009-2011 Broad Institute
Copyright 2003 Lee Kamentsky
Copyright 2003-2005 Peter J. Verveer
```

The applicable BSD 3-Clause license follows:

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Future third-party integrations

The planned refined CCL implementation is expected to derive from CuPy
14.1.x-era upstream kernels and to vendor the required templated traits. That
code is not yet present in this repository, so no CuPy source notice is claimed
here yet. The integration change must identify the exact upstream revision and
extend this file with the applicable attribution and license terms.
