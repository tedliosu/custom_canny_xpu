# Historical engineering notes

These notes consolidate useful observations from prototypes and experiments
that predate integration into this repository. They preserve engineering
context, not reproducible benchmark results or current support guarantees.
Regenerate important measurements from committed code under the README's
benchmarking contract before citing them publicly.

The prototypes evolved throughout these experiments. In particular, not every
alpha-versus-refined observation used fixed, rigorously controlled source
snapshots. Unless a specific experiment says otherwise, treat all numbers and
derived speedups here as approximate engineering observations rather than
controlled comparisons between immutable revisions.

Unless otherwise stated, the figures below were reported in working notes on
2026-09-19. Exact image, dependency, driver, compiler, runtime, warmup,
allocation, and timing boundaries matter.

## Provenance record

The current double-hysteresis helper was adapted from scikit-image's CCL-based
hysteresis implementation. The maintainer recalls consulting the then-current
scikit-image main branch rather than a pinned release, and records show the
resulting implementation in use by May 16, 2025. The exact upstream commit was
not recorded and is no longer expected to be recoverable. Attribution should
state this best-known source and uncertainty rather than assign a speculative
release or commit. Later internal scikit-image changes do not retroactively
establish the original source revision, even if the core logic remains
substantially similar.

## Architecture evolution

The historical alpha implementations use generic array expressions:

- 2D Gaussian and Sobel windows built with `sliding_window_view`;
- `tensordot` for filtering;
- a vectorized interpolation-based NMS with many full-image intermediates;
- library CCL plus label selection for hysteresis; and
- limited buffer reuse.

The refined prototypes use separable filtering, Numba loops on CPU, explicit
CuPy kernels on GPU, local per-output-pixel NMS, fewer full-image passes, and
reusable workspaces. Their important difference is dataflow and materialization,
not simply “vectorized Python versus a lower-level language.”

The alpha Gaussian kernel width is `K = 2 * ceil(4 * sigma) + 1`. Its logical
float32 window footprint is approximately `H * W * K^2 * 4` bytes. At
approximately 24 MP, that is about 16.2 GB for `sigma=1.5` (`K=13`), 27.7 GB
for `sigma=2.0` (`K=17`), 42.3 GB for `sigma=2.5` (`K=21`), and 60 GB for
`sigma=3.0` (`K=25`). The strided view does not allocate that complete tensor,
but a generic contraction can involve packing, temporary storage, or workspace
while other full-image arrays remain live. Alpha NMS also materializes many
image-sized coordinate, gather, interpolation, and mask arrays.

The observed result was that the alpha GPU prototype approached OOM near the
25 MP class at approximately `sigma=1.5` on an RTX 5000 Ada with 32 GB VRAM.
It is not a universal pixel threshold: the dominant exposure scales with
pixels times Gaussian kernel area, so increasing `sigma` can cause OOM at the
same or smaller dimensions. The refined separable approach does not share this
pathological scaling. See `alpha-memory-scaling.md` for the source-inspected
parameter inventory and evidence classifications.

## Correctness observations

Historical disagreement with scikit-image varied with image, configuration,
dtype, and revision:

- about 2.485% on one 1920x1280 natural image;
- about 3.687% on one roughly 122.99 MP natural forest/path image; and
- about 0.750–0.751% in newer roughly 23.23 MP alpha/refined comparisons.

In the newer comparison, alpha CPU and alpha GPU differed by only a few pixels,
and refined GPU versus alpha GPU differed by about four pixels out of roughly
23.23 million. These observations support the claim that architecture and
dataflow changed more than the high-level result, but they do not establish a
general correctness tolerance or exact semantic equivalence.

## CPU observations

On a Ryzen 5700X-era system and a roughly 23.23 MP natural image, representative
warm refined CPU results were:

- with the optional cc3d nogil binding: roughly 0.233–0.252 s for grayscale
  plus Canny and 0.194–0.214 s for Canny-only;
- with default SciPy labeling: roughly 0.248–0.262 s total and 0.208–0.224 s
  for Canny-only;
- common stage ranges of roughly 43–51 ms blur, 47–55 ms gradients, 27–31 ms
  NMS, and 75–79 ms hysteresis with the optional binding; and
- roughly 0.57–0.58 s for the scikit-image Canny reference.

A more recent run on `piermanuele-sberni-unsplash_convd.png` (3938x5899,
23,230,262 pixels) reported Canny-only trials of about 203.491 ms and 197.835
ms. The corresponding blur, gradient, NMS, and hysteresis times were about
45.6–45.8 ms, 48.5–54.5 ms, 27.2–28.1 ms, and 75.7–76.0 ms. Hysteresis was
therefore roughly 37–38% of CPU time, and disagreement with scikit-image was
about 0.751%. These figures remain approximate historical context.

The result was therefore in the 100+ MPix/s class for Canny-only, but it was not
a claim about maximum possible CPU performance.

For a roughly 122.99 MP natural image on the same CPU, reported warm results
were approximately:

- optional cc3d binding: 1.35–1.37 s total, 1.15–1.18 s Canny-only, and
  0.563–0.565 s hysteresis;
- default SciPy labeling: 1.44–1.46 s total, 1.24–1.27 s Canny-only, and
  0.651–0.654 s hysteresis.

The alpha CPU implementation took roughly 6.0 s for Canny on the 23.23 MP
image, versus roughly 0.20–0.22 s for the refined prototype. The approximate
25–30x change should be described as an architectural improvement involving
data movement, temporary arrays, generic formulations, and compilation—not
merely as “Numba is faster.”

### Loop organization

A roughly 123 MP proxy experiment illustrated the cost of flattened indexing:

- serial flat loop with division: about 921 ms;
- serial flat loop avoiding division: about 434 ms;
- raveled flat loop with division: about 645 ms;
- raveled flat loop avoiding division: about 299 ms;
- parallel flattened `prange`: about 82 ms; and
- row-oriented parallel `prange(height)`: about 39 ms.

The durable lesson is to avoid unnecessary integer division in inner loops and
prefer row-oriented decomposition for conventional image shapes. Unusual images
with very few, extremely long rows may behave differently.

### Operator-class proxy benchmark

An external compact proxy benchmark at approximately the same 23.23 MP shape
reported recent medians of about 3.985 ms for NumPy copy, 6.931 ms for a Numba
row-parallel copy, 68.973 ms for SciPy `ndimage.label`, and 13.648 ms for Numba
NMS-lite. Corresponding GPU medians were about 0.394 ms for a CuPy elementwise
copy, 0.716 ms for CuPy NMS-lite, and 2.418 ms for CuPyX labeling. The resulting
ratios were approximately 10.1x, 17.6x, 19.1x, and 28.5x respectively.

The suite samples streaming, regular local-neighborhood, and irregular
connectivity workload classes; it does not predict full Canny exactly. If CPU
stage-time fractions are used, a weighted harmonic combination is appropriate:

```text
S_total ~= 1 / sum(f_i / S_i)
```

Using an approximately 0.63 regular/local CPU share, 0.37 connectivity share,
19.1x NMS-lite ratio, and 28.5x label ratio gives a coarse expectation near
22x. The optimized operator was closer to the 30x class because separability,
materialization avoidance, heavier specialized NMS, and custom hysteresis are
not represented by the proxy.

The proxy's existing NMS-lite payload comment undercounts source-level logical
accesses. Its kernel reads center, left, right, up, down, `gx`, and `gy`, then
writes one float32 result: seven reads plus one write, or 32 bytes/pixel rather
than 16. This correction does not change timings or CPU/GPU ratios. Even the
corrected source-level payload is not literal DRAM traffic because caches,
coalescing, and reuse affect physical transfers. The proxy source remains an
external context artifact rather than repository code.

### Allocation and first touch

One approximately 0.492 GB float32 allocation experiment with 16 threads
reported about 27.5 ms for touching a reused/backed buffer versus about 44.5 ms
for `np.empty` plus first touch under default glibc behavior. With mimalloc
configured to retain backed pages, the nominally fresh case approached reused
buffer behavior. The interpretation was page reuse, not magically increased
page-mapping bandwidth. Once the operator became heavily preallocated, page
faults no longer explained most warm steady-state time.

## CCL and cc3d observations

The custom Cython experiment releases the GIL around the cc3d C++ call and
accepts a caller-provided output buffer. For 2D 8-connectivity, the safe label
bound used by that experiment is:

```text
ceil(height / 2) * ceil(width / 2) + 1
```

The simpler `(height * width) // 4 + 1` expression is unsafe for odd
dimensions.

In one 4320x7680, batch-32, 16-worker labeling experiment:

- SciPy with reusable output: about 0.685 s per batch / 1550 MPix/s;
- SciPy with per-call allocation: about 0.819 s / 1297 MPix/s;
- public cc3d: about 3.867 s / 275 MPix/s; and
- custom cc3d nogil plus reusable output: about 0.802 s / 1324 MPix/s.

Releasing the GIL fixed major thread serialization, but did not make cc3d
universally faster than SciPy. SciPy with per-worker reusable output remains a
strong default direction for CPU frame batching. The custom binding can remain
an optional single-image or experimental path.

cc3d is LGPL-licensed. Keep the custom binding/build logic separate, require an
external source checkout, and do not vendor or automatically clone cc3d.

The binding layer and much of its build/compilation scaffolding were
substantially ChatGPT-assisted. The maintainer identified the public API's
GIL/threading limitation, set the API and preallocation goals, benchmarked
SciPy and public cc3d, validated single-frame and batched behavior, and decided
the integration and licensing boundaries. The accurate summary is that
ChatGPT substantially assisted implementation while the maintainer directed,
tested, benchmarked, validated, and integrated it.

### Why deeper CCL work stops here

YACCLAB and *Parallel Light Speed Labeling: An Efficient Connected Component
Labeling Algorithm for Multi-Core Processors* reinforce that CCL optimization
is a substantial field. Memory accesses are first-class costs; run-based
methods can reduce temporary labels; serious multicore designs may require
strip partitioning, label-range management, and specialized boundary merging;
and some application-specific algorithms can accumulate component properties
without producing a final dense label image.

Canny hysteresis ultimately asks whether each weak-edge component contains a
strong pixel, so a specialized design might reduce final label
materialization. Implementing that idea would require major semantic and
correctness work. Applying a deep run-based or tiled redesign to only the CPU
would also distort the comparison; a meaningful counterpart could require
GPU-local labeling, boundary equivalence merging, warp/wavefront-specific
choices, and a new materialization strategy. That would make the project a
comparative CCL study with Canny as a wrapper. The literature therefore
supports the present stopping point rather than creating an implementation
obligation.

## CUDA GPU observations

On an RTX 5000 Ada with 32 GB VRAM and a CuPy 14.0.1/CUDA 13.0-era runtime
environment, the roughly 23.23 MP alpha GPU prototype showed an unusual
repeatable Gaussian timing transition: about 170 ms in a cold-ish trial, then
about 97 ms, then about 325 ms in later trials. Total later device time was
roughly 419–420 ms and Canny-only time roughly 414–415 ms. The exact reason was
not established. Plausible profiling directions included contraction packing
or workspace behavior, allocator state below CuPy's ordinary memory pool,
backend-library behavior, and clock/power behavior. Do not assert a cause.

Representative warm refined results on the same image were:

- about 7.7–9.0 ms total device work;
- about 6.2–7.4 ms Canny-only;
- about 1.5–1.7 ms reorganized grayscale/cast;
- about 1.1–1.6 ms blur;
- about 2.3–2.7 ms gradients;
- about 0.70–0.72 ms NMS; and
- commonly about 2.0–2.5 ms hysteresis.

Recent trials on the named 23.23 MP image placed custom Canny around 6.2–7.3
ms, with blur around 1.1–1.2 ms, gradients 2.3–2.6 ms, NMS 0.69–0.73 ms, and
hysteresis 2.0–2.9 ms. Compared with the recent CPU observations, the stage
ratios were approximately 40x blur, 20–23x gradients, 40x NMS, 30–35x
hysteresis, and roughly 30x for Canny overall. Run-to-run variation makes these
classes, not exact constants.

The roughly 40x NMS ratio is not a claim about raw FP32 throughput. Both custom
backends quickly enter compiled per-pixel code, but real interpolation NMS does
more coordinate arithmetic, conversion, clamping, gathers, and bilinear work
than NMS-lite. SIMT lets that scalar-looking work be expressed as independent
pixel threads while hardware groups them into vector-like warps or wavefronts;
it is related to, but not programmer-equivalent to, explicit SIMD. The observed
throughput also depends on execution lanes, registers, bandwidth, caches,
occupancy, and latency hiding. NVIDIA-style independent thread scheduling is
not central to this shallowly divergent NMS, though divergence behavior matters
more for variable root traversal, retry loops, and atomics in CCL.
The maintainer also observed 100+ GPix/s-class NMS throughput on MI300X; that
result reflects the full hardware and execution shape, not SIMT alone.

The approximate observed stable-regime improvement was about 45–50x for the
whole device operator and 55–61x for Canny-only. NMS exceeded 100x and Gaussian
blur reached roughly 200–300x in the alpha slow state, but those stage-specific
figures must not be presented as whole-operator speedups. Hysteresis improved
only about 1.25x in one comparison because alpha already used a GPU-native CCL
decomposition.

On a roughly 122.99 MP natural image, prior RTX 5000 Ada observations included
roughly 35 ms custom Canny-only, 44.8 ms for a broader device boundary, and
12.6 ms for hysteresis. One profile placed CCL kernels around 8.5 ms, with
roughly half in the union-find kernel and one kernel showing more than 60%
theoretical excess L2 sectors. This supports treating GPU CCL as irregular,
graph-like work inside a pixel-oriented API.

## ROCm and portability observations

### Separate ROCm bootstrap compatibility experiment

A CuPy runtime-compiled `atomicCAS` compatibility case for signed 64-bit
values was verified on both CUDA/NVRTC and ROCm/HIPRTC. The compact solution
selected an unsigned atomic representation for `long long` using
`cuda::std` type traits on CUDA and version-sensitive `__hip_internal` traits
on ROCm. This was a deliberately small primitive-level compatibility smoke for
the separate ROCm bootstrap project. Its choice of traits and its preference
for a minimal shim do not set the implementation policy for this Canny
repository.

### Refined CCL vendoring decision

The planned refined CCL kernels derive from CuPy 14.1.x-era upstream source.
For this repository, the selected approach is to vendor the required templated
traits together with those kernels rather than depend on `cuda::std` and
version-sensitive HIP-internal traits. This keeps the derived implementation
coherent with its upstream source and makes the relevant CUDA/ROCm behavior
explicit. When the code is integrated, record the exact CuPy revision and add
the applicable attribution and license text to `THIRD_PARTY_NOTICES.md`.

This is a narrow project-specific provenance and portability decision. It does
not reverse the smaller-shim decision in the ROCm bootstrap repository and
should not be exported there implicitly.

One AMD DevCloud MI300X VF observation used a ROCm 7.0.2-era stack, Python
3.12.3, CuPy 13.5.1 for ROCm, and AMD-packaged hipCIM 25.10.0. For a roughly
122.99 MP workload it reported about 28.4 ms total device work, 20.9 ms
Canny-only, and 13.5–13.7 ms hysteresis. The hipCIM-style reference was about
31.2 ms Canny-only. This one environment suggested roughly a 1.5x custom
advantage, not a general MI300X result.

Do not compare those figures directly with Ada as though hardware were the only
variable. CuPy generations, CUDA versus ROCm compilers, warp32 versus wave64,
cloud/VF state, package versions, driver/runtime state, and unequal tuning all
confound the comparison.

## Reference implementation context

cuCIM became the primary serious CUDA library baseline because it accepts CuPy
arrays directly and supplies a scikit-image-style Canny operator. An informal
Python Community Discord inquiry did not identify a faster directly
CuPy-compatible alternative. This was community reconnaissance, not a formal
literature review, expert certification, or proof that cuCIM is globally
fastest.

In one roughly 122.99 MP fp32 grayscale CUDA comparison, cuCIM took about
40.1 ms and the custom implementation about 35.0 ms. The roughly 14–15%
difference is evidence that cuCIM is a strong directly interoperable baseline,
not evidence that it is poorly optimized or globally fastest.

Kornia was also evaluated as a PyTorch-native reference. Reported results
included roughly 1.96 s CPU execution at 3840x2160, roughly 693 ms GPU-only on a
122.99 MP random/synthetic case, and roughly 6.36 s GPU-only on a natural image
of the same scale. The large content sensitivity was consistent with iterative
hysteresis propagation over connected weak-edge structures. Kornia targets
different goals—PyTorch integration, differentiability, augmentation, and
batching—so these observations are workload-specific, not a universal quality
judgment.

That comparison is also a warning against treating hysteresis as a fixed
MPix/s workload. Weak-edge density, component count and sizes, chain structure,
root-equivalence behavior, strong-component count, and propagation convergence
can change at fixed dimensions. Nsight observations of root chasing, atomics,
equivalence updates, and inefficient cache/L2 behavior point in the same
direction for the custom GPU CCL. Benchmark multiple content classes and
investigate content before attributing an unusually fast or slow result solely
to hardware or an implementation change. Regular stages are generally less
content-sensitive.

OpenCV frequently outperformed the refined scientific-Python CPU prototype and
should remain mature production-native context. The project does not need to
beat it at all costs. OpenCV's OpenCL/UMat path can also be fast, but it does
not directly answer the CUDA/ROCm/CuPy question.

When timing scikit-image or cuCIM grayscale conversion, remember that
`rgb2gray(uint8).astype(float32)` may perform grayscale work in a default
floating dtype before the final cast. A clearer public-API path is
`img_as_float32` followed by `rgb2gray`.

Detailed internal stage instrumentation of third-party references is a
non-goal unless a public API already exposes a clean boundary. Whole-Canny,
grayscale-plus-Canny, device-only, transfer-inclusive, and selected public
stage measurements are sufficient. Vendoring reference source, patching
site-packages, carrying forks, or using private APIs only to obtain matching
stage breakdowns would add disproportionate maintenance and provenance cost.

## Interpreting the CPU/GPU comparison

The comparison is meaningful but not architecture-pure. Hysteresis is already
a major wall-time component on both backends, while the CPU CCL direction does
not exploit multicore parallelism as effectively as the regular stages and the
custom GPU formulation exposes much more parallel work. Part of the observed
speedup therefore reflects implementation parallelism rather than intrinsic
hardware alone.

This is not merely an unfinished obvious optimization. Eliminating the
asymmetry rigorously leads into the deeper CCL designs discussed above and can
increase provenance and semantic risk relative to the intended scikit-image
decomposition: threshold weak and strong pixels, label 8-connected components,
identify components containing a strong pixel, and retain them. The current
implementations are sufficiently optimized for a bounded engineering
comparison when this limitation is explicit; they are neither a perfectly
fair hardware experiment nor a meaningless comparison.

CPU remains valuable for deployment availability, CPU-resident applications,
small and medium workloads where launch and transfer costs matter, debugging,
simpler packaging, and systems without usable CUDA or ROCm. Large-workload GPU
advantages and remaining CPU optimization headroom are compatible facts.

## Future directions kept outside current scope

Potential later work includes explicit GPU batch dimensions and reusable
`[B, H, W]` workspaces, CPU frame pools with per-worker workspaces, and a
bounded classic-image-processing backend with selected separable filtering or
feature-extraction operations. None of these directions authorizes premature
batch APIs, an array framework, a SciPy/scikit-image replacement, or a new CCL
research project.

The operator may be used as one low-arithmetic-intensity case with mostly
regular access patterns over pixel data in a broader GPU execution-motif study
or a short ROCm presentation. Those projects remain separate. A concise
presentation should favor scikit-image, the refined CPU operator, and the
refined GPU operator, with OpenCV and cuCIM/hipCIM as optional context. The
central audience messages are “throughput, not magic,” “workload shape
matters,” and “regular access patterns of pixel data fly; irregular
connectivity remains hard.”
