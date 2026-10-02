# Historical engineering notes

These notes consolidate useful observations from prototypes and experiments
that predate integration into this repository. They preserve engineering
context, not reproducible benchmark results or current support guarantees.
Regenerate important measurements from committed code under the README's
benchmarking contract before citing them publicly.

The refined CPU and GPU prototype observations in this document generally
predate removal of per-image maximum-magnitude normalization. They remain
historical evidence for those prototype generations, but they are not current
performance, disagreement, stage-fraction, or cross-backend speedup evidence
for the absolute-threshold operator. See
[`benchmark-evidence-lifecycle.md`](benchmark-evidence-lifecycle.md) for the
status definitions and selective revalidation procedure.

The prototypes evolved throughout these experiments. In particular, not every
alpha-versus-refined observation used fixed, rigorously controlled source
snapshots. Unless a specific experiment says otherwise, treat all numbers and
derived speedups here as approximate engineering observations rather than
controlled comparisons between immutable revisions.

Unless otherwise stated, the figures below were reported in working notes on
2026-09-19. Exact image, dependency, driver, compiler, runtime, warmup,
allocation, and timing boundaries matter.

Centralized third-party source and AI-assistance provenance lives in
[`provenance-and-ai-assistance.md`](provenance-and-ai-assistance.md).
Experiment-specific authorship notes remain here where they are necessary to
interpret the historical evidence.

## Architecture evolution

The historical alpha implementations use generic array expressions:

- 2D Gaussian and Sobel windows built with `sliding_window_view`;
- `tensordot` for filtering;
- a vectorized interpolation-based NMS with many full-image intermediates;
- library CCL plus label selection for hysteresis; and
- limited buffer reuse.

The refined prototypes use separable Numba filtering on CPU, CuPyX-generated
Gaussian and Sobel filters on GPU, explicit custom GPU kernels where appropriate
for NMS and CCL, local per-output-pixel NMS, fewer full-image passes, and
reusable workspaces. Their important difference is dataflow and materialization,
not simply “vectorized Python versus a lower-level language.” Neither refined
prototype materializes an explicit full-frame padded image for blur or
gradients.

That state was reached incrementally. A May 2026 suspect-operator proxy was
prompted around padding, vertical convolution, and a serial BBDT-like scan, but
the evolving CPU operator also had substantial intermediate-allocation churn,
a slow NumPy `hypot` path, and extra padding passes. Reusable preallocation, a
custom parallel compiled magnitude operation, and direct boundary handling
later removed those costs. The proxy code was fully ChatGPT-generated from the
maintainer's operator-suspect prompt, so its results are exploratory workload
samples rather than a complete diagnosis of the later refined CPU operator.

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

### Tiny-image constraint inspection

On 2026-09-29, the latest external refined prototypes were identified from
their filenames, modification times, and contents as
`pseudo_cpu_canny_numpy_mod9_nms_conv_kern_hyccl_fused_grey_prealloc_bconv_ski_fp32.py`
and
`pseudo_gpu_canny_cupy_opt4_kern_nms_ccl_greyscl_fused_prealloc_ski_cucim_amd_cupy_fp32.py`.
They remain external inspection sources, not integrated repository code.

The CPU separable convolution uses an eight-output register-blocked vertical
interior followed by a scalar tail. The block width does not impose an
eight-pixel image-width minimum: a narrower image has no full block and is
handled by the tail. The actual restriction comes from the horizontal and
vertical border loops. The left/top input coordinates are clamped only at zero,
and the right/bottom coordinates only at the final index. Their opposite border
regions and index ranges are valid when the relevant dimension is at least
twice the kernel radius. Equality is sufficient; the full odd kernel width is
not required.

For the active Gaussian settings, `radius = int(4.0 * 1.5 + 0.5) = 6`, so the
custom CPU convolution requires height and width of at least 12. The radius-one
Sobel passes independently require only 2. A 12-by-12 Gaussian result was
checked against SciPy nearest-boundary filtering and agreed within roughly
`1.2e-7` maximum absolute error. The CPU minimum should therefore be derived as
`2 * max(1, gaussian_radius)` from the same radius calculation used to build
the kernel. The old 3-by-3 statement was too weak for this implementation, and
the convolution docstring's full-kernel-size claim was unnecessarily strict.
The repository's initial workspace factory also enforced 3 by 3, but a
workspace allocation has no `sigma` with which to enforce the operator
requirement. That factory now accepts any positive 2D shape; the future public
Canny boundary owns the active image/workspace compatibility check.

The GPU stages have no corresponding footprint-derived minimum. RGB matrix
multiplication, same-shaped workspaces, and CuPyX Gaussian and Sobel operations
with nearest-boundary handling accommodate singleton axes. The custom CCL also
has a valid one-pixel path by source inspection:

- initialization writes background label 0 or foreground label 1;
- the local-union kernel rejects all four neighbor candidates as out of bounds;
- root compression either skips background or counts label 1 as the root at
  flattened index 0; and
- the keep-label kernel can write label index 1 into the two-element
  `max_label + 1` table before `take` maps label 0 or 1.

These conclusions are source-derived because no CUDA device was available for
a runtime boundary test. They cover both the no-component early return and the
one-component foreground path. The custom CCL and its one-pixel launches impose
a nonempty 1-by-1 floor but no larger dimension constraint.

Both prototypes also contain a separate NMS boundary defect that is not solved
by any minimum dimension. Forward and backward bilinear coordinates clamp the
left/down index only at the lower bound and the right/up index only at the upper
bound. An exactly axis-aligned gradient pointing out through a right or bottom
edge can leave the complementary coordinate one past the array. Running the
CPU kernel's Python-level implementation on a crafted 3-by-3 horizontal
gradient reproduced an out-of-bounds `IndexError`. During integration, clamp
every NMS coordinate to both bounds in the CPU and GPU implementations before
claiming their backend-specific minima. This is a correctness fix, not a reason
to force the GPU to adopt the CPU convolution minimum.

With that correction, the NMS kernels accept any nonempty dimensions,
including 1 by 1: both coordinates of every sample clamp to zero on a singleton
axis. The uncorrected external GPU NMS kernel is not independently safe for an
arbitrary singleton-shaped magnitude/direction input. A complete 1-by-1 call is
only source-expected to avoid the defect because nearest-boundary Sobel produces
zero gradients and therefore zero NMS offsets. Treat 1 by 1 as the integrated,
corrected GPU contract and require an actual device test rather than presenting
the external prototype as already verified.

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

Later 2026-09-29 runs changed the vertical convolution interior to update eight
neighboring outputs per tap loop while preserving a scalar tail. These are a
newer source generation, not replacements for the earlier observations above.
On the roughly 23.23 MP workload, Gaussian blur improved from about 44 ms to 36
ms while the three-tap Sobel gradient stage remained in the high-40 to roughly
50 ms range; the final binary output was byte-for-byte identical between the
two convolution variants on that image.

On a roughly 122.99 MP natural image, direct original-versus-blocked trials put
blur around 241–249 versus 161–176 ms, gradients around 254 versus 248 ms, NMS
around 130 ms in both, and hysteresis around 570 versus 568 ms. Total execution
including grayscale improved from roughly 1.40 s to 1.30 s, or about 7%, while
the blur stage improved by about 1.46x. The resulting roughly 94–95 MPix/s
complete CPU rate is approximate external-prototype evidence. It shows a
specific filtering improvement and bottleneck migration toward hysteresis, not
an across-the-board stage speedup or a theoretical CPU limit.

In the corresponding external CPU/GPU comparison, RTX 5000 Ada device-side
grayscale plus Canny was roughly 44–45 ms, Canny-only roughly 34–35 ms, and a
later transfer-inclusive boundary roughly 90–106 ms. CPU and GPU final edge
maps differed at 48 of 122,991,175 pixels, approximately 0.000039%. Preserve
the device-only, transfer-inclusive, CPU-source-generation, and semantic
boundaries when citing those results.

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

This investigation was mostly ChatGPT-generated over several turns. Multiple
candidate scripts were produced, and the final figures came from at least two
scripts; the exact number and turn-by-turn provenance were not retained and
should remain unknown rather than reconstructed speculatively.

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

This proxy was mostly ChatGPT-generated over several turns, with intermediate
candidate scripts. All final measurements above came from one identified
external script, `copy_label_and_nms_proxy_bench.py`. That source remains
outside this repository pending a separate decision about whether this reusable
operator-class proxy merits inclusion.

The suite samples streaming, regular local-neighborhood, and irregular
connectivity workload classes; it does not predict full Canny exactly. The
historical analysis used a weighted harmonic combination:

```text
S_total ~= 1 / sum(f_i / S_i)
```

Using an approximately 0.63 regular/local CPU share, 0.37 connectivity share,
19.1x NMS-lite ratio, and 28.5x label ratio gave approximately:

```text
S_proxy ~= 1 / (0.63 / 19.1 + 0.37 / 28.5) ~= 21.7x
```

This was informally compared with an approximately 30x observed operator ratio.
That numerical relationship is now historical rather than current evidence.
The CPU baseline later changed, particularly through vertical-convolution
register blocking, so the old 0.63/0.37 stage weights no longer characterize
the refined direction. The proxy also does not model the actual Gaussian
footprint, Sobel and interpolation-NMS arithmetic, materialization choices,
complete hysteresis preprocessing, or interaction among stages. Do not claim
that it explains a fraction of current Canny speedup and do not use statistical
language such as variance explained or `R^2` for this one comparison.

Image dimensions and CCL content provide separate limitations. Fixed overhead,
cache and working-set changes, CPU scaling, GPU occupancy, and memory behavior
can change CPU/GPU ratios with size; do not repair a cross-size comparison by
scaling timings with pixel count. At fixed dimensions, foreground density,
component count and sizes, provisional equivalences, root chains, and
compression or convergence behavior can make a synthetic label mask easier or
harder than a real Canny workload. Random input is not an assumed worst case,
one large component is not automatically difficult, and CPU and GPU CCL may
respond differently to the same topology.

The harmonic form itself remains useful when `f_i` is the current CPU
baseline's time fraction and `S_i` is the currently measured class speedup. A
future quantitative comparison should match:

- CPU and GPU code revisions and the current CPU stage fractions;
- dimensions, dtype, sigma or footprint, and relevant thresholds;
- timing boundary and allocation/workspace policy; and
- broad source-content and connectivity class.

It should then measure current proxy ratios and actual Canny speedup under that
setup and compare them descriptively. Such a comparison remains approximate
and is not required for ordinary Canny benchmarking.

The proxy's existing NMS-lite payload comment undercounts source-level logical
accesses. Its kernel reads center, left, right, up, down, `gx`, and `gy`, then
writes one float32 result: seven reads plus one write, or 32 bytes/pixel rather
than 16. This correction does not change timings or CPU/GPU ratios. Even the
corrected source-level payload is not literal DRAM traffic because caches,
coalescing, and reuse affect physical transfers. The proxy source remains an
external context artifact rather than repository code.

The later reduction and Gaussian-footprint experiments in
`filtering-and-reduction-experiments.md` show why “regular/local” is not one
homogeneous class: footprint, layout, cache reuse, compiler SIMD behavior, and
device characteristics changed CPU/GPU ratios substantially. This reinforces
using the proxy as an execution-motif probe rather than explaining the gap
between its old 21.7x estimate and one full-operator result.

If the proxy is refined, keep it small and inspectable rather than adding
Gaussian, Sobel, interpolation NMS, Canny hysteresis preprocessing, internal
artifact exchange, public validation, backend dispatch, or workspace logic. A
lightweight option is to run both real Canny and the proxy on the same small
deterministic family of source images, then derive the proxy label input with a
trivial direct threshold. Smooth, sparse, fragmented/noise-like, dense
connected, and late-merge patterns are possible choices, not a mandatory
matrix. The proxy mask is only broad workload-shape alignment and is not
expected to match Canny's post-NMS low-threshold mask.

Before a material behavior change to the current external unversioned script,
make one `.bakN` backup so the old measurements remain reconstructable. If a
proxy is later admitted to this Git repository, preserve the original through
version history rather than an in-repository backup file.

### Allocation and first touch

One approximately 0.492 GB float32 allocation experiment with 16 threads
reported about 27.5 ms for touching a reused/backed buffer versus about 44.5 ms
for `np.empty` plus first touch under default glibc behavior. With mimalloc
configured to retain backed pages, the nominally fresh case approached reused
buffer behavior. The interpretation was page reuse, not magically increased
page-mapping bandwidth. Once the operator became heavily preallocated, page
faults no longer explained most warm steady-state time.

Gemini mostly authored this experiment, with possible later ChatGPT corrections.
The detailed provenance is scattered and is not worth reconstructing; do not
claim a more exact authorship history unless new primary evidence appears.

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

The batch-32 SciPy/cc3d benchmark harness was also largely ChatGPT-generated
and iterated over multiple LLM turns. Preserve that distinction from the
maintainer's experimental design, execution, validation, and interpretation.

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

Codex generated this compatibility experiment in one pass for the separate
ROCm bootstrap project, with the intent to test a small independently written
primitive rather than imitate known third-party code. That intent is not a
substitute for source review or a claim of provable originality.

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

## Content-sensitive benchmark suite design

The broad candidate taxonomy for CCL-sensitive Canny inputs includes:

1. low-edge or smooth images;
2. ordinary natural images;
3. highly textured natural images;
4. random or noise images;
5. sparse isolated-edge patterns;
6. dense, simply connected structures;
7. long continuous strokes, mazes, or space-filling-like patterns;
8. late-merge or provisional-component torture patterns; and
9. near-threshold or weak-edge-heavy images.

This is a menu, not a mandatory test matrix. Choose a small, partially
overlapping subset according to the question being answered:

- A correctness-oriented subset can use smooth or low-edge input, ordinary
  natural content, near-threshold weak edges, sparse isolated components, and
  a dense but simply connected structure. Add one late-merge pattern only if it
  exposes useful equivalence-resolution behavior. Together these cover flat or
  nearly flat input, threshold boundaries, isolated components, weak-component
  retention based on strong-pixel membership, one large component, and a
  representative difficult merge case.
- A performance-oriented subset can use ordinary and highly textured natural
  content, random/noise input, sparse isolated components, dense/simple
  connectivity, and a late-merge pattern. Add a long stroke or maze-like case
  only if its measured behavior is not already represented. These cases sample
  many small components, one large simple component, dense candidates, and
  many provisional roots that merge later.
- A deep-profiler subset should contain only cases that are diagnostically
  interesting or explain an observed outlier. It need not repeat every routine
  correctness or performance case.

Random/noise input is not inherently a worst case: it may produce many small
fragments that resolve more easily than a large or late-merging connected
structure. Likewise, a long snake, Hilbert-like curve, Morton/Z-order-like
path, or maze has large geometric extent but does not necessarily create a
difficult parent tree. Comb, stripe, or maze patterns that force many
provisional components to merge later may be more adversarial, but they remain
tested synthetic cases rather than formal worst cases.

Routine reports should record, where practical, whole/operator and Canny-only
timing; blur, gradient, NMS, and hysteresis/CCL stage timing; dimensions;
content class; dtype; sigma and thresholds; and final edge-pixel fraction. A
targeted CCL investigation may additionally record weak/strong candidate
fractions, component count and size distribution, convergence or pass count,
CCL-kernel-only timing, and relevant profiler counters such as L2/DRAM traffic,
excess sectors, atomics, and stalls. Those diagnostics are not required for
every routine benchmark.

Avoid crossing every content class with every resolution, sigma, threshold
pair, backend, CCL implementation, timing boundary, and profiler setup. Report
throughput ranges across tested connectivity classes, representative or median
natural-image performance, best and worst observed tested classes, and
stage-level explanations for content-sensitive outliers. Do not elevate one
synthetic result into a universal characterization.

## Reference implementation context

cuCIM became the primary serious CUDA library baseline because it accepts CuPy
arrays directly and supplies a scikit-image-style Canny operator. hipCIM ports
the same API lineage to ROCm and is the natural corresponding baseline where
supported. The primary semantic/reference baseline remains scikit-image.

Two separate pieces of informal reconnaissance support this practical baseline
choice:

- An inquiry in the Python Community Discord did not identify a faster Canny
  implementation that directly accepts CuPy arrays.
- On 2026-09-27, the maintainer informally asked Gemini-powered Google AI Mode,
  Muse Spark, and DeepSeek about maintained or production-oriented Canny
  implementations with that same direct-CuPy public interface. All three
  returned substantially the same practical ecosystem picture and did not
  identify another such implementation.

Neither exercise was a systematic literature review, exhaustive package
survey, expert certification, proof that no alternative exists, or evidence
that cuCIM/hipCIM is globally fastest. Keep them separate rather than combining
their agreement into a stronger claim.

The commonly mentioned alternatives occupy different interface or maintenance
categories:

- OpenCV CUDA provides Canny through its CUDA interfaces, which use
  `cv2.cuda.GpuMat` rather than directly accepting a `cupy.ndarray`. It remains
  mature production-native context.
- Kornia provides a GPU-capable Canny operator for PyTorch tensors rather than
  a CuPy-native interface.
- The historical `cupyimg` project includes a CuPy/skimage-style Canny and was
  described during this reconnaissance as having been superseded by migration
  toward cuCIM; it is not treated as the current maintained baseline.
- `cupyx.scipy.ndimage` provides lower-level operations such as Sobel, not a
  complete Canny operator.

Relevant primary project pages include the
[cuCIM API index](https://docs.rapids.ai/api/cucim/legacy/genindex/),
[official hipCIM repository](https://github.com/ROCm-LS/hipCIM),
[OpenCV CUDA API](https://docs.opencv.org/4.7.0/d1/d1a/namespacecv_1_1cuda.html),
[Kornia 0.8.2 filter API](https://kornia.readthedocs.io/en/v0.8.2/filters.html),
[CuPy 14.0.1 Sobel API](https://docs.cupy.dev/en/v14.0.1/reference/generated/cupyx.scipy.ndimage.sobel.html),
and the [cupyimg repository](https://github.com/mritools/cupyimg). No analogous
CPU ecosystem survey was performed, intentionally: scikit-image is already the
selected semantic reference for the specialized custom contract, and this
documentation update does not open a separate CPU-baseline search.

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

The Kornia and OpenCV measurement harnesses were largely Gemini-generated,
possibly with some ChatGPT involvement. Their detailed provenance is scattered;
do not claim a more exact division of authorship without new primary evidence.

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
