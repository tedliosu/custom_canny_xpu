# custom_canny_xpu

`custom_canny_xpu` is a learning- and performance-oriented implementation of a
Canny-style edge-detection operator for NumPy/SciPy on CPU and CuPy/CuPyX on
GPU. The project explores how backend-appropriate implementations of one
high-level contract behave across CPU, CUDA, and ROCm execution.

For ordinary use, prefer a mature library implementation when it satisfies the
required semantics, deployment constraints, and performance needs. This
project studies the operator stack itself: allocations, materialization,
dataflow, workspace reuse, dtype behavior, backend differences, NMS, and
CCL-based hysteresis that a black-box call normally hides. It is not a general
recommendation to rewrite Canny by hand.

This is an early experimental source repository, not a stable packaged release
or production-supported product. The checkout currently contains initial
package scaffolding, CPU input validation and workspace management, a small
NumPy/CuPy backend registry, and a connected-component-labeling (CCL) form of
double hysteresis. The complete CPU and GPU operators are not yet present.
Design and experiment notes discuss development prototypes that live outside
this checkout; those notes are historical context, not evidence that the code
is installed or supported here.

## Project goals

- Build maintainable CPU and GPU implementations of the same specialized
  Canny-style operator contract.
- Compare backend-appropriate algorithms rather than require identical
  low-level source or execution strategies.
- Study memory traffic, intermediate materialization, workspace reuse, and the
  contrast between regular access patterns over pixel data and irregular,
  data-dependent connectivity work.
- Keep a historical alpha implementation as a learning artifact so later
  architectural improvements can be explained and measured fairly.
- Support CuPy on CUDA and ROCm where the required runtime-compiler and library
  features are available.
- Provide a compact non-generative-AI GPGPU case study without turning this
  repository into a general benchmark framework.

## Illustrative business case

The following scenario is fictitious and exists only to give the educational
project plausible industrial grounding. It does not describe an actual client,
contract, deployment, or validated manufacturing system.

Hawkeye Computer Vision Quality Assurance Inc. has hired you as a data-science
contractor to investigate whether its existing Python ndarray-based
computer-vision inspection application for CNC-machined parts can be
accelerated without materially changing its edge-detection behavior. Its
customers want to increase machining throughput, which increases the volume of
inspection images that the quality-assurance software must process. Profiling
has identified Canny edge detection as a significant contributor to
end-to-end inspection latency.

The task is to investigate whether CPU and GPU implementations can accelerate
the Canny portion while preserving its intended semantics. Performance must be
evaluated together with correctness by comparing custom output against the
existing reference implementation and measuring disagreement. The exercise
does not claim to establish end-to-end defect-detection accuracy for a complete
manufacturing system; its narrower goal is improved Canny throughput with edge
maps sufficiently similar for a hypothetical downstream inspection workflow.
Accordingly, no speed result should be interpreted without its accompanying
semantic-agreement evidence.

## Engineering lessons

The main engineering lesson so far is that controlling dataflow often matters
more than merely choosing a lower-level language. Separable filtering, local
per-pixel non-maximum suppression, fewer full-image passes, fewer temporary
arrays, and reusable workspaces account for much of the improvement in the
newer prototypes. Recent filtering experiments also show that regular stencil
work cannot be reduced to a raw memory-bandwidth comparison: overlapping
windows, footprint size, execution resources, cache behavior, instruction
throughput, and latency hiding can change CPU/GPU ratios substantially. A
same-CPU Numba experiment on an AVX-512-capable i5-11400H further showed that
the enabled SIMD feature set can materially change wider-footprint convolution
performance and even contribute to a cross-CPU ranking reversal. AVX-512 was
not uniformly faster, and later experiments localized most sensitivity to the
row-strided vertical pass without isolating one exact instruction-level cause.
Footprint, layout, compiler lowering, ISA, and thread count can therefore alter
CPU ranking even for the same high-level operation. Detailed evidence lives in
[`docs/filtering-and-reduction-experiments.md`](docs/filtering-and-reduction-experiments.md).

## Related studies and terminology

“Refined” identifies the maintainable CPU/GPU direction being integrated into
the package; “alpha” identifies older, materialization-heavy implementations
preserved only as historical learning artifacts. The optional cc3d work is an
experimental custom binding to the external `connected-components-3d` project,
while the compact copy/label/NMS-lite proxy is a small operator-class benchmark,
not another Canny implementation. Neither is required for ordinary CPU use.

A separate GPU execution-motif study compares representative computational
patterns across GPU environments, including patterns distinguished by
arithmetic intensity and memory-access regularity. Canny motivates some of
those patterns: device-resident pixel-disagreement counting is a planned
low-arithmetic-intensity, regular-access transform-reduction candidate for a
later CUDA CUB or ROCm rocPRIM library experiment. That operation belongs to
reference comparison, outside timed Canny execution. The broader motif taxonomy
and cross-environment study are separate work and are not prerequisites for
completing or publishing this repository.

References to the separate ROCm cloud-bootstrap work mean environment bring-up
and small CUDA/HIP portability experiments performed elsewhere. That work is
not a dependency or a required part of this repository. More terminology and
the bounded public-release proposal are in
[`docs/publication-readiness.md`](docs/publication-readiness.md).

## Operator model

The Canny operator accepts a two-dimensional float32 grayscale array and has
these high-level stages:

1. Gaussian filtering;
2. Sobel gradient extraction;
3. interpolation-based non-maximum suppression;
4. double hysteresis through connected-component labeling; and
5. reusable workspace management around those stages.

RGB-to-grayscale conversion is a separate composable operator, not an implicit
Canny stage. The complete initial API behavior, including shapes, dtypes,
absolute-threshold semantics, backend residency, outputs, errors, and
workspace ownership, is defined in
[`docs/operator-contract.md`](docs/operator-contract.md).

Hysteresis follows the scikit-image-style decomposition: label the
low-threshold mask, identify components containing at least one high-threshold
pixel, and retain those components. CPU and GPU implementations may use
different labeling machinery while preserving that high-level behavior.
The refined prototype comparison is useful but not architecture-pure: its CPU
CCL exposes less multicore parallelism than its custom GPU CCL.
Removing that asymmetry rigorously would require deeper, cross-backend CCL
design and validation that is outside the current scope.

Grayscale conversion, filtering, gradients, and NMS mostly use predictable,
local access patterns over pixel data. CCL-based hysteresis involves root
chasing, equivalence resolution, atomics, irregular memory access, and
synchronization. It can therefore remain the largest stage even when its
absolute throughput is high: regular access patterns of pixel data fly;
irregular connectivity remains hard.

## Current repository contents

- `_backends_bridge.py` registers the always-available NumPy/SciPy backend and,
  when importable, the optional CuPy/CuPyX backend.
- `_canny_double_hysteresis.py` implements the current shared, vectorized CCL
  decomposition of double hysteresis. It still returns `uint8` values 0/255 and
  must be adapted during refined integration to meet the Boolean public-output
  contract.
- `LICENSE` applies the Apache License 2.0 to this repository.
- `THIRD_PARTY_NOTICES.md` records attribution and license terms for adapted
  or vendored third-party material.
- `AGENTS.md` contains the durable working agreement for human and
  agent-assisted development.
- `docs/historical-engineering-notes.md` preserves dated prototype findings
  without treating them as current benchmark guarantees.
- `docs/alpha-memory-scaling.md` records the inspected alpha prototypes'
  parameter-dependent memory risks and separates observations from estimates.
- `docs/filtering-and-reduction-experiments.md` records the informal NumPy
  reduction and CPU/GPU Gaussian-footprint experiments and their limits.
- `docs/benchmark-evidence-lifecycle.md` defines how measurements remain
  current, become historical or pending, or are marked invalid as the project
  evolves.
- `docs/publication-readiness.md` explains related-work terminology and the
  bounded proposal for an initial public experimental release.
- `docs/operator-contract.md` defines the reviewed initial public behavior that
  the refined implementation must satisfy.
- `docs/provenance-and-ai-assistance.md` separates third-party algorithm
  provenance, AI-assisted contributions, maintainer decisions, and known
  inventory limitations.
- `src/custom_canny_xpu/` contains the initial importable package, CPU CCL
  strategy boundary, Canny input validation, and reusable `CpuWorkspace`
  factory.
- `tests/` contains initial package, validation, and CPU-workspace success and
  failure tests.

## Package organization

The current root-level modules are implementation scaffolding, not the stable
public API. The refined implementation is being built in the conventional
`src/custom_canny_xpu/` package, which will export `canny` and
`rgb_to_grayscale`. A root-level `main.py` will call those operators and remain
the application/CLI boundary rather than contain their implementations.
Supported versions, dependency metadata, and installation commands remain to
be defined.

The initial CPU package should keep ordinary CPU stages, their immediately
associated compiled kernels, and the CPU workspace together in a cohesive
backend module. The scikit-image-derived hysteresis decomposition and the
repository-specific CPU CCL machinery should remain in separate modules so
their provenance is clear without splitting every numerical stage into its own
file. Backend resolution belongs once near the public API boundary; numerical
stages should not repeatedly dispatch between array implementations.

The initial target modules are deliberately narrow: `__init__.py` re-exports
the public operators, `_api.py` owns public validation and one-time dispatch,
`_cpu.py` owns the CPU workspace and ordinary CPU stages with their immediate
kernels, `_hysteresis.py` owns the attributed high-level decomposition, and
`_cpu_ccl.py` owns repository-specific CCL strategies and label-retention
helpers. The optional compiled cc3d binding and its build tool should be added
only in their separately reviewed integration change.

## Intended implementation families

### Refined implementation

The refined CPU direction uses NumPy for array storage and ordinary operations,
Numba for explicit native-speed loops, SciPy for the default CCL path, and
reusable NumPy workspaces. An optional experimental Cython binding to the
external `connected-components-3d` project may be retained for selected
single-image or frame-level experiments, but SciPy is the intended public
default. When available, the explicitly selected nogil/preallocated cc3d path
is preferred for the primary optimized custom CPU-versus-GPU benchmark; report
the SciPy path separately as supplementary context. This does not make the
comparison architecture-pure or establish that cc3d is universally faster.
The reviewed prototype performs Gaussian and Sobel work with one generic
separable Numba convolution, explicit nearest-boundary indexing, and
preallocated buffers; it does not materialize full-frame padded inputs. Its
horizontal pass remains direct, while its vertical interior updates eight
neighboring outputs together and uses a scalar tail. Both Gaussian and Sobel
use this implementation so the project does not maintain a second convolution
or a footprint-dependent dispatcher. That external draft still normalizes
gradient magnitudes with a NumPy maximum reduction.
Integration must remove that per-image normalization so thresholds follow the
reviewed absolute-magnitude contract.

The unbatched refined CPU implementation uses the full logical-core worker set
available to its process/runtime for parallel-capable stages. It does not
expose per-operator thread-count tuning or switch to a separate single-thread
algorithm. Such partitioning is deferred to future batch/video work. The
component-retention vector used by hysteresis is deliberately allocated from
the current label-index requirement rather than reserved at full-image-derived
capacity in the reusable workspace.

The refined GPU direction uses CuPy/CuPyX, runtime-compiled CUDA/HIP-compatible
kernels, custom NMS and CCL operations, and reusable device workspaces. CuPy
and GPU reference libraries must remain optional: importing or running the CPU
path must not assume that CuPy, cuCIM, or hipCIM is installed.
The reviewed prototype uses `cupyx.scipy.ndimage.gaussian_filter` and `sobel`
for filtering, then ordinary CuPy operations and custom NMS. Its external draft
still includes per-image maximum normalization that must be removed during
integration. The CuPyX filters are not hand-written repository kernels.
Inspection of the relevant tagged CuPy source showed generated
elementwise-filter machinery with explicit footprint indexing; performance
alone is not evidence of bespoke shared-memory tiling or another hidden
stencil algorithm.

The planned refined CCL implementation derives from CuPy 14.1.x-era upstream
source. For this repository, the selected approach is to vendor the required
templated traits together with the derived kernels rather than depend on
`cuda::std` and version-sensitive HIP-internal traits. Record the exact CuPy
source revision and update `THIRD_PARTY_NOTICES.md` when that code is
integrated. This is a project-specific provenance and portability decision,
not a policy for the separate ROCm cloud-bootstrap work used to test
environment setup and small compatibility shims. That work is not required to
use this repository.

### Historical alpha implementation

The alpha CPU and GPU implementations are intended to preserve the older
high-level vectorized architecture:

- two-dimensional `sliding_window_view` plus `tensordot` filtering;
- vectorized, materialization-heavy interpolation-based NMS;
- comparatively limited workspace reuse; and
- library CCL followed by component selection for hysteresis.

They are learning and comparison artifacts, not templates for the refined
implementation. Once added to the repository, do not modernize away their
architectural character. Correctness fixes, shared orchestration, and rigorous
timing may be appropriate, but large images are unsafe for these versions: the
logical window tensors and NMS intermediates can exhaust even large GPU memory.
There is no universal pixel-count cutoff. The dominant Gaussian exposure scales
approximately with pixels times kernel area, and kernel area grows with
`sigma`; see [`docs/alpha-memory-scaling.md`](docs/alpha-memory-scaling.md).
The final location and degree of separation between alpha and refined code are
still open design decisions.

## Correctness contract

The operator is specialized and is not currently promised to be an exact
semantic clone of scikit-image or OpenCV. scikit-image is the primary semantic
reference because the intended high-level structure includes 3x3 Sobel
gradients, interpolation-based NMS, absolute thresholds, and CCL-based
hysteresis. The initial custom contract is narrower: float32 grayscale input
only and fixed nearest-boundary semantics.

Correctness work should include:

- focused unit tests for individual stages and boundary cases;
- paired expected-success and representative expected-failure tests for each
  public or cross-module behavior;
- CPU/GPU agreement tests under the same logical contract;
- disagreement counts and rates against scikit-image on multiple images;
- explicit dtype, threshold, border-mode, shape, and residency information;
- regression tests before and after meaningful performance rewrites; and
- no post-processing introduced merely to improve a disagreement metric.

GPU disagreement counting should keep both Boolean results device-resident,
perform the count on device, and transfer only the scalar result. This
reference-only transform-reduction is the selected future CUB/rocPRIM
primitive opportunity; its time must not be included in Canny-only
measurements.

A few differing pixels between closely matched CPU/GPU implementations may be
reasonable near floating-point, interpolation, or threshold boundaries. A
general acceptance threshold has not yet been chosen and must be based on
representative evidence rather than a convenient result. Historical
measurements in development prototypes outside this checkout are engineering
context, not repository guarantees. In particular, not every
alpha-versus-refined observation used fixed, rigorously controlled source
snapshots. Treat historical numbers and derived speedups as approximate unless
a specific experiment records otherwise.

## Benchmarking contract

Benchmark evidence belongs to a specific implementation generation, workload,
environment, and experimental contract. Current, historical, pending
revalidation, and invalid evidence are distinct: a sound superseded result
remains useful history, while an uncertain current-facing claim should be
marked pending rather than left misleadingly current. Revalidate selectively
when a change can affect a result or its derived conclusions; do not rerun an
independent experiment whose own contract is unchanged. The detailed status
definitions and procedure are in
[`docs/benchmark-evidence-lifecycle.md`](docs/benchmark-evidence-lifecycle.md).

In particular, measurements from refined prototypes that still performed
per-image magnitude normalization are not current evidence for the integrated
absolute-threshold operator. Removing the reduction can also change threshold
populations, CCL topology, output disagreement, stage fractions, and total
timing, so new performance cannot be inferred by merely subtracting the old
reduction cost. A current CPU/GPU speedup requires both backends to implement a
comparable semantic generation.

Performance results are meaningful only when their boundaries are stated.
Record, as applicable:

- hardware, OS, Python, library, compiler, driver, and runtime versions,
  including exposed CPU ISA features or compiler feature overrides when they
  materially affect the result;
- image dimensions, content class, dtype, and operator parameters;
- final edge-pixel fraction;
- cold/JIT/initialization runs separately from warm steady-state runs;
- CPU versus GPU and CUDA versus ROCm;
- device-resident/operator-only versus transfer-inclusive/end-to-end timing;
- allocation-inclusive versus preallocated/reused-workspace execution;
- stage-only, Canny-only, grayscale-plus-Canny, and full application bounds;
- synthetic/random versus natural-image inputs; and
- semantic/reference versus specialized/performance-oriented implementations.

For reductions and stencils, label calculated GB/s as input-payload or logical
effective throughput rather than literal physical DRAM traffic. Record filter
radius or tap count: image size alone does not characterize stencil work. Raw
CPU-RAM versus GPU-VRAM bandwidth is not an expected ceiling on speedup when
overlapping windows create cache reuse and increase regular arithmetic,
addressing, or instruction work per externally fetched byte.

Hysteresis must be exercised on multiple content classes. Its connectivity
graph, component distribution, root-chain behavior, and convergence can change
substantially even at fixed dimensions, so a single MPix/s figure is not a
content-independent characterization. Keep the suite small and choose cases by
purpose: correctness testing covers semantic behaviors, performance testing
covers distinct workload shapes, and deep profiling is reserved for
diagnostically useful cases and outliers. Do not require every candidate class
for every test or cross every class with every resolution, parameter, backend,
CCL implementation, timing boundary, and profiler configuration.

An initial correctness subset may include smooth or low-edge input, ordinary
natural content, near-threshold weak edges, sparse isolated components, and a
dense but simply connected structure. An initial performance subset may add
highly textured natural content, random/noise input, and a late-merging
provisional-component pattern. Include a long stroke or maze-like pattern only
if it exposes behavior not already represented. Random noise is not assumed to
be a worst case, and geometric path length alone does not establish difficult
union-find behavior. More detailed guidance lives in the historical engineering
notes.

Use `time.perf_counter()` as the host timing source. GPU timing regions must
have an explicit synchronization boundary so asynchronous work is not reported
as completed. A future harness should support blocks of iterations with one
device-wide synchronization per block and expose both block count and
iterations per block. The CPU harness should use the same block structure where
practical.

Do not collapse unlike measurements into a universal leaderboard. In
particular, do not describe the CPU implementation as a theoretical CPU limit,
cuCIM/hipCIM as globally fastest, OpenCV as the only legitimate baseline, or a
cross-vendor MI300X-versus-Ada result as a hardware-only comparison.

Useful baselines serve different purposes:

- scikit-image: semantic/reference behavior;
- the refined NumPy/Numba/SciPy implementation: tuned scientific Python CPU;
- the refined CuPy/CuPyX implementation: custom CUDA/ROCm GPU;
- cuCIM/hipCIM: directly interoperable GPU-library context where available;
- OpenCV: mature production-native context, not a required semantic match.

cuCIM is the primary serious CUDA library baseline because it accepts CuPy
arrays directly and offers a scikit-image-style Canny operator. hipCIM is the
corresponding ROCm-side baseline where supported. An informal inquiry in the
Python Community Discord did not identify a faster directly CuPy-compatible
alternative. Separately, an informal 2026-09-27 survey using Gemini-powered
Google AI Mode, Muse Spark, and DeepSeek did not identify another maintained,
production-oriented Canny implementation with the same direct-CuPy ndarray
interface. Neither exercise was an exhaustive literature or ecosystem review,
expert certification, proof of exclusivity, or evidence that cuCIM/hipCIM is
globally fastest. OpenCV CUDA uses its own `GpuMat` interface, while Kornia is
PyTorch Tensor based, so they remain useful context in different
interoperability categories. No analogous CPU ecosystem survey was performed;
scikit-image remains the deliberately selected CPU semantic reference.

Detailed internal stage timing is primarily for code owned by this repository.
For third-party references, prefer public timing boundaries such as the whole
Canny operator, grayscale plus Canny, and device-only versus transfer-inclusive
execution. Do not vendor or patch a reference, modify site-packages, or depend
on private APIs merely to manufacture matching stage boundaries.

## Dependencies and environments

There is not yet a checked-in dependency specification or supported-version
matrix. Until one is added:

- do not infer reproducibility from a developer's local environment;
- do not install packages automatically on behalf of this project;
- use a maintainer-designated virtual environment when one is supplied;
- keep ROCm/CUDA selection and loader variables command-scoped or in
  repository-owned wrappers rather than modifying shell startup files or the
  system dynamic-loader configuration; and
- probe and report actual runtime versions in benchmark output.

The optional cc3d experiment must build against a maintainer-supplied external
source checkout. Do not vendor, clone, or modify cc3d automatically. cc3d is
LGPL-licensed, and any eventual binding/build instructions and third-party
notices must preserve that boundary.

## Scope and non-goals

This repository is not intended to:

- beat OpenCV at all costs or claim state-of-the-art Canny performance;
- implement every scikit-image or OpenCV option;
- expose extra knobs such as alternate gradient norms or larger Sobel
  apertures merely because another library does;
- become a general array framework, SciPy replacement, scikit-image
  replacement, OpenCV replacement, or benchmark framework;
- make OpenVX or OpenCV's OpenCL path central to the CUDA/ROCm study;
- absorb the separate ROCm environment-bootstrap project or the broader GPU
  execution-motif study;
- add batch/video APIs before the single-image contract is stable; or
- begin a major new CCL algorithm or framework rewrite without a separate
  scope decision;
- become a comparative multicore-CPU/GPU CCL study with Canny only as its
  wrapper; or
- reverse-engineer and instrument every internal stage of third-party Canny
  implementations.

Future batch/video experiments are plausible. Likely directions include
per-worker reusable CPU workspaces without nested oversubscription, and
explicit `[B, H, W]` GPU kernels/workspaces rather than host-thread submission
as the default scaling model.

Narrow diagnostic microbenchmarks are normally disposable investigation tools,
not repository deliverables. Preserve durable results and caveats in the
documentation while keeping their scripts external so this repository does
not become an informal workspace. The compact copy/label/NMS-lite
operator-class proxy is the sole current candidate for later inclusion as a
reusable benchmark, subject to its own reviewed decision. Treat it as a small
execution-motif probe, not a quantitative Canny predictor: its historical
harmonic comparison uses stale CPU stage weights, and neither one image size
nor one synthetic label topology represents current Canny generally. Any
refinement should preserve its separation from the real operator and prefer a
few shared deterministic source-image classes over recreating Canny stages.

## Publication status

Making this repository public would publish an experimental project; it would
not imply a stable API, completed roadmap, packaged release, or production
support. The proposed first useful publication milestone is a working,
documented, reasonably tested refined CPU operator with verified basic
installation and execution instructions. This proposal remains subject to
maintainer review and is not yet satisfied by the current scaffolding.

Optional GPU backends, CUDA/ROCm validation, alpha integration, comprehensive
benchmarking, batch/video support, proxy refinements, and the separate
execution-motif study need not block publication. Licensing, attribution,
provenance, accurate capability claims, and a review of both tracked content
and Git history for unintended private material do remain mandatory for what is
actually published. See
[`docs/publication-readiness.md`](docs/publication-readiness.md).

## Roadmap

This is the canonical high-level backlog. Keep separate future work visible
when completing a parent item.

1. Continue integrating the reviewed refined CPU code against the documented
   operator contract without treating the external prototype as finalized
   source. The `src/custom_canny_xpu/` package, explicit CPU workspace object,
   SciPy-default CCL resolver boundary, and their initial tests are present.
   Next add public validation and CPU dispatch, then complete the SciPy-default
   CPU implementation with the ordinary CPU stages and attributed hysteresis
   decomposition, keeping that decomposition separate from
   repository-specific CCL helpers. Derive the CPU minimum dimensions from the
   active Gaussian and Sobel radii; at `sigma=1.5` the reviewed custom
   convolution requires 12 by 12, not 3 by 3 or 13 by 13. Correct the external
   prototype's one-sided NMS coordinate clamps during integration. Add root
   `main.py` as the application boundary, and bring integrated functions under
   the documented docstring, Pyright, lint, and paired-test rules. Finally,
   review and integrate the optional cc3d binding as a separate change behind
   explicit selection and a lazy optional-import boundary.
2. Integrate the optional GPU implementation with graceful CPU-only import and
   clear forced-GPU failure behavior. Preserve its independently derived 1-by-1
   minimum rather than copying the CPU convolution restriction, and correct the
   same one-sided NMS coordinate-clamping defect during integration. Record the
   exact upstream CuPy revision for derived CCL code and add its attribution and
   license terms to `THIRD_PARTY_NOTICES.md` in the same reviewed change.
3. Decide where alpha artifacts live and add them without erasing their
   historical architecture or allowing unsafe default image sizes.
4. Add a CLI so image paths, Canny parameters, backend selection, references,
   stage reporting, and timing blocks do not require source edits. Keep
   reference comparison and stage reporting orthogonal.
5. Add a `pytest` suite for stage behavior, CPU/GPU agreement, reference
   disagreement, optional-dependency behavior, and representative error paths.
   Cover each backend's minimum width and height, each minimum minus one,
   non-square boundary cases, a normal image, and axis-aligned NMS gradients at
   every border. Exercise the GPU singleton CCL path with both background and
   foreground/strong-pixel cases when GPU tests become available.
6. Establish dependency metadata, supported-version policy, and
   lint/type-check configuration for refined code. Decide separately how much
   checking applies to frozen alpha artifacts.
7. Regenerate reproducible benchmark evidence before publishing exact
   performance or speedup claims. Decide whether the compact
   copy/label/NMS-lite operator-class proxy merits inclusion as a reusable
   harness; keep the other narrow diagnostic scripts external.

The full roadmap is not a public-release checklist. The proposed CPU-first
publication milestone and its narrower review criteria are documented
separately in [`docs/publication-readiness.md`](docs/publication-readiness.md).

## Provenance and LLM assistance

The high-level CCL hysteresis decomposition is derived from scikit-image-style
behavior, and the planned refined GPU CCL derives from CuPy upstream source.
Those source and licensing obligations are distinct from AI-assisted design or
implementation work. Current and planned provenance duties are described in
the provenance document; notices applicable to integrated material live in
`THIRD_PARTY_NOTICES.md`.

All code and documentation in this repository have been drafted with help from
ChatGPT-family models available circa 2025–2026. Architecture, support
boundaries, semantics, interfaces, invariants, acceptance decisions, testing,
profiling, rewriting, and final integration remain human-owned. This disclosure
is transparency about the development process, not proof of originality,
complete provenance, or license compliance. The material contributions
identified so far—including generated code and scaffolding, optimization
ideas, and benchmark/methodology work—are categorized in
[`docs/provenance-and-ai-assistance.md`](docs/provenance-and-ai-assistance.md).
That inventory may be incomplete and should be extended when additional
influence is recognized. Third-party source and licensing obligations remain
independent of AI-assistance disclosure.

## License and project policies

The repository is licensed under the [Apache License 2.0](LICENSE). Adapted or
vendored third-party material is documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and retains its applicable
license terms. Optional dependencies and reference implementations retain
their own licenses.

Reports of suspected undocumented similarity to third-party material or other
licensing concerns are welcome. Any affected code or documentation can then be
reviewed and, when appropriate, corrected, attributed, or replaced.

This is primarily a solo learning project, so there is currently no
`CONTRIBUTING.md`. There is also no separate `SECURITY.md`: the repository is
experimental image-processing code, does not operate a service, and has no
project-specific security-reporting process at present. These files can be
added later if the participation or risk model changes.
