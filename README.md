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

This is an early repository. The checkout currently contains a small
NumPy/CuPy backend registry and a connected-component-labeling (CCL) form of
double hysteresis. The complete experimental CPU and GPU implementations still
need to be integrated from reviewed prototypes. Nothing in this README should
be read as a claim that the full operator or benchmark harness is already
available here.

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

The main engineering lesson so far is that controlling dataflow often matters
more than merely choosing a lower-level language. Separable filtering, local
per-pixel non-maximum suppression, fewer full-image passes, fewer temporary
arrays, and reusable workspaces account for much of the improvement in the
newer prototypes. Recent filtering experiments also show that regular stencil
work cannot be reduced to a raw memory-bandwidth comparison: overlapping
windows, footprint size, execution resources, cache behavior, instruction
throughput, and latency hiding can change CPU/GPU ratios substantially.

## Operator model

The intended operator has these high-level stages:

1. optional RGB-to-grayscale conversion;
2. Gaussian filtering;
3. Sobel gradient extraction;
4. interpolation-based non-maximum suppression;
5. double hysteresis through connected-component labeling;
6. reusable workspace management around those stages.

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
  decomposition of double hysteresis.
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

The current modules are implementation scaffolding, not a stable public API.
Function names, package layout, supported versions, and installation commands
remain to be defined.

## Intended implementation families

### Refined implementation

The refined CPU direction uses NumPy for array storage and ordinary operations,
Numba for explicit native-speed loops, SciPy for the default CCL path, and
reusable NumPy workspaces. An optional experimental Cython binding to the
external `connected-components-3d` project may be retained for selected
single-image or frame-level experiments, but SciPy is the intended default.
The reviewed prototype performs Gaussian and Sobel work with generic separable
Numba convolution passes, explicit nearest-boundary indexing, and preallocated
buffers; it does not materialize full-frame padded inputs. Its gradient stage
comprises separable Sobel X and Y, a compiled magnitude pass, and a NumPy
maximum reduction.

The refined GPU direction uses CuPy/CuPyX, runtime-compiled CUDA/HIP-compatible
kernels, custom NMS and CCL operations, and reusable device workspaces. CuPy
and GPU reference libraries must remain optional: importing or running the CPU
path must not assume that CuPy, cuCIM, or hipCIM is installed.
The reviewed prototype uses `cupyx.scipy.ndimage.gaussian_filter` and `sobel`
for filtering, then CuPy clip, magnitude, and maximum operations. Those filters
are not hand-written repository kernels. Inspection of the relevant tagged
CuPy source showed generated elementwise-filter machinery with explicit
footprint indexing; performance alone is not evidence of bespoke shared-memory
tiling or another hidden stencil algorithm.

The planned refined CCL implementation derives from CuPy 14.1.x-era upstream
source. For this repository, the selected approach is to vendor the required
templated traits together with the derived kernels rather than depend on
`cuda::std` and version-sensitive HIP-internal traits. Record the exact CuPy
source revision and update `THIRD_PARTY_NOTICES.md` when that code is
integrated. This is a project-specific provenance and portability decision,
not a policy for the separate ROCm bootstrap repository.

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
reference because the intended high-level structure includes configurable
border handling, 3x3 Sobel gradients, interpolation-based NMS, and CCL-based
hysteresis.

Correctness work should include:

- focused unit tests for individual stages and boundary cases;
- CPU/GPU agreement tests under the same logical contract;
- disagreement counts and rates against scikit-image on multiple images;
- explicit dtype, threshold, border-mode, shape, and residency information;
- regression tests before and after meaningful performance rewrites; and
- no post-processing introduced merely to improve a disagreement metric.

A few differing pixels between closely matched CPU/GPU implementations may be
reasonable near floating-point, interpolation, or threshold boundaries. A
general acceptance threshold has not yet been chosen and must be based on
representative evidence rather than a convenient result. Historical
measurements in private prototypes are engineering context, not repository
guarantees. In particular, not every alpha-versus-refined observation used
fixed, rigorously controlled source snapshots. Treat historical numbers and
derived speedups as approximate unless a specific experiment records otherwise.

## Benchmarking contract

Performance results are meaningful only when their boundaries are stated.
Record, as applicable:

- hardware, OS, Python, library, compiler, driver, and runtime versions;
- image dimensions, content class, dtype, and operator parameters;
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
content-independent characterization.

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
arrays directly and offers a scikit-image-style Canny operator. An informal
inquiry in the Python Community Discord did not identify a faster directly
CuPy-compatible alternative. That reconnaissance was not a literature review,
expert certification, or proof that no faster implementation exists.

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
reusable benchmark, subject to its own reviewed decision.

## Roadmap

This is the canonical high-level backlog. Keep separate future work visible
when completing a parent item.

1. Define the public operator contract: accepted shapes/dtypes, threshold
   semantics, border modes, outputs, exceptions, workspace ownership, and CPU
   fallback behavior.
2. Choose the package/module layout and integrate reviewed refined CPU code
   without treating the external prototype as finalized source.
3. Integrate the optional GPU implementation with graceful CPU-only import and
   clear forced-GPU failure behavior. Record the exact upstream CuPy revision
   for derived CCL code and add its attribution and license terms to
   `THIRD_PARTY_NOTICES.md` in the same reviewed change.
4. Decide where alpha artifacts live and add them without erasing their
   historical architecture or allowing unsafe default image sizes.
5. Add a CLI so image paths, Canny parameters, backend selection, references,
   stage reporting, and timing blocks do not require source edits. Keep
   reference comparison and stage reporting orthogonal.
6. Add a `pytest` suite for stage behavior, CPU/GPU agreement, reference
   disagreement, optional-dependency behavior, and representative error paths.
7. Establish dependency metadata, supported-version policy, and
   lint/type-check configuration for refined code. Decide separately how much
   checking applies to frozen alpha artifacts.
8. Organize workspaces behind explicit objects after the operator contract is
   understood; do not perform a broad refactor solely for tidiness.
9. Regenerate reproducible benchmark evidence before publishing exact
   performance or speedup claims. Decide whether the compact
   copy/label/NMS-lite operator-class proxy merits inclusion as a reusable
   harness; keep the other narrow diagnostic scripts external.

## Provenance and LLM assistance

The CCL-based hysteresis decomposition is derived from the approach used by
scikit-image. Repository history describes the current helper as vendored, and
its structure shows a backend-generalized adaptation. The best available
provenance is that the reference came from scikit-image's then-current main
branch and the resulting implementation was already in use by May 16, 2025.
The exact upstream commit was not recorded and should remain explicitly
unknown. Some historical NMS and alpha convolution code was drafted
substantially with ChatGPT assistance. The experimental cc3d Cython/C++ binding
and much of its build scaffolding were also substantially ChatGPT-assisted;
the maintainer defined the API and preallocation goals, tested and benchmarked
the result, evaluated alternatives, and set its integration and licensing
boundaries. Avoid implying that these algorithms were invented here, and
prefer small, explicit compatibility code over recreating third-party
infrastructure.

The external diagnostic benchmarks have mixed LLM provenance. The May 2026 CPU
suspect-operator microbenchmark was fully generated by ChatGPT from the
maintainer's prompt. The maximum-reduction and Gaussian-footprint benchmarks
were each generated by ChatGPT in one pass. Other experiment-specific details,
including irrecoverable uncertainty, are recorded in the engineering notes.
These probes informed maintainer-directed investigation; they do not establish
the bottlenecks of the later refined operator by themselves.

All code and documentation in this repository have been drafted with help from
ChatGPT-family models available circa 2025–2026. Architecture, support
boundaries, semantics, interfaces, invariants, acceptance decisions, testing,
profiling, rewriting, and final integration remain human-owned. This disclosure
is transparency about the development process, not proof of originality,
complete provenance, or license compliance.

## License and project policies

The repository is licensed under the [Apache License 2.0](LICENSE). Adapted or
vendored third-party material is documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and retains its applicable
license terms. Optional dependencies and reference implementations retain
their own licenses.

This is primarily a solo learning project, so there is currently no
`CONTRIBUTING.md`. There is also no separate `SECURITY.md`: the repository is
experimental image-processing code, does not operate a service, and has no
project-specific security-reporting process at present. These files can be
added later if the participation or risk model changes.
