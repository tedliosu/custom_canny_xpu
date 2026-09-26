# Repository Agent Instructions

These instructions apply to the entire repository. They are the durable
working agreement for agent-assisted development and remain applicable without
access to prior chats or files outside this checkout.

## Sources of truth

1. Read this file and the complete top-level `README.md` before planning work.
2. Treat the current Git checkout as authoritative for implementation and
   commit state. External notes and conversations may explain intent, but they
   do not prove that code was integrated or tested.
3. Treat the README roadmap as the single canonical project backlog. Do not
   create a parallel `TODO.md`, raw chat dump, or machine-local context file.
4. If external context conflicts with committed code or documentation, report
   the discrepancy. Do not silently choose one version.
5. Keep durable, project-specific decisions in the repository. Do not make the
   project depend on private chats, Downloads directories, absolute home paths,
   or one developer machine.

`AGENTS.md` is a maintained working agreement, not an immutable history. Propose
focused updates when durable scope, workflow, review, or safety decisions
change. Do not use it as a second technical backlog or silently override
current code.

When completing a README roadmap item, inspect its details before removing it.
Preserve separate deferred work, durable constraints, unresolved decisions,
and non-goals in the appropriate section.

## Collaboration and learning model

The maintainer owns architecture, semantics, interfaces, invariants, scope,
and final acceptance. This project is partly a deliberate learning exercise.
When a task is reasonable for a junior software engineer and is neither
critical nor mechanical busywork, give the maintainer one or two attempts
before supplying a complete implementation. Plans, questions, references, and
small illustrative snippets are welcome forms of scaffolding.

This learning-first rule does not require delay for:

- critical correctness, destructive-action, security, licensing, or
  provenance issues;
- a task the maintainer explicitly asks an agent to implement;
- repetitive mechanical work with little learning value; or
- a failure for which the maintainer requests direct help after attempting it.

Use risk-weighted review. Passing tests are evidence for the behavior encoded
by those tests, not proof that the intended contract is correct. Explain
meaningful assumptions and tradeoffs so the maintainer can evaluate them.

The maintainer has limited time while completing a master's program. Prefer
small, reproducible, high-leverage changes. Continue performance tuning only
when there is a specific testable hypothesis and bounded cost. Treat a new CCL
algorithm, general framework, or broad infrastructure rewrite as a separate
scope decision.

## Editing and approval workflow

Before editing:

1. inspect `git status`, relevant committed files, and the README roadmap;
2. identify what is implemented versus merely described in historical notes;
3. state the smallest coherent file and diff boundary; and
4. identify behavior, public interfaces, dependencies, or support claims that
   could change.

The maintainer requires approval for destructive or irreversible edits. A
direct request to create or revise named repository files authorizes those
ordinary edits, but does not authorize unrelated deletion, history rewriting,
staging, committing, pushing, or modification outside the repository. Ask
before destructive action when scope is not explicit.

During editing:

- preserve unrelated user changes in a dirty worktree;
- use the smallest coherent diff and avoid opportunistic refactors;
- do not modify files outside this repository unless explicitly authorized;
- outside Git, make one `.bakN` backup before an authorized edit when no other
  recovery mechanism exists;
- do not create backup/context files inside the repository as an alternative
  to Git;
- leave agent-made edits unstaged for maintainer review;
- treat permission to edit as distinct from permission to stage, commit, or
  push; and
- never modify a maintainer-supplied external cc3d checkout without explicit,
  target-specific permission.

After editing:

- inspect the complete affected files and `git diff`;
- run `git diff --check`;
- run focused syntax, unit, type, and lint checks that already exist and are
  relevant to the change;
- report exactly what ran and what could not run;
- distinguish unstaged, staged, committed, and pushed state precisely; and
- reconcile completed work with the README without deleting distinct future
  tasks.

Stop with an unstaged diff after agent-made edits so the maintainer can review
it. Stage only after the maintainer explicitly approves the reviewed diff for
commit, then inspect the staged diff before committing. A commit request does
not authorize a push; pushing always requires separate explicit authorization.

### Commit messages and co-authorship

Commit a completed and reviewed unit as a focused change with a coherent
behavior and rollback boundary. Do not combine unrelated implementation,
testing, or documentation work merely to reduce the number of commits.

Use a concise commit title and explanatory body paragraphs consistent with
recent repository history. For a commit co-authored by the maintainer and
Codex, end the commit message with this exact established closing clause:

```text
Co-authored by Yensong Ted Li and Codex
```

Preserve the wording and capitalization unless the maintainer explicitly
changes the repository convention. The co-authorship clause does not alter the
review gate: stage only the approved files, inspect the staged diff, and obtain
explicit commit authorization before creating the commit.

## Project and architecture contract

Use “operator” for the Canny computation. Avoid calling it a pipeline except
when discussing a larger application or an execution sequence for which that
term is materially useful.

The Canny operator accepts only a two-dimensional, finite, `[0, 1]` float32
grayscale array. It consists of Gaussian filtering, Sobel gradients,
interpolation-based NMS, CCL-based double hysteresis, and reusable workspace
management. RGB-to-grayscale conversion is a separate composable operator.
scikit-image is the primary semantic reference, but the custom operator is not
assumed to match every scikit-image or OpenCV behavior exactly. Treat
`docs/operator-contract.md` as the durable detailed API contract.

The initial Canny contract uses fixed nearest-boundary behavior and absolute
thresholds on NMS-suppressed Sobel magnitude. Do not normalize magnitudes by
the maximum magnitude of each image. Quantile thresholds, masks, other border
modes, wider dtypes, integer Canny input, and implicit RGB handling are deferred
contract decisions.

Prefer backend-appropriate implementations of a shared high-level contract.
Fair CPU/GPU comparison does not require identical low-level algorithms or
source structure.

The refined CPU direction may use NumPy, Numba, SciPy, and reusable workspaces.
SciPy CCL is the default direction. A custom nogil Cython/cc3d binding remains
optional and experimental; releasing the GIL does not establish that cc3d is
universally faster than SciPy. Select cc3d explicitly rather than through an
availability global that silently changes execution. When available, prefer
the nogil/preallocated cc3d path for the primary optimized custom CPU-versus-GPU
benchmark, with the SciPy result labeled as supplementary or appendix context.
The public `ccl` argument defaults to `None`, meaning the backend default, and
is mutually exclusive with an explicitly supplied workspace. A CPU workspace
records its selected CCL implementation; do not accept a second CCL selection
that could disagree with it. Resolve CCL names through a narrow callable
strategy boundary rather than storing implementation modules in a registry.

The refined GPU direction may use CuPy/CuPyX, `ElementwiseKernel`,
runtime-compiled CUDA/HIP-compatible code, custom CCL operations, and reusable
device workspaces. CuPy should remain useful on both CUDA and ROCm where the
required features work. Do not describe a CUDA pass as ROCm acceptance.

Prefer reducing redundant materialization and full-image passes, reusing
buffers, using separable operations where appropriate, and computing local
neighborhood math directly. Do not assume that GPU is automatically fast,
Python is automatically slow, shared memory/LDS always helps, or lower-level
code is inherently the right next step.

For small filtering kernels, benchmark a simple cache-friendly global-memory
implementation before assuming shared-memory/LDS tiling will win. For CCL,
recognize that parent traversal can escape a tile; shared-memory CCL normally
requires an algorithmic redesign rather than a local substitution.

The reviewed refined prototypes do not materialize full-frame padding for blur
or gradients. The CPU prototype uses separable Numba loops with explicit
nearest-boundary handling and reusable buffers. The GPU prototype uses CuPyX
Gaussian and Sobel filters, followed by ordinary CuPy clip and magnitude
operations. Both external drafts still contain a per-image maximum reduction
used for magnitude normalization; remove that normalization during integration
to satisfy the absolute-threshold contract. Do not attribute historical
timings to explicit `pad` passes or describe those CuPyX filters as custom
repository kernels.

CuPyX filtering can perform well through generated elementwise kernels with
explicit footprint indexing; do not infer shared-memory tiling, warp-specialized
stencils, or another exotic algorithm from timing alone. When source internals
support a durable claim, cite the relevant tagged CuPy version rather than the
moving `main` branch. Distinguish source inspection from profiler evidence and
do not assert the exact cache, instruction, occupancy, or memory bottleneck
without appropriate measurements.

The refined CPU/GPU prototype comparison is meaningful but not
architecture-pure: its CPU CCL exposes less multicore parallelism than its
custom GPU CCL.
Do not present that asymmetry as either a perfectly fair hardware comparison or
an obvious unfinished loop optimization. Serious multicore CCL can require
partitioning, label-range management, border-equivalence merging, and a
specialized solver. A major algorithmic change to one backend needs a
meaningful counterpart, analogue, or explicit justification on the other when
it materially changes the comparison. Deep cross-backend CCL redesign is a
separate scope decision.

## Alpha artifact policy

Historical alpha implementations are learning artifacts representing a
high-level vectorized, materialization-heavy architecture. Once they are added
to the repository, preserve these defining traits:

- `sliding_window_view` plus `tensordot` filtering;
- vectorized/intermediate-heavy NMS;
- older allocation and dataflow choices; and
- library-label-based CCL hysteresis.

Do not silently modernize alpha code until it resembles the refined
implementation. Changes should be limited to justified correctness or safety
fixes, shared orchestration and timing needed for fair comparison, and explicit
maintainer-approved exceptions. Do not run very large flat images through
alpha implementations: their logical working sets and temporary arrays can
cause severe memory growth or GPU OOM.

Do not express alpha safety as a universal pixel-count threshold. In the
inspected full-2D Gaussian, `K = 2 * ceil(4 * sigma) + 1` and the logical
float32 window footprint is approximately `height * width * K^2 * 4` bytes.
The view is strided rather than fully allocated, but contraction packing,
workspace, and materialization can still exhaust memory. The near-25-MP OOM
observation applies only around `sigma=1.5` on the tested RTX 5000 Ada 32 GB
configuration. Consult `docs/alpha-memory-scaling.md` before changing or
running alpha artifacts.

The alpha location and its exact lint/type-check policy remain undecided. Do
not decide them implicitly while doing unrelated work.

## API and code-quality rules

- Document public and cross-module interfaces: purpose, parameters, accepted
  shapes/dtypes, return value, mutation, workspace ownership, and meaningful
  exceptions.
- Do not expose more end-user Canny parameters than scikit-image without an
  explicit contract decision. In particular, do not add OpenCV-style gradient
  norm or Sobel-aperture controls by default.
- Use `custom` or `cust`, not `my`, in refined public names. Historical alpha
  names may remain frozen.
- Keep code derived from or closely adapted from upstream in clearly separated
  files with the best available source/version and licensing information. If
  an exact historical revision is irrecoverable, document the known branch or
  timeframe and the uncertainty instead of guessing.
- Do not access a variable outside a control-flow region that is its only
  declaration. Initialize it explicitly in the owning scope.
- Prefer `pathlib.Path` for new path-oriented Python interfaces unless an
  existing local API or concrete semantics favor another representation.
- Prefer exceptions for library/API failures and return codes for process-level
  CLI status, unless a reviewed contract calls for something else.
- Standardize repeated choices when doing so clarifies semantics, but do not
  perform broad style churn without a concrete benefit.
- Group related workspace arrays behind an explicit object only after their
  ownership and lifetime contract is clear.
- Keep output buffers separate from temporary workspaces. Returned output must
  remain valid across workspace reuse; a caller-supplied output remains
  caller-owned.
- If no workspace is supplied, the public operator constructs a temporary
  backend-appropriate workspace after validation and backend resolution, then
  executes through the same internal path used for a reusable workspace.
  Distinguish allocation-inclusive calls from reusable-workspace timing.
- Treat workspace buffers as private implementation state. Normal first use and
  reuse must initialize every scratch value that a call reads, so call `n + 1`
  does not depend on call `n`. Caller replacement, resizing, retyping, or
  arbitrary mutation of internal buffers is unsupported and has undefined
  operator-level behavior. Do not add full-buffer integrity scans, defensive
  clearing, checksums, or reconstruction solely to guard against such
  tampering; inexpensive backend/shape/capacity/dtype compatibility checks are
  still appropriate.
- The refined implementation should eventually pass the selected type and lint
  checks. Suppress a diagnostic only when the code is correct and the reason is
  documented or obvious from local context.

Prefer explicit organization around ownership, lifetime, and dispatch
boundaries. Use objects for stateful resources such as reusable workspaces and
narrow strategy interfaces for genuinely interchangeable implementations.
Prefer ordinary functions for stateless numerical operations. Do not introduce
inheritance, indirection, or generalized backend abstractions solely for
hypothetical future frameworks. Resolve CPU/GPU dispatch once near the public
boundary rather than repeat backend conditionals throughout every stage.

Public backend choices are `auto`, `cpu`, and `gpu`. Automatic selection
follows array residency and never transfers an array: NumPy selects CPU and
CuPy selects GPU. Explicit backend/residency mismatches and unavailable forced
backends fail clearly. Library code uses exceptions; only a process-level CLI
uses return codes.

Validate a caller-controlled invariant once at the narrowest public or object
construction boundary that owns it. The public Canny operator owns its full
input, parameter, output, backend, and workspace compatibility checks;
workspace/strategy factories own their construction arguments. Private stages
and compiled kernels may rely on those established invariants rather than
repeat the same validation.

The CLI should eventually expose image path, ordinary Canny parameters,
backend choice, reference comparison, stage reporting, and benchmark block
settings. Reference comparison, stage reporting, and backend choice should be
orthogonal. Automatic selection must work in a CPU-only environment; an
explicit request for an unavailable GPU or cuCIM/hipCIM reference must fail
clearly.

Batch/video processing is future work. Do not add broad batching machinery
until the single-image interface is stable. Retain deliberate single-threaded
CPU fallbacks where they help evaluate frame threading versus inner-operator
parallelism, and avoid nested oversubscription.

## Correctness and testing

Use `pytest` for the eventual Python test suite unless a concrete need supports
another choice. Include both positive behavior and representative failure
paths.

After meaningful numerical or performance changes, verify:

- stage-level behavior and edge cases;
- refined CPU versus refined GPU agreement for the same contract;
- disagreement against the selected scikit-image reference;
- shapes, dtypes, thresholds, border modes, and zero/empty cases;
- optional-dependency absence and explicit forced-backend failures; and
- workspace first use and ordinary reuse without stale-data contamination.

Test flat inputs through the ordinary operator path. They must produce an
all-false Boolean result without division warnings, NaNs, or infinities; do not
add a separate image-level flat-input fast path merely to satisfy the test.

For GPU reference comparisons, keep the custom and reference Boolean outputs
device-resident, count differing pixels on device, and transfer only the final
scalar after synchronization. Treat this reference-only transform-reduction as
the selected future CUB/rocPRIM primitive opportunity. Keep its time outside
Canny-only measurements, and do not retain per-image magnitude normalization
merely to preserve a reduction inside the operator.

Exercise hysteresis on multiple content classes, not resolution alone. Record
foreground density and component/connectivity characteristics when practical;
component counts, sizes, chain structure, and propagation behavior can change
CCL performance dramatically at fixed dimensions.

A few CPU/GPU pixel differences near floating-point or threshold boundaries
may be acceptable. Do not establish a numeric tolerance from one convenient
image, and do not game disagreement measurements with dilation or other
post-processing. If a train/test-style image split is adopted, the held-out
set must remain a genuine check rather than a target for tuning.

Small, deterministic tests come before giant benchmark images. Tests for GPU
code must synchronize at boundaries where completion is part of the assertion.

## Performance evidence rules

Treat historical timings as informal engineering context. Some
alpha-versus-refined observations did not use fixed, rigorously controlled
source snapshots, so their numbers and derived speedups are approximate.
Regenerate evidence from identified committed code before publishing exact
performance or speedup claims.

Every performance claim must preserve the relevant distinctions among:

- cold/JIT/initialization and warm steady state;
- stage-only, Canny-only, grayscale-plus-Canny, and end-to-end;
- CPU and GPU, plus CUDA and ROCm;
- device-only and transfer-inclusive;
- allocation-inclusive and preallocated/reused workspace;
- image shape/content and synthetic versus natural inputs;
- dtype and precision;
- dependency, compiler, driver, and runtime versions; and
- semantic/reference and specialized implementations.

Use `time.perf_counter()` for host timing. Synchronize GPU work explicitly.
For amortized timing, use blocks of iterations and synchronize at a documented
block boundary; apply the same block structure to CPU measurements when
practical. Do not report coarse logical bytes-per-pass estimates as literal
DRAM bandwidth.

Treat reduction and stencil microbenchmarks as operation-class evidence, not
whole-operator predictors. A logical-tap or input-payload throughput can exceed
or differ from physical memory traffic because of cache reuse and reduction
behavior. Raw CPU-RAM versus GPU-VRAM bandwidth is not an upper bound on a
stage speedup. Report filter footprint as well as image size, and compare GPU
platforms without assigning a difference to microarchitecture alone when
generation, resource count, cache, memory, power, clocks, or software differ.

The inspected NumPy `ndarray.max()` results are consistent with an efficient
streaming reduction, not an obvious pathological primitive. A full-frame
reduction still has a measurable pass cost; if reductions matter, first test a
hypothesis about eliminating or fusing passes rather than assuming `max()`
itself needs replacement.

Do not claim:

- exact semantic equivalence without configuration-specific evidence;
- theoretical maximum CPU performance;
- that cuCIM/hipCIM is proven globally fastest or generally poor;
- that Kornia is universally poor rather than a different abstraction with a
  workload-sensitive iterative hysteresis implementation;
- that an entire operator improved by a stage-specific speedup; or
- that MI300X-versus-Ada results isolate hardware architecture from software,
  compiler, runtime, environment, and tuning differences.

Do not create matching internal stage timings for third-party references by
vendoring their implementations, patching installed packages, or depending on
private APIs. Use clean public timing boundaries when available. Asymmetric
stage detail is acceptable because the references are semantic or contextual
baselines rather than reverse-engineering targets.

## Dependencies and environment safety

Do not install, upgrade, downgrade, or remove Python or system packages unless
the maintainer explicitly requests it. If a dependency is missing, report it
and give the exact proposed dependency change. The maintainer maintains
separate environment-update records.

When the maintainer designates an environment, use its executables. Do not
hardcode that machine-local absolute path into repository code or durable
documentation; add a repository-owned configuration or wrapper only through a
reviewed change.

Avoid persistent ambient GPU-stack mutation. Keep `LD_LIBRARY_PATH`,
`ROCM_HOME`, `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, and similar selection
variables scoped to the relevant command, environment, or repository wrapper.
Do not edit shell startup files, `/etc/ld.so.conf*`, or run system-wide
`ldconfig` for this project without an explicit separate decision.

The code must not assume CuPy, cuCIM, hipCIM, or the optional cc3d binding is
installed. Import the CPU path cleanly without them. Do not vendor or
automatically clone cc3d. Require an explicit external source path for its
optional build and preserve its LGPL boundary and notices.

Generated Cython C/C++, object files, extension modules, benchmark images, and
other local artifacts are not source-of-truth inputs unless deliberately added
under a reviewed repository policy.

Keep narrow, throwaway diagnostic microbenchmarks outside the repository by
default. Preserve their durable findings, boundaries, and provenance in
documentation without turning this checkout into an informal experiment
workspace. The compact copy/label/NMS-lite operator-class proxy is the only
current candidate exception because it may serve as a reusable benchmark; add
it only through a separate reviewed decision. Do not import other external
proxy scripts merely because their results are documented here.

## Provenance and LLM assistance

The project values clear provenance. If code is copied, translated, or closely
adapted from an external source, identify the source, version or commit,
relevant license, and nature of the adaptation before integration. Do not
describe adapted CCL-based hysteresis as an original algorithm.

For the existing double-hysteresis helper, the best available record is that
it was adapted from the then-current scikit-image main branch and was in use by
May 16, 2025. The exact upstream commit was not recorded and is considered
irrecoverable unless new primary evidence appears. Preserve that uncertainty;
do not substitute a convenient release tag or a later matching revision.

Prefer implementation from reviewed requirements and primary documentation.
By default, use the smallest explicit backend-specific compatibility shim
needed. Avoid cosmetically rewriting possibly copied code, hidden vendoring,
or adding third-party license headers without understanding why they apply.

The refined CCL implementation is a deliberate repository-specific exception:
it is expected to derive from CuPy 14.1.x-era upstream source and to vendor the
required templated traits with the derived kernels. Preserve the upstream
structure when needed for correctness and CUDA/ROCm portability, record the
exact source revision, and update `THIRD_PARTY_NOTICES.md` in the same change.
Do not generalize this decision into a preference for recreating template
machinery, and do not apply it to the separate ROCm bootstrap repository. That
project's `cuda::std` and version-sensitive `__hip_internal` compatibility
smoke has its own narrower contract.

Some NMS and historical convolution work was substantially LLM-assisted. The
experimental cc3d binding layer and much of its compilation/build scaffolding
were also substantially ChatGPT-assisted; the maintainer directed, tested,
benchmarked, validated, and integrated that work. Review distinctive or
substantial generated code more closely for correctness
and provenance. LLM disclosure does not substitute for source review,
validation, or licensing analysis, and agents must not replace the
maintainer's engineering judgment.

## Scope boundaries

Do not turn this repository into:

- a general benchmark or GPU framework;
- a full replacement for SciPy, scikit-image, OpenCV, or an array library;
- a broad ROCm environment/bootstrap project;
- the repository for the larger MI300X-versus-H200 execution-motif study;
- an OpenVX or OpenCL demonstration project;
- a generalized batch/video system before the single-image contract is ready;
  or
- a C++/pybind11/CMake rewrite merely because lower-level code is available.

OpenCV is useful mature implementation context, not an optimization mandate.
cuCIM/hipCIM is a useful directly interoperable GPU reference where available.
Kornia may be useful PyTorch-native context but need not be a primary baseline.
For ordinary usage, prefer mature libraries when they meet the required
semantics and deployment constraints. The custom operator exists to study the
operator stack, not to recommend hand-written Canny as the default production
choice. CPU remains a meaningful backend for deployment availability,
CPU-resident data, smaller workloads, debugging, packaging, and systems without
a usable GPU stack.

No `CONTRIBUTING.md` or `SECURITY.md` is required under the current solo,
experimental project model. Reconsider them only if external participation,
distribution, service exposure, or the risk model changes.

## Completion checklist

Before claiming a task complete:

1. verify requested behavior rather than syntax alone;
2. confirm no unrelated files changed;
3. document meaningful interface, dependency, support, or timing-boundary
   changes;
4. report checks run and checks still requiring optional hardware/software;
5. reconcile the README roadmap without losing distinct future work;
6. report repository state accurately; and
7. leave unresolved policy choices visible for the maintainer instead of
   silently deciding them.
