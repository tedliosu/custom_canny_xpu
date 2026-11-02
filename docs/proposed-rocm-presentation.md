# Approved November 17, 2026 ROCm presentation

The Ohio State Open Source Club approved this presentation around October 26,
2026. It was originally scheduled for November 3, 2026 and is distinct from the
maintainer's autumn 2025 ROCm/OpenCV Canny presentation. Current scheduling
status: **presentation rescheduled for November 17, 2026**.

The approved title and subtitle are:

> **ROCm Beyond GenAI: Finding Edges with AMD GPUs**
>
> **Building and benchmarking a parallel Canny edge detector with Python,
> Numba, CuPy, and HIP**

This note records the repository-owned relationship so readers do not need
access to earlier talks, private prototypes, cloud systems, or AI-assistant
conversations to understand the presentation. Presentation preparation now
lives in a separate local-only repository and dedicated session. This canonical
repository retains only the durable publication gate, project boundaries, and
evidence constraints; it does not own talk-specific scripts or orchestration.
The rescheduled presentation does not make package integration a presentation
milestone or certify any result as current.

## Purpose and audience

The intended audience includes students who are relatively new to GPU
computing. Canny edge detection provides a visually understandable application
for explaining why general-purpose GPU computing is worth learning:

- GPUs use a throughput-oriented execution model that suits many forms of
  abundant parallel work.
- Suitable workloads differ in arithmetic intensity, access regularity,
  synchronization, and control flow.
- GPGPU applies well beyond the current emphasis on generative AI, while still
  carrying real portability, programming, transfer, and workload-shape
  tradeoffs.
- Representative real-world applications can show where those computational
  patterns arise beyond this Canny example.
- Regular filtering and irregular connected-component work provide an
  approachable contrast within one operator.

The presentation is not merely a hardware competition or a generic
accelerated-computing survey. It must neither claim that accelerators make every
program faster nor imply that GPUs are limited to embarrassingly parallel work.
The goal is an accessible account of how execution behavior depends on the
computation, without requiring an extended survey of every architectural
tradeoff.

## Environment and demonstration context

The earlier 2025 presentation used an Azure Radeon Pro V710 environment. The
2026 work reflects later access to MI300X environments through
HotAisle and AMD Developer Cloud. These names describe the history and planned
demonstration context; they do not establish supported deployment targets for
the package.

Any presentation material must distinguish:

- historical observations from measurements validated against an identified
  current revision;
- packaged functionality under `src/custom_canny_xpu` from functionality in
  separately maintained experimental scripts;
- demonstrated behavior from planned work; and
- Canny-only timing from reference comparison, transfers, and other timing
  boundaries.

Controlled Canny measurements are the primary performance evidence. Patched
BabelStream results are approximate same-instance streaming-bandwidth context,
not a direct Canny predictor or percentage-of-peak claim. The failed
TheBandwidthBenchmark HIP smoke is not usable evidence. One-shot scikit-image
and hipCIM timings are correctness-reference timing; a custom-versus-library
speedup requires a future warmed, repeated, boundary-matched comparison.

If necessary, the demonstration may use bug-fixed CPU/GPU prototype scripts
that remain external to this package. Their use must be disclosed accurately;
it does not make them packaged functionality or current repository evidence.

### Investigated standalone RPP demonstration

One tentative direction was a standalone host/device Canny example using AMD
ROCm Performance Primitives (RPP) on the AMD Developer Cloud MI300X
environment. The feasibility investigation found that the ROCm 7.2.3 shipped
RPP package/binary does not export the legacy Canny API even though related
legacy-support source remains visible. The older exported implementation also
has material semantic defects: the CPU weak-edge promotion is clobbered, the
GPU promotion is only one hop rather than transitive, and NMS uses fixed angle
bins. RPP is therefore not the current demonstration candidate or a
same-semantics baseline.

This conclusion does not make RPP an implemented package backend and does not
generalize to ROCm computer vision. If a materially different supported RPP
interface appears, it would require a new bounded feasibility/correctness
decision rather than silently reviving the old plan. Separately maintained,
validated CPU/GPU Canny prototypes remain possible presentation alternatives.

Any future RPP demonstration would own its own implementation contract and
benchmark evidence. Host and GPU RPP paths must not be assumed to have identical
algorithms, optimization, or threading behavior, and their measurements are
not interchangeable with this project's results without checking semantics,
dtype, residency, transfer and allocation boundaries, batching, and timing
scope. The completed RPP feasibility conclusion does not start the separately
deferred validation-cost investigation.

## Scope and publication relationship

The preferred development milestone remains the documented and tested refined
CPU operator. The repository was made public as a clearly labeled experimental
work in progress before the presentation's original November 3, 2026 date,
satisfying the earlier presentation-day gate subject to the same
publication-safety review. Current scheduling status: **presentation
rescheduled for November 17, 2026**. See
[`publication-readiness.md`](publication-readiness.md) for the complete gate
and checklist.

Presentation implementation and orchestration stay in the separate local-only
presentation repository. Preparing the presentation does not require
completion of the refined GPU package, the broader GPU execution-motif study,
the separate ROCm bootstrap work, integration of an RPP demonstration, or every
planned benchmark. The optional validation-cost investigation described in
[`operator-contract.md`](operator-contract.md#reference-disagreement) is
likewise neither presentation work nor a publication prerequisite.

The club announcement and slides should link both the stable public repository
URL and the immutable canonical commit used for publication. Presentation-only
work does not require this canonical repository to move that pinned revision.
