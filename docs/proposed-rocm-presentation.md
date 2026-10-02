# Proposed autumn 2026 ROCm presentation

The maintainer has proposed a presentation for the Ohio State Open Source Club
in autumn 2026. Its date is not confirmed; the current club officers have not
yet responded about scheduling. It is distinct from the maintainer's autumn
2025 ROCm/OpenCV Canny presentation.

Update this note if the club confirms a date, the proposal is withdrawn, or
the intended demonstration scope materially changes.

This note records the proposal so readers do not need access to earlier talks,
private prototypes, cloud systems, or AI-assistant conversations to understand
its relationship to this repository. It does not make the presentation an
implementation milestone or certify any result as current.

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
proposed 2026 work reflects later access to MI300X environments through
HotAisle and AMD Developer Cloud. These names describe the history and planned
demonstration context; they do not establish supported deployment targets for
the package.

Any presentation material must distinguish:

- historical observations from measurements validated against an identified
  current revision;
- functionality installed from `src/custom_canny_xpu` from separately
  maintained experimental scripts;
- demonstrated behavior from planned work; and
- Canny-only timing from reference comparison, transfers, and other timing
  boundaries.

If necessary, the demonstration may use bug-fixed CPU/GPU prototype scripts
that remain external to this package. Their use must be disclosed accurately;
it does not make them packaged functionality or current repository evidence.

## Scope and publication relationship

The preferred publication target remains the documented and tested refined CPU
operator. If this presentation is scheduled before that target can reasonably
be completed, the repository should instead become public no later than
presentation day as a clearly labeled experimental work in progress, subject
to the same publication-safety review. See
[`publication-readiness.md`](publication-readiness.md) for the complete trigger
and checklist.

Preparing the presentation does not require completion of the refined GPU
package, the broader GPU execution-motif study, the separate ROCm bootstrap
work, or every planned benchmark. The optional validation-cost investigation
described in [`operator-contract.md`](operator-contract.md#reference-disagreement)
is likewise neither presentation work nor a publication prerequisite.
