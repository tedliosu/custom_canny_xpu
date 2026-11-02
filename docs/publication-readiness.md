# Publication readiness

This repository is intended to become public as an experimental open-source
project. Public availability is not the same promise as a stable packaged
release or a production-supported software product, and it does not require
completion of the full README roadmap.

This document defines the approved publication paths, records the satisfied
presentation-driven public-visibility gate, and preserves a bounded review
checklist. It does not declare a stable packaged release or authorize
destructive cleanup.

## Release levels

- A **public experimental repository** makes source, history, documentation,
  and limitations available for inspection and learning. Work may remain
  planned or experimental when that status is accurate and visible.
- A **stable packaged release** additionally implies a versioned installation
  path, a deliberately supported API and dependency range, and stronger
  compatibility expectations. Public visibility alone does not make those
  commitments.
- A **production-supported product** implies operational support, maintenance,
  security, reliability, and deployment commitments that this solo learning
  project does not make.

The README must describe the level actually offered rather than relying on a
repository's visibility setting to communicate maturity.

## Project terminology and related work

- **Refined CPU/GPU implementations** are the maintainable, workspace-aware,
  backend-appropriate operator direction being integrated into
  `src/custom_canny_xpu`. Prototype observations do not prove that a stage is
  already present in the package; the current checkout and README status are
  authoritative.
- **Historical alpha implementations** are older, high-level vectorized and
  materialization-heavy learning artifacts retained to explain architectural
  evolution. They are not the production fallback or the model for refined
  code.
- **Operator-class proxy** means the compact copy/label/NMS-lite benchmark that
  historically sampled streaming, regular-local, and generic-labeling work. It
  is not a Canny implementation, current harness candidate, or exact predictor.
- **Streaming-bandwidth context** means an external, minimally patched
  BabelStream v5.0 run used only to contextualize same-instance CPU/GPU
  streaming behavior. It is not a direct Canny predictor, a percentage-of-peak
  result, or a replacement for controlled Canny evidence. The earlier
  TheBandwidthBenchmark HIP smoke failed with an illegal memory access and is
  invalid as HIP bandwidth or Canny evidence.
- **Historical cc3d binding** means the superseded custom binding to the
  external LGPL `connected-components-3d` project. Specialized CPU CCA removes
  the need to integrate or depend on it; its old experiments and provenance
  remain useful history.
- **ROCm cloud-bootstrap work** is separate environment bring-up and
  CUDA/HIP-compatibility experimentation. It is not contained in, required by,
  or a support claim of this repository.
- **GPU execution-motif study** is separate work comparing representative GPU
  computational patterns across execution environments, including differences
  in arithmetic intensity and memory-access regularity. Canny supplies
  application motivation for some patterns but does not absorb that project.

The planned device-resident Boolean disagreement count is one bridge between
the projects. It is a low-arithmetic-intensity, regular-access
transform-reduction that may later be expressed through one CUDA CUB or ROCm
rocPRIM library primitive. It belongs to reference comparison after Canny
output exists and must remain outside Canny-only timing. Completing that
experiment or the wider motif taxonomy is not a condition for publishing this
repository.

## Preferred CPU-first milestone

The approved preferred normal target is a working, documented, and reasonably
tested refined CPU Canny implementation. It would give an external reader a
concrete operator to install, execute, inspect, and learn from without requiring
a GPU.

Its minimum criteria are:

1. The documented refined CPU Canny API works under the stated contract.
2. Basic CPU dependency installation and an execution example have been
   verified from the repository instructions.
3. Representative correctness, boundary, reuse, and expected-failure tests
   pass in the stated environment.
4. Implemented, experimental, historical, and planned functionality is clearly
   distinguished.
5. Current-facing performance and correctness claims are supported under the
   benchmark-evidence lifecycle or explicitly qualified.
6. The README and linked documents are understandable without private files,
   repositories, conversations, or personal terminology.
7. `LICENSE`, `THIRD_PARTY_NOTICES.md`, and material third-party and
   AI-assistance provenance have been reviewed against the content being
   published.
8. Tracked files and Git history have been reviewed for credentials, secrets,
   unintended private information, machine-specific paths, generated artifacts,
   and other material not intended for publication.

The current checkout remains earlier than this milestone: the complete
CPU operator, supported dependency metadata, and verified installation/use
instructions are not yet present.

## Presentation publication gate

The Ohio State Open Source Club ROCm presentation was approved around October
26, 2026 and was originally scheduled for November 3, 2026. The repository was
made public as a clearly labeled experimental work in progress before that
date, satisfying the presentation-day gate while its CPU-first milestone
remained incomplete. Current scheduling status: **presentation postponed; new
date to be announced**. The refined CPU-first milestone remains the preferred
development target.

Presentation preparation now lives in a separate local-only repository and
dedicated session. Do not move presentation-specific wrappers, orchestration,
or deadline-driven prototype changes into `src/custom_canny_xpu`. Any external
demonstration must remain accurately distinguished from installed package
functionality and from evidence reproduced from committed canonical code.

The original publication deadline did not waive the review below. A material
license, attribution, provenance, credential, privacy, or other
publication-safety blocker must be escalated to the maintainer. Do not publish
unsafe material or perform destructive cleanup without separate approval. The
presentation's purpose, audience, environment history, and scope boundaries
are recorded in
[`proposed-rocm-presentation.md`](proposed-rocm-presentation.md).

## Non-blocking future work

Neither the preferred CPU-first readiness level nor the completed WIP
publication path requires:

- the optional refined GPU implementation;
- complete CUDA or ROCm validation;
- historical alpha integration;
- a complete benchmark harness or every content/connectivity class;
- batch or video processing;
- completion of the separate GPU execution-motif study;
- every proposed proxy refinement;
- exhaustive performance optimization or a definitive cross-hardware report;
  or
- completion of every README roadmap item.

These items may remain planned or experimental. Their deferral does not excuse
an unresolved license, attribution, provenance, credential, or publication
safety issue in material that is actually included.

## Pre-publication review

Before changing repository visibility or announcing a release:

1. Verify README capability, maturity, dependency, installation, and execution
   claims against the checkout being published.
2. Review `LICENSE`, `THIRD_PARTY_NOTICES.md`, and provenance disclosures for
   every copied, adapted, vendored, generated, or externally dependent item
   actually present.
3. Confirm optional dependencies and unavailable backends fail as documented
   without preventing the supported CPU path from importing and running.
4. Classify current-facing measurements under the benchmark-evidence lifecycle
   and remove or qualify unsupported claims.
5. Inspect tracked files, ignored/untracked artifacts, repository configuration,
   and the full Git history for credentials, private paths, unpublished data,
   personal infrastructure, large generated files, and unintended artifacts.
6. Verify that references to external prototypes or separate projects contain
   enough context and do not require access to private material to understand
   this repository's behavior or claims.
7. Record unresolved publication blockers and obtain maintainer acceptance of
   the chosen public milestone.

Removing a file in a new commit does not erase it from earlier Git history. If
a review discovers sensitive historical content, stop and assess the exact
objects and exposure. Any history rewrite, object filtering, force push, or
other destructive cleanup requires separate maintainer review and explicit
approval. Re-run the publication audit after an approved cleanup rather than
assuming the rewrite was complete.

## Administrative files

Public visibility alone does not require `CONTRIBUTING.md` or `SECURITY.md`.
Add them when participation, contribution workflow, service exposure, or the
project's risk/support model gives them a real purpose. Until then, accurate
README guidance, license/provenance records, and a bounded publication review
are more useful than generic boilerplate.
