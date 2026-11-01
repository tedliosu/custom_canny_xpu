# Benchmark evidence lifecycle

Benchmark results in this project are evidence for a particular implementation
generation, workload, environment, and experimental contract. Performance and
correctness measurements do not automatically remain representative when one
of those inputs changes. This document defines a lightweight review process
that preserves useful engineering history without allowing old measurements to
remain unsupported current claims.

This policy does not require a comprehensive benchmark rerun after every code
change. Revalidation is selective and proportional to the dependencies of the
affected claim.

## Evidence identity

A result should retain enough context to determine what it measured. Depending
on the claim, that context includes:

- the source revision or identifiable implementation generation;
- operator semantics, algorithms, and intermediate representations;
- image dimensions, content, dtype, sigma, thresholds, and other relevant
  workload parameters;
- hardware, compiler, runtime, dependency, driver, and backend configuration;
- allocation and workspace policy, synchronization, warmup, and timing
  boundaries;
- compilation and cache state when generated-code identity or call-target
  resolution can affect the executed kernel; and
- correctness/reference configuration when disagreement is reported.

Not every report needs every field. Record the fields that can materially
change the interpretation of the result, following the README benchmarking
contract.

## Evidence status

Use these categories without conflating them:

- **Current evidence:** the measurement remains applicable to the current
  implementation and its explicitly stated experimental configuration.
- **Historical evidence:** the measurement was valid for an earlier
  implementation, environment, or methodology but does not necessarily
  characterize the current implementation. Supersession alone does not make a
  result erroneous.
- **Pending revalidation:** a change has made the measurement's or conclusion's
  applicability to the current implementation uncertain, or current evidence
  is known to be required before making the claim again.
- **Invalid evidence:** a confirmed defect or experimental error undermines the
  result's original conclusion. Preserve useful records of what went wrong,
  but do not present the result as sound historical evidence.

An entire document need not have one status. Raw measurements can remain valid
historical evidence while a derived comparison becomes pending or invalid. Add
an explicit status qualification where readers could otherwise mistake an old
or uncertain result for current evidence; do not mechanically label every
number when the enclosing historical context is already unambiguous.

## Review triggers

Review potentially dependent evidence when a material change includes, but is
not limited to:

- operator semantics or numerical behavior;
- algorithms, intermediate representations, or execution strategies;
- CPU or GPU optimizations that alter stage timing or workload composition;
- allocation, workspace reuse, synchronization, or timing boundaries;
- image dimensions, content, dtype, thresholds, or other relevant parameters;
- compilers, runtimes, dependencies, hardware, or backend configuration; or
- a newly discovered methodological defect, incorrect assumption, or
  uncontrolled variable.

A trigger starts a dependency review, not an automatic invalidation of every
recorded experiment.

## Selective revalidation procedure

When a change may affect existing evidence:

1. Identify the implementation or methodology change and the experimental
   contract it alters.
2. Trace which raw measurements and derived conclusions depend on that altered
   behavior.
3. Mark affected current-facing claims as pending revalidation, or invalid if a
   confirmed defect undermined the original result.
4. Preserve sound historical results with their original context rather than
   silently overwriting them.
5. Regenerate only the measurements necessary to establish the desired current
   evidence.
6. Recompute every dependent speedup, stage fraction, proxy estimate, or other
   derived quantity from compatible measurements.
7. Update current-facing documentation and retain links to useful historical
   evidence.

Independent experiments retain their own status. For example, removing
per-image magnitude normalization does not inherently invalidate an isolated
Gaussian-footprint experiment. A changed CPU filtering implementation can make
a proxy-to-Canny interpretation stale without changing the raw measurements
from the original proxy script.

Controlled Canny measurements are the primary performance evidence. The
reviewed TheBandwidthBenchmark HIP smoke failed with an illegal memory access,
so it produced no valid HIP bandwidth result and cannot support a Canny proxy
interpretation. Preserve the failure only as diagnostic history.

The accepted presentation instead uses minimally patched BabelStream v5.0
results as approximate same-instance streaming-bandwidth context. Those raw
measurements retain their own evidence identity, but their CPU/GPU ratios are
not a direct Canny prediction or percentage-of-advertised-peak result. They do
not model specialized CCA topology and parent chasing, CPU seams, GPU atomics
and contention, NMS gathers, stencil reuse, or synchronization. Integrating
the presentation patch or runner remains separately scoped future work.

Any descriptive bandwidth-to-Canny comparison still depends on the matched
operator generation, dimensions, threading, compiler/runtime, hardware
configuration, and selected kernel mix. A CCA or stage-composition change can
invalidate that interpretation without invalidating the underlying valid
streaming measurements themselves.

The accepted one-shot scikit-image and hipCIM timings have correctness-reference
status rather than controlled performance-baseline status. A future speedup
claim requires warmed, repeated, boundary-matched custom and library runs with
the same semantic and workload contract, explicit allocation/compilation/
transfer/residency boundaries, and paired correctness evidence.

## Absolute-threshold and prototype-generation transition

Older external refined CPU and GPU prototypes divided gradient magnitude by the
per-image maximum and sampled NMS neighbors using a different
normalized-gradient coordinate formulation. The Oct. 22 external references
remove that normalization, use scikit-image-style major-axis
linear-interpolation NMS, and use specialized CCA hysteresis, but those
numerical stages are not integrated into this checkout. Moving between these
generations changes more than the cost of one reduction.

It can affect:

- CPU and GPU stage timings;
- weak- and strong-candidate populations;
- candidate-connectivity topology and hysteresis cost;
- final edge maps and reference disagreement;
- overall CPU/GPU speedup; and
- derived stage fractions and proxy interpretations.

Measurements from the normalized prototypes can remain valid historical
evidence for those generations. Their raw NMS timings likewise describe the
superseded coordinate formulation, while any claim that it represented the
current scikit-image-conformant NMS semantics is invalid. Measurements from the
Oct. 22 references are also external-prototype evidence until reproduced from
identified integrated code. Neither may be presented as current checkout
evidence, and future timing must not be estimated as the old timing minus the
maximum-reduction cost. If CPU and GPU temporarily implement different semantic
generations, their measurements must not be combined into a current
cross-backend speedup until comparability is established.

## Generated-code and cache-state transition

For affected Numba `parallel=True` kernels, fresh JIT compilation and cache
loading can resolve calls to different physical parfor/gufunc copies. Identical
source and output therefore do not guarantee identical executed machine code or
steady-state timing. A cache-loaded run is not automatically the corrected
version of a fresh run, nor vice versa.

When this mechanism may affect a conclusion, preserve the old measurement as
evidence for its recorded regime if that regime is known, and mark cross-regime
or unknown-state comparisons pending revalidation. Reproduce algorithm
comparisons with matched cache/JIT state and process boundaries. Experimental
call-target redirection or symbol-internalization results demonstrate a
mechanism but are not current operator evidence or an approved default patch.

## Reporting and publication

Do not combine incompatible semantic, implementation, workload, or timing
generations into one current comparison. When current numbers are unavailable,
state that the claim is pending revalidation instead of retaining an obsolete
number without qualification.

Historical evidence remains useful for explaining optimization decisions,
bottleneck migration, unsuccessful hypotheses, and the evolution of the
operator. Publication does not require regenerating every historical
experiment. It requires current-facing claims to be supported by compatible
current evidence and older results to remain honestly contextualized.
