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
  boundaries; and
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

## Absolute-threshold transition

The external refined CPU and GPU prototypes still divide gradient magnitude by
the per-image maximum. The reviewed operator contract removes that
normalization and applies absolute thresholds to NMS-suppressed Sobel
magnitudes. Integration therefore changes more than the cost of one reduction.
It can affect:

- CPU and GPU stage timings;
- weak- and strong-candidate populations;
- CCL input topology and hysteresis cost;
- final edge maps and reference disagreement;
- overall CPU/GPU speedup; and
- derived stage fractions and proxy interpretations.

Measurements from the normalized prototypes can remain valid historical
evidence for those generations. They must not be presented as current evidence
for the refined absolute-threshold operator, and its future timing must not be
estimated as the old timing minus the maximum-reduction cost. If CPU and GPU
temporarily implement different semantic generations, their measurements must
not be combined into a current cross-backend speedup until comparability is
established.

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
