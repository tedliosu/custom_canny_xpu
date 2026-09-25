# Public operator contract

This document defines the intended initial public behavior of the refined
operators. It is an integration target, not a claim that the current
scaffolding already implements every rule below. Package layout and final
public names remain roadmap decisions.

## Canny operator

### Input

The Canny operator accepts exactly one two-dimensional, backend-native
grayscale array:

- a NumPy `ndarray` for CPU execution; or
- a CuPy `ndarray` for GPU execution.

The array must have shape `[H, W]`, with `H >= 3` and `W >= 3`. Empty, color,
channel-first, and batched arrays are rejected. RGB conversion is a separate
operator rather than an implicit Canny stage.

The only accepted dtype is `float32`. Every value must be finite and in the
closed interval `[0.0, 1.0]`. The public operator does not silently narrow
`float64`, normalize integer input, or transfer an array between host and
device. In particular, reference comparisons must give the custom and
reference implementations the same `float32` grayscale input rather than
compare an explicitly narrowed custom input with a `float64` reference input.
Any future trusted internal entry point that avoids repeated value validation
must remain distinct from the checked public API, and its use must be disclosed
in performance boundaries.

### Parameters and numerical semantics

The initial public parameters are `sigma`, `low_threshold`, and
`high_threshold`, plus backend, workspace, and output-buffer selection as
described below.

- `sigma` is a finite scalar greater than or equal to zero. Zero skips
  Gaussian smoothing. Gaussian truncation is fixed at `4.0` rather than
  exposed as another parameter.
- Thresholds are finite scalars satisfying
  `0.0 <= low_threshold <= high_threshold <= 1.0`. The initial defaults are
  `0.1` and `0.2`.
- Thresholds are absolute values applied to the NMS-suppressed Sobel magnitude.
  The refined implementation must not divide magnitudes by the maximum
  magnitude of each image. Requiring normalized `float32` input gives the
  thresholds a stable image-value convention, but does not imply that Sobel
  magnitudes themselves are bounded by one.
- Quantile thresholds are not part of the initial contract. Quantiles are
  data-dependent percentiles of the edge-magnitude distribution and are
  independent of whether integer input is accepted; they may be considered
  later only through a separate scope decision.
- Gaussian, Sobel, and NMS neighborhood reads use nearest/clamped-boundary
  semantics. There is no public border-mode parameter initially.

The operator uses interpolation-based non-maximum suppression followed by the
scikit-image-style CCL decomposition of double hysteresis: label the
low-threshold mask, identify components containing at least one high-threshold
pixel, and keep those components. This is a shared high-level contract, not a
requirement that CPU and GPU use identical CCL algorithms.

Flat inputs follow the ordinary operator path rather than a separate
image-level fast path. Zero gradient magnitude must be handled locally without
division warnings, NaNs, or infinities, and a flat input must produce an
all-false result.

### Output and mutation

The result is a backend-native Boolean array of shape `[H, W]`. Conversion to
`uint8` values such as 0 and 255 belongs to image serialization or another
explicit operation outside Canny.

The operator never mutates its input. Returned output is not implicit
workspace storage and remains valid when a workspace is reused. When an
explicit output buffer is supported, it is caller-owned and must be writable,
Boolean, backend-matched, and shape-matched. Output buffers and temporary
workspaces remain separate concepts.

### Backend selection and transfers

The public backend choices are `"auto"`, `"cpu"`, and `"gpu"`.

- `"auto"` follows array residency: NumPy selects CPU and CuPy selects GPU.
- Backend selection never changes array residency.
- Explicit CPU execution with a CuPy input, or explicit GPU execution with a
  NumPy input, fails instead of copying the array.
- An explicit unavailable GPU request fails clearly and never silently retries
  on CPU.
- The CPU path must import and run without CuPy, cuCIM, hipCIM, or the optional
  cc3d binding installed.

A future CLI may perform explicit, separately timed transfers before invoking
the operator. Those transfers are not part of the device-resident Canny
boundary.

### Workspaces

A caller may eventually supply a reusable workspace. The durable ownership
rules are:

- the caller owns the workspace and the operator borrows it;
- the operator may overwrite all temporary contents;
- a workspace is backend-specific and shape- or capacity-constrained;
- one workspace is not safe for concurrent calls;
- input and returned output do not alias workspace storage; and
- under the operator-controlled lifecycle, every scratch value read by a call
  is initialized as needed so reuse by call `n + 1` does not depend on results
  or transient state left by call `n`.

Workspace contents are private implementation state even if Python-level
introspection makes an underlying array reachable. Callers must not replace,
resize, retype, or inject arbitrary data into those buffers. Behavior after
such mutation is undefined by the operator contract. The implementation may
perform inexpensive structural compatibility checks for backend, shape,
capacity, and dtype, but it must not add full-buffer integrity scans, defensive
clearing, checksums, or reconstruction solely to protect against caller
tampering. Those passes would undermine the performance purpose of workspace
reuse. This does not relax the normal first-use and cross-call initialization
requirements above.

These rules deliberately do not freeze a public workspace class or its buffer
layout. Initial correctness integration may retain the prototypes' individual
temporary buffers. A later roadmap item will group them behind explicit,
opaque CPU and GPU workspace objects without changing output ownership.

### Failures

Library APIs use exceptions. A future process-level CLI may translate them
into exit statuses.

- `TypeError`: unsupported array implementation or dtype, and backend-residency
  mismatches.
- `ValueError`: invalid shape, non-finite or out-of-range image values, invalid
  sigma or thresholds, or an incompatible output/workspace shape.
- `RuntimeError`: an explicitly requested backend or optional implementation
  is unavailable.

Native allocation and backend-execution failures, including out-of-memory
errors, should retain their meaningful underlying exception rather than cause
an automatic backend fallback.

## RGB-to-grayscale operator

RGB conversion is a separate composable public operator. It accepts exactly a
backend-native `[H, W, 3]` array and never transfers or mutates it.

The initial specialized paths are:

- `uint8` input, using float32 weights
  `[0.2125 / 255.0, 0.7154 / 255.0, 0.0721 / 255.0]`; and
- finite `float32` input in `[0.0, 1.0]`, using float32 weights
  `[0.2125, 0.7154, 0.0721]`.

Both paths use backend matrix multiplication with safe casting into a
caller-provided or newly allocated `[H, W]` float32 output. This preserves the
refined prototypes' simple matrix-vector formulation and avoids the alpha
implementation's separately materialized channel products. `float64`, other
integer dtypes, RGBA, channel-first, and batched inputs are initially rejected.

The grayscale result must be valid input for the Canny operator. Do not add an
unconditional full-image clipping pass merely as a precaution. Tests should
first establish whether backend rounding can escape `[0.0, 1.0]`; add a
correctness measure only if evidence requires it.

## CPU CCL selection

SciPy CCL remains the supported default CPU direction. The optional custom
nogil/preallocated cc3d binding is experimental and must be selected
explicitly; merely importing it must not change execution behavior.

When the binding is available, the primary optimized CPU-versus-GPU benchmark
should prefer the cc3d CPU path so that the comparison uses specialized
hysteresis implementations on both sides. The SciPy result should remain a
clearly labeled supplementary or appendix result. This improves the
engineering comparison but does not make it architecture-pure: the CPU and GPU
still use different CCL algorithms and expose different parallelism.

## Reference disagreement

Reference comparison is orthogonal to Canny execution. On GPU, count differing
pixels while both Boolean outputs remain device-resident and transfer only the
resulting scalar after the required synchronization. Record disagreement count
and rate, image identity, parameters, and reference configuration.

If a reference implementation produces a host result, transfer that Boolean
result to the device outside the Canny-only timing boundary before performing
the disagreement count. Report that setup transfer separately when it is
included in a broader reference-comparison measurement.

The selected lower-level primitive opportunity is this on-device disagreement
count, conceptually a fused transform-reduction of
`custom_output[i] != reference_output[i]`. A later execution-motif experiment
may map it to one CUB or rocPRIM reduction call with reusable temporary
storage. Do not retain the semantically unnecessary per-image magnitude
normalization merely to preserve a reduction, and do not include reference
comparison time in Canny-only measurements.

## Deferred features

The initial contract does not include masks, quantile thresholds, alternative
border modes, implicit RGB handling, `float64` Canny input, integer Canny
input, gradient-norm or Sobel-aperture controls, batches, video, or implicit
host/device transfers. Adding one requires an explicit contract and scope
decision rather than opportunistic implementation.
