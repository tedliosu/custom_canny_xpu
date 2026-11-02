# Public operator contract

This document defines the intended initial public behavior of the refined
operators. It is an integration target, not a claim that the current
scaffolding already implements every rule below. The intended public operator
names are `canny` and `rgb_to_grayscale`, exported by a
`src/custom_canny_xpu/` package. A root-level `main.py` is an application/CLI
boundary that calls those operators rather than containing their
implementations.

## Canny operator

### Input

The Canny operator accepts exactly one two-dimensional, backend-native
grayscale array:

- a NumPy `ndarray` for CPU execution; or
- a CuPy `ndarray` for GPU execution.

The array must have shape `[H, W]`. Empty, color, channel-first, and batched
arrays are rejected. RGB conversion is a separate operator rather than an
implicit Canny stage. Minimum nonempty dimensions are backend-specific because
the initial CPU implementation retains the reviewed custom separable
convolution while the GPU implementation uses library filters with different
border machinery:

- For CPU, let `radius = int(4.0 * sigma + 0.5)`. Both `H` and `W` must be at
  least `2 * max(1, radius)`. The `1` accounts for the radius-one Sobel stage;
  the Gaussian term applies when its radius is larger. At the reviewed
  prototype setting `sigma=1.5`, the Gaussian radius is 6, so the minimum is
  12 by 12.
- For GPU, both `H` and `W` must be at least 1. CuPyX nearest-boundary Gaussian
  and Sobel filters support singleton axes, and the remaining corrected custom
  stages require a nonempty image but no larger fixed extent.

The CPU requirement is twice the active maximum radius, not the odd kernel
width and not the eight-output vertical register-block size. The custom
convolution's scalar tail handles widths below eight, while its paired border
regions assume each convolved dimension is at least twice the corresponding
radius. Validate this requirement from the same radius calculation used to
construct the Gaussian kernel rather than duplicating a fixed size. Backend
resolution and dimension validation occur before workspace allocation or
compiled/device execution where practical.

The stage-derived dimension constraints are:

- RGB-to-grayscale is a separate operator and accepts any nonempty height and
  width.
- The custom CPU Gaussian convolution requires each convolved dimension to be
  at least twice its radius; a zero-radius/skipped Gaussian adds no restriction.
  The custom radius-one CPU Sobel convolution requires at least 2 by 2.
- CuPyX Gaussian and Sobel filtering with nearest-boundary semantics supports
  singleton axes.
- Corrected CPU and GPU NMS supports every nonempty shape. Clamping each
  selected dominant-axis and diagonal integer neighbor to both bounds maps
  every neighbor to zero on a singleton axis.
- The specialized CPU/GPU CCA designs support a single pixel and impose no
  larger minimum, while CPU convolution independently imposes at least 2 by 2.
- Workspace allocation supports any positive 2D shape. Zero-sized dimensions
  are rejected before reductions, index arithmetic, or kernel launches.

The backend-wide minimum is the maximum of its active stage requirements, not
a dimension rule copied from another backend.

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
described below. The specialized CCA is the CPU hysteresis implementation, not
a public algorithm selector. The checked-in unfinished scaffold still exposes
an older `ccl` argument for SciPy/cc3d; remove that argument and its generic
labeling metadata during CCA integration before `canny` becomes public.

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
  semantics. Every selected NMS neighbor index must be clamped at both its
  lower and upper bound; older external prototypes' one-sided clamps were a
  known out-of-bounds defect. This contract deliberately retains clamped NMS
  borders rather than scikit-image v0.25.2's eroded-mask behavior, which zeros
  the outermost one-pixel ring. There is no public border-mode parameter
  initially.

NMS follows
[scikit-image v0.25.2's directional rule](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/feature/_canny_cy.pyx)
despite the upstream helper's `bilinear` name. For each side of a pixel, step
one integer position along the dominant gradient axis and choose the adjacent
diagonal position in the gradient direction. Linearly blend those two
magnitudes with diagonal weight

```text
min(abs(sobel_x), abs(sobel_y)) / max(abs(sobel_x), abs(sobel_y))
```

and retain the current magnitude when it is greater than or equal to both
interpolated neighbors. Compute the weight directly from gradient components;
do not construct absolute float32 subpixel coordinates or sample one Euclidean
unit along the normalized gradient. Handle a zero dominant component locally
without division warnings or nonfinite values.

The low threshold defines the candidate predicate on the NMS-suppressed
magnitude. An implementation may apply that predicate in NMS, as scikit-image
does, or fuse it into specialized CCA initialization; changing the stage
boundary must not change candidate membership. A zero magnitude is never a
candidate even when `low_threshold == 0`, so flat inputs remain all false.
Double hysteresis then keeps a candidate pixel exactly when its 8-connected
candidate component contains at least one high-threshold pixel. Dense
consecutive component IDs and a relabeled image are not required outputs. This
is a shared high-level contract, not a requirement that CPU and GPU use
identical CCA algorithms or synchronization.

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
- The CPU path must import and run without CuPy, cuCIM, or hipCIM installed and
  must not depend on cc3d.

A future CLI may perform explicit, separately timed transfers before invoking
the operator. Those transfers are not part of the device-resident Canny
boundary.

### Workspaces

The initial CPU integration groups its reusable scratch arrays behind a
`CpuWorkspace` dataclass created by `create_cpu_workspace`. The factory owns
the details of allocating compatible intermediate buffers for specialized CCA.
The exact buffer fields remain private implementation details and may preserve
deliberate storage reuse where stage lifetimes do not overlap.

`create_cpu_workspace` accepts any positive two-dimensional shape because it
does not receive `sigma` and allocation itself has no convolution-radius
minimum. Constructing a small workspace does not establish that the Canny
operator can use it for a particular call.

Every call first validates the image, parameters, backend, and required
dimensions. The subsequent workspace path is explicit:

- If `workspace is None`, the operator constructs a temporary compatible
  workspace only after that validation.
- If the caller supplies a workspace, the operator constructs nothing; it
  validates that workspace's backend, shape/capacity, dtype, and CCA metadata
  against the already validated call, then borrows it for execution.

The `workspace` argument is optional. After validating the public arguments
and resolving the backend, `canny` constructs an appropriate temporary
workspace when `workspace is None`; CPU construction uses
`create_cpu_workspace`. This keeps one buffer-management path rather than a
second non-workspace execution path. A temporary workspace lives only for that
call and is neither returned nor attached to the result.

The current unfinished scaffold's `ccl`/workspace mutual-exclusion behavior is
tested only as a record of transitional code. Remove the `ccl` selector,
generic labeling callable/dtype metadata, and obsolete label buffers when the
specialized CCA workspace is integrated; do not carry them into the exported
contract.

Calls that omit a workspace include temporary allocation costs. Calls supplied
with a compatible reusable workspace form the preallocated/steady-state timing
boundary. Benchmarks must preserve that distinction.

The durable workspace ownership rules are:

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

The initial `CpuWorkspace` establishes the ownership boundary without freezing
each buffer field as permanent public API. A later GPU integration should use
an equivalent backend-specific workspace rather than put host and device
arrays into one object. Do not repeatedly reorganize either workspace merely
for tidiness after the ownership and lifetime rules are established.

The current generic-labeling scaffold's `edges_kept_bool_mask` policy is
historical implementation state. Specialized CCA carries component strong
state in its own representation and must not preserve that temporary merely
for compatibility with the older workspace layout.

### Failures

Library APIs use exceptions. A future process-level CLI may translate them
into exit statuses.

Validate caller-controlled invariants once at the narrowest public or object
construction boundary that owns them. `canny` owns validation of its image,
parameters, output, backend/residency selection, and workspace compatibility;
`rgb_to_grayscale` owns its separate input/output contract; and workspace
constructors own their construction arguments. Private numerical stages and
compiled kernels may rely on those established invariants and should not
duplicate the same full-array checks.

- `TypeError`: unsupported array implementation or dtype, and backend-residency
  mismatches.
- `ValueError`: invalid shape, non-finite or out-of-range image values, invalid
  sigma or thresholds, or an incompatible output/workspace shape.
- `RuntimeError`: an explicitly requested backend or optional implementation
  is unavailable.

Native allocation and backend-execution failures, including out-of-memory
errors, should retain their meaningful underlying exception rather than cause
an automatic backend fallback.

A minimum-dimension `ValueError` must state the received shape, selected
backend, actual required height and width, and the stage or active radius that
imposes the restriction. Do not report a backend-independent 3-by-3 rule.

## RGB-to-grayscale operator

RGB conversion is a separate composable public operator. It accepts exactly a
nonempty backend-native `[H, W, 3]` array with `H >= 1` and `W >= 1`, and never
transfers or mutates it.

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

The result is a backend-native `[H, W]` float32 array with the same nonempty
spatial dimensions as the RGB input. The grayscale operator does not enforce
Canny-specific dimension requirements. If a caller later passes that result to
Canny, the Canny operator independently validates its backend- and
sigma-dependent contract. Do not add an unconditional full-image clipping pass
merely as a precaution. Tests should first establish whether backend rounding
can escape `[0.0, 1.0]`; add a correctness measure only if evidence requires
it.

## CPU hysteresis integration transition

The current refined direction is a specialized row-partitioned CPU CCA with
parallel local unions, explicit seam reconciliation, component strong-pixel
state, and direct Boolean emission. This replaces the earlier plan to make
generic SciPy CCL the default and prefer an optional nogil/preallocated cc3d
binding for the optimized comparison.

The checked-in workspace and dispatch scaffold still records `"scipy"` and
`"cc3d"` strategies. That machinery is superseded: remove the public selector,
generic label buffers/callables, optional cc3d path, and related dependency
surface while integrating specialized CCA. Preserve the older code's evidence
and provenance in documentation rather than retaining it as a runtime fallback.

## CPU threading

The unbatched CPU operator uses the full logical-core worker set available to
the process and numerical runtime for stages that expose CPU parallelism. It
does not expose thread-count tuning, select a different algorithm from the
detected thread count, or retain a single-thread execution fallback. Operating
system affinity, container limits, and numerical-runtime configuration define
the actually available worker set. Frame-level worker partitioning and nested
oversubscription are deferred to separately scoped batch/video work.

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
may map it to one CUDA CUB or ROCm rocPRIM reduction call with reusable
temporary storage. The separate execution-motif study compares GPU
computational patterns by properties such as arithmetic intensity and
memory-access regularity; it is not part of the Canny operator or a prerequisite
for this repository. Do not retain the semantically unnecessary per-image
magnitude normalization merely to preserve a reduction, and do not include
reference-comparison time in Canny-only measurements.

Measuring the cost of correctness/reference validation itself is an
intentionally deferred, separately scoped investigation. A future study may
compare CPU and GPU validation cost while separating device-resident
computation from transfer and reference-setup overhead, but only when a
concrete validation-cost question and comparable implementations justify the
work. It is not part of the core benchmark contract, an implementation
requirement, or a prerequisite for publication or the 2026 ROCm presentation.
The presentation was originally scheduled for November 3; current scheduling
status is **presentation rescheduled for November 17, 2026**.

## Deferred features

The initial contract does not include masks, quantile thresholds, alternative
border modes, implicit RGB handling, `float64` Canny input, integer Canny
input, gradient-norm or Sobel-aperture controls, batches, video, or implicit
host/device transfers. Adding one requires an explicit contract and scope
decision rather than opportunistic implementation.
