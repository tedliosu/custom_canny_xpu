# Historical alpha memory scaling

This note records a source inspection of the historical NumPy and CuPy alpha
prototypes that have not yet been integrated into this checkout. It is a guide
for preserving and safely benchmarking those artifacts, not evidence about the
current repository modules. Recheck these conclusions against the reviewed
files when roadmap item 4 integrates the alpha implementations.

## Evidence categories

- **Observed:** the alpha CuPy prototype approached out-of-memory behavior near
  the 25 MP class at `sigma` approximately 1.5 on an RTX 5000 Ada with 32 GB of
  VRAM. This is one configuration-specific observation, not a pixel-count
  limit.
- **Code-derived:** an allocation or logical-view relationship follows from
  the inspected source, but its actual peak memory and OOM point have not been
  measured systematically.
- **Speculative:** a possible future interface or execution mode is absent
  from the inspected source and therefore cannot establish current behavior.

## Dominant Gaussian relationship

The alpha Gaussian constructs a full two-dimensional kernel with

```text
K = 2 * ceil(4 * sigma) + 1
```

and applies it with `sliding_window_view` plus `tensordot`. For `P = H * W`
float32 output pixels, the logical window footprint is approximately

```text
P * K^2 * 4 bytes
```

The strided view does not itself allocate all those bytes. The generic
contraction may nevertheless require packing, workspace, or materialized
temporaries while other full-image arrays remain live. At approximately 24
million pixels, the logical footprints are:

| `sigma` | `K` | samples/pixel | logical float32 footprint |
| ---: | ---: | ---: | ---: |
| 1.5 | 13 | 169 | about 16.2 GB |
| 2.0 | 17 | 289 | about 27.7 GB |
| 2.5 | 21 | 441 | about 42.3 GB |
| 3.0 | 25 | 625 | about 60 GB |

Memory pressure therefore scales approximately with pixels times kernel area,
not pixels alone. Increasing `sigma` can cause OOM at unchanged or smaller
image dimensions. The refined separable formulation does not have this
pathological `P * K^2` logical-window scaling.

## Inspected parameters and retained state

| Parameter or source variable | Current alpha status | Memory effect | Applies to | Evidence |
| --- | --- | --- | --- | --- |
| Image height and width, selected indirectly by `IMG_FILE` | Hardcoded image path; dimensions come from the decoded image | Nearly every image-sized buffer is `O(P)`. Gaussian contraction exposure is `O(P * K^2)`. Padding adds `O((H + 2r)(W + 2r))`, where `r = ceil(4 * sigma)`. | CPU and GPU | Observed GPU OOM only for the configuration above; scaling is code-derived. |
| `GAUSS_SIGMA` / `sigma` | Hardcoded as 1.5 in both drivers; functions accept it as an argument | Sets `K` and `r`; kernel storage is `O(K^2)`, padded input grows with `r`, and the logical Gaussian window is `O(P * K^2)`. This is multiplicative with image size. | CPU and GPU | Code-derived; the near-25-MP observation was specifically around 1.5. |
| Working dtype | Drivers explicitly cast the decoded image to float32; not configurable | Raw input is retained alongside an `O(P * C)` float32 cast. Most dense working buffers scale approximately with bytes per element; CCL masks and labels have their own Boolean/integer dtypes. A future wider working dtype would increase pressure, but current drivers do not expose one. | CPU and GPU | Code-derived for current float32 storage; wider-dtype OOM behavior is unmeasured. |
| Decoded channel count `C` | Not a supported parameter; grayscale indexes the first three channels | The raw and float32 color images occupy `O(P * C)`. Extra decoded channels may remain resident even though only the first three are used. Channel count does not multiply the grayscale Gaussian window after conversion. | CPU and GPU | Code-derived. Treat broader channel support as undefined, not as a proposed CLI feature. |
| `DHYST_LOW` and `DHYST_HIGH` | Hardcoded; passed to the hysteresis helper | Dense weak/strong masks and a dense label image remain `O(P)`. Thresholds and image content affect foreground density, an `O(S)` strong-pixel label selection, component count `C_comp`, and an `O(C_comp)` keep table, where `S <= P` and `C_comp <= P`. | CPU and GPU | Code-derived possibility; no threshold-driven alpha OOM was recorded. It interacts with image content and dimensions, not directly with `sigma`. |
| Reference comparison (`DO_REF` on CPU; unconditional scikit-image and cuCIM work in the GPU driver) | CPU flag is hardcoded true; GPU reference paths are always executed | Custom outputs and several source/stage arrays remain live while reference casts, grayscale images, Canny outputs, and library-private temporaries are created. This adds `O(P)` visible storage plus unknown library workspace. The cuCIM path competes for VRAM with live custom arrays. | CPU and GPU; GPU driver also uses host RAM for scikit-image | Code-derived. No separate reference-induced OOM threshold was established. |
| NMS representation | Fixed materialization-heavy implementation | Coordinate grids, normalized gradients, four subpixel-coordinate fields, integer coordinate fields, gathered values, interpolation weights, neighbor results, and masks are all image-sized. Their count is fixed, so scaling is `O(P)` but with a large constant; the coordinate grids begin with an integer dtype. | CPU and GPU | Code-derived; not controlled by an independent current parameter. |
| Number of driver trials | Hardcoded as four CPU trials and six GPU trials | Results are reassigned rather than appended, so live storage is not multiplied by trial count. Some prior-trial arrays remain referenced until their variables are reassigned during the next trial, and CuPy's allocator may retain freed blocks and a high-water reservation. These details can affect peak availability and timing without constituting cumulative result retention. | CPU and GPU, with allocator-reservation concern primarily on GPU | Code-derived; allocator-specific peak behavior is not quantified. |

Padding mode is fixed to `edge` and Sobel's neighborhood is fixed at 3x3;
neither is an independent memory-size control in these prototypes. The
Gaussian radius is the only inspected neighborhood dimension that grows.

## Possibilities not established by the current alpha code

There is no batch dimension and no list retaining every trial's results. A
future batch parameter could multiply image-sized storage if an implementation
retained per-image intermediates, but that behavior is speculative and should
not be attributed to these prototypes. Similarly, benchmark repetitions do
not currently multiply the live working set, and no general channel-count,
dtype, padding-width, NMS-window, or label-dtype CLI exists.

When alpha artifacts are integrated, safe defaults or explicit guards may be
appropriate, but this note does not authorize new CLI options or behavior
changes. Preserve the alpha architecture and distinguish a measured OOM from a
logical footprint estimate and from an unimplemented future possibility.
