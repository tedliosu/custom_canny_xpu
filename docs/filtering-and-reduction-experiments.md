# Filtering and reduction experiments

These notes preserve informal experiments consolidated on 2026-09-24. The CPU
suspect-operator proxy originated in May 2026; the reduction and
Gaussian-footprint work came later. The benchmark scripts and complete refined
operators were external prototypes and are not implementation evidence for
this checkout. Regenerate publishable results from identified committed code
under the README's benchmarking contract.

The reviewed public contract now removes the prototypes' per-image magnitude
normalization in favor of absolute thresholds. References below to maximum
reduction and the historical gradient-stage timing describe those external
drafts and their recorded experiments, not the intended integrated operator.

Calculated GB/s below is input-payload or logical effective throughput, not a
measurement of literal physical DRAM transactions. Timings establish observed
behavior in the recorded configurations; they do not by themselves prove a
cache, instruction-throughput, occupancy, register-pressure, or memory-system
bottleneck.

## Refined prototype filtering topology

The reviewed CPU prototype uses a generic Numba separable convolution with
`parallel=True`, `nogil=True`, and `fastmath=True`. It performs horizontal and
vertical 1D passes with explicit nearest-boundary indexing and reusable
intermediate/output buffers. It does not perform an explicit full-frame
`np.pad` for blur or gradients.

For Gaussian blur, `truncate=4.0` and

```text
radius = int(sigma * truncate + 0.5)
width = 2 * radius + 1
```

The Canny prototype uses `sigma=1.5`, so its kernel has radius 6 and width 13.
The same 1D Gaussian kernel is applied horizontally and vertically.

CPU Sobel X uses separable factors `[-1, 0, 1]` and `[1, 2, 1]`; Sobel Y uses
the transposed factorization. The reported CPU gradient stage therefore
contains two separable Sobel operations, a separate parallel Numba
`sqrt(x*x + y*y)`-style magnitude pass, and a NumPy full-array maximum
reduction.

The reviewed GPU prototype uses:

- `cupyx.scipy.ndimage.gaussian_filter` with `mode="nearest"`, `truncate=4.0`,
  and preallocated output, followed by a CuPy clip;
- `cupyx.scipy.ndimage.sobel` for X and Y; and
- `cupy.hypot` followed by `cupy.ndarray.max`.

It does not perform an explicit full-frame `cupy.pad` for these stages, and its
Gaussian and Sobel operations are not hand-written repository kernels. Source
inspection, including the tagged CuPy 14.0.1 implementation, found generated
elementwise filtering with explicit/manual footprint indexing. That is
consistent with profiler kernel names, but it does not establish the exact
hardware bottleneck. Generated machinery may still benefit from filter-shape
specialization, compiler-visible loops and indexing, skipped zero footprint
entries, and ordinary compiler/runtime optimization. If this source is linked
later, use the applicable tagged version rather than GitHub's moving `main`
branch.

## NumPy versus CuPy maximum reduction

The FP32 reduction microbenchmark generated one seeded NumPy array and copied
the same data to GPU before timing. It ran configurable warmups and repeats,
synchronized CuPy, reported mean/median/best latency, and checked that both
backends returned the same maximum. The recorded CPU was a Ryzen 5700X with
dual-channel DDR4-3200; the GPU was an RTX 5000 Ada.

ChatGPT generated this benchmark in one pass from the maintainer's request.

| FP32 elements | Input size | NumPy median | NumPy payload rate | CuPy median | CuPy payload rate | Rate ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| about 23.3 M | about 93 MB | 2.699 ms | 34.53 GB/s | 0.191 ms | 487.20 GB/s | 14.11x |
| about 122.99 M | about 492 MB | 14.320 ms | 34.36 GB/s | 0.940 ms | 523.37 GB/s | 15.23x |

The results matched numerically. NumPy sustained about 34 GB/s, versus a prior
roughly 39 GB/s simple system-memory streaming observation. This is consistent
with a reasonably efficient streaming reduction rather than an unexpectedly
pathological NumPy primitive and did not reproduce the previously observed
slow NumPy `hypot` path. That path was later replaced in the refined CPU
prototype by the parallel compiled magnitude operation described above. A
full-frame maximum still costs a complete scan. If several reductions become
important, reducing or fusing full-frame passes is a better first hypothesis
than replacing `np.max()` on the assumption that it is inherently broken.

The 14–15x GPU/CPU ratio reflects the complete reduction executions and memory
systems. It is not evidence that NumPy is poorly implemented, nor does it alone
explain the roughly 20–23x refined gradient-stage ratio.

## Exploratory CPU suspect operations

This May 2026 proxy began with a maintainer prompt hypothesizing three CPU
bottlenecks: serial `np.pad`, a row-parallel vertical 1D convolution, and a
possibly vectorized but serial row-oriented BBDT-like CCL scan. ChatGPT then
generated the complete microbenchmark code, including separate operator
timings and the requested minimum/maximum comparison across the candidates.
The resulting 122.99 MP-shape experiment reported:

| Operation | Representative median | Effective-throughput model |
| --- | ---: | ---: |
| `np.pad` | about 44–45 ms | about 22 GB/s, read plus write payload |
| 3-tap vertical Numba convolution | about 47 ms | about 42 GB/s logical |
| 5-tap vertical Numba convolution | about 63 ms | about 47 GB/s logical |
| 13-tap vertical Numba convolution | about 135 ms | about 51 GB/s logical |
| BBDT-like serial row scan | about 132 ms | about 5.6 GB/s effective |

The convolution model counts logical tap reads, so cache reuse can make its
reported rate exceed physical DRAM bandwidth. The serial BBDT-like scan did
not behave like a simple bandwidth-saturated stream. These were candidate
operations selected before the CPU operator's main remaining costs were fully
understood, not a complete bottleneck diagnosis. The then-evolving Canny code
also incurred large intermediate-allocation churn, a slow `np.hypot` path, and
extra padding passes. Later work introduced reusable preallocated buffers, a
custom parallel compiled magnitude operation, and direct boundary handling
that removed the padding passes.

Consequently, the `np.pad` result remains valid only as an exploratory
operation measurement. The reviewed refined prototype no longer contains
explicit full-frame pad passes for blur or Sobel, so this timing must not be
used to explain its current stage results. The generated benchmark's
provenance and its hypothesis-driven operator selection should accompany any
reuse of these figures.

## Gaussian-footprint benchmark

The benchmark compared the same basic preallocated, parallel Numba separable
CPU convolution with preallocated CuPyX `gaussian_filter` using FP32,
`mode="nearest"`, and `truncate=4.0`. GPU events bounded GPU work. It reported
latency, MPix/s, and CPU/GPU ratios rather than a misleading logical-tap GB/s.
ChatGPT generated this benchmark in one pass from the maintainer's request.

| `sigma` | Radius | 1D width |
| ---: | ---: | ---: |
| 0.5 | 2 | 5 |
| 1.0 | 4 | 9 |
| 1.5 | 6 | 13 |
| 2.0 | 8 | 17 |
| 3.0 | 12 | 25 |

Across all recorded cases, CPU/GPU output differences were tiny: maximum
absolute differences were roughly 3e-7 to 4.2e-7 and mean absolute differences
roughly 4e-8 to 4.9e-8. The timing gap was not obtained through a meaningfully
lower-accuracy GPU approximation.

### RTX 5000 Ada desktop

The desktop record used an RTX 5000 Ada, 16 Numba threads, and FP32. The
experiment record did not independently pin every software version for this
particular sweep; do not silently borrow a version from another run.

| Shape | Pixels | Width | CPU median | GPU median | CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920x1280 | 2.46 MP | 5 | 2.082 ms | 0.200 ms | 10.40x |
| 1920x1280 | 2.46 MP | 9 | 4.268 ms | 0.207 ms | 20.58x |
| 1920x1280 | 2.46 MP | 13 | 4.854 ms | 0.214 ms | 22.64x |
| 1920x1280 | 2.46 MP | 17 | 6.063 ms | 0.222 ms | 27.30x |
| 1920x1280 | 2.46 MP | 25 | 9.412 ms | 0.256 ms | 36.79x |
| 7680x4320 | 33.18 MP | 5 | 29.297 ms | 1.256 ms | 23.32x |
| 7680x4320 | 33.18 MP | 9 | 44.819 ms | 1.294 ms | 34.64x |
| 7680x4320 | 33.18 MP | 13 | 60.144 ms | 1.281 ms | 46.95x |
| 7680x4320 | 33.18 MP | 17 | 101.273 ms | 1.287 ms | 78.70x |
| 7680x4320 | 33.18 MP | 25 | 146.636 ms | 1.673 ms | 87.63x |
| 12325x9979 | 122.99 MP | 5 | 105.458 ms | 4.210 ms | 25.05x |
| 12325x9979 | 122.99 MP | 9 | 153.885 ms | 4.143 ms | 37.15x |
| 12325x9979 | 122.99 MP | 13 | 208.048 ms | 4.157 ms | 50.05x |
| 12325x9979 | 122.99 MP | 17 | 273.097 ms | 4.567 ms | 59.80x |
| 12325x9979 | 122.99 MP | 25 | 415.399 ms | 6.094 ms | 68.16x |

At 122.99 MP, increasing width from 5 to 13 nearly doubled CPU time while GPU
time stayed around 4.1–4.2 ms. From width 5 to 25, CPU time rose about 3.94x
and GPU time about 1.45x. GPU time began increasing more materially at widths
17 and 25, so the device was not insensitive to footprint without limit.

These results reject the use of an approximately 11x external-memory-bandwidth
ratio as an expected ceiling for this stencil. Neighboring outputs reuse
overlapping input regions, and a wider footprint adds arithmetic, addressing,
loads, and instructions faster than it necessarily adds external-memory
traffic. The observations are consistent with cache reuse plus the GPU's
parallel instruction resources and latency hiding, but they do not isolate
which resource limited any particular case.

### RTX 3050 Laptop platform

The second record used an Intel Core i5-11400H, 12 Numba threads, and an RTX
3050 Laptop GPU with 4 GB VRAM and an approximately 64 W power cap. Reported
software included CuPy 14.1.1, Numba 0.65.1, NumPy 2.4.6, and a CUDA 13.x
environment.

| Shape | Pixels | Width | CPU median | GPU median | CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920x1280 | 2.46 MP | 5 | 2.650 ms | 0.410 ms | 6.47x |
| 1920x1280 | 2.46 MP | 9 | 4.707 ms | 0.491 ms | 9.59x |
| 1920x1280 | 2.46 MP | 13 | 4.309 ms | 0.574 ms | 7.50x |
| 1920x1280 | 2.46 MP | 17 | 4.024 ms | 0.665 ms | 6.05x |
| 1920x1280 | 2.46 MP | 25 | 6.019 ms | 0.952 ms | 6.32x |
| 7680x4320 | 33.18 MP | 5 | 35.408 ms | 3.745 ms | 9.46x |
| 7680x4320 | 33.18 MP | 9 | 37.459 ms | 4.848 ms | 7.73x |
| 7680x4320 | 33.18 MP | 13 | 61.475 ms | 6.126 ms | 10.04x |
| 7680x4320 | 33.18 MP | 17 | 61.303 ms | 8.424 ms | 7.28x |
| 7680x4320 | 33.18 MP | 25 | 88.780 ms | 10.720 ms | 8.28x |
| 12325x9979 | 122.99 MP | 5 | 148.627 ms | 13.559 ms | 10.96x |
| 12325x9979 | 122.99 MP | 9 | 164.258 ms | 17.367 ms | 9.46x |
| 12325x9979 | 122.99 MP | 13 | 248.483 ms | 22.593 ms | 11.00x |
| 12325x9979 | 122.99 MP | 17 | 230.625 ms | 27.607 ms | 8.35x |
| 12325x9979 | 122.99 MP | 25 | 296.928 ms | 41.682 ms | 7.12x |

At 122.99 MP, GPU time normalized to each platform's 5-tap result was:

| Width | RTX 5000 Ada | RTX 3050 Laptop |
| ---: | ---: | ---: |
| 5 | 1.00x | 1.00x |
| 9 | about 0.98x | about 1.28x |
| 13 | about 0.99x | about 1.67x |
| 17 | about 1.08x | about 2.04x |
| 25 | about 1.45x | about 3.07x |

The RTX 5000 Ada's direct GPU-time advantage over the RTX 3050 Laptop grew
from about 3.2x at 5 taps to 4.2x, 5.4x, 6.0x, and 6.8x at 9, 13, 17, and 25
taps. This shows that “GPU” is not one performance category and that platform
characteristics materially affect footprint scaling. It does not isolate Ada
microarchitecture: generation, execution-resource count, caches, memory,
power, clocks, host platform, and CuPy patch version all differed.

### i5-11400H AVX-512 control experiment

The laptop CPU results showed an unexpected cross-system pattern: the Ryzen 7
5700X generally led at smaller footprints, while the smaller 6-core/12-thread
Tiger Lake i5-11400H became faster or more competitive at wider footprints.
The maintainer investigated whether AVX-512-enabled Numba/LLVM code generation
contributed to that crossover. This used Numba 0.65.1 and the same separable
FP32 convolution from the footprint benchmark. AVX-512 was exposed by the
laptop OS and detected by Numba.

Inspection of the native generated assembly found extensive `zmm` operations,
vector gathers including `vgatherdps` and `vgatherqps`, FMA instructions, and
substantial vectorized index/address-generation machinery. Repeating the same
benchmark after disabling AVX-512 features through `NUMBA_CPU_FEATURES`
produced `ymm`/`xmm` vector code instead. Explicitly restoring the AVX-512
feature set returned timings close to the native run without an explicit
override. The recorded controls were:

```text
NUMBA_CPU_FEATURES="-avx512f,-avx512cd,-avx512vl,-avx512dq"
NUMBA_CPU_FEATURES="+avx512f,+avx512cd,+avx512vl,+avx512dq"
```

The first disabled the tested AVX-512 feature set; the second explicitly
re-enabled it as a control against the ordinary no-override native run.

The most relevant approximate 122.99 MP results were:

| Width | Native / AVX-512 enabled | AVX-512 disabled | Disabled/native time |
| ---: | ---: | ---: | ---: |
| 5 | 127 ms | 118 ms | 0.93x |
| 9 | 159 ms | 162 ms | 1.02x |
| 13 | 248 ms | 220 ms | 0.89x |
| 17 | 230 ms | 304 ms | about 1.33x |
| 25 | 295 ms | 415 ms | 1.41x |

The roughly 1.3–1.4x penalty without AVX-512 at 17 and 25 taps also appeared
consistently at the smaller tested image sizes. The 13-to-17-tap discontinuity
is especially informative: the native path made 17 taps faster than 13, while
the AVX-512-disabled timings returned to more conventional monotonic scaling.
The separate native run above differs at some widths from the earlier laptop
table, most visibly at five taps; preserve them as distinct approximate runs
rather than silently replacing one with the other.

This establishes that AVX-512-enabled Numba/LLVM code generation materially
affected the i5-11400H's wide-footprint results. It does not establish a simple
mapping such as “17 taps equals one 16-float AVX-512 vector.” Although `zmm`
registers were common, much of the floating-point work still appeared in
`ymm` registers, and the wider ISA also participated in gather-heavy index and
address work. The exact loop transformation and limiting resource have not
been isolated.

AVX-512 was not universally beneficial: the disabled path matched or beat the
native path at some smaller and moderate widths. The durable interpretation is
that Numba/LLVM selected a substantially different AVX-512-enabled vectorized
implementation which became more favorable around 17 taps and above in this
workload. Do not generalize the result to other CPUs, Numba convolutions, or
AVX-512 workloads.

This controlled result helps explain, but does not completely isolate, the
earlier 5700X/i5-11400H crossover. Disabling AVX-512 substantially reduced the
laptop's large-footprint advantage. At 25 taps and 122.99 MP, the earlier
5700X and the AVX-512-disabled laptop both landed near 415 ms; that numerical
coincidence should not be overinterpreted because their hardware and software
environments differed. More broadly, changing filter footprint can change
which architectural resources matter enough to alter CPU rankings, just as the
RTX 5000 Ada and RTX 3050 Laptop records showed different GPU scaling.

There is no project requirement to reproduce an analogous CUDA ISA-control
experiment. Isolating CuPyX code generation that way would require more custom
kernel or compiler-control work than this bounded Canny investigation
justifies.

## Relevance to the Canny observations

For the named 23.23 MP Canny workload, representative warm RTX 5000 Ada stage
ratios remain approximately 40x blur, 20–23x gradients, 40x NMS, 30–35x
hysteresis, and 30x overall. The gradient ratio is not 40x. Its two Sobel
filters, magnitude arithmetic and square root, and maximum reduction make it
more than a bandwidth stream. The independent maximum benchmark's 14–15x
ratio does not by itself explain the complete gradient result.

The real blur uses the same `sigma=1.5`, radius-6, 13-tap footprint as the
isolated experiment. At 122.99 MP, that isolated case reported about 208 ms on
CPU and 4.16 ms on the RTX 5000 Ada, or about 50x. The Canny blur comparison
was closer to 40x, plausibly because its timing boundary includes surrounding
work such as clipping and differs in context. The isolated result independently
supports the broad scale without proving an exact architectural cause.

The older operation-class proxy combined a roughly 19.1x NMS-lite ratio for
regular/local work and a 28.5x labeling ratio into a coarse expectation near
22x, versus an observed roughly 30x operator ratio. It remains useful, but its
regular/local proxy did not model the actual blur and gradient filters. The
13-tap Gaussian's roughly 50x isolated result helps explain why the coarse
model was conservative. Separability itself is not the explanation: both
reviewed CPU and GPU filtering paths already exploit separable structure.

## Durable interpretation

- NumPy maximum reduction was a useful negative optimization result: efficient
  enough to resemble a streaming reduction, though still a full-frame pass.
- Raw external-memory bandwidth ratios are not stage-speedup ceilings.
- Stencil footprint can materially change CPU/GPU ratios, and moderate widths
  were nearly flat in GPU time on one large Ada workload.
- On the i5-11400H, AVX-512-enabled Numba/LLVM code generation materially
  improved the wider 17- and 25-tap cases, but was not consistently faster at
  smaller widths; CPU ISA effects are workload-shape dependent.
- The mobile Ampere platform scaled very differently, so hardware and platform
  details must accompany GPU filtering claims.
- Generated general-purpose CuPyX filter kernels can map regular stencil work
  effectively without an established exotic shared-memory implementation.
- Use “in this workload,” “consistent with,” and “suggests.” Profiling counters
  are required before assigning the observations to a specific cache,
  instruction, occupancy, register, or memory bottleneck.
