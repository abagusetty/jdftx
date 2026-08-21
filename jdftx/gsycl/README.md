# gsycl — CUDA→SYCL compatibility shim for JDFTx

This directory lets JDFTx target Intel GPUs through SYCL **without forking the
sources**. The `.cu` files stay CUDA sources; `gsycl/` supplies SYCL definitions
for the CUDA constructs they use. Both backends build from the same tree:

```
cmake -DEnableCUDA=ON  …   # NVIDIA, nvcc, unchanged
cmake -DUSE_SYCL=ON    …   # Intel, icpx, via this shim
```

The two are mutually exclusive (CMake errors if both are set), because
`gsycl/` is prepended to the include path and shadows the CUDA toolkit headers.

## How the .cu files reach the shim

`jdftx/CMakeLists.txt`, in the `USE_SYCL` branch:

1. `include_directories(BEFORE gsycl)` — so `#include <cuda_runtime.h>`,
   `<cublas_v2.h>`, `<cufft.h>`, `<cusolverDn.h>`, `<driver_types.h>`,
   `<vector_types.h>` and `<thrust/…>` resolve here instead of to the toolkit.
2. `.cu` sources are compiled with `LANGUAGE CXX` and `-x c++`, plus
   `-include gsycl/cuda_prologue.hpp`. That prologue plays nvcc's role: it
   predefines `__device__`/`__host__`/`__global__`/`__constant__` (so
   `core/scalar.h` takes its "in a .cu file" branch) and defines
   `__CUDA_ARCH__` during the SYCL *device* pass (so the existing
   `#ifdef __CUDA_ARCH__` host/device splits in `RadialFunction.h` and
   `Spline.h` pick the device path unchanged).
   `.cpp` sources deliberately do **not** get the prologue — they take the host
   path, exactly as under nvcc.
3. `-DUSE_SYCL` is defined on the target; it is the only guard the JDFTx
   sources themselves test.

## What `sycl_device.hpp` provides

| CUDA construct | SYCL mapping |
| --- | --- |
| `threadIdx` / `blockIdx` / `blockDim` / `gridDim` | macros building a `dim3` from `this_work_item::get_nd_item<3>()`; CUDA x/y/z ↔ SYCL dim 2/1/0 |
| `__syncthreads()` | `sycl::group_barrier` |
| `extern __shared__ T x[]` | `JDFTX_DYNAMIC_SHARED(T, x)` → `get_work_group_scratch_memory()` |
| `sincos(x,&s,&c)` | device definition of the glibc-declared symbol |
| `cudaMalloc`/`Free`/`Memcpy`/`Memset`/… | USM on a process-wide in-order queue |
| `cudaDeviceProp`, `cudaFuncGetAttributes` | queried from `sycl::device` |
| `kernel<<<grid,block>>>(args)` | `jdftx_sycl::launchKernel(...)` (see below) |

Because thread indexing is a free-function query, **kernels keep their original
signatures** — no `nd_item` threading, no lambda rewriting, and
`core/LoopMacros.h` is untouched.

`cublas_v2.h`, `cusolverDn.h` and `cufft.h` are declaration-only headers
mirroring the real CUDA ones; the oneMKL calls live in the matching
`*_impl.cpp`, which are the only TUs that include an oneMKL header. That keeps
`mkl_cblas.h` out of any TU that also sees `gsl/gsl_cblas.h`, avoiding the
CBLAS enum redefinition clash.

## Kernel launches — the one thing the sources had to change

`kernel<<<grid,block>>>(args)` has no portable spelling, so every launch site
now goes through a macro defined in `core/GpuKernelUtils.h`:

```cpp
JDFTX_LAUNCH(RealG_kernel, glc, zBlock, S, vFull, vHalf, scaleFac);
JDFTX_LAUNCH_T(lGradientStress_kernel, (l,m), glc, zBlock, S, G, w, X, Y, grad_RRT, lPhase);
JDFTX_LAUNCH_SHARED(eblas_capMinMax_kernel, glc, sharedMemBytes, N, x, …);
JDFTX_LAUNCH_T_DIM_SHARED(nAugmentGrad_kernel, (Nlm), nBlocks, nPerBlock, shBytes, …);
```

Under CUDA these expand right back to the original `<<< >>>` syntax. Template
arguments are parenthesised in the `_T` forms so their commas survive macro
expansion.

Under SYCL the kernel is passed twice: as a capture-less generic lambda (the
device-side call — SYCL forbids indirect calls) and as a plain function pointer
that `launchKernel()` uses to deduce the kernel's parameter types. That second
role matters: it is what reproduces the implicit conversions `<<< >>>` performs
at a launch (`std::vector` → `array<>`, `GpuBuffer` → `double*`, …), which
plain template deduction over the arguments would lose.

## Device copyability

SYCL requires kernel arguments to be trivially copyable unless told otherwise.
`vector3`/`matrix3`/`tensor3`/`symmetricMatrix3` declare user-provided copy
constructors, so `sycl_device.hpp` ends with `is_device_copyable`
specializations for them; their copy constructors are plain field copies that
the device compiler handles fine.

`RadialFunctionG` needs more than that. It carries a `std::vector<double>` for
its CPU-side copy, and passing it *by value* to a kernel would make the device
copy-construct that vector — which needs heap allocation, so the device image
ends up with unresolved `operator new` / `__throw_bad_array_new_length`. It is
the only kernel-argument type in JDFTx with that problem (a
`is_trivially_destructible` assertion over every kernel parameter across all
`.cu` files finds no others).

The fix is `JDFTX_KERNEL_ARG(T)` in `core/GpuKernelUtils.h`: `const T` on CUDA
(unchanged), `const T&` on SYCL. The launcher stores such a parameter as
`jdftx_sycl::Raw<T>` — a trivially copyable byte image, which is exactly how
CUDA passes kernel arguments anyway — and binds the reference into it, so no
constructor runs on the device. `RadialFunctionG::getCoeff()` already selects
the GPU pointer under `__CUDA_ARCH__`, so the vector member is never read there.

## Two places oneMKL is not a drop-in for its CUDA counterpart

Both were found by CPU-vs-GPU numerical comparison, and both are fixed inside
this directory — the JDFTx sources are untouched.

**cuBLAS scalar results are host pointers.** `cublasDdot`/`Zdotc`/`Dnrm2`/
`Dznrm2`/`Dasum` take a `result` pointer that JDFTx always fills with a host
stack address (`&result` of a local, in `core/BlasExtra.cu`). Real cuBLAS
defaults to `CUBLAS_POINTER_MODE_HOST` and syncs the device-side reduction back
to it; oneMKL has no such mode and requires device-writable USM. Passing the
host pointer through made the GPU write a scalar to an unmapped address — the
page fault the port used to die on. `cublas_impl.cpp` routes every such result
through a process-lifetime `malloc_device` scratch and then copies it out with a
blocking `memcpy`. A `malloc_shared` scratch read directly on the host was tried
first and segfaulted *intermittently*: shared-USM host visibility is not
synchronous with the compute event's completion on this driver.

**FFTW and oneMKL disagree on non-Hermitian c2r input.** A half-space (CCE)
array stores the k2=0 and k2=S2/2 planes redundantly, so a well-formed c2r input
must satisfy `X[k] == conj(X[-k])` there. JDFTx's half-space kernels do not
always produce that: they evaluate radial functions at the *folded* index, and
on the k2=S2/2 plane the fold cannot represent the negation (−S2/2 ≡ +S2/2
mod S2), so entry `(k0,k1,S2/2)` is labelled `G(k0,k1,+S2/2)` while its stored
partner `(−k0,−k1,S2/2)` is labelled `G(−k0,−k1,+S2/2)` rather than
`G(k0,k1,−S2/2)`. Whenever the metric has a cross term — any hexagonal cell —
those two have different `|G|` and the plane is genuinely non-Hermitian. The
input is then out of contract and the two libraries resolve it differently;
neither is wrong. FFTW's answer turns out to be exactly the Hermitian
symmetrization `X[k] <- (X[k] + conj(X[-k]))/2`, so `cufft_impl.cpp` applies
that to both planes before `compute_backward` and the two agree to 1e-15.
Mutating the input is allowed: cuFFT and FFTW both document that c2r may destroy
it, and `core/Operators.cpp`'s `I()` clones first unless handed an expiring
value.

Left unfixed, this shifted every eigenvalue of the graphene test by up to
6e-4 Eh (its Fermi level missed the 1e-4 reference by 4.6e-4) while leaving the
total energy right to 1e-6, because the error rides entirely on Nyquist
components. `tests/test_c2r_nyquist.cpp` is a standalone demonstration: it
builds a JDFTx-shaped grid, perturbs the redundant planes, and exits non-zero if
FFTW and oneMKL differ. The build line is in its header comment.

## Floating-point model

icpx defaults to `-fp-model=fast`, and even `-fp-model=precise` leaves FMA
contraction on. Host and device codegen make those choices independently, so the
CPU and GPU builds can disagree for reasons that have nothing to do with the
port. `jdftx/CMakeLists.txt` therefore applies `-fp-model=precise
-ffp-contract=off -fno-fast-math` to every target by default (the `PreciseFP`
option), and repeats the first two in `SYCL_COMPILE_FLAGS` so device codegen
stays pinned even if `CMAKE_CXX_FLAGS` is overridden. `CompileNative` no longer
passes Intel's `-fast`, which would have re-enabled `-fp-model=fast=2`; it uses
`-xHost` alone.

## Running one system per tile

`run_per_tile.sh` launches a batch of independent inputs as separate
single-rank processes, one per PVC tile — no MPI, since a tile is its own HBM
domain and each input is a different system. It reads the GPU-to-NUMA map from
sysfs, gives each process a disjoint range of physical cores on the socket its
GPU is attached to, caps threads to that count, binds memory to the local NUMA
node, and applies an equal-share `RLIMIT_DATA` cap so one runaway job fails by
itself instead of drawing the OOM killer onto its neighbours. `DRY_RUN=1` prints
the placement without running anything.

## Total footprint in the JDFTx sources

* 104 kernel-launch sites rewritten to `JDFTX_LAUNCH*` (mechanical, one for one).
* `core/GpuKernelUtils.h`: one additive block defining those macros, plus
  `JDFTX_KERNEL_ARG`, for both backends.
* 15 `RadialFunctionG` kernel parameters wrapped in `JDFTX_KERNEL_ARG(...)`
  (`core/Operators.cu`, `fluid/PCM_internal.cu`, `electronic/SpeciesInfo.cu`).
* `core/Spline.h` and `core/BlasExtra.cu`: three lines each, guarding the
  `extern __shared__` declaration and adding the SYCL scratch pointer.

Everything else lives in this directory.
