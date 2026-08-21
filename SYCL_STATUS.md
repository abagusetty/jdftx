# JDFTx SYCL port — status

**Status: PASS** against the JDFTx test suite (`jdftx/test/`, 11 tests / 47
checks) on Intel PVC — the same gate the CUDA and CPU builds are held to.

Branch `feature/sycl`. Design and rationale for the shim itself live in
[`jdftx/gsycl/README.md`](jdftx/gsycl/README.md); this file is the operational
summary.

## Validation

Aurora, 6x Intel Data Center GPU Max 1550, oneAPI 2026.1, AOT `spir64_gen`,
`-DEnableMPI=OFF`. Every run started from a cleaned test tree.

| Configuration | Result |
| --- | --- |
| GPU, one tile, `OMP_NUM_THREADS=1` | 11/11 pass, 307 s |
| GPU, one tile, `OMP_NUM_THREADS=8` | 11/11 pass, 281 s |
| CPU, `OMP_NUM_THREADS=1` | 11/11 pass, 554 s |
| 13 concurrent jobs over 12 tiles (`run_per_tile.sh`) | 13/13, physics matches reference |

Reproduced across two separate Aurora nodes. GPU and CPU agree to 9e-14 on
graphene's total energy and to 9 digits on its Fermi level.

> **Clean before you validate.** `runTest.sh` skips any run whose `<run>.out`
> already exists and ends in `Done!`, so a second `ctest` in a build dir with
> previous outputs just re-parses them and reports a vacuous 100% pass — even
> after a rebuild or on different hardware. Deleting `results`/`summary` is not
> enough. Run `make testclean` first.

## Build

```bash
mkdir -p build-prod && cd build-prod
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DUSE_SYCL=ON -DEnableMPI=OFF \
      -DFFTW3_PATH=$FFTW3_ROOT -DGSL_PATH=$HOME/gsl-install ../jdftx
make -j48        # ~4 min
```

* `-DSyclAOT=OFF` → JIT device codegen; much faster to build while iterating.
* `-DSyclDeviceCodeSplit=` → `per_kernel` (default) / `per_source` / `off`.
* `-DPreciseFP=OFF` disables the strict floating-point model — don't, unless
  you re-validate the suite. icpx defaults to `-fp-model=fast`, and even
  `precise` leaves FMA contraction on; host and device decide that
  independently, so the two builds can silently disagree.
* oneMKL is found from `$MKLROOT`; override with `-DMKL_DIR=`.
* The source tree must contain no `CMakeCache.txt` — CMake silently reuses an
  in-source configuration if one exists.

## Running the tests

Single-tile pinning, thread count, timeout and serialization are CMake test
properties (`jdftx/test/CMakeLists.txt`), so plain `ctest` is already correct —
no wrapper environment needed.

```bash
make testclean
JDFTX_SUFFIX=_gpu ctest      # GPU (tile 0.0)
ctest                        # CPU
make testresults             # human-readable per-check summary
```

Knobs: `-DTestTile=<device>.<subdevice>`, `-DTestThreads=<n>`,
`-DTestTimeout=<s>`.

Threads default to 1 because this node exports `OMP_NUM_THREADS=208`; inherited,
that makes every JDFTx operator spawn 208 `std::thread`s and the CPU binary
aborts in thread creation (`std::__throw_system_error` out of `eblas_zgemm`).
8 threads is validated and slightly faster.

Running a test input by hand needs `SRCDIR=<test source dir>` exported —
inputs `include ${SRCDIR}/common.in`.

## Running production

One system per tile, MPI-free — a tile is its own 64 GB HBM domain and each
input is a different system, so there is nothing for MPI to coordinate.

```bash
DRY_RUN=1 jdftx/gsycl/run_per_tile.sh build-prod/jdftx_gpu out/ *.in   # show placement
          jdftx/gsycl/run_per_tile.sh build-prod/jdftx_gpu out/ *.in   # run
```

It reads the GPU→NUMA map from sysfs and the job's cpuset from
`/proc/self/status`, gives each process a disjoint physical-core range on its
GPU's own socket, caps threads to that count, binds memory to the local NUMA
node, and applies an equal-share `RLIMIT_DATA` cap. Overrides: `NTILES`,
`CORES_PER_TILE`, `MEM_PER_TILE_GB` (0 disables the cap), `MEM_FRACTION`,
`PSEUDO`, `HIERARCHY`, `JDFTX_ARGS`.

On an Aurora node that comes out as 12 tiles × 8–9 cores × 80 GiB.

> Core specialization reserves core 0 of each socket, so the cpuset is
> `1-51,53-103` — 51 usable physical cores per socket, not 52. `numactl`
> refuses any binding outside it with the unhelpful `<0,1,2,...> is invalid`,
> which is why the launcher reads the cpuset rather than trusting `lscpu`.

## Scope of the PASS

**Covered:** plane-wave DFT with LDA/GGA functionals, ultrasoft
pseudopotentials, Fermi smearing, DFT+U, spin-orbit coupling,
isolated/slab/periodic Coulomb truncation, LinearPCM solvation, ionic and
lattice optimization, vibrational modes, checkpoint/restart — single tile, no
MPI. That is the production configuration this port targets.

**Not covered.** Tracing every kernel launch across the suite
(`JDFTX_SYCL_TRACE=1`) shows **46 of 97 GPU kernels execute**. The rest belong
to features no test exercises:

* `phonon` and `wannier` — built, never run by the suite.
* The ClassicalDFT / joint-DFT fluid path: `MixedFMT.cu`,
  `TranslationOperator.cu`, `Fex_ScalarEOS.cu`,
  `Fex_H2O_FittedCorrelations.cu` are *entirely* unexercised. Of the five fluid
  models, only `LinearPCM` is tested.
* NonlinearPCM and SaLSA (`DielectricApply_kernel`, `ScreeningApply_kernel`).
* Hybrid functionals / exact exchange (`exchangeAnalytic_kernel`).
* meta-GGA (`mGGA_kernel`); noncollinear spin (`spinDiagonalize_kernel`).
* The numerical Coulomb kernels `multRealKernel` / `multTransformedKernel`.
* Multi-rank MPI — not built here. `ZE_AFFINITY_MASK` filters what SYCL
  enumerates, but `gpuInit()` round-robins ranks over visible devices, so a
  multi-rank single-tile run needs the mask set **per rank** (e.g.
  `0.$PALS_LOCAL_RANKID`). Not handled inside the shim.

Stress is partly covered: `coulombAnalyticStress`, `gradLocalToStress`,
`reducedLstress` and `VnlStress` run under `latticeOpt`; the EXX,
real-space-kernel and full-grid L/Linv stress variants do not.

## Known gaps and caveats

* **Not bitwise reproducible.** Two identical GPU runs differ in the 8th
  significant digit at LCAO iteration 0. There is no `atomicAdd` anywhere in
  the sources, so this is oneMKL's internal BLAS/DFT reduction order. ~1e-8
  relative — far below every test tolerance, but bitwise determinism is not
  available.
* `cudaFuncGetAttributes` reports the device work-group limit for every kernel;
  SYCL has no per-kernel equivalent. Register-hungry kernels may need a smaller
  cap via `JDFTX_SYCL_MAX_WG`.
* `cudaMemPrefetchAsync` has no host-direction equivalent; the shim reports no
  concurrent managed access, so JDFTx leaves `prefetchSupported` false.
* The CUDA build has not been recompiled here (no nvcc on this machine). The
  CUDA branch of the launch macros was verified by preprocessing only.

## Debug facilities

| Env var | Effect |
| --- | --- |
| `JDFTX_SYCL_TRACE=1` | print and synchronise on every kernel launch |
| `JDFTX_SYCL_MAX_WG=<n>` | cap the work-group size `cudaFuncGetAttributes` reports |
| `ZE_SERIALIZE=2` | driver-level serialization — debugging only, never production |

Standalone checks that need no JDFTx build live in `jdftx/gsycl/tests/`; see
its README for build lines.

### Tooling notes

* `module load pti-gpu` **silently no-ops** — there is no default version. Use
  `module load pti-gpu/1.0.1-rc1` (or `0.17.0`); `unitrace` is then on `PATH`.
* `unitrace -c` (`--call-logging`) **crashes itself** in
  `ZeCollector::zeModuleCreateOnExit` during `urProgramLinkExp` on this stack.
  Use `SYCL_UR_TRACE=2` instead.
* `SYCL_CACHE_PERSISTENT=1` caches JIT kernels across runs; the AOT build
  avoids the startup cost entirely.
* Catching a GPU page fault live needs EU debug mode enabled, then
  `ZET_ENABLE_PROGRAM_DEBUGGING=1 gdb-oneapi -batch -ex run -ex "bt full"
  --args <exe> <args>`. A symbolicated backtrace needs `-g`, so use a
  `-DSyclAOT=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo` build for that.

## Build-system gotchas

Non-obvious things that already bit this port; all are handled in
`jdftx/CMakeLists.txt`, listed here so a rebuild elsewhere doesn't rediscover
them.

* **`MKLConfig.cmake` leaks `LINK_PREFIX="-l"` into directory scope**, which
  poisons `add_JDFTx_executable()` and emits a bare `-l` before `jdftxlib` on
  every executable link — it breaks the plain CPU `jdftx` target too, not just
  the SYCL ones. CMake saves and restores it.
* **Host MKL must be LP64** (`set(MKL_INTERFACE lp64)`). JDFTx and GSL both
  call CBLAS/LAPACK with 32-bit integers; the default ilp64 is silently wrong.
* **AOT requires `-fno-sycl-allow-device-image-dependencies`.** With
  dependencies allowed, a kernel image only *references* its device functions,
  leaving them in a separate image the runtime links at launch
  (`urProgramLinkExp`). That works for SPIR-V/JIT but fails with
  `UR_RESULT_ERROR_INVALID_ARGUMENT` under AOT, because native PVC binaries
  cannot be runtime-linked. oneMKL's imported targets force the flag back *on*
  as an `INTERFACE_LINK_LIBRARIES` item, so it always lands last and wins —
  CMake rewrites it in place on `MKL::MKL_SYCL{,::BLAS,::LAPACK,::DFT}`.
* **GSL and oneMKL CBLAS headers clash.** `oneapi/mkl/types.hpp` pulls in
  `mkl_cblas.h`, which redefines `CBLAS_ORDER`/`CBLAS_TRANSPOSE`/etc. as
  `typedef enum`; GSL's `gsl_cblas.h` (via `core/BlasExtra.h`) defines the same
  names as plain `enum`. Hence the shim's decl/impl split: `gsycl/{cublas_v2,
  cusolverDn,cufft}.h` are declarations-only and never include a oneMKL C++
  header, so the two never meet in one translation unit.
* `oneapi::mkl::lapack::{potrf,potrs,getrf,getrs,heevd,gesvd}` are overloads,
  not templates — no explicit `<cplx>` template argument.
* `gpuInit()` rejects devices reporting compute capability < 1.3 or
  `integrated`; the shim reports 9.0 and non-integrated.
* `gsycl/thrust/{device_ptr,sort}.h` are empty stubs — `SpeciesInfo.cu`
  includes them but uses no thrust symbol.

## Environment

* icpx / Intel oneAPI 2026.1; Intel Data Center GPU Max 1550 (PVC)
* GSL `$HOME/gsl-install` (2.8, built `-fPIC`, pristine upstream headers)
* FFTW3 `/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/fftw-3.3.10-l6f476v`
* MKLROOT `/opt/aurora/26.181.0/oneapi/mkl/latest`
* CMake 3.31.11; no internet from the compute node
* Pre-SYCL baseline for diffs: `52548a96`
* An earlier abandoned approach is kept at
  `.sycl-backup/wip-dec74808.patch` (a `Ptr<>` wrapper plus `KERNEL_SETUP()`
  everywhere) — superseded, do not restore.
