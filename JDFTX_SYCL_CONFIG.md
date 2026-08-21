# JDFTx SYCL backend — project configuration

Working notes for building and running the SYCL backend on Aurora.
Current port status, validation evidence and coverage: **[`SYCL_STATUS.md`](SYCL_STATUS.md)**.
Design of the shim itself: [`jdftx/gsycl/README.md`](jdftx/gsycl/README.md).

## How long things actually take

Measured on this node (2x 52-core sockets, 6x PVC), so you can tell a slow step
from a hung one:

| Step | Time |
| --- | --- |
| Clean full AOT build, `make -j48` (315 TUs, 8 links) | **109 s** |
| GPU test suite, one tile | ~305 s |
| CPU test suite, single-threaded | ~554 s |
| graphene `totalE.in`, GPU | ~11 s |

Build time scales with `-j`; the old `-j16` recipe is roughly 3x the above, so
still minutes, not hours. An earlier version of this file claimed AOT
compilation took "30–60+ minutes" and told you never to put a timeout on a
build — that was wrong by more than an order of magnitude, and it is why the
test suite now carries a perfectly ordinary `TIMEOUT 1800` per test.

Timeouts are fine. What is not fine is *assuming* a hang: if a step runs
dramatically past the numbers above, investigate it (attach a debugger, check
`SYCL_UR_TRACE=2`) rather than killing and retrying blindly.

## Build

```bash
# GSL, built locally once (no system package on Aurora).
# Must be -fPIC, and leave gsl_cblas.h pristine -- see the CBLAS note below.
cd /tmp && curl -L -o gsl-2.8.tar.gz https://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz
tar -xzf gsl-2.8.tar.gz && cd gsl-2.8 && mkdir -p build && cd build
../configure --prefix=$HOME/gsl-install --with-pic && make -j8 && make install

# Out-of-source build. The source tree must stay free of CMakeCache.txt, or
# CMake silently reuses that in-source configuration instead.
cd /lus/tegu/projects/Performance/abagusetty/JDFTx/jdftx
mkdir -p build-prod && cd build-prod
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DUSE_SYCL=ON -DEnableMPI=OFF \
      -DFFTW3_PATH=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/fftw-3.3.10-l6f476v \
      -DGSL_PATH=$HOME/gsl-install \
      ../jdftx
make -j48
```

Options that matter:

* `-DEnableMPI=OFF` — production here is one system per tile, so there is
  nothing for MPI to coordinate. Leaving it out also removes a whole class of
  rank-placement problems. Multi-rank is *not* validated (see `SYCL_STATUS.md`).
* `-DSyclAOT=OFF` — JIT device codegen instead of `spir64_gen -device pvc`.
  Faster to build while iterating; pair with `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
  when you need a symbolicated device backtrace.
* `-DSyclDeviceCodeSplit=` — `per_kernel` (default) / `per_source` / `off`.
* `-DPreciseFP=OFF` — turns off the strict floating-point model. Don't, unless
  you re-validate the suite: icpx defaults to `-fp-model=fast`, and even
  `precise` leaves FMA contraction enabled, decided independently for host and
  device.
* oneMKL is located from `$MKLROOT` (CONFIG mode); pass
  `-DMKL_DIR=$MKLROOT/lib/cmake/mkl` if it is not set.

Targets: `libjdftx_gpu_sycl.so` plus `jdftx_gpu` / `wannier_gpu` / `phonon_gpu`,
alongside the usual CPU `libjdftx.so` and `jdftx` / `wannier` / `phonon`.

## Running

```bash
make testclean               # REQUIRED first -- see SYCL_STATUS.md
JDFTX_SUFFIX=_gpu ctest      # GPU suite, pinned to one tile
ctest                        # CPU suite
```

Single-tile pinning, thread count and timeout are CMake test properties, so
plain `ctest` is already correct with no wrapper environment.

Production, one system per tile, MPI-free:

```bash
DRY_RUN=1 jdftx/gsycl/run_per_tile.sh build-prod/jdftx_gpu out/ *.in
```

## Environment

* MKLROOT: `/opt/aurora/26.181.0/oneapi/mkl/latest`
* FFTW3: `/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/fftw-3.3.10-l6f476v`
* GSL: `$HOME/gsl-install` (2.8, `-fPIC`, pristine upstream headers)
* Compiler: icpx, Intel oneAPI 2026.1; target Intel PVC (Aurora)
* CMake 3.31.11; no internet from the compute node
* `OMP_NUM_THREADS` is exported as 208 by the environment — override it.
  Inherited, JDFTx spawns 208 `std::thread`s per operator and aborts in thread
  creation. The tests set it to 1; `run_per_tile.sh` sets it per tile.
* The job cpuset is `1-51,53-103` — core 0 of each socket is reserved, so 51
  usable physical cores per socket, not 52.

## Design

`jdftx/gsycl/` is a CUDA→SYCL compatibility shim: the `.cu` files stay CUDA
sources and both backends build from the same tree (`-DEnableCUDA=ON` or
`-DUSE_SYCL=ON`, mutually exclusive, because `gsycl/` is prepended to the
include path and shadows the CUDA toolkit headers).

Key files:

* `jdftx/gsycl/sycl_device.hpp` — the shim proper: thread indexing,
  `__syncthreads`, dynamic shared memory, `sincos`, the CUDA runtime API,
  kernel-launch helpers, `is_device_copyable` specializations.
* `jdftx/gsycl/cuda_prologue.hpp` — force-included into `.cu` units only;
  predefines `__device__` / `__global__` / `__CUDA_ARCH__` the way nvcc would.
* `jdftx/gsycl/{cublas_v2,cusolverDn,cufft}.h` + the matching `*_impl.cpp` —
  declaration-only headers plus the only TUs that touch oneMKL. This exists
  because `oneapi/mkl/types.hpp` pulls in `mkl_cblas.h`, which redefines
  `CBLAS_ORDER`/`CBLAS_TRANSPOSE`/etc. as `typedef enum`, while GSL's
  `gsl_cblas.h` (via `core/BlasExtra.h`) defines the same names as plain
  `enum`. The split keeps them out of any single translation unit.
* `jdftx/core/GpuKernelUtils.h` — the `JDFTX_LAUNCH*` macros and
  `JDFTX_KERNEL_ARG`, defined for both backends.
* `jdftx/gsycl/run_per_tile.sh` — MPI-free production launcher, one system per
  tile, with CPU/NUMA/DRAM rationing.
* `jdftx/gsycl/tests/` — standalone checks that need no JDFTx build
  (`fft_check.cpp` for the cuFFT shim, `test_c2r_nyquist.cpp` for the c2r
  Nyquist convention). See its README for build lines.
* `jdftx/test/CMakeLists.txt` — single-tile test properties and the
  `TestTile` / `TestThreads` / `TestTimeout` knobs.

## Orch agent configuration

Project-level Orch config at `.pi/orch/config.json`:

* orchestrator / worker / validator / research / plan_codebase:
  `argo-claude/claudesonnet5`
* smart_friend: `argo-claude/claudeopus5`
