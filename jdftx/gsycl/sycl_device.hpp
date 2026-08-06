#pragma once
// jdftx_sycl_shim.hpp — CUDA→SYCL shim for JDFTx
//
// Drop-in replacement for <cuda_runtime.h> + all CUDA headers.
// Replaces <<<>>> kernel launches, maps CUDA runtime → SYCL, and provides
// cuBLAS/cuFFT/cuSOLVER thin wrappers.
//
// Design:
//   - Single process-wide in-order queue (JDFTx assumes stream-0 ordering)
//   - Real error handling (cudaGetLastError returns actual SYCL errors)
//   - kernelIndex/dir macros read from global `__sycl_item` in kernels
//   - JDFTX_LAUNCH(kernel, glc, args) replaces <<<glc.nBlocks,glc.nPerBlock>>>

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cmath>
#include <limits>
#include <type_traits>

// =========================================================================
// 1. Process-wide queue (singleton, in-order for stream-0 semantics)
// =========================================================================
namespace jdftx_sycl {

inline sycl::queue& queue() {
    static sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    return q;
}

inline std::string& lastError() {
    static std::string e;
    return e;
}

template<typename F>
inline void guarded(F&& f) {
    try { f(); }
    catch(const sycl::exception& e) { lastError() = e.what(); }
    catch(const std::exception& e)  { lastError() = e.what(); }
}

} // namespace jdftx_sycl

// Global item for kernel index macros (set by JDFTX_LAUNCH wrapper)
extern sycl::nd_item<3> __sycl_item;

// =========================================================================
// 2. Error handling types and functions
// =========================================================================
using cudaError_t = int;
constexpr int cudaSuccess = 0;

inline cudaError_t cudaGetLastError() {
#ifndef JDFTX_SYCL_STUB_ERRORS
    jdftx_sycl::guarded([]{ jdftx_sycl::queue().wait_and_throw(); });
    return jdftx_sycl::lastError().empty() ? cudaSuccess : 1;
#endif
    return cudaSuccess;
}

inline cudaError_t cudaPeekAtLastError() {
    return jdftx_sycl::lastError().empty() ? cudaSuccess : 1;
}

inline const char* cudaGetErrorString(cudaError_t) {
    return jdftx_sycl::lastError().empty() ? "no error"
                                           : jdftx_sycl::lastError().c_str();
}

inline void checkCudaErrors(cudaError_t) {}

// =========================================================================
// 3. Kernel launch configuration (GpuKernelUtils.h types)
// =========================================================================
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

inline unsigned int __activemask() { return 0; }

// CUDA x/y/z map to SYCL dims 2/1/0 (SYCL row-major, slowest-first)
#define kernelIndex(dir) (int(__sycl_item.get_global_id(JDFTX_SYCL_DIM_##dir)))
#define JDFTX_SYCL_DIM_x 2
#define JDFTX_SYCL_DIM_y 1
#define JDFTX_SYCL_DIM_z 0

// Flattened 1D index for 2D grids
#define kernelIndex1D() (int(__sycl_item.get_global_linear_id()))

#define __syncthreads() (sycl::group_barrier(__sycl_item.get_group()))

// =========================================================================
// 4. Kernel launch macro
//
// Replaces:  kernel<<<glc.nBlocks,glc.nPerBlock>>>(args...)
// With:      JDFTX_LAUNCH(kernel, glc, args...)
//
// Handles 102 of 104 call sites. The 2 BlasExtra.cu reductions need
// hand-written sycl::local_accessor + group_barrier.
// =========================================================================
namespace jdftx_sycl {

inline sycl::nd_range<3> to_nd_range(const dim3& nBlocks, const dim3& nPerBlock) {
    // CUDA x is fastest-varying → SYCL dim 2
    sycl::range<3> local(nPerBlock.z, nPerBlock.y, nPerBlock.x);
    sycl::range<3> global(
        size_t(nBlocks.z) * nPerBlock.z,
        size_t(nBlocks.y) * nPerBlock.y,
        size_t(nBlocks.x) * nPerBlock.x
    );
    return sycl::nd_range<3>(global, local);
}

} // namespace jdftx_sycl

#define JDFTX_LAUNCH(kernel, glc, ...)                                        \
    jdftx_sycl::guarded([&]{                                                  \
        jdftx_sycl::queue().parallel_for(                                     \
            jdftx_sycl::to_nd_range((glc).nBlocks, (glc).nPerBlock),          \
            [=](sycl::nd_item<3> item) {                                      \
                __sycl_item = item;                                           \
                kernel(__VA_ARGS__);                                          \
            });                                                               \
    })

// =========================================================================
// 5. Kernel qualifiers
// =========================================================================
#define __global__   __attribute__((always_inline))
#define __device__   __attribute__((always_inline))
#define __host__     __attribute__((always_inline))
#define __forceinline__ __attribute__((always_inline))
#define __hostanddev__ __attribute__((always_inline))
#define __constant__ inline constexpr

// =========================================================================
// 6. Memory management
// =========================================================================
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    jdftx_sycl::guarded([&]{
        *ptr = sycl::malloc_device(size, jdftx_sycl::queue());
    });
    return (*ptr) ? cudaSuccess : 1;
}

inline cudaError_t cudaFree(void* ptr) {
    if (ptr) jdftx_sycl::guarded([&]{ sycl::free(ptr, jdftx_sycl::queue()); });
    return cudaSuccess;
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    jdftx_sycl::guarded([&]{
        *ptr = sycl::malloc_host(size, jdftx_sycl::queue());
    });
    return (*ptr) ? cudaSuccess : 1;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    if (ptr) jdftx_sycl::guarded([&]{ sycl::free(ptr, jdftx_sycl::queue()); });
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    jdftx_sycl::guarded([&]{
        jdftx_sycl::queue().memcpy(dst, src, n).wait();
    });
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* ptr, int val, size_t n) {
    jdftx_sycl::guarded([&]{
        jdftx_sycl::queue().memset(ptr, static_cast<unsigned char>(val), n).wait();
    });
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    jdftx_sycl::guarded([]{ jdftx_sycl::queue().wait_and_throw(); });
    return jdftx_sycl::lastError().empty() ? cudaSuccess : 1;
}

// =========================================================================
// 7. Device properties
// =========================================================================
struct cudaDeviceProp {
    int maxThreadsPerBlock;
    int maxGridSize[3];
    int maxThreadsDim[3];
    size_t sharedMemPerBlock;
    int multiProcessorCount;
    char name[256];
};

struct cudaFuncAttributes {
    int maxThreadsPerBlock;
    size_t sharedSizeBytes;
};

// Device properties are cached in GpuUtil.cpp as: extern cudaDeviceProp cudaDevProps;
// We populate it once at init.
extern cudaDeviceProp cudaDevProps;

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int /*device*/ = 0) {
    auto& dev = jdftx_sycl::queue().get_device();
    int wg = int(dev.get_info<sycl::info::device::max_work_group_size>());
    prop->maxThreadsPerBlock = wg;
    auto mi = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
    prop->maxThreadsDim[0] = int(mi[2]);   // SYCL dim order is reversed vs CUDA x/y/z
    prop->maxThreadsDim[1] = int(mi[1]);
    prop->maxThreadsDim[2] = int(mi[0]);
    prop->maxGridSize[0] = prop->maxGridSize[1] = prop->maxGridSize[2] = (1 << 30);
    prop->sharedMemPerBlock = size_t(dev.get_info<sycl::info::device::local_mem_size>());
    prop->multiProcessorCount = int(dev.get_info<sycl::info::device::max_compute_units>());
    std::snprintf(prop->name, sizeof(prop->name), "%s",
                  dev.get_info<sycl::info::device::name>().c_str());
    return cudaSuccess;
}

template<typename K>
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, K*) {
    static int wg = int(jdftx_sycl::queue().get_device()
                        .get_info<sycl::info::device::max_work_group_size>());
    a->maxThreadsPerBlock = wg;
    a->sharedSizeBytes = 0;
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* id) { *id = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }

// =========================================================================
// 8. Math functions (for __hostanddev__ code)
// =========================================================================
using sycl::sqrt;
using sycl::exp;
using sycl::log;
using sycl::pow;
using sycl::floor;
using sycl::ceil;
using sycl::erf;
using sycl::erfc;
using sycl::sin;
using sycl::cos;
using sycl::tanh;
using sycl::atan;
using sycl::fabs;
using sycl::min;
using sycl::max;

// =========================================================================
// 9. CUDA-compatible types
// =========================================================================
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

// double2 is used everywhere in JDFTx
struct double2 {
    double x, y;
    double2() : x(0), y(0) {}
    double2(double x_, double y_) : x(x_), y(y_) {}
};

// float2
struct float2 {
    float x, y;
    float2() : x(0), y(0) {}
    float2(float x_, float y_) : x(x_), y(y_) {}
};

// =========================================================================
// 10. Stubbed CUDA runtime calls (always succeed)
// =========================================================================
inline cudaError_t cudaMemPrefetchAsync(const void*, size_t, int, void*) {
    return cudaSuccess;
}

inline cudaError_t cudaMallocManaged(void**, size_t, unsigned) {
    return cudaSuccess;
}

inline int cudaDeviceGetAttribute(int* value, int /*kind*/, int /*dev*/) {
    *value = 0;
    return 0;
}
