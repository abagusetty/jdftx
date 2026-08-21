#pragma once
/*-------------------------------------------------------------------
gsycl/sycl_device.hpp -- CUDA -> SYCL compatibility shim for JDFTx.

Design goals (see gsycl/README.md):
  * The JDFTx sources stay CUDA sources.  This header supplies SYCL
    definitions for every CUDA construct they use, so the only edits
    required in core/, electronic/ and fluid/ are the kernel-launch
    sites (which cannot be expressed portably in C++) and two
    dynamic-shared-memory declarations.
  * Nothing here is CUDA-visible: the header is only reachable when
    USE_SYCL is on, because gsycl/ is prepended to the include path and
    gsycl/cuda_runtime.h shadows the CUDA toolkit header of that name.

Thread indexing uses the sycl_ext_oneapi_free_function_queries
extension (this_work_item::get_nd_item), so kernels keep their original
signatures -- no nd_item threading, no lambda rewriting.
-------------------------------------------------------------------*/

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

//=========================================================================
// 1. Process-wide in-order queue (matches CUDA default-stream semantics)
//=========================================================================
namespace jdftx_sycl {

//! All SYCL GPU devices visible to this process, in a stable order.
//! (Honours ZE_AFFINITY_MASK etc., since that filters what SYCL reports.)
inline const std::vector<sycl::device>& devices()
{	static const std::vector<sycl::device> devs = []
	{	std::vector<sycl::device> result;
		for(const sycl::platform& p: sycl::platform::get_platforms())
			for(const sycl::device& d: p.get_devices(sycl::info::device_type::gpu))
				result.push_back(d);
		if(result.empty()) result.push_back(sycl::device{sycl::default_selector_v});
		return result;
	}();
	return devs;
}

//! Index of the device selected by cudaSetDevice() (JDFTx picks one per process
//! in gpuInit(), round-robin over ranks sharing a node).
inline int& currentDevice()
{	static int iDevice = 0;
	return iDevice;
}

//! In-order queue per device, matching CUDA default-stream semantics.
inline sycl::queue& queue()
{	static std::vector<std::unique_ptr<sycl::queue>> queues(devices().size());
	const int iDevice = currentDevice();
	if(not queues[iDevice])
		queues[iDevice] = std::make_unique<sycl::queue>(devices()[iDevice],
			sycl::property::queue::in_order());
	return *queues[iDevice];
}

//! Pending error, cleared by cudaGetLastError() (CUDA semantics).
inline std::string& lastError()
{	static std::string e;
	return e;
}

//! Text of the most recent error; kept after lastError() is cleared so that
//! cudaGetErrorString() still has something to print (JDFTx's gpuErrorCheck()
//! calls the two in that order).
inline std::string& lastErrorText()
{	static std::string e{"no error"};
	return e;
}

template<typename F> inline void guarded(F&& f)
{	try { f(); }
	catch(const sycl::exception& e) { lastError() = lastErrorText() = e.what(); }
	catch(const std::exception& e) { lastError() = lastErrorText() = e.what(); }
}

//! Set JDFTX_SYCL_TRACE=1 to print and synchronise on every kernel launch.
//! A GPU page fault aborts the process inside the driver rather than raising a
//! catchable exception, so this is the way to find which kernel caused one.
inline bool trace()
{	static const bool enabled = (std::getenv("JDFTX_SYCL_TRACE") != nullptr);
	return enabled;
}

//! Upper bound on the work-group size reported by cudaFuncGetAttributes().
//! SYCL has no per-kernel equivalent of that query, so JDFTx would otherwise
//! size every block at the device maximum. Override with JDFTX_SYCL_MAX_WG.
inline int maxWorkGroupSize()
{	static const int wg = []
	{	int deviceMax = int(queue().get_device()
			.get_info<sycl::info::device::max_work_group_size>());
		if(const char* env = std::getenv("JDFTX_SYCL_MAX_WG"))
		{	int requested = std::atoi(env);
			if(requested > 0) deviceMax = std::min(deviceMax, requested);
		}
		return deviceMax;
	}();
	return wg;
}

} //namespace jdftx_sycl

//=========================================================================
// 2. Error handling (cudaError_t and friends)
//=========================================================================
using cudaError_t = int;
constexpr int cudaSuccess = 0;

inline cudaError_t cudaGetLastError()
{	jdftx_sycl::guarded([]{ jdftx_sycl::queue().wait_and_throw(); });
	if(jdftx_sycl::lastError().empty()) return cudaSuccess;
	jdftx_sycl::lastError().clear(); //consumed, as cudaGetLastError() does
	return 1;
}
inline cudaError_t cudaPeekAtLastError()
{	return jdftx_sycl::lastError().empty() ? cudaSuccess : 1;
}
inline const char* cudaGetErrorString(cudaError_t err)
{	return err ? jdftx_sycl::lastErrorText().c_str() : "no error";
}
inline void checkCudaErrors(cudaError_t) {}

//=========================================================================
// 3. dim3 and the CUDA vector types
//=========================================================================
struct dim3
{	unsigned x, y, z;
	dim3(unsigned x_=1, unsigned y_=1, unsigned z_=1) : x(x_), y(y_), z(z_) {}
};

struct double2 { double x, y; };
struct double3 { double x, y, z; };
struct double4 { double x, y, z, w; };
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };
struct uint2 { unsigned x, y; };
struct uint3 { unsigned x, y, z; };
struct uint4 { unsigned x, y, z, w; };

//=========================================================================
// 4. Thread indexing
//
// CUDA x/y/z are fastest/middle/slowest varying; SYCL nd_range<3> uses
// dimension 2/1/0 for the same roles.  threadIdx & co. are macros that
// build a value on the spot from the free-function work-item query, so
// kernel bodies (and LoopMacros.h) need no changes at all.
//=========================================================================
namespace jdftx_sycl {

__attribute__((always_inline)) inline sycl::nd_item<3> item()
{	return sycl::ext::oneapi::this_work_item::get_nd_item<3>();
}
__attribute__((always_inline)) inline dim3 localId()
{	auto it = item();
	return dim3(unsigned(it.get_local_id(2)), unsigned(it.get_local_id(1)), unsigned(it.get_local_id(0)));
}
__attribute__((always_inline)) inline dim3 groupId()
{	auto it = item();
	return dim3(unsigned(it.get_group(2)), unsigned(it.get_group(1)), unsigned(it.get_group(0)));
}
__attribute__((always_inline)) inline dim3 localRange()
{	auto it = item();
	return dim3(unsigned(it.get_local_range(2)), unsigned(it.get_local_range(1)), unsigned(it.get_local_range(0)));
}
__attribute__((always_inline)) inline dim3 groupRange()
{	auto it = item();
	return dim3(unsigned(it.get_group_range(2)), unsigned(it.get_group_range(1)), unsigned(it.get_group_range(0)));
}

} //namespace jdftx_sycl

#define threadIdx (::jdftx_sycl::localId())
#define blockIdx  (::jdftx_sycl::groupId())
#define blockDim  (::jdftx_sycl::localRange())
#define gridDim   (::jdftx_sycl::groupRange())

#define __syncthreads() (sycl::group_barrier(::jdftx_sycl::item().get_group()))
#define __threadfence_block() (sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group))

//! Declare a pointer to the dynamic shared memory of the current work-group.
//! CUDA spells this `extern __shared__ Type name[];` at namespace scope; SYCL
//! needs a per-kernel local pointer, hence the macro (used at 2 sites).
#define JDFTX_DYNAMIC_SHARED(Type, name) \
	Type* name = (Type*)sycl::ext::oneapi::experimental::get_work_group_scratch_memory()

//=========================================================================
// 5. Math helpers missing from the SYCL device library
//=========================================================================
//! glibc declares sincos() but there is no device definition; supply one.
//! (Same signature as <math.h>, so this is a definition of that function.)
inline void sincos(double x, double* s, double* c) { *s = sycl::sin(x); *c = sycl::cos(x); }
inline void sincosf(float x, float* s, float* c) { *s = sycl::sin(x); *c = sycl::cos(x); }

inline unsigned int __activemask() { return 0; }

//=========================================================================
// 6. Memory management
//=========================================================================
enum cudaMemcpyKind
{	cudaMemcpyHostToHost = 0,
	cudaMemcpyHostToDevice = 1,
	cudaMemcpyDeviceToHost = 2,
	cudaMemcpyDeviceToDevice = 3,
	cudaMemcpyDefault = 4
};

inline cudaError_t cudaMalloc(void** ptr, size_t size)
{	*ptr = nullptr;
	jdftx_sycl::guarded([&]{ *ptr = sycl::malloc_device(size, jdftx_sycl::queue()); });
	return (*ptr) ? cudaSuccess : 1;
}
inline cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned flags=1)
{	*ptr = nullptr;
	jdftx_sycl::guarded([&]{ *ptr = sycl::malloc_shared(size, jdftx_sycl::queue()); });
	return (*ptr) ? cudaSuccess : 1;
}
inline cudaError_t cudaMallocHost(void** ptr, size_t size)
{	*ptr = nullptr;
	jdftx_sycl::guarded([&]{ *ptr = sycl::malloc_host(size, jdftx_sycl::queue()); });
	return (*ptr) ? cudaSuccess : 1;
}
inline cudaError_t cudaFree(void* ptr)
{	if(ptr) jdftx_sycl::guarded([&]{ sycl::free(ptr, jdftx_sycl::queue()); });
	return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* ptr) { return cudaFree(ptr); }

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind)
{	jdftx_sycl::guarded([&]{ jdftx_sycl::queue().memcpy(dst, src, n).wait(); });
	return cudaSuccess;
}
inline cudaError_t cudaMemset(void* ptr, int val, size_t n)
{	jdftx_sycl::guarded([&]{ jdftx_sycl::queue().memset(ptr, (unsigned char)val, n).wait(); });
	return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize()
{	jdftx_sycl::guarded([]{ jdftx_sycl::queue().wait_and_throw(); });
	return jdftx_sycl::lastError().empty() ? cudaSuccess : 1;
}

//! No host-direction equivalent in SYCL; a device-direction hint is all we can do.
inline cudaError_t cudaMemPrefetchAsync(const void* ptr, size_t n, int dstDevice, int stream=0)
{	if(dstDevice >= 0) jdftx_sycl::guarded([&]{ jdftx_sycl::queue().prefetch(ptr, n); });
	return cudaSuccess;
}

//=========================================================================
// 7. Device properties / introspection
//=========================================================================
struct cudaDeviceProp
{	int maxThreadsPerBlock;
	int warpSize;
	int maxThreadsPerMultiProcessor;
	unsigned int maxGridSize[3];
	int maxThreadsDim[3];
	int multiProcessorCount;
	size_t sharedMemPerBlock;
	size_t totalGlobalMem;
	int major, minor;
	int integrated;
	int computeMode;
	char name[256];
};

struct cudaFuncAttributes
{	int maxThreadsPerBlock;
	size_t sharedSizeBytes;
};

constexpr int cudaDevAttrConcurrentManagedAccess = 64;
constexpr int cudaCpuDeviceId = -1;

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device = -1)
{	const std::vector<sycl::device>& devs = jdftx_sycl::devices();
	if(device < 0 or device >= int(devs.size())) device = jdftx_sycl::currentDevice();
	const sycl::device& dev = devs[device];
	const int wg = int(dev.get_info<sycl::info::device::max_work_group_size>());
	prop->maxThreadsPerBlock = wg;
	prop->warpSize = int(dev.get_info<sycl::info::device::sub_group_sizes>().back());
	prop->maxThreadsPerMultiProcessor = wg;
	auto mi = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
	prop->maxThreadsDim[0] = int(mi[2]); //CUDA x <-> SYCL dim 2
	prop->maxThreadsDim[1] = int(mi[1]);
	prop->maxThreadsDim[2] = int(mi[0]);
	prop->maxGridSize[0] = prop->maxGridSize[1] = prop->maxGridSize[2] = (1u << 30);
	prop->sharedMemPerBlock = size_t(dev.get_info<sycl::info::device::local_mem_size>());
	prop->totalGlobalMem = size_t(dev.get_info<sycl::info::device::global_mem_size>());
	prop->multiProcessorCount = int(dev.get_info<sycl::info::device::max_compute_units>());
	//JDFTx's gpuInit() requires (major,minor) >= (1,3) for double precision and
	//rejects integrated devices; report a discrete, double-capable device.
	prop->major = 9; prop->minor = 0;
	prop->integrated = 0; //gpuInit() rejects integrated devices; PVC is discrete
	prop->computeMode = 0;
	std::snprintf(prop->name, sizeof(prop->name), "%s", dev.get_info<sycl::info::device::name>().c_str());
	return cudaSuccess;
}

//! Kernel attributes: SYCL has no per-kernel query at this level, so report
//! the device work-group limit (JDFTx only uses maxThreadsPerBlock).
template<typename K> inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, K*)
{	a->maxThreadsPerBlock = jdftx_sycl::maxWorkGroupSize();
	a->sharedSizeBytes = 0;
	return cudaSuccess;
}
template<typename K> inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, K& k)
{	return cudaFuncGetAttributes(a, &k);
}

inline cudaError_t cudaGetDeviceCount(int* count)
{	*count = int(jdftx_sycl::devices().size());
	return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device)
{	if(device < 0 or device >= int(jdftx_sycl::devices().size())) return 1;
	jdftx_sycl::currentDevice() = device;
	jdftx_sycl::queue(); //create it now, so failures surface here
	return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* device) { *device = jdftx_sycl::currentDevice(); return cudaSuccess; }
inline cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp*) { *device = 0; return cudaSuccess; }

inline cudaError_t cudaDeviceGetAttribute(int* value, int attr, int /*device*/)
{	//Report no concurrent managed access: that keeps JDFTx's prefetchSupported
	//false, and cudaMemPrefetchAsync() has no host-direction equivalent in SYCL.
	*value = 0;
	(void)attr;
	return cudaSuccess;
}

//=========================================================================
// 8. Kernel launch
//
// launchKernel() reproduces CUDA's by-value argument semantics: every
// argument is copied into the parameter pack and then captured by value in
// the kernel lambda.  The kernel itself is wrapped in a capture-less
// generic lambda by the JDFTX_LAUNCH macros in GpuKernelUtils.h, which
// keeps the call device-side and avoids function pointers.
//=========================================================================
namespace jdftx_sycl {

inline sycl::nd_range<3> ndRange(const dim3& nBlocks, const dim3& nPerBlock)
{	return sycl::nd_range<3>(
		sycl::range<3>(size_t(nBlocks.z)*nPerBlock.z, size_t(nBlocks.y)*nPerBlock.y, size_t(nBlocks.x)*nPerBlock.x),
		sycl::range<3>(nPerBlock.z, nPerBlock.y, nPerBlock.x));
}

//! Byte image of a kernel argument that must not be copy-constructed on the
//! device. RadialFunctionG is the case that matters: it carries a std::vector
//! for its host-side copy, and reconstructing that on the device would need
//! heap allocation (the device only ever reads dGinv/nCoeff/coeffGpu). Raw is
//! trivially copyable, so SYCL ships it as bytes -- exactly what CUDA does with
//! every kernel argument. Kernels declare such parameters with
//! JDFTX_KERNEL_ARG(), which is a reference under SYCL and by-value on CUDA.
template<typename T> struct Raw
{	alignas(T) unsigned char bytes[sizeof(T)];
	Raw() = default;
	Raw(const T& v) { __builtin_memcpy(bytes, &v, sizeof(T)); }
	const T& ref() const { return *reinterpret_cast<const T*>(bytes); }
};

//! What the launcher stores for one kernel parameter, and how it hands it over.
//! Naming Store through KernelArg also blocks template deduction, so the
//! kernel's own parameter list drives the conversion of every argument -- that
//! is what reproduces the implicit conversions `kernel<<<...>>>(args)` performs
//! (std::vector -> array<>, GpuBuffer -> double*, ...).
template<typename P> struct KernelArg
{	using Store = P;
	static const P& get(const Store& s) { return s; }
};
template<typename P> struct KernelArg<const P&>
{	using Store = Raw<P>;
	static const P& get(const Store& s) { return s.ref(); }
};

inline void traceBegin(const char* label, const dim3& nBlocks, const dim3& nPerBlock)
{	std::fprintf(stderr, "[sycl] launch %s grid=(%u,%u,%u) block=(%u,%u,%u)\n", label,
		nBlocks.x, nBlocks.y, nBlocks.z, nPerBlock.x, nPerBlock.y, nPerBlock.z);
	std::fflush(stderr);
}
inline void traceEnd(const char* label)
{	guarded([]{ queue().wait_and_throw(); });
	std::fprintf(stderr, "[sycl] done   %s%s%s\n", label,
		lastError().empty() ? "" : " -- ", lastError().c_str());
	std::fflush(stderr);
}

template<typename F, typename... Params>
inline void launchKernel(const char* label, const dim3& nBlocks, const dim3& nPerBlock,
	F f, void(*)(Params...), typename KernelArg<Params>::Store... args)
{	if(trace()) traceBegin(label, nBlocks, nPerBlock);
	guarded([&]
	{	queue().parallel_for(ndRange(nBlocks, nPerBlock),
			[=](sycl::nd_item<3>) { f(KernelArg<Params>::get(args)...); });
	});
	if(trace()) traceEnd(label);
}

template<typename F, typename... Params>
inline void launchKernelShared(const char* label, const dim3& nBlocks, const dim3& nPerBlock,
	size_t sharedBytes, F f, void(*)(Params...), typename KernelArg<Params>::Store... args)
{	namespace ex = sycl::ext::oneapi::experimental;
	if(trace()) traceBegin(label, nBlocks, nPerBlock);
	guarded([&]
	{	ex::nd_launch(queue(),
			ex::launch_config(ndRange(nBlocks, nPerBlock),
				ex::properties{ex::work_group_scratch_size(sharedBytes)}),
			[=](sycl::nd_item<3>) { f(KernelArg<Params>::get(args)...); });
	});
	if(trace()) traceEnd(label);
}

} //namespace jdftx_sycl

//=========================================================================
// 9. Device-copyability of JDFTx types that SYCL cannot deduce
//
// vector3/matrix3/tensor3/symmetricMatrix3 hold plain scalar arrays but declare
// user-provided copy constructors, and RadialFunctionG carries a std::vector
// for its CPU-side copy -- so none of them are trivially copyable, even though
// a bitwise copy to the device is exactly what CUDA already does with them.
// (RadialFunctionG::getCoeff() selects the GPU pointer under __CUDA_ARCH__, so
// the vector member is never touched on the device.)
//=========================================================================
template<typename scalar> class vector3;
template<typename scalar> class matrix3;
template<typename scalar> class tensor3;
template<typename scalar> struct symmetricMatrix3;
struct RadialFunctionG;

namespace sycl {
template<typename T> struct is_device_copyable<vector3<T>> : std::true_type {};
template<typename T> struct is_device_copyable<matrix3<T>> : std::true_type {};
template<typename T> struct is_device_copyable<tensor3<T>> : std::true_type {};
template<typename T> struct is_device_copyable<symmetricMatrix3<T>> : std::true_type {};
template<> struct is_device_copyable<RadialFunctionG> : std::true_type {};
}
