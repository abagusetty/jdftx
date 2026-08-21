#pragma once
/*-------------------------------------------------------------------
gsycl/cuda_prologue.hpp -- force-included (-include) into the .cu
translation units of the SYCL build, before anything else.

It plays the role nvcc plays for a CUDA build:
  * defines the CUDA function qualifiers, which makes core/scalar.h take
    its "in a .cu file" branch (__hostanddev__, __in_a_cu_file__);
  * defines __CUDA_ARCH__ during the SYCL *device* compilation pass, so
    the existing #ifdef __CUDA_ARCH__ host/device splits in
    RadialFunction.h and Spline.h select the device path unchanged.

It is deliberately NOT applied to .cpp sources: those must keep seeing
the host path, exactly as under nvcc.
-------------------------------------------------------------------*/

//SYCL needs no per-function qualifiers: everything is ordinary C++ that
//the device compiler inlines into the kernel.
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __constant__
#define __constant__ const
#endif

//Device pass of the SYCL compiler == CUDA device pass.
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 800
#endif

#include "sycl_device.hpp"
