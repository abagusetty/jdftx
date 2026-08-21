#pragma once
// Drop-in replacement for the CUDA toolkit's <driver_types.h> when building
// with SYCL: cudaError_t, cudaMemcpyKind and cudaDeviceProp live in
// sycl_device.hpp.
#include "sycl_device.hpp"
