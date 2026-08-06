#pragma once
// Drop-in replacement for <cuda.h> when building with SYCL.
// This is the main entry point — all CUDA code includes <cuda.h> or <cuda_runtime.h>.
#include "sycl_device.hpp"
