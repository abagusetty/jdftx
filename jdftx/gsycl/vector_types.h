#pragma once
// Drop-in replacement for the CUDA toolkit's <vector_types.h> when building
// with SYCL: dim3 and the doubleN/floatN/intN types live in sycl_device.hpp.
#include "sycl_device.hpp"
