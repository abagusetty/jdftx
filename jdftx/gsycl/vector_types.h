#pragma once
// Vector types for JDFTx SYCL port
// Replaces <vector_types.h> from CUDA toolkit
// Provides: dim3, double2, vector3 types used in kernel index math

#include <cmath>
#include "sycl_device.hpp"

// dim3 is defined in sycl_device.hpp as SYCL-compatible wrapper
// Provide additional vector3 alias if JDFTx uses it
typedef struct {
    int x, y, z;
    int& operator[](int i) { return (&x)[i]; }
    int operator[](int i) const { return (&x)[i]; }
} vector3_int;
