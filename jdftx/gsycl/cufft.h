#pragma once
// cuFFT → oneMKL DFT shim
// Maps cufft API to Intel MKL DFTI calls.
// KEY: JDFTx uses OUT-OF-PLACE UNPACKED layout, which is the default for
// oneMKL when DFTI_CONJUGATE_EVEN_STORAGE=DFTI_NOT_APPLICABLE.
// Set strides explicitly to match CUDA's CCE (Conjugate Complex Even) layout.

#include <mkl_dfti.h>
#include <sycl/sycl.hpp>
#include "sycl_device.hpp"

// =====================================================================
// Type aliases and stubs
// =====================================================================
typedef int cudaError_t;
inline constexpr int cudaSuccess = 0;

// =====================================================================
// cuFFT → oneMKL mapping
//
// JDFTx uses these types:
//   cufftHandle planZ2Z, planD2Z, planZ2D  (in GridInfo)
//   cufftPlan3d(&plan, S[0],S[1],S[2], CUFFT_*)  (in GridInfo.cpp)
//   cufftExecZ2D/Z2Z/D2Z(...)                  (in Operators.cpp)
//   cufftDestroy(...)                          (in GridInfo.cpp)
//
// Handle: wrap oneMKL descriptor
typedef struct {
    DFTI_DESCRIPTOR_HANDLE desc;
    bool initialized;
} cufftHandle;

// =====================================================================
// cufftPlan3d → DftiMakeDescriptor
//
// JDFTx uses:
//   cufftPlan3d(&plan, S[0],S[1],S[2], CUFFT_Z2Z)
//   cufftPlan3d(&plan, S[0],S[1],S[2], CUFFT_D2Z)  (R→G)
//   cufftPlan3d(&plan, S[0],S[1],S[2], CUFFT_Z2D)  (G→R)
//
// oneMKL equivalent:
//   DftiMakeDescriptor(handle, MKL_Z2Z or MKL_D2Z or MKL_Z2D,
//                      MKL_COMPLEX or double, 3, dims)
//   DftiSetValue(handle, DFTI_COMPLEX_OUTPUT, ...)  // controls packed/unpacked
//   DftiCommitDescriptor(handle)
//
// CRITICAL: oneMKL defaults to IN-PLACE PADDED for real transforms.
// JDFTx always does OUT-OF-PLACE. Set DFTI_CONJUGATE_EVEN_STORAGE
// to DFTI_NOT_APPLICABLE for complex transforms (Z2Z) — no packing needed.
// For real transforms (D2Z, Z2D), set DFTI_PACKED_FORMAT = DFTI_NOT_APPLICABLE
// to get unpacked output.
// =====================================================================

inline cudaError_t cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, int type) {
    plan->desc = 0;
    plan->initialized = false;

    // Create descriptor
    int rank = 3;
    size_t n[3] = { (size_t)nz, (size_t)ny, (size_t)nx };  // oneMKL: slowest→fastest

    switch(type) {
        case CUFFT_Z2Z:
            // Complex → Complex, unpacked (default)
            DftiMakeDescriptor(&plan->desc, MKL_Z2Z, MKL_DOUBLE, rank, n);
            DftiSetValue(plan->desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_NOT_APPLICABLE);
            DftiSetValue(plan->desc, DFTI_INPUT_STRIDES, (size_t[3]){0, 0, 0});
            DftiSetValue(plan->desc, DFTI_OUTPUT_STRIDES, (size_t[3]){0, 0, 0});
            DftiSetValue(plan->desc, DFTI_OUTPUT_BATCH_STRIDE,
                         (size_t)nz * ny * sizeof(double) * 2);
            DftiSetValue(plan->desc, DFTI_INPUT_BATCH_STRIDE,
                         (size_t)nz * ny * sizeof(double) * 2);
            break;

        case CUFFT_D2Z:  // Real (double) → Complex (Z2Z)
            DftiMakeDescriptor(&plan->desc, MKL_D2Z, MKL_DOUBLE, rank, n);
            // Real→Complex: output is conjugate-even packed by default
            // JDFTx uses UNPACKED, so:
            DftiSetValue(plan->desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            DftiSetValue(plan->desc, DFTI_PACKED_FORMAT, DFTI_NOT_APPLICABLE);
            DftiSetValue(plan->desc, DFTI_INPUT_STRIDES, (size_t[3]){0, 0, 0});
            DftiSetValue(plan->desc, DFTI_OUTPUT_STRIDES, (size_t[3]){0, 0, 0});
            DftiSetValue(plan->desc, DFTI_OUTPUT_BATCH_STRIDE,
                         (size_t)(nz/2 + 1) * ny * sizeof(double) * 2);
            DftiSetValue(plan->desc, DFTI_INPUT_BATCH_STRIDE,
                         (size_t)nz * ny * sizeof(double));
            break;

        case CUFFT_Z2D:  // Complex → Real (double)
            DftiMakeDescriptor(&plan->desc, MKL_Z2D, MKL_DOUBLE, rank, n);
            DftiSetValue(plan->desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            DftiSetValue(plan->desc, DFTI_PACKED_FORMAT, DFTI_NOT_APPLICABLE);
            DftiSetValue(plan->desc, DFTI_INPUT_STRIDES, (size_t[3]){0, 0, 0});
            DftiSetValue(plan->desc, DFTI_OUTPUT_STRIDES, (size_t[3]){0, 0, 0});
            DftiSetValue(plan->desc, DFTI_INPUT_BATCH_STRIDE,
                         (size_t)(nz/2 + 1) * ny * sizeof(double) * 2);
            DftiSetValue(plan->desc, DFTI_OUTPUT_BATCH_STRIDE,
                         (size_t)nz * ny * sizeof(double));
            break;

        default:
            return 1;  // unsupported
    }

    DftiCommitDescriptor(plan->desc);
    plan->initialized = true;
    return cudaSuccess;
}

// =====================================================================
// cufftExecZ2Z → DftiComputeForward/Backward
// direction: CUFFT_FORWARD or CUFFT_INVERSE
// =====================================================================
inline cudaError_t cufftExecZ2Z(cufftHandle plan, const cuDoubleComplex* in,
                                 cuDoubleComplex* out, int direction) {
    if (!plan.initialized) return 1;

    if (direction == CUFFT_FORWARD || direction == CUFFT_INVERSE) {
        DftiComputeDescriptor(
            (direction == CUFFT_FORWARD) ? DFTI_FORWARD : DFTI_BACKWARD,
            plan.desc,
            (const MKL_Complex16*)in,
            (MKL_Complex16*)out);
    }
    return cudaSuccess;
}

// =====================================================================
// cufftExecD2Z → DftiComputeForward
// =====================================================================
inline cudaError_t cufftExecD2Z(cufftHandle plan, const double* in,
                                 cuDoubleComplex* out) {
    if (!plan.initialized) return 1;

    DftiComputeDescriptor(DFTI_FORWARD, plan.desc,
                          (const double*)in,
                          (MKL_Complex16*)out);
    return cudaSuccess;
}

// =====================================================================
// cufftExecZ2D → DftiComputeBackward
// =====================================================================
inline cudaError_t cufftExecZ2D(cufftHandle plan, const cuDoubleComplex* in,
                                 double* out) {
    if (!plan.initialized) return 1;

    DftiComputeDescriptor(DFTI_BACKWARD, plan.desc,
                          (const MKL_Complex16*)in,
                          (double*)out);
    return cudaSuccess;
}

// =====================================================================
// cufftDestroy
// =====================================================================
inline cudaError_t cufftDestroy(cufftHandle plan) {
    if (plan.initialized && plan.desc) {
        DftiFreeDescriptor(&plan.desc);
    }
    plan.initialized = false;
    plan.desc = 0;
    return cudaSuccess;
}
