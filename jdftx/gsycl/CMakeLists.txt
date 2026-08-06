# CMakeLists.txt for jdftx gsycl SYCL shim
#
# This provides:
#   1. sycl_compat — INTERFACE target injecting gsycl/ include path
#   2. When building with SYCL, all #include <cuda_runtime.h> etc.
#      resolve to the gsycl shim headers instead of the CUDA toolkit

# Create INTERFACE library that adds gsycl/ to include paths
add_library(jdftx_sycl_compat INTERFACE)
target_include_directories(jdftx_sycl_compat INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/jdftx/gsycl>"
  "$<INSTALL_INTERFACE:include/gsycl>"
)
