cmake_minimum_required(VERSION 3.12)

project(JDFTx)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake-Modules/")

#Package configuration:
set(CPACK_PACKAGE_NAME "JDFTx")
set(CPACK_PACKAGE_DESCRIPTION "The playground for joint density functional theory")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Toolkit for rapid development of density-functional methods")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "7")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION_STRING "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

#Get revision information
find_package(Git)
if(GIT_FOUND)
	execute_process(
		COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-work-tree
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE IS_GIT
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if("true" STREQUAL "${IS_GIT}")	
		execute_process(
			COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
			WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
			OUTPUT_VARIABLE GIT_HASH
			OUTPUT_STRIP_TRAILING_WHITESPACE)
		message(STATUS "Git revision hash: ${GIT_HASH}")
	endif()
endif()

#Static linking:
option(StaticLinking "Link statically instead of dynamically" OFF)
if(CYGWIN) #Force static linking on Windows:
	if(NOT StaticLinking)
		message(STATUS "Forcing static linking for Cygwin/Windows")
		set(StaticLinking ON)
	endif()
endif()
if(StaticLinking)
	if(WIN32)
		set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
	else()
		set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
	endif()
	set(CMAKE_EXE_LINKER_FLAGS -static)
	set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
	set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
	set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
	set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
	set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
endif()

#Intel GPU support using SYCL (declared early: it selects between the CBLAS/LAPACK
#packages below and oneMKL, and gates the gsycl/ CUDA-compatibility shim)
option(USE_SYCL "Enable Intel PVC (SYCL) GPU support" OFF)

#GSL: honor an explicit GSL_PATH (for installs outside the system prefixes), else find it
if(GSL_PATH)
	message(STATUS "Using GSL from GSL_PATH: ${GSL_PATH}")
	find_path(GSL_INCLUDE_DIR gsl/gsl_cblas.h PATHS ${GSL_PATH}/include NO_DEFAULT_PATH)
	find_library(GSL_LIBRARY NAMES gsl PATHS ${GSL_PATH}/lib ${GSL_PATH}/lib64 NO_DEFAULT_PATH)
	if((NOT GSL_INCLUDE_DIR) OR (NOT GSL_LIBRARY))
		message(FATAL_ERROR "GSL not found under GSL_PATH=${GSL_PATH}")
	endif()
	set(GSL_FOUND TRUE)
else()
	find_package(GSL REQUIRED)
endif()
include_directories(${GSL_INCLUDE_DIR})

#Use indepedent CBLAS, LAPACK and FFTW (default) or MKL for all three
option(EnableMKL "Use Intel MKL to provide BLAS, LAPACK and FFTs")
option(ForceFFTW "Force usage of FFTW (even if MKL is enabled)")
option(ThreadedBLAS "Used built-in threading of the BLAS library if yes; thread in JDFTx if no (currently affects only MKL)" ON)
option(EnableScaLAPACK "Enable ScaLAPACK support (currently used only by the BerkeleyGW output option)")
option(ForceScaLAPACK "Force usage of an external ScaLAPACK when MKL is enabled (to circumvent MKL ScaLAPACK bugs)")
option(EnableLibSci "Use built-in LibSci support (for BLAS, LAPACK and ScaLAPACK if enabled) from Cray compiler wrapper.")
set(CMAKE_THREAD_PREFER_PTHREAD)
find_package(Threads REQUIRED)
if(EnableMKL)
	find_package(MKL REQUIRED)
	set(CBLAS_LAPACK_FFT_LIBRARIES ${MKL_LIBRARIES}) #MKL provides CBLAS, FFTW3 and LAPACK
	add_definitions("-DMKL_PROVIDES_BLAS") #Special handling is required for BLAS calls from threaded sections
	include_directories(${MKL_INCLUDE_DIR})
	if(ForceFFTW)
		find_package(FFTW3 REQUIRED)
		set(CBLAS_LAPACK_FFT_LIBRARIES ${FFTW3_THREADS_LIBRARY} ${FFTW3_LIBRARY} ${MKL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}) #Explicit FFTW3, rest from MKL
	else()
		add_definitions("-DMKL_PROVIDES_FFT") #Special handling is required for FFT initialization
		set(CBLAS_LAPACK_FFT_LIBRARIES ${MKL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}) #MKL provides CBLAS, FFTW3 and LAPACK
	endif()
	if(ThreadedBLAS)
		add_definitions("-DTHREADED_BLAS")
	endif()
elseif(EnableLibSci)
	message(STATUS "Assuming built-in LibSci support from Cray compiler wrappers (not checked).")
	find_package(FFTW3 REQUIRED)
	if(EnableScaLAPACK)
		add_definitions("-DSCALAPACK_ENABLED")  # built-in
	endif()
	set(CBLAS_LAPACK_FFT_LIBRARIES ${FFTW3_THREADS_LIBRARY} ${FFTW3_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
else()
	find_package(FFTW3 REQUIRED)
	if(USE_SYCL)
		#oneMKL provides CBLAS and LAPACK for the host side (LP64: JDFTx and GSL
		#both call them with 32-bit integers). The SYCL device domains
		#(MKL::MKL_SYCL::BLAS/LAPACK/DFT) are linked onto the GPU library below,
		#from this same find_package.
		set(MKL_INTERFACE lp64)
		find_package(MKL REQUIRED CONFIG PATHS "$ENV{MKLROOT}")
		#oneMKL's SYCL targets carry -fsycl-allow-device-image-dependencies as an
		#INTERFACE_LINK_LIBRARIES item, so it always lands after anything we add
		#and wins. With per-kernel device code split that leaves kernel images
		#referencing a separate shared image, which the runtime links at launch --
		#fine for SPIR-V, but AOT native binaries cannot be linked at runtime
		#(urProgramLinkExp -> UR_RESULT_ERROR_INVALID_ARGUMENT). Flip it in place.
		foreach(mklTarget MKL::MKL_SYCL MKL::MKL_SYCL::BLAS MKL::MKL_SYCL::LAPACK MKL::MKL_SYCL::DFT)
			if(TARGET ${mklTarget})
				get_target_property(mklLinkItems ${mklTarget} INTERFACE_LINK_LIBRARIES)
				if(mklLinkItems)
					string(REPLACE "-fsycl-allow-device-image-dependencies"
						"-fno-sycl-allow-device-image-dependencies" mklLinkItems "${mklLinkItems}")
					set_target_properties(${mklTarget} PROPERTIES INTERFACE_LINK_LIBRARIES "${mklLinkItems}")
				endif()
			endif()
		endforeach()
		set(CBLAS_LAPACK_FFT_LIBRARIES ${FFTW3_THREADS_LIBRARY} ${FFTW3_LIBRARY} MKL::MKL ${CMAKE_THREAD_LIBS_INIT})
		#MKLConfig.cmake leaves LINK_PREFIX="-l" / LINK_SUFFIX behind in this
		#directory scope; they would otherwise be picked up by the link options
		#set just below and by add_JDFTx_executable(), emitting a bare "-l".
		unset(LINK_PREFIX)
		unset(LINK_SUFFIX)
	else()
		find_package(LAPACK_ATLAS REQUIRED)
		find_package(CBLAS REQUIRED)
		set(CBLAS_LAPACK_FFT_LIBRARIES ${FFTW3_THREADS_LIBRARY} ${FFTW3_LIBRARY} ${CBLAS_LIBRARY} ${LAPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
	endif()
endif()
include_directories(${FFTW3_INCLUDE_DIR})

if(EnableScaLAPACK AND ((NOT (EnableMKL OR EnableLibSci) OR ForceScaLAPACK)))
	find_package(ScaLAPACK REQUIRED)
	set(CBLAS_LAPACK_FFT_LIBRARIES ${SCALAPACK_LIBRARY} ${CBLAS_LAPACK_FFT_LIBRARIES})
	add_definitions("-DSCALAPACK_ENABLED")
endif()

option(EnableMPI "Use MPI parallelization (in addition to threads / gpu)" ON)
if(EnableMPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_INCLUDE_PATH})
	set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${MPI_INCLUDE_PATH}") #To find libraries installed to MPI specific paths
	add_definitions("-DMPI_ENABLED")
endif()

option(EnableLibXC "Use LibXC to provide additional exchange-correlation functionals")
option(CudaLibXC "Indicate that LibXC is built with CUDA (if so, libxc will only work from jdftx_gpu)")
if(EnableLibXC)
	find_package(LIBXC REQUIRED)
	include_directories(${LIBXC_INCLUDE_DIR})
	add_definitions("-DLIBXC_ENABLED")
	if(CudaLibXC)
		add_definitions("-DLIBXC_CUDA")
	endif()
	#Generate functional list available from LibXC:
	execute_process(COMMAND ${CMAKE_SOURCE_DIR}/opt/indexLibXC.sh ${LIBXC_INCLUDE_DIR}/xc_funcs.h OUTPUT_FILE ${CMAKE_BINARY_DIR}/xcMap.h)
endif()

option(EnableHDF5 "Enable HDF5 features (currently required only by Berkeley GW dump option)")
if(EnableHDF5)
	find_package(HDF5 REQUIRED)
	include_directories(${HDF5_INCLUDE_DIRS})
	add_definitions("-DHDF5_ENABLED")
endif()

#Process configuration information into config.h (with config.in.h as a template)
configure_file(${CMAKE_SOURCE_DIR}/config.in.h ${CMAKE_BINARY_DIR}/config.h)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

option(EnableProfiling "Optionally enable profiling to determine ideal functions to optimize (for developers)")
if(EnableProfiling)
	add_definitions("-DENABLE_PROFILING")
endif()

#------------------------- Compilation flags ------------------------

#Base flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -Wall")

#Additional compiler-dependent flags:
include(CheckCXXCompilerFlag)
#--- Suppress gcc warning
check_cxx_compiler_flag(-Wno-unused-result HAS_NO_UNUSED_RESULT)
if(HAS_NO_UNUSED_RESULT)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()
option(LinkTimeOptimization "Enable link-time optimizations (eg. flto on GNU, ipo on Intel).")
if(LinkTimeOptimization)
	#--- Link-time optimizations (GCC / CLang)
	if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
		check_cxx_compiler_flag(-flto HAS_FLTO)
		if(HAS_FLTO)
			set(JDFTX_CPU_FLAGS "${JDFTX_CPU_FLAGS} -flto")
		endif()
	endif()
	#--- Link-time optimizations (ICC)
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		check_cxx_compiler_flag(-ipo HAS_IPO)
		if(HAS_IPO)
			set(JDFTX_CPU_FLAGS "${JDFTX_CPU_FLAGS} -ipo")
		endif()
	endif()
endif()
#--- Floating-point model: strict by default.
# icpx defaults to -fp-model=fast, which permits reassociation and other
# value-changing transforms, and even -fp-model=precise leaves FMA contraction
# enabled. Neither default is acceptable here. JDFTx has kernels that rely on
# exact cancellation -- CoulombSlab_calc's in-plane |G| in
# core/Coulomb_internal.h computes it as a difference that must come out
# exactly zero -- and host and device codegen decide contraction
# independently, so the CPU and GPU builds can disagree for reasons that have
# nothing to do with the port and are very hard to attribute after the fact.
# Both are therefore disabled by default, uniformly, for every target.
#
# For the record: this was NOT the cause of the graphene Fermi-level error
# (that was a c2r Nyquist convention mismatch, fixed in gsycl/cufft_impl.cpp).
# Setting -ffp-contract=off did not move that result by even 1e-9. It is the
# default because a reproducible floating-point model is worth more than the
# contraction, not because it is known to have fixed something.
# Only turn this off (-DPreciseFP=OFF) if you have re-validated the test suite.
option(PreciseFP "Compile with a strict (non-fast-math) floating-point model" ON)
if(PreciseFP)
	# Order matters defensively: Clang documents -fno-fast-math as implying
	# -ffp-contract=on, so it goes first and the explicit contract setting last.
	# (icpx 2026.1 keeps -ffp-contract=off either way -- checked with -###.)
	foreach(fpFlag "-fno-fast-math" "-fp-model=precise" "-ffp-contract=off")
		string(MAKE_C_IDENTIFIER "HAS${fpFlag}" fpFlagVar)
		check_cxx_compiler_flag("${fpFlag}" ${fpFlagVar})
		if(${fpFlagVar})
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${fpFlag}")
		endif()
	endforeach()
endif()

#--- Template recursion depth (more needed in Clang)
check_cxx_compiler_flag(-ftemplate-depth-512 HAS_TEMPLATE_DEPTH)
if(HAS_TEMPLATE_DEPTH)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth-512")
endif()

#Architecture dependent optimizations
option(CompileNative "Enable aggressive architecture-dependent optimizations for current CPU.")
if(CompileNative)
	#--- GCC architecture-dependent optimization
	check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
	if(HAS_MARCH_NATIVE)
		set(JDFTX_CPU_FLAGS "${JDFTX_CPU_FLAGS} -march=native")
	endif()
	#--- Intel architecture-dependent optimization
	# Deliberately NOT -fast: on the Intel compilers that expands to
	# -ipo -O3 -static -fp-model=fast=2 -xHost, and the fp-model part would
	# undo the strict floating-point model set above. Take only the -xHost part.
	if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
		check_cxx_compiler_flag(-xHost HAS_XHOST)
		if(HAS_XHOST)
			set(JDFTX_CPU_FLAGS "${JDFTX_CPU_FLAGS} -xHost")
		endif()
	endif()
endif()

#MPI I/O options:
option(MPISafeWrite "Safe mode for MPI writes: write from one process; may be needed on some NFS installations")
if(MPISafeWrite)
	add_definitions("-DMPI_SAFE_WRITE")
endif()

#Workarounds for Windows compilation:
if(CYGWIN)
	add_definitions("-D_GNU_SOURCE")
	file(WRITE ${CMAKE_BINARY_DIR}/execinfo.h
"#ifndef EXECINFO_H
#define EXECINFO_H
inline int backtrace(void **buffer, int size) { return 0; }
inline char **backtrace_symbols(void *const *buffer, int size) { return 0; }
#endif //EXECINFO_H
"
	)
endif()

#------------- Macro to set JDFTx flags -------------

macro(set_JDFTx_flags targetName isGpu)
	if(${isGpu})
		set(GPU_ENABLED "-DGPU_ENABLED")
	else()
		set(GPU_ENABLED "")
	endif()
	set_target_properties(${targetName} PROPERTIES
		COMPILE_FLAGS "${EXTRA_CXX_FLAGS} ${GPU_ENABLED}"
		LINK_FLAGS "${EXTRA_CXX_FLAGS} ${MPI_CXX_LINK_FLAGS}"
		INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
		INSTALL_RPATH_USE_LINK_PATH TRUE)
endmacro()

#----------------------- Regular CPU targets ----------------

#External libraries to link to
set(EXTERNAL_LIBS ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GSL_LIBRARY} ${CBLAS_LAPACK_FFT_LIBRARIES} ${LIBXC_LIBRARY} ${EXTRA_LIBRARIES})

#Link options:
if(StaticLinking)
	set(LINK_TYPE STATIC)
	set(LINK_PREFIX -Wl,--whole-archive)
	set(LINK_SUFFIX -Wl,--no-whole-archive)
else()
	set(LINK_TYPE SHARED)
endif()

#Library with all the functionality:
FILE(GLOB jdftxlibSources core/*.cpp fluid/*.cpp electronic/*.cpp perturb/*.cpp commands/*.cpp)
add_library(jdftxlib ${LINK_TYPE} ${jdftxlibSources})
target_link_libraries(jdftxlib ${EXTERNAL_LIBS})
set_target_properties(jdftxlib PROPERTIES OUTPUT_NAME "jdftx")
install(TARGETS jdftxlib LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
set_JDFTx_flags(jdftxlib OFF)

#--------------- Nvidia GPU support using CUDA -----------------------
#----- Add another set of libraries and executables with a _gpu suffix
option(EnableCUDA "Enable NVIDIA GPU support using CUDA" OFF)
option(EnableCuSolver "Whether to use cuSolver GPU LAPACK (Requires CUDA >= 9)" OFF)
if(EnableCUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS}")
	include_directories(${CUDAToolkit_INCLUDE_DIRS})

	#Library with all the functionality:
	FILE(GLOB jdftxlibSources core/*.cpp fluid/*.cpp electronic/*.cpp perturb/*.cpp commands/*.cpp)
	FILE(GLOB gpukernelsSources core/*.cu fluid/*.cu electronic/*.cu commands/*.cu tests/*.cu)
	add_library(jdftxlib_gpu ${LINK_TYPE} ${jdftxlibSources} ${gpukernelsSources})
	target_link_libraries(jdftxlib_gpu ${EXTERNAL_LIBS} CUDA::cublas CUDA::cufft)
	set_target_properties(jdftxlib_gpu PROPERTIES OUTPUT_NAME "jdftx_gpu")
	install(TARGETS jdftxlib_gpu LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
	set_JDFTx_flags(jdftxlib_gpu ON)

	#Add cuSolver support:
	if(EnableCuSolver)
		target_link_libraries(jdftxlib_gpu CUDA::cusolver)
		add_definitions("-DCUSOLVER_ENABLED")
	endif()
endif()
#--------------- Intel GPU support using SYCL (AOT PVC build) ------------------------
#(the USE_SYCL option itself is declared near the top, before the BLAS/LAPACK search)
if(USE_SYCL)
    if(EnableCUDA)
        message(FATAL_ERROR "EnableCUDA and USE_SYCL are mutually exclusive: gsycl/ shadows the CUDA toolkit headers")
    endif()
    message(STATUS "Building with SYCL support (USE_SYCL=ON)")

    # Honor user-supplied CMAKE_BUILD_TYPE; default to Release.
    # For AOT (-fsycl-targets=spir64_gen) a full -g injects into the device path
    # and reaches ocloc which disables optimization. We downgrade C++ -g to
    # -gline-tables-only so host frames get file:line for gdb-oneapi while the
    # device build stays optimized.
    if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    message(STATUS "SYCL build: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

    # # Downgrade -g to -gline-tables-only for C++ per-config flags.
    # # This prevents ocloc from receiving -options "-g" and keeps device code optimized.
    # foreach(_cfg RELWITHDEBINFO DEBUG)
    #     if(CMAKE_CXX_FLAGS_${_cfg})
    #         string(REPLACE "-g" "-gline-tables-only"
    #                CMAKE_CXX_FLAGS_${_cfg} "${CMAKE_CXX_FLAGS_${_cfg}}")
    #     endif()
    # endforeach()

    # # gsycl INTERFACE target - carries the shim include path (matches gpu4pyscf pattern)
    # add_library(sycl_compat INTERFACE)
    # target_include_directories(sycl_compat INTERFACE
    #     "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/gsycl>"
    # )

    # Add gsycl/ shim to include path (BEFORE system paths to shadow CUDA headers)
    include_directories(BEFORE ${CMAKE_SOURCE_DIR}/gsycl)

    # SYCL compile/link flags for AOT PVC (Ahead-Of-Time for Intel PVC)
    # NOTE: no -DMKL_CBLAS_H needed -- the gsycl/{cublas_v2,cusolverDn,cufft}.h
    # shims are declarations-only headers (mirroring real CUDA headers) and
    # never include any oneMKL C++ header, so mkl_cblas.h never appears
    # alongside gsl/gsl_cblas.h in the same translation unit. The actual
    # oneMKL calls live in gsycl/*_impl.cpp, compiled separately.
    # Ahead-of-time device codegen for PVC. Turn off (-DSyclAOT=OFF) to fall back
    # to JIT: much faster to build, which is what you want while iterating.
    option(SyclAOT "Ahead-of-time compile the SYCL device code for Intel PVC" ON)

    # One device image per kernel: smallest images, so the driver only loads and
    # (under JIT) compiles what a launch actually needs.
    #
    # This must be paired with -fno-sycl-allow-device-image-dependencies at link
    # (below). With dependencies allowed, a kernel image only *references* the
    # device functions it calls, leaving them in a separate shared image that
    # the runtime links in at launch via urProgramLinkExp. That works for SPIR-V
    # (JIT) but fails with UR_RESULT_ERROR_INVALID_ARGUMENT under AOT, because
    # native PVC binaries cannot be linked at runtime. Disallowing dependencies
    # makes every kernel image self-contained, which costs some duplication but
    # is what AOT requires. oneMKL's imported targets turn the dependencies back
    # on, hence the explicit negation after them on the link line.
    set(SyclDeviceCodeSplit "per_kernel" CACHE STRING "-fsycl-device-code-split mode")
    set_property(CACHE SyclDeviceCodeSplit PROPERTY STRINGS per_kernel per_source off)

    set(SYCL_COMPILE_FLAGS
        -fsycl
        -fsycl-device-code-split=${SyclDeviceCodeSplit}
        -fp-model=precise
        # -fp-model=precise alone does NOT disable FMA contraction; without
        # this the device pass would contract a-b*c into fma() independently of
        # the host pass. Stated here as well as in the global PreciseFP block
        # so device codegen stays pinned even if CMAKE_CXX_FLAGS is overridden.
        -ffp-contract=off
        -fno-system-debug
        -fsycl-default-sub-group-size=32
    )
    set(SYCL_LINK_FLAGS
        -fsycl
    )
    if(SyclAOT)
        list(APPEND SYCL_COMPILE_FLAGS
            -fsycl-targets=spir64_gen
            "SHELL:-Xsycl-target-backend=spir64_gen \"-device pvc\""
        )
        list(APPEND SYCL_LINK_FLAGS
            -ftarget-register-alloc-mode=pvc:auto
            -fsycl-targets=spir64_gen
            "SHELL:-Xsycl-target-backend=spir64_gen \"-device pvc\""
        )
    else()
        message(STATUS "SYCL build: AOT disabled, device code will be JIT compiled")
    endif()

    # Library with GPU kernels (compiled as C++ via gsycl shim)
    FILE(GLOB jdftxlibSources core/*.cpp fluid/*.cpp electronic/*.cpp perturb/*.cpp commands/*.cpp)
    FILE(GLOB gpukernelsSources core/*.cu fluid/*.cu electronic/*.cu commands/*.cu tests/*.cu)
    # oneMKL-touching implementation TUs for the cublas_v2.h/cusolverDn.h/cufft.h
    # shims (declarations-only headers) -- see gsycl/*_impl.cpp for rationale.
    set(gsyclImplSources
        ${CMAKE_SOURCE_DIR}/gsycl/cublas_impl.cpp
        ${CMAKE_SOURCE_DIR}/gsycl/cusolver_impl.cpp
        ${CMAKE_SOURCE_DIR}/gsycl/cufft_impl.cpp
    )
    add_library(jdftxlib_gpu_sycl ${LINK_TYPE} ${jdftxlibSources} ${gpukernelsSources} ${gsyclImplSources})

    # Compile .cu files as C++ (not CUDA) so gsycl headers shadow <cuda_runtime.h>.
    # gsycl/cuda_prologue.hpp is force-included into the .cu units only: it plays
    # nvcc's role of predefining __device__/__global__/__CUDA_ARCH__ so that
    # core/scalar.h and the __CUDA_ARCH__ splits behave as in the CUDA build.
    # The .cpp units must NOT get it -- they take the host path, as under nvcc.
    set_source_files_properties(${gpukernelsSources} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${gpukernelsSources} PROPERTIES
        COMPILE_FLAGS "-x c++ -include ${CMAKE_SOURCE_DIR}/gsycl/cuda_prologue.hpp")
    target_compile_definitions(jdftxlib_gpu_sycl PRIVATE USE_SYCL)
    target_compile_options(jdftxlib_gpu_sycl PRIVATE ${SYCL_COMPILE_FLAGS})
    target_link_options(jdftxlib_gpu_sycl PRIVATE ${SYCL_LINK_FLAGS})

    # Link the oneMKL SYCL device domains (found above, in CONFIG mode).
    target_link_libraries(jdftxlib_gpu_sycl
        MKL::MKL_SYCL::BLAS
        MKL::MKL_SYCL::LAPACK
        MKL::MKL_SYCL::DFT
    )

    target_link_libraries(jdftxlib_gpu_sycl ${EXTERNAL_LIBS})

    set_target_properties(jdftxlib_gpu_sycl PROPERTIES OUTPUT_NAME "jdftx_gpu_sycl")
    set_JDFTx_flags(jdftxlib_gpu_sycl ON)
endif()

option(PinnedHostMemory "Use page-locked host memory to speed up GPU transfers" OFF)
if(PinnedHostMemory)
	add_definitions("-DPINNED_HOST_MEMORY")
endif()

option(CudaAwareMPI "Whether MPI library supports directly accessing CUDA pointer" OFF)
if(CudaAwareMPI)
	add_definitions("-DCUDA_AWARE_MPI")
endif()

option(CudaManagedMemory "Whether to use CUDA Managed/Unified memory" OFF)
if(CudaManagedMemory)
	add_definitions("-DCUDA_MANAGED_MEMORY")
endif()


#------------- Macro to add an executable pair (regular and GPU) -------------

macro(add_JDFTx_executable execName execSources)
	#CPU version
	add_executable(${execName} ${ARGN} ${execSources})
	target_link_libraries(${execName} ${LINK_PREFIX} jdftxlib ${LINK_SUFFIX})
	if(NOT "${ARGN}" MATCHES EXCLUDE_FROM_ALL)
		install(TARGETS ${execName} RUNTIME DESTINATION bin)
	endif()
	set_JDFTX_flags(${execName} OFF)
	
	if(EnableCUDA) #NVIDIA GPU version
		add_executable(${execName}_gpu ${ARGN} ${execSources})
		target_link_libraries(${execName}_gpu ${LINK_PREFIX} jdftxlib_gpu CUDA::cudart ${LINK_SUFFIX})
		if(NOT "${ARGN}" MATCHES EXCLUDE_FROM_ALL)
			install(TARGETS ${execName}_gpu RUNTIME DESTINATION bin)
		endif()
		set_JDFTx_flags(${execName}_gpu ON)
	endif()

	if(USE_SYCL) #Intel GPU version (same _gpu suffix; the two backends are mutually exclusive)
		add_executable(${execName}_gpu ${ARGN} ${execSources})
		target_compile_definitions(${execName}_gpu PRIVATE USE_SYCL)
		target_compile_options(${execName}_gpu PRIVATE ${SYCL_COMPILE_FLAGS})
		target_link_options(${execName}_gpu PRIVATE ${SYCL_LINK_FLAGS})
		target_link_libraries(${execName}_gpu ${LINK_PREFIX} jdftxlib_gpu_sycl ${LINK_SUFFIX})
		if(NOT "${ARGN}" MATCHES EXCLUDE_FROM_ALL)
			install(TARGETS ${execName}_gpu RUNTIME DESTINATION bin)
		endif()
		set_JDFTx_flags(${execName}_gpu ON)
	endif()
endmacro()

#----------------------- Primary interface executable ------------------------

add_JDFTx_executable(jdftx jdftx.cpp)

#----------------------- Secondary executables ------------------------

#Wannier:
FILE(GLOB wannierSources wannier/*.cpp)
add_JDFTx_executable(wannier "${wannierSources}")

#Phonon:
FILE(GLOB phononSources phonon/*.cpp)
add_JDFTx_executable(phonon "${phononSources}")

#-----------------------------------------------------------------------------

#Documentation via Doxygen:
find_package(Doxygen)
if(DOXYGEN_FOUND)
	set(DOC_TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/doc.tmp")
	file(MAKE_DIRECTORY ${DOC_TMP_DIR})
	file(MAKE_DIRECTORY ${DOC_TMP_DIR}/commands)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)
	#Target for running doxygen:
	add_custom_target(doc
		COMMAND echo "--- Autogenerating command documentation"
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/jdftx   -w jdftx   > "${DOC_TMP_DIR}/commands/jdftx.dox"
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/wannier -w wannier > "${DOC_TMP_DIR}/commands/wannier.dox"
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/phonon  -w phonon  > "${DOC_TMP_DIR}/commands/phonon.dox"
		COMMAND echo "--- Updating command index"
		COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/doc/commands/category_update.sh "${DOC_TMP_DIR}" > "${DOC_TMP_DIR}/commands_categorized.dox"
		COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/doc/commands/index_update.sh "${DOC_TMP_DIR}" > "${DOC_TMP_DIR}/commands_index.dox"
		COMMAND echo "--- Updating scripts documentation"
		COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/doc/scripts/index_update.sh "${CMAKE_CURRENT_SOURCE_DIR}" > "${DOC_TMP_DIR}/scripts_manual.dox"
		COMMAND echo "--- Running doxygen"
		COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/doc/html
		COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
		COMMENT "Generating API documentation with Doxygen" VERBATIM
		DEPENDS jdftx wannier phonon
	)
endif()

#Tests
enable_testing()
add_subdirectory(test)

#Optional: additional auxiliary test executables, mostly related to fluid development
add_subdirectory(aux)

#Pseudopotential library
add_subdirectory(pseudopotentials)


#----------------------- Installation --------------------------
#Executables and library install rules already added above
#--- headers
install(DIRECTORY
	commands
	core
	electronic
	fluid
	phonon
	wannier
	DESTINATION include/jdftx
	FILES_MATCHING PATTERN "*.h")
install(FILES "${CMAKE_BINARY_DIR}/config.h" DESTINATION include/jdftx)
#--- pseudopotentials
install(DIRECTORY
	"${CMAKE_BINARY_DIR}/pseudopotentials/GBRV"
	"${CMAKE_BINARY_DIR}/pseudopotentials/SG15"
	DESTINATION share/jdftx/pseudopotentials)
#--- scripts
install(DIRECTORY scripts DESTINATION share/jdftx USE_SOURCE_PERMISSIONS)
