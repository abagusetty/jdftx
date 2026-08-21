add_custom_target(testresults COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/printResults.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} )
add_custom_target(testclean COMMAND rm -f */*.out */*.wfns */*.fillings */*.ionpos */*.eigenvals */*.fluidState */results */summary WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )

#--- Test execution environment ------------------------------------------
# Every test is one JDFTx process on one GPU tile: that is also how production
# runs are launched here (one system per tile, no MPI -- see
# gsycl/run_per_tile.sh), so the tests exercise the same configuration.
#
# TestThreads defaults to 1 deliberately. JDFTx sizes its own std::thread pool
# from OMP_NUM_THREADS (core/Util.cpp), and this node exports OMP_NUM_THREADS
# equal to its full 208 logical CPUs; inherited, that makes every operator
# spawn 208 threads and the CPU binary aborts in thread creation
# (std::__throw_system_error out of eblas_zgemm). Single-threaded is also
# measurably faster for these small test cells on the GPU binary, where the
# device does the work. For a CPU-only sweep (plain `ctest`, no JDFTX_SUFFIX)
# raise it -- -DTestThreads=16 is a reasonable value on this node; anything
# near the full 208 hits the thread-creation failure above.
set(TestTile "0.0" CACHE STRING "GPU tile the tests run on (ZE_AFFINITY_MASK, <device>.<subdevice>)")
set(TestThreads "1" CACHE STRING "CPU threads per test process")
set(TestTimeout "1800" CACHE STRING "Per-test timeout in seconds")

macro(add_jdftx_test testName)
	add_test(NAME ${testName} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/runTest.sh ${testName} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR})
	set(testEnv "OMP_NUM_THREADS=${TestThreads}" "MKL_NUM_THREADS=${TestThreads}")
	if(USE_SYCL)
		# Pin to a single tile. ZE_AFFINITY_MASK filters what Level Zero
		# enumerates, so the process sees exactly one device and gpuInit()
		# cannot round-robin onto a neighbour's tile.
		list(APPEND testEnv "ZE_AFFINITY_MASK=${TestTile}" "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE")
	endif()
	# RUN_SERIAL: even under `ctest -j`, keep one test at a time on that tile.
	set_tests_properties(${testName} PROPERTIES
		ENVIRONMENT "${testEnv}"
		TIMEOUT ${TestTimeout}
		RUN_SERIAL TRUE)
endmacro()

add_jdftx_test(openShell)
add_jdftx_test(vibrations)
add_jdftx_test(moleculeSolvation)
add_jdftx_test(ionSolvation)
add_jdftx_test(latticeOpt)
add_jdftx_test(metalBulk)
add_jdftx_test(plusU)
add_jdftx_test(spinOrbit)
add_jdftx_test(graphene)
add_jdftx_test(metalSurface)
add_jdftx_test(checkpoint)
