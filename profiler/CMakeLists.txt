if(BUILD_PROFILER AND ENABLE_GPU_BUILD)
   add_executable(gpu_kernel_profiler main.c profile_kernels.c)
   target_include_directories(gpu_kernel_profiler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels)
   target_sources(gpu_kernel_profiler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/coulomb.c
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/coulomb.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/coulomb_singularity_subtraction.c
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/coulomb_singularity_subtraction.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/yukawa.c
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/yukawa.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/yukawa_singularity_subtraction.c
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/yukawa_singularity_subtraction.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src/tools.c)
   target_compile_definitions(gpu_kernel_profiler PRIVATE OPENACC_ENABLED)   
   target_link_libraries(gpu_kernel_profiler PRIVATE MPI::MPI_C)
   install(TARGETS gpu_kernel_profiler DESTINATION bin)

   add_executable(gpu_kernel_profiler_cuda main_cuda.c profile_kernels_cuda.cu)
   set_target_properties(gpu_kernel_profiler_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
   target_include_directories(gpu_kernel_profiler_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels)
   target_sources(gpu_kernel_profiler_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/tools.c)
   target_link_libraries(gpu_kernel_profiler_cuda PRIVATE MPI::MPI_C)
   install(TARGETS gpu_kernel_profiler_cuda DESTINATION bin)
endif()
