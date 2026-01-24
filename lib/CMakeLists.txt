# CFD Library CMake Configuration
# Modular Backend Architecture - Split into separate libraries per backend
cmake_minimum_required(VERSION 3.10)

# Option for CUDA support
option(CFD_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# CUDA architecture configuration
# Default: Support GPUs from Turing (RTX 20 series) onwards
# Users can override with -DCFD_CUDA_ARCHITECTURES="75;80" etc.
if(NOT DEFINED CFD_CUDA_ARCHITECTURES)
    set(CFD_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
        "CUDA architectures to build for (75=Turing, 80/86=Ampere, 89=Ada, 90=Hopper)")
endif()

# Set CMAKE_CUDA_ARCHITECTURES before project() to satisfy CMP0104
if(CFD_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CFD_CUDA_ARCHITECTURES}")
endif()

# Define the library project with optional CUDA
if(CFD_ENABLE_CUDA)
    project(cfd_library VERSION 0.1.6 LANGUAGES C CUDA)
    message(STATUS "CUDA support enabled")
else()
    project(cfd_library VERSION 0.1.6 LANGUAGES C)
    message(STATUS "CUDA support disabled (use -DCFD_ENABLE_CUDA=ON to enable)")
endif()

# OpenMP Support
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found. Multithreading support enabled.")
    add_compile_definitions(CFD_ENABLE_OPENMP)
else()
    message(WARNING "OpenMP not found. Multithreaded solvers will run in serial.")
endif()

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Option to enable AVX2 SIMD optimizations (requires AVX2-capable CPU)
option(CFD_ENABLE_AVX2 "Enable AVX2 SIMD optimizations (requires AVX2-capable CPU)" OFF)

# Detect x86/x64 architecture (case-insensitive)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CFD_PROCESSOR_LOWER)
if(CFD_PROCESSOR_LOWER MATCHES "x86_64|amd64|i686|x86")
    set(CFD_IS_X86 TRUE CACHE INTERNAL "Is x86/x64 architecture")
else()
    set(CFD_IS_X86 FALSE CACHE INTERNAL "Is x86/x64 architecture")
endif()

# Detect ARM64 architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(CFD_IS_ARM64 TRUE CACHE INTERNAL "Is ARM64 architecture")
else()
    set(CFD_IS_ARM64 FALSE CACHE INTERNAL "Is ARM64 architecture")
endif()

# Report SIMD optimization status
if(CFD_ENABLE_AVX2 AND CFD_IS_X86)
    message(STATUS "AVX2 SIMD optimizations ENABLED (${CMAKE_SYSTEM_PROCESSOR})")
    add_compile_definitions(CFD_HAS_AVX2)
elseif(CFD_IS_ARM64)
    message(STATUS "ARM NEON SIMD optimizations available (${CMAKE_SYSTEM_PROCESSOR})")
else()
    message(STATUS "SIMD optimizations DISABLED - using scalar fallback")
endif()

#=============================================================================
# Source File Definitions - Organized by Backend
#=============================================================================

# Core sources (always required)
# Note: API sources are NOT part of core because they depend on solver backends.
# They are in a separate cfd_api library that links all backends.
set(CFD_CORE_SOURCES
    src/core/grid.c
    src/core/memory.c
    src/core/filesystem.c
    src/core/logging.c
    src/core/cfd_version.c
    src/core/cfd_init.c
    src/core/math_utils.c
    src/core/field_ops.c
    src/core/derived_fields.c
    src/core/cpu_features.c
)

# API sources - these depend on solver backends and must link against all of them
# Includes dispatcher files that reference multiple backends
set(CFD_API_SOURCES
    src/api/simulation_api.c
    src/api/solver_registry.c
    src/api/output_registry.c
    # Dispatcher files that select between backends at runtime
    src/boundary/boundary_conditions.c
    src/solvers/linear/linear_solver.c
)

# I/O sources (part of core)
set(CFD_IO_SOURCES
    src/io/csv_output.c
    src/io/vtk_output.c
)

# Scalar CPU sources (baseline implementation)
# Note: Dispatcher files (boundary_conditions.c, linear_solver.c) are in CFD_API_SOURCES
# because they reference all backends and must link against them.
set(CFD_SCALAR_SOURCES
    # Boundary conditions - scalar implementations only
    src/boundary/cpu/boundary_conditions_scalar.c
    src/boundary/cpu/boundary_conditions_inlet_scalar.c
    src/boundary/cpu/boundary_conditions_inlet_time_scalar.c
    src/boundary/cpu/boundary_conditions_outlet_scalar.c
    # Navier-Stokes solvers - scalar
    src/solvers/navier_stokes/cpu/solver_explicit_euler.c
    src/solvers/navier_stokes/cpu/solver_projection.c
    # Linear solvers - scalar implementations only
    src/solvers/linear/cpu/linear_solver_jacobi.c
    src/solvers/linear/cpu/linear_solver_sor.c
    src/solvers/linear/cpu/linear_solver_redblack.c
    src/solvers/linear/cpu/linear_solver_cg.c
)

# SIMD sources (AVX2/NEON optimized)
set(CFD_SIMD_SOURCES
    # Boundary conditions - SIMD
    src/boundary/simd/boundary_conditions_simd_dispatch.c
    src/boundary/avx2/boundary_conditions_avx2.c
    src/boundary/avx2/boundary_conditions_inlet_avx2.c
    src/boundary/avx2/boundary_conditions_outlet_avx2.c
    src/boundary/neon/boundary_conditions_neon.c
    src/boundary/neon/boundary_conditions_inlet_neon.c
    src/boundary/neon/boundary_conditions_outlet_neon.c
    # Navier-Stokes solvers - SIMD
    src/solvers/navier_stokes/avx2/solver_explicit_euler_avx2.c
    src/solvers/navier_stokes/avx2/solver_projection_avx2.c
    # Linear solvers - SIMD
    src/solvers/linear/simd/linear_solver_simd_dispatch.c
    src/solvers/linear/avx2/linear_solver_jacobi_avx2.c
    src/solvers/linear/avx2/linear_solver_redblack_avx2.c
    src/solvers/linear/avx2/linear_solver_cg_avx2.c
    src/solvers/linear/neon/linear_solver_jacobi_neon.c
    src/solvers/linear/neon/linear_solver_redblack_neon.c
    src/solvers/linear/neon/linear_solver_cg_neon.c
)

# OpenMP sources
set(CFD_OMP_SOURCES
    # Boundary conditions - OMP
    src/boundary/omp/boundary_conditions_omp.c
    src/boundary/omp/boundary_conditions_inlet_omp.c
    src/boundary/omp/boundary_conditions_outlet_omp.c
    # Navier-Stokes solvers - OMP
    src/solvers/navier_stokes/omp/solver_explicit_euler_omp.c
    src/solvers/navier_stokes/omp/solver_projection_omp.c
    # Linear solvers - OMP
    src/solvers/linear/omp/linear_solver_redblack_omp.c
)

# Note: API sources (solver_registry, simulation_api, output_registry) are in
# a separate cfd_api library that links against all solver backends.

# GPU stub (when CUDA not available)
set(CFD_GPU_STUB_SOURCES
    src/solvers/gpu/solver_gpu_stub.c
)

# GPU CUDA sources
set(CFD_CUDA_SOURCES
    src/boundary/gpu/boundary_conditions_gpu.cu
    src/boundary/gpu/boundary_conditions_inlet_gpu.cu
    src/solvers/gpu/solver_projection_jacobi_gpu.cu
)

#=============================================================================
# Function to apply compiler options
#=============================================================================
function(apply_compiler_options target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
        if(CFD_ENABLE_AVX2 AND CFD_IS_X86)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -O3>)
        if(CFD_ENABLE_AVX2 AND CFD_IS_X86)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-mavx2 -mfma>)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-mcpu=native>)
        endif()
        target_link_libraries(${target_name} PUBLIC m)
    endif()
endfunction()

function(apply_omp_options target_name)
    if(OpenMP_C_FOUND)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_C)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/openmp:llvm>)
        endif()
    endif()
endfunction()

function(apply_cuda_options target_name)
    if(CFD_HAS_CUDA)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
        )
    endif()
endfunction()

#=============================================================================
# Library Targets - Modular Backend Architecture
#=============================================================================

# --- CFD Core Library ---
# Contains: grid, memory, I/O, status, utilities, API layer
# Dependencies: None (base library)
# Note: GPU stub is always included in cfd_core to provide gpu_is_available().
#       The unified library (cfd_library) handles CUDA vs stub selection separately.
# Note: OpenMP is optional for core - used for parallelizing derived_fields.c
add_library(cfd_core STATIC
    ${CFD_CORE_SOURCES}
    ${CFD_IO_SOURCES}
    ${CFD_GPU_STUB_SOURCES}
)
set_target_properties(cfd_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cfd_core
)
target_include_directories(cfd_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
apply_compiler_options(cfd_core)
# Core uses OpenMP for derived_fields.c parallelization (optional)
apply_omp_options(cfd_core)
add_library(CFD::Core ALIAS cfd_core)

# --- CFD Scalar Library ---
# Contains: Scalar CPU solvers (baseline)
# Dependencies: cfd_core
add_library(cfd_scalar STATIC
    ${CFD_SCALAR_SOURCES}
)
set_target_properties(cfd_scalar PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cfd_scalar
)
target_include_directories(cfd_scalar
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cfd_scalar PUBLIC cfd_core)
# Note: cfd_scalar calls poisson_solve() which is defined in cfd_api (a dispatcher).
# We can't make cfd_scalar depend on cfd_api at build time (circular dependency),
# but we need the symbols at link time. CMake handles this automatically when
# linking the final executable/library, so no special handling needed here.
apply_compiler_options(cfd_scalar)
add_library(CFD::Scalar ALIAS cfd_scalar)

# --- CFD SIMD Library ---
# Contains: AVX2/NEON optimized solvers
# Dependencies: cfd_core, cfd_scalar (for fallback)
add_library(cfd_simd STATIC
    ${CFD_SIMD_SOURCES}
)
set_target_properties(cfd_simd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cfd_simd
)
target_include_directories(cfd_simd
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cfd_simd PUBLIC cfd_core cfd_scalar)
apply_compiler_options(cfd_simd)
add_library(CFD::SIMD ALIAS cfd_simd)

# --- CFD OpenMP Library ---
# Contains: OpenMP parallelized solvers (or stubs when OpenMP not available)
# Dependencies: cfd_core, cfd_scalar
# Note: Always build this library to provide bc_impl_omp symbol needed by cfd_api dispatcher.
#       When OpenMP is not available, only boundary condition stubs are compiled (they have guards).
if(OpenMP_C_FOUND)
    # Full OpenMP implementation
    set(CFD_OMP_BUILD_SOURCES ${CFD_OMP_SOURCES})
    set(CFD_HAS_OMP TRUE)
else()
    # Only boundary condition files (they have #ifdef CFD_ENABLE_OPENMP guards)
    # Solver files require omp.h and can't be compiled without OpenMP
    set(CFD_OMP_BUILD_SOURCES
        src/boundary/omp/boundary_conditions_omp.c
        src/boundary/omp/boundary_conditions_inlet_omp.c
        src/boundary/omp/boundary_conditions_outlet_omp.c
    )
    set(CFD_HAS_OMP FALSE)
endif()

add_library(cfd_omp STATIC
    ${CFD_OMP_BUILD_SOURCES}
)
set_target_properties(cfd_omp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cfd_omp
)
target_include_directories(cfd_omp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cfd_omp PUBLIC cfd_core cfd_scalar)
apply_compiler_options(cfd_omp)
if(OpenMP_C_FOUND)
    apply_omp_options(cfd_omp)
endif()
add_library(CFD::OMP ALIAS cfd_omp)

# --- CFD CUDA Library ---
# Contains: CUDA GPU solvers
# Dependencies: cfd_core, cfd_scalar
if(CFD_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        find_package(CUDAToolkit REQUIRED)

        add_library(cfd_cuda STATIC
            ${CFD_CUDA_SOURCES}
        )
        set_target_properties(cfd_cuda PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME cfd_cuda
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES "${CFD_CUDA_ARCHITECTURES}"
        )
        target_include_directories(cfd_cuda
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_compile_definitions(cfd_cuda PRIVATE CFD_HAS_CUDA=1)
        target_link_libraries(cfd_cuda PUBLIC cfd_core cfd_scalar CUDA::cudart)
        apply_compiler_options(cfd_cuda)
        apply_cuda_options(cfd_cuda)
        add_library(CFD::CUDA ALIAS cfd_cuda)
        set(CFD_HAS_CUDA TRUE)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "CUDA requested but no CUDA compiler found, using CPU stub")
        set(CFD_HAS_CUDA FALSE)
    endif()
else()
    set(CFD_HAS_CUDA FALSE)
endif()

# --- CFD API Library ---
# Contains: solver_registry, simulation_api, output_registry, dispatchers
# Dependencies: cfd_core, cfd_scalar, cfd_simd, cfd_omp (always), cfd_cuda (optional)
# This library provides the high-level API and must link all solver backends.
# Note: cfd_omp is always linked because the boundary_conditions dispatcher references
#       bc_impl_omp. When OpenMP is disabled, cfd_omp contains only NULL stub tables.
set(CFD_API_DEPS cfd_core cfd_scalar cfd_simd cfd_omp)
if(CFD_HAS_CUDA)
    list(APPEND CFD_API_DEPS cfd_cuda)
endif()

add_library(cfd_api STATIC
    ${CFD_API_SOURCES}
)
set_target_properties(cfd_api PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cfd_api
)
target_include_directories(cfd_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cfd_api PUBLIC ${CFD_API_DEPS})
# Handle circular dependency: cfd_scalar/cfd_simd call poisson_solve() from cfd_api
# On Linux/GCC, we need to use linker groups. Add custom link flags to wrap all the
# modular libraries in --start-group/--end-group.
# This is handled at the cfd_library (unified) level, not here.
# The circular reference: cfd_scalar/simd -> poisson_solve (in cfd_api) -> cfd_scalar/simd
apply_compiler_options(cfd_api)
if(CFD_HAS_OMP)
    apply_omp_options(cfd_api)
endif()
if(CFD_HAS_CUDA)
    target_compile_definitions(cfd_api PRIVATE CFD_HAS_CUDA=1)
endif()
add_library(CFD::API ALIAS cfd_api)

#=============================================================================
# Unified Library (cfd_library) - Combines All Backends
# This maintains backward compatibility with existing code
#
# The unified library links against the modular libraries rather than
# recompiling sources. This ensures consistency and reduces build time.
#=============================================================================

# Collect modular libraries to link
# cfd_api already links all the backend libraries, so we just need cfd_api
set(CFD_MODULAR_LIBS cfd_api)

# For static builds, create a unified library by linking modular libraries
# For shared builds, we need to recompile to properly export symbols
if(BUILD_SHARED_LIBS)
    # Shared library needs source recompilation for proper symbol export
    set(CFD_UNIFIED_SOURCES
        ${CFD_CORE_SOURCES}
        ${CFD_IO_SOURCES}
        ${CFD_API_SOURCES}
        ${CFD_SCALAR_SOURCES}
        ${CFD_SIMD_SOURCES}
    )
    if(OpenMP_C_FOUND)
        list(APPEND CFD_UNIFIED_SOURCES ${CFD_OMP_SOURCES})
    endif()
    if(CFD_HAS_CUDA)
        list(APPEND CFD_UNIFIED_SOURCES ${CFD_CUDA_SOURCES})
    else()
        list(APPEND CFD_UNIFIED_SOURCES ${CFD_GPU_STUB_SOURCES})
    endif()

    add_library(cfd_library SHARED ${CFD_UNIFIED_SOURCES})

    # Set properties for shared library
    set_target_properties(cfd_library PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    target_include_directories(cfd_library
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    apply_compiler_options(cfd_library)
    apply_omp_options(cfd_library)

    # CUDA settings for shared library
    if(CFD_HAS_CUDA)
        set_target_properties(cfd_library PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES "${CFD_CUDA_ARCHITECTURES}"
        )
        target_compile_definitions(cfd_library PRIVATE CFD_HAS_CUDA=1)
        target_link_libraries(cfd_library PUBLIC CUDA::cudart)
        apply_cuda_options(cfd_library)
    endif()

    # Also create a static INTERFACE library to aggregate modular libraries
    add_library(cfd_library_static INTERFACE)
    target_link_libraries(cfd_library_static INTERFACE ${CFD_MODULAR_LIBS})
    target_include_directories(cfd_library_static INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
else()
    # Static build: use INTERFACE library to aggregate modular libraries
    # This ensures consistency between modular and unified builds and reduces build time
    add_library(cfd_library INTERFACE)

    # On Linux, wrap modular libraries in linker group to resolve circular dependencies
    # (cfd_scalar/cfd_simd call poisson_solve from cfd_api)
    if(UNIX AND NOT APPLE)
        # Explicitly list all backend libraries for the linker group
        # Note: cfd_omp is always included (contains stubs when OpenMP disabled)
        set(LINKER_GROUP_LIBS cfd_api cfd_core cfd_scalar cfd_simd cfd_omp)
        if(CFD_HAS_CUDA)
            list(APPEND LINKER_GROUP_LIBS cfd_cuda)
        endif()

        target_link_libraries(cfd_library INTERFACE
            "-Wl,--start-group"
            ${LINKER_GROUP_LIBS}
            "-Wl,--end-group"
        )
    else()
        target_link_libraries(cfd_library INTERFACE ${CFD_MODULAR_LIBS})
    endif()

    target_include_directories(cfd_library INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# Symbol Visibility and Export Header (needed for both builds)
include(GenerateExportHeader)
# For INTERFACE libraries, we can't generate export header from it, use cfd_core instead
if(BUILD_SHARED_LIBS)
    generate_export_header(cfd_library
        BASE_NAME CFD_LIBRARY
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h"
    )
    target_include_directories(cfd_library PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )
else()
    # For static builds, generate export header from cfd_core
    generate_export_header(cfd_core
        BASE_NAME CFD_LIBRARY
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h"
        STATIC_DEFINE CFD_LIBRARY_STATIC_DEFINE
    )
endif()

# Create alias for backward compatibility
add_library(CFD::Library ALIAS cfd_library)

#=============================================================================
# Installation
#=============================================================================
include(GNUInstallDirs)

# Collect headers
set(CFD_LIBRARY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/grid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/cfd_status.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/filesystem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/logging.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/cfd_init.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/cfd_version.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/math_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/field_ops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/derived_fields.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/gpu_device.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/core/cpu_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/solvers/navier_stokes_solver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/solvers/poisson_solver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/boundary/boundary_conditions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/io/vtk_output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/io/csv_output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/io/output_registry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/api/simulation_api.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h
)

# Targets to install
# Note: cfd_omp is always included (contains stubs when OpenMP disabled)
set(INSTALL_TARGETS cfd_library cfd_core cfd_scalar cfd_simd cfd_omp cfd_api)
if(CFD_HAS_CUDA)
    list(APPEND INSTALL_TARGETS cfd_cuda)
endif()
if(BUILD_SHARED_LIBS)
    list(APPEND INSTALL_TARGETS cfd_library_static)
endif()

install(TARGETS ${INSTALL_TARGETS}
    EXPORT cfd_libraryTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/cfd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cfd
)

install(EXPORT cfd_libraryTargets
    FILE cfd_libraryTargets.cmake
    NAMESPACE CFD::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cfd_library
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cfd_libraryConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cfd_libraryConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cfd_library
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cfd_libraryConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cfd_libraryConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cfd_libraryConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cfd_library
)

#=============================================================================
# Summary
#=============================================================================
message(STATUS "")
message(STATUS "CFD Library Configuration:")
message(STATUS "  Libraries built:")
message(STATUS "    - cfd_core     (grid, memory, I/O, utilities)")
message(STATUS "    - cfd_scalar   (scalar CPU solvers)")
message(STATUS "    - cfd_simd     (AVX2/NEON optimized solvers)")
if(CFD_HAS_OMP)
    message(STATUS "    - cfd_omp      (OpenMP parallelized solvers)")
else()
    message(STATUS "    - cfd_omp      (stub - OpenMP not available)")
endif()
if(CFD_HAS_CUDA)
    message(STATUS "    - cfd_cuda     (CUDA GPU solvers)")
    message(STATUS "      CUDA Architectures: ${CFD_CUDA_ARCHITECTURES}")
endif()
message(STATUS "    - cfd_api      (solver registry, simulation API)")
message(STATUS "    - cfd_library  (unified library - all backends)")
message(STATUS "")
message(STATUS "  Usage:")
message(STATUS "    target_link_libraries(app PRIVATE CFD::Library)  # All backends (recommended)")
message(STATUS "    target_link_libraries(app PRIVATE CFD::API)      # API + all backends")
message(STATUS "")
