# CFD Library CMake Configuration
cmake_minimum_required(VERSION 3.10)

# Option for CUDA support
option(CFD_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# CUDA architecture configuration
# Default: Support wide range of GPUs from Pascal (GTX 10xx) onwards
# Users can override with -DCFD_CUDA_ARCHITECTURES="70;75" etc.
if(NOT DEFINED CFD_CUDA_ARCHITECTURES)
    set(CFD_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
        "CUDA architectures to build for (75=Turing, 80=Ampere, 86=Ampere, 89=Ada, 90=Hopper)")
endif()

# Set CMAKE_CUDA_ARCHITECTURES before project() to satisfy CMP0104
if(CFD_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CFD_CUDA_ARCHITECTURES}")
endif()

# Define the library project with optional CUDA
if(CFD_ENABLE_CUDA)
    project(cfd_library VERSION 0.1.0 LANGUAGES C CUDA)
    message(STATUS "CUDA support enabled")
else()
    project(cfd_library VERSION 0.1.0 LANGUAGES C)
    message(STATUS "CUDA support disabled (use -DCFD_ENABLE_CUDA=ON to enable)")
endif()

# OpenMP Support
set(CFD_OMP_SOLVER_SOURCES "")
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found. Multithreading support enabled.")
    add_compile_definitions(CFD_ENABLE_OPENMP)
    set(CFD_OMP_SOLVER_SOURCES
        src/solvers/navier_stokes/omp/solver_explicit_euler_omp.c
        src/solvers/navier_stokes/omp/solver_projection_omp.c
    )
else()
    message(WARNING "OpenMP not found. Multithreaded solvers will run in serial.")
endif()

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Collect core source files
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

# Collect boundary condition source files (organized by backend)
# AVX2 and NEON implementations are in separate files; the dispatcher
# selects the correct one at runtime based on CPU feature detection.
set(CFD_BOUNDARY_SOURCES
    src/boundary/boundary_conditions.c
    src/boundary/cpu/boundary_conditions_scalar.c
    src/boundary/cpu/boundary_conditions_inlet_scalar.c
    src/boundary/omp/boundary_conditions_omp.c
    src/boundary/omp/boundary_conditions_inlet_omp.c
    src/boundary/avx2/boundary_conditions_avx2_omp.c
    src/boundary/avx2/boundary_conditions_inlet_avx2_omp.c
    src/boundary/neon/boundary_conditions_neon_omp.c
    src/boundary/neon/boundary_conditions_inlet_neon_omp.c
    src/boundary/simd_omp/boundary_conditions_simd_omp_dispatch.c
)

# Collect I/O source files
set(CFD_IO_SOURCES
    src/io/csv_output.c
    src/io/vtk_output.c
)

# Collect API source files
set(CFD_API_SOURCES
    src/api/simulation_api.c
    src/api/solver_registry.c
    src/api/output_registry.c
)

# Collect Navier-Stokes solver source files (organized by backend)
set(CFD_NAVIER_STOKES_SOURCES
    # Scalar CPU implementations
    src/solvers/navier_stokes/cpu/solver_explicit_euler.c
    src/solvers/navier_stokes/cpu/solver_projection.c
    # AVX2+OMP implementations (x86-64)
    src/solvers/navier_stokes/avx2_omp/solver_explicit_euler_avx2_omp.c
    src/solvers/navier_stokes/avx2_omp/solver_projection_avx2_omp.c
)

# Collect Poisson solver sources (abstraction layer)
# Uses runtime SIMD detection via SIMD+OMP dispatcher (AVX2 on x86-64, NEON on ARM64)
set(CFD_POISSON_SOLVER_SOURCES
    src/solvers/linear/linear_solver.c
    src/solvers/linear/cpu/linear_solver_jacobi.c
    src/solvers/linear/cpu/linear_solver_sor.c
    src/solvers/linear/cpu/linear_solver_redblack.c
    # SIMD+OMP dispatcher (runtime architecture detection)
    src/solvers/linear/simd_omp/linear_solver_simd_omp_dispatch.c
    # AVX2+OMP implementations (x86-64)
    src/solvers/linear/avx2/linear_solver_jacobi_avx2_omp.c
    src/solvers/linear/avx2/linear_solver_redblack_avx2_omp.c
    # NEON+OMP implementations (ARM64)
    src/solvers/linear/neon/linear_solver_jacobi_neon_omp.c
    src/solvers/linear/neon/linear_solver_redblack_neon_omp.c
)

# OMP Poisson solver (if OpenMP is available)
if(OpenMP_C_FOUND)
    list(APPEND CFD_POISSON_SOLVER_SOURCES
        src/solvers/linear/omp/linear_solver_redblack_omp.c
    )
endif()

# Combine all source files
set(CFD_LIBRARY_SOURCES
    ${CFD_CORE_SOURCES}
    ${CFD_BOUNDARY_SOURCES}
    ${CFD_IO_SOURCES}
    ${CFD_API_SOURCES}
    ${CFD_NAVIER_STOKES_SOURCES}
    ${CFD_OMP_SOLVER_SOURCES}
    ${CFD_POISSON_SOLVER_SOURCES}
)

# Add GPU sources based on CUDA availability
if(CFD_ENABLE_CUDA)
    # Check if CUDA toolkit is actually available
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        list(APPEND CFD_LIBRARY_SOURCES
            src/boundary/gpu/boundary_conditions_gpu.cu
            src/boundary/gpu/boundary_conditions_inlet_gpu.cu
            src/solvers/gpu/solver_projection_jacobi_gpu.cu
        )
        set(CFD_HAS_CUDA TRUE)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "CUDA requested but no CUDA compiler found, using CPU stub")
        list(APPEND CFD_LIBRARY_SOURCES src/solvers/gpu/solver_gpu_stub.c)
        set(CFD_HAS_CUDA FALSE)
    endif()
else()
    # Use CPU stub when CUDA is disabled
    list(APPEND CFD_LIBRARY_SOURCES src/solvers/gpu/solver_gpu_stub.c)
    set(CFD_HAS_CUDA FALSE)
endif()

# Collect all header files
set(CFD_LIBRARY_HEADERS
    # Core headers
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
    # Solver headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/solvers/navier_stokes_solver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/solvers/poisson_solver.h
    # Boundary headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/boundary/boundary_conditions.h
    # IO headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/io/vtk_output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/io/csv_output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/io/output_registry.h
    # API headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cfd/api/simulation_api.h
)

# Create libraries (both static and shared if requested)
if(BUILD_SHARED_LIBS)
    # Create shared library
    add_library(cfd_library SHARED ${CFD_LIBRARY_SOURCES})
    # Also create static library with different name
    add_library(cfd_library_static STATIC ${CFD_LIBRARY_SOURCES})
    set_target_properties(cfd_library_static PROPERTIES OUTPUT_NAME cfd_library)
else()
    # Create static library only
    add_library(cfd_library STATIC ${CFD_LIBRARY_SOURCES})
    target_compile_definitions(cfd_library PUBLIC CFD_LIBRARY_STATIC_DEFINE)
endif()

# Enable Position Independent Code for all library targets
# This is required when building Python extensions that link the static library
# Being explicit ensures consistent behavior across all platforms
set_target_properties(cfd_library PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(BUILD_SHARED_LIBS)
    set_target_properties(cfd_library_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Set CUDA properties if enabled
if(CFD_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)

    set_target_properties(cfd_library PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CFD_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(cfd_library PRIVATE CFD_HAS_CUDA=1)
    target_link_libraries(cfd_library PUBLIC CUDA::cudart)

    if(BUILD_SHARED_LIBS)
        set_target_properties(cfd_library_static PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES "${CFD_CUDA_ARCHITECTURES}"
        )
        target_compile_definitions(cfd_library_static PRIVATE CFD_HAS_CUDA=1)
        target_link_libraries(cfd_library_static PUBLIC CUDA::cudart)
    endif()
endif()

# Set library properties
set_target_properties(cfd_library PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Include directories for main library
target_include_directories(cfd_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Symbol Visibility and Export Header
include(GenerateExportHeader)
generate_export_header(cfd_library
    BASE_NAME CFD_LIBRARY
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h"
)

# Set visibility properties
set_target_properties(cfd_library PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Add export header to includes
target_include_directories(cfd_library PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add generated header to headers list for installation
list(APPEND CFD_LIBRARY_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h")

# If we built both libraries, set properties for static version too
if(BUILD_SHARED_LIBS)
    set_target_properties(cfd_library_static PROPERTIES
        VERSION ${PROJECT_VERSION}
    )

    target_include_directories(cfd_library_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    # Define STATIC macro to disable export declarations
    target_compile_definitions(cfd_library_static PUBLIC CFD_LIBRARY_STATIC_DEFINE)
endif()

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
    # Define CFD_HAS_AVX2 for source code - MSVC doesn't define __AVX2__ automatically
    # even when compiling with /arch:AVX2, so we need this explicit define
    add_compile_definitions(CFD_HAS_AVX2)
elseif(CFD_IS_ARM64)
    message(STATUS "ARM NEON SIMD optimizations available (${CMAKE_SYSTEM_PROCESSOR})")
else()
    message(STATUS "SIMD optimizations DISABLED - using scalar fallback")
endif()

# Function to apply compiler options to a target
function(apply_compiler_options target_name)
    if(MSVC)
        # Only apply MSVC flags to C/CXX, not CUDA
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
        # Enable AVX2 only if explicitly requested
        if(CFD_ENABLE_AVX2 AND CFD_IS_X86)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -O3>)
        # Enable AVX2 only if explicitly requested
        if(CFD_ENABLE_AVX2 AND CFD_IS_X86)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-mavx2 -mfma>)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            # ARM64-specific optimizations
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-mcpu=native>)
        endif()
        # Link math library on Unix systems
        target_link_libraries(${target_name} PUBLIC m)
    endif()

    # OpenMP linkage
    if(OpenMP_C_FOUND)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_C)
        # Attempt to enable modern LLVM-based OpenMP for MSVC to support newer standards (OpenMP 3.1+)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/openmp:llvm>)
        endif()
    endif()
    # CUDA-specific options
    if(CFD_HAS_CUDA)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
        )
    endif()
endfunction()

# Apply compiler options to libraries
apply_compiler_options(cfd_library)
if(BUILD_SHARED_LIBS)
    apply_compiler_options(cfd_library_static)
endif()

# Optional: Create an alias for easier usage
add_library(CFD::Library ALIAS cfd_library)

# Installation rules (optional, for library distribution)
include(GNUInstallDirs)

# Installation rules
set(INSTALL_TARGETS cfd_library)
if(BUILD_SHARED_LIBS)
    list(APPEND INSTALL_TARGETS cfd_library_static)
endif()

install(TARGETS ${INSTALL_TARGETS}
    EXPORT cfd_libraryTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cfd  <-- Removed to prevent flattening
)

# Install headers preserving directory structure
install(DIRECTORY include/cfd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Install generated export header explicitly
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/cfd/cfd_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cfd
)

install(EXPORT cfd_libraryTargets
    FILE cfd_libraryTargets.cmake
    NAMESPACE CFD::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cfd_library
)

# Create a config file for find_package support
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

# Print summary
message(STATUS "CFD Library Configuration:")
message(STATUS "  CUDA Support: ${CFD_HAS_CUDA}")
if(CFD_HAS_CUDA)
    message(STATUS "  CUDA Architectures: ${CFD_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
