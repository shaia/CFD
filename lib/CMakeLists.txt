# CFD Library CMake Configuration
cmake_minimum_required(VERSION 3.10)

# Option for CUDA support
option(CFD_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# CUDA architecture configuration
# Default: Support wide range of GPUs from Pascal (GTX 10xx) onwards
# Users can override with -DCFD_CUDA_ARCHITECTURES="70;75" etc.
if(NOT DEFINED CFD_CUDA_ARCHITECTURES)
    set(CFD_CUDA_ARCHITECTURES "60;70;75;80;86;89;90" CACHE STRING
        "CUDA architectures to build for (60=Pascal, 70=Volta, 75=Turing, 80=Ampere, 86=Ampere, 89=Ada, 90=Hopper)")
endif()

# Define the library project with optional CUDA
if(CFD_ENABLE_CUDA)
    project(cfd_library VERSION 1.0.0 LANGUAGES C CUDA)
    message(STATUS "CUDA support enabled")
else()
    project(cfd_library VERSION 1.0.0 LANGUAGES C)
    message(STATUS "CUDA support disabled (use -DCFD_ENABLE_CUDA=ON to enable)")
endif()

# OpenMP Support
set(CFD_OMP_SOLVER_SOURCES "")
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found. Multithreading support enabled.")
    add_compile_definitions(CFD_ENABLE_OPENMP)
    set(CFD_OMP_SOLVER_SOURCES
        src/solvers/omp/solver_explicit_euler_omp.c
        src/solvers/omp/solver_projection_omp.c
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
    src/core/utils.c
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

# Collect CPU solver source files
set(CFD_CPU_SOLVER_SOURCES
    src/solvers/cpu/solver_explicit_euler.c
    src/solvers/cpu/solver_projection.c
)

# Collect SIMD-optimized solver source files
set(CFD_SIMD_SOLVER_SOURCES
    src/solvers/simd/solver_explicit_euler_simd.c
    src/solvers/simd/solver_projection_simd.c
)

# Combine all source files
set(CFD_LIBRARY_SOURCES
    ${CFD_CORE_SOURCES}
    ${CFD_IO_SOURCES}
    ${CFD_API_SOURCES}
    ${CFD_CPU_SOLVER_SOURCES}
    ${CFD_SIMD_SOLVER_SOURCES}
    ${CFD_OMP_SOLVER_SOURCES}
)

# Add GPU sources based on CUDA availability
if(CFD_ENABLE_CUDA)
    # Check if CUDA toolkit is actually available
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        list(APPEND CFD_LIBRARY_SOURCES src/solvers/gpu/solver_projection_jacobi_gpu.cu)
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
    include/grid.h
    include/solver_interface.h
    include/solver_gpu.h
    include/utils.h
    include/vtk_output.h
    include/csv_output.h
    include/output_registry.h
    include/simulation_api.h
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
    PUBLIC_HEADER "${CFD_LIBRARY_HEADERS}"
)

# Include directories for main library
target_include_directories(cfd_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# If we built both libraries, set properties for static version too
if(BUILD_SHARED_LIBS)
    set_target_properties(cfd_library_static PROPERTIES
        VERSION ${PROJECT_VERSION}
        PUBLIC_HEADER "${CFD_LIBRARY_HEADERS}"
    )

    target_include_directories(cfd_library_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
endif()

# Function to apply compiler options to a target
function(apply_compiler_options target_name)
    if(MSVC)
        # Only apply MSVC flags to C/CXX, not CUDA
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
        # Enable AVX2 if available on x86/x64
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -O3>)
        # Enable AVX2 and other optimizations if available on x86/x64
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
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
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:llvm>)
        endif()
    endif()
    # CUDA-specific options
    if(CFD_HAS_CUDA)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
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
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cfd
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
