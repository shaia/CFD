# CFD Library CMake Configuration
cmake_minimum_required(VERSION 3.10)

# Define the library project
project(cfd_library VERSION 1.0.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Collect all source files
set(CFD_LIBRARY_SOURCES
    src/grid.c
    src/solver.c
    src/solver_optimized.c
    src/utils.c
    src/vtk_output.c
    src/simulation_api.c
)

# Collect all header files
set(CFD_LIBRARY_HEADERS
    include/grid.h
    include/solver.h
    include/utils.h
    include/vtk_output.h
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
        target_compile_options(${target_name} PRIVATE /W3)
        # Enable AVX2 if available on x86/x64
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
            target_compile_options(${target_name} PRIVATE /arch:AVX2)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -O3)
        # Enable AVX2 and other optimizations if available on x86/x64
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
            target_compile_options(${target_name} PRIVATE -mavx2 -mfma)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            # ARM64-specific optimizations
            target_compile_options(${target_name} PRIVATE -mcpu=native)
        endif()
        # Link math library on Unix systems
        target_link_libraries(${target_name} PUBLIC m)
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