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

# Create the static library
add_library(cfd_library STATIC ${CFD_LIBRARY_SOURCES})

# Set library properties
set_target_properties(cfd_library PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CFD_LIBRARY_HEADERS}"
)

# Include directories
target_include_directories(cfd_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler-specific options
if(MSVC)
    target_compile_options(cfd_library PRIVATE /W3)
    # Enable AVX2 if available
    target_compile_options(cfd_library PRIVATE /arch:AVX2)
else()
    target_compile_options(cfd_library PRIVATE -Wall -Wextra -O3)
    # Enable AVX2 and other optimizations if available
    target_compile_options(cfd_library PRIVATE -mavx2 -mfma)
    # Link math library on Unix systems
    target_link_libraries(cfd_library PUBLIC m)
endif()

# Optional: Create an alias for easier usage
add_library(CFD::Library ALIAS cfd_library)

# Installation rules (optional, for library distribution)
include(GNUInstallDirs)

install(TARGETS cfd_library
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