cmake_minimum_required(VERSION 3.10)
project(cfd_framework C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add the library subdirectory
add_subdirectory(lib)

# Example executables
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    # Basic simulation example (original main.c)
    add_executable(basic_simulation examples/basic_simulation.c)
    target_link_libraries(basic_simulation PRIVATE CFD::Library)

    # Minimal example - simplest usage
    add_executable(minimal_example examples/minimal_example.c)
    target_link_libraries(minimal_example PRIVATE CFD::Library)

    # Performance comparison between solvers
    add_executable(performance_comparison examples/performance_comparison.c)
    target_link_libraries(performance_comparison PRIVATE CFD::Library)

    # Custom boundary conditions example
    add_executable(custom_boundary_conditions examples/custom_boundary_conditions.c)
    target_link_libraries(custom_boundary_conditions PRIVATE CFD::Library)
endif()

# Optionally build tests
option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    # Fetch Unity testing framework
    include(FetchContent)
    FetchContent_Declare(
        Unity
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        GIT_TAG v2.6.1
    )

    # Make Unity available
    FetchContent_MakeAvailable(Unity)

    # Add the test executables
    add_executable(test_simulation_basic tests/test_simulation_basic.c)
    add_executable(test_runner tests/test_runner.c)

    # Link test executables with CFD library and Unity
    target_link_libraries(test_simulation_basic PRIVATE CFD::Library unity)
    target_link_libraries(test_runner PRIVATE CFD::Library unity)

    # Enable CTest support (optional)
    enable_testing()
    add_test(NAME BasicSimulationTest COMMAND test_simulation_basic)
    add_test(NAME TestRunner COMMAND test_runner)
endif()