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

    # Velocity visualization example
    add_executable(velocity_visualization examples/velocity_visualization.c)
    target_link_libraries(velocity_visualization PRIVATE CFD::Library)

    # Animated flow simulation example
    add_executable(animated_flow_simulation examples/animated_flow_simulation.c)
    target_link_libraries(animated_flow_simulation PRIVATE CFD::Library)

    # Simple animated flow demo
    add_executable(simple_animated_flow examples/simple_animated_flow.c)
    target_link_libraries(simple_animated_flow PRIVATE CFD::Library)

    # Custom source terms example
    add_executable(custom_source_terms examples/custom_source_terms.c)
    target_link_libraries(custom_source_terms PRIVATE CFD::Library)
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
    add_executable(test_solver_consistency tests/test_solver_consistency.c)
    add_executable(test_physics_validation tests/test_physics_validation.c)
    add_executable(test_decay_prevention tests/test_decay_prevention.c)
    add_executable(test_output_paths tests/test_output_paths.c)
    add_executable(test_comprehensive tests/test_comprehensive.c)
    add_executable(test_fixed_comprehensive tests/test_fixed_comprehensive.c)
    add_executable(test_solver_basic tests/test_solver_basic.c)
    add_executable(test_solver_debug tests/test_solver_debug.c)

    # Link test executables with CFD library and Unity
    target_link_libraries(test_simulation_basic PRIVATE CFD::Library unity)
    target_link_libraries(test_runner PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_consistency PRIVATE CFD::Library unity)
    target_link_libraries(test_physics_validation PRIVATE CFD::Library unity)
    target_link_libraries(test_decay_prevention PRIVATE CFD::Library unity)
    target_link_libraries(test_output_paths PRIVATE CFD::Library unity)
    target_link_libraries(test_comprehensive PRIVATE CFD::Library unity)
    target_link_libraries(test_fixed_comprehensive PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_basic PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_debug PRIVATE CFD::Library unity)

    # Enable CTest support (optional)
    enable_testing()
    add_test(NAME BasicSimulationTest COMMAND test_simulation_basic)
    add_test(NAME TestRunner COMMAND test_runner)
    add_test(NAME SolverConsistencyTest COMMAND test_solver_consistency)
    add_test(NAME PhysicsValidationTest COMMAND test_physics_validation)
    add_test(NAME DecayPreventionTest COMMAND test_decay_prevention)
    add_test(NAME OutputPathsTest COMMAND test_output_paths)
    add_test(NAME ComprehensiveTest COMMAND test_comprehensive)
    add_test(NAME SolverBasicTest COMMAND test_solver_basic)
endif()