cmake_minimum_required(VERSION 3.10)

# CUDA support option (must be set before project() for proper language detection)
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

# Define project with appropriate languages
if(CFD_ENABLE_CUDA)
    project(cfd_framework C CUDA)
    message(STATUS "CUDA support enabled at top level")
else()
    project(cfd_framework C)
endif()

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Sanitizer options (for Debug builds on GCC/Clang)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Code coverage option (for GCC/Clang)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

# Explicitly default to static libraries to simplifiy running tests on Windows
set(BUILD_SHARED_LIBS OFF)

if(ENABLE_COVERAGE)
    if(MSVC)
        message(WARNING "Code coverage is not supported with MSVC, skipping")
    else()
        set(COVERAGE_FLAGS "--coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${COVERAGE_FLAGS}")
        message(STATUS "Code coverage enabled")
    endif()
endif()

if(ENABLE_ASAN OR ENABLE_UBSAN)                                                                                                                                             
    if(MSVC)
        message(WARNING "Sanitizers are not fully supported with MSVC, skipping")
    else()
        set(SANITIZER_FLAGS "")

        if(ENABLE_ASAN)
            set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=address")
            message(STATUS "AddressSanitizer enabled")
        endif()

        if(ENABLE_UBSAN)
            set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=undefined")
            message(STATUS "UndefinedBehaviorSanitizer enabled")
        endif()

        # Add common sanitizer flags
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fno-omit-frame-pointer -g")

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    endif()
endif()

# Add the library subdirectory
add_subdirectory(lib)

# Ensure consumers (tests/examples) know we are using statistic library
if(NOT BUILD_SHARED_LIBS)
    add_compile_definitions(CFD_LIBRARY_STATIC_DEFINE)
endif()

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

    # Solver selection example - demonstrates pluggable solver architecture
    add_executable(solver_selection examples/solver_selection.c)
    target_link_libraries(solver_selection PRIVATE CFD::Library)

    # Runtime comparison - benchmarks CUDA vs SIMD implementations
    add_executable(runtime_comparison examples/runtime_comparison.c)
    target_link_libraries(runtime_comparison PRIVATE CFD::Library)

    # CSV data export example - demonstrates CSV output capabilities
    add_executable(csv_data_export examples/csv_data_export.c)
    target_link_libraries(csv_data_export PRIVATE CFD::Library)

    # Lid-driven cavity example - demonstrates Dirichlet boundary conditions
    add_executable(lid_driven_cavity examples/lid_driven_cavity.c)
    target_link_libraries(lid_driven_cavity PRIVATE CFD::Library)
endif()

# Optionally build tests
option(BUILD_TESTS "Build test executables" ON)

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
    
    # Enable double precision for Unity assertions (PUBLIC propagates to tests)
    target_compile_definitions(unity PUBLIC UNITY_INCLUDE_DOUBLE)

    # Add the test executables
    add_executable(test_simulation_basic tests/simulation/test_simulation_basic.c)
    add_executable(test_runner tests/test_runner.c)
    add_executable(test_physics_validation tests/simulation/test_physics_validation.c)
    add_executable(test_decay_prevention tests/simulation/test_decay_prevention.c)
    add_executable(test_output_paths tests/io/test_output_paths.c)

    add_executable(test_csv_output tests/io/test_csv_output.c)
    add_executable(test_vtk_output tests/io/test_vtk_output.c)
    add_executable(test_derived_fields tests/simulation/test_derived_fields.c)
    add_executable(test_simulation_api tests/simulation/test_simulation_api.c)
    add_executable(test_solver_gpu_api tests/solvers/gpu/test_solver_gpu_api.c)
    add_executable(test_core_functionality tests/core/test_core_functionality.c)
    add_executable(test_grid tests/core/test_grid.c)
    add_executable(test_error_handling tests/core/test_error_handling.c)
    add_executable(test_reentrancy tests/simulation/test_reentrancy.c)
    add_executable(test_input_validation tests/core/test_input_validation.c)
    add_executable(test_logging tests/core/test_logging.c)
    add_executable(test_init tests/core/test_init.c)
    add_executable(test_boundary_conditions_dirichlet tests/core/test_boundary_conditions_dirichlet.c)
    add_executable(test_boundary_conditions_noslip tests/core/test_boundary_conditions_noslip.c)
    add_executable(test_boundary_conditions_inlet tests/core/test_boundary_conditions_inlet.c)
    add_executable(test_boundary_conditions_outlet tests/core/test_boundary_conditions_outlet.c)

    # Navier-Stokes solver tests (CPU)
    add_executable(test_solver_explicit_euler tests/solvers/navier_stokes/cpu/test_solver_explicit_euler.c)
    add_executable(test_solver_projection tests/solvers/navier_stokes/cpu/test_solver_projection.c)

    # Navier-Stokes solver tests (AVX2)
    add_executable(test_solver_explicit_euler_avx2 tests/solvers/navier_stokes/avx2/test_solver_explicit_euler_avx2.c)
    add_executable(test_solver_projection_avx2 tests/solvers/navier_stokes/avx2/test_solver_projection_avx2.c)

    # Navier-Stokes solver tests (OpenMP)
    add_executable(test_solver_explicit_euler_omp tests/solvers/navier_stokes/omp/test_solver_explicit_euler_omp.c)
    add_executable(test_solver_projection_omp tests/solvers/navier_stokes/omp/test_solver_projection_omp.c)

    # Navier-Stokes solver tests (GPU)
    add_executable(test_solver_projection_jacobi_gpu tests/solvers/navier_stokes/gpu/test_solver_projection_jacobi_gpu.c)

    # Linear solver abstraction tests
    add_executable(test_linear_solver tests/solvers/test_linear_solver.c)

    # Solver backend API tests
    add_executable(test_solver_backend_api tests/solvers/test_solver_backend_api.c)

    # Validation tests - cavity setup and flow
    add_executable(test_cavity_setup tests/validation/test_cavity_setup.c)
    add_executable(test_cavity_flow tests/validation/test_cavity_flow.c)
    add_executable(test_cavity_validation tests/validation/test_cavity_validation.c)
    add_executable(test_cavity_reference tests/validation/test_cavity_reference.c)

    # Ghia validation - per-architecture test files
    add_executable(test_ghia_euler_cpu tests/validation/test_ghia_euler_cpu.c)
    add_executable(test_ghia_euler_avx2 tests/validation/test_ghia_euler_avx2.c)
    add_executable(test_ghia_euler_omp tests/validation/test_ghia_euler_omp.c)
    add_executable(test_ghia_projection_cpu tests/validation/test_ghia_projection_cpu.c)
    add_executable(test_ghia_projection_avx2 tests/validation/test_ghia_projection_avx2.c)
    add_executable(test_ghia_projection_omp tests/validation/test_ghia_projection_omp.c)
    add_executable(test_ghia_projection_gpu tests/validation/test_ghia_projection_gpu.c)

    # Taylor-Green vortex validation test
    add_executable(test_taylor_green_vortex tests/validation/test_taylor_green_vortex.c)

    # Mathematical accuracy tests
    add_executable(test_finite_differences tests/math/test_finite_differences.c)

    # Cross-architecture consistency tests
    add_executable(test_solver_architecture tests/validation/test_solver_architecture.c)

    # Modular backend library tests
    add_executable(test_modular_libraries tests/core/test_modular_libraries.c)

    # Backend-focused tests - verify specific backend functionality
    # Note: Currently linked against CFD::Library because the architecture has
    # cross-backend symbol dependencies. True modular linking requires refactoring
    # to use weak symbols or conditional registration (see ROADMAP.md 4.2).
    add_executable(test_modular_core_scalar tests/core/test_modular_core_scalar.c)
    add_executable(test_modular_core_simd tests/core/test_modular_core_simd.c)

    # Link test executables with CFD library and Unity
    target_link_libraries(test_simulation_basic PRIVATE CFD::Library unity)
    target_link_libraries(test_runner PRIVATE CFD::Library unity)
    target_link_libraries(test_physics_validation PRIVATE CFD::Library unity)
    target_link_libraries(test_decay_prevention PRIVATE CFD::Library unity)
    target_link_libraries(test_output_paths PRIVATE CFD::Library unity)

    target_link_libraries(test_csv_output PRIVATE CFD::Library unity)
    target_link_libraries(test_vtk_output PRIVATE CFD::Library unity)
    target_link_libraries(test_derived_fields PRIVATE CFD::Library unity)
    target_link_libraries(test_simulation_api PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_gpu_api PRIVATE CFD::Library unity)
    target_link_libraries(test_core_functionality PRIVATE CFD::Library unity)
    target_link_libraries(test_grid PRIVATE CFD::Library unity)
    target_link_libraries(test_error_handling PRIVATE CFD::Library unity)
    target_link_libraries(test_reentrancy PRIVATE CFD::Library unity)
    target_link_libraries(test_input_validation PRIVATE CFD::Library unity)
    target_link_libraries(test_logging PRIVATE CFD::Library unity)
    target_link_libraries(test_init PRIVATE CFD::Library unity)
    target_link_libraries(test_boundary_conditions_dirichlet PRIVATE CFD::Library unity)
    target_link_libraries(test_boundary_conditions_noslip PRIVATE CFD::Library unity)
    target_link_libraries(test_boundary_conditions_inlet PRIVATE CFD::Library unity)
    target_link_libraries(test_boundary_conditions_outlet PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_explicit_euler PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_projection PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_explicit_euler_avx2 PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_projection_avx2 PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_explicit_euler_omp PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_projection_omp PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_projection_jacobi_gpu PRIVATE CFD::Library unity)
    target_link_libraries(test_linear_solver PRIVATE CFD::Library unity)
    target_link_libraries(test_solver_backend_api PRIVATE CFD::Library unity)
    target_link_libraries(test_modular_libraries PRIVATE CFD::Library unity)

    # Backend-focused tests - linked against unified library due to cross-backend dependencies
    # TODO: Refactor architecture to support true modular linking (weak symbols/conditional registration)
    target_link_libraries(test_modular_core_scalar PRIVATE CFD::Library unity)
    target_link_libraries(test_modular_core_simd PRIVATE CFD::Library unity)

    target_link_libraries(test_cavity_setup PRIVATE CFD::Library unity)
    target_link_libraries(test_cavity_flow PRIVATE CFD::Library unity)
    target_link_libraries(test_cavity_validation PRIVATE CFD::Library unity)
    target_link_libraries(test_cavity_reference PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_euler_cpu PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_euler_avx2 PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_euler_omp PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_projection_cpu PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_projection_avx2 PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_projection_omp PRIVATE CFD::Library unity)
    target_link_libraries(test_ghia_projection_gpu PRIVATE CFD::Library unity)
    target_link_libraries(test_taylor_green_vortex PRIVATE CFD::Library unity)
    target_link_libraries(test_finite_differences PRIVATE unity $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
    target_include_directories(test_finite_differences PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/include)
    target_link_libraries(test_solver_architecture PRIVATE CFD::Library unity)

    # Enable CTest support (optional)
    enable_testing()
    add_test(NAME BasicSimulationTest COMMAND test_simulation_basic)
    add_test(NAME TestRunner COMMAND test_runner)
    add_test(NAME PhysicsValidationTest COMMAND test_physics_validation)
    add_test(NAME DecayPreventionTest COMMAND test_decay_prevention)
    add_test(NAME OutputPathsTest COMMAND test_output_paths)

    add_test(NAME CsvOutputTest COMMAND test_csv_output)
    add_test(NAME VtkOutputTest COMMAND test_vtk_output)
    add_test(NAME DerivedFieldsTest COMMAND test_derived_fields)
    add_test(NAME SimulationApiTest COMMAND test_simulation_api)
    add_test(NAME SolverGpuApiTest COMMAND test_solver_gpu_api)
    add_test(NAME CoreFunctionalityTest COMMAND test_core_functionality)
    add_test(NAME GridTest COMMAND test_grid)
    add_test(NAME ErrorHandlingTest COMMAND test_error_handling)
    add_test(NAME ReentrancyTest COMMAND test_reentrancy)
    add_test(NAME InputValidationTest COMMAND test_input_validation)
    add_test(NAME LoggingTest COMMAND test_logging)
    add_test(NAME InitializationTest COMMAND test_init)
    add_test(NAME BoundaryConditionsDirichletTest COMMAND test_boundary_conditions_dirichlet)
    add_test(NAME BoundaryConditionsNoslipTest COMMAND test_boundary_conditions_noslip)
    add_test(NAME BoundaryConditionsInletTest COMMAND test_boundary_conditions_inlet)
    add_test(NAME BoundaryConditionsOutletTest COMMAND test_boundary_conditions_outlet)
    add_test(NAME SolverExplicitEulerTest COMMAND test_solver_explicit_euler)
    add_test(NAME SolverProjectionTest COMMAND test_solver_projection)
    add_test(NAME SolverExplicitEulerAvx2Test COMMAND test_solver_explicit_euler_avx2)
    add_test(NAME SolverProjectionAvx2Test COMMAND test_solver_projection_avx2)
    add_test(NAME SolverExplicitEulerOmpTest COMMAND test_solver_explicit_euler_omp)
    add_test(NAME SolverProjectionOmpTest COMMAND test_solver_projection_omp)
    add_test(NAME SolverProjectionJacobiGpuTest COMMAND test_solver_projection_jacobi_gpu)
    add_test(NAME LinearSolverTest COMMAND test_linear_solver)
    add_test(NAME SolverBackendApiTest COMMAND test_solver_backend_api)
    add_test(NAME ModularLibrariesTest COMMAND test_modular_libraries)
    add_test(NAME ModularCoreScalarTest COMMAND test_modular_core_scalar)
    add_test(NAME ModularCoreSIMDTest COMMAND test_modular_core_simd)
    add_test(NAME CavitySetupTest COMMAND test_cavity_setup)
    add_test(NAME CavityFlowTest COMMAND test_cavity_flow)
    add_test(NAME CavityValidationTest COMMAND test_cavity_validation)
    add_test(NAME CavityReferenceTest COMMAND test_cavity_reference)

    # Ghia validation tests - per architecture
    # These run long simulations (3000-5000 steps) to validate physics accuracy
    add_test(NAME GhiaEulerCpuTest COMMAND test_ghia_euler_cpu)
    add_test(NAME GhiaEulerAvx2Test COMMAND test_ghia_euler_avx2)
    add_test(NAME GhiaEulerOmpTest COMMAND test_ghia_euler_omp)
    add_test(NAME GhiaProjectionCpuTest COMMAND test_ghia_projection_cpu)
    add_test(NAME GhiaProjectionAvx2Test COMMAND test_ghia_projection_avx2)
    add_test(NAME GhiaProjectionOmpTest COMMAND test_ghia_projection_omp)
    add_test(NAME GhiaProjectionGpuTest COMMAND test_ghia_projection_gpu)

    # Taylor-Green vortex validation test
    add_test(NAME TaylorGreenVortexTest COMMAND test_taylor_green_vortex)

    # Mathematical accuracy tests
    add_test(NAME FiniteDifferencesTest COMMAND test_finite_differences)

    # Label long-running validation tests to exclude from sanitizer job
    # These tests validate physics accuracy, not memory safety
    set_tests_properties(
        CavityValidationTest
        CavityReferenceTest
        GhiaEulerCpuTest
        GhiaEulerAvx2Test
        GhiaEulerOmpTest
        GhiaProjectionCpuTest
        GhiaProjectionAvx2Test
        GhiaProjectionOmpTest
        GhiaProjectionGpuTest
        TaylorGreenVortexTest
        PROPERTIES LABELS "validation"
    )

    # Cross-architecture consistency tests
    # This test requires AVX2+OpenMP and is run only in the dedicated CI job
    add_test(NAME SolverArchitectureTest COMMAND test_solver_architecture)
    set_tests_properties(SolverArchitectureTest PROPERTIES LABELS "cross-arch")
endif()

# Documentation target (Doxygen)
option(BUILD_DOCS "Build API documentation with Doxygen" OFF)

if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        message(STATUS "Doxygen found - 'docs' target available")
    else()
        message(WARNING "Doxygen not found - documentation will not be built")
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "CFD Framework Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${CFD_ENABLE_CUDA}")
if(CFD_ENABLE_CUDA)
    message(STATUS "  CUDA Architectures: ${CFD_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Docs: ${BUILD_DOCS}")
message(STATUS "")