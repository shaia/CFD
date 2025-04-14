# Add source files
set(SOURCES
    main.c
    grid.c
    solver.c
    solver_optimized.c
    utils.c
)

# Add header files
set(HEADERS
    grid.h
    solver.h
    utils.h
)

# Create executable
add_executable(cfd_framework ${SOURCES} ${HEADERS})

# Link against OpenGL and GLFW
target_link_libraries(cfd_framework PRIVATE
    ${EXTRA_LIBS}
)

# Link the math library
target_link_libraries(cfd_framework PRIVATE m)

# Include directories
target_include_directories(cfd_framework PRIVATE ${CMAKE_SOURCE_DIR}/src)