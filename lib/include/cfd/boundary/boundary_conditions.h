#ifndef CFD_BOUNDARY_CONDITIONS_H
#define CFD_BOUNDARY_CONDITIONS_H

#include "cfd/cfd_export.h"
#include "cfd/core/cfd_status.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Boundary Condition Types
 *
 * Defines the types of boundary conditions available for CFD simulations.
 */
typedef enum {
    BC_TYPE_PERIODIC,   // Wrap-around: left boundary = right interior, etc.
    BC_TYPE_NEUMANN,    // Zero gradient: boundary = adjacent interior value
    BC_TYPE_DIRICHLET,  // Fixed value: boundary = specified constant value
    BC_TYPE_NOSLIP,     // No-slip wall: velocity = 0 at all boundaries
    BC_TYPE_INLET,      // Inlet velocity specification (placeholder for future)
    BC_TYPE_OUTLET      // Outlet/convective (placeholder for future)
} bc_type_t;

/**
 * Boundary Condition Backend Types
 *
 * Specifies which implementation backend to use for boundary conditions.
 * This allows runtime selection of the most appropriate implementation
 * based on the solver type being used.
 */
typedef enum {
    BC_BACKEND_AUTO,    // Auto-select best available (SIMD > OMP > Scalar)
    BC_BACKEND_SCALAR,  // Force scalar implementation (single-threaded)
    BC_BACKEND_OMP,     // Force OpenMP implementation (multi-threaded, scalar loops)
    BC_BACKEND_SIMD,    // Force SIMD + OpenMP (runtime: AVX2 on x86, NEON on ARM)
    BC_BACKEND_CUDA     // Force CUDA GPU implementation
} bc_backend_t;

/**
 * Dirichlet Boundary Condition Values
 *
 * Specifies fixed values for each boundary in Dirichlet (fixed value) BCs.
 * Used with bc_apply_dirichlet_* functions.
 */
typedef struct {
    double left;    // Value at x=0 boundary (column 0)
    double right;   // Value at x=Lx boundary (column nx-1)
    double top;     // Value at y=Ly boundary (row ny-1)
    double bottom;  // Value at y=0 boundary (row 0)
} bc_dirichlet_values_t;

/**
 * Inlet Velocity Profile Types
 *
 * Defines the velocity profile shape for inlet boundary conditions.
 */
typedef enum {
    BC_INLET_PROFILE_UNIFORM,     // Constant velocity across inlet
    BC_INLET_PROFILE_PARABOLIC,   // Parabolic profile (fully-developed flow)
    BC_INLET_PROFILE_CUSTOM       // User-defined profile via callback
} bc_inlet_profile_t;

/**
 * Outlet Boundary Condition Types
 *
 * Defines the type of outlet boundary condition.
 */
typedef enum {
    BC_OUTLET_ZERO_GRADIENT,   // Neumann: boundary = adjacent interior value
    BC_OUTLET_CONVECTIVE       // Advective: du/dt + U*du/dn = 0
} bc_outlet_type_t;

/**
 * Inlet Velocity Specification Type
 *
 * Defines how the inlet velocity is specified.
 */
typedef enum {
    BC_INLET_SPEC_VELOCITY,       // Fixed velocity components (u, v)
    BC_INLET_SPEC_MAGNITUDE_DIR,  // Velocity magnitude + direction angle
    BC_INLET_SPEC_MASS_FLOW       // Mass flow rate (requires density)
} bc_inlet_spec_type_t;

/**
 * Boundary Edge Identifier
 *
 * Specifies which boundary edge an inlet applies to.
 */
typedef enum {
    BC_EDGE_LEFT   = 0x01,   // x=0 boundary (column 0)
    BC_EDGE_RIGHT  = 0x02,   // x=Lx boundary (column nx-1)
    BC_EDGE_BOTTOM = 0x04,   // y=0 boundary (row 0)
    BC_EDGE_TOP    = 0x08    // y=Ly boundary (row ny-1)
} bc_edge_t;

/* ============================================================================
 * Time-Varying Boundary Condition Types
 *
 * Support for boundary conditions that vary with simulation time.
 * Enables pulsatile flow, ramp-up transients, and oscillating boundaries.
 * ============================================================================ */

/**
 * Time Profile Types
 *
 * Defines how a boundary condition varies with time.
 */
typedef enum {
    BC_TIME_PROFILE_CONSTANT,     // No time variation (default)
    BC_TIME_PROFILE_SINUSOIDAL,   // Sinusoidal: offset + amplitude * sin(2*pi*freq*t + phase)
    BC_TIME_PROFILE_RAMP,         // Linear ramp between two times
    BC_TIME_PROFILE_STEP,         // Step change at specified time
    BC_TIME_PROFILE_CUSTOM        // User-defined time function
} bc_time_profile_t;

/**
 * Time Context
 *
 * Provides current simulation time information to time-varying BC functions.
 */
typedef struct {
    double time;    // Current simulation time
    double dt;      // Current time step size
} bc_time_context_t;

/**
 * Sinusoidal Time Profile Configuration
 *
 * Produces: offset + amplitude * sin(2*pi*frequency*t + phase)
 */
typedef struct {
    double frequency;   // Frequency in Hz
    double amplitude;   // Amplitude (multiplier)
    double phase;       // Phase offset in radians
    double offset;      // DC offset (1.0 = no attenuation at mean)
} bc_time_sinusoidal_t;

/**
 * Ramp Time Profile Configuration
 *
 * Linear interpolation from value_start to value_end over [t_start, t_end].
 */
typedef struct {
    double t_start;      // Ramp start time
    double t_end;        // Ramp end time
    double value_start;  // Multiplier at t_start
    double value_end;    // Multiplier at t_end
} bc_time_ramp_t;

/**
 * Step Time Profile Configuration
 *
 * Instantaneous change from value_before to value_after at t_step.
 */
typedef struct {
    double t_step;        // Time of step change
    double value_before;  // Multiplier before t_step
    double value_after;   // Multiplier after t_step
} bc_time_step_t;

/**
 * Custom time modulation callback function type.
 *
 * @param time       Current simulation time
 * @param dt         Current time step size
 * @param user_data  User-provided context pointer
 * @return           Modulation factor to multiply base velocity by
 */
typedef double (*bc_time_custom_fn)(double time, double dt, void* user_data);

/**
 * Time Profile Configuration
 *
 * Unified structure for all time profile types.
 */
typedef struct {
    bc_time_profile_t profile;    // Which time profile to use

    union {
        bc_time_sinusoidal_t sinusoidal;
        bc_time_ramp_t ramp;
        bc_time_step_t step;
    } params;

    // Custom callback (used when profile == BC_TIME_PROFILE_CUSTOM)
    bc_time_custom_fn custom_fn;
    void* custom_user_data;
} bc_time_config_t;

/**
 * Custom inlet profile callback function type.
 *
 * @param position   Normalized position along the inlet (0.0 to 1.0)
 * @param u_out      Output: x-velocity component at this position
 * @param v_out      Output: y-velocity component at this position
 * @param user_data  User-provided context pointer
 *
 * The callback is called for each grid point along the inlet boundary.
 * Position 0.0 is at the start of the boundary, 1.0 is at the end.
 */
typedef void (*bc_inlet_profile_fn)(double position, double* u_out, double* v_out, void* user_data);

/**
 * Time-varying inlet profile callback function type.
 *
 * Extended callback that receives both spatial position and time information.
 *
 * @param position   Normalized position along the inlet (0.0 to 1.0)
 * @param time       Current simulation time
 * @param dt         Current time step size
 * @param u_out      Output: x-velocity component at this position and time
 * @param v_out      Output: y-velocity component at this position and time
 * @param user_data  User-provided context pointer
 */
typedef void (*bc_inlet_profile_time_fn)(double position, double time, double dt,
                                          double* u_out, double* v_out, void* user_data);

/**
 * Inlet Boundary Condition Configuration
 *
 * Comprehensive structure for specifying inlet velocity boundary conditions.
 * Supports uniform, parabolic, and custom velocity profiles.
 */
typedef struct {
    bc_edge_t edge;                    // Which boundary edge this inlet applies to
    bc_inlet_profile_t profile;        // Velocity profile type
    bc_inlet_spec_type_t spec_type;    // How velocity is specified

    // Velocity specification (interpretation depends on spec_type)
    union {
        struct {
            double u;                  // x-velocity component
            double v;                  // y-velocity component
        } velocity;                    // For BC_INLET_SPEC_VELOCITY

        struct {
            double magnitude;          // Velocity magnitude
            double direction;          // Direction angle in radians (0 = +x, pi/2 = +y)
        } magnitude_dir;               // For BC_INLET_SPEC_MAGNITUDE_DIR

        struct {
            double mass_flow_rate;     // Mass flow rate (kg/s per unit depth for 2D)
            double density;            // Fluid density (kg/m^3)
            double inlet_length;       // Physical length of inlet (m)
        } mass_flow;                   // For BC_INLET_SPEC_MASS_FLOW
    } spec;

    // Custom profile callback (only used when profile == BC_INLET_PROFILE_CUSTOM)
    bc_inlet_profile_fn custom_profile;
    void* custom_profile_user_data;

    // Time variation configuration (optional, zero-initialized = no time variation)
    bc_time_config_t time_config;

    // Time-varying custom profile (overrides custom_profile when set)
    bc_inlet_profile_time_fn custom_profile_time;
    void* custom_profile_time_user_data;
} bc_inlet_config_t;

/**
 * Outlet Boundary Condition Configuration
 *
 * Configuration structure for specifying outlet boundary conditions.
 * Supports zero-gradient (Neumann) and convective outlet types.
 */
typedef struct {
    bc_edge_t edge;              // Which boundary edge this outlet applies to
    bc_outlet_type_t type;       // Type of outlet BC

    // Convective outlet parameters (only used when type == BC_OUTLET_CONVECTIVE)
    double advection_velocity;   // Advection velocity for convective outlet (m/s)
                                 // Typically the mean outflow velocity
} bc_outlet_config_t;

/**
 * Apply boundary conditions to a scalar field (raw array)
 *
 * @param field Pointer to the scalar field array (size nx*ny)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 * @param type  Type of boundary condition to apply
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions to velocity components (u, v arrays)
 *
 * @param u     Pointer to x-velocity array (size nx*ny)
 * @param v     Pointer to y-velocity array (size nx*ny)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 * @param type  Type of boundary condition to apply
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/* ============================================================================
 * Convenience Macros
 *
 * Shorthand macros for applying common boundary condition types.
 * These use the global backend setting (see bc_set_backend()).
 * ============================================================================ */

/**
 * Apply Neumann (zero-gradient) boundary conditions to a scalar field.
 *
 * Sets boundary values equal to adjacent interior values:
 *   - Left:   field[0,j] = field[1,j]
 *   - Right:  field[nx-1,j] = field[nx-2,j]
 *   - Bottom: field[i,0] = field[i,1]
 *   - Top:    field[i,ny-1] = field[i,ny-2]
 *
 * @param field Pointer to scalar field array (size nx*ny, row-major)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 */
#define bc_apply_neumann(field, nx, ny)  bc_apply_scalar((field), (nx), (ny), BC_TYPE_NEUMANN)

/**
 * Apply periodic boundary conditions to a scalar field.
 *
 * Wraps values from opposite boundaries:
 *   - Left:   field[0,j] = field[nx-2,j]
 *   - Right:  field[nx-1,j] = field[1,j]
 *   - Bottom: field[i,0] = field[i,ny-2]
 *   - Top:    field[i,ny-1] = field[i,1]
 *
 * @param field Pointer to scalar field array (size nx*ny, row-major)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 */
#define bc_apply_periodic(field, nx, ny) bc_apply_scalar((field), (nx), (ny), BC_TYPE_PERIODIC)

/* ============================================================================
 * Error Handler API
 *
 * Allows users to customize error handling behavior for internal errors.
 * By default, errors are logged to stderr. Users can provide a custom handler
 * for integration with their own logging/error management systems.
 * ============================================================================ */

/**
 * Error codes for boundary condition operations.
 */
typedef enum {
    BC_ERROR_NONE = 0,
    BC_ERROR_NO_SIMD_BACKEND,    /**< SIMD backend called but no SIMD available */
    BC_ERROR_INTERNAL            /**< Internal library error */
} bc_error_code_t;

/**
 * Error handler callback function type.
 *
 * @param error_code  The error code indicating the type of error
 * @param function    Name of the function where the error occurred
 * @param message     Human-readable error message
 * @param user_data   User-provided context pointer (from bc_set_error_handler)
 *
 * The handler is called when an internal error occurs that cannot be reported
 * through the normal return value mechanism (e.g., in dispatcher functions).
 *
 * After the handler returns, the library will fall back to scalar implementation
 * if possible, or return without applying the boundary condition.
 */
typedef void (*bc_error_handler_t)(bc_error_code_t error_code,
                                    const char* function,
                                    const char* message,
                                    void* user_data);

/**
 * Set a custom error handler for boundary condition operations.
 *
 * @param handler    The error handler callback, or NULL to restore default behavior
 * @param user_data  User-provided context pointer passed to the handler
 *
 * The default handler prints errors to stderr.
 * Setting handler to NULL restores this default behavior.
 *
 * Thread-safety: This function is NOT thread-safe. Set the handler once
 * during initialization before any boundary condition operations.
 */
CFD_LIBRARY_EXPORT void bc_set_error_handler(bc_error_handler_t handler, void* user_data);

/**
 * Get the current error handler.
 *
 * @return The current error handler, or NULL if using default behavior
 */
CFD_LIBRARY_EXPORT bc_error_handler_t bc_get_error_handler(void);

/* ============================================================================
 * Backend Selection API
 * ============================================================================ */

/**
 * Get the currently active BC backend.
 *
 * @return The current backend type
 */
CFD_LIBRARY_EXPORT bc_backend_t bc_get_backend(void);

/**
 * Get the name of the currently active BC backend as a string.
 *
 * @return Human-readable backend name (e.g., "scalar", "omp", "simd", "cuda")
 *         For simd, may include architecture detail like "simd (avx2)" or "simd (neon)"
 */
CFD_LIBRARY_EXPORT const char* bc_get_backend_name(void);

/**
 * Set the BC backend to use for subsequent operations.
 *
 * @param backend The backend to use
 * @return true if the backend was set successfully, false if unavailable
 *
 * Note: BC_BACKEND_AUTO always succeeds and selects the best available.
 *       Other backends may fail if not compiled in or not supported.
 */
CFD_LIBRARY_EXPORT bool bc_set_backend(bc_backend_t backend);

/**
 * Check if a specific backend is available.
 *
 * @param backend The backend to check
 * @return true if the backend is available, false otherwise
 */
CFD_LIBRARY_EXPORT bool bc_backend_available(bc_backend_t backend);

/* ============================================================================
 * Explicit Backend API
 *
 * These functions allow direct selection of a specific implementation,
 * bypassing the global backend setting. Useful when different solvers
 * need different BC implementations.
 * ============================================================================ */

/**
 * Apply boundary conditions using scalar implementation.
 * Always available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar_cpu(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar_simd(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar_omp(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply velocity boundary conditions using scalar implementation.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity_cpu(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply velocity boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity_simd(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply velocity boundary conditions using OpenMP implementation.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/* ============================================================================
 * Dirichlet Boundary Conditions API
 *
 * Dirichlet BCs set boundary values to fixed specified values.
 * Unlike Neumann/Periodic, these require explicit values to be provided.
 * ============================================================================ */

/**
 * Apply Dirichlet (fixed value) boundary conditions to a scalar field.
 *
 * Sets each boundary to the corresponding value in the values struct:
 *   - Left:   field[0,j] = values->left
 *   - Right:  field[nx-1,j] = values->right
 *   - Bottom: field[i,0] = values->bottom
 *   - Top:    field[i,ny-1] = values->top
 *
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param field  Pointer to scalar field array (size nx*ny, row-major)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 * @param values Pointer to struct containing boundary values
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar(double* field, size_t nx, size_t ny,
                                                           const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet boundary conditions to velocity components (u, v).
 *
 * @param u        Pointer to x-velocity array (size nx*ny)
 * @param v        Pointer to y-velocity array (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param u_values Pointer to struct containing u-velocity boundary values
 * @param v_values Pointer to struct containing v-velocity boundary values
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                                             const bc_dirichlet_values_t* u_values,
                                                             const bc_dirichlet_values_t* v_values);

/* Backend-specific Dirichlet implementations */

/**
 * Apply Dirichlet boundary conditions using scalar implementation.
 * Always available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar_cpu(double* field, size_t nx, size_t ny,
                                                               const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar_simd(double* field, size_t nx, size_t ny,
                                                                    const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet boundary conditions using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar_omp(double* field, size_t nx, size_t ny,
                                                               const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet velocity boundary conditions using scalar implementation.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity_cpu(double* u, double* v, size_t nx, size_t ny,
                                                                 const bc_dirichlet_values_t* u_values,
                                                                 const bc_dirichlet_values_t* v_values);

/**
 * Apply Dirichlet velocity boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity_simd(double* u, double* v, size_t nx, size_t ny,
                                                                      const bc_dirichlet_values_t* u_values,
                                                                      const bc_dirichlet_values_t* v_values);

/**
 * Apply Dirichlet velocity boundary conditions using OpenMP implementation.
 * @return CFD_ERROR_UNSUPPORTED if OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity_omp(double* u, double* v, size_t nx, size_t ny,
                                                                 const bc_dirichlet_values_t* u_values,
                                                                 const bc_dirichlet_values_t* v_values);

/**
 * Convenience macro for applying Dirichlet BCs to a scalar field.
 */
#define bc_apply_dirichlet(field, nx, ny, values) \
    bc_apply_dirichlet_scalar((field), (nx), (ny), (values))

/* ============================================================================
 * No-Slip Wall Boundary Conditions API
 *
 * No-slip BCs enforce zero velocity at solid walls.
 * This is the standard wall boundary condition for viscous flows.
 * Equivalent to Dirichlet BCs with all values set to 0.
 * ============================================================================ */

/**
 * Apply no-slip wall boundary conditions to velocity components.
 *
 * Sets both u and v velocity components to zero at all boundaries:
 *   - u = 0, v = 0 at left, right, top, and bottom walls
 *
 * This is the standard boundary condition for solid walls in viscous flow.
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip(double* u, double* v, size_t nx, size_t ny);

/**
 * Apply no-slip wall boundary conditions using scalar implementation.
 * Always available.
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip_cpu(double* u, double* v, size_t nx, size_t ny);

/**
 * Apply no-slip wall boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip_simd(double* u, double* v, size_t nx, size_t ny);

/**
 * Apply no-slip wall boundary conditions using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip_omp(double* u, double* v, size_t nx, size_t ny);

/**
 * Convenience macro for applying no-slip BCs to velocity fields.
 */
#define bc_apply_noslip_velocity(u, v, nx, ny) bc_apply_noslip((u), (v), (nx), (ny))

/* ============================================================================
 * Inlet Velocity Boundary Conditions API
 *
 * Inlet BCs specify velocity at inflow boundaries.
 * Supports uniform, parabolic, and custom velocity profiles.
 * ============================================================================ */

/**
 * Create a default inlet configuration for uniform velocity.
 *
 * Creates a uniform velocity inlet with the specified velocity components.
 * The inlet is configured for the left boundary by default.
 *
 * @param u_velocity  x-velocity component at inlet
 * @param v_velocity  y-velocity component at inlet
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_uniform(double u_velocity, double v_velocity);

/**
 * Create a default inlet configuration for parabolic velocity profile.
 *
 * Creates a parabolic velocity inlet (fully-developed laminar flow).
 * For left/right inlets: u is parabolic, v is zero.
 * For top/bottom inlets: v is parabolic, u is zero.
 *
 * @param max_velocity  Maximum velocity at center of inlet
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_parabolic(double max_velocity);

/**
 * Create an inlet configuration from velocity magnitude and direction.
 *
 * @param magnitude  Velocity magnitude
 * @param direction  Direction angle in radians (0 = +x, pi/2 = +y)
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_magnitude_dir(double magnitude, double direction);

/**
 * Create an inlet configuration from mass flow rate.
 *
 * @param mass_flow_rate  Mass flow rate (kg/s per unit depth for 2D)
 * @param density         Fluid density (kg/m^3)
 * @param inlet_length    Physical length of inlet (m)
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_mass_flow(double mass_flow_rate, double density, double inlet_length);

/**
 * Create an inlet configuration with custom profile callback.
 *
 * @param callback   Function called for each grid point to get velocity
 * @param user_data  User-provided context pointer passed to callback
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_custom(bc_inlet_profile_fn callback, void* user_data);

/* ============================================================================
 * Time-Varying Inlet Configuration Builders
 * ============================================================================ */

/**
 * Create a sinusoidal time-varying inlet.
 *
 * Velocity = (u,v) * (offset + amplitude * sin(2*pi*frequency*t + phase))
 *
 * @param u_velocity   Base x-velocity component
 * @param v_velocity   Base y-velocity component
 * @param frequency    Oscillation frequency in Hz
 * @param amplitude    Oscillation amplitude (multiplier)
 * @param phase        Phase offset in radians
 * @param offset       DC offset (1.0 = oscillates around base velocity)
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_time_sinusoidal(
    double u_velocity, double v_velocity,
    double frequency, double amplitude, double phase, double offset);

/**
 * Create a ramp inlet (smooth start-up).
 *
 * Velocity ramps linearly from value_start to value_end over [t_start, t_end].
 * Before t_start: velocity = (u,v) * value_start
 * After t_end: velocity = (u,v) * value_end
 *
 * @param u_velocity   Target x-velocity component (at value_end=1.0)
 * @param v_velocity   Target y-velocity component (at value_end=1.0)
 * @param t_start      Ramp start time
 * @param t_end        Ramp end time
 * @param value_start  Initial velocity multiplier (typically 0.0)
 * @param value_end    Final velocity multiplier (typically 1.0)
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_time_ramp(
    double u_velocity, double v_velocity,
    double t_start, double t_end,
    double value_start, double value_end);

/**
 * Create a step inlet (sudden change).
 *
 * Velocity changes instantaneously at t_step.
 * Before t_step: velocity = (u,v) * value_before
 * After t_step: velocity = (u,v) * value_after
 *
 * @param u_velocity    Base x-velocity component
 * @param v_velocity    Base y-velocity component
 * @param t_step        Time of step change
 * @param value_before  Velocity multiplier before step
 * @param value_after   Velocity multiplier after step
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_time_step(
    double u_velocity, double v_velocity,
    double t_step, double value_before, double value_after);

/**
 * Create an inlet with custom time+space callback.
 *
 * @param callback   Function called for each grid point with position and time
 * @param user_data  User-provided context pointer passed to callback
 * @return Configured inlet structure
 */
CFD_LIBRARY_EXPORT bc_inlet_config_t bc_inlet_config_time_custom(
    bc_inlet_profile_time_fn callback, void* user_data);

/**
 * Set sinusoidal time variation on an existing inlet configuration.
 *
 * @param config     Pointer to inlet configuration to modify
 * @param frequency  Oscillation frequency in Hz
 * @param amplitude  Oscillation amplitude
 * @param phase      Phase offset in radians
 * @param offset     DC offset
 */
CFD_LIBRARY_EXPORT void bc_inlet_set_time_sinusoidal(
    bc_inlet_config_t* config,
    double frequency, double amplitude, double phase, double offset);

/**
 * Set ramp time variation on an existing inlet configuration.
 *
 * @param config       Pointer to inlet configuration to modify
 * @param t_start      Ramp start time
 * @param t_end        Ramp end time
 * @param value_start  Initial multiplier
 * @param value_end    Final multiplier
 */
CFD_LIBRARY_EXPORT void bc_inlet_set_time_ramp(
    bc_inlet_config_t* config,
    double t_start, double t_end,
    double value_start, double value_end);

/**
 * Set step time variation on an existing inlet configuration.
 *
 * @param config        Pointer to inlet configuration to modify
 * @param t_step        Time of step change
 * @param value_before  Multiplier before step
 * @param value_after   Multiplier after step
 */
CFD_LIBRARY_EXPORT void bc_inlet_set_time_step(
    bc_inlet_config_t* config,
    double t_step, double value_before, double value_after);

/**
 * Set the boundary edge for an inlet configuration.
 *
 * @param config  Pointer to inlet configuration to modify
 * @param edge    Which boundary edge to apply inlet to
 */
CFD_LIBRARY_EXPORT void bc_inlet_set_edge(bc_inlet_config_t* config, bc_edge_t edge);

/**
 * Apply inlet velocity boundary condition to velocity fields.
 *
 * Applies the configured inlet velocity to the specified boundary edge.
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to inlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet(double* u, double* v, size_t nx, size_t ny,
                                                const bc_inlet_config_t* config);

/**
 * Apply inlet velocity boundary condition using scalar implementation.
 * Always available.
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to inlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_cpu(double* u, double* v, size_t nx, size_t ny,
                                                    const bc_inlet_config_t* config);

/**
 * Apply inlet velocity boundary condition using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to inlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_simd(double* u, double* v, size_t nx, size_t ny,
                                                         const bc_inlet_config_t* config);

/**
 * Apply inlet velocity boundary condition using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to inlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_omp(double* u, double* v, size_t nx, size_t ny,
                                                    const bc_inlet_config_t* config);

/* ============================================================================
 * Time-Varying Inlet Application API
 * ============================================================================ */

/**
 * Apply time-varying inlet velocity boundary condition.
 *
 * Applies the configured inlet velocity modulated by the time profile.
 * If no time variation is configured, delegates to bc_apply_inlet().
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param u         Pointer to x-velocity array (size nx*ny)
 * @param v         Pointer to y-velocity array (size nx*ny)
 * @param nx        Number of grid points in x-direction
 * @param ny        Number of grid points in y-direction
 * @param config    Pointer to inlet configuration
 * @param time_ctx  Pointer to time context with current time and dt
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_time(
    double* u, double* v, size_t nx, size_t ny,
    const bc_inlet_config_t* config,
    const bc_time_context_t* time_ctx);

/**
 * Apply time-varying inlet using scalar implementation.
 * Always available.
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_time_cpu(
    double* u, double* v, size_t nx, size_t ny,
    const bc_inlet_config_t* config,
    const bc_time_context_t* time_ctx);

/**
 * Apply time-varying inlet using SIMD implementation.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD not available.
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_time_simd(
    double* u, double* v, size_t nx, size_t ny,
    const bc_inlet_config_t* config,
    const bc_time_context_t* time_ctx);

/**
 * Apply time-varying inlet using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_inlet_time_omp(
    double* u, double* v, size_t nx, size_t ny,
    const bc_inlet_config_t* config,
    const bc_time_context_t* time_ctx);

/**
 * Helper macro to create a time context.
 */
#define BC_TIME_CONTEXT(t, delta_t) ((bc_time_context_t){.time = (t), .dt = (delta_t)})

/* ============================================================================
 * Outlet Boundary Conditions API
 *
 * Outlet BCs specify conditions at outflow boundaries.
 * Supports zero-gradient (Neumann) and convective outlet types.
 *
 * Zero-gradient: Sets boundary value equal to adjacent interior value.
 *   This is the simplest outlet BC, appropriate when flow is well-developed.
 *
 * Convective: Advects the field out of the domain using du/dt + U*du/dn = 0.
 *   This prevents wave reflections at the outlet, useful for unsteady flows.
 * ============================================================================ */

/**
 * Create a zero-gradient outlet configuration.
 *
 * Creates an outlet BC that copies the adjacent interior value to the boundary.
 * The outlet is configured for the right boundary by default.
 *
 * @return Configured outlet structure
 */
CFD_LIBRARY_EXPORT bc_outlet_config_t bc_outlet_config_zero_gradient(void);

/**
 * Create a convective outlet configuration.
 *
 * Creates an outlet BC that advects the field out of the domain.
 * Uses the equation: du/dt + U_adv * du/dn = 0
 * The outlet is configured for the right boundary by default.
 *
 * @param advection_velocity  Advection velocity (m/s), typically mean outflow velocity
 * @return Configured outlet structure
 */
CFD_LIBRARY_EXPORT bc_outlet_config_t bc_outlet_config_convective(double advection_velocity);

/**
 * Set the boundary edge for an outlet configuration.
 *
 * @param config  Pointer to outlet configuration to modify
 * @param edge    Which boundary edge to apply outlet to
 */
CFD_LIBRARY_EXPORT void bc_outlet_set_edge(bc_outlet_config_t* config, bc_edge_t edge);

/**
 * Apply outlet boundary condition to a scalar field.
 *
 * Applies the configured outlet BC to the specified boundary edge.
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param field   Pointer to scalar field array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_scalar(double* field, size_t nx, size_t ny,
                                                        const bc_outlet_config_t* config);

/**
 * Apply outlet boundary condition to velocity fields.
 *
 * Applies the configured outlet BC to both u and v velocity components.
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                                          const bc_outlet_config_t* config);

/**
 * Apply outlet boundary condition to a scalar field using scalar implementation.
 * Always available.
 *
 * @param field   Pointer to scalar field array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_scalar_cpu(double* field, size_t nx, size_t ny,
                                                            const bc_outlet_config_t* config);

/**
 * Apply outlet boundary condition using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 *
 * @param field   Pointer to scalar field array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_scalar_simd(double* field, size_t nx, size_t ny,
                                                                 const bc_outlet_config_t* config);

/**
 * Apply outlet boundary condition using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 *
 * @param field   Pointer to scalar field array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_scalar_omp(double* field, size_t nx, size_t ny,
                                                            const bc_outlet_config_t* config);

/**
 * Apply outlet velocity boundary condition using scalar implementation.
 * Always available.
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_velocity_cpu(double* u, double* v, size_t nx, size_t ny,
                                                              const bc_outlet_config_t* config);

/**
 * Apply outlet velocity boundary condition using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_velocity_simd(double* u, double* v, size_t nx, size_t ny,
                                                                   const bc_outlet_config_t* config);

/**
 * Apply outlet velocity boundary condition using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 *
 * @param u       Pointer to x-velocity array (size nx*ny)
 * @param v       Pointer to y-velocity array (size nx*ny)
 * @param nx      Number of grid points in x-direction
 * @param ny      Number of grid points in y-direction
 * @param config  Pointer to outlet configuration
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_outlet_velocity_omp(double* u, double* v, size_t nx, size_t ny,
                                                              const bc_outlet_config_t* config);

/**
 * Convenience macro for applying zero-gradient outlet BC to a scalar field.
 * Uses the right boundary by default.
 */
#define bc_apply_outlet(field, nx, ny) \
    bc_apply_outlet_scalar((field), (nx), (ny), \
        &(bc_outlet_config_t){.edge = BC_EDGE_RIGHT, .type = BC_OUTLET_ZERO_GRADIENT})

#ifdef __cplusplus
}
#endif

#endif  // CFD_BOUNDARY_CONDITIONS_H
