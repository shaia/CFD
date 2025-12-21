/**
 * Boundary Conditions - Public API and Dispatcher
 *
 * Technology-agnostic dispatcher for boundary conditions.
 * This file contains ONLY:
 * - Public API functions
 * - Backend selection and dispatch logic
 *
 * The dispatcher uses function pointer tables provided by each backend.
 * It has NO knowledge of SIMD, OpenMP, or other technologies.
 *
 * Backend implementations are in separate folders:
 * - cpu/boundary_conditions_scalar.c (baseline, single-threaded)
 * - omp/boundary_conditions_omp.c (OpenMP, multi-threaded scalar loops)
 * - simd_omp/boundary_conditions_simd_omp.c (OpenMP + SIMD: AVX2 on x86, NEON on ARM)
 */

#include "boundary_conditions_internal.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/logging.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/* ============================================================================
 * Error Handler State and Implementation
 * ============================================================================ */

static bc_error_handler_t g_error_handler = NULL;
static void* g_error_handler_user_data = NULL;

/**
 * Default error handler - prints to stderr.
 */
static void default_error_handler(bc_error_code_t error_code,
                                   const char* function,
                                   const char* message,
                                   void* user_data) {
    (void)user_data;  /* Unused in default handler */
    fprintf(stderr, "BC ERROR [%d] in %s: %s\n", (int)error_code, function, message);
}

void bc_set_error_handler(bc_error_handler_t handler, void* user_data) {
    g_error_handler = handler;
    g_error_handler_user_data = user_data;
}

bc_error_handler_t bc_get_error_handler(void) {
    return g_error_handler;
}

void bc_report_error(bc_error_code_t error_code, const char* function, const char* message) {
    if (g_error_handler != NULL) {
        g_error_handler(error_code, function, message, g_error_handler_user_data);
    } else {
        default_error_handler(error_code, function, message, NULL);
    }
}

/* ============================================================================
 * Backend State and Selection
 * ============================================================================ */

static bc_backend_t g_current_backend = BC_BACKEND_AUTO;

/**
 * Get the implementation table for a given backend.
 * Returns NULL if the backend is not available.
 */
static const bc_backend_impl_t* get_backend_impl(bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_SCALAR:
            return &bc_impl_scalar;
        case BC_BACKEND_OMP:
            return (bc_impl_omp.apply_neumann != NULL) ? &bc_impl_omp : NULL;
        case BC_BACKEND_SIMD_OMP:
            /* Use runtime check - bc_impl_simd_omp has non-NULL dispatchers,
             * but the underlying SIMD backend may not be available */
            return bc_simd_omp_backend_available() ? &bc_impl_simd_omp : NULL;
        case BC_BACKEND_CUDA:
            /* CUDA not yet implemented for boundary conditions */
            return NULL;
        case BC_BACKEND_AUTO:
        default:
            /* Auto: priority SIMD_OMP > OMP > Scalar (with runtime detection) */
            if (bc_simd_omp_backend_available()) {
                return &bc_impl_simd_omp;
            }
            if (bc_impl_omp.apply_neumann != NULL) {
                return &bc_impl_omp;
            }
            return &bc_impl_scalar;
    }
}

bool bc_backend_available(bc_backend_t backend) {
    return get_backend_impl(backend) != NULL;
}

bc_backend_t bc_get_backend(void) {
    return g_current_backend;
}

const char* bc_get_backend_name(void) {
    const bc_backend_impl_t* impl = get_backend_impl(g_current_backend);

    if (impl == &bc_impl_simd_omp) {
        /* Report which SIMD variant is active based on runtime detection.
         * Use enum comparison for robustness instead of string comparison. */
        cfd_simd_arch_t arch = cfd_detect_simd_arch();
        if (g_current_backend == BC_BACKEND_AUTO) {
            /* Auto mode selected SIMD+OMP */
            switch (arch) {
                case CFD_SIMD_AVX2:
                    return "auto (simd_omp/avx2)";
                case CFD_SIMD_NEON:
                    return "auto (simd_omp/neon)";
                default:
                    /* Should not happen: impl == bc_impl_simd_omp implies SIMD available */
                    return "auto (simd_omp)";
            }
        } else {
            /* Explicit SIMD+OMP selection */
            switch (arch) {
                case CFD_SIMD_AVX2:
                    return "simd_omp (avx2)";
                case CFD_SIMD_NEON:
                    return "simd_omp (neon)";
                default:
                    /* Should not happen: impl == bc_impl_simd_omp implies SIMD available */
                    return "simd_omp";
            }
        }
    }
    if (impl == &bc_impl_omp) {
        return (g_current_backend == BC_BACKEND_AUTO) ? "auto (omp)" : "omp";
    }
    return (g_current_backend == BC_BACKEND_AUTO) ? "auto (scalar)" : "scalar";
}

bool bc_set_backend(bc_backend_t backend) {
    if (!bc_backend_available(backend) && backend != BC_BACKEND_AUTO) {
        g_current_backend = backend;
        return false;
    }
    g_current_backend = backend;
    return true;
}

/* ============================================================================
 * Internal Dispatch Helpers
 * ============================================================================ */

static cfd_status_t apply_neumann_with_backend(double* field, size_t nx, size_t ny,
                                                const bc_backend_impl_t* impl) {
    if (impl == NULL || impl->apply_neumann == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    impl->apply_neumann(field, nx, ny);
    return CFD_SUCCESS;
}

static cfd_status_t apply_periodic_with_backend(double* field, size_t nx, size_t ny,
                                                 const bc_backend_impl_t* impl) {
    if (impl == NULL || impl->apply_periodic == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    impl->apply_periodic(field, nx, ny);
    return CFD_SUCCESS;
}

static cfd_status_t apply_dirichlet_with_backend(double* field, size_t nx, size_t ny,
                                                  const bc_dirichlet_values_t* values,
                                                  const bc_backend_impl_t* impl) {
    if (impl == NULL || impl->apply_dirichlet == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    impl->apply_dirichlet(field, nx, ny, values);
    return CFD_SUCCESS;
}

static cfd_status_t apply_scalar_field_bc(double* field, size_t nx, size_t ny,
                                           bc_type_t type, const bc_backend_impl_t* impl) {
    if (!field || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    switch (type) {
        case BC_TYPE_NEUMANN:
            return apply_neumann_with_backend(field, nx, ny, impl);

        case BC_TYPE_PERIODIC:
            return apply_periodic_with_backend(field, nx, ny, impl);

        case BC_TYPE_DIRICHLET:
            cfd_warning("BC_TYPE_DIRICHLET requires bc_apply_dirichlet_*() functions with values");
            return CFD_ERROR_INVALID;

        case BC_TYPE_NOSLIP:
            /* No-slip is Dirichlet with all zeros - use dedicated bc_apply_noslip() */
            cfd_warning("BC_TYPE_NOSLIP requires bc_apply_noslip() for velocity fields");
            return CFD_ERROR_INVALID;

        case BC_TYPE_INLET:
            cfd_warning("BC_TYPE_INLET not implemented");
            return CFD_ERROR_UNSUPPORTED;

        case BC_TYPE_OUTLET:
            cfd_warning("BC_TYPE_OUTLET not implemented");
            return CFD_ERROR_UNSUPPORTED;

        default:
            cfd_warning("Unknown BC type requested");
            return CFD_ERROR_INVALID;
    }
}

/* ============================================================================
 * Public API - Global Backend
 * ============================================================================ */

cfd_status_t bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type) {
    const bc_backend_impl_t* impl = get_backend_impl(g_current_backend);
    return apply_scalar_field_bc(field, nx, ny, type, impl);
}

cfd_status_t bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(g_current_backend);
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, impl);
}

/* ============================================================================
 * Public API - Explicit Backend Selection
 * ============================================================================ */

cfd_status_t bc_apply_scalar_cpu(double* field, size_t nx, size_t ny, bc_type_t type) {
    return apply_scalar_field_bc(field, nx, ny, type, &bc_impl_scalar);
}

cfd_status_t bc_apply_scalar_simd_omp(double* field, size_t nx, size_t ny, bc_type_t type) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_scalar_field_bc(field, nx, ny, type, impl);
}

cfd_status_t bc_apply_scalar_omp(double* field, size_t nx, size_t ny, bc_type_t type) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_scalar_field_bc(field, nx, ny, type, impl);
}

cfd_status_t bc_apply_velocity_cpu(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, &bc_impl_scalar);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, &bc_impl_scalar);
}

cfd_status_t bc_apply_velocity_simd_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, impl);
}

cfd_status_t bc_apply_velocity_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, impl);
}

/* ============================================================================
 * Public API - Dirichlet Boundary Conditions
 * ============================================================================ */

cfd_status_t bc_apply_dirichlet_scalar(double* field, size_t nx, size_t ny,
                                        const bc_dirichlet_values_t* values) {
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(g_current_backend);
    return apply_dirichlet_with_backend(field, nx, ny, values, impl);
}

cfd_status_t bc_apply_dirichlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                          const bc_dirichlet_values_t* u_values,
                                          const bc_dirichlet_values_t* v_values) {
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(g_current_backend);
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, impl);
}

/* Backend-specific Dirichlet implementations */

cfd_status_t bc_apply_dirichlet_scalar_cpu(double* field, size_t nx, size_t ny,
                                            const bc_dirichlet_values_t* values) {
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, &bc_impl_scalar);
}

cfd_status_t bc_apply_dirichlet_scalar_simd_omp(double* field, size_t nx, size_t ny,
                                                 const bc_dirichlet_values_t* values) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, impl);
}

cfd_status_t bc_apply_dirichlet_scalar_omp(double* field, size_t nx, size_t ny,
                                            const bc_dirichlet_values_t* values) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, impl);
}

cfd_status_t bc_apply_dirichlet_velocity_cpu(double* u, double* v, size_t nx, size_t ny,
                                              const bc_dirichlet_values_t* u_values,
                                              const bc_dirichlet_values_t* v_values) {
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, &bc_impl_scalar);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, &bc_impl_scalar);
}

cfd_status_t bc_apply_dirichlet_velocity_simd_omp(double* u, double* v, size_t nx, size_t ny,
                                                   const bc_dirichlet_values_t* u_values,
                                                   const bc_dirichlet_values_t* v_values) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, impl);
}

cfd_status_t bc_apply_dirichlet_velocity_omp(double* u, double* v, size_t nx, size_t ny,
                                              const bc_dirichlet_values_t* u_values,
                                              const bc_dirichlet_values_t* v_values) {
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, impl);
}

/* ============================================================================
 * Public API - No-Slip Wall Boundary Conditions
 *
 * No-slip conditions set velocity to zero at all boundaries.
 * Implemented using Dirichlet BCs with zero values for efficiency.
 * ============================================================================ */

/** Static zero-valued Dirichlet BC for no-slip walls */
static const bc_dirichlet_values_t g_noslip_zero = {
    .left = 0.0,
    .right = 0.0,
    .top = 0.0,
    .bottom = 0.0
};

/** Helper: apply no-slip to both velocity components using specified backend */
static cfd_status_t apply_noslip_with_backend(double* u, double* v, size_t nx, size_t ny,
                                               const bc_backend_impl_t* impl) {
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, &g_noslip_zero, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, &g_noslip_zero, impl);
}

cfd_status_t bc_apply_noslip(double* u, double* v, size_t nx, size_t ny) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_noslip_with_backend(u, v, nx, ny, get_backend_impl(g_current_backend));
}

cfd_status_t bc_apply_noslip_cpu(double* u, double* v, size_t nx, size_t ny) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_noslip_with_backend(u, v, nx, ny, &bc_impl_scalar);
}

cfd_status_t bc_apply_noslip_simd_omp(double* u, double* v, size_t nx, size_t ny) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_noslip_with_backend(u, v, nx, ny, impl);
}

cfd_status_t bc_apply_noslip_omp(double* u, double* v, size_t nx, size_t ny) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_noslip_with_backend(u, v, nx, ny, impl);
}

/* ============================================================================
 * Public API - Inlet Velocity Boundary Conditions
 *
 * Inlet conditions specify velocity at inflow boundaries.
 * ============================================================================ */

/** Helper: apply inlet BC with specified backend */
static cfd_status_t apply_inlet_with_backend(double* u, double* v, size_t nx, size_t ny,
                                              const bc_inlet_config_t* config,
                                              const bc_backend_impl_t* impl) {
    if (impl == NULL || impl->apply_inlet == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return impl->apply_inlet(u, v, nx, ny, config);
}

/* Inlet configuration factory functions */

bc_inlet_config_t bc_inlet_config_uniform(double u_velocity, double v_velocity) {
    bc_inlet_config_t config = {0};
    config.edge = BC_EDGE_LEFT;  /* Default to left boundary */
    config.profile = BC_INLET_PROFILE_UNIFORM;
    config.spec_type = BC_INLET_SPEC_VELOCITY;
    config.spec.velocity.u = u_velocity;
    config.spec.velocity.v = v_velocity;
    config.custom_profile = NULL;
    config.custom_profile_user_data = NULL;
    return config;
}

bc_inlet_config_t bc_inlet_config_parabolic(double max_velocity) {
    bc_inlet_config_t config = {0};
    config.edge = BC_EDGE_LEFT;  /* Default to left boundary */
    config.profile = BC_INLET_PROFILE_PARABOLIC;
    config.spec_type = BC_INLET_SPEC_VELOCITY;
    /* For left boundary, parabolic flow is in +x direction */
    config.spec.velocity.u = max_velocity;
    config.spec.velocity.v = 0.0;
    config.custom_profile = NULL;
    config.custom_profile_user_data = NULL;
    return config;
}

bc_inlet_config_t bc_inlet_config_magnitude_dir(double magnitude, double direction) {
    bc_inlet_config_t config = {0};
    config.edge = BC_EDGE_LEFT;  /* Default to left boundary */
    config.profile = BC_INLET_PROFILE_UNIFORM;
    config.spec_type = BC_INLET_SPEC_MAGNITUDE_DIR;
    config.spec.magnitude_dir.magnitude = magnitude;
    config.spec.magnitude_dir.direction = direction;
    config.custom_profile = NULL;
    config.custom_profile_user_data = NULL;
    return config;
}

bc_inlet_config_t bc_inlet_config_mass_flow(double mass_flow_rate, double density, double inlet_length) {
    bc_inlet_config_t config = {0};
    config.edge = BC_EDGE_LEFT;  /* Default to left boundary */
    config.profile = BC_INLET_PROFILE_UNIFORM;
    config.spec_type = BC_INLET_SPEC_MASS_FLOW;
    config.spec.mass_flow.mass_flow_rate = mass_flow_rate;
    config.spec.mass_flow.density = density;
    config.spec.mass_flow.inlet_length = inlet_length;
    config.custom_profile = NULL;
    config.custom_profile_user_data = NULL;
    return config;
}

bc_inlet_config_t bc_inlet_config_custom(bc_inlet_profile_fn callback, void* user_data) {
    bc_inlet_config_t config = {0};
    config.edge = BC_EDGE_LEFT;  /* Default to left boundary */
    config.profile = BC_INLET_PROFILE_CUSTOM;
    config.spec_type = BC_INLET_SPEC_VELOCITY;
    config.spec.velocity.u = 0.0;
    config.spec.velocity.v = 0.0;
    config.custom_profile = callback;
    config.custom_profile_user_data = user_data;
    return config;
}

void bc_inlet_set_edge(bc_inlet_config_t* config, bc_edge_t edge) {
    if (config != NULL) {
        config->edge = edge;
    }
}

/* Inlet BC application functions */

cfd_status_t bc_apply_inlet(double* u, double* v, size_t nx, size_t ny,
                             const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_inlet_with_backend(u, v, nx, ny, config, get_backend_impl(g_current_backend));
}

cfd_status_t bc_apply_inlet_cpu(double* u, double* v, size_t nx, size_t ny,
                                 const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_inlet_with_backend(u, v, nx, ny, config, &bc_impl_scalar);
}

cfd_status_t bc_apply_inlet_simd_omp(double* u, double* v, size_t nx, size_t ny,
                                      const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_inlet_with_backend(u, v, nx, ny, config, impl);
}

cfd_status_t bc_apply_inlet_omp(double* u, double* v, size_t nx, size_t ny,
                                 const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_inlet_with_backend(u, v, nx, ny, config, impl);
}

/* ============================================================================
 * Public API - Outlet Boundary Conditions
 *
 * Outlet conditions specify conditions at outflow boundaries.
 * Supports zero-gradient (Neumann) and convective outlet types.
 * ============================================================================ */

/** Helper: apply outlet BC with specified backend */
static cfd_status_t apply_outlet_with_backend(double* field, size_t nx, size_t ny,
                                               const bc_outlet_config_t* config,
                                               const bc_backend_impl_t* impl) {
    if (impl == NULL || impl->apply_outlet == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return impl->apply_outlet(field, nx, ny, config);
}

/* Outlet configuration factory functions */

bc_outlet_config_t bc_outlet_config_zero_gradient(void) {
    bc_outlet_config_t config = {0};
    config.edge = BC_EDGE_RIGHT;  /* Default to right boundary */
    config.type = BC_OUTLET_ZERO_GRADIENT;
    config.advection_velocity = 0.0;
    return config;
}

bc_outlet_config_t bc_outlet_config_convective(double advection_velocity) {
    bc_outlet_config_t config = {0};
    config.edge = BC_EDGE_RIGHT;  /* Default to right boundary */
    config.type = BC_OUTLET_CONVECTIVE;
    config.advection_velocity = advection_velocity;
    return config;
}

void bc_outlet_set_edge(bc_outlet_config_t* config, bc_edge_t edge) {
    if (config != NULL) {
        config->edge = edge;
    }
}

/* Outlet BC application functions */

cfd_status_t bc_apply_outlet_scalar(double* field, size_t nx, size_t ny,
                                     const bc_outlet_config_t* config) {
    if (!field || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_outlet_with_backend(field, nx, ny, config, get_backend_impl(g_current_backend));
}

cfd_status_t bc_apply_outlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                       const bc_outlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_outlet_with_backend(u, nx, ny, config, get_backend_impl(g_current_backend));
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_outlet_with_backend(v, nx, ny, config, get_backend_impl(g_current_backend));
}

cfd_status_t bc_apply_outlet_scalar_cpu(double* field, size_t nx, size_t ny,
                                         const bc_outlet_config_t* config) {
    if (!field || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_outlet_with_backend(field, nx, ny, config, &bc_impl_scalar);
}

cfd_status_t bc_apply_outlet_scalar_simd_omp(double* field, size_t nx, size_t ny,
                                              const bc_outlet_config_t* config) {
    if (!field || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_outlet_with_backend(field, nx, ny, config, impl);
}

cfd_status_t bc_apply_outlet_scalar_omp(double* field, size_t nx, size_t ny,
                                         const bc_outlet_config_t* config) {
    if (!field || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    return apply_outlet_with_backend(field, nx, ny, config, impl);
}

cfd_status_t bc_apply_outlet_velocity_cpu(double* u, double* v, size_t nx, size_t ny,
                                           const bc_outlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_outlet_with_backend(u, nx, ny, config, &bc_impl_scalar);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_outlet_with_backend(v, nx, ny, config, &bc_impl_scalar);
}

cfd_status_t bc_apply_outlet_velocity_simd_omp(double* u, double* v, size_t nx, size_t ny,
                                                const bc_outlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_SIMD_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    cfd_status_t status = apply_outlet_with_backend(u, nx, ny, config, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_outlet_with_backend(v, nx, ny, config, impl);
}

cfd_status_t bc_apply_outlet_velocity_omp(double* u, double* v, size_t nx, size_t ny,
                                           const bc_outlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    const bc_backend_impl_t* impl = get_backend_impl(BC_BACKEND_OMP);
    if (impl == NULL) {
        return CFD_ERROR_UNSUPPORTED;
    }
    cfd_status_t status = apply_outlet_with_backend(u, nx, ny, config, impl);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_outlet_with_backend(v, nx, ny, config, impl);
}
