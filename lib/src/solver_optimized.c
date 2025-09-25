// Enable C11 features for aligned_alloc
#define _POSIX_C_SOURCE 200112L
#define _ISOC11_SOURCE

#include "solver.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "vtk_output.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical stability limits for numerical computation (same as basic solver)
#define MAX_DERIVATIVE_LIMIT 100.0     // Maximum allowed first derivative magnitude (1/s)
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0  // Maximum allowed second derivative magnitude (1/s²)
#define MAX_VELOCITY_LIMIT 100.0       // Maximum allowed velocity magnitude (m/s)
#define MAX_DIVERGENCE_LIMIT 10.0      // Maximum allowed velocity divergence (1/s)

// Use centralized aligned memory allocation from utils

// Block size for cache-friendly memory access
#define BLOCK_SIZE 32

// Optimized version of the solver using SIMD and cache-friendly memory access
void solve_navier_stokes_optimized(FlowField* field, const Grid* grid, const SolverParams* params) {
    // Validate input parameters
    if (!field || !grid || !params) {
        return;
    }

    // Check for minimum grid size - prevent crashes on small grids
    if (field->nx < 3 || field->ny < 3) {
        return; // Skip solver for grids too small for finite differences
    }

    // Pre-compute memory size for better readability and maintainability
    const size_t field_size_bytes = field->nx * field->ny * sizeof(double);

    // Allocate temporary arrays with aligned memory for SIMD
    double* u_new = (double*)cfd_aligned_malloc(field_size_bytes);
    double* v_new = (double*)cfd_aligned_malloc(field_size_bytes);
    double* p_new = (double*)cfd_aligned_malloc(field_size_bytes);
    double* rho_new = (double*)cfd_aligned_malloc(field_size_bytes);
    double* T_new = (double*)cfd_aligned_malloc(field_size_bytes);

    // Check if memory allocation succeeded
    if (!u_new || !v_new || !p_new || !rho_new || !T_new) {
        // Clean up any allocated memory
        if (u_new) cfd_aligned_free(u_new);
        if (v_new) cfd_aligned_free(v_new);
        if (p_new) cfd_aligned_free(p_new);
        if (rho_new) cfd_aligned_free(rho_new);
        if (T_new) cfd_aligned_free(T_new);
        return;
    }

    // Pre-compute grid spacing inverses
    double* dx_inv = (double*)cfd_aligned_malloc(field->nx * sizeof(double));
    double* dy_inv = (double*)cfd_aligned_malloc(field->ny * sizeof(double));

    // Check if grid allocation succeeded
    if (!dx_inv || !dy_inv) {
        // Clean up all allocated memory
        cfd_aligned_free(u_new);
        cfd_aligned_free(v_new);
        cfd_aligned_free(p_new);
        cfd_aligned_free(rho_new);
        cfd_aligned_free(T_new);
        if (dx_inv) cfd_aligned_free(dx_inv);
        if (dy_inv) cfd_aligned_free(dy_inv);
        return;
    }
    
    for (size_t i = 0; i < field->nx; i++) {
        dx_inv[i] = (i < field->nx - 1) ? 1.0 / (2.0 * grid->dx[i]) : 0.0;
    }
    for (size_t j = 0; j < field->ny; j++) {
        dy_inv[j] = (j < field->ny - 1) ? 1.0 / (2.0 * grid->dy[j]) : 0.0;
    }

    // Initialize temporary arrays with current values to prevent uninitialized memory
    memcpy(u_new, field->u, field_size_bytes);
    memcpy(v_new, field->v, field_size_bytes);
    memcpy(p_new, field->p, field_size_bytes);
    memcpy(rho_new, field->rho, field_size_bytes);
    memcpy(T_new, field->T, field_size_bytes);

    // Use conservative time step to prevent instabilities
    double conservative_dt = fmin(params->dt, 0.0001);

    // Main time-stepping loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Use conservative dt instead of dynamically computed one
        
        // Update solution using block-based computation
        for (size_t j_block = 1; j_block < field->ny - 1; j_block += BLOCK_SIZE) {
            size_t j_end = (j_block + BLOCK_SIZE < field->ny - 1) ? j_block + BLOCK_SIZE : field->ny - 1;

            for (size_t i_block = 1; i_block < field->nx - 1; i_block += BLOCK_SIZE) {
                size_t i_end = (i_block + BLOCK_SIZE < field->nx - 1) ? i_block + BLOCK_SIZE : field->nx - 1;
                
                // Process each block
                for (size_t j = j_block; j < j_end; j++) {
                    for (size_t i = i_block; i < i_end; i++) {
                        size_t idx = j * field->nx + i;
                        
                        // Compute spatial derivatives using pre-computed inverses
                        double du_dx = (field->u[idx + 1] - field->u[idx - 1]) * dx_inv[i];
                        double du_dy = (field->u[idx + field->nx] - field->u[idx - field->nx]) * dy_inv[j];
                        double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) * dx_inv[i];
                        double dv_dy = (field->v[idx + field->nx] - field->v[idx - field->nx]) * dy_inv[j];

                        // Pressure gradients
                        double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) * dx_inv[i];
                        double dp_dy = (field->p[idx + field->nx] - field->p[idx - field->nx]) * dy_inv[j];

                        // Second derivatives for viscous terms
                        double dx2 = grid->dx[i] * grid->dx[i];
                        double dy2 = grid->dy[j] * grid->dy[j];
                        double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) / dx2;
                        double d2u_dy2 = (field->u[idx + field->nx] - 2.0 * field->u[idx] + field->u[idx - field->nx]) / dy2;
                        double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) / dx2;
                        double d2v_dy2 = (field->v[idx + field->nx] - 2.0 * field->v[idx] + field->v[idx - field->nx]) / dy2;

                        // Viscosity coefficient (kinematic viscosity = dynamic viscosity / density) with safety
                        double nu = params->mu / fmax(field->rho[idx], 1e-10);
                        nu = fmin(nu, 1.0);  // Limit maximum viscosity

                        // Limit derivatives to prevent instabilities (same as basic solver)
                        du_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dx));
                        du_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dy));
                        dv_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dx));
                        dv_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dy));
                        dp_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dx));
                        dp_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dy));
                        d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
                        d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
                        d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
                        d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));

                        // Source terms to maintain flow (prevents decay)
                        double x = grid->x[i];
                        double y = grid->y[j];
                        double source_u, source_v;
                        compute_source_terms(x, y, iter, conservative_dt, params, &source_u, &source_v);

                        // Conservative velocity updates with limited changes
                        double du = conservative_dt * (
                            -field->u[idx] * du_dx - field->v[idx] * du_dy  // Convection
                            - dp_dx / fmax(field->rho[idx], 1e-10)          // Pressure gradient (safe division)
                            + nu * (d2u_dx2 + d2u_dy2)                      // Viscous diffusion
                            + source_u                                       // Source term
                        );

                        double dv = conservative_dt * (
                            -field->u[idx] * dv_dx - field->v[idx] * dv_dy  // Convection
                            - dp_dy / fmax(field->rho[idx], 1e-10)          // Pressure gradient (safe division)
                            + nu * (d2v_dx2 + d2v_dy2)                      // Viscous diffusion
                            + source_v                                       // Source term
                        );

                        // Limit velocity changes
                        du = fmax(-1.0, fmin(1.0, du));
                        dv = fmax(-1.0, fmin(1.0, dv));

                        u_new[idx] = field->u[idx] + du;
                        v_new[idx] = field->v[idx] + dv;

                        // Limit velocity magnitudes
                        u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, u_new[idx]));
                        v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, v_new[idx]));

                        // Simplified stable pressure update
                        double divergence = du_dx + dv_dy;
                        divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));

                        double dp = -0.1 * conservative_dt * field->rho[idx] * divergence;
                        dp = fmax(-1.0, fmin(1.0, dp));  // Limit pressure changes
                        p_new[idx] = field->p[idx] + dp;

                        // Keep density and temperature constant for this simplified model
                        rho_new[idx] = field->rho[idx];
                        T_new[idx] = field->T[idx];
                    }
                }
            }
        }
        
        // Copy new solution to old solution using SIMD if available
#ifdef __AVX2__
        size_t size = field->nx * field->ny;
        size_t vec_size = size / 4 * 4;  // Process 4 doubles at a time

        // Use aligned SIMD operations for maximum performance
        // Both temporary arrays and FlowField arrays are now 32-byte aligned
        for (size_t i = 0; i < vec_size; i += 4) {
            __m256d u_vec = _mm256_load_pd(&u_new[i]);
            __m256d v_vec = _mm256_load_pd(&v_new[i]);
            __m256d p_vec = _mm256_load_pd(&p_new[i]);
            __m256d rho_vec = _mm256_load_pd(&rho_new[i]);
            __m256d T_vec = _mm256_load_pd(&T_new[i]);

            _mm256_store_pd(&field->u[i], u_vec);
            _mm256_store_pd(&field->v[i], v_vec);
            _mm256_store_pd(&field->p[i], p_vec);
            _mm256_store_pd(&field->rho[i], rho_vec);
            _mm256_store_pd(&field->T[i], T_vec);
        }

        // Handle remaining elements
        for (size_t i = vec_size; i < size; i++) {
            field->u[i] = u_new[i];
            field->v[i] = v_new[i];
            field->p[i] = p_new[i];
            field->rho[i] = rho_new[i];
            field->T[i] = T_new[i];
        }
#else
        memcpy(field->u, u_new, field_size_bytes);
        memcpy(field->v, v_new, field_size_bytes);
        memcpy(field->p, p_new, field_size_bytes);
        memcpy(field->rho, rho_new, field_size_bytes);
        memcpy(field->T, T_new, field_size_bytes);
#endif
        
        // Apply boundary conditions
        apply_boundary_conditions(field, grid);

        // Check for NaN/Inf values and stop if found
        int has_nan = 0;
        for (size_t k = 0; k < field->nx * field->ny; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
                has_nan = 1;
                break;
            }
        }

        if (has_nan) {
            printf("Warning: NaN/Inf detected in optimized solver iteration %d, stopping solver\n", iter);
            break;
        }

        // Output solution every 100 iterations
        if (iter % 100 == 0) {
            char artifacts_path[256];
            char output_path[256];
            char filename[256];

            // Create cross-platform paths
            make_artifacts_path(artifacts_path, sizeof(artifacts_path), "");
            make_artifacts_path(output_path, sizeof(output_path), "output");

            ensure_directory_exists(artifacts_path);
            ensure_directory_exists(output_path);

            // Create output filename with proper path separator
            char base_filename[128];
            snprintf(base_filename, sizeof(base_filename), "output_optimized_%d.vtk", iter);
            make_output_path(filename, sizeof(filename), base_filename);

            write_vtk_output(filename, "pressure", field->p, field->nx, field->ny,
                           grid->xmin, grid->xmax, grid->ymin, grid->ymax);
        }
    }
    
    // Free temporary arrays
    cfd_aligned_free(u_new);
    cfd_aligned_free(v_new);
    cfd_aligned_free(p_new);
    cfd_aligned_free(rho_new);
    cfd_aligned_free(T_new);
    cfd_aligned_free(dx_inv);
    cfd_aligned_free(dy_inv);
}