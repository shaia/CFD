#include "solver.h"
#include "utils.h"
#include <math.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Block size for cache-friendly memory access
#define BLOCK_SIZE 32

// Optimized version of the solver using SIMD and cache-friendly memory access
void solve_navier_stokes_optimized(FlowField* field, const Grid* grid, const SolverParams* params) {
    // Allocate temporary arrays with aligned memory for SIMD
    double* u_new = (double*)aligned_alloc(32, field->nx * field->ny * sizeof(double));
    double* v_new = (double*)aligned_alloc(32, field->nx * field->ny * sizeof(double));
    double* p_new = (double*)aligned_alloc(32, field->nx * field->ny * sizeof(double));
    double* rho_new = (double*)aligned_alloc(32, field->nx * field->ny * sizeof(double));
    double* T_new = (double*)aligned_alloc(32, field->nx * field->ny * sizeof(double));
    
    // Pre-compute grid spacing inverses
    double* dx_inv = (double*)aligned_alloc(32, (field->nx - 1) * sizeof(double));
    double* dy_inv = (double*)aligned_alloc(32, (field->ny - 1) * sizeof(double));
    
    for (size_t i = 0; i < field->nx - 1; i++) {
        dx_inv[i] = 1.0 / (2.0 * grid->dx[i]);
    }
    for (size_t j = 0; j < field->ny - 1; j++) {
        dy_inv[j] = 1.0 / (2.0 * grid->dy[j]);
    }
    
    // Main time-stepping loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Compute time step
        SolverParams params_copy = *params;
        compute_time_step(field, grid, &params_copy);
        
        // Update solution using block-based computation
        for (size_t j_block = 1; j_block < field->ny - 1; j_block += BLOCK_SIZE) {
            size_t j_end = min_double(j_block + BLOCK_SIZE, field->ny - 1);
            
            for (size_t i_block = 1; i_block < field->nx - 1; i_block += BLOCK_SIZE) {
                size_t i_end = min_double(i_block + BLOCK_SIZE, field->nx - 1);
                
                // Process each block
                for (size_t j = j_block; j < j_end; j++) {
                    for (size_t i = i_block; i < i_end; i++) {
                        size_t idx = j * field->nx + i;
                        
                        // Compute spatial derivatives using pre-computed inverses
                        double du_dx = (field->u[idx + 1] - field->u[idx - 1]) * dx_inv[i];
                        double du_dy = (field->u[idx + field->nx] - field->u[idx - field->nx]) * dy_inv[j];
                        double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) * dx_inv[i];
                        double dv_dy = (field->v[idx + field->nx] - field->v[idx - field->nx]) * dy_inv[j];
                        
                        // Update velocity components
                        u_new[idx] = field->u[idx] - params_copy.dt * 
                                   (field->u[idx] * du_dx + field->v[idx] * du_dy);
                        v_new[idx] = field->v[idx] - params_copy.dt * 
                                   (field->u[idx] * dv_dx + field->v[idx] * dv_dy);
                        
                        // Update pressure and density (simplified)
                        p_new[idx] = field->p[idx];
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
        memcpy(field->u, u_new, field->nx * field->ny * sizeof(double));
        memcpy(field->v, v_new, field->nx * field->ny * sizeof(double));
        memcpy(field->p, p_new, field->nx * field->ny * sizeof(double));
        memcpy(field->rho, rho_new, field->nx * field->ny * sizeof(double));
        memcpy(field->T, T_new, field->nx * field->ny * sizeof(double));
#endif
        
        // Apply boundary conditions
        apply_boundary_conditions(field, grid);
        
        // Output solution every 100 iterations
        if (iter % 100 == 0) {
            char filename[256];
            sprintf(filename, "output_optimized_%d.vtk", iter);
            write_vtk_output(filename, "pressure", field->p, field->nx, field->ny,
                           grid->xmin, grid->xmax, grid->ymin, grid->ymax);
        }
    }
    
    // Free temporary arrays
    free(u_new);
    free(v_new);
    free(p_new);
    free(rho_new);
    free(T_new);
    free(dx_inv);
    free(dy_inv);
} 