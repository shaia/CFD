#include "solver.h"
#include "utils.h"
#include <math.h>
#include <string.h>

FlowField* flow_field_create(size_t nx, size_t ny) {
    FlowField* field = (FlowField*)cfd_malloc(sizeof(FlowField));
    
    field->nx = nx;
    field->ny = ny;
    
    // Allocate memory for flow variables
    field->u = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->v = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->p = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->rho = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->T = (double*)cfd_calloc(nx * ny, sizeof(double));
    
    return field;
}

void flow_field_destroy(FlowField* field) {
    if (field != NULL) {
        cfd_free(field->u);
        cfd_free(field->v);
        cfd_free(field->p);
        cfd_free(field->rho);
        cfd_free(field->T);
        cfd_free(field);
    }
}

void initialize_flow_field(FlowField* field, const Grid* grid) {
    // Initialize with a simple test case: uniform flow with a small perturbation
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            
            // Set initial conditions
            field->u[idx] = 1.0;  // Uniform flow in x-direction
            field->v[idx] = 0.0;  // No flow in y-direction
            field->p[idx] = 1.0;  // Reference pressure
            field->rho[idx] = 1.0;  // Reference density
            field->T[idx] = 300.0;  // Reference temperature (K)
            
            // Add a small perturbation in the center
            double x = grid->x[i];
            double y = grid->y[j];
            double r = sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
            if (r < 0.1) {
                field->p[idx] += 0.1 * exp(-r * r / 0.01);
            }
        }
    }
}

void compute_time_step(FlowField* field, const Grid* grid, SolverParams* params) {
    double max_speed = 0.0;
    double dx_min = grid->dx[0];
    double dy_min = grid->dy[0];
    
    // Find minimum grid spacing
    for (size_t i = 0; i < grid->nx - 1; i++) {
        dx_min = min_double(dx_min, grid->dx[i]);
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        dy_min = min_double(dy_min, grid->dy[j]);
    }
    
    // Find maximum wave speed
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double speed = sqrt(field->u[idx] * field->u[idx] + 
                              field->v[idx] * field->v[idx]) +
                         sqrt(params->gamma * field->p[idx] / field->rho[idx]);
            max_speed = max_double(max_speed, speed);
        }
    }
    
    // Compute time step based on CFL condition
    params->dt = params->cfl * min_double(dx_min, dy_min) / max_speed;
}

void apply_boundary_conditions(FlowField* field, const Grid* grid) {
    // Apply periodic boundary conditions in x-direction
    for (size_t j = 0; j < field->ny; j++) {
        field->u[j * field->nx + 0] = field->u[j * field->nx + field->nx - 2];
        field->v[j * field->nx + 0] = field->v[j * field->nx + field->nx - 2];
        field->p[j * field->nx + 0] = field->p[j * field->nx + field->nx - 2];
        field->rho[j * field->nx + 0] = field->rho[j * field->nx + field->nx - 2];
        field->T[j * field->nx + 0] = field->T[j * field->nx + field->nx - 2];
        
        field->u[j * field->nx + field->nx - 1] = field->u[j * field->nx + 1];
        field->v[j * field->nx + field->nx - 1] = field->v[j * field->nx + 1];
        field->p[j * field->nx + field->nx - 1] = field->p[j * field->nx + 1];
        field->rho[j * field->nx + field->nx - 1] = field->rho[j * field->nx + 1];
        field->T[j * field->nx + field->nx - 1] = field->T[j * field->nx + 1];
    }
    
    // Apply wall boundary conditions in y-direction
    for (size_t i = 0; i < field->nx; i++) {
        // Bottom wall
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        
        // Top wall
        size_t top_idx = (field->ny - 1) * field->nx + i;
        field->u[top_idx] = 0.0;
        field->v[top_idx] = 0.0;
    }
}

void solve_navier_stokes(FlowField* field, const Grid* grid, const SolverParams* params) {
    // Allocate temporary arrays for the solution update
    double* u_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* v_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* p_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* rho_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* T_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    
    // Main time-stepping loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Compute time step
        SolverParams params_copy = *params;
        compute_time_step(field, grid, &params_copy);
        
        // Update solution using explicit Euler method
        for (size_t j = 1; j < field->ny - 1; j++) {
            for (size_t i = 1; i < field->nx - 1; i++) {
                size_t idx = j * field->nx + i;
                
                // Compute spatial derivatives
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / 
                             (2.0 * grid->dx[i]);
                double du_dy = (field->u[idx + field->nx] - field->u[idx - field->nx]) / 
                             (2.0 * grid->dy[j]);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / 
                             (2.0 * grid->dx[i]);
                double dv_dy = (field->v[idx + field->nx] - field->v[idx - field->nx]) / 
                             (2.0 * grid->dy[j]);
                
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
        
        // Copy new solution to old solution
        memcpy(field->u, u_new, field->nx * field->ny * sizeof(double));
        memcpy(field->v, v_new, field->nx * field->ny * sizeof(double));
        memcpy(field->p, p_new, field->nx * field->ny * sizeof(double));
        memcpy(field->rho, rho_new, field->nx * field->ny * sizeof(double));
        memcpy(field->T, T_new, field->nx * field->ny * sizeof(double));
        
        // Apply boundary conditions
        apply_boundary_conditions(field, grid);
        
        // Output solution every 100 iterations
        if (iter % 100 == 0) {
            char filename[256];
            sprintf(filename, "output_%d.vtk", iter);
            write_vtk_output(filename, "pressure", field->p, field->nx, field->ny,
                           grid->xmin, grid->xmax, grid->ymin, grid->ymax);
        }
    }
    
    // Free temporary arrays
    cfd_free(u_new);
    cfd_free(v_new);
    cfd_free(p_new);
    cfd_free(rho_new);
    cfd_free(T_new);
} 