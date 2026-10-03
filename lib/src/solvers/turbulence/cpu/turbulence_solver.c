/**
 * @file turbulence_solver.c
 * @brief Scalar CPU implementation of the RANS turbulence solver
 *
 * Public entry points, model dispatch, boundary conditions, and the wall
 * functions (log law or Spalding's law of the wall). The per-model transport kernels live in
 * turbulence_kepsilon.c and turbulence_sa.c.
 */

#include "cfd/solvers/turbulence_solver.h"
#include "../turbulence_solver_internal.h"
#include "boundary/bc_edge_range.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/nn/cfdnn.h"

#include <math.h>
#include <string.h>

/* Kinematic viscosity at a grid point (density-floored, like the RHS kernel). */
static double local_nu(const ns_solver_params_t* params, const flow_field* field,
                       size_t idx) {
    return params->mu / fmax(field->rho[idx], 1e-10);
}

/* SA viscous damping function fv1(chi). */
static double sa_fv1(double chi) {
    double chi3 = chi * chi * chi;
    return chi3 / (chi3 + SA_CV1 * SA_CV1 * SA_CV1);
}

void turb_update_nu_t(flow_field* field, const ns_solver_params_t* params) {
    size_t total = field->nx * field->ny * field->nz;

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        for (size_t n = 0; n < total; n++) {
            double nu = local_nu(params, field, n);
            double k_c = fmax(field->turb_k[n], 0.0);
            double eps_c = fmax(field->turb_eps[n], TURB_EPS_MIN);
            double nu_t = TURB_C_MU * k_c * k_c / eps_c;
            field->nu_t[n] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
        }
    } else if (params->turb_model == TURB_MODEL_SPALART_ALLMARAS) {
        for (size_t n = 0; n < total; n++) {
            double nu = local_nu(params, field, n);
            double nt = fmax(field->turb_nu_tilde[n], 0.0);
            double nu_t = nt * sa_fv1(nt / nu);
            field->nu_t[n] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
        }
    }
}

/* The k-epsilon eddy viscosity before the realizability clamp.
 *
 * turb_update_nu_t stores the CLAMPED value, so a correction that scaled
 * field->nu_t directly would be scaling the cap wherever the clamp had bound:
 * beta < 1 would shrink the cap rather than the model's own answer, and
 * beta > 1 could not lift a clamped value at all. Both corrections are
 * k-epsilon only, so this reproduces that branch of turb_update_nu_t exactly
 * and the single clamp then lands where the docs say it does -- after the
 * correction, not before and after. */
static double turb_nu_t_raw(const flow_field* field, size_t n) {
    const double k_c   = fmax(field->turb_k[n], 0.0);
    const double eps_c = fmax(field->turb_eps[n], TURB_EPS_MIN);
    return TURB_C_MU * k_c * k_c / eps_c;
}


/* ==========================================================================
 * Optional eddy-viscosity corrections: algebraic, or learned
 *
 * Two alternatives on one seam, and the algebraic one is the incumbent: a
 * power law in the dimensionless strain rate, which is a C_mu that varies with
 * the local strain. Per the design note's governing rule -- ML earns its place
 * only where no analytic answer exists -- a learned closure has to beat it on a
 * held-out Reynolds number to be worth shipping. Setting both is refused.
 *
 * Either way the correction is applied ON TOP of an active closure, never
 * instead of one: the features are built from k and epsilon, which only exist
 * when a transport model is running.
 *
 * Feature set, chosen by the locality study in
 * docs/technical-notes/ml-integration-design.md:
 *
 *   ln S*     = ln(|S| k / eps)     dimensionless strain rate
 *   ln Re_t   = ln(k^2 / (nu eps))  turbulent Reynolds number
 *   ln y+     = ln(y u_tau / nu)    approximated here by ln(nu_t/nu)
 *
 * Those were measured to generalise across Reynolds number better than the
 * non-local coordinate y/delta, which is what makes a local closure viable at
 * all. Note that ln Re_t and ln(nu_t/nu) are nearly collinear for k-epsilon
 * (nu_t/nu = C_mu Re_t by construction; measured correlation 0.93), so they
 * carry less independent information than their count suggests.
 *
 * Safety is structural, not hoped for -- but note what it does and does not
 * promise. The correction is MULTIPLICATIVE and the DNS data asks for beta < 1
 * in the outer layer, so this removes eddy viscosity as well as adding it. It
 * is not a dissipation-only correction, and the claim that it can only
 * over-diffuse does not hold:
 *   - beta is clamped to [TURB_CLOSURE_BETA_MIN, TURB_CLOSURE_BETA_MAX], so a
 *     wrong model can scale nu_t by at most 10x either way, never to zero and
 *     never negative;
 *   - the existing realizability clamp still runs afterwards, bounding the
 *     upper side a second time;
 *   - a non-finite prediction fails the step with CFD_ERROR_DIVERGED rather
 *     than seeding NaN into the momentum equation.
 * A reduced nu_t means less damping, so stability is bounded by the floor
 * rather than argued away: at beta = BETA_MIN the momentum equation sees at
 * worst the laminar-viscosity-dominated limit it already handles when the
 * turbulence model is off.
 *
 * Scope, for whoever trains the model: turbulence_apply_bcs() runs after this
 * and overwrites nu_t at wall nodes and at the first interior node next to a
 * no-slip wall with the wall-shear-matching value. The correction therefore
 * cannot move those points, which is consistent with the design note's caveat
 * that they measure the wall treatment rather than the transport equations.
 * ========================================================================== */

/**
 * Strain-rate magnitude |S| = sqrt(2 S_ij S_ij) at cell n, from central
 * differences.
 *
 * Boundary nodes reuse the nearest interior value rather than a one-sided
 * stencil, since the wall function governs them anyway. Shared by both
 * corrections so they cannot drift apart in how they see the flow.
 */
static double turb_strain_magnitude(const flow_field* field, const grid* grid, size_t n) {
    const size_t nx = field->nx, ny = field->ny;
    const double dx = grid->dx ? grid->dx[0] : 1.0;
    const double dy = grid->dy ? grid->dy[0] : 1.0;

    size_t i = n % nx;
    size_t j = (n / nx) % ny;
    size_t ic = i == 0 ? 1 : (i == nx - 1 ? nx - 2 : i);
    size_t jc = j == 0 ? 1 : (j == ny - 1 ? ny - 2 : j);
    size_t c = (n / (nx * ny)) * nx * ny + jc * nx + ic;

    double dudy = (field->u[c + nx] - field->u[c - nx]) / (2.0 * dy);
    double dvdx = (field->v[c + 1] - field->v[c - 1]) / (2.0 * dx);
    double dudx = (field->u[c + 1] - field->u[c - 1]) / (2.0 * dx);
    double dvdy = (field->v[c + nx] - field->v[c - nx]) / (2.0 * dy);
    double s12 = 0.5 * (dudy + dvdx);
    return sqrt(2.0 * (dudx * dudx + dvdy * dvdy + 2.0 * s12 * s12));
}

/** Dimensionless strain rate S* = |S| k / epsilon at cell n. */
static double turb_s_star(const flow_field* field, const grid* grid, size_t n) {
    const double k_c = fmax(field->turb_k[n], TURB_K_MIN);
    const double eps_c = fmax(field->turb_eps[n], TURB_EPS_MIN);
    return turb_strain_magnitude(field, grid, n) * k_c / eps_c;
}

/**
 * Per-cell features for one tile of cells [base, base + count).
 *
 * Split out from the apply loop so the two array walks stay readable while the
 * grid is traversed in tiles; see TURB_CLOSURE_TILE for why it is tiled.
 */
static void turb_closure_features(const flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params,
                                  size_t base, size_t count, double* feats) {
    for (size_t t = 0; t < count; t++) {
        const size_t n = base + t;
        const double nu = local_nu(params, field, n);
        const double k_c = fmax(field->turb_k[n], TURB_K_MIN);
        const double eps_c = fmax(field->turb_eps[n], TURB_EPS_MIN);

        double s_star = turb_s_star(field, grid, n);
        double re_t = k_c * k_c / (nu * eps_c);
        double nut_p = field->nu_t[n] / nu;

        double* f = feats + t * TURB_CLOSURE_FEATURES;
        f[0] = log(fmax(s_star, 1e-30));
        f[1] = log(fmax(re_t, 1e-30));
        f[2] = log(fmax(nut_p, 1e-30));
    }
}

/**
 * Algebraic correction: nu_t *= clamp(A * (S*)^B).
 *
 * Equivalent to a C_mu that varies with the local strain rate, since
 * nu_t = C_mu k^2 / eps. Coefficients and their provenance are at
 * TURB_ALG_BETA_A in the internal header.
 */
static void turb_apply_algebraic_correction(flow_field* field, const grid* grid,
                                            const ns_solver_params_t* params) {
    const size_t total = field->nx * field->ny * field->nz;
    for (size_t n = 0; n < total; n++) {
        const double s_star = fmax(turb_s_star(field, grid, n), 1e-30);
        double b = TURB_ALG_BETA_A * pow(s_star, TURB_ALG_BETA_B);
        b = fmin(fmax(b, TURB_CLOSURE_BETA_MIN), TURB_CLOSURE_BETA_MAX);
        const double nu = local_nu(params, field, n);
        field->nu_t[n] = fmin(turb_nu_t_raw(field, n) * b, TURB_NU_T_MAX_FACTOR * nu);
    }
}

static int is_nonnegative_finite(double v) {
    return isfinite(v) && v >= 0.0;
}

/* The model-independent rules for one segment; see turbulence_bc_add_segment.
 * allow_detached accepts a segment whose profile a checkpoint could not store
 * (turb_segment_profile_detached), which only solver init may do. */
static cfd_status_t validate_segment(const ns_turbulence_bc_segment_t* seg,
                                     int allow_detached) {
    const char* reason = NULL;
    if (seg->edge != BC_EDGE_LEFT && seg->edge != BC_EDGE_RIGHT &&
        seg->edge != BC_EDGE_BOTTOM && seg->edge != BC_EDGE_TOP) {
        reason = "turbulence BC segment: edge must be BC_EDGE_LEFT, _RIGHT, _BOTTOM or _TOP";
    } else if (!(seg->start >= 0.0 && seg->start < seg->end && seg->end <= 1.0)) {
        reason = "turbulence BC segment: range must satisfy 0 <= start < end <= 1";
    } else if (seg->type != BC_TYPE_NEUMANN && seg->type != BC_TYPE_DIRICHLET &&
               seg->type != BC_TYPE_NOSLIP) {
        reason = "turbulence BC segment: type must be NEUMANN, DIRICHLET or NOSLIP "
                 "(PERIODIC has no meaning on part of a face)";
    } else if (turb_segment_profile_detached(seg)) {
        if (!allow_detached) {
            reason = "turbulence BC segment: its profile was not stored in the checkpoint "
                     "it was loaded from; re-attach segments[n].profile before stepping";
        }
    } else if (seg->type == BC_TYPE_DIRICHLET && !seg->profile &&
               (!is_nonnegative_finite(seg->k) || !is_nonnegative_finite(seg->eps) ||
                !is_nonnegative_finite(seg->nu_tilde))) {
        reason = "turbulence BC segment: DIRICHLET values must be finite and >= 0";
    }
    if (reason) {
        cfd_set_error(CFD_ERROR_INVALID, reason);
        return CFD_ERROR_INVALID;
    }
    return CFD_SUCCESS;
}

cfd_status_t turb_check_segments(const ns_solver_params_t* params, int allow_detached) {
    const ns_turbulence_bc_config_t* tbc = &params->turb_bc;
    if (tbc->n_segments > NS_TURB_BC_MAX_SEGMENTS) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: turb_bc.n_segments exceeds "
                      "NS_TURB_BC_MAX_SEGMENTS");
        return CFD_ERROR_INVALID;
    }
    const int is_ke = (params->turb_model == TURB_MODEL_K_EPSILON);
    for (size_t n = 0; n < tbc->n_segments; n++) {
        const ns_turbulence_bc_segment_t* seg = &tbc->segments[n];
        cfd_status_t status = validate_segment(seg, allow_detached);
        if (status != CFD_SUCCESS) {
            return status;
        }
        if (is_ke && seg->type == BC_TYPE_DIRICHLET && !seg->profile &&
            !turb_segment_profile_detached(seg) && !(seg->eps > 0.0)) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "turbulence_solver: a k-epsilon DIRICHLET segment needs "
                          "eps > 0 or a profile");
            return CFD_ERROR_INVALID;
        }
    }
    return CFD_SUCCESS;
}

cfd_status_t turb_check_closure_config(const ns_solver_params_t* params) {
    if (!params) {
        return CFD_SUCCESS;
    }
    if (params->turb_nut_correction != NS_NUT_CORRECTION_NONE &&
        params->turb_nut_correction != NS_NUT_CORRECTION_S_STAR) {
        cfd_set_error(CFD_ERROR_INVALID, "Unknown ns_solver_params_t.turb_nut_correction");
        return CFD_ERROR_INVALID;
    }
    /* Two multipliers on nu_t would compound, and the result would be neither
     * the fitted algebraic correction nor the trained one. Pick one. */
    if (params->turb_closure && params->turb_nut_correction != NS_NUT_CORRECTION_NONE) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "params.turb_closure and params.turb_nut_correction are "
                      "alternatives: set one, not both");
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!params->turb_closure && params->turb_nut_correction == NS_NUT_CORRECTION_NONE) {
        return CFD_SUCCESS; /* feature off */
    }
    /* SA carries no k/epsilon, so S* and its companions are undefined for it,
     * and with no model at all there is nothing to correct. Refused rather than
     * ignored: a caller who configured a correction asked for it to run. */
    if (params->turb_model != TURB_MODEL_K_EPSILON) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "an eddy-viscosity correction requires TURB_MODEL_K_EPSILON: its "
                      "features are built from k and epsilon, which no other model carries");
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!params->turb_closure) {
        return CFD_SUCCESS; /* algebraic correction needs no model */
    }

    const cfd_nn_model_t* model = cfd_nn_context_model(params->turb_closure);
    if (!model) {
        cfd_set_error(CFD_ERROR_INVALID, "params.turb_closure carries no model");
        return CFD_ERROR_INVALID;
    }
    if (cfd_nn_model_inputs(model) != TURB_CLOSURE_FEATURES ||
        cfd_nn_model_outputs(model) != TURB_CLOSURE_OUTPUTS) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "params.turb_closure model shape must be 3 inputs -> 1 output "
                      "(ln S*, ln Re_t, ln nu_t/nu -> beta)");
        return CFD_ERROR_INVALID;
    }
    if (cfd_nn_context_capacity(params->turb_closure) == 0) {
        cfd_set_error(CFD_ERROR_INVALID, "params.turb_closure has zero batch capacity");
        return CFD_ERROR_INVALID;
    }
    return CFD_SUCCESS;
}

cfd_status_t turb_apply_nu_t_correction(flow_field* field, const grid* grid,
                                        const ns_solver_params_t* params) {
    if (!params || (!params->turb_closure &&
                    params->turb_nut_correction == NS_NUT_CORRECTION_NONE)) {
        return CFD_SUCCESS; /* feature off: bit-identical to the un-corrected path */
    }

    cfd_status_t status = turb_check_closure_config(params);
    if (status != CFD_SUCCESS) {
        return status;
    }
    if (!field || !grid || !field->nu_t || !field->turb_k || !field->turb_eps) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turb_apply_nu_t_correction: missing field, grid or k-epsilon state");
        return CFD_ERROR_INVALID;
    }

    const size_t nx = field->nx, ny = field->ny;
    const size_t total = nx * ny * field->nz;
    if (total == 0) {
        return CFD_SUCCESS;
    }
    /* The central-difference features need one interior neighbour each way.
     * The step validator already enforces this; a direct caller may not have. */
    if (nx < 3 || ny < 3) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turb_apply_nu_t_correction requires nx >= 3 and ny >= 3");
        return CFD_ERROR_INVALID;
    }

    if (params->turb_nut_correction == NS_NUT_CORRECTION_S_STAR) {
        turb_apply_algebraic_correction(field, grid, params);
        return CFD_SUCCESS;
    }

    /* Tiled so the scratch is a fixed stack allocation rather than a per-step
     * malloc, and so a context smaller than the grid still works -- it just
     * makes the tiles smaller. */
    size_t tile = cfd_nn_context_capacity(params->turb_closure);
    if (tile > TURB_CLOSURE_TILE) {
        tile = TURB_CLOSURE_TILE;
    }
    double feats[TURB_CLOSURE_TILE * TURB_CLOSURE_FEATURES];
    double beta[TURB_CLOSURE_TILE];

    for (size_t base = 0; base < total; base += tile) {
        const size_t count = (total - base < tile) ? (total - base) : tile;

        turb_closure_features(field, grid, params, base, count, feats);

        status = cfd_nn_predict_batch(params->turb_closure, count, feats,
                                      count * TURB_CLOSURE_FEATURES, beta, count);
        if (status != CFD_SUCCESS) {
            cfd_set_error(status,
                          "learned eddy-viscosity closure: inference failed; nu_t is "
                          "left as the k-epsilon model produced it and the step fails");
            return status;
        }

        for (size_t t = 0; t < count; t++) {
            const size_t n = base + t;
            const double b = fmin(fmax(beta[t], TURB_CLOSURE_BETA_MIN), TURB_CLOSURE_BETA_MAX);
            const double nu = local_nu(params, field, n);
            field->nu_t[n] =
                fmin(turb_nu_t_raw(field, n) * b, TURB_NU_T_MAX_FACTOR * nu);
        }
    }
    return CFD_SUCCESS;
}

/* Type of the last segment on `edge` whose range strictly contains normalized
 * position t, or face_type. Used for wall distance, where t is an interval
 * midpoint and never sits on a bound, so the node-level slack does not apply. */
static bc_type_t segment_type_at(const ns_turbulence_bc_config_t* tbc, bc_edge_t edge,
                                 bc_type_t face_type, double t) {
    for (size_t n = tbc->n_segments; n-- > 0;) {
        const ns_turbulence_bc_segment_t* seg = &tbc->segments[n];
        if (seg->edge == edge && t >= seg->start && t <= seg->end) {
            return seg->type;
        }
    }
    return face_type;
}

/*
 * Fold one face's wall into the running minimum distance *d.
 *
 * normal: distance from the point to the face's line. along: the point's
 * coordinate along the face, which spans [lo, hi].
 *
 * Without segments on this face the face is a wall or not, and the distance is
 * `normal` exactly, as it always was. With segments, the face is cut at every
 * segment bound; each piece is a wall when its midpoint resolves to NOSLIP, and
 * the distance to a wall piece is the distance to that line segment. Position
 * maps to coordinate linearly, which is exact on the uniform grids the
 * turbulence models require.
 */
static void face_wall_distance(const ns_turbulence_bc_config_t* tbc, bc_edge_t edge,
                               bc_type_t face_type, double normal, double along,
                               double lo, double hi, double* d, int* found) {
    double cuts[2 + 2 * NS_TURB_BC_MAX_SEGMENTS];
    size_t n_cuts = 0;
    cuts[n_cuts++] = 0.0;
    cuts[n_cuts++] = 1.0;
    for (size_t n = 0; n < tbc->n_segments; n++) {
        if (tbc->segments[n].edge == edge) {
            cuts[n_cuts++] = tbc->segments[n].start;
            cuts[n_cuts++] = tbc->segments[n].end;
        }
    }

    if (n_cuts == 2) {
        if (face_type == BC_TYPE_NOSLIP) {
            *d = *found ? fmin(*d, normal) : normal;
            *found = 1;
        }
        return;
    }

    /* Insertion sort: at most 2 + 2 * NS_TURB_BC_MAX_SEGMENTS values. */
    for (size_t a = 1; a < n_cuts; a++) {
        double v = cuts[a];
        size_t b = a;
        while (b > 0 && cuts[b - 1] > v) {
            cuts[b] = cuts[b - 1];
            b--;
        }
        cuts[b] = v;
    }

    const double span = hi - lo;
    for (size_t a = 0; a + 1 < n_cuts; a++) {
        if (!(cuts[a + 1] > cuts[a])) {
            continue;
        }
        double mid = 0.5 * (cuts[a] + cuts[a + 1]);
        if (segment_type_at(tbc, edge, face_type, mid) != BC_TYPE_NOSLIP) {
            continue;
        }
        double p_lo = lo + cuts[a] * span;
        double p_hi = lo + cuts[a + 1] * span;
        double off = 0.0;
        if (along < p_lo) {
            off = p_lo - along;
        } else if (along > p_hi) {
            off = along - p_hi;
        }
        double c = (off > 0.0) ? hypot(normal, off) : normal;
        *d = *found ? fmin(*d, c) : c;
        *found = 1;
    }
}

double turb_wall_distance(const grid* grid, const ns_turbulence_bc_config_t* tbc,
                          size_t i, size_t j, int* has_wall) {
    double d = 0.0;
    int found = 0;
    const double x = grid->x[i];
    const double y = grid->y[j];
    const double x0 = grid->x[0];
    const double x1 = grid->x[grid->nx - 1];
    const double y0 = grid->y[0];
    const double y1 = grid->y[grid->ny - 1];

    face_wall_distance(tbc, BC_EDGE_LEFT, tbc->left, x - x0, y, y0, y1, &d, &found);
    face_wall_distance(tbc, BC_EDGE_RIGHT, tbc->right, x1 - x, y, y0, y1, &d, &found);
    face_wall_distance(tbc, BC_EDGE_BOTTOM, tbc->bottom, y - y0, x, x0, x1, &d, &found);
    face_wall_distance(tbc, BC_EDGE_TOP, tbc->top, y1 - y, x, x0, x1, &d, &found);

    *has_wall = found;
    return d;
}

/* sum_{n >= n0} x^n / n!, the tail of e^x after its first n0 terms. Summed as
 * a series for x <= 1, where e^x minus the partial sum would cancel and leave
 * Spalding's sublayer correction as rounding noise. */
static double exp_tail(double x, int n0) {
    if (x > 1.0) {
        double partial = 1.0;
        double t = 1.0;
        for (int n = 1; n < n0; n++) {
            t *= x / n;
            partial += t;
        }
        return exp(x) - partial;
    }
    double t = 1.0;
    for (int n = 1; n <= n0; n++) {
        t *= x / n;
    }
    double sum = 0.0;
    for (int n = n0; n < n0 + 20; n++) {
        sum += t;
        t *= x / (n + 1);
    }
    return sum;
}

/*
 * Spalding's law of the wall, one smooth y+(u+) through all three layers:
 *   y+ = u+ + e^{-kappa B} [e^{kappa u+} - 1 - kappa u+ - (kappa u+)^2/2 - (kappa u+)^3/6]
 * Linear (y+ -> u+) in the viscous sublayer, logarithmic (u+ -> ln(y+)/kappa + B)
 * deep in the log layer, blended through the buffer layer.
 *
 * With u+ = u_p/u_tau and y+ = u_tau*y_p/nu, their product is the known wall
 * Reynolds number Re_p = u_p*y_p/nu, so solve for u+ alone:
 *   G(u+) = ln(u+) + ln(y+(u+)) - ln(Re_p) = 0.
 * G increases with u+, so the root is unique. Since y+ >= u+, sqrt(Re_p) (the
 * linear-law answer) bounds it above; Newton is safeguarded by bisection
 * inside that bracket.
 */
static double wall_u_tau_spalding(double u_p, double y_p, double nu) {
    const double c = exp(-WALL_KAPPA * WALL_B);
    const double log_re = log(u_p) + log(y_p) - log(nu);

    /* The cap keeps e^{kappa u+} finite; u+ reaches it only for Re_p ~ 1e300. */
    double lo = 0.0;
    double hi = fmin(exp(0.5 * log_re), 700.0 / WALL_KAPPA);
    /* Log-law estimate: close wherever the linear-law bound is not. */
    double up = fmin(hi, WALL_B + log_re / WALL_KAPPA);
    if (!(up > 0.0)) {
        up = hi;
    }

    for (int it = 0; it < 100; it++) {
        const double ku = WALL_KAPPA * up;
        const double yplus = up + c * exp_tail(ku, 4);
        const double dyplus = 1.0 + c * WALL_KAPPA * exp_tail(ku, 3);
        const double g = log(up) + log(yplus) - log_re;
        if (g == 0.0) {
            break;
        }
        if (g > 0.0) {
            hi = up;
        } else {
            lo = up;
        }
        double next = up - g / (1.0 / up + dyplus / yplus);
        if (!(next > lo && next <= hi)) {
            next = 0.5 * (lo + hi);
        }
        const int converged = fabs(next - up) <= 1e-14 * up;
        up = next;
        if (converged) {
            break;
        }
    }
    return u_p / up;
}

/*
 * y+ where the linear law u+ = y+ meets the log law u+ = ln(y+)/kappa + B:
 * 11.06 at kappa = 0.41, B = 5.2. Solved rather than hard-coded so it follows
 * the constants; a switch placed anywhere else (the former 11.63) makes u_tau
 * jump, 3.3% there. f(y) = y - ln(y)/kappa - B has a second root below y = 1;
 * Newton from 11 converges to this one in a few steps.
 */
static double wall_yplus_crossover(void) {
    double y = 11.0;
    for (int it = 0; it < 20; it++) {
        double next = y - (y - log(y) / WALL_KAPPA - WALL_B) / (1.0 - 1.0 / (WALL_KAPPA * y));
        const int converged = fabs(next - y) <= 1e-14 * y;
        y = next;
        if (converged) {
            break;
        }
    }
    return y;
}

/*
 * Linear law below the crossover, log law above it. With u+ = u_p/u_tau and
 * y+ = u_tau*y_p/nu, u+ * y+ = Re_p = u_p*y_p/nu, so the switch is
 * Re_p <= y+_c^2, known before u_tau is. There the linear answer
 * sqrt(nu*u_p/y_p) puts u+ = y+ = y+_c on both laws, so u_tau is continuous.
 * Above it, Newton on ut*(ln(ut*y_p/nu)/kappa + B) = u_p from the linear value.
 */
static double wall_u_tau_log(double u_p, double y_p, double nu) {
    const double yc = wall_yplus_crossover();
    double ut = sqrt(nu * u_p / y_p);
    if (u_p * y_p / nu <= yc * yc) {
        return ut;
    }
    for (int it = 0; it < 50; it++) {
        const double yplus = ut * y_p / nu;
        const double f = ut * (log(yplus) / WALL_KAPPA + WALL_B) - u_p;
        const double fp = (log(yplus) + 1.0) / WALL_KAPPA + WALL_B;
        double next = ut - f / fp;
        if (next < 1e-12) {
            next = 1e-12;
        }
        const int converged = fabs(next - ut) <= 1e-14 * ut;
        ut = next;
        if (converged) {
            break;
        }
    }
    return ut;
}

double turbulence_wall_u_tau(ns_wall_law_t law, double u_p, double y_p, double nu) {
    if (u_p <= 0.0 || y_p <= 0.0 || nu <= 0.0) {
        return 0.0;
    }
    switch (law) {
        case NS_WALL_LAW_LOG:
            return wall_u_tau_log(u_p, y_p, nu);
        case NS_WALL_LAW_SPALDING:
            return wall_u_tau_spalding(u_p, y_p, nu);
    }
    return 0.0;
}

/* Validate uniform spacing (the transport stencils assume it, like energy). */
static cfd_status_t validate_uniform_spacing(const grid* grid, size_t nx, size_t ny) {
    const double dx0 = grid->dx[0];
    const double tol_x = 1e-12 * fmax(1.0, fabs(dx0));
    for (size_t i = 1; i < nx - 1; i++) {
        if (fabs(grid->dx[i] - dx0) > tol_x) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                          "turbulence_solver: non-uniform dx not supported");
            return CFD_ERROR_UNSUPPORTED;
        }
    }
    const double dy0 = grid->dy[0];
    const double tol_y = 1e-12 * fmax(1.0, fabs(dy0));
    for (size_t j = 1; j < ny - 1; j++) {
        if (fabs(grid->dy[j] - dy0) > tol_y) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                          "turbulence_solver: non-uniform dy not supported");
            return CFD_ERROR_UNSUPPORTED;
        }
    }
    return CFD_SUCCESS;
}

/* Shared validation for step and BC application. */
static cfd_status_t validate_turbulence_args(const flow_field* field, const grid* grid,
                                             const ns_solver_params_t* params) {
    if (!field || !grid || !params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: field, grid, and params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (!field->turb_k || !field->turb_eps || !field->turb_nu_tilde || !field->nu_t) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: missing turbulence fields");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model != TURB_MODEL_K_EPSILON &&
        params->turb_model != TURB_MODEL_SPALART_ALLMARAS) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: unknown turbulence model");
        return CFD_ERROR_INVALID;
    }
    /* nu = mu/rho appears in denominators (SA chi, wall functions); mu <= 0
     * would produce Inf/NaN rather than a diagnosable error. */
    if (params->mu <= 0.0) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: mu must be positive when a "
                      "turbulence model is active");
        return CFD_ERROR_INVALID;
    }
    if (field->nz > 1) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "turbulence_solver: 3D turbulence not supported");
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!grid->dx || !grid->dy || field->nx < 3 || field->ny < 3) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: grid too small or missing dx/dy");
        return CFD_ERROR_INVALID;
    }
    /* Shared by the step and the BCs, not left to the BCs: the transport step
     * reads the segments too (SA wall distance) and runs before the BCs do. */
    return turb_check_segments(params, 0);
}

cfd_status_t turb_validate_step_args(const flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params) {
    cfd_status_t status = validate_turbulence_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return validate_uniform_spacing(grid, field->nx, field->ny);
}

cfd_status_t turbulence_step_explicit_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* workspace, size_t workspace_size) {
    (void)time;

    if (!params) {
        cfd_set_error(CFD_ERROR_INVALID, "turbulence_solver: params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    /* Checked before the disabled-model exit below: a correction configured
     * against no turbulence model is refused here exactly as it is at solver
     * init, rather than silently skipped because the step has nothing to do. */
    cfd_status_t status = turb_check_closure_config(params);
    if (status != CFD_SUCCESS) {
        return status;
    }
    /* Turbulence disabled: a legitimate no-op. */
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }

    status = turb_validate_step_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }

    size_t total = field->nx * field->ny * field->nz;

    /* Use caller's workspace or allocate internally */
    int owns_buffer = 0;
    double* buf;
    if (workspace && workspace_size >= TURB_WORKSPACE_SIZE(total)) {
        buf = workspace;
    } else {
        buf = (double*)cfd_calloc(TURB_WORKSPACE_SIZE(total), sizeof(double));
        if (!buf) {
            return CFD_ERROR_NOMEM;
        }
        owns_buffer = 1;
    }

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        double* k_new = buf;
        double* eps_new = buf + total;
        memcpy(k_new, field->turb_k, total * sizeof(double));
        memcpy(eps_new, field->turb_eps, total * sizeof(double));

        turb_kepsilon_step_scalar(field, grid, params, dt, k_new, eps_new);

        for (size_t n = 0; n < total; n++) {
            if (!isfinite(k_new[n]) || !isfinite(eps_new[n])) {
                cfd_set_error(CFD_ERROR_DIVERGED,
                              "NaN/Inf detected in turbulence_step_explicit (k-epsilon)");
                if (owns_buffer) cfd_free(buf);
                return CFD_ERROR_DIVERGED;
            }
        }
        memcpy(field->turb_k, k_new, total * sizeof(double));
        memcpy(field->turb_eps, eps_new, total * sizeof(double));
    } else {
        double* nt_new = buf;
        memcpy(nt_new, field->turb_nu_tilde, total * sizeof(double));

        turb_sa_step_scalar(field, grid, params, dt, nt_new);

        for (size_t n = 0; n < total; n++) {
            if (!isfinite(nt_new[n])) {
                cfd_set_error(CFD_ERROR_DIVERGED,
                              "NaN/Inf detected in turbulence_step_explicit (SA)");
                if (owns_buffer) cfd_free(buf);
                return CFD_ERROR_DIVERGED;
            }
        }
        memcpy(field->turb_nu_tilde, nt_new, total * sizeof(double));
    }

    turb_update_nu_t(field, params);
    cfd_status_t closure_status = turb_apply_nu_t_correction(field, grid, params);

    if (owns_buffer) cfd_free(buf);
    return closure_status;
}

cfd_status_t turbulence_step_explicit(flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params,
                                      double dt, double time) {
    return turbulence_step_explicit_with_workspace(field, grid, params,
                                                   dt, time, NULL, 0);
}

/* A face may only request a turbulence BC type the solver implements. */
static int is_supported_turb_bc(bc_type_t type) {
    return type == BC_TYPE_PERIODIC || type == BC_TYPE_NEUMANN ||
           type == BC_TYPE_DIRICHLET || type == BC_TYPE_NOSLIP;
}

/*
 * Wall function at one wall node, u_tau from the law in params->turb_bc.wall_law.
 *
 * idx_w: wall node, idx_p: first interior node at distance y_p, u_p: wall-
 * parallel speed at idx_p. Sets equilibrium turbulence values at the first
 * interior node (fixed-value wall-function variant):
 *   k_p  = u_tau^2 / sqrt(C_mu)      eps_p = u_tau^3 / (kappa*y_p)
 *   nt_p = kappa*u_tau*y_p
 * and imposes the wall-law momentum sink through the wall-face viscosity: the
 * discrete wall shear is (nu + 0.5*(nu_t_w + nu_t_p)) * u_p/y_p, so storing
 *   nu_t_w = 0,   nu_t_p = max(2*(u_tau^2*y_p/u_p - nu), 0)
 * makes it exactly u_tau^2. The equilibrium eddy viscosity kappa*u_tau*y_p is
 * deliberately NOT stored at idx_p: the coarse one-sided gradient u_p/y_p
 * cannot represent the log profile's curvature, and pairing it with the
 * physical nu_t would overpredict the wall shear several-fold. Both wall laws
 * have y+ >= u+, so u_tau^2*y_p/u_p = nu*y+/u+ >= nu and nu_t_p is never
 * negative. In the viscous sublayer it is exactly 0 on the log law's linear
 * branch and vanishes as (kappa*u+)^4 on Spalding's (laminar shear).
 */
static void apply_wall_function_node(flow_field* field,
                                     const ns_solver_params_t* params,
                                     size_t idx_w, size_t idx_p,
                                     double y_p, double u_p) {
    double nu = local_nu(params, field, idx_p);
    double ut = turbulence_wall_u_tau(params->turb_bc.wall_law, u_p, y_p, nu);

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        field->turb_k[idx_p] = fmax(ut * ut / sqrt(TURB_C_MU), TURB_K_MIN);
        field->turb_eps[idx_p] = fmax(ut * ut * ut / (WALL_KAPPA * y_p), TURB_EPS_MIN);
        field->turb_k[idx_w] = 0.0;
        field->turb_eps[idx_w] = field->turb_eps[idx_p];
    } else {
        field->turb_nu_tilde[idx_p] = WALL_KAPPA * ut * y_p;
        field->turb_nu_tilde[idx_w] = 0.0;
    }

    /* Wall-shear-matching viscosity (see function comment), subject to the
     * same realizability bound as turb_update_nu_t: on very coarse grids or
     * near-zero u_p the matching formula can blow up and destabilize the
     * momentum step / time-step estimation. */
    if (u_p > 1e-12) {
        double nu_wall_eff = ut * ut * y_p / u_p;
        field->nu_t[idx_p] = fmin(fmax(2.0 * (nu_wall_eff - nu), 0.0),
                                  TURB_NU_T_MAX_FACTOR * nu);
    } else {
        field->nu_t[idx_p] = 0.0;
    }
    field->nu_t[idx_w] = 0.0;
}

/* Apply one BC type to one scalar array cell (periodic/neumann/dirichlet). */
static void apply_scalar_face_bc(double* arr, bc_type_t type, size_t idx,
                                 size_t idx_interior, size_t idx_periodic,
                                 double dirichlet_value) {
    if (type == BC_TYPE_DIRICHLET) {
        arr[idx] = dirichlet_value;
    } else if (type == BC_TYPE_NEUMANN) {
        arr[idx] = arr[idx_interior];
    } else if (type == BC_TYPE_PERIODIC) {
        arr[idx] = arr[idx_periodic];
    }
    /* BC_TYPE_NOSLIP handled by the wall-function path, not here. */
}

/* Set nu_t at a Dirichlet boundary node consistently with the fixed values. */
static void set_dirichlet_nu_t(flow_field* field, const ns_solver_params_t* params,
                               size_t idx) {
    double nu = local_nu(params, field, idx);
    double nu_t;
    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        double k_c = fmax(field->turb_k[idx], 0.0);
        double eps_c = fmax(field->turb_eps[idx], TURB_EPS_MIN);
        nu_t = TURB_C_MU * k_c * k_c / eps_c;
    } else {
        double nt = fmax(field->turb_nu_tilde[idx], 0.0);
        nu_t = nt * sa_fv1(nt / nu);
    }
    field->nu_t[idx] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
}

/* The BC one boundary node receives: its type and, for DIRICHLET, its values. */
typedef struct {
    bc_type_t type;
    double k;
    double eps;
    double nu_tilde;
} turb_node_bc_t;

/*
 * Resolve the BC at normalized position t on `edge`: the last segment covering
 * t, or the face type and face values. A profile is evaluated here and its
 * output checked, since a negative k or a zero epsilon would otherwise enter
 * the transport equations as if it were data.
 */
static cfd_status_t resolve_node_bc(const ns_turbulence_bc_config_t* tbc, bc_edge_t edge,
                                    const turb_node_bc_t* face, int is_ke, double t,
                                    turb_node_bc_t* out) {
    *out = *face;
    for (size_t n = tbc->n_segments; n-- > 0;) {
        const ns_turbulence_bc_segment_t* seg = &tbc->segments[n];
        double position;
        if (seg->edge != edge || !bc_edge_range_position(t, seg->start, seg->end, &position)) {
            continue;
        }
        out->type = seg->type;
        if (seg->type != BC_TYPE_DIRICHLET) {
            return CFD_SUCCESS;
        }
        if (!seg->profile) {
            out->k = seg->k;
            out->eps = seg->eps;
            out->nu_tilde = seg->nu_tilde;
            return CFD_SUCCESS;
        }
        out->k = out->eps = out->nu_tilde = NAN;
        seg->profile(position, &out->k, &out->eps, &out->nu_tilde, seg->profile_user_data);
        const int ok = is_ke ? (is_nonnegative_finite(out->k) && isfinite(out->eps) &&
                                out->eps > 0.0)
                             : is_nonnegative_finite(out->nu_tilde);
        if (!ok) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "turbulence_apply_bcs: a segment profile returned a negative or "
                          "non-finite value (or eps <= 0 under k-epsilon)");
            return CFD_ERROR_INVALID;
        }
        return CFD_SUCCESS;
    }
    return CFD_SUCCESS;
}

/* Boundary node s of a face, its first interior neighbour, and its periodic
 * partner on the opposite face. */
static void face_node_indices(bc_edge_t edge, size_t nx, size_t ny, size_t s,
                              size_t* idx, size_t* idx_int, size_t* idx_per) {
    switch (edge) {
        case BC_EDGE_LEFT:
            *idx = s * nx;
            *idx_int = *idx + 1;
            *idx_per = s * nx + (nx - 2);
            break;
        case BC_EDGE_RIGHT:
            *idx = s * nx + (nx - 1);
            *idx_int = *idx - 1;
            *idx_per = s * nx + 1;
            break;
        case BC_EDGE_BOTTOM:
            *idx = s;
            *idx_int = s + nx;
            *idx_per = (ny - 2) * nx + s;
            break;
        default: /* BC_EDGE_TOP */
            *idx = (ny - 1) * nx + s;
            *idx_int = *idx - nx;
            *idx_per = nx + s;
            break;
    }
}

/* Apply a resolved BC at one boundary node. y_p and u_tan (the wall-parallel
 * speed at idx_int) are read only by the wall function. */
static void apply_node_bc(flow_field* field, const ns_solver_params_t* params,
                          const turb_node_bc_t* bc, int is_ke, size_t idx, size_t idx_int,
                          size_t idx_per, double y_p, double u_tan) {
    if (bc->type == BC_TYPE_NOSLIP) {
        apply_wall_function_node(field, params, idx, idx_int, y_p, u_tan);
        return;
    }
    if (is_ke) {
        apply_scalar_face_bc(field->turb_k, bc->type, idx, idx_int, idx_per, bc->k);
        apply_scalar_face_bc(field->turb_eps, bc->type, idx, idx_int, idx_per, bc->eps);
    } else {
        apply_scalar_face_bc(field->turb_nu_tilde, bc->type, idx, idx_int, idx_per,
                             bc->nu_tilde);
    }
    if (bc->type == BC_TYPE_DIRICHLET) {
        set_dirichlet_nu_t(field, params, idx);
    } else {
        apply_scalar_face_bc(field->nu_t, bc->type, idx, idx_int, idx_per, 0.0);
    }
}

/* Faces in application order: left and right first, then bottom and top, which
 * overwrite the corners. */
static const bc_edge_t face_order[4] = {BC_EDGE_LEFT, BC_EDGE_RIGHT, BC_EDGE_BOTTOM,
                                          BC_EDGE_TOP};

static size_t face_node_count(bc_edge_t edge, size_t nx, size_t ny) {
    return (edge == BC_EDGE_LEFT || edge == BC_EDGE_RIGHT) ? ny : nx;
}

/*
 * Resolve every boundary node of every face, in face_order, into nodes[]
 * (2 * (nx + ny) entries). This is where profiles are called and their output
 * checked, so a refusal happens before any field is written.
 */
static cfd_status_t resolve_all_faces(const ns_turbulence_bc_config_t* tbc, int is_ke,
                                      size_t nx, size_t ny, const turb_node_bc_t faces[4],
                                      turb_node_bc_t* nodes) {
    size_t offset = 0;
    for (int f = 0; f < 4; f++) {
        const size_t count = face_node_count(face_order[f], nx, ny);
        for (size_t s = 0; s < count; s++) {
            cfd_status_t status = resolve_node_bc(tbc, face_order[f], &faces[f], is_ke,
                                                  bc_edge_node_t(s, count),
                                                  &nodes[offset + s]);
            if (status != CFD_SUCCESS) {
                return status;
            }
        }
        offset += count;
    }
    return CFD_SUCCESS;
}

/* Apply one face node by node: node s takes nodes[s], or the face's own BC when
 * nodes is NULL (no segments). Every node reads only its own interior
 * neighbour, so the order along the face does not matter. */
static void apply_face_bcs(flow_field* field, const grid* grid,
                           const ns_solver_params_t* params, bc_edge_t edge,
                           const turb_node_bc_t* face, const turb_node_bc_t* nodes) {
    const size_t nx = field->nx;
    const size_t ny = field->ny;
    const int along_y = (edge == BC_EDGE_LEFT || edge == BC_EDGE_RIGHT);
    const size_t count = face_node_count(edge, nx, ny);
    const int is_ke = (params->turb_model == TURB_MODEL_K_EPSILON);

    double y_p;
    switch (edge) {
        case BC_EDGE_LEFT: y_p = grid->x[1] - grid->x[0]; break;
        case BC_EDGE_RIGHT: y_p = grid->x[nx - 1] - grid->x[nx - 2]; break;
        case BC_EDGE_BOTTOM: y_p = grid->y[1] - grid->y[0]; break;
        default: y_p = grid->y[ny - 1] - grid->y[ny - 2]; break;
    }

    for (size_t s = 0; s < count; s++) {
        size_t idx, idx_int, idx_per;
        face_node_indices(edge, nx, ny, s, &idx, &idx_int, &idx_per);
        const double u_tan = along_y ? fabs(field->v[idx_int]) : fabs(field->u[idx_int]);
        apply_node_bc(field, params, nodes ? &nodes[s] : face, is_ke, idx, idx_int, idx_per,
                      y_p, u_tan);
    }
}

cfd_status_t turbulence_apply_bcs(flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params) {
    if (!params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_apply_bcs: params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }

    cfd_status_t status = validate_turbulence_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }

    const ns_turbulence_bc_config_t* tbc = &params->turb_bc;

    if (!is_supported_turb_bc(tbc->left) || !is_supported_turb_bc(tbc->right) ||
        !is_supported_turb_bc(tbc->bottom) || !is_supported_turb_bc(tbc->top) ||
        (tbc->wall_law != NS_WALL_LAW_LOG && tbc->wall_law != NS_WALL_LAW_SPALDING)) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_apply_bcs: unsupported turbulence BC type on a face "
                      "(only PERIODIC, NEUMANN, DIRICHLET, NOSLIP are valid) or unknown "
                      "turb_bc.wall_law");
        return CFD_ERROR_INVALID;
    }

    /* In face_order */
    const turb_node_bc_t faces[4] = {
        {tbc->left, tbc->k_values.left, tbc->eps_values.left, tbc->nu_tilde_values.left},
        {tbc->right, tbc->k_values.right, tbc->eps_values.right, tbc->nu_tilde_values.right},
        {tbc->bottom, tbc->k_values.bottom, tbc->eps_values.bottom,
         tbc->nu_tilde_values.bottom},
        {tbc->top, tbc->k_values.top, tbc->eps_values.top, tbc->nu_tilde_values.top},
    };
    const size_t nx = field->nx;
    const size_t ny = field->ny;

    /* With segments, every node is resolved first -- profiles called, their
     * output checked -- and only then is any field written, so a refusal leaves
     * the fields as they were (the segments themselves were checked with the
     * other arguments). One small allocation per call, 2 * (nx + ny) nodes, and
     * none without segments. */
    turb_node_bc_t* nodes = NULL;
    if (tbc->n_segments > 0) {
        nodes = (turb_node_bc_t*)cfd_calloc(2 * (nx + ny), sizeof(*nodes));
        if (!nodes) {
            return CFD_ERROR_NOMEM;
        }
        status = resolve_all_faces(tbc, params->turb_model == TURB_MODEL_K_EPSILON, nx, ny,
                                   faces, nodes);
        if (status != CFD_SUCCESS) {
            cfd_free(nodes);
            return status;
        }
    }

    size_t offset = 0;
    for (int f = 0; f < 4; f++) {
        apply_face_bcs(field, grid, params, face_order[f], &faces[f],
                       nodes ? &nodes[offset] : NULL);
        offset += face_node_count(face_order[f], nx, ny);
    }
    cfd_free(nodes);
    return CFD_SUCCESS;
}

cfd_status_t turbulence_bc_add_segment(ns_turbulence_bc_config_t* bc,
                                       const ns_turbulence_bc_segment_t* segment) {
    if (!bc || !segment) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_bc_add_segment: bc and segment must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (bc->n_segments >= NS_TURB_BC_MAX_SEGMENTS) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_bc_add_segment: no room; NS_TURB_BC_MAX_SEGMENTS "
                      "segments are already set");
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = validate_segment(segment, 0);
    if (status != CFD_SUCCESS) {
        return status;
    }
    bc->segments[bc->n_segments++] = *segment;
    return CFD_SUCCESS;
}

cfd_status_t turbulence_init_uniform(flow_field* field,
                                     const ns_solver_params_t* params,
                                     double k0, double eps0, double nu_tilde0) {
    if (!field || !params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_init_uniform: field and params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }
    if (!field->turb_k || !field->turb_eps || !field->turb_nu_tilde || !field->nu_t) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_init_uniform: missing turbulence fields");
        return CFD_ERROR_INVALID;
    }

    size_t total = field->nx * field->ny * field->nz;

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        if (k0 < 0.0 || eps0 <= 0.0) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "turbulence_init_uniform: k0 must be >= 0 and eps0 > 0");
            return CFD_ERROR_INVALID;
        }
        for (size_t n = 0; n < total; n++) {
            field->turb_k[n] = k0;
            field->turb_eps[n] = eps0;
        }
    } else {
        if (nu_tilde0 < 0.0) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "turbulence_init_uniform: nu_tilde0 must be >= 0");
            return CFD_ERROR_INVALID;
        }
        for (size_t n = 0; n < total; n++) {
            field->turb_nu_tilde[n] = nu_tilde0;
        }
    }

    /* No learned correction here: this is initialization, and the features it
     * consumes (a strain rate, a turbulent Reynolds number) are meaningless
     * before the flow has developed. The correction belongs to the per-step
     * path only. */
    turb_update_nu_t(field, params);
    return CFD_SUCCESS;
}
