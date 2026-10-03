#ifndef CFD_NN_INTERNAL_H
#define CFD_NN_INTERNAL_H

#include "cfd/nn/cfdnn.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ==========================================================================
 * Robustness caps for malformed or hostile files (not physical limits).
 * Every cap is checked BEFORE the corresponding allocation, and all size
 * arithmetic is done in uint64_t before narrowing to size_t.
 * ========================================================================== */

#define CFD_NN_MAX_LAYERS   64u
#define CFD_NN_MAX_FEATURES 4096u
#define CFD_NN_MAX_WEIGHTS  (1u << 22) /* 4M floats = 16 MB per layer */
#define CFD_NN_MAX_STRING   (1u << 12)

/* ==========================================================================
 * Model representation
 *
 * Weights for all layers live in one contiguous allocation so a model is one
 * malloc plus one free, and so the hot loop walks memory forwards.
 * ========================================================================== */

typedef struct {
    cfd_nn_activation_t activation;
    float               act_param;
    size_t              in_features;
    size_t              out_features;
    const float*        weights; /* [out][in], row-major; into model->blob */
    const float*        bias;    /* [out], or NULL; into model->blob */
} cfd_nn_layer_t;

struct cfd_nn_model {
    char*           name;
    float*          blob; /* single allocation backing every layer's weights */
    cfd_nn_layer_t* layers;
    size_t          layer_count;
    size_t          inputs;
    size_t          outputs;
    size_t          widest; /* max layer width; sizes the scratch buffers */
};

/* ==========================================================================
 * Backend dispatch
 *
 * A function-pointer table, mirroring bc_backend_impl_t in
 * boundary/boundary_conditions_internal.h. NN kernels are the same arithmetic
 * on different hardware -- the boundary-condition case, not the "rk2 vs rk4"
 * case that the string registry exists for. A NULL entry means the backend
 * does not provide that op, and context creation rejects the model rather than
 * silently falling back.
 * ========================================================================== */

/**
 * One Dense layer over a batch.
 *
 * @param l        Layer to apply.
 * @param batch    Number of samples.
 * @param in       batch * l->in_features floats, sample-major.
 * @param out      batch * l->out_features floats, sample-major.
 * @param scratch  scratch_lanes * model->widest floats owned by the context,
 *                 or NULL when the table's scratch_lanes is 0.
 *
 * Implementations MUST accumulate over the input-feature axis in ascending
 * index order and MUST NOT reduce across SIMD lanes or threads: lanes and
 * threads carry distinct SAMPLES. That is what keeps OpenMP bit-identical to
 * scalar, and keeps SIMD's sums within FMA-contraction distance of it.
 */
typedef void (*cfd_nn_dense_fn)(const cfd_nn_layer_t* l, size_t batch,
                                const float* in, float* out, float* scratch);

typedef struct {
    const char*     name;
    cfd_nn_dense_fn dense;
    /** Samples the kernel transposes at once. The context allocates
     *  scratch_lanes * widest floats of scratch for it; 0 means none. */
    size_t          scratch_lanes;
} cfd_nn_backend_impl_t;

extern const cfd_nn_backend_impl_t cfd_nn_impl_scalar;
extern const cfd_nn_backend_impl_t cfd_nn_impl_omp;
/* Architecture tables. Each is defined on every platform and carries a NULL
 * kernel where its instruction set is not compiled in, so the dispatcher can
 * reference both without an #ifdef. */
extern const cfd_nn_backend_impl_t cfd_nn_impl_avx2;
extern const cfd_nn_backend_impl_t cfd_nn_impl_neon;

/** The SIMD table for this build AND this CPU (runtime-detected), or NULL
 *  when neither AVX2 nor NEON is usable. Its name is "avx2" or "neon". */
const cfd_nn_backend_impl_t* cfd_nn_simd_impl(void);

/* Scalar activation, applied in place to n values. Defined once in the scalar
 * backend; OpenMP uses it for everything, which is part of why OpenMP stays
 * bit-identical. SIMD uses it for identity/ReLU/leaky ReLU and its scalar
 * tail, and its own vector approximations for tanh/sigmoid/softplus. */
void cfd_nn_apply_activation(cfd_nn_activation_t act, float param,
                             float* v, size_t n);

/* Format layer: load/write live in cfdnn_format.c. */
cfd_status_t cfd_nn_load_impl(const char* path, const void* bytes, size_t size,
                              cfd_nn_model_t** out_model);

#endif /* CFD_NN_INTERNAL_H */
