/**
 * @file cfdnn.c
 * @brief Technology-agnostic dispatcher for `.cfdnn` inference.
 *
 * Holds the model and context lifecycles, selects a backend implementation
 * table, and runs the layer loop. There are no architecture #ifdefs here: per
 * the project's backend-abstraction rule those belong in the backend files
 * only, and this file must read the same on every platform.
 */

#include "cfdnn_internal.h"

#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <string.h>

/* ==========================================================================
 * Model
 * ========================================================================== */

cfd_status_t cfd_nn_model_load(const char* path, cfd_nn_model_t** out_model) {
    if (!path) {
        return CFD_ERROR_INVALID;
    }
    return cfd_nn_load_impl(path, NULL, 0, out_model);
}

cfd_status_t cfd_nn_model_load_memory(const void* bytes, size_t size,
                                      cfd_nn_model_t** out_model) {
    if (!bytes || size == 0) {
        return CFD_ERROR_INVALID;
    }
    return cfd_nn_load_impl(NULL, bytes, size, out_model);
}

size_t cfd_nn_model_inputs(const cfd_nn_model_t* model) {
    return model ? model->inputs : 0;
}

size_t cfd_nn_model_outputs(const cfd_nn_model_t* model) {
    return model ? model->outputs : 0;
}

const char* cfd_nn_model_name(const cfd_nn_model_t* model) {
    return (model && model->name) ? model->name : "";
}

/* ==========================================================================
 * Backend selection
 * ========================================================================== */

struct cfd_nn_context {
    const cfd_nn_model_t*        model;
    const cfd_nn_backend_impl_t* impl;
    size_t                       max_batch;
    float*                       buf_a; /* ping-pong activation buffers */
    float*                       buf_b;
    size_t                       buf_elems;
};

static const cfd_nn_backend_impl_t* resolve_backend(cfd_nn_backend_t want) {
    switch (want) {
        case CFD_NN_BACKEND_SCALAR:
            return &cfd_nn_impl_scalar;
        case CFD_NN_BACKEND_SIMD:
            return cfd_nn_impl_simd.dense ? &cfd_nn_impl_simd : NULL;
        case CFD_NN_BACKEND_OMP:
            return cfd_nn_impl_omp.dense ? &cfd_nn_impl_omp : NULL;
        case CFD_NN_BACKEND_AUTO:
            /* Priority SIMD > OMP > scalar, matching BC_BACKEND_AUTO. AUTO is
             * allowed to pick; an EXPLICIT request that cannot be honoured is
             * an error, never a silent substitution. */
            if (cfd_nn_impl_simd.dense) {
                return &cfd_nn_impl_simd;
            }
            if (cfd_nn_impl_omp.dense) {
                return &cfd_nn_impl_omp;
            }
            return &cfd_nn_impl_scalar;
        default:
            return NULL;
    }
}

bool cfd_nn_backend_available(cfd_nn_backend_t backend) {
    return resolve_backend(backend) != NULL;
}

/* ==========================================================================
 * Context
 * ========================================================================== */

cfd_status_t cfd_nn_context_create(const cfd_nn_model_t* model, size_t max_batch,
                                   cfd_nn_backend_t backend,
                                   cfd_nn_context_t** out_ctx) {
    if (!out_ctx) {
        return CFD_ERROR_INVALID;
    }
    *out_ctx = NULL;
    if (!model || max_batch == 0) {
        return CFD_ERROR_INVALID;
    }

    const cfd_nn_backend_impl_t* impl = resolve_backend(backend);
    if (!impl) {
        /* No fallback: the caller decides what to do, per the project rule
         * that an unavailable backend is reported, never substituted. */
        return CFD_ERROR_UNSUPPORTED;
    }

    cfd_nn_context_t* ctx = (cfd_nn_context_t*)cfd_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }
    ctx->model     = model;
    ctx->impl      = impl;
    ctx->max_batch = max_batch;
    ctx->buf_elems = max_batch * model->widest;

    ctx->buf_a = (float*)cfd_aligned_malloc(ctx->buf_elems * sizeof(float));
    ctx->buf_b = (float*)cfd_aligned_malloc(ctx->buf_elems * sizeof(float));
    if (!ctx->buf_a || !ctx->buf_b) {
        cfd_nn_context_destroy(ctx);
        return CFD_ERROR_NOMEM;
    }
    *out_ctx = ctx;
    return CFD_SUCCESS;
}

void cfd_nn_context_destroy(cfd_nn_context_t* ctx) {
    if (!ctx) {
        return;
    }
    cfd_aligned_free(ctx->buf_a);
    cfd_aligned_free(ctx->buf_b);
    cfd_free(ctx);
}

const char* cfd_nn_context_backend(const cfd_nn_context_t* ctx) {
    if (!ctx || !ctx->impl) {
        return "none";
    }
    /* The SIMD table reports the arch actually selected at runtime, so a test
     * cannot pass while silently running scalar code. */
    if (ctx->impl == &cfd_nn_impl_simd) {
        return cfd_nn_simd_arch_name();
    }
    return ctx->impl->name;
}

/* ==========================================================================
 * Inference
 * ========================================================================== */

cfd_status_t cfd_nn_predict_batch(cfd_nn_context_t* ctx, size_t batch,
                                  const double* input, size_t input_count,
                                  double* output, size_t output_count) {
    if (!ctx || !input || !output || batch == 0) {
        return CFD_ERROR_INVALID;
    }
    const cfd_nn_model_t* m = ctx->model;
    if (batch > ctx->max_batch) {
        return CFD_ERROR_INVALID;
    }
    if (input_count != batch * m->inputs || output_count != batch * m->outputs) {
        return CFD_ERROR_INVALID;
    }

    /* f64 -> f32 at the boundary; weights and arithmetic are f32. */
    float* src = ctx->buf_a;
    float* dst = ctx->buf_b;
    for (size_t i = 0; i < input_count; i++) {
        src[i] = (float)input[i];
    }

    for (size_t li = 0; li < m->layer_count; li++) {
        ctx->impl->dense(&m->layers[li], batch, src, dst);
        float* tmp = src;
        src        = dst;
        dst        = tmp;
    }

    /* A corrupt or badly scaled model must fail loudly rather than seed NaN
     * into a flow field, so the result is checked before it is handed back. */
    for (size_t i = 0; i < output_count; i++) {
        if (!isfinite(src[i])) {
            return CFD_ERROR_DIVERGED;
        }
        output[i] = (double)src[i];
    }
    return CFD_SUCCESS;
}
