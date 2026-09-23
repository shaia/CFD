#ifndef CFD_NN_CFDNN_H
#define CFD_NN_CFDNN_H

#include "cfd/cfd_export.h"

#include "cfd/core/cfd_status.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file cfdnn.h
 * @brief Pure-C inference for small pointwise networks (no runtime dependencies).
 *
 * Evaluates a pre-trained multilayer perceptron stored in the portable `.cfdnn`
 * binary format. There is no training, no autodiff and no graph optimizer: this
 * is an evaluator for a fixed, validated op set.
 *
 * The motivating consumer is a data-driven turbulence closure, which maps a
 * handful of local flow invariants to a scalar correction at every cell, every
 * step. That shape -- many independent evaluations of a small network -- is why
 * the batch entry point is primary and why only Dense layers are supported.
 * Convolutions are deliberately absent: they solve a different problem (fields
 * with spatial structure) and would be dead weight here. See
 * docs/technical-notes/ml-integration-design.md for why this shape was chosen.
 *
 * Supported layers: Dense (fully connected) with an optional fused activation.
 * Batch normalization is NOT supported and is rejected at load: fold it into the
 * preceding layer's weights at export time, which is exact.
 *
 * Weights are IEEE-754 binary32 and arithmetic is performed in binary32. The
 * public API is binary64 to match the rest of the library; conversion happens at
 * the boundary. A closure correction is clamped by the caller and fed to a
 * model that is itself an approximation, so f32 is not the limiting error.
 *
 * Threading: a model is immutable once loaded and may be shared by any number of
 * threads. A context owns mutable scratch and must NOT be shared; create one per
 * thread. This split is deliberate -- folding the scratch into the model would
 * make inference non-thread-safe the first time it was called from inside an
 * OpenMP region.
 */

/** Current on-disk weight format version. Bumped on any layout change. */
#define CFD_NN_FORMAT_VERSION 1u

/** Recommended file extension for model files. */
#define CFD_NN_EXTENSION ".cfdnn"

/** Execution backend. AUTO resolves SIMD > OMP > SCALAR at context creation. */
typedef enum {
    CFD_NN_BACKEND_AUTO   = 0,
    CFD_NN_BACKEND_SCALAR = 1, /**< Portable reference; always available. */
    CFD_NN_BACKEND_SIMD   = 2, /**< AVX2 or NEON, selected by runtime detection. */
    CFD_NN_BACKEND_OMP    = 3  /**< OpenMP over the batch axis. */
} cfd_nn_backend_t;

/** Layer kinds. Values are on-disk encoding and must never be renumbered. */
typedef enum {
    CFD_NN_LAYER_DENSE = 1
} cfd_nn_layer_kind_t;

/** Fused activations. Values are on-disk encoding and must never be renumbered. */
typedef enum {
    CFD_NN_ACT_IDENTITY   = 0,
    CFD_NN_ACT_RELU       = 1,
    CFD_NN_ACT_LEAKY_RELU = 2, /**< act_param = negative slope */
    CFD_NN_ACT_TANH       = 3,
    CFD_NN_ACT_SIGMOID    = 4,
    CFD_NN_ACT_SOFTPLUS   = 5  /**< log1p(exp(x)); the non-negative output layer */
} cfd_nn_activation_t;

typedef struct cfd_nn_model   cfd_nn_model_t;
typedef struct cfd_nn_context cfd_nn_context_t;

/* ------------------------------------------------------------------ model -- */

/**
 * Load a model from @p path.
 *
 * The whole file is read, validated and copied into freshly allocated storage;
 * the file is not retained. Validation is total -- magic, version, endianness,
 * CRC32, every declared size against a cap, and layer-to-layer shape agreement.
 * A model that loads is guaranteed executable without further shape checks.
 *
 * @param path       Source file path.
 * @param out_model  Receives the loaded model; set to NULL on failure.
 * @return CFD_SUCCESS, or:
 *   - CFD_ERROR_INVALID     bad magic, malformed record, shape mismatch, unknown
 *                           layer kind or activation, or a size beyond a cap;
 *   - CFD_ERROR_UNSUPPORTED format version mismatch, foreign endianness, or an
 *                           unsupported weight dtype;
 *   - CFD_ERROR_IO          open/read failure, truncation, or CRC mismatch;
 *   - CFD_ERROR_NOMEM       allocation failure.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_nn_model_load(const char* path,
                                                  cfd_nn_model_t** out_model);

/**
 * Load a model from a memory image, with identical semantics and identical
 * validation to cfd_nn_model_load(). Lets a model be embedded in a binary (the
 * test suite ships its golden model this way, honouring the convention that no
 * data files live under tests/). @p bytes is not retained.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_nn_model_load_memory(const void* bytes,
                                                         size_t size,
                                                         cfd_nn_model_t** out_model);

/** Free a model. NULL is safe. Destroy dependent contexts first. */
CFD_LIBRARY_EXPORT void cfd_nn_model_destroy(cfd_nn_model_t* model);

/** Number of input features the model expects per sample. */
CFD_LIBRARY_EXPORT size_t cfd_nn_model_inputs(const cfd_nn_model_t* model);

/** Number of output values the model produces per sample. */
CFD_LIBRARY_EXPORT size_t cfd_nn_model_outputs(const cfd_nn_model_t* model);

/** Model name recorded at export, or "" if none. Valid until destroy. */
CFD_LIBRARY_EXPORT const char* cfd_nn_model_name(const cfd_nn_model_t* model);

/* ---------------------------------------------------------------- context -- */

/**
 * Create an execution context bound to a maximum batch size and a backend.
 *
 * @param model       Model to execute (must outlive the context).
 * @param max_batch   Largest sample count a single predict call will pass.
 * @param backend     Requested backend; AUTO picks the best available.
 * @param out_ctx     Receives the context; set to NULL on failure.
 * @return CFD_SUCCESS, or:
 *   - CFD_ERROR_INVALID     NULL model or zero max_batch;
 *   - CFD_ERROR_UNSUPPORTED the requested backend is not compiled in or not
 *                           supported by this CPU. There is no silent fallback:
 *                           the caller chooses what to do, per the project rule.
 *   - CFD_ERROR_NOMEM       arena allocation failure.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_nn_context_create(const cfd_nn_model_t* model,
                                                      size_t max_batch,
                                                      cfd_nn_backend_t backend,
                                                      cfd_nn_context_t** out_ctx);

/** Free a context. NULL is safe. */
CFD_LIBRARY_EXPORT void cfd_nn_context_destroy(cfd_nn_context_t* ctx);

/** "scalar", "avx2", "neon" or "omp" -- the backend actually resolved. */
CFD_LIBRARY_EXPORT const char* cfd_nn_context_backend(const cfd_nn_context_t* ctx);

/** The max_batch the context was created with: the largest sample count a
 *  single predict call may pass. A caller with more samples than that must
 *  split them, which is what the turbulence closure does. 0 if ctx is NULL. */
CFD_LIBRARY_EXPORT size_t cfd_nn_context_capacity(const cfd_nn_context_t* ctx);

/** The model the context executes, so a consumer can check the shape it is
 *  about to feed before the first step. NULL if ctx is NULL. */
CFD_LIBRARY_EXPORT const cfd_nn_model_t* cfd_nn_context_model(const cfd_nn_context_t* ctx);

/* ---------------------------------------------------------------- predict -- */

/**
 * Evaluate the network over a batch of samples.
 *
 * @param ctx          Execution context.
 * @param batch        Number of samples; must not exceed the context's max_batch.
 * @param input        batch * cfd_nn_model_inputs() doubles, sample-major
 *                     (sample 0's features first, then sample 1's, ...).
 * @param input_count  Element count, checked against the expected size.
 * @param output       batch * cfd_nn_model_outputs() doubles, sample-major.
 * @param output_count Element count, checked.
 *
 * @return CFD_SUCCESS, CFD_ERROR_INVALID on a NULL argument, an oversized batch
 *         or a count mismatch, or CFD_ERROR_DIVERGED if any output is
 *         non-finite -- a corrupt or badly scaled model must fail loudly rather
 *         than seed NaN into a flow field.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_nn_predict_batch(cfd_nn_context_t* ctx,
                                                     size_t batch,
                                                     const double* input,
                                                     size_t input_count,
                                                     double* output,
                                                     size_t output_count);

/** Whether @p backend can be used on this build and this CPU. */
CFD_LIBRARY_EXPORT bool cfd_nn_backend_available(cfd_nn_backend_t backend);

/* ---------------------------------------------------------- serialization -- */
/* The library does not train models. The writer exists so that exporters, tools
 * and tests produce .cfdnn files with the same code the reader validates --
 * there is deliberately no second implementation of the format. */

/** One Dense layer. Weights are row-major [out][in], matching PyTorch's
 *  Linear.weight exactly, so an exporter needs no transpose. */
typedef struct {
    cfd_nn_layer_kind_t kind;
    cfd_nn_activation_t activation;
    float               act_param;   /**< leaky-relu slope; 0 when unused */
    size_t              in_features;
    size_t              out_features;
    const float*        weights;     /**< in_features * out_features, row-major */
    const float*        bias;        /**< out_features values, or NULL */
} cfd_nn_layer_desc_t;

/** Model description handed to the writer. */
typedef struct {
    const char*                name;        /**< may be NULL */
    const cfd_nn_layer_desc_t* layers;
    size_t                     layer_count;
} cfd_nn_model_desc_t;

/**
 * Write @p desc to @p path, running the same shape validation the reader runs so
 * an invalid model cannot be produced in the first place.
 * @return CFD_SUCCESS, CFD_ERROR_INVALID, CFD_ERROR_IO or CFD_ERROR_NOMEM.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_nn_model_write(const char* path,
                                                   const cfd_nn_model_desc_t* desc);

#ifdef __cplusplus
}
#endif

#endif /* CFD_NN_CFDNN_H */
