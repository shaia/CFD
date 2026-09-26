/**
 * @file cfdnn_kernels_omp.c
 * @brief OpenMP backend for `.cfdnn` inference.
 *
 * Parallelises the SAMPLE axis only. That axis is not reduced -- each sample's
 * accumulation over input features is untouched and runs in the same ascending
 * order as the scalar reference -- so this backend is BIT-IDENTICAL to scalar,
 * and the cross-backend test asserts exact equality rather than a tolerance.
 * If that test ever fails it means someone parallelised a reduction axis,
 * which is precisely what it exists to catch.
 *
 * This file compiles with or without OpenMP. Without it the table exports a
 * NULL kernel, so the backend reports itself unavailable instead of quietly
 * running serial code under the name "omp" -- a silent cross-backend fallback
 * is exactly what the project's dispatch rules forbid. It must keep compiling
 * either way: cfdnn.c references cfd_nn_impl_omp unconditionally, so dropping
 * this translation unit from a no-OpenMP build would leave that symbol
 * undefined and fail the link for the whole library.
 */

#include "../cfdnn_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CFD_ENABLE_OPENMP

static void dense_omp(const cfd_nn_layer_t* l, size_t batch, const float* in,
                      float* out) {
    const size_t ni = l->in_features;
    const size_t no = l->out_features;

    /* Signed loop variable declared outside the for-init: MSVC's OpenMP
     * rejects a declaration there (error C3015). Matches the idiom used by
     * the turbulence and boundary OMP kernels. */
    const int n = (int)batch;
    int       s;

#pragma omp parallel for schedule(static)
    for (s = 0; s < n; s++) {
        const float* x = in + (size_t)s * ni;
        float*       y = out + (size_t)s * no;
        for (size_t o = 0; o < no; o++) {
            const float* w   = l->weights + o * ni;
            float        acc = l->bias ? l->bias[o] : 0.0f;
            for (size_t i = 0; i < ni; i++) {
                acc += x[i] * w[i];
            }
            y[o] = acc;
        }
        cfd_nn_apply_activation(l->activation, l->act_param, y, no);
    }
}

const cfd_nn_backend_impl_t cfd_nn_impl_omp = {
    "omp",
    dense_omp,
};

#else /* !CFD_ENABLE_OPENMP */

const cfd_nn_backend_impl_t cfd_nn_impl_omp = {
    "omp",
    NULL, /* unavailable: reported, never silently substituted */
};

#endif /* CFD_ENABLE_OPENMP */
