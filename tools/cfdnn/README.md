# tools/cfdnn — Python side of the `.cfdnn` format

Developer tooling for getting a trained network into the solver's learned
eddy-viscosity closure (`ns_solver_params_t.turb_closure`). It sits outside the
build and outside CI; the one CI check on its output is
`tests/nn/test_cfdnn_python_export.c`. Background and the case for (and
against) a learned closure: [ml-integration-design.md](../../docs/technical-notes/ml-integration-design.md).

Requires Python 3.9+ and numpy. PyTorch is optional and never imported.

| file | purpose |
| ---- | ------- |
| `cfdnn.py` | writer, validating reader, float64 reference `predict()`, normalization/BatchNorm folding, PyTorch `Sequential` adapter |
| `distill_algebraic.py` | trains the pipeline's reference model and regenerates `tests/nn/cfdnn_python_golden.h` |
| `test_cfdnn.py` | `py -3 -m unittest tools/cfdnn/test_cfdnn.py` |

## Exporting a PyTorch model for the closure

The closure feeds the network three raw features per cell and multiplies
`nu_t` by its single output (then clamps it to [0.1, 10]):

| input | feature |
| ----- | ------- |
| 0 | `ln S*` = ln(\|S\| k / eps) |
| 1 | `ln Re_t` = ln(k² / (nu eps)) |
| 2 | `ln (nu_t / nu)` |

The output multiplies `nu_t`, not `k`. With the stress fixed, `k ~ beta^(-1/2)`,
so a target fitted as `k_dns / k_model` must be transformed to
`beta = (k_dns / k_model)^(-2)` first (design note §2.7).

```python
import sys; sys.path.insert(0, "tools/cfdnn")
import cfdnn

model.eval()                                   # required; BatchNorm uses running stats
layers = cfdnn.from_torch_sequential(model)    # Linear / activations / BatchNorm1d / Dropout
layers = cfdnn.fold_input_normalization(layers, x_mean, x_std)  # if trained on standardized x
cfdnn.write("closure.cfdnn", layers, name="my-closure")
```

Supported modules: `Linear`, `ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`,
`Softplus` (beta = 1), `BatchNorm1d` directly after a `Linear` (folded, exact),
and `Dropout` / `Identity` (dropped). Anything else is refused, not skipped.
Use `Softplus` on the output so the multiplier is positive by construction.

On the C side:

```c
cfd_nn_model_t* model;   cfd_nn_model_load("closure.cfdnn", &model);
cfd_nn_context_t* ctx;
/* SIMD where the CPU has it, else scalar -- deliberately not AUTO, which would
 * pick OpenMP on a build without AVX2/NEON. */
if (cfd_nn_context_create(model, 256, CFD_NN_BACKEND_SIMD, &ctx) == CFD_ERROR_UNSUPPORTED) {
    cfd_nn_context_create(model, 256, CFD_NN_BACKEND_SCALAR, &ctx);
}
params.turb_model   = TURB_MODEL_K_EPSILON;
params.turb_closure = ctx;               /* not together with turb_nut_correction */
```

The closure evaluates the network in tiles of 256 cells, so a larger capacity
buys nothing. At that size an OpenMP context is slower than scalar: it opens a
parallel region per layer per tile, and the region costs more than the work
(design note §2.3).

## Regenerating the test fixture

```sh
py -3 tools/cfdnn/distill_algebraic.py --out build/beta_s_star_distilled.cfdnn \
      --c-header tests/nn/cfdnn_python_golden.h
```

The run is deterministic for a given seed. After regenerating, rebuild and run
`test_cfdnn_python_export`, and re-run `test_cfdnn.py`, which checks the
header's embedded expectations against the bytes.
