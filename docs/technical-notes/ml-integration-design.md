# ML Integration Design: A Learned Eddy-Viscosity Correction

## Context

`ROADMAP.md` Phase 7 has carried an ML integration section since before v0.2.0, ending in an
unmade decision: *"Two approaches are under consideration — pick one before implementing."*
This note makes that decision, records the reasoning, and corrects the roadmap where
investigation showed it was wrong.

**Audience:** library contributors and future maintainers. This is not user documentation —
for the API, see the public header `lib/include/cfd/nn/cfdnn.h`.

**Why this note exists:** the most valuable output of the investigation was not the design
that was chosen but the design that was *rejected*, and why. Without that written down, the
rejected design will be re-proposed roughly annually, because it is the obvious idea.

**What was decided:**

| Question | Answer |
| -------- | ------ |
| Approach | **A — full C inference**, zero runtime dependencies |
| First consumer | **A learned eddy-viscosity correction** |
| Not the first consumer | An ML-predicted pressure Poisson initial guess |
| Architecture | Pointwise MLP, ~6 local invariants per cell → one scalar |

---

## Part 1 — Why

### 1.1 The rule that governs ML in this library

This repository has twice replaced a tuned constant with something better, and both times
the answer was analytic, not learned:

- A hard-coded `omega = 1.5` became `poisson_solver_resolve_omega()` in
  `lib/src/solvers/linear/linear_solver_internal.h` — Young's formula, derived from the
  grid's Jacobi spectral radius.
- Iteration-count tuning for the pressure solve became geometric multigrid, which converges
  at a fixed rate per cycle regardless of grid size.

That history establishes the test every future ML proposal must pass:

> **ML earns its place only where no analytic or algorithmic answer exists.**

A learned model that competes with a provably optimal algorithm loses — not on accuracy, on
cost, maintenance, and the fact that its failure modes are unbounded while the algorithm's
are proven. The rule is not conservatism; it is the reason the SOR ω fix was a good change
and a learned ω would have been a bad one.

### 1.2 Why the obvious first application was rejected

The intuitive first use of a neural network in a Navier-Stokes solver is to predict the
pressure field and hand it to the Poisson solver as an initial guess. It is attractive
because it looks *safe*: conjugate gradient still iterates to its tolerance, so a bad
network appears to cost iterations rather than accuracy.

It was rejected on three independent findings, each verified in the source.

#### Finding 1 — Under this library's stopping rule, a better guess cannot buy iterations

`lib/src/solvers/linear/cpu/linear_solver_cg.c`, the convergence target:

```c
double initial_res = sqrt(dot_product(r, r, ...));   /* ||r0||, from the initial guess */
...
double tolerance = params->tolerance * initial_res;
if (tolerance < params->absolute_tolerance) {
    tolerance = params->absolute_tolerance;
}
```

The target is `max(tol_rel · ‖r₀‖, abs_tol)`, and `‖r₀‖` is measured **from the initial
guess**. A guess that halves `‖r₀‖` halves the target with it. CG is asked for the same
*relative* reduction no matter where it starts, and its iteration count under a relative
criterion is governed by the operator's spectrum, not by the error's magnitude.

The `absolute_tolerance` floor (default `1e-10`) only takes over once
`tol · ‖r₀‖ < 1e-10`, i.e. `‖r₀‖ < 1e-4` at the default `tol = 1e-6` — far below a typical
pressure residual. The relative regime dominates in practice.

On the GPU path it is not merely dominant but absolute.
`lib/src/solvers/gpu/solver_projection_gpu.cu` passes the absolute tolerance as zero, and
says so:

```c
// The relative tolerance / iteration cap come from the
// GPU config; absolute_tolerance is left at 0 so the relative test governs.
cg_gpu_solve_params cgp = {cfg.poisson_tolerance, 0.0, cfg.poisson_max_iter};
```

There is no floor at all there.

**The practical corollary, which applies to any future solver-acceleration claim:**
comparing raw iteration counts between arms with different initial guesses measures the
moving target, not the work. Every arm must be driven to the *same absolute* residual.
Setting `params.tolerance = 0.0` collapses the expression above to
`tolerance = absolute_tolerance`, which makes the target arm-independent; `poisson_solver_init`
does not validate `tolerance`, so this needs no library change.

#### Finding 2 — A local network removes exactly the error that was already cheap

CG on a Poisson operator converges quickly on high-frequency error and slowly on smooth,
low-frequency error; the iteration count is set by the smallest eigenvalues. A stack of 3×3
convolutions has a receptive field of a few cells, so it can only see locally determined —
that is, high-frequency — structure.

So a local CNN removes precisely the error component that costs few iterations and leaves
precisely the one that costs many. The standard fix is a multi-scale U-Net, whose receptive
field grows as `2^L` with pooling depth `L`. But that is structurally a V-cycle: a U-Net for
an elliptic problem *is* multigrid, learned approximately instead of derived exactly.

This is the same objection as Finding 1 restated spectrally, and it is why the two are not
independent failures but one.

#### Finding 3 — The headroom was already banked, by non-ML means

`CHANGELOG.md` records the multigrid-preconditioned CG result:

> 5 iterations at 33²–129² (tol 1e-8) vs 50–170 unpreconditioned.

Five iterations, independent of grid size. Meanwhile
`lib/src/solvers/navier_stokes/cpu/solver_projection.c` already warm-starts the solve:

```c
memcpy(p_new, field->p, bytes);   /* previous step's converged pressure */
```

so the incumbent initial guess costs nothing and is already in place on every backend.

An order-of-magnitude cost comparison at 129² (16,641 cells) completes the picture:

| Work | Approximate FLOPs |
| ---- | ----------------- |
| MG-PCG pressure solve, 5 iterations | ~3M |
| Plain CG, 50–170 iterations | ~33–110M |
| *One* 3×3 conv layer, 1→16 channels | ~5M |
| A 6-layer, 16–32-channel U-Net | ~50–200M |

A network large enough to be multi-scale — which Finding 2 says it must be — costs roughly
15–60× more arithmetic than the entire solve it is meant to accelerate, and pays that cost
every timestep.

#### What this leaves

Multigrid is absent from the AVX2 and GPU backends, which do use plain CG, so headroom does
exist there. But the fix for a missing multigrid backend is to port multigrid — already a
listed backend-coverage item with a proven correctness bar — not to approximate it with a
network. Using an unvalidatable model to paper over a missing port of a provably correct
algorithm is the worst available trade.

### 1.3 Why turbulence closure instead

The closure problem is the one place in this library where the governing rule of §1.1 points
*toward* ML rather than away from it. RANS closures are empirical: `TURB_C_MU = 0.09`,
`SA_CB1 = 0.1355` and their companions in
`lib/src/solvers/turbulence/turbulence_solver_internal.h` are fitted constants, not derived
ones. After fifty years there is still no analytic answer, which is exactly the condition
the rule requires.

Three further properties make it the right first consumer here specifically:

**The payoff is accuracy, not speed.** This inverts the cost problem. In §1.2 the network
had to be cheaper than the thing it replaced, and could not be. A closure correction has no
such constraint: it buys a better answer, and a 10–20% step-time overhead for a measurably
more accurate one is a trade users make willingly.

**The seam is a single function.** `turb_update_nu_t()` in
`lib/src/solvers/turbulence/cpu/turbulence_solver.c` is the sole definition of `nu_t` for
every backend — scalar, OMP and AVX2 all call this one scalar function:

```c
void turb_update_nu_t(flow_field* field, const ns_solver_params_t* params) {
    size_t total = field->nx * field->ny * field->nz;
    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        for (size_t n = 0; n < total; n++) {
            double nu = local_nu(params, field, n);
            ...
            field->nu_t[n] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
        }
    } else if (params->turb_model == TURB_MODEL_SPALART_ALLMARAS) {
        ...
    }
}
```

Because a learned viscosity carries **no transport equation**, the hand-duplicated k-ε and
SA transport kernels in `omp/turbulence_solver_omp.c` and `avx2/turbulence_solver_avx2.c`
are never touched. A new branch in this one function reaches three backends at once. The
momentum kernels already consume `nu_t` through an existing gate
(`turb_model != TURB_MODEL_NONE && nu_t != NULL`) and need no change at all.

**The architecture is a pointwise MLP.** Features are local invariants per cell — `|S|dx²/ν`,
`|Ω|/|S|`, `y⁺`, cell Reynolds number, `dx/L` — mapping to one scalar. No convolution, no
pooling, no skip connections. This is a ~400-parameter network, not a 14,000-parameter one.

### 1.4 Why Approach A, and three corrections to the roadmap

**Approach A (full C inference) over Approach B (Python + C kernels)** follows directly from
what the library is. The complete third-party surface today is Unity, fetched behind
`if(BUILD_TESTS)`; the library links nothing but libm, OpenMP and cudart. Approach B would
make a Python environment a *runtime* requirement of a C library whose selling point is that
it has no moving parts. The CI matrix that would have to carry it is five configurations
including Windows x86 and macOS ARM64.

Three corrections to Phase 7 as written follow from the same constraint:

**(A) Delete "+ JSON metadata".** There is no JSON parser in this repository. Adding one
means either the first third-party runtime dependency in project history or ~500 lines of
hand-rolled parser hardened against hostile input. A sidecar file also creates a two-file
consistency problem that `.cfdchk` deliberately does not have — nothing stops `model.cfdnn`
and `model.json` from disagreeing, and a CRC over one does not cover the other. Everything
JSON would carry becomes a length-prefixed record in the same byte stream under the same
CRC. The cost is that a human cannot `cat` the model; the answer is a small C dump tool that
shares the reader and therefore cannot drift from it.

**(B) Delete batch-norm and dropout from the layer list.** Dropout is identity at inference —
it is not a layer, it is something the exporter drops. BatchNorm folds exactly into the
preceding layer's weights:

```text
W' = W · γ/√(σ²+ε)
b' = γ(b−μ)/√(σ²+ε) + β
```

Folding is a few lines in the exporter and removes an entire layer kind, its kernel, its
tests, and the whole "running statistics or batch statistics?" bug class. The format should
**reject** a batch-norm layer kind rather than reserve one, so a non-folding exporter fails
loudly instead of silently producing a wrong model.

**(C) Keep "MLP (priority), Conv2D (future)" — it was right.** This is worth stating because
the rejected pressure application would have forced the opposite. A dense layer on a 64×64
field is 4096→4096 = 16.7M parameters and is locked to one grid size, so shipping one means
shipping a model zoo. A pointwise closure network has neither problem. The roadmap's
ordering survives because the first consumer changed.

### 1.5 Why the design is safe

The pressure-guess proposal was chosen partly for a safety property — a bad network costs
iterations, not accuracy. A closure correction does change physics; that is the point. So
safety has to be structural instead, enforced in C rather than hoped for:

1. **The multiplier is bounded on both sides.** This point originally read "a softplus output
   layer means the correction can only *add* dissipation, so it cannot destabilize the
   momentum solve." **That argument is withdrawn**, and it was withdrawn on the evidence in
   §2.5: k-ε *over*-predicts turbulent energy in the outer layer, so the correction the DNS
   asks for is β < 1. A dissipation-only correction could not represent the deficiency this
   work exists to fix. The implementation therefore clamps β to
   `[TURB_CLOSURE_BETA_MIN, TURB_CLOSURE_BETA_MAX]` = `[0.1, 10]`: a wrong model can scale
   `nu_t` by at most ten either way, never to zero and never negative. Stability rests on
   that floor, not on one-sidedness — at β = 0.1 the momentum equation sees a
   laminar-viscosity-dominated limit it already handles with the turbulence model switched
   off. A softplus output layer is still the right choice at export time, because it keeps
   the raw prediction positive before the clamp ever sees it.
2. **The existing clamp still applies.** `fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu)` is already
   in `turb_update_nu_t` and is not bypassed. The learned path inherits the same
   realizability bound the analytic models have.
3. **Non-finite output is an error, not a value.** Inference returns `CFD_ERROR_DIVERGED` if
   any output is non-finite, so a corrupt model fails the step loudly rather than seeding
   NaN into a flow field. The correction propagates that status out of the turbulence step,
   so the step itself fails; it never drops the correction and reports success.
4. **Correction ≡ 0 must be bit-identical to today.** This is a test, not an aspiration:
   `tests/solvers/turbulence/test_nut_correction.c` asserts memory equality between a run
   with no closure and one whose model predicts exactly 1.0.

---

## Part 2 — How

### 2.1 The `.cfdnn` format

Modelled line-for-line on `lib/src/io/checkpoint.c`, which already solved this problem for
`.cfdchk`: a latching I/O struct, explicit little-endian fixed-width encoding (never a raw
`fwrite` of a struct), an endianness marker, a table-less CRC32, and a format version that
**rejects** unknown values rather than guessing. `put_f32`/`get_f32` are added alongside the
existing `put_f64`/`get_f64`, using `memcpy` on the bit pattern for the same alias-safety
reason.

```text
 0   8  magic "CFDNN\0\0\0"
 8   4  format_version u32 = 1       (mismatch -> CFD_ERROR_UNSUPPORTED)
12   4  endian_marker  u32 = 0x01020304
16   6  lib_version    major/minor/patch u16 x3
22   2  flags          u16  bit0 = trailing CRC32 present; other bits rejected
24   1  dtype          u8   1=f32 (only supported), 2=f64 (reserved)
25   1  tensor_layout  u8   0 = row-major, the only layout the kernels index
26   2  reserved       u16  must be 0
28   4  layer_count    u32
32   4  reserved       u32  must be 0
36   4  reserved       u32  must be 0
40  ..  name (u32 len + bytes), then the layer records
 ..  4  crc32 over all preceding bytes
```

Dense layer record:

```text
kind u16 | activation u16 | act_param f32 | in_features u32 | out_features u32
        | weight_count u32 | f32[weight_count] | bias_count u32 | f32[bias_count]
```

Two choices worth justifying:

**Weights are row-major `[out][in]`** — byte-for-byte PyTorch's `Linear.weight` shape. The
exporter is therefore `w.detach().cpu().numpy().astype('<f4').tobytes()` with no transpose,
which removes an entire class of "the model loads and runs but produces garbage" bug.

**Arithmetic is float32 with a float64 API boundary.** PyTorch trains in f32; storing f64
would double the file to record noise the training never produced. AVX2 carries 8 floats per
vector against 4 doubles. The dtype is nevertheless a *field*, and the loader returns
`CFD_ERROR_UNSUPPORTED` for anything but f32 — `checkpoint.c`'s "reject unknown, never
guess" rule applied to precision.

Robustness caps (`NN_MAX_LAYERS`, `NN_MAX_FEATURES`, `NN_MAX_WEIGHTS`) are checked *before*
the corresponding allocation, with size arithmetic in `uint64_t` before narrowing. The loader
parses a file that may be hostile.

### 2.2 Module layout and dispatch

```text
lib/include/cfd/nn/cfdnn.h                      public API + normative format contract
lib/src/nn/cfdnn.c                              dispatcher: lifecycle, backend table, predict
lib/src/nn/cfdnn_internal.h                     tensor/arena structs, backend impl table
lib/src/nn/cfdnn_format.c                       reader + writer + validation
lib/src/nn/cpu/cfdnn_kernels_scalar.c           reference kernels
lib/src/nn/simd/cfdnn_simd_dispatch.c           runtime AVX2/NEON selection
lib/src/nn/avx2/cfdnn_kernels_avx2.c            macros + #include template
lib/src/nn/neon/cfdnn_kernels_neon.c            macros + #include template
lib/src/nn/simd_template/cfdnn_kernels_simd_template.h
lib/src/nn/omp/cfdnn_kernels_omp.c
```

`cfd/nn/` is a sibling of `cfd/io/` and `cfd/boundary/`, not under `cfd/solvers/` — it is not
a solver, and nesting it there would drag in the `<category>/<backend>/` rule for no reason.
The layout is otherwise exactly `boundary/`: technology-agnostic dispatcher at the module
root, backend subfolders as siblings, `simd/` holding only the dispatch file.

Naming the template directory `simd_template` is deliberate — `.clangd`'s existing
`PathMatch: .*/simd_template/.*` suppression already covers it, so no `.clangd` edit is
needed.

**Dispatch is a function-pointer table, not the string registry.** The project's two-level
dispatch splits on what varies: the named factory registry exists for user-selectable
*algorithms* with different numerical behaviour ("rk2" versus "rk4"), while
`bc_backend_impl_t` exists for the *same math on different hardware*. NN kernels are the
latter. A per-op `NULL` then means "this backend lacks this op", checked once at context
creation, so an unsupported model fails with `CFD_ERROR_UNSUPPORTED` early and loudly rather
than mid-prediction. No fallbacks, per the project rule.

**Model and execution scratch are separate objects.** A `cfd_nn_model_t` is immutable after
load and shareable across threads; a `cfd_nn_context_t` owns the arena and is
single-threaded. Folding the arena into the model would make prediction non-thread-safe on a
shared model — a violation of the project's thread-safety rule that would surface the first
time it was called from inside an OMP region.

### 2.3 The kernel decision that matters: vectorize across cells

The obvious way to vectorize a dense layer is to reduce over input features into a vector
accumulator and horizontal-sum at the end. That makes the accumulation order depend on the
vector width, which destroys cross-backend reproducibility.

Assigning each SIMD lane a distinct **cell** instead means the accumulation order over
features is identical to scalar, and there is no cross-lane reduction anywhere in the
kernel. The closure evaluates one network per cell — a 129² grid is 16,641 independent
evaluations per step — so the cell axis is both the natural batch axis and wide enough to
vectorize over.

Consequences, all of which are contracts the tests enforce:

- **OMP is bit-identical to scalar.** OMP parallelizes over the cell axis, which is not
  reduced, so each cell's arithmetic is untouched. The test asserts exact memory equality,
  not a tolerance. A failure there means someone parallelized a reduction axis — which is
  precisely what we want to catch.
- **SIMD differs from scalar only by FMA contraction**, around 1e-7 relative. The test
  tolerance is 1e-5 and the observed maximum is *printed*, so drift toward the tolerance is
  visible before it becomes a failure.
- Vector loop plus scalar remainder tail, per the established SIMD rule.
- Transcendental activations fall back to the scalar tail rather than growing a
  polynomial-approximation dependency; they are a vanishing fraction of the FLOPs and
  hand-rolled approximations would break cross-backend agreement for no measurable gain.

One deliberate deviation from existing code: the AVX2 file is guarded on `CFD_HAS_AVX2`
**alone**. `linear_solver_bicgstab_avx2.c` couples its guard to `CFD_ENABLE_OPENMP`, which
means a build without OpenMP silently loses vectorization. That is a latent wart; new code
should not replicate it.

### 2.4 Integration at `turb_update_nu_t()`

**As built, and where it differs from the plan above.** No
`TURB_MODEL_LEARNED_VISCOSITY` enumerator was added. The correction is not a model of its
own but a multiplier applied *on top of* k-ε, which is what the locality study of §2.6
actually fits (`beta = k_dns / k_model`), so the switch is the presence of
`params.turb_closure` rather than a model selection. Concretely:

1. `turb_apply_learned_correction()` runs immediately after `turb_update_nu_t()` in all three
   turbulence backends (scalar, OMP, AVX2), which keeps `turb_update_nu_t` unchanged for
   every caller that does not set a closure.
2. It walks the grid in tiles of `TURB_CLOSURE_TILE` cells, calling `cfd_nn_predict_batch`
   per tile. Tiling is what keeps the scratch a fixed stack buffer rather than a per-step
   allocation, and it means the context need not be sized to the grid.
3. Each predicted β is clamped to `[0.1, 10]` and multiplied into `nu_t`, which then passes
   through the **existing** `TURB_NU_T_MAX_FACTOR * nu` realizability clamp.
4. The model handle lives on `ns_solver_params_t`, following the established
   `source_func`/`source_context` callback-plus-context precedent. (`solver-params-redesign.md`
   notes that struct is already 24+ fields; this does not block the work, but the field
   belongs in a future grouped struct.)

Rejections, with no silent fallbacks anywhere. Every one of these is covered by
`tests/solvers/turbulence/test_nut_correction.c`:

| Configuration | Result |
| ------------- | ------ |
| GPU backend, any turbulence model | already `CFD_ERROR_UNSUPPORTED` |
| 3D or non-uniform grid | already `CFD_ERROR_UNSUPPORTED` |
| Closure set with any `turb_model` but k-ε, including `TURB_MODEL_NONE` | `CFD_ERROR_UNSUPPORTED` at init, and again at the step |
| Closure whose model is not 3 inputs → 1 output | `CFD_ERROR_INVALID` at init, and again at the step |
| Model predicts a non-finite value | `CFD_ERROR_DIVERGED`, failing the step |

### 2.5 Validation

**Corrected after a failed first attempt — read this before designing the gate.**

The original plan gated this work on the coarse-grid accuracy budget of the lid-driven
cavity at Re=1000: measure Ghia RMS at 65² and 129², and treat the gap as the budget a
learned correction could recover. That gate was wrong, for a reason worth recording.

**The cavity at Re ≤ 1000 is steady laminar flow.** Ghia et al. computed steady solutions
there, and the 2D cavity does not become unsteady until Re ≈ 8000. So the coarse-grid error
in that case is *discretization* error, not *turbulence-closure* error. An eddy-viscosity
correction trained against it would not be learning a closure; it would be learning to mask
numerical error with artificial dissipation — a different and much weaker proposition, and
one that would fail the "must vanish in laminar regions" requirement below by construction.

The coarse-grid-correction protocol this was borrowed from applies to under-resolved
*turbulent* simulation. Applying it to a laminar case was the error.

The attempt also failed operationally, which is its own lesson: the 65² Re=1000 run was
killed after **8 hours** without reaching the kinetic-energy steady-state criterion
(`|ΔKE|/KE < 1e-8`). Per-step cost grew sharply as the flow converged — consistent with
Finding 1 of this note, since a better warm start shrinks `‖r₀‖`, tightens the relative
Poisson target, and makes each solve *more* expensive near steady state. That hypothesis is
untested; logging Poisson iterations per step would settle it, and it may be a real defect
worth its own investigation.

**The correct gate is the turbulent channel, not the cavity.** `test_turbulent_channel` at
Re_τ = 395 is the library's only genuinely turbulent validated case, and it is cheap: a
16×21 grid, `dt = 0.002`, 5,000–40,000 steps, with a 600-second timeout. It also has a
recorded baseline to beat — 2.9% (k-ε) and 3.1% (SA) `u_τ` error. Sweep `CH_RE_TAU` over
{180, 395, 1000} and read the log-law errors the test already prints. If the error curve is
flat across Re_τ, the channel carries no training signal and a separated-flow benchmark is a
prerequisite — which is an honest answer, obtainable in minutes rather than hours.

**E6 result (measured, Release/AVX2).** The first attempt held nothing fixed but `Re_tau`
and appeared to show closure error growing 3% -> 15%. That was an artifact: the stock test
hard-codes both `ny` and `dt`, so refining `Re_tau` moved first-node y+ out of the wall
function's valid band AND drove the convective CFL from 0.3 to 1.5, where the explicit
scheme fails. Both constants now scale with `Re_tau` (defaults unchanged, bit-identical).
With y+ pinned near 39.4 and CFL pinned at its tuned value:

| Re_tau | ny  | u_tau err (k-eps) | u_tau err (SA) | peak log-law err | runtime |
| ------ | --- | ----------------- | -------------- | ---------------- | ------- |
| 395    | 21  | 2.9%              | 3.3%           | 2.4% / 3.5%      | 9 s     |
| 590    | 31  | 2.1%              | 2.1%           | 2.7% / 3.4%      | 61 s    |
| 1000   | 52  | 1.7%              | 1.7%           | 3.1% / 3.0%      | 181 s   |
| 1500   | 77  | 1.7%              | 1.7%           | 3.3% / 2.7%      | 572 s   |
| 2000   | 102 | 1.9%              | (timed out)    | 3.3%             | >900 s  |

**The gate fails, on its own terms.** It was written as: *"If the error curve is flat across
Re_tau, the channel carries no training signal."* Beyond Re_tau = 590 the u_tau error is flat
at 1.7-1.9%, and the earlier growth was numerical. There is no Reynolds-dependent closure
deficiency here to learn against.

**And the instrument is worse than inconclusive -- it is close to circular.** The test scores
the velocity profile against the log law, but the wall-function boundary condition *imposes*
log-law behaviour at the first node. The model is being graded on reproducing the thing it is
constructed to reproduce. Two corroborating signs:

1. k-epsilon and Spalart-Allmaras -- structurally very different closures, one two-equation
   and one one-equation -- agree on `u_tau` to four significant figures at Re_tau >= 590
   (0.9794, 0.9834). When the choice of closure does not move the metric, the metric is not
   measuring the closure.
2. The residual 1.5-3.5% deviation grows with y+ and is largest at the outer edge of the
   probe window, which is where the log law itself stops being valid and the wake begins.
   Part of that "error" is real physics the log law does not describe, so it cannot be
   attributed to the model.

**Consequence: this library currently has no way to measure turbulence-closure error.** The
open question from the start of this work -- what reference data a learned correction could
be trained and validated against -- now has a concrete answer: **external channel DNS**
(Kim, Moin & Moser at Re_tau = 180/395; Moser, Kim & Mansour at 590) is a prerequisite, not
an optimisation. RANS closures are typically several percent off DNS mean profiles and
substantially off in the Reynolds stresses, and none of that is visible against a log-law
yardstick.

Until that data is in the repo and wired into a comparison, the closure work has no
acceptance criterion and should not start. That is the honest state of the gate.

**Cost note:** runtime grows steeply with `Re_tau` (9 s -> 572 s from 395 to 1500, with 2000
exceeding a 900 s cap) because refining the grid shrinks `dt` and multiplies the step count.
Any routine sweep should stop at Re_tau = 1000.

**DNS reference imported, and the circularity is gone.** `tests/validation/channel_dns_reference.h`
now carries the MKM mean-velocity profiles (129 points each, downloaded from
turbulence.oden.utexas.edu and generated into C arrays programmatically -- not
transcribed). `test_turbulent_channel.c` reports RMS and worst-case error against
DNS alongside the log-law check. Note the nominal case labels are not the actual
Reynolds numbers: chan395 is Re_tau = 392.24 and chan590 is 587.19, and the solver
must be driven at the actual value or the comparison carries an offset.

Measured, Release/AVX2, driving the solver at each data set's true Re_tau:

| Re_tau | nodes compared | k-eps DNS-RMS | SA DNS-RMS | runtime |
| ------ | -------------- | ------------- | ---------- | ------- |
| 392.24 | 9              | 1.30%         | 1.76%      | 9 s     |
| 587.19 | 14             | 1.23%         | 1.58%      | 13 s    |

**The DNS comparison does what the log-law check could not: it resolves the
closure.** Against the log law, k-epsilon and Spalart-Allmaras agreed on u_tau to
four significant figures. Against DNS they separate cleanly and consistently --
k-epsilon is better by 0.35-0.45 percentage points at both Reynolds numbers. The
metric is now measuring the model rather than the boundary condition, and there
are two validated operating points, so a train/holdout split is finally possible.

**But on the mean profile the gate still does not pass.** The error is ~1.2-1.3%
(k-epsilon) and ~1.6-1.8% (SA), and it is flat -- slightly *decreasing* with
Re_tau. A learned correction would be chasing about 1.5% with no Reynolds
dependence to learn from. The channel mean profile is the case these models were
calibrated on, so this is close to their best behaviour; it is the wrong place to
look for a deficiency.

**The open question is now specific and answerable: Reynolds stresses.** Mean
velocity is where RANS does well; the turbulent stresses are where it does not.
The same archive publishes `.reystress` profiles (u'u'+, v'v'+, w'w'+, u'v'+) for
both cases, and k-epsilon models these through k and nu_t, so they can be compared
directly. Eddy-viscosity closures are known to misplace the TKE peak substantially
in a way the mean profile hides. **That measurement, not the mean profile, should
decide whether the closure work proceeds.** Until it is run, Milestones 1 and 2
stay blocked.

**GATE RESULT: PASS, on Reynolds stresses -- not on the mean profile.**

> **Revised by §2.7.** The gate passed on two quantities, `-uv+` and `k+`. Measuring the
> algebraic correction showed that the `-uv+` half does not survive: in this flow the mean
> momentum balance fixes the total shear stress, so `-uv+` is nearly closure-independent and
> its apparent error tracks the model's own discretisation error almost exactly. Read the
> `-uv+` column below as a discretisation diagnostic. The gate still passes, on `k+` alone.

`chan<N>.reystress` is now imported alongside the means, and the test compares
three quantities. Measured in Release/AVX2, each case driven at its true Re_tau:

| Re_tau | model   | u+ RMS | -uv+ RMS | k+ RMS | nodes |
| ------ | ------- | ------ | -------- | ------ | ----- |
| 392.24 | k-eps   | 1.30%  | 15.47%   | 14.96% | 9     |
| 392.24 | SA      | 1.76%  | 16.41%   | n/a    | 9     |
| 587.19 | k-eps   | 1.23%  | 13.06%   | 12.77% | 14    |
| 587.19 | SA      | 1.58%  | 12.47%   | n/a    | 14    |

**Reynolds-stress error is roughly ten times the mean-velocity error**, consistent
across both Reynolds numbers and both closures. The mean profile was hiding it:
at 1.2-1.8% and flat in Re_tau it looked like there was nothing to learn, and on
that basis this work was about to be stopped. There is a 12-16% deficiency, in
precisely the quantity an eddy-viscosity closure is responsible for.

The error also has learnable structure rather than being noise: k+ error grows
monotonically toward the centreline (7.5% at y+ = 39 to 27.6% at y+ = 353 for
k-epsilon at Re_tau = 392), i.e. the model systematically over-predicts turbulent
energy in the outer layer.

**Caveats that must travel with these numbers:**

1. The modelled shear stress is `nu_t * dU/dy` with a central difference on a
   21-point grid, which is crude. One point (y+ = 78.4) reports -uv+ = 1.105,
   above the physical total stress there (~0.8), so per-point shear values are
   suspect even though the aggregate is not. Refining the derivative, or
   comparing on a finer grid, should come before any model is trained on it.
2. k+ at the first node is pinned at 3.33 = 1/sqrt(C_mu) by the wall function's
   equilibrium assumption. That point measures the wall treatment, not the
   transport equations.
3. The DNS k+ peak (4.57 at y+ ~ 17) lies *below* the first computational node
   (y+ ~ 39) and is never resolved -- wall functions bridge it by construction.
   So the classic near-wall closure failure is not being measured at all; what is
   measured is outer-layer error. A low-Re model or a wall-resolved grid would be
   needed to see the rest, and would likely show a larger deficiency still.
4. Two operating points is the minimum for a train/holdout split. A third would
   be better; Re_tau = 180 is available from the same archive but its first node
   falls below the wall function's valid band on any grid this coarse.

**Consequence: Milestones 1 and 2 are unblocked.** The acceptance metric is
Reynolds-stress error against MKM DNS, trained at one Re_tau and evaluated at the
other, with the mean profile retained only as a non-regression check.

**Prerequisite, now satisfied:** import channel DNS reference profiles (the same way
`cavity_reference_data.h` carries the Ghia tables) and add a DNS comparison alongside the
log-law check. Only then is there a measurable closure error, and only then can the ship
gate below be evaluated at all. Training data is the DNS; the library's own runs are the
thing being corrected, not the target.

Ship gate (evaluable only once the DNS reference exists):

- Reduces mean-profile and Reynolds-stress error against channel DNS relative to the
  recorded k-ε / SA baseline, on at least one Re_τ held out of training. Measured against
  DNS, never against the log law -- the wall function imposes the log law, so scoring
  against it is close to circular.
- Beats a tuned-`Cs` Smagorinsky baseline (`nu_t = (Cs·dx)²|S|`). This is the honest
  analytic competitor, and unlike the pressure case it is *known* to be mediocre — wrong at
  walls, wrong in 2D, nonzero where it should be zero.
- Does not regress laminar Poiseuille (analytic parabola), Taylor-Green (analytic decay), or
  MMS convergence order. Poiseuille is the direct test of the property Smagorinsky famously
  fails: **the correction must vanish in laminar regions.**
- Correction ≡ 0 is bit-identical to the current build.

All quantitative claims are measured in **Release**; a Debug measurement of an AVX2 path is
meaningless.

---

### 2.6 Locality study: is the correction learnable from local invariants?

Before committing to an inference engine, one question had to be answered: the k+
error collapses on y/delta (an outer-layer, NON-local coordinate), not on y+. If
the required correction is fundamentally a function of y/delta, then (a) an
analytic f(y/delta) captures it and the governing rule says use that, and (b) a
closure built from local invariants could not represent it at all and would fail
to generalise to any other geometry.

Method: export per-node local invariants (S* = |S|k/eps, Re_t = k^2/(nu eps),
nu_t/nu, y+) and the non-local y/delta, with target beta = k_dns/k_model. Fit on
Re_tau = 392.24 (9 nodes), predict Re_tau = 587.19 (14 nodes) -- a genuine
held-out Reynolds number, not a random split.

| model (fit 392, predict 587)              | train RMSE | test RMSE | test R2 |
| ----------------------------------------- | ---------- | --------- | ------- |
| no correction (beta = 1)                  | 0.1266     | 0.1088    | -0.34   |
| NON-LOCAL  [1, y/d, (y/d)^2]              | 0.0493     | 0.0741    | 0.38    |
| **LOCAL [1, ln S*, ln Re_t, ln nu_t+]**   | 0.0161     | **0.0522**| **0.69**|
| LOCAL + ln y+                             | 0.0066     | 0.3135    | -10.15  |
| LOCAL, ln S* only                         | 0.0235     | 0.0622    | 0.56    |

**Local invariants beat the non-local fit on held-out data.** The concern is
resolved in favour of a local closure: y/delta = 0.5 denotes different physics at
different Reynolds numbers, while S* is a proper local similarity variable
(correlation with beta = +0.83, the strongest of any feature). A correction built
from local invariants generalises across Re_tau; the geometry-dependent one does
not.

**But the same table argues against starting with a neural network.** A
four-parameter *linear* fit in log-features reaches R2 = 0.69, and S* alone
reaches 0.56. Adding a fifth parameter collapses to R2 = -10.15 on 9 training
points, so model capacity is hard-capped by the available data. By this note's own
governing rule -- ML only where no analytic answer exists -- the algebraic form
must be tried first.

**Revised plan.** Implement an algebraic correction beta = f(S*) inside
`turb_update_nu_t()` (a few coefficients, ~30 lines, no inference engine, no
weight format, no Python) and measure it against the DNS Reynolds-stress metric.
That is the competitor a network has to beat. Only if it saturates, and a network
clears it on a held-out Reynolds number, is Milestone 1 justified.

Caveats: 9 training points, two Reynolds numbers, one flow, one geometry, and the
shear-stress derivative issue of the previous section still applies. This is
suggestive, not settled.

---

### 2.7 The algebraic correction, measured — and two things §2.5 got wrong

§2.6 said the algebraic form had to be tried before a network. It has been, as
`NS_NUT_CORRECTION_S_STAR`, and measuring it corrected two errors in the gate above.

#### Correction 1 — `-uv+` error is mostly discretisation, not closure error

§2.5 reports a 12–16% Reynolds-shear-stress error and treats it as a closure deficiency.
It is largely not one. In fully developed channel flow the mean momentum balance fixes the
**total** shear stress, `tau_total/u_tau^2 = 1 - y/delta`, whatever the closure says. Any
converged model with the right `u_tau` therefore reproduces `-uv+` automatically; the
closure sets the mean velocity, not the stress.

Measured on the baseline run, the model's own stresses against that balance:

| Re_tau | RMS `-uv+` error vs DNS | RMS balance error vs `1 - y/delta` | DNS vs `1 - y/delta` |
| ------ | ----------------------- | ---------------------------------- | -------------------- |
| 392.24 | 15.5%                   | 15.2%                              | 3.6%                 |
| 587.19 | 13.1%                   | 11.0%                              | 4.8%                 |

The two columns track each other because they are the same quantity. One node dominates
both: the node adjacent to the wall-function node, where `nu_t` jumps and the central
difference straddles the jump, carries a 43% / 39% balance error and reports `-uv+ = 1.105`
where the physical total stress is ~0.8. That is caveat 1 of §2.5 promoted from a footnote
to the explanation.

**Consequence:** of the two quantities the gate passed on, only `k+` is a genuine closure
output. `-uv+` should be reported as a discretisation diagnostic, not as evidence for or
against a closure — and a correction that improves it is suspect rather than encouraging.

#### Correction 2 — the sign of the `nu_t` lever is inverted

The obvious mapping is "k-epsilon over-predicts k, so reduce `nu_t`". Measured, it does the
opposite. Multiplying `nu_t` by `beta` is using `C_mu' = beta * C_mu`; with the stress
pinned by the momentum balance the strain rate absorbs the change, so `P_k = tau^2 / nu_t`
and local equilibrium `P_k = eps` gives

```text
k = tau / sqrt(C_mu')     =>     k ~ beta^(-1/2)
```

**Lowering `nu_t` raises k.** The first a posteriori run confirmed it quantitatively:
`beta = 0.90` predicts +5.4% in k and the run gave +6.0%, and every DNS-RMS got worse —
`k+` 14.96% → 20.40% at the Reynolds number the fit was made on. The fitted TKE ratio must
therefore be inverted and squared, `beta_nu = beta_k^(-2)`, before it is applied.

#### The result, with the sign right

`beta = A (S*)^B`, `A = 1.6945`, `B = -0.2778`, from the least-squares fit of
`ln(k_dns/k_model)` on `ln S*` at Re_tau = 392.24 transformed by `beta_nu = beta_k^(-2)`.
Release/AVX2, each case driven at its true Re_tau, k-epsilon:

| Re_tau | | `u+` RMS | `-uv+` RMS | `k+` RMS | `u_tau` |
| ------ | --- | -------- | ---------- | -------- | ------- |
| 392.24 (fitted) | baseline | 1.30% | 15.47% | 14.96% | 0.9716 |
| 392.24 | corrected | 1.63% | 15.75% | **6.23%** | 0.9723 |
| 587.19 (held out) | baseline | 1.24% | 13.06% | 12.80% | 0.9807 |
| 587.19 | corrected | 1.54% | 13.07% | **9.27%** | 0.9807 |

**A 58% reduction in TKE error where it was fitted, and 28% at a held-out Reynolds
number.** `u_tau` is unmoved, so the friction calibration survives. The mean profile gives
up 0.3 percentage points, which is the expected trade: k-epsilon's constants are calibrated
on that profile, so improving k has to cost it something. `-uv+` does not move, exactly as
Correction 1 predicts it cannot.

**One refit iteration made it worse and was discarded.** Using the measured response to
re-fit (`beta_needed = beta_applied * (k_corr/k_dns)^2`, refit on the corrected run's own
S*) gave `A = 1.9948, B = -0.4364` and `k+` RMS 6.30% / 9.49% — worse at both Reynolds
numbers, with `u_tau` pulled to 0.9529. Nine training points do not support a second
iteration. The transformed first fit is kept.

#### What this does to the case for a network

It raises the bar it has to clear. The analytic competitor is now two constants that cut
TKE error by more than half and generalise to a held-out Reynolds number, and the governing
rule of §1.1 says a network must beat *that*, not the uncorrected model. It also narrows
the target: `-uv+` is not evidence, so the only quantity left to learn against on this case
is `k+`, at 9 and 14 nodes.

**What would change the picture** is a flow where the closure actually sets the stress
distribution, rather than one where the momentum balance pins it. A separated or
adverse-pressure-gradient case is now a prerequisite for the learned closure, for the same
reason DNS data was a prerequisite for measuring closure error at all.

---

## Part 3 — Limitations, stated up front

1. **This is the narrowest beachhead in the library.** Turbulence here is 2D-only,
   uniform-grid-only, and GPU-rejected, so the first ML capability lands in one physics
   module, one dimension, three of five backends. That is the price of putting it where ML
   actually wins. The broader alternative had a payoff of approximately zero, and a broad
   zero is still zero.
2. **The demonstration rests on an unmeasured assumption** — that the turbulent channel
   carries enough closure error across Re_τ to be worth learning against. See §2.5, and note
   that the first attempt at this gate used the wrong case entirely. Do not write inference
   code before measuring it.
3. **Cross-backend results are not bit-identical.** SIMD differs from scalar at FMA
   contraction (~1e-7). OMP *is* bit-identical by construction. Promise no more than that.
4. **Nothing in CI proves the Python exporter matches the C reader.** The exporter is
   developer tooling under `tools/cfdnn/`, outside the build and outside CI; adding a
   `setup-python` step would be a policy change needing its own discussion. The gap is
   narrowed — not closed — by a manual `cfdnn_check` verifier, by the format matching
   PyTorch's native layout and dtype so the serialization has almost no logic to get wrong,
   and by the runtime clamp of §1.5.
5. **Scope discipline.** This is not ONNX and must not grow into one. Every added op should
   have to justify itself against a named consumer in the solver, the same way multigrid and
   turbulence did.

## Findings filed separately

Two defects surfaced during this investigation that owe nothing to ML and should be fixed on
their own merits.

**The CG residual is computed twice per iteration.** In the unpreconditioned branch,
`rho_new = (r,r)` and `res_norm = sqrt((r,r))` are the same quantity computed by two separate
full passes over the residual vector — one of roughly five O(N) passes per iteration is
redundant. Present in all three CPU backends — `cpu/linear_solver_cg.c`,
`avx2/linear_solver_cg_avx2.c` and `omp/linear_solver_cg_omp.c`, each in the `else` branch
of the `use_precond` test (grep `rho_new = dot_product`; line numbers drift). The
preconditioned branch must keep its own `(r,r)`, since `rho_new = (r,z)` there.

**The wall function switches branches discontinuously.** `turbulence_wall_u_tau()` in
`lib/src/solvers/turbulence/cpu/turbulence_solver.c` uses the linear law below
`WALL_YPLUS_LAMINAR` (11.63) and a Newton-solved log law above it, with no buffer-layer
blending. With this library's `WALL_KAPPA = 0.41` and `WALL_B = 5.2`, the two laws do not
intersect exactly at 11.63, so `u_tau` jumps at the switch rather than merely kinking.
Spalding's law is the standard smooth alternative. **Measure the magnitude of the jump before
filing this as a bug** — it is arithmetic, not a hypothesis, but it has not been measured
yet, and the size determines whether it matters. Note also that `test_turbulent_channel` uses
this same function as its measurement instrument, so an analytic copy must be retained as the
yardstick if it is ever replaced.

## Explicitly not doing

- **Pressure-guess ML** — rejected on the evidence in §1.2.
- **A full flow surrogate.** Phase 7's "<5% L2 error vs CFD" is not a defensible acceptance
  bar in a library that validates with MMS convergence *order*, Ghia RMS to four decimals
  across backends, and bit-identical 3D degeneracy. 5% L2 is not a tolerance; it is a picture
  that looks about right.
- **A learned preconditioner.** Multigrid *is* the optimal preconditioner for this operator
  and is already implemented. CG's theory also requires a fixed SPD `M`, which a nonlinear
  network cannot provide without moving to flexible CG or GMRES — trading away CG's short
  recurrence for a preconditioner worse than the one already in the tree.
- **ML for the hand-tuned thresholds** (`gpu_config_t`, `BC_SIMD_THRESHOLD`,
  `OMP_THRESHOLD`). A measured lookup table beats an MLP and is auditable.
- **ML for adaptive dt or adaptive Poisson tolerance.** Embedded Runge-Kutta pairs with a PI
  controller, and Eisenstat-Walker forcing terms, are the analytic winners. Both are open
  roadmap items and neither needs a network.

## A note on the evidence

**Observed directly in the source at the time of writing:** the CG convergence target and its
`absolute_tolerance` floor; the GPU path passing `absolute_tolerance = 0.0`, with its own
comment confirming the intent; the projection warm-start `memcpy`; the full body of
`turb_update_nu_t`; the duplicated residual dot products in all three CPU CG backends; the
branch structure of `turbulence_wall_u_tau`.

**Documented, taken from the repository's own records:** the multigrid iteration counts
(5 at 33²–129² versus 50–170), from `CHANGELOG.md`; the k-ε and SA `u_tau` errors on the
Re_τ = 395 channel, from `ROADMAP.md`.

**Reconstructed, not measured:** the FLOP table in §1.2 is an order-of-magnitude estimate
from layer shapes and iteration counts, not an instrumented profile. It is used only to
establish a ratio of 15–60×, which is far outside the range where estimation error would
change the conclusion — but it is an estimate, and a profile would be better.

**Measured and discarded:** the lid-driven cavity coarse-grid budget at Re=1000. 33² gave
Ghia RMS_u 0.2130 / RMS_v 0.3062, converged in 5,272 steps. 65² was killed after 8 hours
without meeting the steady-state criterion, so no ratio was obtained — and the case was the
wrong one regardless, since that flow is laminar. See §2.5.

**Measured, and it failed the gate:** the channel closure budget of §2.5. Across
Re_τ = 395–2000 with y⁺ and CFL both held fixed, u_τ error is flat at 1.7–1.9% beyond
Re_τ = 590, and k-ε and Spalart-Allmaras agree to four significant figures — so the metric
is not resolving the closure. The apparent 3% → 15% growth in the first attempt was a
timestep artifact. **This blocks Milestones 1 and 2.**

**Not yet established:** whether the Poisson solve's per-step cost really does grow near
steady state via the Finding 1 mechanism (the 8-hour cavity run is suggestive, not
evidence); and the magnitude of the wall-function discontinuity. Both are cheap to measure
and neither has been.

**What I would still change:** the roadmap's Phase 7 success criteria were written for a
surrogate model and survive into this design unamended. They should be rewritten around the
closure acceptance gate in §2.5, and the "~1000× faster inference" framing should go — it was
never achievable against a warm-started, multigrid-preconditioned projection method, and
leaving it in the roadmap invites the rejected design back.

---

**See also:** `docs/reference/nn-inference.md` (API reference) ·
`docs/reference/solvers.md` (turbulence models) ·
`lib/src/io/checkpoint.c` (the binary-format precedent) ·
`ROADMAP.md` Phase 7
