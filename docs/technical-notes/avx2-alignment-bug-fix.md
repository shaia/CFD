# AVX2 SIMD Context Struct Alignment Bug Fix

## Issue Summary

SIMD solver tests (Jacobi, Red-Black SOR, Projection) segfaulted on Linux CI when built with AVX2 enabled (`-DCFD_ENABLE_AVX2=ON`), while passing on Windows. The root cause was improper memory alignment for context structures containing `__m256d` AVX2 vector types.

## Affected Components

- `lib/src/solvers/linear/avx2/linear_solver_jacobi_avx2.c`
- `lib/src/solvers/linear/avx2/linear_solver_redblack_avx2.c`

## Symptoms

- Tests passed on Windows (MSVC) with AVX2 enabled
- Tests segfaulted on Linux (GCC) with AVX2 enabled via `-DCFD_ENABLE_AVX2=ON`
- Standard Linux builds without AVX2 worked correctly
- Affected tests: `PoissonJacobiSimdTest`, `PoissonRedBlackSimdTest`, `SolverProjectionTest`, `SolverProjectionSimdTest`, `LinearSolverTest`

## Root Cause Analysis

### The Problem

Both SIMD solvers defined context structures containing `__m256d` (AVX2 256-bit vector) members:

```c
typedef struct {
    double dx2;
    double dy2;
    double inv_factor;
    __m256d dx2_inv_vec;       // Requires 32-byte alignment
    __m256d dy2_inv_vec;       // Requires 32-byte alignment
    __m256d inv_factor_vec;    // Requires 32-byte alignment
    __m256d neg_inv_factor_vec; // Requires 32-byte alignment
    int initialized;
} jacobi_simd_context_t;
```

These structures were allocated using `cfd_calloc()`:

```c
jacobi_simd_context_t* ctx = (jacobi_simd_context_t*)cfd_calloc(1, sizeof(jacobi_simd_context_t));
```

### Why This Causes a Segfault

1. **AVX2 alignment requirements**: The `__m256d` type requires 32-byte (256-bit) alignment for certain operations.

2. **Standard allocator alignment**: `calloc()` (via `cfd_calloc()`) only guarantees alignment suitable for any standard C type (typically 8 or 16 bytes, depending on platform).

3. **Compiler code generation**: When the compiler accesses struct members of type `__m256d` (e.g., `ctx->dx2_inv_vec`), it may generate aligned load/store instructions (`vmovapd`) that require 32-byte alignment.

4. **Platform differences**:
   - **GCC on Linux**: More likely to generate aligned instructions (`vmovapd`) that crash on misaligned memory
   - **MSVC on Windows**: May use unaligned instructions or have different memory allocator behavior that happens to satisfy alignment by chance

### Alignment in Different Contexts

The use of `_mm256_loadu_pd` (unaligned load) and `_mm256_storeu_pd` (unaligned store) in the computation loop is correct and works with any alignment. However, the struct member access for the pre-computed vectors (`ctx->dx2_inv_vec`) uses the compiler's default access pattern, which assumes proper alignment for `__m256d`.

## The Fix

Replace `cfd_calloc()` with `cfd_aligned_calloc()` (32-byte aligned allocation) for context structures containing `__m256d` members, and use `cfd_aligned_free()` for deallocation.

### Jacobi SIMD Changes

**Before:**
```c
static cfd_status_t jacobi_simd_init(...) {
    jacobi_simd_context_t* ctx = (jacobi_simd_context_t*)cfd_calloc(1, sizeof(jacobi_simd_context_t));
    ...
}

static void jacobi_simd_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}
```

**After:**
```c
static cfd_status_t jacobi_simd_init(...) {
    /* Use aligned allocation for struct containing __m256d members */
    jacobi_simd_context_t* ctx = (jacobi_simd_context_t*)cfd_aligned_calloc(1, sizeof(jacobi_simd_context_t));
    ...
}

static void jacobi_simd_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_aligned_free(solver->context);
        solver->context = NULL;
    }
}
```

### Red-Black SIMD Changes

Identical pattern - replaced `cfd_calloc()` with `cfd_aligned_calloc()` and `cfd_free()` with `cfd_aligned_free()` in the init and destroy functions.

## Technical Background

### Memory Alignment for SIMD Types

| Type | Size | Required Alignment |
|------|------|-------------------|
| `double` | 8 bytes | 8 bytes |
| `__m128d` (SSE2) | 16 bytes | 16 bytes |
| `__m256d` (AVX2) | 32 bytes | 32 bytes |
| `__m512d` (AVX-512) | 64 bytes | 64 bytes |

### Platform-Specific Behavior

**Linux (GCC/Clang with `-mavx2`):**
- Compiler generates aligned instructions for `__m256d` struct members
- Misaligned access causes `SIGBUS` or `SIGSEGV`
- Stack is 32-byte aligned, but heap allocations from `malloc`/`calloc` are not guaranteed

**Windows (MSVC with `/arch:AVX2`):**
- MSVC may generate different instruction sequences
- Windows heap allocators sometimes provide 16-byte or better alignment
- May work by chance but is technically undefined behavior

### CFD Library Aligned Allocation Functions

The CFD library provides portable aligned allocation:

```c
// From lib/src/core/memory.c
void* cfd_aligned_malloc(size_t size);       // 32-byte aligned allocation
void* cfd_aligned_calloc(size_t count, size_t size);  // 32-byte aligned, zero-initialized
void cfd_aligned_free(void* ptr);            // Free aligned memory
```

Implementation details:
- **Linux/Unix**: Uses `posix_memalign()` with 32-byte alignment
- **Windows**: Uses `_aligned_malloc()` with 32-byte alignment

## Lessons Learned

1. **Always use aligned allocation for structs containing SIMD types**: Any structure with `__m128*`, `__m256*`, or `__m512*` members must be allocated with appropriate alignment.

2. **Unaligned loads/stores are not sufficient**: Even if you use `_mm256_loadu_pd` for data access, struct members with SIMD types need aligned allocation.

3. **Platform differences can mask bugs**: Code may work on one platform due to allocator behavior but fail on another. Always verify on multiple platforms.

4. **Use CI with multiple configurations**: The SIMD-specific CI job (`simd-test`) with `-DCFD_ENABLE_AVX2=ON` caught this issue that wouldn't appear in standard builds.

## Testing

After the fix, all tests pass on both platforms:

```
100% tests passed, 0 tests failed out of 27
```

Specifically verified:
- `PoissonJacobiSimdTest` - Uses Jacobi SIMD solver
- `PoissonRedBlackSimdTest` - Uses Red-Black SIMD solver
- `SolverProjectionSimdTest` - Uses SIMD projection method which calls SIMD Poisson solvers
- `LinearSolverTest` - Tests all linear solver variants including SIMD

## Related Files

- `lib/include/cfd/core/memory.h` - Aligned allocation function declarations
- `lib/src/core/memory.c` - Aligned allocation implementations
- `lib/src/solvers/linear/linear_solver_internal.h` - SIMD detection macros
- `.github/workflows/build.yml` - CI configuration with `simd-test` job

## References

- [Intel Intrinsics Guide - _mm256_loadu_pd](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_pd)
- [GCC x86 Options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)
- [MSVC /arch (x64)](https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64)
- [posix_memalign(3) - Linux manual page](https://www.man7.org/linux/man-pages/man3/aligned_alloc.3.html)
