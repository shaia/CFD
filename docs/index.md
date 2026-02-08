# CFD Framework Documentation

Welcome to the CFD Framework documentation. This page provides a comprehensive guide to all available documentation.

## Quick Links

- **[README](../README.md)** - Project overview and quick start
- **[Building](guides/building.md)** - Build instructions for all platforms
- **[Examples](guides/examples.md)** - Example programs and usage patterns
- **[API Reference](reference/api-reference.md)** - Complete API documentation
- **[Solvers](reference/solvers.md)** - Numerical methods and performance
- **[Architecture](architecture/architecture.md)** - Design principles and patterns

## Documentation Structure

### Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| [README](../README.md) | Quick start, installation, basic usage | All users |
| [Building](guides/building.md) | Detailed build instructions, platform-specific notes | Developers |
| [Examples](guides/examples.md) | Example programs with code walkthrough | All users |

### Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [API Reference](reference/api-reference.md) | Complete function and type reference | Developers |
| [Solvers](reference/solvers.md) | Numerical methods, algorithms, performance | Researchers, advanced users |
| [Architecture](architecture/architecture.md) | Design principles, patterns, best practices | Contributors |

### Specialized Topics

| Document | Description | Audience |
|----------|-------------|----------|
| [Validation](validation/) | Benchmark results and validation | Researchers |
| [SIMD Optimization](technical-notes/simd-optimization-analysis.md) | AVX2/NEON implementation details | Performance engineers |

## Documentation by Role

### For New Users

1. Start with [README](../README.md) for project overview
2. Follow [building instructions](guides/building.md#quick-start)
3. Run [minimal example](guides/examples.md#1-minimal_examplec)
4. Explore other [examples](guides/examples.md)

**Estimated time:** 30 minutes to first simulation

### For Application Developers

1. Read [API Reference](reference/api-reference.md) for function documentation
2. Study [Examples](guides/examples.md) for usage patterns
3. Review [Solvers](reference/solvers.md#choosing-a-solver) for solver selection
4. Check [Error Handling](reference/api-reference.md#error-handling) patterns

**Key resources:**
- [Simulation API](reference/api-reference.md#simulation-api)
- [Solver Registry](reference/api-reference.md#solver-registry-api)
- [Error Handling](architecture/architecture.md#6-error-handling)

### For Performance Engineers

1. Read [Solvers Performance](reference/solvers.md#performance-benchmarks)
2. Study [SIMD Optimization Analysis](technical-notes/simd-optimization-analysis.md)
3. Review [Backend Performance](reference/solvers.md#backend-performance)
4. Run [performance_comparison](guides/examples.md#4-performance_comparisonc)

**Key resources:**
- [Backend Abstraction](architecture/architecture.md#4-backend-abstraction)
- [Memory Efficiency](architecture/architecture.md#5-memory-efficiency)
- [SIMD Implementation](reference/solvers.md#simd-avx2neon)

### For Researchers

1. Study [Numerical Methods](reference/solvers.md#solver-families)
2. Review [Validation Results](validation/)
3. Check [Lid-Driven Cavity](validation/lid-driven-cavity.md) benchmark
4. Read [Linear Solvers](reference/solvers.md#linear-solvers-poisson-equation)

**Key resources:**
- [Projection Method](reference/solvers.md#2-projection-method-solvers)
- [Convergence Theory](reference/solvers.md#linear-solver-performance-comparison)
- [Benchmark Results](reference/solvers.md#validation)

### For Contributors

1. Read [Architecture](architecture/architecture.md) for design principles
2. Study [Backend Abstraction](architecture/architecture.md#4-backend-abstraction)
3. Review [Cross-Backend Checklist](architecture/architecture.md#cross-backend-implementation-checklist)
4. Follow [File Organization](architecture/architecture.md#file-organization-best-practices)

**Key resources:**
- [Design Principles](architecture/architecture.md#design-principles)
- [Modular Architecture](architecture/architecture.md#modular-library-architecture)
- [Algorithm-Primitive Separation](architecture/architecture.md#algorithm-primitive-separation)

## Documentation by Topic

### Numerical Methods

- [Solver Families](reference/solvers.md#solver-families) - Euler vs Projection
- [Linear Solvers](reference/solvers.md#linear-solvers-poisson-equation) - Jacobi, SOR, CG, BiCGSTAB
- [Boundary Conditions](reference/solvers.md#boundary-conditions) - Dirichlet, Neumann, periodic
- [Convergence Theory](reference/solvers.md#linear-solver-performance-comparison)

### Performance Optimization

- [Backend Performance](reference/solvers.md#backend-performance) - CPU, SIMD, OpenMP, CUDA
- [SIMD Optimization](technical-notes/simd-optimization-analysis.md) - AVX2/NEON details
- [Memory Layout](architecture/architecture.md#5-memory-efficiency) - Cache-friendly patterns
- [Performance Benchmarks](reference/solvers.md#performance-benchmarks)

### Software Design

- [Design Principles](architecture/architecture.md#design-principles) - Core architectural decisions
- [Zero-Branch Dispatch](architecture/architecture.md#2-zero-branch-dispatch) - Function pointer pattern
- [Error Handling](architecture/architecture.md#6-error-handling) - Status codes and thread safety
- [Modular Libraries](architecture/architecture.md#modular-library-architecture) - CMake targets

### Validation & Testing

- [Lid-Driven Cavity](validation/lid-driven-cavity.md) - Ghia benchmark
- [Taylor-Green Vortex](reference/solvers.md#taylor-green-vortex) - Analytical validation
- [Testing Patterns](../README.md#testing) - Test suite organization

## API Quick Reference

### Core Functions

```c
// Initialization
cfd_status_t cfd_init(void);
void cfd_cleanup(void);

// Simulation
simulation* simulation_create(size_t nx, size_t ny, ...);
cfd_status_t run_simulation_step(simulation* sim, double dt);
cfd_status_t run_simulation_solve(simulation* sim, double final_time, int* steps);
void simulation_destroy(simulation* sim);

// Output
cfd_status_t write_simulation_to_vtk(simulation* sim, const char* filename);

// Error handling
const char* cfd_get_last_error(void);
const char* cfd_get_error_string(cfd_status_t status);
```

Full reference: [API Reference](reference/api-reference.md)

## Common Tasks

### How do I...

**...build the library?**
- Windows: See [Windows Quick Build](guides/building.md#windows)
- Linux/macOS: See [Linux/macOS Quick Build](guides/building.md#linuxmacos-quick-build)
- With CUDA: See [CUDA Support](guides/building.md#cuda-gpu-support)

**...run a simple simulation?**
- See [minimal_example.c](guides/examples.md#1-minimal_examplec)
- Or [Basic Usage](../README.md#basic-usage) in README

**...choose the right solver?**
- See [Choosing a Solver](reference/solvers.md#choosing-a-solver)
- Or [Decision Tree](reference/solvers.md#decision-tree)

**...handle errors?**
- See [Error Handling API](reference/api-reference.md#error-handling)
- Or [Error Handling Pattern](architecture/architecture.md#6-error-handling)

**...optimize performance?**
- See [Backend Performance](reference/solvers.md#backend-performance)
- Or [Performance Benchmarks](reference/solvers.md#performance-benchmarks)

**...add a new solver?**
- See [Cross-Backend Checklist](architecture/architecture.md#cross-backend-implementation-checklist)
- Or [Algorithm-Primitive Separation](architecture/architecture.md#algorithm-primitive-separation)

**...validate my results?**
- See [Validation](reference/solvers.md#validation)
- Or [Lid-Driven Cavity Benchmark](validation/lid-driven-cavity.md)

## External Resources

### Papers & References

- **Chorin, A.J.** (1968) - Projection method foundation
- **Ghia et al.** (1982) - Lid-driven cavity benchmark
- **Ferziger & Peric** - CFD methods textbook
- **Saad, Y.** - Iterative linear solvers

Full bibliography: [Solvers - References](reference/solvers.md#references)

### Tools

- **ParaView** - VTK visualization (https://www.paraview.org/)
- **VisIt** - Scientific visualization (https://visit.llnl.gov/)
- **CMake** - Build system (https://cmake.org/)
- **CUDA Toolkit** - GPU programming (https://developer.nvidia.com/cuda-toolkit)

## Contributing

See [Architecture Guide](architecture/architecture.md) for:
- Design principles to follow
- Code organization patterns
- Testing conventions
- Cross-backend implementation checklist

## Version Information

Current documentation corresponds to:
- **Version:** 0.1.x
- **Status:** Pre-release (approaching v1.0)
- **Last Updated:** 2025-02-07

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/yourusername/cfd/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/cfd/discussions)
- **Email:** your.email@example.com

## Document Index

### Main Documentation

1. [README.md](../README.md) - Project overview
2. [Building.md](guides/building.md) - Build instructions
3. [Examples.md](guides/examples.md) - Example programs
4. [API Reference.md](reference/api-reference.md) - API documentation
5. [Solvers.md](reference/solvers.md) - Numerical methods
6. [Architecture.md](architecture/architecture.md) - Design principles

### Specialized Documentation

7. [Validation/](validation/) - Benchmark validation
   - [Lid-Driven Cavity](validation/lid-driven-cavity.md)
8. [SIMD Optimization Analysis](technical-notes/simd-optimization-analysis.md)
9. [AVX2 Alignment Bug Fix](technical-notes/avx2-alignment-bug-fix.md)

### Development Documentation

11. [ROADMAP.md](../ROADMAP.md) - Development roadmap
12. [CLAUDE.md](../.claude/CLAUDE.md) - Development guidelines
13. [.claude/commands/](../.claude/commands/) - Development tools

---

**Navigation:**
- [↑ Back to Top](#cfd-framework-documentation)
- [← Back to README](../README.md)
