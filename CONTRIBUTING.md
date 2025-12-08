# Contributing to CFD Solver

Thank you for your interest in contributing to the CFD Solver project! We welcome contributions from the community to help improve this high-performance fluid dynamics library.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/CFD.git
    cd CFD
    ```
3.  **Install dependencies**:
    - CMake (3.10+)
    - C Compiler (GCC, Clang, or MSVC)
    - OpenMP (optional, but recommended for performance)
    - CUDA (optional, for GPU solvers)

## Reporting Bugs

If you find a bug, please report it by opening an issue on GitHub. Include:
- A clear title and description.
- Steps to reproduce the issue.
- Your environment (OS, Compiler version, CMake version).
- Expected vs. actual behavior.

## Pull Requests

1.  **Create a new branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Make your changes**. Keep your code clean and readable.
3.  **Run tests** to ensure no regressions:
    ```bash
    mkdir build && cd build
    cmake -DBUILD_TESTS=ON ..
    cmake --build .
    ctest
    ```
4.  **Commit your changes** with descriptive commit messages.
5.  **Push to your fork** and open a Pull Request.

## Coding Guidelines

- **Language Standard**: C11.
- **Style**: Follow the existing coding style (indentation, naming conventions).
    - Use clear variable names (`velocity`, `pressure`, not `v`, `p`).
    - Comment complex logic, especially SIMD and OpenMP sections.
- **Performance**:
    - This is a high-performance library. Avoid unnecessary allocations in tight loops.
    - Use `const` correctness where possible.
    - Verify thread safety for OpenMP implementations.
- **Documentation**: Update code comments and documentation if you change logic.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed under the project's license.
