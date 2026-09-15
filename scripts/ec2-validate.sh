#!/usr/bin/env bash
# ec2-validate.sh — Build and run the 129x129 cavity validation on EC2
#
# Runs every CavityBackend_* ctest entry (Re=100, 400 and 1000), the same cases the
# Cross-Architecture Validation (EC2) workflow runs on each push to master.
#
# Quick-start (on a fresh g4dn.4xlarge with NVIDIA Deep Learning AMI):
#
#   # First time:
#   git clone <your-repo-url> cfd && cd cfd
#   ./scripts/ec2-validate.sh --setup
#   ./scripts/ec2-validate.sh --build --release --run
#
#   # Subsequent runs (after code changes):
#   git pull  # or: rsync from local machine
#   ./scripts/ec2-validate.sh --build --release --run
#
#   # Debug build (much slower; for stepping through with gdb):
#   ./scripts/ec2-validate.sh --build --run
#
# VSCode Remote-SSH debugging:
#   1. Install "Remote - SSH" extension in VSCode
#   2. Ctrl+Shift+P > "Remote-SSH: Connect to Host..." > ubuntu@<ip>
#   3. Open the cfd folder on the remote
#   4. Select "Debug Cavity Backends (Linux/Remote)" or CUDA variant
#   5. Set breakpoints and press F5
#
# Syncing local changes to EC2:
#   rsync -avz --exclude build/ --exclude build_release/ \
#     ./ ubuntu@<ec2-ip>:~/cfd/

set -euo pipefail

# ---------- Defaults ----------
DO_SETUP=false
DO_BUILD=false
DO_RUN=false
BUILD_TYPE="Debug"
OMP_THREADS=""
BUILD_DIR="build"

# ---------- Parse Arguments ----------
if [[ $# -eq 0 ]]; then
    DO_SETUP=true
    DO_BUILD=true
    DO_RUN=true
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)   DO_SETUP=true; shift ;;
        --build)   DO_BUILD=true; shift ;;
        --release) BUILD_TYPE="Release"; BUILD_DIR="build_release"; shift ;;
        --run)     DO_RUN=true; shift ;;
        --all)     DO_SETUP=true; DO_BUILD=true; DO_RUN=true; shift ;;
        --threads) OMP_THREADS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--setup] [--build] [--release] [--run] [--all] [--threads N]"
            echo ""
            echo "  --setup    Install build dependencies (first time only)"
            echo "  --build    Build the project (Debug mode by default)"
            echo "  --release  Use Release mode instead of Debug (matches CI)"
            echo "  --run      Run all 129x129 cavity validation cases (Re=100/400/1000)"
            echo "  --all      Do setup + build + run (default if no flags given)"
            echo "  --threads  Set OMP_NUM_THREADS per test (default: nproc / 4)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- Setup ----------
if [[ "${DO_SETUP}" == "true" ]]; then
    echo "=== Installing build dependencies ==="

    sudo apt-get update
    sudo apt-get install -y build-essential cmake git gdb htop

    echo ""
    echo "--- Checking NVIDIA/CUDA ---"

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        echo "WARNING: nvidia-smi not found. CUDA GPU tests will be skipped."
        echo "Use an NVIDIA Deep Learning AMI or install CUDA toolkit manually."
    fi

    # Add CUDA to PATH if installed but not in PATH
    if [[ -d /usr/local/cuda/bin ]]; then
        export PATH="/usr/local/cuda/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    fi

    if command -v nvcc &>/dev/null; then
        nvcc --version | tail -1
    else
        echo "nvcc not found. Installing CUDA toolkit..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm -f cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit
        export PATH="/usr/local/cuda/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        nvcc --version | tail -1
    fi

    echo ""
    echo "--- Checking CPU features ---"
    if grep -q avx2 /proc/cpuinfo; then
        echo "AVX2: supported"
    else
        echo "AVX2: NOT supported (SIMD backends will be skipped)"
    fi

    echo ""
    echo "--- OpenMP ---"
    NPROC=$(nproc)
    echo "Available cores: ${NPROC}"

    echo ""
    echo "Setup complete."
fi

# ---------- Build ----------
if [[ "${DO_BUILD}" == "true" ]]; then
    echo ""
    echo "=== Building (${BUILD_TYPE}) ==="

    # Ensure CUDA is in PATH
    if [[ -d /usr/local/cuda/bin ]] && ! command -v nvcc &>/dev/null; then
        export PATH="/usr/local/cuda/bin:${PATH}"
    fi

    CUDA_FLAG="OFF"
    if command -v nvcc &>/dev/null; then
        CUDA_FLAG="ON"
    fi

    AVX2_FLAG="OFF"
    if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
        AVX2_FLAG="ON"
    fi

    cmake -B "${BUILD_DIR}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_C_COMPILER=gcc \
        -DBUILD_TESTS=ON \
        -DBUILD_EXAMPLES=OFF \
        -DCFD_ENABLE_AVX2="${AVX2_FLAG}" \
        -DCFD_ENABLE_CUDA="${CUDA_FLAG}" \
        -DCAVITY_FULL_VALIDATION=ON

    cmake --build "${BUILD_DIR}" --parallel "$(nproc)" --target test_cavity_backends

    echo ""
    echo "Build complete: ${BUILD_DIR}/test_cavity_backends"
fi

# ---------- Run ----------
if [[ "${DO_RUN}" == "true" ]]; then
    echo ""
    echo "=== Running 129x129 Cavity Validation (Re=100/400/1000) ==="

    # Same scheduling as CI: 4 ctest slots (each CavityBackend_ entry reserves 2)
    # and nproc/4 OpenMP threads per test
    CTEST_JOBS=4
    if [[ -z "${OMP_THREADS}" ]]; then
        OMP_THREADS=$(( $(nproc) / CTEST_JOBS ))
        OMP_THREADS=$(( OMP_THREADS > 0 ? OMP_THREADS : 1 ))
    fi
    export OMP_NUM_THREADS="${OMP_THREADS}"
    echo "ctest -j ${CTEST_JOBS}, OMP_NUM_THREADS=${OMP_NUM_THREADS} ($(nproc) cores available)"

    BINARY="${BUILD_DIR}/test_cavity_backends"
    if [[ ! -f "${BINARY}" ]]; then
        echo "ERROR: ${BINARY} not found. Run with --build first."
        exit 1
    fi

    LOG="${BUILD_DIR}/validation_output.txt"
    echo ""
    START_TIME=$(date +%s)
    # Capture the status instead of letting set -e end the script before the summary
    EXIT_CODE=0
    ctest --test-dir "${BUILD_DIR}" -R '^CavityBackend_' --no-tests=error \
        --output-on-failure -V -j "${CTEST_JOBS}" 2>&1 | tee "${LOG}" || EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))

    echo ""
    echo "=== Results (full log: ${LOG}) ==="
    grep -E 'Testing backend|Steps run:|RMS_u:|SKIPPED|\[FAILED\]' "${LOG}" || true

    echo ""
    echo "============================================"
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "  PASSED — ${MINUTES}m ${SECONDS}s"
    else
        echo "  FAILED (exit code ${EXIT_CODE}) — ${MINUTES}m ${SECONDS}s"
    fi
    echo "============================================"

    exit ${EXIT_CODE}
fi
