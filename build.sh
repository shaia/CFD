#!/bin/bash

# CFD Project Build Script
# Provides various build, clean, and test operations

set -e  # Exit on error

PROJECT_NAME="CFD Framework"
BUILD_DIR="build"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE} $PROJECT_NAME - Build Script${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Check if we're in the right directory
check_directory() {
    if [[ ! -f "CMakeLists.txt" ]]; then
        print_error "CMakeLists.txt not found. Please run this script from the project root."
        exit 1
    fi
}

# Clean build directory
clean() {
    print_status "Cleaning build directory..."
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    else
        print_warning "Build directory doesn't exist"
    fi
}

# Clean all build and output directories
clean_all() {
    print_status "Cleaning all build and output directories..."

    local cleaned_count=0
    local dirs_to_clean=("build" "build-release" "install" "packages" "output")

    for dir in "${dirs_to_clean[@]}"; do
        if [[ -d "$dir" ]]; then
            print_status "Removing $dir/..."
            if rm -rf "$dir" 2>/dev/null; then
                cleaned_count=$((cleaned_count + 1))
            else
                print_warning "Could not remove $dir (may be in use)"
            fi
        fi
    done

    # Clean artifacts/output but keep artifacts directory structure
    if [[ -d "artifacts/output" ]]; then
        # Use ls to check if directory has contents (more portable than find)
        if ls artifacts/output/* >/dev/null 2>&1; then
            print_status "Cleaning artifacts/output/..."
            if rm -rf artifacts/output/* 2>/dev/null; then
                cleaned_count=$((cleaned_count + 1))
            else
                print_warning "Could not clean artifacts/output (may be in use)"
            fi
        fi
    fi

    if [[ $cleaned_count -gt 0 ]]; then
        print_success "Cleaned $cleaned_count directories"
    else
        print_warning "No directories to clean"
    fi
}

# Clean only output/artifact directories (keep build)
clean_output() {
    print_status "Cleaning output directories..."

    local cleaned_count=0

    if [[ -d "output" ]]; then
        print_status "Removing output/..."
        if rm -rf "output" 2>/dev/null; then
            cleaned_count=$((cleaned_count + 1))
        else
            print_warning "Could not remove output (may be in use)"
        fi
    fi

    if [[ -d "artifacts/output" ]]; then
        # Use ls to check if directory has contents (more portable than find)
        if ls artifacts/output/* >/dev/null 2>&1; then
            print_status "Cleaning artifacts/output/..."
            if rm -rf artifacts/output/* 2>/dev/null; then
                cleaned_count=$((cleaned_count + 1))
            else
                print_warning "Could not clean artifacts/output (may be in use)"
            fi
        fi
    fi

    if [[ $cleaned_count -gt 0 ]]; then
        print_success "Cleaned $cleaned_count output directories"
    else
        print_warning "No output directories to clean"
    fi
}

# Create build directory
create_build_dir() {
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_status "Creating build directory..."
        mkdir -p "$BUILD_DIR"
    fi
}

# Configure CMake
configure() {
    print_status "Configuring CMake (${CMAKE_BUILD_TYPE})..."
    create_build_dir
    cd "$BUILD_DIR"

    # Default options
    local cmake_options=(
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DBUILD_EXAMPLES=ON"
        "-DBUILD_TESTS=OFF"
    )

    # Add any additional arguments passed to the function
    cmake_options+=("$@")

    cmake "${cmake_options[@]}" ..
    cd ..
    print_success "CMake configuration complete"
}

# Build the project
build() {
    print_status "Building project..."
    create_build_dir
    cd "$BUILD_DIR"
    cmake --build . --config "$CMAKE_BUILD_TYPE" -j$(nproc 2>/dev/null || echo 4)
    cd ..
    print_success "Build complete"
}

# Build with tests
build_with_tests() {
    print_status "Configuring with tests enabled..."
    configure -DBUILD_TESTS=ON
    build
}

# Run tests
test() {
    print_status "Running tests..."
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_error "Build directory not found. Run '$0 build-tests' first."
        exit 1
    fi

    cd "$BUILD_DIR"
    if [[ -f "CTestTestfile.cmake" ]]; then
        ctest -C "$CMAKE_BUILD_TYPE" --output-on-failure
        cd ..
        print_success "Tests completed"
    else
        cd ..
        print_error "Tests not configured. Run '$0 build-tests' first."
        exit 1
    fi
}

# Install the project
install() {
    print_status "Installing project..."
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_error "Build directory not found. Run '$0 build' first."
        exit 1
    fi

    cd "$BUILD_DIR"
    cmake --install . --config "$CMAKE_BUILD_TYPE"
    cd ..
    print_success "Installation complete"
}

# Run examples
run_examples() {
    print_status "Running examples..."
    local build_path

    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        build_path="$BUILD_DIR/$CMAKE_BUILD_TYPE"
    else
        build_path="$BUILD_DIR"
    fi

    if [[ ! -d "$build_path" ]]; then
        print_error "Build directory not found. Run '$0 build' first."
        exit 1
    fi

    # Auto-discover examples from the examples/ directory
    local example_files=()
    if [[ -d "examples" ]]; then
        while IFS= read -r -d '' file; do
            local basename=$(basename "$file" .c)
            example_files+=("$basename")
        done < <(find examples -maxdepth 1 -name "*.c" -print0 2>/dev/null | sort -z)
    fi

    if [[ ${#example_files[@]} -eq 0 ]]; then
        print_warning "No example files found in examples/ directory"
        return
    fi

    local ran_count=0
    local skipped_count=0

    for example in "${example_files[@]}"; do
        local exe_name="$example"
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            exe_name="${example}.exe"
        fi

        if [[ -x "$build_path/$exe_name" ]]; then
            print_status "Running $example..."
            cd "$build_path"
            ./"$exe_name"
            cd - > /dev/null
            echo ""
            ran_count=$((ran_count + 1))
        else
            print_warning "Skipping $example (not built or not executable)"
            skipped_count=$((skipped_count + 1))
        fi
    done

    if [[ $ran_count -gt 0 ]]; then
        print_success "Ran $ran_count example(s)"
    fi
    if [[ $skipped_count -gt 0 ]]; then
        print_warning "Skipped $skipped_count example(s)"
    fi
}

# Package the project
package() {
    print_status "Creating package..."
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_error "Build directory not found. Run '$0 build' first."
        exit 1
    fi

    cd "$BUILD_DIR"
    cpack
    cd ..
    print_success "Package created"
}

# Show project status
status() {
    print_header

    echo "Project Status:"
    echo "---------------"

    # Check build directory
    if [[ -d "$BUILD_DIR" ]]; then
        local size=$(du -sh "$BUILD_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
        echo "Build directory: ✓ Exists (${size})"

        # Check for executables
        local exe_count=0
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            exe_count=$(find "$BUILD_DIR" -name "*.exe" 2>/dev/null | wc -l || echo 0)
        else
            exe_count=$(find "$BUILD_DIR" -type f -perm +111 2>/dev/null | wc -l || echo 0)
        fi
        echo "Executables: ${exe_count} built"

        # Check for libraries
        local lib_count=0
        lib_count=$(find "$BUILD_DIR" -name "*.lib" -o -name "*.a" 2>/dev/null | wc -l || echo 0)
        echo "Libraries: ${lib_count} built"
    else
        echo "Build directory: ✗ Not found"
    fi

    # Check output directories
    if [[ -d "artifacts/output" ]]; then
        local vtk_count=$(find artifacts/output -name "*.vtk" 2>/dev/null | wc -l || echo 0)
        echo "VTK files: ${vtk_count} in artifacts/output/"
    else
        echo "Output directory: ✗ Not found"
    fi

    if [[ -d "visualization/visualization_output" ]]; then
        local viz_count=$(find visualization/visualization_output -type f 2>/dev/null | wc -l || echo 0)
        echo "Visualization files: ${viz_count} generated"
    else
        echo "Visualization output: ✗ Not found"
    fi

    echo ""
}

# Show help
help() {
    print_header

    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  clean              Clean build directory"
    echo "  clean-all          Clean all build and output directories"
    echo "  clean-output       Clean only output/artifact directories"
    echo "  configure          Configure CMake"
    echo "  build              Build project (library + examples)"
    echo "  build-tests        Build project with tests enabled"
    echo "  test               Run tests"
    echo "  install            Install project"
    echo "  run                Run example programs"
    echo "  package            Create distribution package"
    echo "  rebuild            Clean and build"
    echo "  full               Full clean, configure, build, and test"
    echo "  status             Show project status"
    echo "  help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CMAKE_BUILD_TYPE   Build type (Debug|Release) [default: Debug]"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build in Debug mode"
    echo "  CMAKE_BUILD_TYPE=Release $0 build  # Build in Release mode"
    echo "  $0 clean-all                # Clean everything (build + output)"
    echo "  $0 full                     # Complete clean build with tests"
    echo "  $0 run                      # Run all example programs"
    echo ""
}

# Main script logic
main() {
    check_directory

    case "${1:-help}" in
        clean)
            clean
            ;;
        clean-all)
            clean_all
            ;;
        clean-output)
            clean_output
            ;;
        configure)
            shift
            configure "$@"
            ;;
        build)
            configure
            build
            ;;
        build-tests)
            build_with_tests
            ;;
        test)
            test
            ;;
        install)
            install
            ;;
        run)
            run_examples
            ;;
        package)
            package
            ;;
        rebuild)
            clean
            configure
            build
            ;;
        full)
            print_header
            clean
            build_with_tests
            test
            print_success "Full build cycle complete!"
            ;;
        status)
            status
            ;;
        help|--help|-h)
            help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"