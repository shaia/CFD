#!/bin/bash

# CFD Library Release Build Script
# This script builds optimized binaries for distribution

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build-release"
INSTALL_DIR="install"
PACKAGE_DIR="packages"

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

detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM=Linux;;
        Darwin*)    PLATFORM=macOS;;
        CYGWIN*|MINGW32*|MSYS*|MINGW64*) PLATFORM=Windows;;
        *)          PLATFORM=Unknown;;
    esac

    case "$(uname -m)" in
        x86_64)     ARCH=x64;;
        i686|i386)  ARCH=x86;;
        arm64|aarch64) ARCH=arm64;;
        *)          ARCH=unknown;;
    esac

    print_status "Detected platform: $PLATFORM-$ARCH"
}

clean_build() {
    print_status "Cleaning previous build..."
    rm -rf "$BUILD_DIR" "$INSTALL_DIR" "$PACKAGE_DIR"
    mkdir -p "$BUILD_DIR" "$INSTALL_DIR" "$PACKAGE_DIR"
}

configure_cmake() {
    print_status "Configuring CMake..."
    cd "$BUILD_DIR"

    # Platform-specific configuration
    CMAKE_ARGS=""
    if [[ "$PLATFORM" == "Windows" ]]; then
        CMAKE_ARGS="-G \"Visual Studio 17 2022\" -A x64"
    elif [[ "$PLATFORM" == "macOS" ]]; then
        # macOS specific settings
        CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15"
        if [[ "$ARCH" == "arm64" ]]; then
            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_OSX_ARCHITECTURES=arm64"
        else
            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_OSX_ARCHITECTURES=x86_64"
        fi
    fi

    # Configure with both static and shared libraries
    eval cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="../$INSTALL_DIR" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_TESTS=ON \
        $CMAKE_ARGS

    cd ..
    print_success "CMake configuration completed"
}

build_project() {
    print_status "Building project..."
    cd "$BUILD_DIR"

    if [[ "$PLATFORM" == "Windows" ]]; then
        cmake --build . --config Release --parallel
    else
        cmake --build . --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi

    cd ..
    print_success "Build completed"
}

run_tests() {
    print_status "Running tests..."
    cd "$BUILD_DIR"

    if [[ "$PLATFORM" == "Windows" ]]; then
        ctest -C Release --output-on-failure
    else
        ctest --output-on-failure
    fi

    cd ..
    print_success "All tests passed"
}

install_project() {
    print_status "Installing project..."
    cd "$BUILD_DIR"

    if [[ "$PLATFORM" == "Windows" ]]; then
        cmake --install . --config Release
    else
        cmake --install .
    fi

    cd ..
    print_success "Installation completed"
}

create_package() {
    print_status "Creating distribution package..."

    PACKAGE_NAME="cfd-${PLATFORM,,}-${ARCH}"
    PACKAGE_PATH="$PACKAGE_DIR/$PACKAGE_NAME"

    mkdir -p "$PACKAGE_PATH"

    # Copy installed files
    if [[ -d "$INSTALL_DIR" ]]; then
        cp -r "$INSTALL_DIR"/* "$PACKAGE_PATH/"
    fi

    # Copy additional files based on platform
    cd "$BUILD_DIR"

    if [[ "$PLATFORM" == "Windows" ]]; then
        # Windows: Copy library files only (.lib, .dll)
        find . -name "*.lib" -exec cp {} "../$PACKAGE_PATH/" \;
        find . -name "*.dll" -exec cp {} "../$PACKAGE_PATH/" \;
        find . -name "*.pdb" -exec cp {} "../$PACKAGE_PATH/" \; 2>/dev/null || true

    else
        # Unix: Copy library files only (.so/.dylib, .a)
        if [[ "$PLATFORM" == "macOS" ]]; then
            find . -name "*.dylib" -exec cp {} "../$PACKAGE_PATH/" \;
        else
            find . -name "*.so*" -exec cp {} "../$PACKAGE_PATH/" \;
        fi

        # Copy static libraries
        find . -name "*.a" -exec cp {} "../$PACKAGE_PATH/" \;

    fi

    cd ..

    # Create documentation
    cat > "$PACKAGE_PATH/README.txt" << EOF
CFD Library - $PLATFORM $ARCH Release Package

Build Information:
- Platform: $PLATFORM
- Architecture: $ARCH
- Build Date: $(date)
- Build Type: Release (Optimized)

Contents:
- lib/ - Static and shared libraries
- include/ - Header files for C/C++ integration
- bin/ - Example executables and tools
- cmake/ - CMake configuration files

Library Files:
EOF

    if [[ "$PLATFORM" == "Windows" ]]; then
        echo "- cfd_library.lib - Static library (link with this)" >> "$PACKAGE_PATH/README.txt"
        echo "- cfd_library.dll - Dynamic library (distribute with your app)" >> "$PACKAGE_PATH/README.txt"
    else
        echo "- libcfd_library.a - Static library (link with this)" >> "$PACKAGE_PATH/README.txt"
        if [[ "$PLATFORM" == "macOS" ]]; then
            echo "- libcfd_library.dylib - Shared library" >> "$PACKAGE_PATH/README.txt"
        else
            echo "- libcfd_library.so - Shared library" >> "$PACKAGE_PATH/README.txt"
        fi
    fi

    cat >> "$PACKAGE_PATH/README.txt" << EOF

Library Package Contents:
- Static libraries for linking into your applications
- Shared libraries for runtime distribution
- Header files for C/C++ integration
- CMake configuration files for easy project integration

Usage (CMake):
  find_package(cfd_library REQUIRED)
  target_link_libraries(your_target CFD::Library)

Usage (Direct linking):
EOF

    if [[ "$PLATFORM" == "Windows" ]]; then
        echo "  cl your_program.c cfd_library.lib" >> "$PACKAGE_PATH/README.txt"
    else
        echo "  gcc your_program.c -lcfd_library -lm" >> "$PACKAGE_PATH/README.txt"
    fi

    # Show package contents
    echo "" >> "$PACKAGE_PATH/README.txt"
    echo "Package Contents:" >> "$PACKAGE_PATH/README.txt"
    ls -la "$PACKAGE_PATH/" >> "$PACKAGE_PATH/README.txt" 2>/dev/null || dir "$PACKAGE_PATH/" >> "$PACKAGE_PATH/README.txt"

    # Create archive
    cd "$PACKAGE_DIR"
    if [[ "$PLATFORM" == "Windows" ]]; then
        powershell -Command "Compress-Archive -Path '$PACKAGE_NAME/*' -DestinationPath '$PACKAGE_NAME.zip'" 2>/dev/null || {
            print_warning "PowerShell not available, creating tar.gz instead"
            tar -czf "$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"/
        }
    else
        tar -czf "$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"/
    fi

    cd ..

    print_success "Package created: $PACKAGE_DIR/$PACKAGE_NAME"

    # Show final package info
    echo ""
    echo "=== PACKAGE INFORMATION ==="
    echo "Name: $PACKAGE_NAME"
    echo "Location: $(pwd)/$PACKAGE_DIR/"
    if [[ "$PLATFORM" == "Windows" ]] && [[ -f "$PACKAGE_DIR/$PACKAGE_NAME.zip" ]]; then
        echo "Archive: $PACKAGE_NAME.zip ($(du -h "$PACKAGE_DIR/$PACKAGE_NAME.zip" | cut -f1))"
    else
        echo "Archive: $PACKAGE_NAME.tar.gz ($(du -h "$PACKAGE_DIR/$PACKAGE_NAME.tar.gz" | cut -f1))"
    fi
    echo ""
    echo "Contents preview:"
    ls -la "$PACKAGE_PATH/" | head -20
}

main() {
    echo "========================================"
    echo "   CFD Library Release Build Script"
    echo "========================================"
    echo ""

    detect_platform

    # Check if we're in the right directory
    if [[ ! -f "CMakeLists.txt" ]]; then
        print_error "CMakeLists.txt not found. Please run this script from the CFD project root directory."
        exit 1
    fi

    # Build steps
    clean_build
    configure_cmake
    build_project
    run_tests
    install_project
    create_package

    print_success "Release build completed successfully!"
    echo ""
    echo "Your binary package is ready for distribution!"
    echo "Upload the archive to GitHub releases or distribute directly."
}

# Run main function
main "$@"