# GitHub Actions Local Testing with Act

This directory contains development tools for testing GitHub Actions workflows locally using [act](https://github.com/nektos/act).

## Files

- `Dockerfile.act` - Custom Docker image for act with CMake and build tools
- `actrc` - Act configuration file specifying platform mappings
- `README.md` - This documentation file

## Setup

1. **Install act**: https://github.com/nektos/act#installation

2. **Build the custom Docker image**:
   ```bash
   docker build -f tools/act/Dockerfile.act -t cfd-act:latest .
   ```

## Usage

**Test specific workflow**:
```bash
act -P ubuntu-latest=catthehacker/ubuntu:act-latest -W .github/workflows/build-and-release.yml --pull=false
```

## Custom Docker Image

The `Dockerfile.act` creates a custom image with:
- CMake
- GCC/G++ compilers
- Build essentials
- Make

This ensures the local testing environment matches the CI environment.