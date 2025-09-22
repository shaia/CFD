# GitHub Actions Act Configuration

This directory contains configuration files for testing GitHub Actions workflows locally using [act](https://github.com/nektos/act).

## Files

- `Dockerfile.act` - Custom Docker image for act with CMake and build tools
- `actrc` - Act configuration file specifying platform mappings

## Usage

To test GitHub Actions workflows locally:

1. Install act: https://github.com/nektos/act#installation
2. From the repository root, run:
   ```bash
   act -P ubuntu-latest=catthehacker/ubuntu:act-latest
   ```

Or use the configuration file:
```bash
act --actrc .github/act/actrc
```

## Custom Docker Image

The `Dockerfile.act` creates a custom image with:
- CMake
- GCC/G++ compilers
- Build essentials
- Make

This ensures the local testing environment matches the CI environment.