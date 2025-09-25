# Branch Protection Setup

This document explains how to configure GitHub branch protection rules to ensure all PRs must pass builds before merging.

## Branch Protection Configuration

To enforce that PRs can only be merged when all builds pass:

1. **Go to your repository on GitHub**
2. **Navigate to Settings → Branches**
3. **Add a branch protection rule for `master` (and `main` if used)**

### Required Settings:

```
Branch name pattern: master
☑️ Require a pull request before merging
  ☑️ Require approvals (optional, but recommended: 1)
  ☑️ Dismiss stale PR approvals when new commits are pushed
☑️ Require status checks to pass before merging
  ☑️ Require branches to be up to date before merging
  Required status checks:
    - Build Status Check
    - build-matrix (Build Linux GCC x64)
    - build-matrix (Build Linux Clang x64)
    - build-matrix (Build Windows MSVC x64)
    - build-matrix (Build Windows MSVC x86)
    - build-matrix (Build macOS x64)
    - build-matrix (Build macOS ARM64)
☑️ Restrict pushes that create matching branches
☑️ Do not allow bypassing the above settings
```

## How It Works

### For Pull Requests:
1. **Automatic Validation**: When a PR is opened/updated, all platforms are built and tested
2. **Merge Blocking**: If ANY build fails, the PR cannot be merged
3. **Status Checks**: GitHub shows clear status for each platform build
4. **Artifacts**: Build artifacts are available for 7 days for PR review

### For Releases (Version Tags):
1. **Tag Triggered**: Only when you push a version tag (e.g., `v1.0.0`)
2. **Build All Platforms**: All 6 platform configurations are built
3. **Create GitHub Release**: Automatically creates a release with all binary downloads
4. **Long Retention**: Release artifacts kept for 90 days

## Creating a Release

To create a new release:

```bash
# Update version and create tag
git tag v1.0.0
git push origin v1.0.0
```

This will:
- Trigger the full build workflow
- Run all tests on all platforms
- Create a GitHub release at: `https://github.com/YOUR_USERNAME/cfd/releases`
- Upload all binary archives for user download

## User Download Experience

Users can find pre-built binaries at:
- **Releases page**: `https://github.com/YOUR_USERNAME/cfd/releases`
- **Direct download**: Each release has platform-specific archives
- **Clear instructions**: Release notes explain what's included and how to use

## Benefits

✅ **Quality Assurance**: No broken code can be merged
✅ **Cross-Platform Testing**: All platforms tested automatically
✅ **Easy Downloads**: Users don't need to build from source
✅ **Professional Releases**: Clean, documented releases with changelogs
✅ **CI/CD Best Practices**: Industry standard workflow