#!/bin/bash

# CFD Framework Release Management Script
# Usage: ./release.sh [version] [--check|--help]

set -e

VERSION_FILE="VERSION"
CURRENT_VERSION=$(cat "$VERSION_FILE" 2>/dev/null || echo "0.0.0")

show_help() {
    cat << EOF
CFD Framework Release Management

USAGE:
    ./release.sh <version>              Create a new release
    ./release.sh --check                Check current version and status
    ./release.sh --help                 Show this help

EXAMPLES:
    ./release.sh 1.2.0                 Create version 1.2.0
    ./release.sh 1.1.5                 Create version 1.1.5
    ./release.sh --check               Show current status

VERSION FORMAT:
    Use semantic versioning: MAJOR.MINOR.PATCH
    - MAJOR: Breaking changes
    - MINOR: New features, backward compatible
    - PATCH: Bug fixes, backward compatible

RELEASE PROCESS:
    1. Update VERSION file with new version
    2. Commit and push changes
    3. GitHub Actions will automatically:
       - Run tests on all platforms
       - Create git tag
       - Build release artifacts
       - Create GitHub release

CURRENT VERSION: $CURRENT_VERSION
EOF
}

check_status() {
    echo "=== CFD Framework Release Status ==="
    echo "Current version: $CURRENT_VERSION"
    echo "VERSION file: $VERSION_FILE"
    echo ""

    # Check if we're on a clean branch
    if ! git diff --quiet; then
        echo "❌ Working directory has uncommitted changes"
        git status --short
        echo ""
    else
        echo "✅ Working directory is clean"
    fi

    # Check current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"

    # Check if current version has a tag
    if git rev-parse "v$CURRENT_VERSION" >/dev/null 2>&1; then
        echo "✅ Tag v$CURRENT_VERSION exists"
        tag_commit=$(git rev-list -n 1 "v$CURRENT_VERSION")
        head_commit=$(git rev-parse HEAD)
        if [ "$tag_commit" = "$head_commit" ]; then
            echo "✅ Current commit matches tag"
        else
            echo "⚠️  Current commit differs from tag"
        fi
    else
        echo "❌ Tag v$CURRENT_VERSION does not exist"
    fi

    # Check recent tags
    echo ""
    echo "Recent tags:"
    git tag --sort=-version:refname | head -5 || echo "No tags found"
}

validate_version() {
    local version="$1"
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "❌ Invalid version format: $version"
        echo "Please use semantic versioning format: MAJOR.MINOR.PATCH (e.g., 1.2.3)"
        exit 1
    fi
}

compare_versions() {
    local current="$1"
    local new="$2"

    # Split versions into components
    IFS='.' read -ra CURR <<< "$current"
    IFS='.' read -ra NEW <<< "$new"

    # Compare major.minor.patch
    for i in 0 1 2; do
        curr_part=${CURR[$i]:-0}
        new_part=${NEW[$i]:-0}

        if (( new_part > curr_part )); then
            return 0  # New version is greater
        elif (( new_part < curr_part )); then
            return 1  # New version is less
        fi
    done
    return 1  # Versions are equal
}

create_release() {
    local new_version="$1"

    validate_version "$new_version"

    # Check if new version is greater than current
    if ! compare_versions "$CURRENT_VERSION" "$new_version"; then
        echo "❌ New version $new_version must be greater than current version $CURRENT_VERSION"
        exit 1
    fi

    # Check if tag already exists
    if git rev-parse "v$new_version" >/dev/null 2>&1; then
        echo "❌ Tag v$new_version already exists"
        exit 1
    fi

    # Check if working directory is clean
    if ! git diff --quiet; then
        echo "❌ Working directory has uncommitted changes. Please commit or stash them first."
        exit 1
    fi

    # Check if we're on main/master branch
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
        echo "⚠️  Warning: You're on branch '$current_branch', not main/master"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted"
            exit 1
        fi
    fi

    echo "=== Creating Release v$new_version ==="
    echo "Current version: $CURRENT_VERSION"
    echo "New version: $new_version"
    echo ""

    # Update VERSION file
    echo "$new_version" > "$VERSION_FILE"
    echo "✅ Updated $VERSION_FILE"

    # Add and commit the version change
    git add "$VERSION_FILE"
    git commit -m "Release v$new_version

- Update VERSION file to $new_version
- This will trigger automatic tag creation and release build

🤖 Generated with [Claude Code](https://claude.ai/code)"

    echo "✅ Committed version update"

    # Push the change
    echo "Pushing to origin..."
    git push origin "$current_branch"

    echo ""
    echo "🚀 Release initiated!"
    echo ""
    echo "Next steps:"
    echo "1. GitHub Actions will run tests on all platforms"
    echo "2. If tests pass, tag v$new_version will be created automatically"
    echo "3. Release artifacts will be built and published"
    echo ""
    echo "Monitor progress at:"
    echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"
}

# Main script logic
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    --check|-c)
        check_status
        ;;
    "")
        echo "❌ No version specified"
        echo "Use --help for usage information"
        exit 1
        ;;
    *)
        create_release "$1"
        ;;
esac