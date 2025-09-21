#!/usr/bin/env python3
"""
Simple script to run CFD simulation and create visualizations
"""

import subprocess
import os
import sys

def run_simulation():
    """Build and run the CFD simulation"""
    print("Building CFD simulation...")

    # Go to parent directory (where CMakeLists.txt is)
    os.chdir("..")

    # Build the project
    if not os.path.exists("build"):
        os.makedirs("build")

    os.chdir("build")

    # Configure and build
    result = subprocess.run(["cmake", ".."], capture_output=True, text=True)
    if result.returncode != 0:
        print("CMake configuration failed:")
        print(result.stderr)
        return False

    result = subprocess.run(["cmake", "--build", "."], capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        return False

    print("Running CFD simulation...")

    # Run the simulation
    result = subprocess.run(["./cfd_framework"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Simulation failed:")
        print(result.stderr)
        return False

    print("Simulation output:")
    print(result.stdout)

    os.chdir("../..")
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize-only":
        print("Skipping simulation, using existing VTK files...")
    else:
        if not run_simulation():
            print("Failed to run simulation!")
            return

    # Move VTK files to output directory if they're in build/
    if os.path.exists("build"):
        import glob
        import shutil
        os.makedirs("output", exist_ok=True)
        build_vtk_files = glob.glob("build/output_optimized_*.vtk")
        for file in build_vtk_files:
            filename = os.path.basename(file)
            shutil.move(file, f"output/{filename}")

    print("\nCreating visualizations...")

    # Import and run visualization
    try:
        from visualize_cfd import main as viz_main
        viz_main()
    except ImportError as e:
        print(f"Failed to import visualization module: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()